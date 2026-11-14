// Written with AI assistance (Claude Code).
#include "ccip/cpu/nn.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "ccip/cpu/kernels.hpp"

namespace gufo::models::ccip::nn {

void Activate(float* x, std::size_t n, Act act) {
  if (act == Act::kNone) return;
  const auto count = static_cast<std::ptrdiff_t>(n);
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t i = 0; i < count; ++i) {
    const float v = x[i];
    switch (act) {
      case Act::kRelu:
        x[i] = v > 0.0f ? v : 0.0f;
        break;
      case Act::kSilu:
        x[i] = v / (1.0f + std::exp(-v));
        break;
      case Act::kGelu:
        x[i] = 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752f));
        break;
      case Act::kNone:
        break;
    }
  }
}

bool Weights::Load(const std::filesystem::path& file, std::string* error) {
  return file_.Load(file, error);
}

const Tensor& Weights::Get(std::string_view name) const {
  const Tensor* t = file_.Find(name);
  if (t == nullptr) throw std::runtime_error("missing tensor " + std::string(name));
  return *t;
}

Map Conv(const Map& x, const float* weight, const float* bias, int out, int k, int stride, int pad,
         int groups, Act act) {
  Map y;
  y.c = out;
  y.h = (x.h + 2 * pad - k) / stride + 1;
  y.w = (x.w + 2 * pad - k) / stride + 1;
  const int n = y.h * y.w;
  y.d.assign(static_cast<std::size_t>(out) * n, 0.0f);
  if (groups == x.c && groups == out && groups > 1) {
    // Depthwise.
#pragma omp parallel for schedule(static)
    for (int c = 0; c < out; ++c) {
      const float* wc = weight + static_cast<std::size_t>(c) * k * k;
      const float* xc = x.d.data() + static_cast<std::size_t>(c) * x.h * x.w;
      float* yc = y.d.data() + static_cast<std::size_t>(c) * n;
      const float b = bias != nullptr ? bias[c] : 0.0f;
      for (int oy = 0; oy < y.h; ++oy) {
        for (int ox = 0; ox < y.w; ++ox) {
          float s = b;
          for (int ky = 0; ky < k; ++ky) {
            const int iy = oy * stride + ky - pad;
            if (iy < 0 || iy >= x.h) continue;
            for (int kx = 0; kx < k; ++kx) {
              const int ix = ox * stride + kx - pad;
              if (ix < 0 || ix >= x.w) continue;
              s += wc[ky * k + kx] * xc[iy * x.w + ix];
            }
          }
          yc[oy * y.w + ox] = s;
        }
      }
    }
    Activate(y.d.data(), y.d.size(), act);
    return y;
  }
  if (groups != 1) throw std::runtime_error("grouped conv not supported");
  if (k == 1 && stride == 1 && pad == 0) {
    cpu::Gemm(weight, x.d.data(), nullptr, y.d.data(), out, n, x.c);
  } else {
    const int kd = x.c * k * k;
    std::vector<float> cols(static_cast<std::size_t>(kd) * n);
#pragma omp parallel for schedule(static)
    for (int r = 0; r < kd; ++r) {
      const int ci = r / (k * k);
      const int ky = (r / k) % k;
      const int kx = r % k;
      float* dst = cols.data() + static_cast<std::size_t>(r) * n;
      const float* src = x.d.data() + static_cast<std::size_t>(ci) * x.h * x.w;
      for (int oy = 0; oy < y.h; ++oy) {
        const int iy = oy * stride + ky - pad;
        for (int ox = 0; ox < y.w; ++ox) {
          const int ix = ox * stride + kx - pad;
          dst[oy * y.w + ox] =
              (iy >= 0 && iy < x.h && ix >= 0 && ix < x.w) ? src[iy * x.w + ix] : 0.0f;
        }
      }
    }
    cpu::Gemm(weight, cols.data(), nullptr, y.d.data(), out, n, kd);
  }
  if (bias != nullptr) {
#pragma omp parallel for schedule(static)
    for (int o = 0; o < out; ++o) {
      float* r = y.d.data() + static_cast<std::size_t>(o) * n;
      for (int i = 0; i < n; ++i) r[i] += bias[o];
    }
  }
  Activate(y.d.data(), y.d.size(), act);
  return y;
}

Map Conv(const Weights& w, const std::string& name, const Map& x, int stride, int pad, Act act,
         int groups) {
  const Tensor& t = w.Get(name + ".weight");
  if (t.shape.size() != 4 || t.shape[2] != t.shape[3]) throw std::runtime_error("bad conv " + name);
  const int out = static_cast<int>(t.shape[0]);
  const int k = static_cast<int>(t.shape[2]);
  if (t.shape[1] * groups != x.c) throw std::runtime_error("channel mismatch at " + name);
  const float* b = w.Has(name + ".bias") ? w.Raw(name + ".bias").data() : nullptr;
  return Conv(x, t.f32.data(), b, out, k, stride, pad, groups, act);
}

Map MaxPool(const Map& x, int k, int stride, int pad) {
  Map y;
  y.c = x.c;
  y.h = (x.h + 2 * pad - k) / stride + 1;
  y.w = (x.w + 2 * pad - k) / stride + 1;
  y.d.resize(static_cast<std::size_t>(y.c) * y.h * y.w);
#pragma omp parallel for schedule(static)
  for (int c = 0; c < x.c; ++c) {
    for (int oy = 0; oy < y.h; ++oy) {
      for (int ox = 0; ox < y.w; ++ox) {
        float m = -std::numeric_limits<float>::infinity();
        for (int ky = 0; ky < k; ++ky) {
          const int iy = oy * stride + ky - pad;
          if (iy < 0 || iy >= x.h) continue;
          for (int kx = 0; kx < k; ++kx) {
            const int ix = ox * stride + kx - pad;
            if (ix < 0 || ix >= x.w) continue;
            m = std::max(m, x.d[(static_cast<std::size_t>(c) * x.h + iy) * x.w + ix]);
          }
        }
        y.d[(static_cast<std::size_t>(c) * y.h + oy) * y.w + ox] = m;
      }
    }
  }
  return y;
}

Map AvgPoolCeil(const Map& x, int k, int stride) {
  Map y;
  y.c = x.c;
  y.h = (x.h - k + stride - 1) / stride + 1;
  y.w = (x.w - k + stride - 1) / stride + 1;
  y.d.resize(static_cast<std::size_t>(y.c) * y.h * y.w);
#pragma omp parallel for schedule(static)
  for (int c = 0; c < x.c; ++c) {
    for (int oy = 0; oy < y.h; ++oy) {
      for (int ox = 0; ox < y.w; ++ox) {
        float s = 0.0f;
        int count = 0;
        for (int ky = 0; ky < k; ++ky) {
          const int iy = oy * stride + ky;
          if (iy >= x.h) continue;
          for (int kx = 0; kx < k; ++kx) {
            const int ix = ox * stride + kx;
            if (ix >= x.w) continue;
            s += x.d[(static_cast<std::size_t>(c) * x.h + iy) * x.w + ix];
            ++count;
          }
        }
        y.d[(static_cast<std::size_t>(c) * y.h + oy) * y.w + ox] = s / static_cast<float>(count);
      }
    }
  }
  return y;
}

Map UpsampleNearest2(const Map& x) {
  Map y{x.c, x.h * 2, x.w * 2, {}};
  y.d.resize(static_cast<std::size_t>(y.c) * y.h * y.w);
#pragma omp parallel for schedule(static)
  for (int c = 0; c < y.c; ++c) {
    for (int oy = 0; oy < y.h; ++oy) {
      for (int ox = 0; ox < y.w; ++ox) {
        y.d[(static_cast<std::size_t>(c) * y.h + oy) * y.w + ox] =
            x.d[(static_cast<std::size_t>(c) * x.h + oy / 2) * x.w + ox / 2];
      }
    }
  }
  return y;
}

Map UpsampleBilinear(const Map& x, int s) {
  Map y{x.c, x.h * s, x.w * s, {}};
  y.d.resize(static_cast<std::size_t>(y.c) * y.h * y.w);
  const float inv = 1.0f / static_cast<float>(s);
  auto src = [inv](int dst, int n, int* i0, int* i1, float* l1) {
    float v = (static_cast<float>(dst) + 0.5f) * inv - 0.5f;
    if (v < 0.0f) v = 0.0f;
    *i0 = static_cast<int>(v);
    *i1 = *i0 < n - 1 ? *i0 + 1 : *i0;
    *l1 = v - static_cast<float>(*i0);
  };
#pragma omp parallel for schedule(static)
  for (int c = 0; c < y.c; ++c) {
    const float* xc = x.d.data() + static_cast<std::size_t>(c) * x.h * x.w;
    for (int oy = 0; oy < y.h; ++oy) {
      int y0 = 0, y1 = 0;
      float ly = 0;
      src(oy, x.h, &y0, &y1, &ly);
      for (int ox = 0; ox < y.w; ++ox) {
        int x0 = 0, x1 = 0;
        float lx = 0;
        src(ox, x.w, &x0, &x1, &lx);
        const float v = (1.0f - ly) * ((1.0f - lx) * xc[y0 * x.w + x0] + lx * xc[y0 * x.w + x1]) +
                        ly * ((1.0f - lx) * xc[y1 * x.w + x0] + lx * xc[y1 * x.w + x1]);
        y.d[(static_cast<std::size_t>(c) * y.h + oy) * y.w + ox] = v;
      }
    }
  }
  return y;
}

Map Concat(const Map& a, const Map& b) {
  if (a.h != b.h || a.w != b.w) throw std::runtime_error("concat size mismatch");
  Map y{a.c + b.c, a.h, a.w, a.d};
  y.d.insert(y.d.end(), b.d.begin(), b.d.end());
  return y;
}

void AddInPlace(Map* a, const Map& b) {
  if (a->d.size() != b.d.size()) throw std::runtime_error("add size mismatch");
  for (std::size_t i = 0; i < b.d.size(); ++i) a->d[i] += b.d[i];
}

std::vector<float> Linear(const Weights& w, const std::string& name, const float* x, int rows,
                          Act act) {
  const Tensor& t = w.Get(name + ".weight");
  if (t.shape.size() != 2) throw std::runtime_error("bad linear " + name);
  const int out = static_cast<int>(t.shape[0]);
  const int in = static_cast<int>(t.shape[1]);
  const float* b = w.Has(name + ".bias") ? w.Raw(name + ".bias").data() : nullptr;
  std::vector<float> y(static_cast<std::size_t>(rows) * out);
  cpu::GemmBT(x, t.f32.data(), b, y.data(), rows, out, in);
  Activate(y.data(), y.size(), act);
  return y;
}

void LayerNorm(float* x, int rows, int dim, const float* g, const float* b, float eps) {
#pragma omp parallel for schedule(static)
  for (int r = 0; r < rows; ++r) {
    float* v = x + static_cast<std::size_t>(r) * dim;
    double mean = 0.0;
    for (int i = 0; i < dim; ++i) mean += v[i];
    mean /= dim;
    double var = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double d = v[i] - mean;
      var += d * d;
    }
    var /= dim;
    const auto inv = static_cast<float>(1.0 / std::sqrt(var + eps));
    const auto m = static_cast<float>(mean);
    for (int i = 0; i < dim; ++i) v[i] = (v[i] - m) * inv * g[i] + (b != nullptr ? b[i] : 0.0f);
  }
}

void LayerNorm(const Weights& w, const std::string& name, float* x, int rows, int dim, float eps) {
  LayerNorm(x, rows, dim, w.Raw(name + ".weight").data(),
            w.Has(name + ".bias") ? w.Raw(name + ".bias").data() : nullptr, eps);
}

void SoftmaxRows(float* x, int rows, int cols) { cpu::SoftmaxRows(x, rows, cols); }

std::vector<float> Attention(const float* q, const float* k, const float* v, int lq, int lk, int dim,
                             int heads) {
  const int hd = dim / heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  std::vector<float> out(static_cast<std::size_t>(lq) * dim);
#pragma omp parallel for schedule(dynamic)
  for (int t = 0; t < heads * lq; ++t) {
    const int h = t / lq;
    const int i = t % lq;
    std::vector<float> s(static_cast<std::size_t>(lk));
    const float* qi = q + static_cast<std::size_t>(i) * dim + h * hd;
    float mx = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < lk; ++j) {
      const float* kj = k + static_cast<std::size_t>(j) * dim + h * hd;
      float d = 0.0f;
      for (int e = 0; e < hd; ++e) d += qi[e] * kj[e];
      s[j] = d * scale;
      mx = std::max(mx, s[j]);
    }
    float sum = 0.0f;
    for (int j = 0; j < lk; ++j) {
      s[j] = std::exp(s[j] - mx);
      sum += s[j];
    }
    float* oi = out.data() + static_cast<std::size_t>(i) * dim + h * hd;
    for (int e = 0; e < hd; ++e) oi[e] = 0.0f;
    for (int j = 0; j < lk; ++j) {
      const float p = s[j] / sum;
      const float* vj = v + static_cast<std::size_t>(j) * dim + h * hd;
      for (int e = 0; e < hd; ++e) oi[e] += p * vj[e];
    }
  }
  return out;
}

std::vector<float> ToTokens(const Map& x) {
  const int n = x.h * x.w;
  std::vector<float> t(static_cast<std::size_t>(n) * x.c);
  for (int c = 0; c < x.c; ++c) {
    for (int i = 0; i < n; ++i) t[static_cast<std::size_t>(i) * x.c + c] = x.d[static_cast<std::size_t>(c) * n + i];
  }
  return t;
}

Map FromTokens(const std::vector<float>& t, int c, int h, int w) {
  Map y{c, h, w, std::vector<float>(static_cast<std::size_t>(c) * h * w)};
  const int n = h * w;
  for (int ch = 0; ch < c; ++ch) {
    for (int i = 0; i < n; ++i) y.d[static_cast<std::size_t>(ch) * n + i] = t[static_cast<std::size_t>(i) * c + ch];
  }
  return y;
}

}  // namespace gufo::models::ccip::nn
