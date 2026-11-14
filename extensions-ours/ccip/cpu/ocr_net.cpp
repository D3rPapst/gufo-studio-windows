// Written with AI assistance (Claude Code).
#include "ccip/cpu/ocr_net.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <gufo/ext/json.hpp>

#include "ccip/cpu/kernels.hpp"
#include "ccip/tensors.hpp"

namespace gufo::models::ccip::cpu {

std::int64_t NetTensor::numel() const {
  std::int64_t n = 1;
  for (std::int64_t d : shape) n *= d;
  return n;
}

namespace {

using Shape = std::vector<std::int64_t>;

[[noreturn]] void Fail(const std::string& what) { throw std::runtime_error(what); }

enum class Op {
  kAdd, kAveragePool, kBatchNorm, kCast, kClip, kConcat, kConv, kConvTranspose, kDiv,
  kGlobalAveragePool, kHardSigmoid, kMatMul, kMul, kPow, kReduceMean, kRelu, kReshape, kResize,
  kShape, kSigmoid, kSlice, kSoftmax, kSqrt, kSqueeze, kSub, kTranspose,
};

Op ParseOp(const std::string& s) {
  static const std::map<std::string, Op> ops = {
      {"Add", Op::kAdd}, {"AveragePool", Op::kAveragePool},
      {"BatchNormalization", Op::kBatchNorm}, {"Cast", Op::kCast}, {"Clip", Op::kClip},
      {"Concat", Op::kConcat}, {"Conv", Op::kConv}, {"ConvTranspose", Op::kConvTranspose},
      {"Div", Op::kDiv}, {"GlobalAveragePool", Op::kGlobalAveragePool},
      {"HardSigmoid", Op::kHardSigmoid}, {"MatMul", Op::kMatMul}, {"Mul", Op::kMul},
      {"Pow", Op::kPow}, {"ReduceMean", Op::kReduceMean}, {"Relu", Op::kRelu},
      {"Reshape", Op::kReshape}, {"Resize", Op::kResize}, {"Shape", Op::kShape},
      {"Sigmoid", Op::kSigmoid}, {"Slice", Op::kSlice}, {"Softmax", Op::kSoftmax},
      {"Sqrt", Op::kSqrt}, {"Squeeze", Op::kSqueeze}, {"Sub", Op::kSub},
      {"Transpose", Op::kTranspose},
  };
  auto it = ops.find(s);
  if (it == ops.end()) Fail("unsupported op " + s);
  return it->second;
}

struct Attrs {
  std::map<std::string, std::vector<std::int64_t>> ints;
  std::map<std::string, double> floats;
  std::map<std::string, std::string> strings;
  std::int64_t Int(const std::string& k, std::int64_t def) const {
    auto it = ints.find(k);
    return it == ints.end() || it->second.size() != 1 ? def : it->second[0];
  }
  std::vector<std::int64_t> List(const std::string& k, std::vector<std::int64_t> def) const {
    auto it = ints.find(k);
    return it == ints.end() ? def : it->second;
  }
  double Float(const std::string& k, double def) const {
    auto it = floats.find(k);
    return it == floats.end() ? def : it->second;
  }
};

constexpr int kNone = std::numeric_limits<int>::min();

struct Node {
  Op op{};
  std::string name;          // first output (error messages)
  std::vector<int> in;       // run slot (>= 0), constant -(index + 1), or kNone
  std::vector<int> out;
  std::vector<int> release;  // run slots whose last use is this node
  Attrs attrs;
};

std::size_t Sz(std::int64_t v) { return static_cast<std::size_t>(v); }

Shape Strides(const Shape& s) {
  Shape st(s.size(), 1);
  for (int d = static_cast<int>(s.size()) - 2; d >= 0; --d) st[Sz(d)] = st[Sz(d) + 1] * s[Sz(d) + 1];
  return st;
}

NetTensor FloatTensor(Shape shape) {
  NetTensor t;
  t.shape = std::move(shape);
  t.f.resize(Sz(t.numel()));
  return t;
}

// ------------------------------------------------------------------ elementwise
enum class Bin { kAdd, kSub, kMul, kDiv, kPow };

inline float Apply(Bin op, float a, float b) {
  switch (op) {
    case Bin::kAdd: return a + b;
    case Bin::kSub: return a - b;
    case Bin::kMul: return a * b;
    case Bin::kDiv: return a / b;
    case Bin::kPow: return b == 2.0f ? a * a : std::pow(a, b);
  }
  return 0.0f;
}

NetTensor Binary(Bin op, const NetTensor& a, const NetTensor& b) {
  if (a.is_int || b.is_int) Fail("elementwise ops on int tensors are not implemented");
  const std::size_t rank = std::max(a.shape.size(), b.shape.size());
  Shape sa(rank, 1), sb(rank, 1), so(rank, 1);
  std::copy(a.shape.begin(), a.shape.end(), sa.begin() + static_cast<std::ptrdiff_t>(rank - a.shape.size()));
  std::copy(b.shape.begin(), b.shape.end(), sb.begin() + static_cast<std::ptrdiff_t>(rank - b.shape.size()));
  for (std::size_t d = 0; d < rank; ++d) {
    if (sa[d] != sb[d] && sa[d] != 1 && sb[d] != 1) Fail("broadcast mismatch");
    so[d] = (sa[d] == 0 || sb[d] == 0) ? 0 : std::max(sa[d], sb[d]);
  }
  NetTensor out = FloatTensor(so);
  if (out.f.empty()) return out;
  // Element strides (0 on broadcast dims); merge dims that stay contiguous for
  // both inputs so the inner loop is long.
  Shape ta = Strides(sa), tb = Strides(sb);
  for (std::size_t d = 0; d < rank; ++d) {
    if (sa[d] == 1) ta[d] = 0;
    if (sb[d] == 1) tb[d] = 0;
  }
  Shape dims, st_a, st_b;
  for (std::size_t d = 0; d < rank; ++d) {
    if (so[d] == 1) continue;
    if (!dims.empty() && st_a.back() == ta[d] * so[d] && st_b.back() == tb[d] * so[d]) {
      dims.back() *= so[d];
      st_a.back() = ta[d];
      st_b.back() = tb[d];
    } else {
      dims.push_back(so[d]);
      st_a.push_back(ta[d]);
      st_b.push_back(tb[d]);
    }
  }
  if (dims.empty()) {
    out.f[0] = Apply(op, a.f[0], b.f[0]);
    return out;
  }
  const std::int64_t inner = dims.back();
  const std::int64_t ia = st_a.back(), ib = st_b.back();
  const std::int64_t outer = out.numel() / inner;
  const int nd = static_cast<int>(dims.size()) - 1;
  const float* pa = a.f.data();
  const float* pb = b.f.data();
  float* po = out.f.data();
#pragma omp parallel for schedule(static) if (outer > 16 && outer * inner > 32768)
  for (std::int64_t o = 0; o < outer; ++o) {
    std::int64_t rem = o, oa = 0, ob = 0;
    for (int d = nd - 1; d >= 0; --d) {
      const std::int64_t idx = rem % dims[Sz(d)];
      rem /= dims[Sz(d)];
      oa += idx * st_a[Sz(d)];
      ob += idx * st_b[Sz(d)];
    }
    const float* xa = pa + oa;
    const float* xb = pb + ob;
    float* y = po + o * inner;
    if (ia == 1 && ib == 1) {
      for (std::int64_t j = 0; j < inner; ++j) y[j] = Apply(op, xa[j], xb[j]);
    } else if (ia == 1 && ib == 0) {
      const float v = xb[0];
      for (std::int64_t j = 0; j < inner; ++j) y[j] = Apply(op, xa[j], v);
    } else if (ia == 0 && ib == 1) {
      const float v = xa[0];
      for (std::int64_t j = 0; j < inner; ++j) y[j] = Apply(op, v, xb[j]);
    } else {
      for (std::int64_t j = 0; j < inner; ++j) y[j] = Apply(op, xa[j * ia], xb[j * ib]);
    }
  }
  return out;
}

template <typename F>
NetTensor Unary(const NetTensor& x, F fn) {
  if (x.is_int) Fail("unary op on an int tensor");
  NetTensor out = FloatTensor(x.shape);
  const std::int64_t n = x.numel();
  const float* px = x.f.data();
  float* po = out.f.data();
#pragma omp parallel for schedule(static) if (n > 65536)
  for (std::int64_t k = 0; k < n; ++k) po[k] = fn(px[k]);
  return out;
}

// ------------------------------------------------------------------ conv
struct ConvParams {
  int group{1};
  int kh{1}, kw{1};
  int sh{1}, sw{1};
  int pt{0}, pl{0}, pb{0}, pr{0};
};

ConvParams ReadConv(const Attrs& at, const Shape& wshape) {
  ConvParams p;
  p.group = static_cast<int>(at.Int("group", 1));
  const auto k = at.List("kernel_shape", {wshape[2], wshape[3]});
  p.kh = static_cast<int>(k[0]);
  p.kw = static_cast<int>(k[1]);
  const auto s = at.List("strides", {1, 1});
  p.sh = static_cast<int>(s[0]);
  p.sw = static_cast<int>(s[1]);
  const auto pads = at.List("pads", {0, 0, 0, 0});
  p.pt = static_cast<int>(pads[0]);
  p.pl = static_cast<int>(pads[1]);
  p.pb = static_cast<int>(pads[2]);
  p.pr = static_cast<int>(pads[3]);
  return p;
}

NetTensor Conv(const NetTensor& x, const NetTensor& w, const NetTensor* b, const Attrs& at) {
  if (x.shape.size() != 4 || w.shape.size() != 4) Fail("Conv expects 4-D tensors");
  const ConvParams p = ReadConv(at, w.shape);
  const int n = static_cast<int>(x.shape[0]);
  const int c = static_cast<int>(x.shape[1]);
  const int h = static_cast<int>(x.shape[2]);
  const int wd = static_cast<int>(x.shape[3]);
  const int m = static_cast<int>(w.shape[0]);
  const int cg = static_cast<int>(w.shape[1]);
  if (cg * p.group != c || m % p.group != 0 || w.shape[2] != p.kh || w.shape[3] != p.kw)
    Fail("Conv weight shape mismatch");
  const int oh = (h + p.pt + p.pb - p.kh) / p.sh + 1;
  const int ow = (wd + p.pl + p.pr - p.kw) / p.sw + 1;
  NetTensor out = FloatTensor({n, m, oh, ow});
  const std::size_t ohw = Sz(oh) * Sz(ow);
  const float* bias = b != nullptr ? b->f.data() : nullptr;
  const int mg = m / p.group;
  if (cg == 1 && mg == 1) {
    // Depthwise: one output plane per input channel.
    for (int bn = 0; bn < n; ++bn) {
#pragma omp parallel for schedule(static)
      for (int ch = 0; ch < m; ++ch) {
        const float* xp = x.f.data() + (Sz(bn) * Sz(c) + Sz(ch)) * Sz(h) * Sz(wd);
        const float* wp = w.f.data() + Sz(ch) * Sz(p.kh) * Sz(p.kw);
        float* yp = out.f.data() + (Sz(bn) * Sz(m) + Sz(ch)) * ohw;
        const float bv = bias != nullptr ? bias[ch] : 0.0f;
        for (int oy = 0; oy < oh; ++oy) {
          float* yrow = yp + Sz(oy) * Sz(ow);
          for (int ox = 0; ox < ow; ++ox) yrow[ox] = bv;
          for (int ky = 0; ky < p.kh; ++ky) {
            const int iy = oy * p.sh - p.pt + ky;
            if (iy < 0 || iy >= h) continue;
            const float* xrow = xp + Sz(iy) * Sz(wd);
            for (int kx = 0; kx < p.kw; ++kx) {
              const float wv = wp[ky * p.kw + kx];
              const int x0 = kx - p.pl;
              // ox with 0 <= ox * sw + x0 < wd
              const int lo = x0 >= 0 ? 0 : (-x0 + p.sw - 1) / p.sw;
              int hi = (wd - 1 - x0) >= 0 ? (wd - 1 - x0) / p.sw + 1 : 0;
              hi = std::min(hi, ow);
              if (p.sw == 1) {
                for (int ox = lo; ox < hi; ++ox) yrow[ox] += wv * xrow[ox + x0];
              } else {
                for (int ox = lo; ox < hi; ++ox) yrow[ox] += wv * xrow[ox * p.sw + x0];
              }
            }
          }
        }
      }
    }
    return out;
  }
  const int kk = cg * p.kh * p.kw;
  const bool pointwise = p.kh == 1 && p.kw == 1 && p.sh == 1 && p.sw == 1 && p.pt == 0 &&
                         p.pl == 0 && p.pb == 0 && p.pr == 0;
  std::vector<float> col;
  if (!pointwise) col.resize(Sz(kk) * ohw);
  for (int bn = 0; bn < n; ++bn) {
    for (int g = 0; g < p.group; ++g) {
      const float* xg = x.f.data() + (Sz(bn) * Sz(c) + Sz(g) * Sz(cg)) * Sz(h) * Sz(wd);
      const float* src = xg;
      if (!pointwise) {
#pragma omp parallel for schedule(static)
        for (int r = 0; r < kk; ++r) {
          const int ci = r / (p.kh * p.kw);
          const int ky = (r / p.kw) % p.kh;
          const int kx = r % p.kw;
          float* dst = col.data() + Sz(r) * ohw;
          const float* xc = xg + Sz(ci) * Sz(h) * Sz(wd);
          for (int oy = 0; oy < oh; ++oy) {
            const int iy = oy * p.sh - p.pt + ky;
            float* drow = dst + Sz(oy) * Sz(ow);
            if (iy < 0 || iy >= h) {
              std::fill(drow, drow + ow, 0.0f);
              continue;
            }
            const float* xrow = xc + Sz(iy) * Sz(wd);
            for (int ox = 0; ox < ow; ++ox) {
              const int ix = ox * p.sw - p.pl + kx;
              drow[ox] = (ix >= 0 && ix < wd) ? xrow[ix] : 0.0f;
            }
          }
        }
        src = col.data();
      }
      float* yg = out.f.data() + (Sz(bn) * Sz(m) + Sz(g) * Sz(mg)) * ohw;
      Gemm(w.f.data() + Sz(g) * Sz(mg) * Sz(kk), src, nullptr, yg, mg, static_cast<int>(ohw), kk);
      if (bias != nullptr) {
#pragma omp parallel for schedule(static)
        for (int o = 0; o < mg; ++o) {
          float* yr = yg + Sz(o) * ohw;
          const float bv = bias[g * mg + o];
          for (std::size_t j = 0; j < ohw; ++j) yr[j] += bv;
        }
      }
    }
  }
  return out;
}

NetTensor ConvTranspose(const NetTensor& x, const NetTensor& w, const NetTensor* b,
                        const Attrs& at) {
  const ConvParams p = ReadConv(at, w.shape);
  if (p.group != 1) Fail("ConvTranspose with groups is not implemented");
  const int n = static_cast<int>(x.shape[0]);
  const int ci = static_cast<int>(x.shape[1]);
  const int h = static_cast<int>(x.shape[2]);
  const int wd = static_cast<int>(x.shape[3]);
  if (w.shape[0] != ci) Fail("ConvTranspose weight shape mismatch");
  const int co = static_cast<int>(w.shape[1]);
  const int oh = (h - 1) * p.sh - p.pt - p.pb + p.kh;
  const int ow = (wd - 1) * p.sw - p.pl - p.pr + p.kw;
  NetTensor out = FloatTensor({n, co, oh, ow});
  const int rows = co * p.kh * p.kw;
  // wt[rows, ci] = w[ci, rows]^T
  std::vector<float> wt(Sz(rows) * Sz(ci));
  for (int i = 0; i < ci; ++i)
    for (int r = 0; r < rows; ++r) wt[Sz(r) * Sz(ci) + Sz(i)] = w.f[Sz(i) * Sz(rows) + Sz(r)];
  const std::size_t hw = Sz(h) * Sz(wd);
  std::vector<float> col(Sz(rows) * hw);
  for (int bn = 0; bn < n; ++bn) {
    Gemm(wt.data(), x.f.data() + Sz(bn) * Sz(ci) * hw, nullptr, col.data(), rows,
         static_cast<int>(hw), ci);
    float* y = out.f.data() + Sz(bn) * Sz(co) * Sz(oh) * Sz(ow);
#pragma omp parallel for schedule(static)
    for (int o = 0; o < co; ++o) {
      float* yo = y + Sz(o) * Sz(oh) * Sz(ow);
      const float bv = b != nullptr ? b->f[Sz(o)] : 0.0f;
      std::fill(yo, yo + Sz(oh) * Sz(ow), bv);
      for (int ky = 0; ky < p.kh; ++ky) {
        for (int kx = 0; kx < p.kw; ++kx) {
          const float* cr = col.data() + (Sz(o) * Sz(p.kh) * Sz(p.kw) + Sz(ky * p.kw + kx)) * hw;
          for (int iy = 0; iy < h; ++iy) {
            const int yy = iy * p.sh - p.pt + ky;
            if (yy < 0 || yy >= oh) continue;
            for (int ix = 0; ix < wd; ++ix) {
              const int xx = ix * p.sw - p.pl + kx;
              if (xx < 0 || xx >= ow) continue;
              yo[Sz(yy) * Sz(ow) + Sz(xx)] += cr[Sz(iy) * Sz(wd) + Sz(ix)];
            }
          }
        }
      }
    }
  }
  return out;
}

NetTensor BatchNorm(const NetTensor& x, const NetTensor& scale, const NetTensor& bias,
                    const NetTensor& mean, const NetTensor& var, double eps) {
  const std::int64_t n = x.shape[0], c = x.shape[1];
  const std::int64_t inner = x.numel() / (n * c);
  NetTensor out = FloatTensor(x.shape);
  std::vector<float> s(Sz(c)), t(Sz(c));
  for (std::int64_t ch = 0; ch < c; ++ch) {
    const float inv = 1.0f / std::sqrt(var.f[Sz(ch)] + static_cast<float>(eps));
    s[Sz(ch)] = scale.f[Sz(ch)] * inv;
    t[Sz(ch)] = bias.f[Sz(ch)] - mean.f[Sz(ch)] * s[Sz(ch)];
  }
#pragma omp parallel for schedule(static)
  for (std::int64_t r = 0; r < n * c; ++r) {
    const float sv = s[Sz(r % c)], tv = t[Sz(r % c)];
    const float* xp = x.f.data() + Sz(r) * Sz(inner);
    float* yp = out.f.data() + Sz(r) * Sz(inner);
    for (std::int64_t j = 0; j < inner; ++j) yp[j] = xp[j] * sv + tv;
  }
  return out;
}

NetTensor GlobalAveragePool(const NetTensor& x) {
  const std::int64_t n = x.shape[0], c = x.shape[1];
  const std::int64_t inner = x.numel() / (n * c);
  NetTensor out = FloatTensor({n, c, 1, 1});
#pragma omp parallel for schedule(static) if (n * c * inner > 65536)
  for (std::int64_t r = 0; r < n * c; ++r) {
    const float* xp = x.f.data() + Sz(r) * Sz(inner);
    double s = 0.0;
    for (std::int64_t j = 0; j < inner; ++j) s += xp[j];
    out.f[Sz(r)] = static_cast<float>(s / static_cast<double>(inner));
  }
  return out;
}

NetTensor AveragePool(const NetTensor& x, const Attrs& at) {
  const auto k = at.List("kernel_shape", {});
  const auto s = at.List("strides", {1, 1});
  if (k.size() != 2 || x.shape.size() != 4) Fail("AveragePool expects 2-D kernels");
  const std::int64_t n = x.shape[0], c = x.shape[1], h = x.shape[2], w = x.shape[3];
  const std::int64_t oh = (h - k[0]) / s[0] + 1, ow = (w - k[1]) / s[1] + 1;
  NetTensor out = FloatTensor({n, c, oh, ow});
  const float inv = 1.0f / static_cast<float>(k[0] * k[1]);
  for (std::int64_t r = 0; r < n * c; ++r) {
    const float* xp = x.f.data() + Sz(r) * Sz(h) * Sz(w);
    float* yp = out.f.data() + Sz(r) * Sz(oh) * Sz(ow);
    for (std::int64_t oy = 0; oy < oh; ++oy)
      for (std::int64_t ox = 0; ox < ow; ++ox) {
        float acc = 0.0f;
        for (std::int64_t ky = 0; ky < k[0]; ++ky)
          for (std::int64_t kx = 0; kx < k[1]; ++kx)
            acc += xp[Sz(oy * s[0] + ky) * Sz(w) + Sz(ox * s[1] + kx)];
        yp[Sz(oy) * Sz(ow) + Sz(ox)] = acc * inv;
      }
  }
  return out;
}

NetTensor Resize(const NetTensor& x, const NetTensor* scales, const NetTensor* sizes) {
  const std::size_t rank = x.shape.size();
  Shape os(rank);
  std::vector<double> sc(rank, 1.0);
  if (sizes != nullptr && sizes->numel() == static_cast<std::int64_t>(rank)) {
    for (std::size_t d = 0; d < rank; ++d) {
      os[d] = sizes->is_int ? sizes->i[d] : static_cast<std::int64_t>(sizes->f[d]);
      sc[d] = static_cast<double>(os[d]) / static_cast<double>(x.shape[d]);
    }
  } else if (scales != nullptr && scales->numel() == static_cast<std::int64_t>(rank)) {
    for (std::size_t d = 0; d < rank; ++d) {
      sc[d] = scales->f[d];
      os[d] = static_cast<std::int64_t>(static_cast<double>(x.shape[d]) * sc[d]);
    }
  } else {
    Fail("Resize needs scales or sizes");
  }
  if (rank != 4 || os[0] != x.shape[0] || os[1] != x.shape[1]) Fail("Resize only over H, W");
  NetTensor out = FloatTensor(os);
  const std::int64_t h = x.shape[2], w = x.shape[3], oh = os[2], ow = os[3];
  std::vector<std::int64_t> ys(Sz(oh)), xs(Sz(ow));
  // coordinate_transformation_mode asymmetric, nearest_mode floor.
  for (std::int64_t y = 0; y < oh; ++y)
    ys[Sz(y)] = std::min<std::int64_t>(
        h - 1, static_cast<std::int64_t>(std::floor(static_cast<double>(y) / sc[2])));
  for (std::int64_t xx = 0; xx < ow; ++xx)
    xs[Sz(xx)] = std::min<std::int64_t>(
        w - 1, static_cast<std::int64_t>(std::floor(static_cast<double>(xx) / sc[3])));
  const std::int64_t planes = os[0] * os[1];
#pragma omp parallel for schedule(static)
  for (std::int64_t r = 0; r < planes; ++r) {
    const float* xp = x.f.data() + Sz(r) * Sz(h) * Sz(w);
    float* yp = out.f.data() + Sz(r) * Sz(oh) * Sz(ow);
    for (std::int64_t y = 0; y < oh; ++y) {
      const float* xrow = xp + Sz(ys[Sz(y)]) * Sz(w);
      float* yrow = yp + Sz(y) * Sz(ow);
      for (std::int64_t xx = 0; xx < ow; ++xx) yrow[xx] = xrow[xs[Sz(xx)]];
    }
  }
  return out;
}

std::int64_t Axis(std::int64_t a, std::size_t rank) {
  if (a < 0) a += static_cast<std::int64_t>(rank);
  if (a < 0 || a >= static_cast<std::int64_t>(rank)) Fail("axis out of range");
  return a;
}

NetTensor Concat(const std::vector<const NetTensor*>& xs, std::int64_t axis_attr) {
  const NetTensor& first = *xs[0];
  const std::size_t rank = first.shape.size();
  const std::int64_t axis = Axis(axis_attr, rank);
  NetTensor out;
  out.is_int = first.is_int;
  out.shape = first.shape;
  out.shape[Sz(axis)] = 0;
  for (const NetTensor* t : xs) {
    if (t->is_int != out.is_int || t->shape.size() != rank) Fail("Concat input mismatch");
    out.shape[Sz(axis)] += t->shape[Sz(axis)];
  }
  std::int64_t outer = 1;
  for (std::int64_t d = 0; d < axis; ++d) outer *= out.shape[Sz(d)];
  std::int64_t inner = 1;
  for (std::size_t d = Sz(axis) + 1; d < rank; ++d) inner *= out.shape[d];
  if (out.is_int)
    out.i.resize(Sz(out.numel()));
  else
    out.f.resize(Sz(out.numel()));
  std::int64_t offset = 0;
  const std::int64_t row = out.shape[Sz(axis)] * inner;
  for (const NetTensor* t : xs) {
    const std::int64_t chunk = t->shape[Sz(axis)] * inner;
    for (std::int64_t o = 0; o < outer; ++o) {
      if (out.is_int)
        std::copy_n(t->i.data() + o * chunk, chunk, out.i.data() + o * row + offset);
      else
        std::copy_n(t->f.data() + o * chunk, chunk, out.f.data() + o * row + offset);
    }
    offset += chunk;
  }
  return out;
}

std::vector<std::int64_t> IntValues(const NetTensor& t) {
  if (t.is_int) return t.i;
  std::vector<std::int64_t> v;
  for (float f : t.f) v.push_back(static_cast<std::int64_t>(f));
  return v;
}

NetTensor Slice(const NetTensor& x, const NetTensor& starts, const NetTensor* ends_t,
                const NetTensor* axes_t, const NetTensor* steps_t) {
  if (ends_t == nullptr) Fail("Slice without ends");
  const std::size_t rank = x.shape.size();
  const auto st = IntValues(starts);
  const auto en = IntValues(*ends_t);
  std::vector<std::int64_t> ax, sp;
  if (axes_t != nullptr) {
    ax = IntValues(*axes_t);
  } else {
    for (std::size_t k = 0; k < st.size(); ++k) ax.push_back(static_cast<std::int64_t>(k));
  }
  if (steps_t != nullptr) sp = IntValues(*steps_t);
  else sp.assign(st.size(), 1);
  std::vector<std::int64_t> b(rank, 0), step(rank, 1);
  Shape os = x.shape;
  for (std::size_t k = 0; k < st.size(); ++k) {
    const std::int64_t a = Axis(ax[k], rank);
    const std::int64_t dim = x.shape[Sz(a)];
    if (sp[k] <= 0) Fail("Slice with step <= 0 is not implemented");
    std::int64_t s = st[k], e = en[k];
    if (s < 0) s += dim;
    if (e < 0) e += dim;
    s = std::clamp<std::int64_t>(s, 0, dim);
    e = std::clamp<std::int64_t>(e, 0, dim);
    b[Sz(a)] = s;
    step[Sz(a)] = sp[k];
    os[Sz(a)] = e > s ? (e - s + sp[k] - 1) / sp[k] : 0;
  }
  NetTensor out;
  out.is_int = x.is_int;
  out.shape = os;
  const std::int64_t n = out.numel();
  if (out.is_int) out.i.resize(Sz(n));
  else out.f.resize(Sz(n));
  const Shape xs = Strides(x.shape);
  for (std::int64_t idx = 0; idx < n; ++idx) {
    std::int64_t rem = idx, src = 0;
    for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
      const std::int64_t v = rem % os[Sz(d)];
      rem /= os[Sz(d)];
      src += (b[Sz(d)] + v * step[Sz(d)]) * xs[Sz(d)];
    }
    if (out.is_int) out.i[Sz(idx)] = x.i[Sz(src)];
    else out.f[Sz(idx)] = x.f[Sz(src)];
  }
  return out;
}

NetTensor Transpose(const NetTensor& x, const std::vector<std::int64_t>& perm_attr) {
  const std::size_t rank = x.shape.size();
  if (x.is_int) Fail("Transpose on int tensors is not implemented");
  std::vector<std::int64_t> perm = perm_attr;
  if (perm.empty())
    for (std::size_t d = 0; d < rank; ++d) perm.push_back(static_cast<std::int64_t>(rank - 1 - d));
  if (perm.size() != rank) Fail("Transpose perm rank");
  Shape os(rank);
  for (std::size_t d = 0; d < rank; ++d) os[d] = x.shape[Sz(perm[d])];
  const Shape xs = Strides(x.shape);
  Shape src_stride(rank);
  for (std::size_t d = 0; d < rank; ++d) src_stride[d] = xs[Sz(perm[d])];
  NetTensor out = FloatTensor(os);
  const std::int64_t n = out.numel();
  if (n == 0) return out;
  const std::int64_t inner = os[rank - 1];
  const std::int64_t is = src_stride[rank - 1];
  const std::int64_t outer = n / inner;
#pragma omp parallel for schedule(static) if (n > 65536)
  for (std::int64_t o = 0; o < outer; ++o) {
    std::int64_t rem = o, src = 0;
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d) {
      const std::int64_t v = rem % os[Sz(d)];
      rem /= os[Sz(d)];
      src += v * src_stride[Sz(d)];
    }
    float* y = out.f.data() + o * inner;
    const float* xp = x.f.data() + src;
    for (std::int64_t j = 0; j < inner; ++j) y[j] = xp[j * is];
  }
  return out;
}

NetTensor Reshape(const NetTensor& x, const NetTensor& shape_t) {
  const auto req = IntValues(shape_t);
  Shape os(req.size());
  std::int64_t known = 1;
  int infer = -1;
  for (std::size_t d = 0; d < req.size(); ++d) {
    if (req[d] == 0) {
      if (d >= x.shape.size()) Fail("Reshape 0 beyond the input rank");
      os[d] = x.shape[d];
    } else if (req[d] == -1) {
      if (infer >= 0) Fail("Reshape with two -1");
      infer = static_cast<int>(d);
      os[d] = 1;
      continue;
    } else {
      os[d] = req[d];
    }
    known *= os[d];
  }
  if (infer >= 0) os[Sz(infer)] = known == 0 ? 0 : x.numel() / known;
  NetTensor out;
  out.is_int = x.is_int;
  out.shape = os;
  if (out.numel() != x.numel()) Fail("Reshape element count mismatch");
  out.f = x.f;
  out.i = x.i;
  return out;
}

NetTensor MatMul(const NetTensor& a, const NetTensor& b) {
  if (a.shape.size() < 2 || b.shape.size() < 2) Fail("MatMul expects rank >= 2");
  const std::int64_t m = a.shape[a.shape.size() - 2], k = a.shape.back();
  const std::int64_t k2 = b.shape[b.shape.size() - 2], n = b.shape.back();
  if (k != k2) Fail("MatMul inner dims");
  Shape os = a.shape;
  os.back() = n;
  NetTensor out = FloatTensor(os);
  if (b.shape.size() == 2) {
    const std::int64_t rows = a.numel() / k;
    Gemm(a.f.data(), b.f.data(), nullptr, out.f.data(), static_cast<int>(rows),
         static_cast<int>(n), static_cast<int>(k));
    return out;
  }
  if (a.shape.size() != b.shape.size() ||
      !std::equal(a.shape.begin(), a.shape.end() - 2, b.shape.begin()))
    Fail("MatMul batch broadcast is not implemented");
  const std::int64_t batches = a.numel() / (m * k);
  for (std::int64_t bt = 0; bt < batches; ++bt)
    Gemm(a.f.data() + bt * m * k, b.f.data() + bt * k * n, nullptr, out.f.data() + bt * m * n,
         static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
  return out;
}

NetTensor Softmax(const NetTensor& x, std::int64_t axis_attr) {
  // opset < 13: softmax over the flattened dims [axis, rank).
  const std::int64_t axis = Axis(axis_attr, x.shape.size());
  std::int64_t inner = 1;
  for (std::size_t d = Sz(axis); d < x.shape.size(); ++d) inner *= x.shape[d];
  const std::int64_t rows = x.numel() / inner;
  NetTensor out = FloatTensor(x.shape);
#pragma omp parallel for schedule(static) if (rows > 64)
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* xp = x.f.data() + r * inner;
    float* yp = out.f.data() + r * inner;
    float mx = xp[0];
    for (std::int64_t j = 1; j < inner; ++j) mx = std::max(mx, xp[j]);
    float sum = 0.0f;
    for (std::int64_t j = 0; j < inner; ++j) {
      yp[j] = std::exp(xp[j] - mx);
      sum += yp[j];
    }
    for (std::int64_t j = 0; j < inner; ++j) yp[j] /= sum;
  }
  return out;
}

NetTensor ReduceMeanLast(const NetTensor& x, const Attrs& at) {
  const auto axes = at.List("axes", {-1});
  if (axes.size() != 1 ||
      Axis(axes[0], x.shape.size()) != static_cast<std::int64_t>(x.shape.size()) - 1)
    Fail("ReduceMean only over the last axis");
  const bool keep = at.Int("keepdims", 1) != 0;
  const std::int64_t inner = x.shape.back();
  const std::int64_t rows = x.numel() / inner;
  Shape os = x.shape;
  if (keep) os.back() = 1;
  else os.pop_back();
  NetTensor out = FloatTensor(os);
  for (std::int64_t r = 0; r < rows; ++r) {
    const float* xp = x.f.data() + r * inner;
    float s = 0.0f;
    for (std::int64_t j = 0; j < inner; ++j) s += xp[j];
    out.f[Sz(r)] = s / static_cast<float>(inner);
  }
  return out;
}

NetTensor Squeeze(const NetTensor& x, const std::vector<std::int64_t>& axes) {
  NetTensor out = x;
  Shape os;
  for (std::size_t d = 0; d < x.shape.size(); ++d) {
    bool drop = axes.empty() && x.shape[d] == 1;
    for (std::int64_t a : axes)
      if (Axis(a, x.shape.size()) == static_cast<std::int64_t>(d)) drop = true;
    if (drop && x.shape[d] != 1) Fail("Squeeze of a dim != 1");
    if (!drop) os.push_back(x.shape[d]);
  }
  out.shape = os;
  return out;
}

class CpuOcrNet final : public OcrNet {
public:
  bool Init(const std::filesystem::path& dir, const std::string& stem, std::string* error) {
    TensorFile weights;
    if (!weights.Load(dir / (stem + ".safetensors"), error)) return false;
    std::ifstream in(dir / (stem + ".json"), std::ios::binary);
    if (!in) {
      if (error) *error = "missing " + (dir / (stem + ".json")).string();
      return false;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    try {
      const json::Value arch = json::parse(ss.str());
      std::unordered_map<std::string, int> names;  // run slots (>= 0), constants (< 0)
      for (const auto& [name, t] : weights.all()) {
        NetTensor c;
        c.shape = t.shape;
        c.f = t.f32;
        if (c.f.size() != Sz(c.numel())) Fail("tensor " + name + " is not F32");
        consts_.push_back(std::move(c));
        names[name] = -static_cast<int>(consts_.size());
      }
      if (const json::Value* ic = arch.find("int_constants"); ic != nullptr) {
        for (const auto& [name, t] : ic->members()) {
          NetTensor c;
          c.is_int = true;
          for (const auto& d : t.find("shape")->items())
            c.shape.push_back(static_cast<std::int64_t>(d.as_double()));
          for (const auto& v : t.find("data")->items()) {
            const double dv = v.as_double();
            c.i.push_back(dv >= 9.2e18    ? std::numeric_limits<std::int64_t>::max()
                          : dv <= -9.2e18 ? std::numeric_limits<std::int64_t>::min()
                                          : static_cast<std::int64_t>(dv));
          }
          if (c.i.size() != Sz(c.numel())) Fail("int constant " + name + " size");
          consts_.push_back(std::move(c));
          names[name] = -static_cast<int>(consts_.size());
        }
      }
      const std::string input = arch.member_str("input");
      const std::string output = arch.member_str("output");
      input_ = slots_++;
      names[input] = input_;
      for (const auto& nj : arch.find("nodes")->items()) {
        Node node;
        node.op = ParseOp(nj.member_str("op"));
        for (const auto& i : nj.find("inputs")->items()) {
          if (i.str().empty()) {
            node.in.push_back(kNone);
            continue;
          }
          auto it = names.find(i.str());
          if (it == names.end()) Fail("input " + i.str() + " is not produced");
          node.in.push_back(it->second);
        }
        const auto& outs = nj.find("outputs")->items();
        if (outs.size() != 1) Fail("nodes with several outputs are not implemented");
        node.name = outs[0].str();
        names[node.name] = slots_;
        node.out.push_back(slots_++);
        if (const json::Value* at = nj.find("attrs"); at != nullptr) {
          for (const auto& [key, v] : at->members()) {
            if (v.is_string()) {
              node.attrs.strings[key] = v.str();
            } else if (v.is_array()) {
              std::vector<std::int64_t> l;
              for (const auto& e : v.items()) l.push_back(static_cast<std::int64_t>(e.as_double()));
              node.attrs.ints[key] = l;
            } else if (v.is_number()) {
              const double dv = v.as_double();
              node.attrs.floats[key] = dv;
              if (std::floor(dv) == dv) node.attrs.ints[key] = {static_cast<std::int64_t>(dv)};
            }
          }
        }
        nodes_.push_back(std::move(node));
      }
      auto it = names.find(output);
      if (it == names.end() || it->second < 0) Fail("output " + output + " is not produced");
      output_ = it->second;
      // Free every run slot after its last use.
      std::vector<int> last(Sz(slots_), -1);
      for (std::size_t k = 0; k < nodes_.size(); ++k)
        for (int s : nodes_[k].in)
          if (s >= 0) last[Sz(s)] = static_cast<int>(k);
      for (int s = 0; s < slots_; ++s)
        if (last[Sz(s)] >= 0 && s != output_) nodes_[Sz(last[Sz(s)])].release.push_back(s);
    } catch (const std::exception& e) {
      if (error) *error = "ocr model " + stem + ": " + e.what();
      return false;
    }
    return true;
  }

  NetTensor Run(NetTensor input) const override {
    std::vector<NetTensor> slots(Sz(slots_));
    slots[Sz(input_)] = std::move(input);
    for (const Node& n : nodes_) {
      NetTensor y;
      try {
        y = Eval(n, slots);
      } catch (const std::exception& e) {
        Fail(std::string(e.what()) + " (node " + n.name + ")");
      }
      slots[Sz(n.out[0])] = std::move(y);
      for (int s : n.release) slots[Sz(s)] = NetTensor{};
    }
    return std::move(slots[Sz(output_)]);
  }

private:
  const NetTensor& Get(const std::vector<NetTensor>& slots, int id) const {
    if (id >= 0) return slots[Sz(id)];
    if (id == kNone) Fail("missing input");
    return consts_[Sz(-id - 1)];
  }
  const NetTensor* Opt(const std::vector<NetTensor>& slots, const Node& n, std::size_t k) const {
    if (k >= n.in.size() || n.in[k] == kNone) return nullptr;
    return &Get(slots, n.in[k]);
  }

  NetTensor Eval(const Node& n, const std::vector<NetTensor>& s) const {
    const NetTensor& x = Get(s, n.in[0]);
    auto in = [&](std::size_t k) -> const NetTensor& { return Get(s, n.in.at(k)); };
    switch (n.op) {
      case Op::kAdd: return Binary(Bin::kAdd, x, in(1));
      case Op::kSub: return Binary(Bin::kSub, x, in(1));
      case Op::kMul: return Binary(Bin::kMul, x, in(1));
      case Op::kDiv: return Binary(Bin::kDiv, x, in(1));
      case Op::kPow: return Binary(Bin::kPow, x, in(1));
      case Op::kRelu: return Unary(x, [](float v) { return v > 0.0f ? v : 0.0f; });
      case Op::kSigmoid: return Unary(x, [](float v) { return 1.0f / (1.0f + std::exp(-v)); });
      case Op::kSqrt: return Unary(x, [](float v) { return std::sqrt(v); });
      case Op::kHardSigmoid: {
        const float alpha = static_cast<float>(n.attrs.Float("alpha", 0.2));
        const float beta = static_cast<float>(n.attrs.Float("beta", 0.5));
        return Unary(x, [alpha, beta](float v) {
          return std::max(0.0f, std::min(1.0f, alpha * v + beta));
        });
      }
      case Op::kClip: {
        float lo = -std::numeric_limits<float>::infinity();
        float hi = std::numeric_limits<float>::infinity();
        if (const NetTensor* t = Opt(s, n, 1)) lo = t->f.at(0);
        else lo = static_cast<float>(n.attrs.Float("min", lo));
        if (const NetTensor* t = Opt(s, n, 2)) hi = t->f.at(0);
        else hi = static_cast<float>(n.attrs.Float("max", hi));
        return Unary(x, [lo, hi](float v) { return std::min(hi, std::max(lo, v)); });
      }
      case Op::kConv: return Conv(x, in(1), Opt(s, n, 2), n.attrs);
      case Op::kConvTranspose: return ConvTranspose(x, in(1), Opt(s, n, 2), n.attrs);
      case Op::kBatchNorm:
        return BatchNorm(x, in(1), in(2), in(3), in(4), n.attrs.Float("epsilon", 1e-5));
      case Op::kGlobalAveragePool: return GlobalAveragePool(x);
      case Op::kAveragePool: return AveragePool(x, n.attrs);
      case Op::kResize: return Resize(x, Opt(s, n, 2), Opt(s, n, 3));
      case Op::kConcat: {
        std::vector<const NetTensor*> xs;
        for (int id : n.in) xs.push_back(&Get(s, id));
        return Concat(xs, n.attrs.Int("axis", 0));
      }
      case Op::kShape: {
        NetTensor t;
        t.is_int = true;
        t.shape = {static_cast<std::int64_t>(x.shape.size())};
        t.i = x.shape;
        return t;
      }
      case Op::kCast: {
        const std::int64_t to = n.attrs.Int("to", 1);
        NetTensor y;
        y.shape = x.shape;
        if (to == 1) {
          if (!x.is_int) return x;
          for (std::int64_t v : x.i) y.f.push_back(static_cast<float>(v));
          return y;
        }
        y.is_int = true;
        if (x.is_int) {
          y.i = x.i;
          if (to == 6)
            for (auto& v : y.i) v = static_cast<std::int32_t>(v);
        } else {
          for (float v : x.f) y.i.push_back(static_cast<std::int64_t>(v));
        }
        return y;
      }
      case Op::kSlice: return Slice(x, in(1), Opt(s, n, 2), Opt(s, n, 3), Opt(s, n, 4));
      case Op::kSqueeze: return Squeeze(x, n.attrs.List("axes", {}));
      case Op::kReshape: return Reshape(x, in(1));
      case Op::kTranspose: return Transpose(x, n.attrs.List("perm", {}));
      case Op::kReduceMean: return ReduceMeanLast(x, n.attrs);
      case Op::kMatMul: return MatMul(x, in(1));
      case Op::kSoftmax: return Softmax(x, n.attrs.Int("axis", 1));
    }
    Fail("unhandled op");
  }

  std::vector<NetTensor> consts_;
  std::vector<Node> nodes_;
  int slots_{0};
  int input_{0};
  int output_{0};
};

}  // namespace

std::unique_ptr<OcrNet> OcrNet::Load(const std::filesystem::path& dir, const std::string& stem,
                                     std::string* error) {
  auto net = std::make_unique<CpuOcrNet>();
  if (!net->Init(dir, stem, error)) return nullptr;
  return net;
}

}  // namespace gufo::models::ccip::cpu
