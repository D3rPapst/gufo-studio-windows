// WD14 SwinV2 tagger, CPU fp32 (see wd14.hpp). Linear layers run through a
// packed AVX2/FMA GEMM (B packed in 16-column panels at load time, 6x16
// register tiles); window attention per (window, head) in parallel.
// Written with AI assistance (Claude Code).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <gufo/ext/json.hpp>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define GUFO_WD14_AVX2 1
#endif

#include "ccip/cpu/kernels.hpp"
#include "ccip/tensors.hpp"
#include "ccip/wd14.hpp"

namespace gufo::models::ccip {
namespace {

namespace json = gufo::json;

constexpr int kPanel = 16;  // B columns per packed panel
constexpr int kTileRows = 6;
constexpr int kRowChunk = 48;

// y = x @ W + b with W [k, n] packed as panels [ceil(n/16)][k][16].
struct Linear {
  int k{0};
  int n{0};
  std::vector<float> packed;
  std::vector<float> bias;  // empty: none
};

Linear MakeLinear(const std::vector<float>& w, int k, int n, const std::vector<float>* bias) {
  Linear l;
  l.k = k;
  l.n = n;
  const int panels = (n + kPanel - 1) / kPanel;
  l.packed.assign(static_cast<std::size_t>(panels) * k * kPanel, 0.0f);
  for (int p = 0; p < panels; ++p) {
    for (int kk = 0; kk < k; ++kk) {
      float* dst = l.packed.data() + (static_cast<std::size_t>(p) * k + kk) * kPanel;
      for (int j = 0; j < kPanel; ++j) {
        const int col = p * kPanel + j;
        if (col < n) dst[j] = w[static_cast<std::size_t>(kk) * n + col];
      }
    }
  }
  if (bias != nullptr) l.bias = *bias;
  return l;
}

// rows (1..6) x 16 tile: c[r][0..16) = bias + sum_k a[r][k] * panel[k][..].
template <int R>
void Tile(const float* a, int lda, const float* panel, int k, const float* bias16, float* out16,
          int ldo, int cols) {
#ifdef GUFO_WD14_AVX2
  __m256 acc[R][2];
  for (int r = 0; r < R; ++r) {
    acc[r][0] = bias16 != nullptr ? _mm256_loadu_ps(bias16) : _mm256_setzero_ps();
    acc[r][1] = bias16 != nullptr ? _mm256_loadu_ps(bias16 + 8) : _mm256_setzero_ps();
  }
  for (int p = 0; p < k; ++p) {
    const __m256 b0 = _mm256_loadu_ps(panel + static_cast<std::size_t>(p) * kPanel);
    const __m256 b1 = _mm256_loadu_ps(panel + static_cast<std::size_t>(p) * kPanel + 8);
    for (int r = 0; r < R; ++r) {
      const __m256 av = _mm256_broadcast_ss(a + static_cast<std::size_t>(r) * lda + p);
      acc[r][0] = _mm256_fmadd_ps(av, b0, acc[r][0]);
      acc[r][1] = _mm256_fmadd_ps(av, b1, acc[r][1]);
    }
  }
  for (int r = 0; r < R; ++r) {
    float* o = out16 + static_cast<std::size_t>(r) * ldo;
    if (cols == kPanel) {
      _mm256_storeu_ps(o, acc[r][0]);
      _mm256_storeu_ps(o + 8, acc[r][1]);
    } else {
      alignas(32) float tmp[kPanel];
      _mm256_store_ps(tmp, acc[r][0]);
      _mm256_store_ps(tmp + 8, acc[r][1]);
      std::memcpy(o, tmp, sizeof(float) * cols);
    }
  }
#else
  for (int r = 0; r < R; ++r) {
    float acc[kPanel];
    for (int j = 0; j < kPanel; ++j) acc[j] = bias16 != nullptr ? bias16[j] : 0.0f;
    const float* ar = a + static_cast<std::size_t>(r) * lda;
    for (int p = 0; p < k; ++p) {
      const float av = ar[p];
      const float* b = panel + static_cast<std::size_t>(p) * kPanel;
      for (int j = 0; j < kPanel; ++j) acc[j] += av * b[j];
    }
    std::memcpy(out16 + static_cast<std::size_t>(r) * ldo, acc, sizeof(float) * cols);
  }
#endif
}

// y[m, n] = x[m, k] @ W (+ b).
void Apply(const Linear& l, const float* x, int m, float* y) {
  const int panels = (l.n + kPanel - 1) / kPanel;
  const int chunks = (m + kRowChunk - 1) / kRowChunk;
  // Bias padded to whole panels.
  std::vector<float> bias;
  if (!l.bias.empty()) {
    bias.assign(static_cast<std::size_t>(panels) * kPanel, 0.0f);
    std::copy(l.bias.begin(), l.bias.end(), bias.begin());
  }
  const float* bp = bias.empty() ? nullptr : bias.data();
#pragma omp parallel for schedule(dynamic, 4)
  for (int t = 0; t < chunks * panels; ++t) {
    const int chunk = t / panels;
    const int p = t % panels;
    const int r0 = chunk * kRowChunk;
    const int r1 = std::min(m, r0 + kRowChunk);
    const int cols = std::min(kPanel, l.n - p * kPanel);
    const float* panel = l.packed.data() + static_cast<std::size_t>(p) * l.k * kPanel;
    const float* b16 = bp != nullptr ? bp + static_cast<std::size_t>(p) * kPanel : nullptr;
    for (int r = r0; r < r1; r += kTileRows) {
      const float* a = x + static_cast<std::size_t>(r) * l.k;
      float* o = y + static_cast<std::size_t>(r) * l.n + static_cast<std::size_t>(p) * kPanel;
      switch (std::min(kTileRows, r1 - r)) {
        case 6: Tile<6>(a, l.k, panel, l.k, b16, o, l.n, cols); break;
        case 5: Tile<5>(a, l.k, panel, l.k, b16, o, l.n, cols); break;
        case 4: Tile<4>(a, l.k, panel, l.k, b16, o, l.n, cols); break;
        case 3: Tile<3>(a, l.k, panel, l.k, b16, o, l.n, cols); break;
        case 2: Tile<2>(a, l.k, panel, l.k, b16, o, l.n, cols); break;
        default: Tile<1>(a, l.k, panel, l.k, b16, o, l.n, cols); break;
      }
    }
  }
}

// LayerNorm with weight and bias over rows x dim; y may alias x.
void LayerNormBias(const float* x, const float* w, const float* b, float* y, int rows, int dim,
                   float eps) {
#pragma omp parallel for schedule(static)
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<std::size_t>(r) * dim;
    float* yr = y + static_cast<std::size_t>(r) * dim;
    double mean = 0.0;
    for (int i = 0; i < dim; ++i) mean += xr[i];
    mean /= dim;
    double var = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double d = xr[i] - mean;
      var += d * d;
    }
    var /= dim;
    const float inv = static_cast<float>(1.0 / std::sqrt(var + eps));
    const float m = static_cast<float>(mean);
    for (int i = 0; i < dim; ++i) yr[i] = (xr[i] - m) * inv * w[i] + b[i];
  }
}

// x += LayerNorm(y) per row.
void AddLayerNorm(float* x, const float* y, const float* w, const float* b, int rows, int dim,
                  float eps) {
#pragma omp parallel for schedule(static)
  for (int r = 0; r < rows; ++r) {
    const float* yr = y + static_cast<std::size_t>(r) * dim;
    float* xr = x + static_cast<std::size_t>(r) * dim;
    double mean = 0.0;
    for (int i = 0; i < dim; ++i) mean += yr[i];
    mean /= dim;
    double var = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double d = yr[i] - mean;
      var += d * d;
    }
    var /= dim;
    const float inv = static_cast<float>(1.0 / std::sqrt(var + eps));
    const float m = static_cast<float>(mean);
    for (int i = 0; i < dim; ++i) xr[i] += (yr[i] - m) * inv * w[i] + b[i];
  }
}

void GeluTanh(float* x, std::size_t n) {
  const auto count = static_cast<std::ptrdiff_t>(n);
  constexpr float kC = 0.7978845608028654f;  // sqrt(2 / pi)
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t i = 0; i < count; ++i) {
    const float v = x[i];
    x[i] = 0.5f * v * (1.0f + std::tanh(kC * (v + 0.044715f * v * v * v)));
  }
}

struct Block {
  Linear qkv, proj, fc1, fc2;
  const float* scale{nullptr};    // [heads]
  const float* relbias{nullptr};  // [heads, N, N]
  const float *n1w{nullptr}, *n1b{nullptr}, *n2w{nullptr}, *n2b{nullptr};
  int shift{0};
};

struct Stage {
  int dim{0}, heads{0}, window{0}, res{0};
  bool down{false};
  Linear down_lin;
  const float *down_nw{nullptr}, *down_nb{nullptr};
  const float* mask{nullptr};  // [windows, N, N] for shifted blocks
  std::vector<Block> blocks;
};

class CpuWd14 final : public Wd14Tagger {
public:
  bool Init(const std::filesystem::path& dir, const std::string& stem, std::string* error) {
    if (!weights_.Load(dir / (stem + ".safetensors"), error)) return false;
    std::ifstream in(dir / (stem + ".json"), std::ios::binary);
    if (!in) {
      if (error) *error = "missing " + (dir / (stem + ".json")).string();
      return false;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    try {
      const json::Value arch = json::parse(ss.str());
      if (arch.member_str("arch") != "swinv2_wd14") throw std::runtime_error("not a swinv2_wd14 export");
      size_ = static_cast<int>(arch.member_size("img_size"));
      patch_ = static_cast<int>(arch.member_size("patch"));
      embed_ = static_cast<int>(arch.member_size("embed_dim"));
      eps_ = static_cast<float>(arch.member_double("eps"));
      const json::Value* input = arch.find("input");
      if (input == nullptr || input->member_str("order") != "BGR")
        throw std::runtime_error("input order must be BGR");
      in_scale_ = static_cast<float>(input->member_double("scale"));
      in_offset_ = static_cast<float>(input->member_double("offset"));
      features_ = static_cast<int>(arch.member_size("features"));
      classes_ = static_cast<int>(arch.member_size("classes"));
      patch_w_ = F32("patch.w", static_cast<std::int64_t>(embed_) * 3 * patch_ * patch_);
      patch_b_ = F32("patch.b", embed_);
      patch_nw_ = F32("patch.norm.w", embed_);
      patch_nb_ = F32("patch.norm.b", embed_);
      int s = 0;
      for (const json::Value& sj : arch.find("stages")->items()) {
        Stage st;
        st.dim = static_cast<int>(sj.member_size("dim"));
        st.heads = static_cast<int>(sj.member_size("heads"));
        st.window = static_cast<int>(sj.member_size("window"));
        st.res = static_cast<int>(sj.member_size("resolution"));
        st.down = sj.find("downsample")->as_bool();
        const int hidden = static_cast<int>(sj.member_size("mlp_hidden"));
        const int c = st.dim;
        const int n_tok = st.window * st.window;
        const std::string sp = "s" + std::to_string(s);
        if (st.down) {
          st.down_lin = Lin(sp + ".down", 2 * c, c, false);
          st.down_nw = F32(sp + ".down.norm.w", c);
          st.down_nb = F32(sp + ".down.norm.b", c);
        }
        int b = 0;
        bool any_shift = false;
        for (const json::Value& sh : sj.find("shifts")->items()) {
          Block bl;
          const std::string bp = sp + ".b" + std::to_string(b);
          bl.shift = static_cast<int>(sh.as_double());
          any_shift = any_shift || bl.shift != 0;
          bl.qkv = Lin(bp + ".qkv", c, 3 * c, true);
          bl.proj = Lin(bp + ".proj", c, c, true);
          bl.fc1 = Lin(bp + ".fc1", c, hidden, true);
          bl.fc2 = Lin(bp + ".fc2", hidden, c, true);
          bl.scale = F32(bp + ".scale", st.heads);
          bl.relbias = F32(bp + ".relbias", static_cast<std::int64_t>(st.heads) * n_tok * n_tok);
          bl.n1w = F32(bp + ".norm1.w", c);
          bl.n1b = F32(bp + ".norm1.b", c);
          bl.n2w = F32(bp + ".norm2.w", c);
          bl.n2b = F32(bp + ".norm2.b", c);
          st.blocks.push_back(std::move(bl));
          ++b;
        }
        if (any_shift) {
          const int nw = (st.res / st.window) * (st.res / st.window);
          st.mask = F32(sp + ".mask", static_cast<std::int64_t>(nw) * n_tok * n_tok);
        }
        if (st.res % st.window != 0 || st.dim % st.heads != 0)
          throw std::runtime_error("bad stage geometry");
        stages_.push_back(std::move(st));
        ++s;
      }
      norm_w_ = F32("norm.w", features_);
      norm_b_ = F32("norm.b", features_);
      head_ = Lin("head", features_, classes_, true);
      const json::Value* tags = arch.find("tags");
      for (const auto& n : tags->find("names")->items()) names_.push_back(n.str());
      for (const auto& c : tags->find("categories")->items())
        categories_.push_back(static_cast<int>(c.as_double()));
      if (static_cast<int>(names_.size()) != classes_ ||
          static_cast<int>(categories_.size()) != classes_)
        throw std::runtime_error("tag list does not match the classes");
      if (size_ != patch_ * stages_.front().res) throw std::runtime_error("bad patch geometry");
    } catch (const std::exception& e) {
      if (error) *error = std::string("wd14 ") + stem + ": " + e.what();
      return false;
    }
    return true;
  }

  int InputSize() const override { return size_; }

  void Forward(const float* bgr, std::vector<float>* probs,
               std::vector<float>* embedding) const override {
    // Input: x / scale - offset, NCHW (the export's Transpose, Div, Sub).
    const int sz = size_;
    std::vector<float> chw(static_cast<std::size_t>(3) * sz * sz);
    for (int c = 0; c < 3; ++c) {
      float* dst = chw.data() + static_cast<std::size_t>(c) * sz * sz;
      for (int i = 0; i < sz * sz; ++i) dst[i] = bgr[static_cast<std::size_t>(i) * 3 + c] / in_scale_ - in_offset_;
    }
    int res = stages_.front().res;
    std::vector<float> x(static_cast<std::size_t>(res) * res * embed_);
    int oh = 0, ow = 0;
    cpu::ConvToNhwc(chw.data(), true, sz, sz, 3, patch_w_, patch_b_, embed_, patch_, patch_, 0,
                    x.data(), &oh, &ow);
    LayerNormBias(x.data(), patch_nw_, patch_nb_, x.data(), res * res, embed_, eps_);
    int dim = embed_;
    for (const Stage& st : stages_) {
      if (st.down) {
        const int r2 = res / 2;
        std::vector<float> merged(static_cast<std::size_t>(r2) * r2 * 4 * dim);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < r2; ++i) {
          for (int j = 0; j < r2; ++j) {
            float* dst = merged.data() + (static_cast<std::size_t>(i) * r2 + j) * 4 * dim;
            for (int dx = 0; dx < 2; ++dx) {
              for (int dy = 0; dy < 2; ++dy) {
                const float* src =
                    x.data() + (static_cast<std::size_t>(2 * i + dy) * res + (2 * j + dx)) * dim;
                std::memcpy(dst + static_cast<std::size_t>(dx * 2 + dy) * dim, src, sizeof(float) * dim);
              }
            }
          }
        }
        res = r2;
        dim *= 2;
        x.assign(static_cast<std::size_t>(res) * res * dim, 0.0f);
        Apply(st.down_lin, merged.data(), res * res, x.data());
        LayerNormBias(x.data(), st.down_nw, st.down_nb, x.data(), res * res, dim, eps_);
      }
      for (const Block& bl : st.blocks) RunBlock(st, bl, x.data());
    }
    const int tokens = res * res;
    LayerNormBias(x.data(), norm_w_, norm_b_, x.data(), tokens, dim, eps_);
    std::vector<float> emb(static_cast<std::size_t>(dim), 0.0f);
    std::vector<double> sum(static_cast<std::size_t>(dim), 0.0);
    for (int t = 0; t < tokens; ++t) {
      const float* xr = x.data() + static_cast<std::size_t>(t) * dim;
      for (int c = 0; c < dim; ++c) sum[c] += xr[c];
    }
    for (int c = 0; c < dim; ++c) emb[c] = static_cast<float>(sum[c] / tokens);
    std::vector<float> logits(static_cast<std::size_t>(classes_));
    Apply(head_, emb.data(), 1, logits.data());
    for (float& v : logits) v = 1.0f / (1.0f + std::exp(-v));
    if (probs != nullptr) *probs = std::move(logits);
    if (embedding != nullptr) *embedding = std::move(emb);
  }

  Wd14Tags Select(const std::vector<float>& probs, double general_threshold,
                  double character_threshold) const override {
    Wd14Tags out;
    for (int i = 0; i < classes_ && i < static_cast<int>(probs.size()); ++i) {
      const double p = probs[static_cast<std::size_t>(i)];
      switch (categories_[static_cast<std::size_t>(i)]) {
        case 9: out.rating.emplace_back(names_[i], probs[i]); break;
        case 0:
          if (p > general_threshold) out.general.emplace_back(names_[i], probs[i]);
          break;
        case 4:
          if (p > character_threshold) out.character.emplace_back(names_[i], probs[i]);
          break;
        default: break;
      }
    }
    return out;
  }

private:
  const float* F32(const std::string& name, std::int64_t numel) const {
    return weights_.F32(name, numel).data();
  }

  // Packs <name>.w [k, n] (and .b) and frees the unpacked copy.
  Linear Lin(const std::string& name, int k, int n, bool bias) {
    const std::vector<float>& w = weights_.F32(name + ".w", static_cast<std::int64_t>(k) * n);
    const std::vector<float>* b = bias ? &weights_.F32(name + ".b", n) : nullptr;
    Linear l = MakeLinear(w, k, n, b);
    const_cast<std::vector<float>&>(w) = std::vector<float>();  // owned by weights_; packed copy kept
    return l;
  }

  void RunBlock(const Stage& st, const Block& bl, float* x) const {
    const int res = st.res, c = st.dim, ws = st.window, heads = st.heads;
    const int hd = c / heads;
    const int n_tok = ws * ws;
    const int per_row = res / ws;
    const int windows = per_row * per_row;
    const int tokens = res * res;
    const int shift = bl.shift;
    // Windows of the image rolled by -shift: token (ty, tx) of window (wy, wx)
    // is pixel ((wy*ws + ty + shift) % res, (wx*ws + tx + shift) % res).
    std::vector<int> src(static_cast<std::size_t>(tokens));
    for (int w = 0; w < windows; ++w) {
      const int wy = w / per_row, wx = w % per_row;
      for (int t = 0; t < n_tok; ++t) {
        const int y = (wy * ws + t / ws + shift) % res;
        const int xx = (wx * ws + t % ws + shift) % res;
        src[static_cast<std::size_t>(w) * n_tok + t] = y * res + xx;
      }
    }
    std::vector<float> win(static_cast<std::size_t>(tokens) * c);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < tokens; ++i) {
      std::memcpy(win.data() + static_cast<std::size_t>(i) * c,
                  x + static_cast<std::size_t>(src[i]) * c, sizeof(float) * c);
    }
    std::vector<float> qkv(static_cast<std::size_t>(tokens) * 3 * c);
    Apply(bl.qkv, win.data(), tokens, qkv.data());
    // win now receives the attention output (heads concatenated).
#pragma omp parallel
    {
      std::vector<float> q(static_cast<std::size_t>(n_tok) * hd), k(q.size()), v(q.size());
      std::vector<float> s(static_cast<std::size_t>(n_tok) * n_tok);
#pragma omp for schedule(dynamic, 1)
      for (int task = 0; task < windows * heads; ++task) {
        const int w = task / heads, h = task % heads;
        const float* base = qkv.data() + static_cast<std::size_t>(w) * n_tok * 3 * c;
        for (int t = 0; t < n_tok; ++t) {
          const float* row = base + static_cast<std::size_t>(t) * 3 * c + static_cast<std::size_t>(h) * hd;
          float* qd = q.data() + static_cast<std::size_t>(t) * hd;
          float* kd = k.data() + static_cast<std::size_t>(t) * hd;
          float* vd = v.data() + static_cast<std::size_t>(t) * hd;
          float qn = 0.0f, kn = 0.0f;
          for (int d = 0; d < hd; ++d) {
            qn += row[d] * row[d];
            kn += row[c + d] * row[c + d];
          }
          const float qi = 1.0f / std::max(std::sqrt(qn), 1e-12f);
          const float ki = 1.0f / std::max(std::sqrt(kn), 1e-12f);
          for (int d = 0; d < hd; ++d) {
            qd[d] = row[d] * qi;
            kd[d] = row[c + d] * ki;
            vd[d] = row[2 * c + d];
          }
        }
        const float scale = bl.scale[h];
        const float* rb = bl.relbias + static_cast<std::size_t>(h) * n_tok * n_tok;
        const float* mk = (shift != 0 && st.mask != nullptr)
                              ? st.mask + static_cast<std::size_t>(w) * n_tok * n_tok
                              : nullptr;
        for (int i = 0; i < n_tok; ++i) {
          const float* qi = q.data() + static_cast<std::size_t>(i) * hd;
          float* si = s.data() + static_cast<std::size_t>(i) * n_tok;
          const float* rbi = rb + static_cast<std::size_t>(i) * n_tok;
          float mx = -1e30f;
          for (int j = 0; j < n_tok; ++j) {
            const float* kj = k.data() + static_cast<std::size_t>(j) * hd;
            float dot = 0.0f;
#pragma omp simd reduction(+ : dot)
            for (int d = 0; d < hd; ++d) dot += qi[d] * kj[d];
            float val = dot * scale + rbi[j];
            if (mk != nullptr) val += mk[static_cast<std::size_t>(i) * n_tok + j];
            si[j] = val;
            mx = std::max(mx, val);
          }
          float sum = 0.0f;
          for (int j = 0; j < n_tok; ++j) {
            si[j] = std::exp(si[j] - mx);
            sum += si[j];
          }
          const float inv = 1.0f / sum;
          float* out = win.data() + (static_cast<std::size_t>(w) * n_tok + i) * c +
                       static_cast<std::size_t>(h) * hd;
          std::fill(out, out + hd, 0.0f);
          for (int j = 0; j < n_tok; ++j) {
            const float pj = si[j] * inv;
            const float* vj = v.data() + static_cast<std::size_t>(j) * hd;
#pragma omp simd
            for (int d = 0; d < hd; ++d) out[d] += pj * vj[d];
          }
        }
      }
    }
    std::vector<float> proj(static_cast<std::size_t>(tokens) * c);
    Apply(bl.proj, win.data(), tokens, proj.data());
    // Back to image order (roll by +shift), then x += norm1(attn).
#pragma omp parallel for schedule(static)
    for (int i = 0; i < tokens; ++i) {
      std::memcpy(win.data() + static_cast<std::size_t>(src[i]) * c,
                  proj.data() + static_cast<std::size_t>(i) * c, sizeof(float) * c);
    }
    AddLayerNorm(x, win.data(), bl.n1w, bl.n1b, tokens, c, eps_);
    std::vector<float> hidden(static_cast<std::size_t>(tokens) * bl.fc1.n);
    Apply(bl.fc1, x, tokens, hidden.data());
    GeluTanh(hidden.data(), hidden.size());
    Apply(bl.fc2, hidden.data(), tokens, proj.data());
    AddLayerNorm(x, proj.data(), bl.n2w, bl.n2b, tokens, c, eps_);
  }

  TensorFile weights_;
  int size_{448}, patch_{4}, embed_{128}, features_{1024}, classes_{0};
  float eps_{1e-5f}, in_scale_{127.5f}, in_offset_{1.0f};
  const float *patch_w_{nullptr}, *patch_b_{nullptr}, *patch_nw_{nullptr}, *patch_nb_{nullptr};
  const float *norm_w_{nullptr}, *norm_b_{nullptr};
  std::vector<Stage> stages_;
  Linear head_;
  std::vector<std::string> names_;
  std::vector<int> categories_;
};

}  // namespace

std::unique_ptr<Wd14Tagger> Wd14Tagger::Load(const std::filesystem::path& dir,
                                             const std::string& stem, std::string* error) {
  auto net = std::make_unique<CpuWd14>();
  if (!net->Init(dir, stem, error)) return nullptr;
  return net;
}

Wd14Tags Wd14Tagger::Tag(const RgbImage& image, double general_threshold,
                         double character_threshold) const {
  const std::vector<float> input = Wd14Input(image, InputSize());
  std::vector<float> probs;
  Forward(input.data(), &probs, nullptr);
  return Select(probs, general_threshold, character_threshold);
}

std::vector<float> Wd14Input(const RgbImage& image, int size) {
  const int side = std::max(image.width, image.height);
  RgbImage square;
  square.width = side;
  square.height = side;
  square.rgb.assign(static_cast<std::size_t>(side) * side * 3, 255);
  const int left = (side - image.width) / 2;
  const int top = (side - image.height) / 2;
  for (int y = 0; y < image.height; ++y) {
    std::memcpy(square.rgb.data() + (static_cast<std::size_t>(y + top) * side + left) * 3,
                image.rgb.data() + static_cast<std::size_t>(y) * image.width * 3,
                static_cast<std::size_t>(image.width) * 3);
  }
  const RgbImage sized = side == size ? std::move(square) : ResizeBicubic(square, size, size);
  std::vector<float> out(static_cast<std::size_t>(size) * size * 3);
  for (std::size_t i = 0; i < static_cast<std::size_t>(size) * size; ++i) {
    out[i * 3 + 0] = sized.rgb[i * 3 + 2];
    out[i * 3 + 1] = sized.rgb[i * 3 + 1];
    out[i * 3 + 2] = sized.rgb[i * 3 + 0];
  }
  return out;
}

std::array<int, 4> MarginCropBox(const std::array<double, 4>& box, double margin, int width,
                                 int height) {
  const double rand = margin * std::max(box[2] - box[0], box[3] - box[1]);
  // Python round(): half to even (nearbyint in the default rounding mode).
  // Clamped first: a value outside int would be undefined behaviour (the
  // router limits boxes to +-1e6, this keeps other callers safe too).
  const auto r = [](double v) {
    return static_cast<int>(std::nearbyint(std::clamp(v, -1e9, 1e9)));
  };
  return {r(std::max(0.0, box[0] - rand)), r(std::max(0.0, box[1] - rand)),
          r(std::min(static_cast<double>(width), box[2] + rand)),
          r(std::min(static_cast<double>(height), box[3] + rand))};
}

}  // namespace gufo::models::ccip
