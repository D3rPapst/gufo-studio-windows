// gufo_windows addition: DSpark drafter on the GPU (see src/dspark.hpp). Generic GEMMs from kernels.hpp,
// small kernels here. fp32 activations; drafter weights as stored (Q8_0 / BF16 / F32).
#include "src/dspark.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <hip/hip_runtime.h>

#include <gufo/ext/gguf_reader.hpp>

#include "src/hip/kernels.hpp"

namespace gufo::models::inkling2 {
namespace {

constexpr int E = 2304, NL = 3, NH = 24, NKV = 8, HD = 128, FF = 9216, RANK = 256, BLOCK = 8, HN = 16;
constexpr int KVD = NKV * HD, QD = NH * HD;

void Chk(hipError_t e, const char* what) {
  if (e != hipSuccess) throw std::runtime_error(std::string("inkling2 dspark: ") + what + ": " + hipGetErrorString(e));
}

__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }

// out[t] = rms(sum_j w_j F_j) * norm, F = the 4 features of token t ([T][4][E])
__global__ void k_fuse(const float* feat, float w0, float w1, float w2, float w3, const float* norm, float* out, float eps) {
  __shared__ float red[256];
  const size_t t = blockIdx.x;
  const float* f = feat + t * 4 * E;
  float ss = 0.f;
  for (int i = threadIdx.x; i < E; i += blockDim.x) {
    const float v = w0 * f[i] + w1 * f[E + i] + w2 * f[2 * E + i] + w3 * f[3 * E + i];
    ss += v * v;
  }
  red[threadIdx.x] = ss;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if (static_cast<int>(threadIdx.x) < o) red[threadIdx.x] += red[threadIdx.x + o];
    __syncthreads();
  }
  const float r = rsqrtf(red[0] / E + eps);
  for (int i = threadIdx.x; i < E; i += blockDim.x) {
    const float v = w0 * f[i] + w1 * f[E + i] + w2 * f[2 * E + i] + w3 * f[3 * E + i];
    out[t * E + i] = v * r * norm[i];
  }
}

// per-head rms norm (weight w[128]) then RoPE NEOX on (x_i, x_{i+64}) at position pos0 + row; x [rows][heads][128]
__global__ void k_headnorm_rope(float* x, int heads, const float* w, int pos0, float theta, float eps) {
  __shared__ float red[64];
  const size_t row = static_cast<size_t>(blockIdx.x) * heads + blockIdx.y;
  float* v = x + row * HD;
  const int i = threadIdx.x;
  float a = v[i], b = v[i + 64];
  red[i] = a * a + b * b;
  __syncthreads();
  for (int o = 32; o > 0; o >>= 1) {
    if (i < o) red[i] += red[i + o];
    __syncthreads();
  }
  const float r = rsqrtf(red[0] / 128.f + eps);
  a *= r * w[i];
  b *= r * w[i + 64];
  const float ang = static_cast<float>(pos0 + static_cast<int>(blockIdx.x)) * powf(theta, -2.f * i / 128.f);
  const float c = cosf(ang), s = sinf(ang);
  v[i] = a * c - b * s;
  v[i + 64] = a * s + b * c;
}

__global__ void k_store(const float* K, const float* V, float* kc, float* vc, int pos0, int cap) {
  const int t = blockIdx.x;
  const size_t o = static_cast<size_t>((pos0 + t) % cap) * KVD;
  for (int i = threadIdx.x; i < KVD; i += blockDim.x) {
    kc[o + i] = K[static_cast<size_t>(t) * KVD + i];
    vc[o + i] = V[static_cast<size_t>(t) * KVD + i];
  }
}

// block attention (non-causal inside the block): keys = cache positions [lo, hi) then the block's own keys.
// grid (rows, heads), 128 threads (one per head dim).
__global__ void k_attn(const float* q, const float* kb, const float* vb, const float* kc, const float* vc, int cap, int lo,
                       int hi, int rows, float* out) {
  __shared__ float qs[128], lg[128], red[128];
  const int r = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, kv = h / (NH / NKV);
  qs[tid] = q[(static_cast<size_t>(r) * NH + h) * HD + tid];
  __syncthreads();
  float m = -1e30f, l = 0.f, acc = 0.f;
  const float scale = rsqrtf(128.f);
  const int total = (hi - lo) + rows;
  for (int j0 = 0; j0 < total; j0 += 128) {
    const int cnt = min(128, total - j0);
    float s = -1e30f;
    if (tid < cnt) {
      const int j = j0 + tid;
      const float* kr = j < hi - lo ? kc + static_cast<size_t>((lo + j) % cap) * KVD + kv * HD
                                    : kb + static_cast<size_t>(j - (hi - lo)) * KVD + kv * HD;
      float d = 0.f;
      for (int i = 0; i < HD; ++i) d += qs[i] * kr[i];
      s = d * scale;
    }
    lg[tid] = s;
    red[tid] = s;
    __syncthreads();
    for (int o = 64; o > 0; o >>= 1) {
      if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]);
      __syncthreads();
    }
    const float mn = fmaxf(m, red[0]), corr = expf(m - mn);
    acc *= corr;
    l *= corr;
    m = mn;
    __syncthreads();
    float a2 = 0.f, l2 = 0.f;
    for (int i = 0; i < cnt; ++i) {
      const int j = j0 + i;
      const float w = expf(lg[i] - m);
      l2 += w;
      const float* vr = j < hi - lo ? vc + static_cast<size_t>((lo + j) % cap) * KVD + kv * HD
                                    : vb + static_cast<size_t>(j - (hi - lo)) * KVD + kv * HD;
      a2 += w * vr[tid];
    }
    acc += a2;
    l += l2;
    __syncthreads();
  }
  out[(static_cast<size_t>(r) * NH + h) * HD + tid] = l > 0.f ? acc / l : 0.f;
}

__global__ void k_fill_rows(float* X, const float* row0, const float* mask) {
  const int t = blockIdx.x;
  for (int i = threadIdx.x; i < E; i += blockDim.x) X[static_cast<size_t>(t) * E + i] = t == 0 ? row0[i] : mask[i];
}
__global__ void k_add(const float* a, const float* b, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}
__global__ void k_swiglu(float* g, const float* u, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float x = g[i];
    g[i] = x / (1.f + expf(-x)) * u[i];
  }
}
// joint = [state(256) | markov_w1[prev](256) | hidden row (E) | context (E)]
__global__ void k_joint(const float* state, const uint16_t* markov, const int* prev, const float* row, const float* ctx,
                        float* joint) {
  const int tot = 2 * RANK + 2 * E;
  for (int i = threadIdx.x; i < tot; i += blockDim.x) {
    float v;
    if (i < RANK) v = state[i];
    else if (i < 2 * RANK) v = bf2f(markov[static_cast<size_t>(prev[0]) * RANK + (i - RANK)]);
    else if (i < 2 * RANK + E) v = row[i - 2 * RANK];
    else v = ctx[i - 2 * RANK - E];
    joint[i] = v;
  }
}
// c += bias; gate = sigmoid(c0), cand = tanh(c1), state = cand + gate * (state - cand), latent = tanh(c2)
__global__ void k_gru(const float* c, const float* bias, float* state, float* latent) {
  const int i = threadIdx.x;
  if (i >= RANK) return;
  const float gate = 1.f / (1.f + expf(-(c[i] + bias[i])));
  const float cand = tanhf(c[RANK + i] + bias[RANK + i]);
  state[i] = cand + gate * (state[i] - cand);
  latent[i] = tanhf(c[2 * RANK + i] + bias[2 * RANK + i]);
}
// argmax over base + corr + fbias (the softcap is monotonic and does not change it); lowest index on ties
__global__ void k_argmax_part(const float* base, const float* corr, const float* fbias, int n, float* pv, int* pi) {
  __shared__ float rv[256];
  __shared__ int ri[256];
  float best = -INFINITY;
  int bi = n;
  for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
    const float x = base[i] + corr[i] + fbias[i];
    if (x > best || (x == best && i < bi)) { best = x; bi = i; }
  }
  rv[threadIdx.x] = best;
  ri[threadIdx.x] = bi;
  __syncthreads();
  for (int o = 128; o > 0; o >>= 1) {
    if (static_cast<int>(threadIdx.x) < o) {
      const float a = rv[threadIdx.x], b = rv[threadIdx.x + o];
      const int ia = ri[threadIdx.x], ib = ri[threadIdx.x + o];
      if (b > a || (b == a && ib < ia)) { rv[threadIdx.x] = b; ri[threadIdx.x] = ib; }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) { pv[blockIdx.x] = rv[0]; pi[blockIdx.x] = ri[0]; }
}
__global__ void k_argmax_final(const float* pv, const int* pi, int nb, int* out, int slot, int* prev) {
  __shared__ float rv[256];
  __shared__ int ri[256];
  float best = -INFINITY;
  int bi = 1 << 30;
  for (int i = threadIdx.x; i < nb; i += 256) {
    if (pv[i] > best || (pv[i] == best && pi[i] < bi)) { best = pv[i]; bi = pi[i]; }
  }
  rv[threadIdx.x] = best;
  ri[threadIdx.x] = bi;
  __syncthreads();
  for (int o = 128; o > 0; o >>= 1) {
    if (static_cast<int>(threadIdx.x) < o) {
      const float a = rv[threadIdx.x], b = rv[threadIdx.x + o];
      const int ia = ri[threadIdx.x], ib = ri[threadIdx.x + o];
      if (b > a || (b == a && ib < ia)) { rv[threadIdx.x] = b; ri[threadIdx.x] = ib; }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) { out[slot] = ri[0]; prev[0] = ri[0]; }
}
// foldT [256][rows] (F32) -> fold [rows][256] (BF16, round to nearest even)
__global__ void k_fold_pack(const float* in, uint16_t* out, int rows) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<size_t>(rows) * RANK) return;
  const size_t v = i / RANK, r = i % RANK;
  const uint32_t u = __float_as_uint(in[r * rows + v]);
  out[i] = static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

struct DevMat {
  void* p = nullptr;
  int type = 0, k = 0, n = 0;  // n rows of k
};

}  // namespace

struct DSparkDrafter::Impl {
  Engine* target = nullptr;
  Engine::DraftTargetView tv;
  std::shared_ptr<const core::GgufReader> g;
  hipStream_t st{};
  float eps = 1e-6f, theta = 1e6f;
  int window = 1024;
  bool swa[NL] = {true, false, true};
  float fuse[4][4]{};
  float *out_norm = nullptr, *dspark_norm = nullptr, *mask = nullptr, *joint_b = nullptr;
  DevMat markov, joint, fold;  // fold: [head_rows][256] BF16
  float* fbias = nullptr;      // [head_rows]
  struct Layer {
    float *attn_norm, *q_norm, *k_norm, *kctx_norm, *dflare, *ffn_norm;
    DevMat q, k, v, o, kctx, vctx, gate, up, down;
    float *kc = nullptr, *vc = nullptr;
    int cap = 0;
  } L[NL];
  std::vector<void*> allocs;
  // work buffers
  float *fused = nullptr, *kk = nullptr, *vv = nullptr;
  int inject_cap = 0;
  float *X, *xn, *qb, *kb, *vb, *ab, *ob, *gb, *ub, *db, *hid, *row, *row0, *base, *state, *jb, *gates, *latent, *corr, *pv, *hn;
  int *d_anchor, *d_prev, *d_toks, *pi;
  std::vector<long long> hn_pos;
  long long ctx_from = 0;
  double last_ms = 0;

  void* Alloc(std::size_t bytes) {
    void* p = nullptr;
    Chk(hipMalloc(&p, std::max<std::size_t>(bytes, 256)), "alloc");
    allocs.push_back(p);
    return p;
  }
  float* AllocF(std::size_t n) { return static_cast<float*>(Alloc(n * 4)); }
  const core::GgufTensorInfo& T(const std::string& name) const {
    const auto* t = g->FindTensor(name);
    if (!t) throw std::runtime_error("inkling2 dspark: missing tensor " + name);
    return *t;
  }
  void* Up(const core::GgufTensorInfo& t) {
    void* d = Alloc(t.size_bytes);
    Chk(hipMemcpy(d, t.data, t.size_bytes, hipMemcpyHostToDevice), "upload");
    return d;
  }
  float* UpF32(const std::string& name) {  // F32 or BF16 vector -> F32 on the device
    const auto& t = T(name);
    const std::size_t n = t.ElementCount();
    std::vector<float> h(n);
    if (t.type == core::GgmlType::kF32) {
      std::memcpy(h.data(), t.data, n * 4);
    } else if (t.type == core::GgmlType::kBF16) {
      const auto* b = static_cast<const uint16_t*>(t.data);
      for (std::size_t i = 0; i < n; ++i) {
        const uint32_t u = static_cast<uint32_t>(b[i]) << 16;
        std::memcpy(&h[i], &u, 4);
      }
    } else {
      throw std::runtime_error("inkling2 dspark: vector type of " + name);
    }
    float* d = AllocF(n);
    Chk(hipMemcpy(d, h.data(), n * 4, hipMemcpyHostToDevice), "upload vec");
    return d;
  }
  DevMat Mat(const std::string& name) {
    const auto& t = T(name);
    DevMat m;
    m.p = Up(t);
    m.type = static_cast<int>(t.type);
    m.k = static_cast<int>(t.dimensions[0]);
    m.n = t.dimensions.size() > 1 ? static_cast<int>(t.dimensions[1]) : 1;
    return m;
  }
  void Gemm(const DevMat& m, const float* x, int ldx, float* out, int ldo, int t) {
    hip::Gemm(st, m.type, m.p, m.n, m.k, x, ldx, out, ldo, t);
  }

  void Load(const std::string& path, const Options& opt) {
    std::string err;
    core::GgufOpenOptions open;
    open.allow_unsized_tensors = true;
    g = core::GgufReader::OpenFile(path, &err, open);
    if (!g) throw std::runtime_error("inkling2 dspark: " + err);
    if (auto v = g->GetMetadataUint32("dflash.block_count"); !v || *v != NL) throw std::runtime_error("inkling2 dspark: block_count");
    if (auto v = g->GetMetadataUint32("dflash.block_size"); !v || *v != BLOCK) throw std::runtime_error("inkling2 dspark: block_size");
    if (auto v = g->GetMetadataUint32("dflash.dspark.markov_rank"); !v || *v != RANK) throw std::runtime_error("inkling2 dspark: markov_rank");
    if (auto v = g->GetMetadataFloat32("dflash.attention.layer_norm_rms_epsilon")) eps = *v;
    if (auto v = g->GetMetadataFloat32("dflash.rope.freq_base")) theta = *v;
    if (auto v = g->GetMetadataUint32("dflash.attention.sliding_window")) window = static_cast<int>(*v);
    window += opt.window_extra;
    Chk(hipStreamCreateWithFlags(&st, hipStreamNonBlocking), "stream");
    {
      const auto& fz = T("dspark_fuse.weight");
      const float* f = static_cast<const float*>(fz.data);
      for (int r = 0; r < 4; ++r) {
        float mx = -1e30f, s = 0.f, w[4];
        for (int j = 0; j < 4; ++j) mx = std::max(mx, f[r * 4 + j]);
        for (int j = 0; j < 4; ++j) { w[j] = std::exp(f[r * 4 + j] - mx); s += w[j]; }
        for (int j = 0; j < 4; ++j) fuse[r][j] = w[j] / s;
      }
    }
    out_norm = UpF32("output_norm.weight");
    dspark_norm = UpF32("dspark_norm.weight");
    mask = UpF32("mask_embd.weight");
    joint_b = UpF32("joint_proj.bias");
    markov = Mat("markov_w1.weight");
    joint = Mat("joint_proj.weight");
    if (markov.type != static_cast<int>(core::GgmlType::kBF16)) throw std::runtime_error("inkling2 dspark: markov_w1 must be BF16");
    for (int l = 0; l < NL; ++l) {
      auto n = [&](const char* s) { return "blk." + std::to_string(l) + "." + s; };
      Layer& y = L[l];
      y.attn_norm = UpF32(n("attn_norm.weight"));
      y.q_norm = UpF32(n("attn_q_norm.weight"));
      y.k_norm = UpF32(n("attn_k_norm.weight"));
      y.kctx_norm = UpF32(n("attn_k_ctx_norm.weight"));
      y.dflare = UpF32(n("dflare_norm.weight"));
      y.ffn_norm = UpF32(n("ffn_norm.weight"));
      y.q = Mat(n("attn_q.weight"));
      y.k = Mat(n("attn_k.weight"));
      y.v = Mat(n("attn_v.weight"));
      y.o = Mat(n("attn_output.weight"));
      y.kctx = Mat(n("attn_k_ctx.weight"));
      y.vctx = Mat(n("attn_v_ctx.weight"));
      y.gate = Mat(n("ffn_gate.weight"));
      y.up = Mat(n("ffn_up.weight"));
      y.down = Mat(n("ffn_down.weight"));
      y.cap = swa[l] ? 2048 : 16384;
      y.kc = AllocF(static_cast<std::size_t>(y.cap) * KVD);
      y.vc = AllocF(static_cast<std::size_t>(y.cap) * KVD);
    }
    // work buffers
    X = AllocF(BLOCK * E); xn = AllocF(BLOCK * E); qb = AllocF(BLOCK * QD); kb = AllocF(BLOCK * KVD); vb = AllocF(BLOCK * KVD);
    ab = AllocF(BLOCK * QD); ob = AllocF(BLOCK * E); gb = AllocF(static_cast<std::size_t>(BLOCK) * FF);
    ub = AllocF(static_cast<std::size_t>(BLOCK) * FF); db = AllocF(BLOCK * E); hid = AllocF(BLOCK * E); row = AllocF(E);
    row0 = AllocF(E); base = AllocF(static_cast<std::size_t>(BLOCK) * tv.head_rows); state = AllocF(RANK);
    jb = AllocF(2 * RANK + 2 * E); gates = AllocF(3 * RANK); latent = AllocF(RANK); corr = AllocF(tv.head_rows);
    pv = AllocF(256); hn = AllocF(static_cast<std::size_t>(HN) * E);
    d_anchor = static_cast<int*>(Alloc(4)); d_prev = static_cast<int*>(Alloc(4)); d_toks = static_cast<int*>(Alloc(BLOCK * 4));
    pi = static_cast<int*>(Alloc(256 * 4));
    hn_pos.assign(HN, -1);
    // fold the Markov correction into the lm_head (ciru-ai): fold[v] = lm_head[v] . hidden_out, fbias[v] = lm_head[v] . bias
    {
      const auto& ho = T("dspark_hidden_out.weight");  // [256 in][E out]
      const std::size_t cnt = static_cast<std::size_t>(E) * RANK;
      std::vector<float> h(cnt), hT(cnt);
      for (std::size_t i = 0; i < cnt; ++i) {
        if (ho.type == core::GgmlType::kBF16) {
          const uint32_t u = static_cast<uint32_t>(static_cast<const uint16_t*>(ho.data)[i]) << 16;
          std::memcpy(&h[i], &u, 4);
        } else {
          h[i] = static_cast<const float*>(ho.data)[i];
        }
      }
      // h is [E rows][256]: column r of hidden_out = h[e * 256 + r]; GEMM inputs need [256][E]
      for (int e = 0; e < E; ++e)
        for (int r = 0; r < RANK; ++r) hT[static_cast<std::size_t>(r) * E + e] = h[static_cast<std::size_t>(e) * RANK + r];
      float* dhT = AllocF(cnt);
      Chk(hipMemcpy(dhT, hT.data(), cnt * 4, hipMemcpyHostToDevice), "fold in");
      float* foldT = nullptr;
      Chk(hipMalloc(&foldT, static_cast<std::size_t>(RANK) * tv.head_rows * 4), "fold tmp");
      hip::Gemm(st, tv.head_type, tv.head, tv.head_rows, E, dhT, E, foldT, tv.head_rows, RANK);
      fold.type = static_cast<int>(core::GgmlType::kBF16);
      fold.k = RANK;
      fold.n = tv.head_rows;
      fold.p = Alloc(static_cast<std::size_t>(tv.head_rows) * RANK * 2);
      const std::size_t tot = static_cast<std::size_t>(tv.head_rows) * RANK;
      k_fold_pack<<<static_cast<unsigned>((tot + 255) / 256), 256, 0, st>>>(foldT, static_cast<uint16_t*>(fold.p), tv.head_rows);
      float* hb = UpF32("dspark_hidden_out.bias");
      fbias = AllocF(tv.head_rows);
      hip::Gemm(st, tv.head_type, tv.head, tv.head_rows, E, hb, E, fbias, tv.head_rows, 1);
      Chk(hipStreamSynchronize(st), "fold");
      (void)hipFree(foldT);
    }
  }

  void EnsureInject(int n) {
    if (n <= inject_cap) return;
    inject_cap = std::max(n, 64);
    fused = AllocF(static_cast<std::size_t>(inject_cap) * E);
    kk = AllocF(static_cast<std::size_t>(inject_cap) * KVD);
    vv = AllocF(static_cast<std::size_t>(inject_cap) * KVD);
  }

  void Inject(long long pos0, int n) {
    if (n <= 0) return;
    if (!tv.features) throw std::runtime_error("inkling2 dspark: the target has no feature capture");
    EnsureInject(n);
    for (int l = 0; l < NL; ++l) {
      Layer& y = L[l];
      k_fuse<<<n, 256, 0, st>>>(tv.features, fuse[l][0], fuse[l][1], fuse[l][2], fuse[l][3], y.dflare, fused, eps);
      Gemm(y.kctx, fused, E, kk, KVD, n);
      Gemm(y.vctx, fused, E, vv, KVD, n);
      k_headnorm_rope<<<dim3(n, NKV), 64, 0, st>>>(kk, NKV, y.kctx_norm, static_cast<int>(pos0), theta, eps);
      k_store<<<n, 256, 0, st>>>(kk, vv, y.kc, y.vc, static_cast<int>(pos0), y.cap);
    }
    k_fuse<<<n, 256, 0, st>>>(tv.features, fuse[3][0], fuse[3][1], fuse[3][2], fuse[3][3], dspark_norm, fused, eps);
    for (int i = std::max(0, n - HN); i < n; ++i) {
      const long long p = pos0 + i;
      Chk(hipMemcpyAsync(hn + static_cast<std::size_t>(p % HN) * E, fused + static_cast<std::size_t>(i) * E, E * 4,
                         hipMemcpyDeviceToDevice, st),
          "hn");
      hn_pos[p % HN] = p;
    }
    Chk(hipStreamSynchronize(st), "inject");
  }

  std::vector<std::int32_t> Draft(int anchor, long long pos0, int k) {
    std::vector<std::int32_t> out;
    const auto t0 = std::chrono::steady_clock::now();
    if (k <= 0 || pos0 < 1 || ctx_from > pos0 - 1 || hn_pos[(pos0 - 1) % HN] != pos0 - 1) return out;
    k = std::min(k, BLOCK - 1);
    Chk(hipMemcpyAsync(d_anchor, &anchor, 4, hipMemcpyHostToDevice, st), "anchor");
    Chk(hipMemcpyAsync(d_prev, d_anchor, 4, hipMemcpyDeviceToDevice, st), "prev");
    hip::EmbedRows(st, tv.embd_type, tv.embd, E, d_anchor, 1, row, E);
    hip::RmsNorm(st, row, E, tv.embd_norm, row0, E, E, 1, tv.eps);
    k_fill_rows<<<BLOCK, 256, 0, st>>>(X, row0, mask);
    for (int l = 0; l < NL; ++l) {
      Layer& y = L[l];
      hip::RmsNorm(st, X, E, y.attn_norm, xn, E, E, BLOCK, eps);
      Gemm(y.q, xn, E, qb, QD, BLOCK);
      Gemm(y.k, xn, E, kb, KVD, BLOCK);
      Gemm(y.v, xn, E, vb, KVD, BLOCK);
      k_headnorm_rope<<<dim3(BLOCK, NH), 64, 0, st>>>(qb, NH, y.q_norm, static_cast<int>(pos0), theta, eps);
      k_headnorm_rope<<<dim3(BLOCK, NKV), 64, 0, st>>>(kb, NKV, y.k_norm, static_cast<int>(pos0), theta, eps);
      const long long win = swa[l] ? window : 8192;
      const long long lo = std::max<long long>(ctx_from, pos0 - win), hi = pos0;
      k_attn<<<dim3(BLOCK, NH), 128, 0, st>>>(qb, kb, vb, y.kc, y.vc, y.cap, static_cast<int>(lo),
                                              static_cast<int>(std::max(hi, lo)), BLOCK, ab);
      Gemm(y.o, ab, QD, ob, E, BLOCK);
      k_add<<<(BLOCK * E + 255) / 256, 256, 0, st>>>(X, ob, ob, BLOCK * E);
      hip::RmsNorm(st, ob, E, y.ffn_norm, xn, E, E, BLOCK, eps);
      Gemm(y.gate, xn, E, gb, FF, BLOCK);
      Gemm(y.up, xn, E, ub, FF, BLOCK);
      k_swiglu<<<(BLOCK * FF + 255) / 256, 256, 0, st>>>(gb, ub, BLOCK * FF);
      Gemm(y.down, gb, FF, db, E, BLOCK);
      k_add<<<(BLOCK * E + 255) / 256, 256, 0, st>>>(ob, db, X, BLOCK * E);
    }
    hip::RmsNorm(st, X, E, out_norm, hid, E, E, BLOCK, eps);
    // base logits of draft rows 1..k in one lm_head pass
    hip::Gemm(st, tv.head_type, tv.head, tv.head_rows, E, hid + E, E, base, tv.head_rows, k);
    Chk(hipMemsetAsync(state, 0, RANK * 4, st), "state");
    const float* ctx = hn + static_cast<std::size_t>((pos0 - 1) % HN) * E;
    for (int i = 1; i <= k; ++i) {
      k_joint<<<1, 256, 0, st>>>(state, static_cast<const uint16_t*>(markov.p), d_prev, hid + static_cast<std::size_t>(i) * E, ctx, jb);
      Gemm(joint, jb, 2 * RANK + 2 * E, gates, 3 * RANK, 1);
      k_gru<<<1, 256, 0, st>>>(gates, joint_b, state, latent);
      Gemm(fold, latent, RANK, corr, tv.head_rows, 1);
      k_argmax_part<<<256, 256, 0, st>>>(base + static_cast<std::size_t>(i - 1) * tv.head_rows, corr, fbias, tv.vocab, pv, pi);
      k_argmax_final<<<1, 256, 0, st>>>(pv, pi, 256, d_toks, i - 1, d_prev);
    }
    out.resize(k);
    Chk(hipMemcpyAsync(out.data(), d_toks, static_cast<std::size_t>(k) * 4, hipMemcpyDeviceToHost, st), "toks");
    Chk(hipStreamSynchronize(st), "draft");
    last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
  }

  ~Impl() {
    for (void* p : allocs) (void)hipFree(p);
    if (st) (void)hipStreamDestroy(st);
  }
};

std::unique_ptr<DSparkDrafter> DSparkDrafter::Create(Engine& target, const std::string& path, const Options& options,
                                                     std::string* error) {
  try {
    std::unique_ptr<DSparkDrafter> d(new DSparkDrafter());
    d->impl_ = std::make_unique<Impl>();
    d->impl_->target = &target;
    d->impl_->tv = target.draft_target();
    d->impl_->Load(path, options);
    return d;
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return nullptr;
  }
}
DSparkDrafter::~DSparkDrafter() = default;
void DSparkDrafter::Reset(std::size_t pos) {
  impl_->ctx_from = static_cast<long long>(pos);
  impl_->hn_pos.assign(HN, -1);
}
void DSparkDrafter::Inject(std::size_t pos0, std::size_t n) {
  impl_->tv.features = impl_->target->draft_target().features;
  impl_->Inject(static_cast<long long>(pos0), static_cast<int>(n));
}
std::vector<std::int32_t> DSparkDrafter::Draft(std::int32_t anchor, std::size_t pos0, int k) {
  return impl_->Draft(anchor, static_cast<long long>(pos0), k);
}
double DSparkDrafter::last_draft_ms() const { return impl_->last_ms; }

}  // namespace gufo::models::inkling2
