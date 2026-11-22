// gufo_windows addition (not part of headbouyJB/inkling2-gufo): DSpark acceptance measurement.
//
//   inkling2_dspark --model GGUF --drafter dspark-Inkling-2-Mini-Q8_0.gguf --prompt FILE [--gen 256] [--k 3,5,7]
//                   [--chat 1] [--ctx 8192] [--threads 16] [--tap-shift 0] [--rope neox|adjacent]
//
// 1 Target: greedy decode of --gen tokens (the k = 0 reference, single-token decode path) with feature capture:
//   per token the residual stream entering visits 5 / 23 / 41 and the final normalised state (target_layers
//   [5, 23, 41, 48] of the drafter). The prompt is also fed token by token so every position has features.
// 2 Drafter (CPU, fp32, this file; after sebt3's engine/dspark.hpp and ciru-ai's vLLM draft): per position the
//   fused features give the drafter's context keys / values (3 layers, RoPE NEOX base 1e6, window 1024 on the
//   local layers 0 and 2), and the Markov head's context vector. A draft = anchor embedding (target token_embd +
//   token_embd_norm) + 7 mask rows through the 3 layers (non-causal inside the block), then up to 7 sequential
//   steps of the recurrent Markov head on top of the target's lm_head.
// 3 Simulation per K: cycle at committed position c (S[c] = the newest token, features known up to c - 1): draft
//   K tokens, accept the longest prefix equal to the reference, the cycle yields a + 1 tokens, c += a + 1.
//   Because verification is lossless, the reference IS what a speculative run must output, so the accepted
//   lengths are exact; the projected tok/s uses the measured chain cost (verify T = K + 1) when K + 1 <= 4.
// Speed of the CPU drafter is irrelevant here; only the acceptance is measured.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>

#include "src/engine.hpp"
#include "text/tokenizer.hpp"

namespace {
using gufo::core::GgufReader;
using gufo::core::GgufTensorInfo;
using Clock = std::chrono::steady_clock;

constexpr int E = 2304, NL = 3, NH = 24, NKV = 8, HD = 128, FF = 9216, RANK = 256, BLOCK = 8;
constexpr int KVD = NKV * HD, QD = NH * HD;
int g_threads = 16;

template <class F>
void ParallelFor(int n, F&& f) {
  const int nt = std::max(1, std::min(g_threads, n));
  std::atomic<int> next{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < nt; ++t)
    ts.emplace_back([&] {
      for (;;) {
        const int i0 = next.fetch_add(64);
        if (i0 >= n) break;
        for (int i = i0; i < std::min(n, i0 + 64); ++i) f(i);
      }
    });
  for (auto& t : ts) t.join();
}

float F16(std::uint16_t h) {
  const std::uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
  std::uint32_t r;
  if (e == 0) {
    if (m == 0) r = s;
    else {  // subnormal
      float v = std::ldexp(static_cast<float>(m), -24);
      return (h & 0x8000u) ? -v : v;
    }
  } else if (e == 31) {
    r = s | 0x7F800000u | (m << 13);
  } else {
    r = s | ((e + 112) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &r, 4);
  return f;
}
float BF16(std::uint16_t b) {
  const std::uint32_t r = static_cast<std::uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &r, 4);
  return f;
}

// A matrix [rows][K] in the file's format (ggml: ne0 = K, ne1 = rows).
struct Mat {
  const std::uint8_t* p = nullptr;
  int type = 0;  // 0 F32, 8 Q8_0, 30 BF16
  int K = 0, rows = 0;
  [[nodiscard]] double Dot(int r, const float* x) const {
    double acc = 0;
    if (type == 8) {
      const std::uint8_t* row = p + static_cast<std::size_t>(r) * (K / 32) * 34;
      for (int b = 0; b < K / 32; ++b) {
        const std::uint8_t* blk = row + b * 34;
        std::uint16_t hs;
        std::memcpy(&hs, blk, 2);
        const auto* q = reinterpret_cast<const std::int8_t*>(blk + 2);
        float s = 0.f;
        for (int i = 0; i < 32; ++i) s += q[i] * x[b * 32 + i];
        acc += static_cast<double>(F16(hs)) * s;
      }
    } else if (type == 30) {
      const auto* row = reinterpret_cast<const std::uint16_t*>(p) + static_cast<std::size_t>(r) * K;
      float s = 0.f;
      for (int i = 0; i < K; ++i) s += BF16(row[i]) * x[i];
      acc = s;
    } else {
      const auto* row = reinterpret_cast<const float*>(p) + static_cast<std::size_t>(r) * K;
      float s = 0.f;
      for (int i = 0; i < K; ++i) s += row[i] * x[i];
      acc = s;
    }
    return acc;
  }
  void Row(int r, float* out) const {  // dequantised row
    if (type == 8) {
      const std::uint8_t* row = p + static_cast<std::size_t>(r) * (K / 32) * 34;
      for (int b = 0; b < K / 32; ++b) {
        std::uint16_t hs;
        std::memcpy(&hs, row + b * 34, 2);
        const float d = F16(hs);
        const auto* q = reinterpret_cast<const std::int8_t*>(row + b * 34 + 2);
        for (int i = 0; i < 32; ++i) out[b * 32 + i] = d * q[i];
      }
    } else if (type == 30) {
      const auto* row = reinterpret_cast<const std::uint16_t*>(p) + static_cast<std::size_t>(r) * K;
      for (int i = 0; i < K; ++i) out[i] = BF16(row[i]);
    } else {
      std::memcpy(out, reinterpret_cast<const float*>(p) + static_cast<std::size_t>(r) * K, K * 4);
    }
  }
  // y[n][rows] = W x[n][K]
  void Apply(const float* x, float* y, int n) const {
    ParallelFor(rows, [&](int r) {
      for (int t = 0; t < n; ++t) y[static_cast<std::size_t>(t) * rows + r] = static_cast<float>(Dot(r, x + static_cast<std::size_t>(t) * K));
    });
  }
};

Mat GetMat(const GgufReader& g, const std::string& name) {
  const GgufTensorInfo* t = g.FindTensor(name);
  if (!t) throw std::runtime_error("missing tensor " + name);
  Mat m;
  m.p = static_cast<const std::uint8_t*>(t->data);
  m.type = static_cast<int>(t->type);
  m.K = static_cast<int>(t->dimensions[0]);
  m.rows = t->dimensions.size() > 1 ? static_cast<int>(t->dimensions[1]) : 1;
  if (m.type != 0 && m.type != 8 && m.type != 30) throw std::runtime_error("unsupported type for " + name);
  return m;
}
std::vector<float> GetVec(const GgufReader& g, const std::string& name) {
  const Mat m = GetMat(g, name);
  std::vector<float> v(static_cast<std::size_t>(m.K) * m.rows);
  for (int r = 0; r < m.rows; ++r) m.Row(r, v.data() + static_cast<std::size_t>(r) * m.K);
  return v;
}

void RmsNorm(const float* x, const float* w, float* out, int n, float eps) {
  double ss = 0;
  for (int i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
  const float r = static_cast<float>(1.0 / std::sqrt(ss / n + eps));
  for (int i = 0; i < n; ++i) out[i] = x[i] * r * (w ? w[i] : 1.f);
}
// per-head rms norm (128) with weight, then RoPE (NEOX pairs (i, i+64) or adjacent (2i, 2i+1)) at `pos`
void HeadNormRope(float* v, const float* w, int pos, float theta, float eps, bool adjacent) {
  RmsNorm(v, w, v, HD, eps);
  for (int i = 0; i < HD / 2; ++i) {
    const int ia = adjacent ? 2 * i : i, ib = adjacent ? 2 * i + 1 : i + HD / 2;
    const double ang = pos * std::pow(static_cast<double>(theta), -2.0 * i / HD);
    const float c = static_cast<float>(std::cos(ang)), s = static_cast<float>(std::sin(ang));
    const float a = v[ia], b = v[ib];
    v[ia] = a * c - b * s;
    v[ib] = a * s + b * c;
  }
}

struct Drafter {
  float eps = 1e-6f, theta = 1e6f;
  int window = 1024;
  bool swa[NL] = {true, false, true};
  bool adjacent = false;
  bool markov_on = true;  // false: base logits lm_head(row) only (sanity check of the Markov head)
  float fuse[4][4];
  std::vector<float> out_norm, dspark_norm, mask, joint_b, hout_b;
  Mat markov, joint, hout;
  struct Layer {
    std::vector<float> attn_norm, q_norm, k_norm, kctx_norm, dflare, ffn_norm;
    Mat q, k, v, o, kctx, vctx, gate, up, down;
  } L[NL];
  // per position: context K / V of every layer, and the Markov context vector
  std::vector<std::vector<float>> ck[NL], cv[NL];
  std::vector<std::vector<float>> hn;

  explicit Drafter(const GgufReader& g) {
    if (auto v = g.GetMetadataUint32("dflash.block_count"); v && *v != NL) throw std::runtime_error("drafter: block_count");
    if (auto v = g.GetMetadataUint32("dflash.block_size"); v && *v != BLOCK) throw std::runtime_error("drafter: block_size");
    if (auto v = g.GetMetadataFloat32("dflash.attention.layer_norm_rms_epsilon")) eps = *v;
    if (auto v = g.GetMetadataFloat32("dflash.rope.freq_base")) theta = *v;
    if (auto v = g.GetMetadataUint32("dflash.attention.sliding_window")) window = static_cast<int>(*v);
    const auto fz = GetVec(g, "dspark_fuse.weight");  // [4][4], softmax over each row
    for (int r = 0; r < 4; ++r) {
      float mx = -1e30f, s = 0.f, w[4];
      for (int j = 0; j < 4; ++j) mx = std::max(mx, fz[r * 4 + j]);
      for (int j = 0; j < 4; ++j) { w[j] = std::exp(fz[r * 4 + j] - mx); s += w[j]; }
      for (int j = 0; j < 4; ++j) fuse[r][j] = w[j] / s;
    }
    out_norm = GetVec(g, "output_norm.weight");
    dspark_norm = GetVec(g, "dspark_norm.weight");
    mask = GetVec(g, "mask_embd.weight");
    joint_b = GetVec(g, "joint_proj.bias");
    hout_b = GetVec(g, "dspark_hidden_out.bias");
    markov = GetMat(g, "markov_w1.weight");
    joint = GetMat(g, "joint_proj.weight");
    hout = GetMat(g, "dspark_hidden_out.weight");
    for (int l = 0; l < NL; ++l) {
      auto n = [&](const char* s) { return "blk." + std::to_string(l) + "." + s; };
      Layer& y = L[l];
      y.attn_norm = GetVec(g, n("attn_norm.weight"));
      y.q_norm = GetVec(g, n("attn_q_norm.weight"));
      y.k_norm = GetVec(g, n("attn_k_norm.weight"));
      y.kctx_norm = GetVec(g, n("attn_k_ctx_norm.weight"));
      y.dflare = GetVec(g, n("dflare_norm.weight"));
      y.ffn_norm = GetVec(g, n("ffn_norm.weight"));
      y.q = GetMat(g, n("attn_q.weight"));
      y.k = GetMat(g, n("attn_k.weight"));
      y.v = GetMat(g, n("attn_v.weight"));
      y.o = GetMat(g, n("attn_output.weight"));
      y.kctx = GetMat(g, n("attn_k_ctx.weight"));
      y.vctx = GetMat(g, n("attn_v_ctx.weight"));
      y.gate = GetMat(g, n("ffn_gate.weight"));
      y.up = GetMat(g, n("ffn_up.weight"));
      y.down = GetMat(g, n("ffn_down.weight"));
    }
  }

  void Fuse(const float* feat /*[4][E]*/, int row, const float* norm, float* out) const {
    std::vector<float> v(E);
    for (int i = 0; i < E; ++i)
      v[i] = fuse[row][0] * feat[i] + fuse[row][1] * feat[E + i] + fuse[row][2] * feat[2 * E + i] + fuse[row][3] * feat[3 * E + i];
    RmsNorm(v.data(), norm, out, E, eps);
  }

  // context of positions [0, n): features [n][4][E]
  void Inject(const std::vector<float>& feat, int n) {
    for (int l = 0; l < NL; ++l) {
      ck[l].assign(n, {});
      cv[l].assign(n, {});
      std::vector<float> fused(static_cast<std::size_t>(n) * E), kk(static_cast<std::size_t>(n) * KVD), vv(static_cast<std::size_t>(n) * KVD);
      ParallelFor(n, [&](int t) { Fuse(feat.data() + static_cast<std::size_t>(t) * 4 * E, l, L[l].dflare.data(), fused.data() + static_cast<std::size_t>(t) * E); });
      L[l].kctx.Apply(fused.data(), kk.data(), n);
      L[l].vctx.Apply(fused.data(), vv.data(), n);
      for (int t = 0; t < n; ++t) {
        for (int h = 0; h < NKV; ++h) HeadNormRope(kk.data() + static_cast<std::size_t>(t) * KVD + h * HD, L[l].kctx_norm.data(), t, theta, eps, adjacent);
        ck[l][t].assign(kk.begin() + static_cast<std::ptrdiff_t>(t) * KVD, kk.begin() + static_cast<std::ptrdiff_t>(t + 1) * KVD);
        cv[l][t].assign(vv.begin() + static_cast<std::ptrdiff_t>(t) * KVD, vv.begin() + static_cast<std::ptrdiff_t>(t + 1) * KVD);
      }
    }
    hn.assign(n, std::vector<float>(E));
    ParallelFor(n, [&](int t) { Fuse(feat.data() + static_cast<std::size_t>(t) * 4 * E, 3, dspark_norm.data(), hn[t].data()); });
  }

  // up to k (<= 7) tokens after `anchor` at position pos0 (context known for positions < pos0)
  std::vector<int> Draft(const float* anchor_row, int anchor, int pos0, int k, const Mat& lm_head, int vocab, float softcap) const {
    std::vector<float> X(static_cast<std::size_t>(BLOCK) * E), xn(X.size()), qb(static_cast<std::size_t>(BLOCK) * QD),
        kb(static_cast<std::size_t>(BLOCK) * KVD), vb(kb.size()), ab(qb.size()), ob(X.size()), gb(static_cast<std::size_t>(BLOCK) * FF),
        ub(gb.size()), db(X.size());
    std::memcpy(X.data(), anchor_row, E * 4);
    for (int r = 1; r < BLOCK; ++r) std::memcpy(X.data() + static_cast<std::size_t>(r) * E, mask.data(), E * 4);
    for (int l = 0; l < NL; ++l) {
      const Layer& y = L[l];
      for (int r = 0; r < BLOCK; ++r) RmsNorm(X.data() + r * E, y.attn_norm.data(), xn.data() + r * E, E, eps);
      y.q.Apply(xn.data(), qb.data(), BLOCK);
      y.k.Apply(xn.data(), kb.data(), BLOCK);
      y.v.Apply(xn.data(), vb.data(), BLOCK);
      for (int r = 0; r < BLOCK; ++r) {
        for (int h = 0; h < NH; ++h) HeadNormRope(qb.data() + r * QD + h * HD, y.q_norm.data(), pos0 + r, theta, eps, adjacent);
        for (int h = 0; h < NKV; ++h) HeadNormRope(kb.data() + r * KVD + h * HD, y.k_norm.data(), pos0 + r, theta, eps, adjacent);
      }
      const int win = swa[l] ? window : 8192;
      const int lo = std::max(0, pos0 - win), hi = pos0;
      ParallelFor(BLOCK * NH, [&](int rh) {
        const int r = rh / NH, h = rh % NH, kv = h / (NH / NKV);
        const float* q = qb.data() + r * QD + h * HD;
        const int total = (hi - lo) + BLOCK;
        std::vector<float> sc(total);
        float mx = -1e30f;
        for (int j = 0; j < total; ++j) {
          const float* key = j < hi - lo ? ck[l][lo + j].data() + kv * HD : kb.data() + (j - (hi - lo)) * KVD + kv * HD;
          float d = 0.f;
          for (int i = 0; i < HD; ++i) d += q[i] * key[i];
          sc[j] = d / std::sqrt(static_cast<float>(HD));
          mx = std::max(mx, sc[j]);
        }
        double sum = 0;
        for (int j = 0; j < total; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
        float* out = ab.data() + r * QD + h * HD;
        for (int i = 0; i < HD; ++i) out[i] = 0.f;
        for (int j = 0; j < total; ++j) {
          const float* val = j < hi - lo ? cv[l][lo + j].data() + kv * HD : vb.data() + (j - (hi - lo)) * KVD + kv * HD;
          const float w = static_cast<float>(sc[j] / sum);
          for (int i = 0; i < HD; ++i) out[i] += w * val[i];
        }
      });
      y.o.Apply(ab.data(), ob.data(), BLOCK);
      for (std::size_t i = 0; i < ob.size(); ++i) ob[i] += X[i];
      for (int r = 0; r < BLOCK; ++r) RmsNorm(ob.data() + r * E, y.ffn_norm.data(), xn.data() + r * E, E, eps);
      y.gate.Apply(xn.data(), gb.data(), BLOCK);
      y.up.Apply(xn.data(), ub.data(), BLOCK);
      for (std::size_t i = 0; i < gb.size(); ++i) gb[i] = gb[i] / (1.f + std::exp(-gb[i])) * ub[i];
      y.down.Apply(gb.data(), db.data(), BLOCK);
      for (std::size_t i = 0; i < X.size(); ++i) X[i] = ob[i] + db[i];
    }
    std::vector<float> hid(X.size());
    for (int r = 0; r < BLOCK; ++r) RmsNorm(X.data() + r * E, out_norm.data(), hid.data() + r * E, E, eps);
    // Markov head: joint = [state | markov_w1[prev] | hidden row | context] -> gates -> latent -> correction
    std::vector<float> state(RANK, 0.f), jb(2 * RANK + 2 * E), gates(3 * RANK), latent(RANK), corr(E), h2(E), mrow(RANK);
    const std::vector<float>& ctx = hn[pos0 - 1];
    std::vector<float> logits(lm_head.rows);
    int prev = anchor;
    std::vector<int> out;
    for (int i = 1; i <= std::min(k, BLOCK - 1); ++i) {
      markov.Row(prev, mrow.data());
      std::copy(state.begin(), state.end(), jb.begin());
      std::copy(mrow.begin(), mrow.end(), jb.begin() + RANK);
      std::copy(hid.begin() + i * E, hid.begin() + (i + 1) * E, jb.begin() + 2 * RANK);
      std::copy(ctx.begin(), ctx.end(), jb.begin() + 2 * RANK + E);
      joint.Apply(jb.data(), gates.data(), 1);
      for (int c = 0; c < 3 * RANK; ++c) gates[c] += joint_b[c];
      for (int c = 0; c < RANK; ++c) {
        const float gate = 1.f / (1.f + std::exp(-gates[c])), cand = std::tanh(gates[RANK + c]);
        state[c] = cand + gate * (state[c] - cand);
        latent[c] = std::tanh(gates[2 * RANK + c]);
      }
      hout.Apply(latent.data(), corr.data(), 1);
      for (int c = 0; c < E; ++c) h2[c] = hid[i * E + c] + (markov_on ? corr[c] + hout_b[c] : 0.f);
      lm_head.Apply(h2.data(), logits.data(), 1);
      int best = 0;
      float bv = -1e30f;
      for (int v = 0; v < vocab; ++v) {
        const float lv = softcap > 0 ? softcap * std::tanh(logits[v] / softcap) : logits[v];
        if (lv > bv) { bv = lv; best = v; }
      }
      out.push_back(best);
      prev = best;
    }
    return out;
  }
};

std::vector<int> ParseList(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) if (!t.empty()) out.push_back(std::atoi(t.c_str()));
  return out;
}
}  // namespace

int main(int argc, char** argv) {
  std::string model, drafter_path, prompt_path, klist = "3,5,7", rope = "neox";
  int gen = 256, chat = 1, tap_shift = 0, window_extra = 0, markov = 1;
  unsigned ctx = 8192;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (k == "--model") model = next();
    else if (k == "--drafter") drafter_path = next();
    else if (k == "--prompt") prompt_path = next();
    else if (k == "--gen") gen = std::atoi(next().c_str());
    else if (k == "--k") klist = next();
    else if (k == "--chat") chat = std::atoi(next().c_str());
    else if (k == "--ctx") ctx = static_cast<unsigned>(std::strtoul(next().c_str(), nullptr, 10));
    else if (k == "--threads") g_threads = std::atoi(next().c_str());
    else if (k == "--tap-shift") tap_shift = std::atoi(next().c_str());
    else if (k == "--rope") rope = next();
    else if (k == "--window-extra") window_extra = std::atoi(next().c_str());
    else if (k == "--markov") markov = std::atoi(next().c_str());
    else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
  }
  (void)tap_shift;  // reserved: the taps sit at the visit inputs 5/23/41 (sebt3, ciru-ai)
  std::string err;
  std::string text;
  {
    std::ifstream f(prompt_path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    text = ss.str();
  }
  auto tok = inkling2::text::Tokenizer::CreateFromGguf(model, &err);
  if (!tok) { std::fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1; }
  std::vector<std::int32_t> ids;
  {
    const std::string full = chat ? "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>" +
                                        text + "<|end_message|>"
                                  : text;
    const auto e = tok->Encode(full, chat != 0);
    ids.assign(e.begin(), e.end());
  }
  gufo::core::GgufOpenOptions open;
  open.allow_unsized_tensors = true;
  std::shared_ptr<const GgufReader> reader = GgufReader::OpenFile(model, &err, open);
  if (!reader) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
  std::shared_ptr<const GgufReader> dreader = GgufReader::OpenFile(drafter_path, &err, open);
  if (!dreader) { std::fprintf(stderr, "drafter: %s\n", err.c_str()); return 1; }
  auto engine = gufo::models::inkling2::Engine::Create(reader, ctx, &err);
  if (!engine) { std::fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
  const int V = static_cast<int>(engine->config().n_vocab_text);
  const float softcap = engine->config().logit_softcap;
  const float teps = engine->config().rms_eps;
  std::printf("prompt %zu tokens, gen %d, chat %d, rope %s\n", ids.size(), gen, chat, rope.c_str());

  // ---- 1 target: token by token with features (prompt and greedy generation)
  auto s = engine->NewSession();
  engine->SetFeatureCapture(true);
  std::vector<std::int32_t> seq = ids;
  std::vector<float> feat;  // [pos][4][E]
  const auto t0 = Clock::now();
  for (std::size_t i = 0; i < ids.size(); ++i) {
    engine->Forward(*s, std::span<const std::int32_t>(&ids[i], 1));
    feat.insert(feat.end(), engine->features().begin(), engine->features().end());
  }
  for (int g = 0; g < gen; ++g) {
    const std::int32_t nx = static_cast<std::int32_t>(std::max_element(s->logits().begin(), s->logits().begin() + V) - s->logits().begin());
    seq.push_back(nx);
    if (g + 1 == gen) break;
    engine->Forward(*s, std::span<const std::int32_t>(&nx, 1));
    feat.insert(feat.end(), engine->features().begin(), engine->features().end());
  }
  engine->SetFeatureCapture(false);
  std::printf("target: %zu positions with features in %.1f s\n", feat.size() / (4 * E),
              std::chrono::duration<double>(Clock::now() - t0).count());
  {
    std::vector<std::int32_t> outp(seq.begin() + static_cast<std::ptrdiff_t>(ids.size()), seq.end());
    std::string txt = tok->Decode(std::span<const inkling2::text::TokenId>(reinterpret_cast<const inkling2::text::TokenId*>(outp.data()), outp.size()));
    for (char& c : txt) if (c == '\n') c = ' ';
    std::printf("reference text: %.300s\n", txt.c_str());
  }

  // ---- 2 drafter context for every position with features
  Drafter d(*dreader);
  d.adjacent = rope == "adjacent";
  d.window += window_extra;
  d.markov_on = markov != 0;
  std::printf("drafter variant: window %d, markov %d\n", d.window, markov);
  const int npos = static_cast<int>(feat.size() / (4 * E));
  const auto t1 = Clock::now();
  d.Inject(feat, npos);
  std::printf("drafter: context of %d positions in %.1f s (fuse rows: %.3f %.3f %.3f %.3f)\n", npos,
              std::chrono::duration<double>(Clock::now() - t1).count(), d.fuse[0][0], d.fuse[0][1], d.fuse[0][2], d.fuse[0][3]);
  const Mat lm_head = GetMat(*reader, "output.weight");
  const Mat embd = GetMat(*reader, "token_embd.weight");
  const auto embd_norm = GetVec(*reader, "token_embd_norm.weight");

  // ---- 3 simulation per K
  const double t_dec = 22.4, t_chain[5] = {0, 22.4, 30.2, 34.8, 39.5};  // ms, measured (round 3/4)
  for (int K : ParseList(klist)) {
    K = std::min(K, BLOCK - 1);
    const int start = static_cast<int>(ids.size()), end = static_cast<int>(seq.size());
    int c = start, cycles = 0, drafted = 0, accepted = 0;
    std::vector<int> hist(K + 1, 0);
    double pos_acc[8] = {}, pos_n[8] = {};
    const auto t2 = Clock::now();
    while (c < end - 1) {
      const int anchor = seq[c];
      std::vector<float> row(E), arow(E);
      embd.Row(anchor, row.data());
      RmsNorm(row.data(), embd_norm.data(), arow.data(), E, teps);
      const int kk = std::min(K, end - 1 - c);
      const auto dr = d.Draft(arow.data(), anchor, c, kk, lm_head, V, softcap);
      int a = 0;
      while (a < static_cast<int>(dr.size()) && dr[a] == seq[c + 1 + a]) ++a;
      for (int i = 0; i < static_cast<int>(dr.size()); ++i) {
        pos_n[i] += (i <= a) ? 1 : 0;  // position i was reached (all before it accepted)
        pos_acc[i] += (i < a) ? 1 : 0;
      }
      ++cycles;
      drafted += static_cast<int>(dr.size());
      accepted += a;
      ++hist[a];
      c += a + 1;
    }
    const int toks = end - 1 - start;
    const double per_cycle = static_cast<double>(toks) / cycles;
    std::printf("K=%d: %d cycles, %d tokens, %.2f tokens/cycle, accepted %d of %d drafted (%.0f%%), draft time %.1f s\n", K, cycles,
                toks, per_cycle, accepted, drafted, drafted ? 100.0 * accepted / drafted : 0.0,
                std::chrono::duration<double>(Clock::now() - t2).count());
    std::printf("  per-position acceptance (given all earlier accepted):");
    for (int i = 0; i < K; ++i) std::printf(" %.2f", pos_n[i] > 0 ? pos_acc[i] / pos_n[i] : 0.0);
    std::printf("\n  accepted-length histogram:");
    for (int i = 0; i <= K; ++i) std::printf(" %d:%d", i, hist[i]);
    std::printf("\n");
    for (double draft_ms : {5.0, 10.0}) {
      const double verify = K + 1 <= 4 ? t_chain[K + 1] : 39.5 + (K + 1 - 4) * 4.7;  // T > 4: extrapolated from T = 2..4
      const double cyc_ms = draft_ms + verify;
      std::printf("  projected (draft %.0f ms, verify T=%d %.1f ms%s): %.1f tok/s vs %.1f plain\n", draft_ms, K + 1, verify,
                  K + 1 <= 4 ? " measured" : " extrapolated", per_cycle * 1000.0 / cyc_ms, 1000.0 / t_dec);
    }
  }
  return 0;
}
