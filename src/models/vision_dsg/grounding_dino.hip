// Grounding DINO on the GPU (see grounding_dino.hpp).
// Written with AI assistance (Claude Code).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>

#include "src/core/json.hpp"
#include "src/models/vision_dsg/grounding_dino.hpp"
#include "src/models/vision_dsg/hip_ops.hpp"

namespace gufo::models::vision_dsg {
namespace {

constexpr int kThreads = 256;
constexpr int kD = 256;          // d_model
constexpr int kLevels = 4;
constexpr int kPoints = 4;
constexpr int kQueries = 900;
constexpr int kMaxText = 256;
constexpr float kEps = 1e-5F;    // layer_norm_eps
constexpr float kBertEps = 1e-12F;
constexpr float kMinFloat = -3.4028234663852886e38F;
constexpr int kMaxWindow = 12;
const char* const kSwin = "model.backbone.conv_encoder.model.";

unsigned Blocks(std::size_t n) {
  return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

// ---- Swin kernels -----------------------------------------------------------

__device__ int ShiftRegion(int coord, int size, int ws, int shift) {
  return (coord >= size - ws ? 1 : 0) + (coord >= size - shift ? 1 : 0);
}

// One thread per (window, head, query): softmax(q k^T * scale + relative
// bias + shift mask) v. qkv rows [q | k | v] of 3C.
__global__ void WindowAttentionKernel(const float* qkv, int c, int heads, int ws,
                                      int shift, int hp, int wp, int windows,
                                      const float* table, float scale, float* out) {
  const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int n = ws * ws;
  if (t >= static_cast<std::size_t>(windows) * heads * n)
    return;
  const int i = static_cast<int>(t % n);
  const int head = static_cast<int>((t / n) % heads);
  const int win = static_cast<int>(t / (static_cast<std::size_t>(n) * heads));
  const int hd = c / heads;
  const int nww = wp / ws;
  const int wy = (win / nww) * ws, wx = (win % nww) * ws;
  const float* base = qkv + static_cast<std::size_t>(win) * n * 3 * c;
  const float* q = base + static_cast<std::size_t>(i) * 3 * c + head * hd;
  const int yi = i / ws, xi = i % ws;
  int region_i = 0;
  if (shift > 0)
    region_i = ShiftRegion(wy + yi, hp, ws, shift) * 3 + ShiftRegion(wx + xi, wp, ws, shift);
  float scores[kMaxWindow * kMaxWindow];
  float maximum = -INFINITY;
  for (int j = 0; j < n; ++j) {
    const float* k = base + static_cast<std::size_t>(j) * 3 * c + c + head * hd;
    float s = 0;
    for (int d = 0; d < hd; ++d)
      s += q[d] * k[d];
    const int yj = j / ws, xj = j % ws;
    const int index = (yi - yj + ws - 1) * (2 * ws - 1) + (xi - xj + ws - 1);
    s = s * scale + table[index * heads + head];
    if (shift > 0) {
      const int region_j =
          ShiftRegion(wy + yj, hp, ws, shift) * 3 + ShiftRegion(wx + xj, wp, ws, shift);
      if (region_j != region_i)
        s += -100.0F;
    }
    scores[j] = s;
    maximum = fmaxf(maximum, s);
  }
  float sum = 0;
  for (int j = 0; j < n; ++j) {
    scores[j] = expf(scores[j] - maximum);
    sum += scores[j];
  }
  const float inv = 1.0F / sum;
  float* o = out + (static_cast<std::size_t>(win) * n + i) * c + head * hd;
  for (int d = 0; d < hd; ++d) {
    float acc = 0;
    for (int j = 0; j < n; ++j)
      acc += scores[j] * base[static_cast<std::size_t>(j) * 3 * c + 2 * c + head * hd + d];
    o[d] = acc * inv;
  }
}

// Swin patch merging input: [ceil(H/2) * ceil(W/2), 4C] of x0 (2i, 2j),
// x1 (2i+1, 2j), x2 (2i, 2j+1), x3 (2i+1, 2j+1), zero padded.
__global__ void PatchMergeKernel(const float* x, int h, int w, int c, float* out) {
  const int h2 = (h + 1) / 2, w2 = (w + 1) / 2;
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(h2) * w2 * 4 * c;
  if (i >= total)
    return;
  const int ch = static_cast<int>(i % (4 * c));
  const std::size_t p = i / (4 * c);
  const int part = ch / c, cc = ch % c;
  const int y = static_cast<int>(p / w2) * 2 + (part == 1 || part == 3 ? 1 : 0);
  const int xx = static_cast<int>(p % w2) * 2 + (part >= 2 ? 1 : 0);
  out[i] = (y < h && xx < w) ? x[(static_cast<std::size_t>(y) * w + xx) * c + cc] : 0.0F;
}

// ---- deformable attention ---------------------------------------------------

struct LevelShapes {
  int h[kLevels];
  int w[kLevels];
  int start[kLevels];
};

// One thread per (query, head, d): bilinear samples (grid_sample,
// align_corners false, zero padding) of the value heads, weighted by the
// softmax of the attention logits. ref: [nq, 2] (encoder) or [nq, 4].
__global__ void DeformableKernel(const float* value, LevelShapes shapes,
                                 const float* offsets, const float* logits,
                                 const float* ref, int ref_dims, int nq,
                                 int heads, int hd, float* out) {
  const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= static_cast<std::size_t>(nq) * heads * hd)
    return;
  const int d = static_cast<int>(t % hd);
  const int head = static_cast<int>((t / hd) % heads);
  const int q = static_cast<int>(t / (static_cast<std::size_t>(hd) * heads));
  constexpr int kLp = kLevels * kPoints;
  const float* lg = logits + (static_cast<std::size_t>(q) * heads + head) * kLp;
  float maximum = -INFINITY;
  for (int k = 0; k < kLp; ++k)
    maximum = fmaxf(maximum, lg[k]);
  float weights[kLp];
  float sum = 0;
  for (int k = 0; k < kLp; ++k) {
    weights[k] = expf(lg[k] - maximum);
    sum += weights[k];
  }
  const float* off = offsets + (static_cast<std::size_t>(q) * heads + head) * kLp * 2;
  const float* r = ref + static_cast<std::size_t>(q) * ref_dims;
  const int dim = heads * hd;
  float acc = 0;
  for (int l = 0; l < kLevels; ++l) {
    const int H = shapes.h[l], W = shapes.w[l];
    const float* v = value + static_cast<std::size_t>(shapes.start[l]) * dim + head * hd + d;
    for (int p = 0; p < kPoints; ++p) {
      const int k = l * kPoints + p;
      float lx, ly;
      if (ref_dims == 2) {
        lx = r[0] + off[2 * k] / static_cast<float>(W);
        ly = r[1] + off[2 * k + 1] / static_cast<float>(H);
      } else {
        lx = r[0] + off[2 * k] / kPoints * r[2] * 0.5F;
        ly = r[1] + off[2 * k + 1] / kPoints * r[3] * 0.5F;
      }
      // grid = 2 * loc - 1; unnormalised ((g + 1) * size - 1) / 2.
      const float gx = 2.0F * lx - 1.0F, gy = 2.0F * ly - 1.0F;
      const float x = ((gx + 1.0F) * W - 1.0F) / 2.0F;
      const float y = ((gy + 1.0F) * H - 1.0F) / 2.0F;
      const float x0f = floorf(x), y0f = floorf(y);
      const int x0 = static_cast<int>(x0f), y0 = static_cast<int>(y0f);
      const float fx = x - x0f, fy = y - y0f;
      float sample = 0;
      const auto at = [&](int yy, int xx) {
        return (yy >= 0 && yy < H && xx >= 0 && xx < W)
                   ? v[(static_cast<std::size_t>(yy) * W + xx) * dim]
                   : 0.0F;
      };
      sample = at(y0, x0) * (1 - fx) * (1 - fy) + at(y0, x0 + 1) * fx * (1 - fy) +
               at(y0 + 1, x0) * (1 - fx) * fy + at(y0 + 1, x0 + 1) * fx * fy;
      acc += sample * weights[k];
    }
  }
  out[static_cast<std::size_t>(q) * dim + head * hd + d] = acc / sum;
}

// rows[r, :] *= keep[r].
__global__ void RowKeepKernel(float* x, const float* keep, int columns, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] *= keep[i / columns];
}

// encode_sinusoidal_position_embedding of one coordinate into `feats`
// values: x * 2 pi / T^(2 floor(c / 2) / feats), sin on even, cos on odd.
void Sine(float coord, int feats, float temperature, float* out) {
  const float scale = 6.283185307179586F;
  for (int c = 0; c < feats; ++c) {
    const float dim_t = std::pow(temperature, 2.0F * static_cast<float>(c / 2) /
                                                  static_cast<float>(feats));
    const float v = coord * scale / dim_t;
    out[c] = (c % 2 == 0) ? std::sin(v) : std::cos(v);
  }
}

float Logit(float x) {
  constexpr float eps = 1e-5F;
  x = std::clamp(x, eps, 1.0F - eps);
  return std::log(x / (1.0F - x));
}

float Sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }

// Oracle debugging: GUFO_VISION_DUMP=<dir> writes intermediate tensors.
void Dump(const char* name, const float* device, std::size_t count) {
  const char* dir = std::getenv("GUFO_VISION_DUMP");
  if (dir == nullptr || device == nullptr)
    return;
  std::vector<float> host(count);
  hipDeviceSynchronize();
  hipMemcpy(host.data(), device, count * 4, hipMemcpyDeviceToHost);
  std::ofstream out(std::string(dir) + "/" + name + ".f32", std::ios::binary);
  out.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(count * 4));
}

}  // namespace

struct GroundingDino::Impl {
  Gpu gpu;
  WeightStore w;
  BertTokenizer tokenizer;
  std::vector<int> depths, swin_heads;
  int embed{128}, window{12};
  int bert_layers{12}, bert_dim{768}, bert_heads{12};
  int enc_layers{6}, dec_layers{6}, ffn{2048}, enc_heads{8}, dec_heads{8};
  std::vector<float> level_embed;
  // MM-Grounding-DINO / LLMDet: contrastive logits / sqrt(256) + a learned
  // bias, and one box head and class bias per decoder layer.
  bool mm{false};
  float enc_bias{0}, final_bias{0}, logit_scale{1};
  std::string BoxHead(int layer) const {
    if (!mm)
      return "model.decoder.bbox_embed.0";
    return layer == 0 ? "bbox_embed.0" : "model.decoder.bbox_embed." + std::to_string(layer);
  }

  const float* W(const std::string& name) const { return w.Get(name); }
  const float* S(const std::string& name) const { return w.Get(kSwin + name); }
  bool Lin(const float* in, int rows_in, const std::string& name, int out_dim,
           int in_dim, float* out, Act act, std::string* error) {
    return gpu.Linear(in, rows_in, W(name + ".weight"),
                      w.Has(name + ".bias") ? W(name + ".bias") : nullptr, out_dim,
                      in_dim, out, act, error);
  }
  bool Swin(const std::vector<float>& pixels, int ph, int pw,
            std::vector<DevBuf>* maps, std::vector<int>* hs, std::vector<int>* ws,
            std::vector<int>* cs, std::string* error);
  bool Bert(const std::vector<int>& ids, const std::vector<int>& positions,
            const DevBuf& mask, DevBuf* text, std::string* error);
  bool Mha(const float* q_in, const float* k_in, const float* v_in, int tq, int tk,
           const std::string& p, int heads, const float* bias, float* out,
           std::string* error);
  bool Deformable(const float* hidden, const float* pos, int nq, const float* memory,
                  int nv, const LevelShapes& shapes, const DevBuf& ref, int ref_dims,
                  const std::string& p, int heads, float* out, std::string* error);
  bool Mlp(const float* in, int rows, const std::string& p, int layers, int in_dim,
           int out_dim, float* out, std::string* error);
  DevBuf t1, t2, t3, t4, t5, t6;
};

GroundingDino::GroundingDino(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GroundingDino::~GroundingDino() = default;

std::size_t GroundingDino::weight_bytes() const { return impl_->w.bytes(); }

std::unique_ptr<GroundingDino> GroundingDino::Load(const std::filesystem::path& dir,
                                                   std::string* error) {
  std::ifstream file(dir / "config.json", std::ios::binary);
  if (!file) {
    SetError(error, "Grounding DINO: no config.json in " + dir.string());
    return nullptr;
  }
  std::stringstream text;
  text << file.rdbuf();
  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  try {
    const json::Value config = json::parse(text.str());
    const std::string type = config.member_str("model_type");
    m.mm = type == "mm-grounding-dino";
    if (type != "grounding-dino" && !m.mm) {
      SetError(error, dir.string() + " is not a Grounding DINO checkpoint");
      return nullptr;
    }
    const json::Value* b = config.find("backbone_config");
    if (b == nullptr || b->member_str("model_type") != "swin") {
      SetError(error, "Grounding DINO: only the Swin backbone is supported");
      return nullptr;
    }
    for (const auto& v : b->find("depths")->items())
      m.depths.push_back(static_cast<int>(v.as_double()));
    for (const auto& v : b->find("num_heads")->items())
      m.swin_heads.push_back(static_cast<int>(v.as_double()));
    m.embed = static_cast<int>(b->member_double("embed_dim", 128));
    m.window = static_cast<int>(b->member_double("window_size", 12));
    if (static_cast<int>(config.member_double("d_model", 256)) != kD ||
        static_cast<int>(config.member_double("num_feature_levels", 4)) != kLevels ||
        static_cast<int>(config.member_double("encoder_n_points", 4)) != kPoints ||
        static_cast<int>(config.member_double("decoder_n_points", 4)) != kPoints ||
        static_cast<int>(config.member_double("num_queries", 900)) != kQueries ||
        !config.find("two_stage")->as_bool() || m.depths.size() != 4 ||
        m.window > kMaxWindow) {
      SetError(error, "Grounding DINO: unsupported configuration");
      return nullptr;
    }
    m.enc_layers = static_cast<int>(config.member_double("encoder_layers", 6));
    m.dec_layers = static_cast<int>(config.member_double("decoder_layers", 6));
    m.ffn = static_cast<int>(config.member_double("encoder_ffn_dim", 2048));
    m.enc_heads = static_cast<int>(config.member_double("encoder_attention_heads", 8));
    m.dec_heads = static_cast<int>(config.member_double("decoder_attention_heads", 8));
  } catch (const std::exception& e) {
    SetError(error, "Grounding DINO config.json: " + std::string(e.what()));
    return nullptr;
  }
  if (!m.tokenizer.Load(dir / "vocab.txt", error) || !m.gpu.Init(error) ||
      !m.w.Load(dir / "model.safetensors", error))
    return nullptr;
  m.level_embed = m.w.Host("model.level_embed");
  if (m.mm) {
    const auto enc = m.w.Host("model.encoder_output_class_embed.bias");
    const std::string last = m.dec_layers == 1
                                 ? std::string("class_embed.0.bias")
                                 : "model.decoder.class_embed." + std::to_string(m.dec_layers - 1) + ".bias";
    const auto fin = m.w.Host(last);
    if (enc.size() != 1 || fin.size() != 1) {
      SetError(error, "MM-Grounding-DINO: class biases missing");
      return nullptr;
    }
    m.enc_bias = enc[0];
    m.final_bias = fin[0];
    m.logit_scale = 1.0F / std::sqrt(static_cast<float>(kD));
  }
  if (m.level_embed.size() != static_cast<std::size_t>(kLevels) * kD ||
      m.W(m.BoxHead(m.dec_layers - 1) + ".layers.2.weight") == nullptr ||
      m.W("model.text_projection.weight") == nullptr) {
    SetError(error, "Grounding DINO: weights do not match the configuration");
    return nullptr;
  }
  return std::unique_ptr<GroundingDino>(new GroundingDino(std::move(impl)));
}

bool GroundingDino::Impl::Mlp(const float* in, int rows, const std::string& p, int layers,
                              int in_dim, int out_dim, float* out, std::string* error) {
  const float* x = in;
  DevBuf* bufs[2] = {&t5, &t6};
  int dim = in_dim;
  for (int l = 0; l < layers; ++l) {
    const bool last = l + 1 == layers;
    const int od = last ? out_dim : kD;
    float* target = last ? out : nullptr;
    if (!last) {
      if (!bufs[l % 2]->Reserve(static_cast<std::size_t>(rows) * od, error))
        return false;
      target = bufs[l % 2]->f();
    }
    if (!Lin(x, rows, p + ".layers." + std::to_string(l), od, dim, target,
             last ? Act::kNone : Act::kRelu, error))
      return false;
    x = target;
    dim = od;
  }
  return true;
}

// GroundingDinoMultiheadAttention: q from q_in, k from k_in, v from v_in.
bool GroundingDino::Impl::Mha(const float* q_in, const float* k_in, const float* v_in,
                              int tq, int tk, const std::string& p, int heads,
                              const float* bias, float* out, std::string* error) {
  if (!t1.Reserve(static_cast<std::size_t>(tq) * kD, error) ||
      !t2.Reserve(static_cast<std::size_t>(tk) * kD, error) ||
      !t3.Reserve(static_cast<std::size_t>(tk) * kD, error) ||
      !t4.Reserve(static_cast<std::size_t>(tq) * kD, error) ||
      !Lin(q_in, tq, p + ".query", kD, kD, t1.f(), Act::kNone, error) ||
      !Lin(k_in, tk, p + ".key", kD, kD, t2.f(), Act::kNone, error) ||
      !Lin(v_in, tk, p + ".value", kD, kD, t3.f(), Act::kNone, error))
    return false;
  const int hd = kD / heads;
  return gpu.Attention(t1.f(), kD, t2.f(), kD, t3.f(), kD, tq, tk, heads, hd,
                       1.0F / std::sqrt(static_cast<float>(hd)), t4.f(), kD, error,
                       nullptr, bias) &&
         Lin(t4.f(), tq, p + ".out_proj", kD, kD, out, Act::kNone, error);
}

bool GroundingDino::Impl::Deformable(const float* hidden, const float* pos, int nq,
                                     const float* memory, int nv,
                                     const LevelShapes& shapes, const DevBuf& ref,
                                     int ref_dims, const std::string& p, int heads,
                                     float* out, std::string* error) {
  const std::size_t qd = static_cast<std::size_t>(nq) * kD;
  DevBuf query, value, offsets, logits, sampled;
  if (!query.Reserve(qd, error) || !value.Reserve(static_cast<std::size_t>(nv) * kD, error) ||
      !offsets.Reserve(static_cast<std::size_t>(nq) * heads * kLevels * kPoints * 2, error) ||
      !logits.Reserve(static_cast<std::size_t>(nq) * heads * kLevels * kPoints, error) ||
      !sampled.Reserve(qd, error))
    return false;
  AddTo(query.f(), hidden, pos, qd);
  const int lp = heads * kLevels * kPoints;
  if (!Lin(memory, nv, p + ".value_proj", kD, kD, value.f(), Act::kNone, error) ||
      !Lin(query.f(), nq, p + ".sampling_offsets", lp * 2, kD, offsets.f(), Act::kNone, error) ||
      !Lin(query.f(), nq, p + ".attention_weights", lp, kD, logits.f(), Act::kNone, error))
    return false;
  const std::size_t n = static_cast<std::size_t>(nq) * kD;
  DeformableKernel<<<Blocks(n), kThreads>>>(value.f(), shapes, offsets.f(), logits.f(),
                                            ref.f(), ref_dims, nq, heads, kD / heads,
                                            sampled.f());
  return Lin(sampled.f(), nq, p + ".output_proj", kD, kD, out, Act::kNone, error);
}

bool GroundingDino::Impl::Swin(const std::vector<float>& pixels, int ph, int pw,
                               std::vector<DevBuf>* maps, std::vector<int>* hs,
                               std::vector<int>* wss, std::vector<int>* cs,
                               std::string* error) {
  // Patch embedding: zero pad to multiples of 4, 4x4 / 4 convolution, norm.
  const int hp4 = (ph + 3) / 4 * 4, wp4 = (pw + 3) / 4 * 4;
  std::vector<float> padded(static_cast<std::size_t>(3) * hp4 * wp4, 0.0F);
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < ph; ++y)
      std::copy_n(pixels.data() + (static_cast<std::size_t>(c) * ph + y) * pw, pw,
                  padded.data() + (static_cast<std::size_t>(c) * hp4 + y) * wp4);
  int h = hp4 / 4, w = wp4 / 4, c = embed;
  DevBuf image, map, hidden, normed, win, qkv, att, proj, mid;
  if (!image.Upload(padded.data(), padded.size(), error) ||
      !map.Reserve(static_cast<std::size_t>(c) * h * w, error) ||
      !gpu.Conv2d(image.f(), 3, hp4, wp4, S("embeddings.patch_embeddings.projection.weight"),
                  S("embeddings.patch_embeddings.projection.bias"), c, 4, 4, 0, map.f(),
                  nullptr, nullptr, error) ||
      !hidden.Reserve(static_cast<std::size_t>(c) * h * w, error))
    return false;
  MapToTokens(map.f(), h * w, c, normed.Reserve(static_cast<std::size_t>(c) * h * w, error)
                                     ? normed.f()
                                     : nullptr);
  LayerNorm(normed.f(), S("embeddings.norm.weight"), S("embeddings.norm.bias"), hidden.f(),
            h * w, c, kEps);
  for (int stage = 0; stage < 4; ++stage) {
    const int heads = swin_heads[stage];
    for (int block = 0; block < depths[stage]; ++block) {
      const std::string p = "encoder.layers." + std::to_string(stage) + ".blocks." +
                            std::to_string(block) + ".";
      int ws = window;
      int shift = block % 2 == 1 ? window / 2 : 0;
      if (std::min(h, w) <= window) {
        SetError(error, "Grounding DINO: image too small for the Swin windows");
        return false;
      }
      const int hp = (h + ws - 1) / ws * ws, wp = (w + ws - 1) / ws * ws;
      const std::size_t tokens = static_cast<std::size_t>(h) * w;
      const std::size_t wtokens = static_cast<std::size_t>(hp) * wp;
      if (!normed.Reserve(tokens * c, error) || !win.Reserve(wtokens * c, error) ||
          !qkv.Reserve(wtokens * 3 * c, error) || !att.Reserve(wtokens * c, error) ||
          !proj.Reserve(wtokens * c, error) || !mid.Reserve(tokens * 4 * c, error))
        return false;
      LayerNorm(hidden.f(), S(p + "layernorm_before.weight"), S(p + "layernorm_before.bias"),
                normed.f(), static_cast<int>(tokens), c, kEps);
      WindowGather(normed.f(), h, w, c, ws, shift, win.f());
      const std::string a = p + "attention.self.";
      if (!gpu.Linear(win.f(), static_cast<int>(wtokens), S(a + "query.weight"),
                      S(a + "query.bias"), c, c, qkv.f(), Act::kNone, error, 0, 3 * c) ||
          !gpu.Linear(win.f(), static_cast<int>(wtokens), S(a + "key.weight"),
                      S(a + "key.bias"), c, c, qkv.f() + c, Act::kNone, error, 0, 3 * c) ||
          !gpu.Linear(win.f(), static_cast<int>(wtokens), S(a + "value.weight"),
                      S(a + "value.bias"), c, c, qkv.f() + 2 * c, Act::kNone, error, 0,
                      3 * c))
        return false;
      const int windows = static_cast<int>(wtokens / (ws * ws));
      const std::size_t threads = static_cast<std::size_t>(windows) * heads * ws * ws;
      WindowAttentionKernel<<<Blocks(threads), kThreads>>>(
          qkv.f(), c, heads, ws, shift, hp, wp, windows,
          S(a + "relative_position_bias_table"),
          1.0F / std::sqrt(static_cast<float>(c / heads)), att.f());
      if (!gpu.Linear(att.f(), static_cast<int>(wtokens), S(p + "attention.output.dense.weight"),
                      S(p + "attention.output.dense.bias"), c, c, proj.f(), Act::kNone, error))
        return false;
      WindowScatterAdd(proj.f(), h, w, c, ws, shift, hidden.f());
      LayerNorm(hidden.f(), S(p + "layernorm_after.weight"), S(p + "layernorm_after.bias"),
                normed.f(), static_cast<int>(tokens), c, kEps);
      if (!gpu.Linear(normed.f(), static_cast<int>(tokens), S(p + "intermediate.dense.weight"),
                      S(p + "intermediate.dense.bias"), 4 * c, c, mid.f(), Act::kGelu, error) ||
          !gpu.Linear(mid.f(), static_cast<int>(tokens), S(p + "output.dense.weight"),
                      S(p + "output.dense.bias"), c, 4 * c, proj.f(), Act::kNone, error))
        return false;
      Add(hidden.f(), proj.f(), tokens * c);
    }
    const std::size_t tokens = static_cast<std::size_t>(h) * w;
    if (stage >= 1) {
      const std::string norm = "hidden_states_norms.stage" + std::to_string(stage + 1);
      DevBuf out;
      if (!normed.Reserve(tokens * c, error) || !out.Reserve(tokens * c, error))
        return false;
      LayerNorm(hidden.f(), S(norm + ".weight"), S(norm + ".bias"), normed.f(),
                static_cast<int>(tokens), c, kEps);
      TokensToMap(normed.f(), static_cast<int>(tokens), c, c, out.f());
      maps->push_back(std::move(out));
      hs->push_back(h);
      wss->push_back(w);
      cs->push_back(c);
    }
    if (stage == 3)
      break;
    const int h2 = (h + 1) / 2, w2 = (w + 1) / 2;
    const std::size_t t2n = static_cast<std::size_t>(h2) * w2;
    const std::string d = "encoder.layers." + std::to_string(stage) + ".downsample.";
    if (!win.Reserve(t2n * 4 * c, error) || !normed.Reserve(t2n * 4 * c, error))
      return false;
    PatchMergeKernel<<<Blocks(t2n * 4 * c), kThreads>>>(hidden.f(), h, w, c, win.f());
    LayerNorm(win.f(), S(d + "norm.weight"), S(d + "norm.bias"), normed.f(),
              static_cast<int>(t2n), 4 * c, kEps);
    if (!hidden.Reserve(t2n * 2 * c, error) ||
        !gpu.Linear(normed.f(), static_cast<int>(t2n), S(d + "reduction.weight"), nullptr,
                    2 * c, 4 * c, hidden.f(), Act::kNone, error))
      return false;
    h = h2;
    w = w2;
    c *= 2;
  }
  return Sync("Swin", error);
}

bool GroundingDino::Impl::Bert(const std::vector<int>& ids, const std::vector<int>& positions,
                               const DevBuf& mask, DevBuf* text, std::string* error) {
  const int n = static_cast<int>(ids.size()), D = bert_dim;
  const std::string e = "model.text_backbone.embeddings.";
  const auto word = w.Host(e + "word_embeddings.weight");
  const auto pos = w.Host(e + "position_embeddings.weight");
  const auto type = w.Host(e + "token_type_embeddings.weight");
  std::vector<float> sum(static_cast<std::size_t>(n) * D);
  for (int i = 0; i < n; ++i)
    for (int c = 0; c < D; ++c)
      sum[static_cast<std::size_t>(i) * D + c] =
          word[static_cast<std::size_t>(ids[i]) * D + c] +
          type[c] + pos[static_cast<std::size_t>(positions[i]) * D + c];
  DevBuf x, h, qkv, att, lin, mid;
  const std::size_t nd = static_cast<std::size_t>(n) * D;
  if (!x.Upload(sum.data(), sum.size(), error) || !h.Reserve(nd, error) ||
      !qkv.Reserve(nd * 3, error) || !att.Reserve(nd, error) || !lin.Reserve(nd, error) ||
      !mid.Reserve(static_cast<std::size_t>(n) * 4 * D, error))
    return false;
  LayerNorm(x.f(), W(e + "LayerNorm.weight"), W(e + "LayerNorm.bias"), h.f(), n, D, kBertEps);
  for (int layer = 0; layer < bert_layers; ++layer) {
    const std::string p = "model.text_backbone.encoder.layer." + std::to_string(layer) + ".";
    const std::string a = p + "attention.self.";
    if (!gpu.Linear(h.f(), n, W(a + "query.weight"), W(a + "query.bias"), D, D, qkv.f(),
                    Act::kNone, error, 0, 3 * D) ||
        !gpu.Linear(h.f(), n, W(a + "key.weight"), W(a + "key.bias"), D, D, qkv.f() + D,
                    Act::kNone, error, 0, 3 * D) ||
        !gpu.Linear(h.f(), n, W(a + "value.weight"), W(a + "value.bias"), D, D,
                    qkv.f() + 2 * D, Act::kNone, error, 0, 3 * D) ||
        !gpu.Attention(qkv.f(), 3 * D, qkv.f() + D, 3 * D, qkv.f() + 2 * D, 3 * D, n, n,
                       bert_heads, D / bert_heads,
                       1.0F / std::sqrt(static_cast<float>(D / bert_heads)), att.f(), D,
                       error, nullptr, mask.f()) ||
        !gpu.Linear(att.f(), n, W(p + "attention.output.dense.weight"),
                    W(p + "attention.output.dense.bias"), D, D, lin.f(), Act::kNone, error))
      return false;
    Add(lin.f(), h.f(), nd);
    LayerNorm(lin.f(), W(p + "attention.output.LayerNorm.weight"),
              W(p + "attention.output.LayerNorm.bias"), h.f(), n, D, kBertEps);
    if (!gpu.Linear(h.f(), n, W(p + "intermediate.dense.weight"),
                    W(p + "intermediate.dense.bias"), 4 * D, D, mid.f(), Act::kGelu, error) ||
        !gpu.Linear(mid.f(), n, W(p + "output.dense.weight"), W(p + "output.dense.bias"), D,
                    4 * D, lin.f(), Act::kNone, error))
      return false;
    Add(lin.f(), h.f(), nd);
    LayerNorm(lin.f(), W(p + "output.LayerNorm.weight"), W(p + "output.LayerNorm.bias"), h.f(),
              n, D, kBertEps);
  }
  return text->Reserve(static_cast<std::size_t>(n) * kD, error) &&
         gpu.Linear(h.f(), n, W("model.text_projection.weight"),
                    W("model.text_projection.bias"), kD, D, text->f(), Act::kNone, error);
}

bool GroundingDino::Run(const std::uint8_t* rgb, int width, int height,
                        const std::string& prompt, float box_threshold,
                        float text_threshold, std::vector<GroundingBox>* boxes,
                        GroundingRaw* raw, std::string* error) {
  Impl& m = *impl_;
  // Text.
  std::vector<int> ids = m.tokenizer.Encode(prompt);
  if (ids.size() > static_cast<std::size_t>(kMaxText))
    ids.resize(kMaxText);
  const int nt = static_cast<int>(ids.size());
  std::vector<float> allowed;
  std::vector<int> positions;
  GroundingTextMasks(ids, &allowed, &positions);
  std::vector<float> bias(allowed.size());
  for (std::size_t i = 0; i < bias.size(); ++i)
    bias[i] = allowed[i] != 0.0F ? 0.0F : kMinFloat;
  DevBuf text_bias, text;
  if (!text_bias.Upload(bias.data(), bias.size(), error) ||
      !m.Bert(ids, positions, text_bias, &text, error))
    return false;
  Dump("text_features", text.f(), static_cast<std::size_t>(nt) * kD);
  // Image.
  int pw = 0, ph = 0;
  const std::vector<float> pixels = GroundingPixels(rgb, width, height, &pw, &ph);
  if (pixels.empty()) {
    SetError(error, "Grounding DINO: invalid image");
    return false;
  }
  std::vector<DevBuf> maps;
  std::vector<int> mh, mw, mc;
  if (!m.Swin(pixels, ph, pw, &maps, &mh, &mw, &mc, error))
    return false;
  for (std::size_t i = 0; i < maps.size(); ++i)
    Dump(("backbone" + std::to_string(i)).c_str(), maps[i].f(),
         static_cast<std::size_t>(mc[i]) * mh[i] * mw[i]);
  // Input projections (1x1 + GroupNorm; the fourth level 3x3 / 2 on the last
  // backbone map) and the flattened sources.
  LevelShapes shapes{};
  std::vector<DevBuf> sources(kLevels);
  int total = 0;
  DevBuf conv;
  for (int l = 0; l < kLevels; ++l) {
    const std::string p = "model.input_proj_vision." + std::to_string(l);
    int h = 0, w = 0;
    if (l < 3) {
      h = mh[l];
      w = mw[l];
      if (!conv.Reserve(static_cast<std::size_t>(kD) * h * w, error) ||
          !m.gpu.Conv2d(maps[l].f(), mc[l], h, w, m.W(p + ".0.weight"), m.W(p + ".0.bias"), kD,
                        1, 1, 0, conv.f(), nullptr, nullptr, error))
        return false;
    } else {
      if (!conv.Reserve(static_cast<std::size_t>(kD) * ((mh[2] + 1) / 2 + 1) *
                            ((mw[2] + 1) / 2 + 1),
                        error) ||
          !m.gpu.Conv2d(maps[2].f(), mc[2], mh[2], mw[2], m.W(p + ".0.weight"),
                        m.W(p + ".0.bias"), kD, 3, 2, 1, conv.f(), &h, &w, error))
        return false;
    }
    shapes.h[l] = h;
    shapes.w[l] = w;
    shapes.start[l] = total;
    total += h * w;
    DevBuf normed;
    if (!normed.Reserve(static_cast<std::size_t>(kD) * h * w, error) ||
        !sources[l].Reserve(static_cast<std::size_t>(kD) * h * w, error))
      return false;
    GroupNorm(conv.f(), m.W(p + ".1.weight"), m.W(p + ".1.bias"), normed.f(), kD,
              static_cast<std::size_t>(h) * w, 32, kEps);
    MapToTokens(normed.f(), h * w, kD, sources[l].f());
  }
  const int nv = total;
  const std::size_t nvd = static_cast<std::size_t>(nv) * kD;
  DevBuf vision, vpos, ref2;
  if (!vision.Reserve(nvd, error))
    return false;
  std::vector<float> pos_host(nvd), ref_host(static_cast<std::size_t>(nv) * 2);
  for (int l = 0; l < kLevels; ++l) {
    const int h = shapes.h[l], w = shapes.w[l];
    hipMemcpy(vision.f() + static_cast<std::size_t>(shapes.start[l]) * kD, sources[l].f(),
              static_cast<std::size_t>(h) * w * kD * 4, hipMemcpyDeviceToDevice);
    const float scale = 6.283185307179586F;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const std::size_t t = static_cast<std::size_t>(shapes.start[l]) + y * w + x;
        float* o = pos_host.data() + t * kD;
        const float ye = static_cast<float>(y + 1) / (static_cast<float>(h) + 1e-6F) * scale;
        const float xe = static_cast<float>(x + 1) / (static_cast<float>(w) + 1e-6F) * scale;
        for (int c = 0; c < kD / 2; ++c) {
          const float dim_t =
              std::pow(20.0F, 2.0F * static_cast<float>(c / 2) / static_cast<float>(kD / 2));
          o[c] = (c % 2 == 0) ? std::sin(ye / dim_t) : std::cos(ye / dim_t);
          o[kD / 2 + c] = (c % 2 == 0) ? std::sin(xe / dim_t) : std::cos(xe / dim_t);
        }
        for (int c = 0; c < kD; ++c)
          o[c] += m.level_embed[static_cast<std::size_t>(l) * kD + c];
        ref_host[t * 2] = (static_cast<float>(x) + 0.5F) / static_cast<float>(w);
        ref_host[t * 2 + 1] = (static_cast<float>(y) + 0.5F) / static_cast<float>(h);
      }
  }
  // Text position embeddings (position ids, 256 features, T 10000).
  std::vector<float> tpos_host(static_cast<std::size_t>(nt) * kD);
  for (int i = 0; i < nt; ++i)
    Sine(static_cast<float>(positions[i]), kD, 10000.0F, tpos_host.data() + static_cast<std::size_t>(i) * kD);
  DevBuf tpos;
  if (!vpos.Upload(pos_host.data(), nvd, error) ||
      !ref2.Upload(ref_host.data(), ref_host.size(), error) ||
      !tpos.Upload(tpos_host.data(), tpos_host.size(), error))
    return false;
  Dump("sources", vision.f(), nvd);
  Dump("vpos", vpos.f(), nvd);
  // Encoder.
  const std::size_t ntd = static_cast<std::size_t>(nt) * kD;
  DevBuf vn, tn, qv, kt, vv, vt, vo, to, dv, dt, tmp, ffn;
  if (!vn.Reserve(nvd, error) || !tn.Reserve(ntd, error) ||
      !qv.Reserve(nvd * 4, error) || !kt.Reserve(ntd * 4, error) ||
      !vv.Reserve(nvd * 4, error) || !vt.Reserve(ntd * 4, error) ||
      !vo.Reserve(nvd * 4, error) || !to.Reserve(ntd * 4, error) || !dv.Reserve(nvd, error) ||
      !dt.Reserve(ntd, error) || !tmp.Reserve(nvd, error) ||
      !ffn.Reserve(static_cast<std::size_t>(std::max(nv, kQueries)) * m.ffn, error))
    return false;
  const int fh = 4, fd = 1024, fhd = fd / fh;
  for (int layer = 0; layer < m.enc_layers; ++layer) {
    const std::string p = "model.encoder.layers." + std::to_string(layer) + ".";
    // Fusion (bi-directional attention between the normalised features).
    const std::string f = p + "fusion_layer.";
    LayerNorm(vision.f(), m.W(f + "layer_norm_vision.weight"), m.W(f + "layer_norm_vision.bias"),
              vn.f(), nv, kD, kEps);
    LayerNorm(text.f(), m.W(f + "layer_norm_text.weight"), m.W(f + "layer_norm_text.bias"),
              tn.f(), nt, kD, kEps);
    if (!m.Lin(vn.f(), nv, f + "attn.vision_proj", fd, kD, qv.f(), Act::kNone, error) ||
        !m.Lin(tn.f(), nt, f + "attn.text_proj", fd, kD, kt.f(), Act::kNone, error) ||
        !m.Lin(vn.f(), nv, f + "attn.values_vision_proj", fd, kD, vv.f(), Act::kNone, error) ||
        !m.Lin(tn.f(), nt, f + "attn.values_text_proj", fd, kD, vt.f(), Act::kNone, error))
      return false;
    const float fs = 1.0F / std::sqrt(static_cast<float>(fhd));
    if (!m.gpu.Attention(qv.f(), fd, kt.f(), fd, vt.f(), fd, nv, nt, fh, fhd, fs, vo.f(), fd,
                         error) ||
        !m.gpu.Attention(kt.f(), fd, qv.f(), fd, vv.f(), fd, nt, nv, fh, fhd, fs, to.f(), fd,
                         error) ||
        !m.Lin(vo.f(), nv, f + "attn.out_vision_proj", kD, fd, dv.f(), Act::kNone, error) ||
        !m.Lin(to.f(), nt, f + "attn.out_text_proj", kD, fd, dt.f(), Act::kNone, error))
      return false;
    ScaledAdd(vn.f(), dv.f(), m.W(f + "vision_param"), nv, kD);
    ScaledAdd(tn.f(), dt.f(), m.W(f + "text_param"), nt, kD);
    hipMemcpy(vision.f(), vn.f(), nvd * 4, hipMemcpyDeviceToDevice);
    hipMemcpy(text.f(), tn.f(), ntd * 4, hipMemcpyDeviceToDevice);
    // Text enhancer.
    const std::string te = p + "text_enhancer_layer.";
    AddTo(tn.f(), text.f(), tpos.f(), ntd);
    if (!m.Mha(tn.f(), tn.f(), text.f(), nt, nt, te + "self_attn", m.enc_heads / 2,
               text_bias.f(), dt.f(), error))
      return false;
    Add(text.f(), dt.f(), ntd);
    LayerNorm(text.f(), m.W(te + "layer_norm_before.weight"), m.W(te + "layer_norm_before.bias"),
              tn.f(), nt, kD, kEps);
    if (!m.Lin(tn.f(), nt, te + "fc1", m.ffn / 2, kD, ffn.f(), Act::kRelu, error) ||
        !m.Lin(ffn.f(), nt, te + "fc2", kD, m.ffn / 2, dt.f(), Act::kNone, error))
      return false;
    Add(dt.f(), tn.f(), ntd);
    LayerNorm(dt.f(), m.W(te + "layer_norm_after.weight"), m.W(te + "layer_norm_after.bias"),
              text.f(), nt, kD, kEps);
    // Deformable self-attention on the vision tokens.
    const std::string dl = p + "deformable_layer.";
    if (!m.Deformable(vision.f(), vpos.f(), nv, vision.f(), nv, shapes, ref2, 2,
                      dl + "self_attn", m.enc_heads, dv.f(), error))
      return false;
    Add(dv.f(), vision.f(), nvd);
    LayerNorm(dv.f(), m.W(dl + "self_attn_layer_norm.weight"),
              m.W(dl + "self_attn_layer_norm.bias"), vision.f(), nv, kD, kEps);
    if (!m.Lin(vision.f(), nv, dl + "fc1", m.ffn, kD, ffn.f(), Act::kRelu, error) ||
        !m.Lin(ffn.f(), nv, dl + "fc2", kD, m.ffn, dv.f(), Act::kNone, error))
      return false;
    Add(dv.f(), vision.f(), nvd);
    LayerNorm(dv.f(), m.W(dl + "final_layer_norm.weight"), m.W(dl + "final_layer_norm.bias"),
              vision.f(), nv, kD, kEps);
    if (!Sync("Grounding DINO encoder", error))
      return false;
  }
  Dump("enc_vision", vision.f(), nvd);
  Dump("enc_text", text.f(), ntd);
  // Two-stage proposals.
  std::vector<float> keep(nv), proposals(static_cast<std::size_t>(nv) * 4);
  for (int l = 0; l < kLevels; ++l) {
    const int h = shapes.h[l], w = shapes.w[l];
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const std::size_t t = static_cast<std::size_t>(shapes.start[l]) + y * w + x;
        const float wh = 0.05F * std::pow(2.0F, static_cast<float>(l));
        const float v[4] = {(static_cast<float>(x) + 0.5F) / static_cast<float>(w),
                            (static_cast<float>(y) + 0.5F) / static_cast<float>(h), wh, wh};
        bool ok = true;
        for (float c : v)
          ok = ok && c > 0.01F && c < 0.99F;
        keep[t] = ok ? 1.0F : 0.0F;
        for (int k = 0; k < 4; ++k)
          proposals[t * 4 + k] = ok ? std::log(v[k] / (1.0F - v[k])) : INFINITY;
      }
  }
  DevBuf keep_d, object, scores_d, delta_d;
  if (!keep_d.Upload(keep.data(), keep.size(), error) || !object.Reserve(nvd, error) ||
      !scores_d.Reserve(static_cast<std::size_t>(nv) * nt, error) ||
      !delta_d.Reserve(static_cast<std::size_t>(nv) * 4, error))
    return false;
  hipMemcpy(tmp.f(), vision.f(), nvd * 4, hipMemcpyDeviceToDevice);
  RowKeepKernel<<<Blocks(nvd), kThreads>>>(tmp.f(), keep_d.f(), kD, nvd);
  if (!m.Lin(tmp.f(), nv, "model.enc_output", kD, kD, dv.f(), Act::kNone, error))
    return false;
  LayerNorm(dv.f(), m.W("model.enc_output_norm.weight"), m.W("model.enc_output_norm.bias"),
            object.f(), nv, kD, kEps);
  if (!m.gpu.Gemm(false, true, nv, nt, kD, m.logit_scale, object.f(), kD, text.f(), kD, 0.0F,
                  scores_d.f(), nt, error) ||
      !m.Mlp(object.f(), nv, "model.encoder_output_bbox_embed", 3, kD, 4, delta_d.f(), error) ||
      !Sync("Grounding DINO proposals", error))
    return false;
  std::vector<float> enc_scores(static_cast<std::size_t>(nv) * nt), delta(static_cast<std::size_t>(nv) * 4);
  if (!scores_d.Download(enc_scores.data(), enc_scores.size(), error) ||
      !delta_d.Download(delta.data(), delta.size(), error))
    return false;
  if (m.mm)
    for (float& v : enc_scores)
      v += m.enc_bias;
  std::vector<float> best(nv);
  for (int t = 0; t < nv; ++t) {
    float b = -INFINITY;
    for (int k = 0; k < nt; ++k)
      b = std::max(b, enc_scores[static_cast<std::size_t>(t) * nt + k]);
    best[t] = b;
  }
  std::vector<int> order(nv);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return best[a] > best[b]; });
  // Oracle debugging: GUFO_VISION_TOPK=<file> replaces the top-k order (one
  // token index per line), to check the decoder apart from near-ties.
  if (const char* topk = std::getenv("GUFO_VISION_TOPK")) {
    std::ifstream in(topk);
    for (int q = 0; q < kQueries && (in >> order[q]); ++q) {
    }
  }
  std::vector<float> ref(static_cast<std::size_t>(kQueries) * 4);
  for (int q = 0; q < kQueries; ++q)
    for (int k = 0; k < 4; ++k) {
      const std::size_t s = static_cast<std::size_t>(order[q]) * 4 + k;
      ref[static_cast<std::size_t>(q) * 4 + k] = Sigmoid(delta[s] + proposals[s]);
    }
  if (std::getenv("GUFO_VISION_DUMP") != nullptr) {
    std::ofstream o(std::string(std::getenv("GUFO_VISION_DUMP")) + "/topk.txt");
    for (int q = 0; q < kQueries; ++q)
      o << order[q] << '\n';
  }
  // Decoder.
  const std::size_t qd = static_cast<std::size_t>(kQueries) * kD;
  DevBuf hidden, qpos, sine_d, ref_d, a, b2, box_d;
  if (!hidden.Reserve(qd, error) || !qpos.Reserve(qd, error) || !a.Reserve(qd, error) ||
      !b2.Reserve(qd, error) || !box_d.Reserve(static_cast<std::size_t>(kQueries) * 4, error))
    return false;
  hipMemcpy(hidden.f(), m.W("model.query_position_embeddings.weight"), qd * 4,
            hipMemcpyDeviceToDevice);
  std::vector<float> sine(static_cast<std::size_t>(kQueries) * 2 * kD);
  std::vector<float> ref_prev = ref;
  for (int layer = 0; layer < m.dec_layers; ++layer) {
    const std::string p = "model.decoder.layers." + std::to_string(layer) + ".";
    for (int q = 0; q < kQueries; ++q) {
      float* o = sine.data() + static_cast<std::size_t>(q) * 2 * kD;
      const float* r = ref.data() + static_cast<std::size_t>(q) * 4;
      // Order y, x, w, h (the first two swapped), 128 features each.
      Sine(r[1], kD / 2, 10000.0F, o);
      Sine(r[0], kD / 2, 10000.0F, o + kD / 2);
      Sine(r[2], kD / 2, 10000.0F, o + kD);
      Sine(r[3], kD / 2, 10000.0F, o + 3 * kD / 2);
    }
    if (!sine_d.Upload(sine.data(), sine.size(), error) ||
        !ref_d.Upload(ref.data(), ref.size(), error) ||
        !m.Mlp(sine_d.f(), kQueries, "model.decoder.reference_points_head", 2, 2 * kD, kD,
               qpos.f(), error))
      return false;
    if (layer == 0) {
      Dump("init_ref", ref_d.f(), ref.size());
      Dump("qpos0", qpos.f(), qd);
    }
    // Self-attention.
    AddTo(a.f(), hidden.f(), qpos.f(), qd);
    if (!m.Mha(a.f(), a.f(), hidden.f(), kQueries, kQueries, p + "self_attn", m.dec_heads,
               nullptr, b2.f(), error))
      return false;
    Add(b2.f(), hidden.f(), qd);
    LayerNorm(b2.f(), m.W(p + "self_attn_layer_norm.weight"), m.W(p + "self_attn_layer_norm.bias"),
              hidden.f(), kQueries, kD, kEps);
    if (layer == 0)
      Dump("l0_self", hidden.f(), qd);
    // Text cross-attention.
    AddTo(a.f(), hidden.f(), qpos.f(), qd);
    if (!m.Mha(a.f(), text.f(), text.f(), kQueries, nt, p + "encoder_attn_text", m.dec_heads,
               nullptr, b2.f(), error))
      return false;
    Add(b2.f(), hidden.f(), qd);
    LayerNorm(b2.f(), m.W(p + "encoder_attn_text_layer_norm.weight"),
              m.W(p + "encoder_attn_text_layer_norm.bias"), hidden.f(), kQueries, kD, kEps);
    if (layer == 0)
      Dump("l0_text", hidden.f(), qd);
    // Deformable cross-attention.
    if (!m.Deformable(hidden.f(), qpos.f(), kQueries, vision.f(), nv, shapes, ref_d, 4,
                      p + "encoder_attn", m.dec_heads, b2.f(), error))
      return false;
    Add(b2.f(), hidden.f(), qd);
    LayerNorm(b2.f(), m.W(p + "encoder_attn_layer_norm.weight"),
              m.W(p + "encoder_attn_layer_norm.bias"), hidden.f(), kQueries, kD, kEps);
    if (layer == 0)
      Dump("l0_deform", hidden.f(), qd);
    if (!m.Lin(hidden.f(), kQueries, p + "fc1", m.ffn, kD, ffn.f(), Act::kRelu, error) ||
        !m.Lin(ffn.f(), kQueries, p + "fc2", kD, m.ffn, b2.f(), Act::kNone, error))
      return false;
    Add(b2.f(), hidden.f(), qd);
    LayerNorm(b2.f(), m.W(p + "final_layer_norm.weight"), m.W(p + "final_layer_norm.bias"),
              hidden.f(), kQueries, kD, kEps);
    Dump(("dec" + std::to_string(layer)).c_str(), hidden.f(), qd);
    // Box refinement.
    std::vector<float> tmp_box(static_cast<std::size_t>(kQueries) * 4);
    if (!m.Mlp(hidden.f(), kQueries, m.BoxHead(layer), 3, kD, 4, box_d.f(), error) ||
        !Sync("Grounding DINO decoder", error) ||
        !box_d.Download(tmp_box.data(), tmp_box.size(), error))
      return false;
    ref_prev = ref;
    for (std::size_t i = 0; i < ref.size(); ++i)
      ref[i] = Sigmoid(tmp_box[i] + Logit(ref[i]));
    if (std::getenv("GUFO_VISION_DUMP") != nullptr) {
      std::ofstream o(std::string(std::getenv("GUFO_VISION_DUMP")) + "/ref" +
                          std::to_string(layer) + ".f32",
                      std::ios::binary);
      o.write(reinterpret_cast<const char*>(ref.data()),
              static_cast<std::streamsize>(ref.size() * 4));
      o.close();
      std::ofstream t(std::string(std::getenv("GUFO_VISION_DUMP")) + "/tmp" +
                          std::to_string(layer) + ".f32",
                      std::ios::binary);
      t.write(reinterpret_cast<const char*>(tmp_box.data()),
              static_cast<std::streamsize>(tmp_box.size() * 4));
    }
  }
  // Heads on the normalised last decoder state.
  DevBuf logits_d;
  LayerNorm(hidden.f(), m.W("model.decoder.layer_norm.weight"), m.W("model.decoder.layer_norm.bias"),
            a.f(), kQueries, kD, kEps);
  if (!logits_d.Reserve(static_cast<std::size_t>(kQueries) * nt, error) ||
      !m.gpu.Gemm(false, true, kQueries, nt, kD, m.logit_scale, a.f(), kD, text.f(), kD, 0.0F,
                  logits_d.f(), nt, error) ||
      !m.Mlp(a.f(), kQueries, m.BoxHead(m.dec_layers - 1), 3, kD, 4, box_d.f(), error) ||
      !Sync("Grounding DINO heads", error))
    return false;
  std::vector<float> logits(static_cast<std::size_t>(kQueries) * nt),
      deltas(static_cast<std::size_t>(kQueries) * 4), out_boxes(deltas.size());
  if (!logits_d.Download(logits.data(), logits.size(), error) ||
      !box_d.Download(deltas.data(), deltas.size(), error))
    return false;
  for (std::size_t i = 0; i < deltas.size(); ++i)
    out_boxes[i] = Sigmoid(deltas[i] + Logit(ref_prev[i]));
  if (m.mm)
    for (float& v : logits)
      v += m.final_bias;
  if (raw != nullptr) {
    raw->logits = logits;
    raw->boxes = out_boxes;
    raw->ids = ids;
  }
  // post_process_grounded_object_detection.
  boxes->clear();
  for (int q = 0; q < kQueries; ++q) {
    float score = 0;
    std::vector<int> label_ids;
    for (int k = 0; k < nt; ++k) {
      const float prob = Sigmoid(logits[static_cast<std::size_t>(q) * nt + k]);
      score = std::max(score, prob);
    }
    if (!(score > box_threshold))
      continue;
    for (int k = 1; k < nt && k < kMaxText - 1; ++k)
      if (Sigmoid(logits[static_cast<std::size_t>(q) * nt + k]) > text_threshold)
        label_ids.push_back(ids[k]);
    const float* b = out_boxes.data() + static_cast<std::size_t>(q) * 4;
    GroundingBox box;
    box.box = {(b[0] - 0.5F * b[2]) * width, (b[1] - 0.5F * b[3]) * height,
               (b[0] + 0.5F * b[2]) * width, (b[1] + 0.5F * b[3]) * height};
    box.score = score;
    box.label = m.tokenizer.Decode(label_ids);
    boxes->push_back(std::move(box));
  }
  return true;
}

}  // namespace gufo::models::vision_dsg
