// SAM ViT-H on the GPU (see sam.hpp).
// Written with AI assistance (Claude Code).
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <utility>

#include "src/core/json.hpp"
#include "src/models/vision_dsg/hip_ops.hpp"
#include "src/models/vision_dsg/sam.hpp"

namespace gufo::models::vision_dsg {
namespace {

constexpr int kThreads = 256;
constexpr int kSide = 1024;
constexpr int kGrid = 64;      // image embedding side
constexpr int kD = 256;        // prompt / decoder width
constexpr int kMaskTokens = 4;
constexpr float kEps = 1e-6F;
constexpr float kTwoPi = 6.283185307179586F;

unsigned Blocks(std::size_t n) {
  return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

// rel[t, k] = sum_c q[t, c] * table[pos(t) - k + side - 1, c] for the
// row (height) or column (width) coordinate of token t inside its window.
__global__ void RelPosKernel(const float* q, int ldq, int tokens, int side,
                             int hd, const float* table, int along_height,
                             float* rel) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(tokens) * side)
    return;
  const int k = static_cast<int>(i % side);
  const int t = static_cast<int>(i / side);
  const int local = t % (side * side);
  const int coord = along_height ? local / side : local % side;
  const float* row = table + static_cast<std::size_t>(coord - k + side - 1) * hd;
  const float* qt = q + static_cast<std::size_t>(t) * ldq;
  float sum = 0;
  for (int c = 0; c < hd; ++c)
    sum += qt[c] * row[c];
  rel[i] = sum;
}

// scores[w, i, j] = scores * scale + rel_h[w*n + i, j / side] + rel_w[.., j % side].
__global__ void RelBiasKernel(float* scores, const float* rel_h, const float* rel_w,
                              int n, int side, float scale, std::size_t total) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total)
    return;
  const int j = static_cast<int>(i % n);
  const std::size_t row = i / n;  // window * n + query
  scores[i] = scores[i] * scale + rel_h[row * side + j / side] + rel_w[row * side + j % side];
}

// Rows [y0 .. y0 + h) x [0, w) of a [rows, stride] map into a dense [h, w].
__global__ void CropKernel(const float* in, int stride, int h, int w, float* out) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(h) * w)
    return;
  out[i] = in[(i / w) * stride + i % w];
}

}  // namespace

struct Sam::Impl {
  Gpu gpu;
  WeightStore w;
  int dim{1280}, heads{16}, layers{32}, window{14}, mlp{5120};
  std::vector<int> global;
  std::vector<float> gauss;  // [2, 128]
  std::vector<float> image_pe;

  const float* W(const std::string& name) const { return w.Get(name); }
  bool Lin(const float* in, int rows, const std::string& name, int out_dim, int in_dim,
           float* out, Act act, std::string* error) {
    return gpu.Linear(in, rows, W(name + ".weight"),
                      w.Has(name + ".bias") ? W(name + ".bias") : nullptr, out_dim, in_dim,
                      out, act, error);
  }
  void Embed(float x, float y, float* out) const {
    // SamPositionalEmbedding on coordinates already in [0, 1].
    const float cx = 2 * x - 1, cy = 2 * y - 1;
    for (int k = 0; k < kD / 2; ++k) {
      const float v = kTwoPi * (cx * gauss[k] + cy * gauss[kD / 2 + k]);
      out[k] = std::sin(v);
      out[kD / 2 + k] = std::cos(v);
    }
  }
  bool Encode(const std::vector<float>& pixels, DevBuf* embedding, std::string* error);
  bool Attn(const float* q_in, int tq, const float* k_in, const float* v_in, int tk,
            const std::string& p, int internal, float* out, std::string* error);
  bool Ffn(const float* in, int rows, const std::string& p, int hidden_layers, int out_dim,
           float* out, std::string* error);
  DevBuf a1, a2, a3, a4, f1, f2;
};

Sam::Sam(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Sam::~Sam() = default;
std::size_t Sam::weight_bytes() const { return impl_->w.bytes(); }

std::unique_ptr<Sam> Sam::Load(const std::filesystem::path& dir, std::string* error) {
  std::ifstream file(dir / "config.json", std::ios::binary);
  if (!file) {
    SetError(error, "SAM: no config.json in " + dir.string());
    return nullptr;
  }
  std::stringstream text;
  text << file.rdbuf();
  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  try {
    const json::Value config = json::parse(text.str());
    const json::Value* v = config.find("vision_config");
    if (v == nullptr || !v->is_object()) {
      SetError(error, dir.string() + " is not a SAM checkpoint");
      return nullptr;
    }
    m.dim = static_cast<int>(v->member_double("hidden_size", 768));
    m.heads = static_cast<int>(v->member_double("num_attention_heads", 12));
    m.layers = static_cast<int>(v->member_double("num_hidden_layers", 12));
    m.window = static_cast<int>(v->member_double("window_size", 14));
    m.mlp = static_cast<int>(v->member_double("mlp_dim", m.dim * 4));
    if (const json::Value* g = v->find("global_attn_indexes"))
      for (const auto& x : g->items())
        m.global.push_back(static_cast<int>(x.as_double()));
    if (static_cast<int>(v->member_double("image_size", 1024)) != kSide ||
        static_cast<int>(v->member_double("patch_size", 16)) != 16 ||
        static_cast<int>(v->member_double("output_channels", 256)) != kD) {
      SetError(error, "SAM: unsupported configuration");
      return nullptr;
    }
  } catch (const std::exception& e) {
    SetError(error, "SAM config.json: " + std::string(e.what()));
    return nullptr;
  }
  if (!m.gpu.Init(error) || !m.w.Load(dir / "model.safetensors", error))
    return nullptr;
  m.gauss = m.w.Host("shared_image_embedding.positional_embedding");
  if (m.gauss.size() != static_cast<std::size_t>(kD) ||
      m.W("vision_encoder.pos_embed") == nullptr ||
      m.W("mask_decoder.iou_prediction_head.proj_out.weight") == nullptr) {
    SetError(error, "SAM: weights do not match the configuration");
    return nullptr;
  }
  m.image_pe.resize(static_cast<std::size_t>(kGrid) * kGrid * kD);
  for (int y = 0; y < kGrid; ++y)
    for (int x = 0; x < kGrid; ++x)
      m.Embed((static_cast<float>(x + 1) - 0.5F) / kGrid, (static_cast<float>(y + 1) - 0.5F) / kGrid,
              m.image_pe.data() + (static_cast<std::size_t>(y) * kGrid + x) * kD);
  return std::unique_ptr<Sam>(new Sam(std::move(impl)));
}

bool Sam::Impl::Encode(const std::vector<float>& pixels, DevBuf* embedding, std::string* error) {
  const int T = kGrid * kGrid, C = dim, hd = dim / heads;
  constexpr int kPatch = 16, pv = 3 * kPatch * kPatch;
  std::vector<float> columns(static_cast<std::size_t>(T) * pv);
  for (int py = 0; py < kGrid; ++py)
    for (int px = 0; px < kGrid; ++px)
      for (int c = 0; c < 3; ++c)
        for (int y = 0; y < kPatch; ++y)
          for (int x = 0; x < kPatch; ++x)
            columns[(static_cast<std::size_t>(py) * kGrid + px) * pv + (c * kPatch + y) * kPatch + x] =
                pixels[(static_cast<std::size_t>(c) * kSide + py * kPatch + y) * kSide + px * kPatch + x];
  const std::size_t TC = static_cast<std::size_t>(T) * C;
  const int wtok_max = ((kGrid + window - 1) / window * window) * ((kGrid + window - 1) / window * window);
  const std::size_t rows_max = std::max<std::size_t>(T, wtok_max);
  DevBuf cols, hidden, normed, win, qkv, att, proj, mlp_buf, scores, rel_h, rel_w;
  if (!cols.Upload(columns.data(), columns.size(), error) || !hidden.Reserve(TC, error) ||
      !normed.Reserve(TC, error) || !win.Reserve(rows_max * C, error) ||
      !qkv.Reserve(rows_max * 3 * C, error) || !att.Reserve(rows_max * C, error) ||
      !proj.Reserve(rows_max * C, error) ||
      !mlp_buf.Reserve(static_cast<std::size_t>(T) * mlp, error) ||
      !scores.Reserve(static_cast<std::size_t>(T) * T, error) ||
      !rel_h.Reserve(rows_max * kGrid, error) || !rel_w.Reserve(rows_max * kGrid, error))
    return false;
  if (!gpu.Linear(cols.f(), T, W("vision_encoder.patch_embed.projection.weight"),
                  W("vision_encoder.patch_embed.projection.bias"), C, pv, hidden.f(), Act::kNone,
                  error))
    return false;
  Add(hidden.f(), W("vision_encoder.pos_embed"), TC);
  const float one = 1.0F, zero = 0.0F;
  const auto blas = static_cast<rocblas_handle>(gpu.blas());
  for (int layer = 0; layer < layers; ++layer) {
    const std::string p = "vision_encoder.layers." + std::to_string(layer) + ".";
    const bool is_global = std::find(global.begin(), global.end(), layer) != global.end();
    const int side = is_global ? kGrid : window;
    LayerNorm(hidden.f(), W(p + "layer_norm1.weight"), W(p + "layer_norm1.bias"), normed.f(), T, C,
              kEps);
    const float* src = normed.f();
    int rows = T;
    if (!is_global) {
      WindowGather(normed.f(), kGrid, kGrid, C, window, 0, win.f());
      src = win.f();
      rows = wtok_max;
    }
    const int n = side * side, windows = rows / n;
    if (!gpu.Linear(src, rows, W(p + "attn.qkv.weight"), W(p + "attn.qkv.bias"), 3 * C, C,
                    qkv.f(), Act::kNone, error))
      return false;
    const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
    const std::size_t per = static_cast<std::size_t>(n) * n;
    for (int h = 0; h < heads; ++h) {
      const float* q = qkv.f() + h * hd;
      const float* k = qkv.f() + C + h * hd;
      const float* v = qkv.f() + 2 * C + h * hd;
      if (rocblas_sgemm_strided_batched(blas, rocblas_operation_transpose, rocblas_operation_none,
                                        n, n, hd, &one, k, 3 * C,
                                        static_cast<rocblas_stride>(n) * 3 * C, q, 3 * C,
                                        static_cast<rocblas_stride>(n) * 3 * C, &zero, scores.f(),
                                        n, static_cast<rocblas_stride>(per),
                                        windows) != rocblas_status_success) {
        SetError(error, "SAM attention scores failed");
        return false;
      }
      const std::size_t rn = static_cast<std::size_t>(rows) * side;
      RelPosKernel<<<Blocks(rn), kThreads>>>(q, 3 * C, rows, side, hd,
                                             W(p + "attn.rel_pos_h") , 1, rel_h.f());
      RelPosKernel<<<Blocks(rn), kThreads>>>(q, 3 * C, rows, side, hd,
                                             W(p + "attn.rel_pos_w"), 0, rel_w.f());
      const std::size_t total = per * windows;
      RelBiasKernel<<<Blocks(total), kThreads>>>(scores.f(), rel_h.f(), rel_w.f(), n, side, scale,
                                                 total);
      Softmax(scores.f(), rows, n, 1.0F);
      if (rocblas_sgemm_strided_batched(blas, rocblas_operation_none, rocblas_operation_none, hd,
                                        n, n, &one, v, 3 * C,
                                        static_cast<rocblas_stride>(n) * 3 * C, scores.f(), n,
                                        static_cast<rocblas_stride>(per), &zero, att.f() + h * hd,
                                        C, static_cast<rocblas_stride>(n) * C,
                                        windows) != rocblas_status_success) {
        SetError(error, "SAM attention values failed");
        return false;
      }
    }
    if (!gpu.Linear(att.f(), rows, W(p + "attn.proj.weight"), W(p + "attn.proj.bias"), C, C,
                    proj.f(), Act::kNone, error))
      return false;
    if (is_global)
      Add(hidden.f(), proj.f(), TC);
    else
      WindowScatterAdd(proj.f(), kGrid, kGrid, C, window, 0, hidden.f());
    LayerNorm(hidden.f(), W(p + "layer_norm2.weight"), W(p + "layer_norm2.bias"), normed.f(), T, C,
              kEps);
    if (!gpu.Linear(normed.f(), T, W(p + "mlp.lin1.weight"), W(p + "mlp.lin1.bias"), mlp, C,
                    mlp_buf.f(), Act::kGelu, error) ||
        !gpu.Linear(mlp_buf.f(), T, W(p + "mlp.lin2.weight"), W(p + "mlp.lin2.bias"), C, mlp,
                    proj.f(), Act::kNone, error))
      return false;
    Add(hidden.f(), proj.f(), TC);
    if (!Sync("SAM encoder", error))
      return false;
  }
  // Neck: 1x1 (no bias), channel layer norm, 3x3 (no bias), layer norm.
  const std::size_t TD = static_cast<std::size_t>(T) * kD;
  if (!embedding->Reserve(TD, error) ||
      !gpu.Linear(hidden.f(), T, W("vision_encoder.neck.conv1.weight"), nullptr, kD, C,
                  normed.f(), Act::kNone, error))
    return false;
  LayerNorm(normed.f(), W("vision_encoder.neck.layer_norm1.weight"),
            W("vision_encoder.neck.layer_norm1.bias"), win.f(), T, kD, kEps);
  TokensToMap(win.f(), T, kD, kD, att.f());
  if (!gpu.Conv2d(att.f(), kD, kGrid, kGrid, W("vision_encoder.neck.conv2.weight"), nullptr, kD, 3,
                  1, 1, proj.f(), nullptr, nullptr, error))
    return false;
  MapToTokens(proj.f(), T, kD, normed.f());
  LayerNorm(normed.f(), W("vision_encoder.neck.layer_norm2.weight"),
            W("vision_encoder.neck.layer_norm2.bias"), embedding->f(), T, kD, kEps);
  return Sync("SAM neck", error);
}

// SamAttention: projections to `internal`, 8 heads, out_proj back to 256.
bool Sam::Impl::Attn(const float* q_in, int tq, const float* k_in, const float* v_in, int tk,
                     const std::string& p, int internal, float* out, std::string* error) {
  if (!a1.Reserve(static_cast<std::size_t>(tq) * internal, error) ||
      !a2.Reserve(static_cast<std::size_t>(tk) * internal, error) ||
      !a3.Reserve(static_cast<std::size_t>(tk) * internal, error) ||
      !a4.Reserve(static_cast<std::size_t>(tq) * internal, error) ||
      !Lin(q_in, tq, p + ".q_proj", internal, kD, a1.f(), Act::kNone, error) ||
      !Lin(k_in, tk, p + ".k_proj", internal, kD, a2.f(), Act::kNone, error) ||
      !Lin(v_in, tk, p + ".v_proj", internal, kD, a3.f(), Act::kNone, error))
    return false;
  constexpr int kHeads = 8;
  const int hd = internal / kHeads;
  return gpu.Attention(a1.f(), internal, a2.f(), internal, a3.f(), internal, tq, tk, kHeads, hd,
                       1.0F / std::sqrt(static_cast<float>(hd)), a4.f(), internal, error) &&
         Lin(a4.f(), tq, p + ".out_proj", kD, internal, out, Act::kNone, error);
}

// SamFeedForward: proj_in, ReLU, hidden layers with ReLU, proj_out.
bool Sam::Impl::Ffn(const float* in, int rows, const std::string& p, int hidden_layers,
                    int out_dim, float* out, std::string* error) {
  const std::size_t n = static_cast<std::size_t>(rows) * kD;
  if (!f1.Reserve(n, error) || !f2.Reserve(n, error) ||
      !Lin(in, rows, p + ".proj_in", kD, kD, f1.f(), Act::kRelu, error))
    return false;
  for (int l = 0; l < hidden_layers; ++l) {
    if (!Lin(f1.f(), rows, p + ".layers." + std::to_string(l), kD, kD, f2.f(), Act::kRelu, error))
      return false;
    std::swap(f1, f2);
  }
  return Lin(f1.f(), rows, p + ".proj_out", out_dim, kD, out, Act::kNone, error);
}

bool Sam::Run(const std::uint8_t* rgb, int width, int height, const SamPrompt& prompt,
              SamOutput* out, std::string* error) {
  Impl& m = *impl_;
  int nw = 0, nh = 0;
  const std::vector<float> pixels = SamPixels(rgb, width, height, &nw, &nh);
  if (pixels.empty()) {
    SetError(error, "SAM: invalid image");
    return false;
  }
  DevBuf embedding;
  if (!m.Encode(pixels, &embedding, error))
    return false;
  const int T = kGrid * kGrid;
  const std::size_t TD = static_cast<std::size_t>(T) * kD;
  std::vector<float> iou_token = m.w.Host("mask_decoder.iou_token.weight");
  std::vector<float> mask_tokens = m.w.Host("mask_decoder.mask_tokens.weight");
  const auto host = [&](const char* name) { return m.w.Host(name); };
  const std::vector<float> pe[4] = {host("prompt_encoder.point_embed.0.weight"),
                                    host("prompt_encoder.point_embed.1.weight"),
                                    host("prompt_encoder.point_embed.2.weight"),
                                    host("prompt_encoder.point_embed.3.weight")};
  const std::vector<float> not_a_point = host("prompt_encoder.not_a_point_embed.weight");
  // Prompt sets: one per box, or one for all points.
  struct Set {
    std::vector<float> sparse;  // [s, 256]
  };
  std::vector<Set> sets;
  const double sx = static_cast<double>(nw) / width, sy = static_cast<double>(nh) / height;
  const auto embed_point = [&](double x, double y, float* o) {
    const float px = static_cast<float>(x * sx) + 0.5F, py = static_cast<float>(y * sy) + 0.5F;
    m.Embed(px / kSide, py / kSide, o);
  };
  for (const auto& b : prompt.boxes) {
    Set s;
    s.sparse.resize(2 * kD);
    embed_point(b[0], b[1], s.sparse.data());
    embed_point(b[2], b[3], s.sparse.data() + kD);
    for (int c = 0; c < kD; ++c) {
      s.sparse[c] += pe[2][c];
      s.sparse[kD + c] += pe[3][c];
    }
    sets.push_back(std::move(s));
  }
  if (!prompt.points.empty()) {
    Set s;
    const std::size_t np = prompt.points.size();
    s.sparse.resize((np + 1) * kD);
    for (std::size_t i = 0; i < np; ++i) {
      float* o = s.sparse.data() + i * kD;
      embed_point(prompt.points[i][0], prompt.points[i][1], o);
      const int label = i < prompt.labels.size() ? prompt.labels[i] : 1;
      if (label == 0 || label == 1)
        for (int c = 0; c < kD; ++c)
          o[c] += pe[label][c];
    }
    // The padding point (label -1) is the not-a-point embedding.
    std::copy(not_a_point.begin(), not_a_point.end(), s.sparse.data() + np * kD);
    sets.push_back(std::move(s));
  }
  DevBuf keys0, kpe;
  if (!keys0.Reserve(TD, error) || !kpe.Upload(m.image_pe.data(), m.image_pe.size(), error))
    return false;
  hipMemcpy(keys0.f(), embedding.f(), TD * 4, hipMemcpyDeviceToDevice);
  AddRowBias(keys0.f(), m.W("prompt_encoder.no_mask_embed.weight"), T, kD);
  out->width = width;
  out->height = height;
  out->masks.clear();
  out->scores.clear();
  out->low_res.clear();
  DevBuf queries, qpe, keys, t1, t2, kq, mlpb, map, up1, up1n, up2, hyper, masks_d, big, crop, full;
  const std::string tp = "mask_decoder.transformer.";
  for (const Set& set : sets) {
    const int s = static_cast<int>(set.sparse.size() / kD);
    const int tq = 1 + kMaskTokens + s;
    std::vector<float> tok(static_cast<std::size_t>(tq) * kD);
    std::copy(iou_token.begin(), iou_token.end(), tok.begin());
    std::copy(mask_tokens.begin(), mask_tokens.end(), tok.begin() + kD);
    std::copy(set.sparse.begin(), set.sparse.end(), tok.begin() + (1 + kMaskTokens) * kD);
    const std::size_t qd = static_cast<std::size_t>(tq) * kD;
    if (!queries.Upload(tok.data(), tok.size(), error) || !qpe.Upload(tok.data(), tok.size(), error) ||
        !keys.Reserve(TD, error) || !t1.Reserve(TD, error) || !t2.Reserve(TD, error) ||
        !kq.Reserve(TD, error) || !mlpb.Reserve(static_cast<std::size_t>(tq) * 2048, error))
      return false;
    hipMemcpy(keys.f(), keys0.f(), TD * 4, hipMemcpyDeviceToDevice);
    for (int layer = 0; layer < 2; ++layer) {
      const std::string p = tp + "layers." + std::to_string(layer) + ".";
      if (layer == 0) {
        if (!m.Attn(queries.f(), tq, queries.f(), queries.f(), tq, p + "self_attn", kD, t1.f(), error))
          return false;
        hipMemcpy(queries.f(), t1.f(), qd * 4, hipMemcpyDeviceToDevice);
      } else {
        AddTo(t2.f(), queries.f(), qpe.f(), qd);
        if (!m.Attn(t2.f(), tq, t2.f(), queries.f(), tq, p + "self_attn", kD, t1.f(), error))
          return false;
        Add(queries.f(), t1.f(), qd);
      }
      LayerNorm(queries.f(), m.W(p + "layer_norm1.weight"), m.W(p + "layer_norm1.bias"), t1.f(), tq,
                kD, kEps);
      hipMemcpy(queries.f(), t1.f(), qd * 4, hipMemcpyDeviceToDevice);
      AddTo(t2.f(), queries.f(), qpe.f(), qd);
      AddTo(kq.f(), keys.f(), kpe.f(), TD);
      if (!m.Attn(t2.f(), tq, kq.f(), keys.f(), T, p + "cross_attn_token_to_image", kD / 2, t1.f(),
                  error))
        return false;
      Add(queries.f(), t1.f(), qd);
      LayerNorm(queries.f(), m.W(p + "layer_norm2.weight"), m.W(p + "layer_norm2.bias"), t1.f(), tq,
                kD, kEps);
      if (!m.Lin(t1.f(), tq, p + "mlp.lin1", 2048, kD, mlpb.f(), Act::kRelu, error) ||
          !m.Lin(mlpb.f(), tq, p + "mlp.lin2", kD, 2048, queries.f(), Act::kNone, error))
        return false;
      Add(queries.f(), t1.f(), qd);
      LayerNorm(queries.f(), m.W(p + "layer_norm3.weight"), m.W(p + "layer_norm3.bias"), t1.f(), tq,
                kD, kEps);
      hipMemcpy(queries.f(), t1.f(), qd * 4, hipMemcpyDeviceToDevice);
      AddTo(t2.f(), queries.f(), qpe.f(), qd);
      if (!m.Attn(kq.f(), T, t2.f(), queries.f(), tq, p + "cross_attn_image_to_token", kD / 2, t1.f(),
                  error))
        return false;
      Add(keys.f(), t1.f(), TD);
      LayerNorm(keys.f(), m.W(p + "layer_norm4.weight"), m.W(p + "layer_norm4.bias"), t1.f(), T, kD,
                kEps);
      hipMemcpy(keys.f(), t1.f(), TD * 4, hipMemcpyDeviceToDevice);
    }
    AddTo(t2.f(), queries.f(), qpe.f(), qd);
    AddTo(kq.f(), keys.f(), kpe.f(), TD);
    if (!m.Attn(t2.f(), tq, kq.f(), keys.f(), T, tp + "final_attn_token_to_image", kD / 2, t1.f(),
                error))
      return false;
    Add(queries.f(), t1.f(), qd);
    LayerNorm(queries.f(), m.W(tp + "layer_norm_final_attn.weight"),
              m.W(tp + "layer_norm_final_attn.bias"), t1.f(), tq, kD, 1e-5F);
    hipMemcpy(queries.f(), t1.f(), qd * 4, hipMemcpyDeviceToDevice);
    // Upscaling: [256, 64, 64] -> conv-T 64 -> LN2d -> GELU -> conv-T 32 -> GELU.
    constexpr int s1 = kGrid * 2, s2 = kGrid * 4;
    if (!map.Reserve(TD, error) || !up1.Reserve(static_cast<std::size_t>(64) * s1 * s1, error) ||
        !up1n.Reserve(static_cast<std::size_t>(64) * s1 * s1, error) ||
        !up2.Reserve(static_cast<std::size_t>(32) * s2 * s2, error))
      return false;
    TokensToMap(keys.f(), T, kD, kD, map.f());
    if (!m.gpu.ConvTransposeKS(map.f(), kD, kGrid, kGrid, m.W("mask_decoder.upscale_conv1.weight"),
                               m.W("mask_decoder.upscale_conv1.bias"), 64, 2, up1.f(), error))
      return false;
    MapToTokens(up1.f(), s1 * s1, 64, up1n.f());
    LayerNorm(up1n.f(), m.W("mask_decoder.upscale_layer_norm.weight"),
              m.W("mask_decoder.upscale_layer_norm.bias"), up1.f(), s1 * s1, 64, kEps);
    TokensToMap(up1.f(), s1 * s1, 64, 64, up1n.f());
    Activate(up1n.f(), static_cast<std::size_t>(64) * s1 * s1, Act::kGelu);
    if (!m.gpu.ConvTransposeKS(up1n.f(), 64, s1, s1, m.W("mask_decoder.upscale_conv2.weight"),
                               m.W("mask_decoder.upscale_conv2.bias"), 32, 2, up2.f(), error))
      return false;
    Activate(up2.f(), static_cast<std::size_t>(32) * s2 * s2, Act::kGelu);
    // Hypernetwork rows and IoU head.
    if (!hyper.Reserve(static_cast<std::size_t>(kMaskTokens) * 32, error) ||
        !masks_d.Reserve(static_cast<std::size_t>(kMaskTokens) * s2 * s2, error))
      return false;
    for (int i = 0; i < kMaskTokens; ++i)
      if (!m.Ffn(queries.f() + static_cast<std::size_t>(1 + i) * kD, 1,
                 "mask_decoder.output_hypernetworks_mlps." + std::to_string(i), 1, 32,
                 hyper.f() + i * 32, error))
        return false;
    DevBuf iou_d;
    if (!iou_d.Reserve(kMaskTokens, error) ||
        !m.Ffn(queries.f(), 1, "mask_decoder.iou_prediction_head", 1, kMaskTokens, iou_d.f(), error) ||
        !m.gpu.Gemm(false, false, kMaskTokens, s2 * s2, 32, 1.0F, hyper.f(), 32, up2.f(), s2 * s2,
                    0.0F, masks_d.f(), s2 * s2, error))
      return false;
    float iou[kMaskTokens];
    if (!Sync("SAM decoder", error) || !iou_d.Download(iou, kMaskTokens, error))
      return false;
    const int first = prompt.multimask ? 1 : 0, last = prompt.multimask ? kMaskTokens : 1;
    for (int i = first; i < last; ++i) {
      const float* low = masks_d.f() + static_cast<std::size_t>(i) * s2 * s2;
      if (!big.Reserve(static_cast<std::size_t>(kSide) * kSide, error) ||
          !crop.Reserve(static_cast<std::size_t>(nh) * nw, error) ||
          !full.Reserve(static_cast<std::size_t>(width) * height, error))
        return false;
      ResizeBilinear(low, 1, s2, s2, big.f(), kSide, kSide, false);
      CropKernel<<<Blocks(static_cast<std::size_t>(nh) * nw), kThreads>>>(big.f(), kSide, nh, nw,
                                                                        crop.f());
      ResizeBilinear(crop.f(), 1, nh, nw, full.f(), height, width, false);
      std::vector<float> values(static_cast<std::size_t>(width) * height), lowres(s2 * s2);
      if (!Sync("SAM masks", error) || !full.Download(values.data(), values.size(), error))
        return false;
      hipMemcpy(lowres.data(), low, lowres.size() * 4, hipMemcpyDeviceToHost);
      std::vector<std::uint8_t> mask(values.size());
      for (std::size_t k = 0; k < values.size(); ++k)
        mask[k] = values[k] > 0.0F ? 1 : 0;
      out->masks.push_back(std::move(mask));
      out->scores.push_back(iou[i]);
      out->low_res.push_back(std::move(lowres));
    }
  }
  return true;
}

}  // namespace gufo::models::vision_dsg
