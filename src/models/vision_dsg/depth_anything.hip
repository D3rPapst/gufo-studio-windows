// Depth Anything V2 on the GPU (see depth_anything.hpp).
// Written with AI assistance (Claude Code).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <utility>

#include "src/core/json.hpp"
#include "src/models/vision_dsg/depth_anything.hpp"
#include "src/models/vision_dsg/hip_ops.hpp"

namespace gufo::models::vision_dsg {
namespace {

constexpr int kPatch = 14;
constexpr float kEps = 1e-6F;

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

int IntMember(const json::Value& object, const char* name, int fallback) {
  const json::Value* v = object.find(name);
  return v != nullptr && v->is_number() ? static_cast<int>(v->as_double())
                                        : fallback;
}

}  // namespace

struct DepthAnything::Impl {
  Gpu gpu;
  WeightStore w;
  int dim{0}, heads{0}, layers{0}, mlp{0}, fusion{0}, head_hidden{0};
  int grid{37};  // position grid side (image_size / patch)
  std::vector<int> out_indices;  // 1-based layer numbers
  std::vector<int> neck_sizes;
  std::vector<double> factors;
  float max_depth{1.0F};
  std::vector<float> pos_class, pos_grid;  // host copies
  std::vector<float> cls;

  const float* W(const std::string& name) const { return w.Get(name); }
  bool Residual(const std::string& p, float* x, int h, int ww, DevBuf* t,
                DevBuf* u, std::string* error);
};

DepthAnything::DepthAnything(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DepthAnything::~DepthAnything() = default;

std::unique_ptr<DepthAnything> DepthAnything::Load(
    const std::filesystem::path& dir, std::string* error) {
  std::ifstream config_file(dir / "config.json", std::ios::binary);
  if (!config_file) {
    SetError(error, "Depth Anything: no config.json in " + dir.string());
    return nullptr;
  }
  std::stringstream text;
  text << config_file.rdbuf();
  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  try {
    const json::Value config = json::parse(text.str());
    if (config.member_str("model_type") != "depth_anything") {
      SetError(error, dir.string() + " is not a Depth Anything checkpoint");
      return nullptr;
    }
    const json::Value* backbone = config.find("backbone_config");
    if (backbone == nullptr || !backbone->is_object()) {
      SetError(error, "Depth Anything: config.json lacks backbone_config");
      return nullptr;
    }
    m.dim = IntMember(*backbone, "hidden_size", 768);
    m.heads = IntMember(*backbone, "num_attention_heads", 12);
    m.layers = IntMember(*backbone, "num_hidden_layers", 12);
    m.mlp = m.dim * IntMember(*backbone, "mlp_ratio", 4);
    m.grid = IntMember(*backbone, "image_size", 518) / kPatch;
    if (const json::Value* idx = backbone->find("out_indices"))
      for (const auto& v : idx->items())
        m.out_indices.push_back(static_cast<int>(v.as_double()));
    if (const json::Value* sizes = config.find("neck_hidden_sizes"))
      for (const auto& v : sizes->items())
        m.neck_sizes.push_back(static_cast<int>(v.as_double()));
    if (const json::Value* f = config.find("reassemble_factors"))
      for (const auto& v : f->items())
        m.factors.push_back(v.as_double());
    m.fusion = IntMember(config, "fusion_hidden_size", 256);
    m.head_hidden = IntMember(config, "head_hidden_size", 32);
    if (const json::Value* md = config.find("max_depth");
        md != nullptr && md->is_number())
      m.max_depth = static_cast<float>(md->as_double());
    const std::string type = config.member_str("depth_estimation_type");
    if (!type.empty() && type != "relative") {
      SetError(error, "Depth Anything: only relative depth checkpoints");
      return nullptr;
    }
    if (IntMember(config, "head_in_index", -1) != -1 ||
        IntMember(*backbone, "patch_size", kPatch) != kPatch) {
      SetError(error, "Depth Anything: unsupported head_in_index/patch_size");
      return nullptr;
    }
  } catch (const std::exception& e) {
    SetError(error, "Depth Anything config.json: " + std::string(e.what()));
    return nullptr;
  }
  if (m.out_indices.size() != 4 || m.neck_sizes.size() != 4 ||
      m.factors.size() != 4 || m.dim % m.heads != 0) {
    SetError(error, "Depth Anything: unexpected config (four stages)");
    return nullptr;
  }
  if (!m.gpu.Init(error) || !m.w.Load(dir / "model.safetensors", error))
    return nullptr;
  const auto pos = m.w.Host("backbone.embeddings.position_embeddings");
  m.cls = m.w.Host("backbone.embeddings.cls_token");
  if (pos.size() !=
          static_cast<std::size_t>(1 + m.grid * m.grid) * m.dim ||
      m.cls.size() != static_cast<std::size_t>(m.dim) ||
      m.W("head.conv3.weight") == nullptr) {
    SetError(error, "Depth Anything: weights do not match config.json");
    return nullptr;
  }
  m.pos_class.assign(pos.begin(), pos.begin() + m.dim);
  // [grid*grid, dim] -> NCHW [dim, grid, grid] for the bicubic resize.
  const int g2 = m.grid * m.grid;
  m.pos_grid.resize(static_cast<std::size_t>(g2) * m.dim);
  for (int p = 0; p < g2; ++p)
    for (int c = 0; c < m.dim; ++c)
      m.pos_grid[static_cast<std::size_t>(c) * g2 + p] =
          pos[static_cast<std::size_t>(1 + p) * m.dim + c];
  return std::unique_ptr<DepthAnything>(new DepthAnything(std::move(impl)));
}

std::string DepthAnything::size_name() const {
  if (impl_->dim <= 384)
    return "small";
  if (impl_->dim <= 768)
    return "base";
  return "large";
}

std::size_t DepthAnything::weight_bytes() const { return impl_->w.bytes(); }

// DepthAnythingPreActResidualLayer in place: x += conv2(relu(conv1(relu(x)))).
bool DepthAnything::Impl::Residual(const std::string& p, float* x, int h,
                                   int ww, DevBuf* t, DevBuf* u,
                                   std::string* error) {
  const std::size_t n = static_cast<std::size_t>(fusion) * h * ww;
  if (!t->Reserve(n, error) || !u->Reserve(n, error))
    return false;
  hipMemcpy(t->f(), x, n * 4, hipMemcpyDeviceToDevice);
  Activate(t->f(), n, Act::kRelu);
  if (!gpu.Conv2d(t->f(), fusion, h, ww, W(p + "convolution1.weight"),
                  W(p + "convolution1.bias"), fusion, 3, 1, 1, u->f(), nullptr,
                  nullptr, error))
    return false;
  Activate(u->f(), n, Act::kRelu);
  if (!gpu.Conv2d(u->f(), fusion, h, ww, W(p + "convolution2.weight"),
                  W(p + "convolution2.bias"), fusion, 3, 1, 1, t->f(), nullptr,
                  nullptr, error))
    return false;
  Add(x, t->f(), n);
  return true;
}

bool DepthAnything::Run(const std::uint8_t* rgb, int width, int height,
                        DepthResult* out, std::string* error) {
  int nw = 0, nh = 0;
  const std::vector<float> pixels = DepthPixels(rgb, width, height, &nw, &nh);
  if (pixels.empty()) {
    SetError(error, "Depth Anything: invalid image");
    return false;
  }
  return RunPixels(pixels.data(), nh, nw, width, height, out, error);
}

bool DepthAnything::RunPixels(const float* pixels, int net_height,
                              int net_width, int width, int height,
                              DepthResult* out, std::string* error) {
  Impl& m = *impl_;
  if (net_height % kPatch != 0 || net_width % kPatch != 0 || net_height < kPatch ||
      net_width < kPatch) {
    SetError(error, "Depth Anything: input size must be a multiple of 14");
    return false;
  }
  const int ph = net_height / kPatch, pw = net_width / kPatch;
  const int patches = ph * pw, T = patches + 1, D = m.dim;
  const int pv = 3 * kPatch * kPatch;
  // Patch columns [patches, 3 * 14 * 14] (ci, ky, kx order of the conv
  // weight).
  std::vector<float> columns(static_cast<std::size_t>(patches) * pv);
  for (int py = 0; py < ph; ++py)
    for (int px = 0; px < pw; ++px)
      for (int c = 0; c < 3; ++c)
        for (int y = 0; y < kPatch; ++y)
          for (int x = 0; x < kPatch; ++x)
            columns[(static_cast<std::size_t>(py) * pw + px) * pv +
                    (c * kPatch + y) * kPatch + x] =
                pixels[(static_cast<std::size_t>(c) * net_height + py * kPatch +
                        y) * net_width +
                       px * kPatch + x];
  // Position embeddings: as stored for the 37 x 37 square, else bicubic.
  std::vector<float> positions(static_cast<std::size_t>(T) * D);
  for (int c = 0; c < D; ++c)
    positions[c] = m.pos_class[c] + m.cls[c];
  if (ph == m.grid && pw == m.grid) {
    for (int p = 0; p < patches; ++p)
      for (int c = 0; c < D; ++c)
        positions[static_cast<std::size_t>(1 + p) * D + c] =
            m.pos_grid[static_cast<std::size_t>(c) * patches + p];
  } else {
    const auto grid =
        HostResizeBicubic(m.pos_grid, D, m.grid, m.grid, ph, pw);
    for (int p = 0; p < patches; ++p)
      for (int c = 0; c < D; ++c)
        positions[static_cast<std::size_t>(1 + p) * D + c] =
            grid[static_cast<std::size_t>(c) * patches + p];
  }
  const std::size_t TD = static_cast<std::size_t>(T) * D;
  DevBuf cols, hidden, pos, normed, qkv, attn, lin, mlp;
  if (!cols.Upload(columns.data(), columns.size(), error) ||
      !pos.Upload(positions.data(), positions.size(), error) ||
      !hidden.Reserve(TD, error) || !normed.Reserve(TD, error) ||
      !qkv.Reserve(TD * 3, error) || !attn.Reserve(TD, error) ||
      !lin.Reserve(TD, error) ||
      !mlp.Reserve(static_cast<std::size_t>(T) * m.mlp, error))
    return false;
  hipMemset(hidden.f(), 0, static_cast<std::size_t>(D) * 4);
  if (!m.gpu.Linear(cols.f(), patches,
                    m.W("backbone.embeddings.patch_embeddings.projection.weight"),
                    m.W("backbone.embeddings.patch_embeddings.projection.bias"),
                    D, pv, hidden.f() + D, Act::kNone, error))
    return false;
  Add(hidden.f(), pos.f(), TD);
  // Encoder; the four out_indices states, layer-normalised, as NCHW maps.
  std::vector<DevBuf> maps(4);
  const int hd = D / m.heads;
  int taken = 0;
  for (int layer = 0; layer < m.layers && taken < 4; ++layer) {
    const std::string p =
        "backbone.encoder.layer." + std::to_string(layer) + ".";
    LayerNorm(hidden.f(), m.W(p + "norm1.weight"), m.W(p + "norm1.bias"),
              normed.f(), T, D, kEps);
    const std::string a = p + "attention.attention.";
    if (!m.gpu.Linear(normed.f(), T, m.W(a + "query.weight"),
                      m.W(a + "query.bias"), D, D, qkv.f(), Act::kNone, error,
                      0, 3 * D) ||
        !m.gpu.Linear(normed.f(), T, m.W(a + "key.weight"),
                      m.W(a + "key.bias"), D, D, qkv.f() + D, Act::kNone,
                      error, 0, 3 * D) ||
        !m.gpu.Linear(normed.f(), T, m.W(a + "value.weight"),
                      m.W(a + "value.bias"), D, D, qkv.f() + 2 * D, Act::kNone,
                      error, 0, 3 * D) ||
        !m.gpu.Attention(qkv.f(), 3 * D, qkv.f() + D, 3 * D, qkv.f() + 2 * D,
                         3 * D, T, T, m.heads, hd,
                         1.0F / std::sqrt(static_cast<float>(hd)), attn.f(), D,
                         error) ||
        !m.gpu.Linear(attn.f(), T, m.W(p + "attention.output.dense.weight"),
                      m.W(p + "attention.output.dense.bias"), D, D, lin.f(),
                      Act::kNone, error))
      return false;
    ScaledAdd(hidden.f(), lin.f(), m.W(p + "layer_scale1.lambda1"), T, D);
    LayerNorm(hidden.f(), m.W(p + "norm2.weight"), m.W(p + "norm2.bias"),
              normed.f(), T, D, kEps);
    if (!m.gpu.Linear(normed.f(), T, m.W(p + "mlp.fc1.weight"),
                      m.W(p + "mlp.fc1.bias"), m.mlp, D, mlp.f(), Act::kGelu,
                      error) ||
        !m.gpu.Linear(mlp.f(), T, m.W(p + "mlp.fc2.weight"),
                      m.W(p + "mlp.fc2.bias"), D, m.mlp, lin.f(), Act::kNone,
                      error))
      return false;
    ScaledAdd(hidden.f(), lin.f(), m.W(p + "layer_scale2.lambda1"), T, D);
    if (layer + 1 == m.out_indices[taken]) {
      LayerNorm(hidden.f(), m.W("backbone.layernorm.weight"),
                m.W("backbone.layernorm.bias"), normed.f(), T, D, kEps);
      if (!maps[taken].Reserve(static_cast<std::size_t>(patches) * D, error))
        return false;
      TokensToMap(normed.f() + D, patches, D, D, maps[taken].f());
      ++taken;
    }
  }
  if (taken != 4 || !Sync("Depth Anything encoder", error))
    return false;
  // Neck: reassemble + 3x3 convs to `fusion` channels.
  std::vector<DevBuf> feats(4);
  std::vector<int> fh(4), fw(4);
  DevBuf proj, resized;
  for (int i = 0; i < 4; ++i) {
    const std::string p = "neck.reassemble_stage.layers." + std::to_string(i) + ".";
    const int c = m.neck_sizes[i];
    if (!proj.Reserve(static_cast<std::size_t>(c) * patches, error) ||
        !m.gpu.Conv2d(maps[i].f(), D, ph, pw, m.W(p + "projection.weight"),
                      m.W(p + "projection.bias"), c, 1, 1, 0, proj.f(), nullptr,
                      nullptr, error))
      return false;
    const double f = m.factors[i];
    int h = ph, ww = pw;
    const float* src = proj.f();
    if (f > 1) {
      const int k = static_cast<int>(std::lround(f));
      h = ph * k;
      ww = pw * k;
      if (!resized.Reserve(static_cast<std::size_t>(c) * h * ww, error) ||
          !m.gpu.ConvTransposeKS(proj.f(), c, ph, pw, m.W(p + "resize.weight"),
                                 m.W(p + "resize.bias"), c, k, resized.f(),
                                 error))
        return false;
      src = resized.f();
    } else if (f < 1) {
      const int s = static_cast<int>(std::lround(1.0 / f));
      const int oh = (ph + 2 - 3) / s + 1, ow = (pw + 2 - 3) / s + 1;
      if (!resized.Reserve(static_cast<std::size_t>(c) * oh * ow, error) ||
          !m.gpu.Conv2d(proj.f(), c, ph, pw, m.W(p + "resize.weight"),
                        m.W(p + "resize.bias"), c, 3, s, 1, resized.f(), &h,
                        &ww, error))
        return false;
      src = resized.f();
    }
    fh[i] = h;
    fw[i] = ww;
    if (!feats[i].Reserve(static_cast<std::size_t>(m.fusion) * h * ww, error) ||
        !m.gpu.Conv2d(src, c, h, ww, m.W("neck.convs." + std::to_string(i) + ".weight"),
                      nullptr, m.fusion, 3, 1, 1, feats[i].f(), nullptr, nullptr,
                      error))
      return false;
  }
  // Fusion, deepest first.
  DevBuf fused, t, u, up;
  int cur_h = 0, cur_w = 0;
  for (int idx = 0; idx < 4; ++idx) {
    const int level = 3 - idx;
    const std::string p = "neck.fusion_stage.layers." + std::to_string(idx) + ".";
    const std::size_t n = static_cast<std::size_t>(m.fusion) * fh[level] * fw[level];
    if (idx == 0) {
      if (!fused.Reserve(n, error))
        return false;
      hipMemcpy(fused.f(), feats[level].f(), n * 4, hipMemcpyDeviceToDevice);
    } else {
      if (cur_h != fh[level] || cur_w != fw[level]) {
        SetError(error, "Depth Anything: fusion size mismatch");
        return false;
      }
      if (!m.Residual(p + "residual_layer1.", feats[level].f(), fh[level],
                      fw[level], &t, &u, error))
        return false;
      Add(fused.f(), feats[level].f(), n);
    }
    cur_h = fh[level];
    cur_w = fw[level];
    if (!m.Residual(p + "residual_layer2.", fused.f(), cur_h, cur_w, &t, &u,
                    error))
      return false;
    const int oh = idx < 3 ? fh[level - 1] : cur_h * 2;
    const int ow = idx < 3 ? fw[level - 1] : cur_w * 2;
    const std::size_t on = static_cast<std::size_t>(m.fusion) * oh * ow;
    if (!up.Reserve(on, error))
      return false;
    ResizeBilinear(fused.f(), m.fusion, cur_h, cur_w, up.f(), oh, ow, true);
    if (!fused.Reserve(on, error) ||
        !m.gpu.Conv2d(up.f(), m.fusion, oh, ow, m.W(p + "projection.weight"),
                      m.W(p + "projection.bias"), m.fusion, 1, 1, 0, fused.f(),
                      nullptr, nullptr, error))
      return false;
    cur_h = oh;
    cur_w = ow;
  }
  // Head.
  const int half = m.fusion / 2;
  DevBuf h1, h2, h3;
  if (!h1.Reserve(static_cast<std::size_t>(half) * cur_h * cur_w, error) ||
      !m.gpu.Conv2d(fused.f(), m.fusion, cur_h, cur_w, m.W("head.conv1.weight"),
                    m.W("head.conv1.bias"), half, 3, 1, 1, h1.f(), nullptr,
                    nullptr, error))
    return false;
  const std::size_t net_pixels = static_cast<std::size_t>(net_height) * net_width;
  if (!h2.Reserve(static_cast<std::size_t>(half) * net_pixels, error))
    return false;
  ResizeBilinear(h1.f(), half, cur_h, cur_w, h2.f(), net_height, net_width, true);
  if (!h3.Reserve(static_cast<std::size_t>(m.head_hidden) * net_pixels, error) ||
      !m.gpu.Conv2d(h2.f(), half, net_height, net_width, m.W("head.conv2.weight"),
                    m.W("head.conv2.bias"), m.head_hidden, 3, 1, 1, h3.f(),
                    nullptr, nullptr, error))
    return false;
  Activate(h3.f(), static_cast<std::size_t>(m.head_hidden) * net_pixels, Act::kRelu);
  if (!m.gpu.Conv2d(h3.f(), m.head_hidden, net_height, net_width,
                    m.W("head.conv3.weight"), m.W("head.conv3.bias"), 1, 1, 1, 0,
                    h2.f(), nullptr, nullptr, error))
    return false;
  Activate(h2.f(), net_pixels, Act::kRelu);
  if (m.max_depth != 1.0F)
    Scale(h2.f(), m.max_depth, net_pixels);
  out->net_width = net_width;
  out->net_height = net_height;
  out->raw.resize(net_pixels);
  if (!Sync("Depth Anything head", error) ||
      !h2.Download(out->raw.data(), net_pixels, error))
    return false;
  out->width = width;
  out->height = height;
  if (width > 0 && height > 0) {
    const std::size_t full = static_cast<std::size_t>(width) * height;
    if (!h1.Reserve(full, error))
      return false;
    ResizeBicubic(h2.f(), 1, net_height, net_width, h1.f(), height, width);
    out->depth.resize(full);
    if (!Sync("Depth Anything resize", error) ||
        !h1.Download(out->depth.data(), full, error))
      return false;
  }
  return true;
}

}  // namespace gufo::models::vision_dsg
