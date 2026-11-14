// ViTPose base with the simple decoder (transformers VitPoseForPoseEstimation +
// VitPoseImageProcessor), CPU fp32. Written with AI assistance (Claude Code).
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <gufo/ext/json.hpp>

#include "ccip/cpu/nn.hpp"
#include "ccip/pose.hpp"

namespace gufo::models::ccip {

using nn::Act;
using nn::Map;

namespace {

constexpr int kInW = 192;
constexpr int kInH = 256;

// box_to_center_and_scale (float32 like numpy), box given as x0, y0, x1, y1.
void CenterScale(const std::array<float, 4>& box, float normalize, float c[2], float s[2]) {
  const double x = box[0], y = box[1];
  double w = static_cast<double>(box[2]) - box[0];
  double h = static_cast<double>(box[3]) - box[1];
  const double ar = static_cast<double>(kInW) / kInH;
  c[0] = static_cast<float>(x + w * 0.5);
  c[1] = static_cast<float>(y + h * 0.5);
  if (w > ar * h) {
    h = w * 1.0 / ar;
  } else if (w < ar * h) {
    w = h * ar;
  }
  s[0] = static_cast<float>(w / normalize) * 1.25f;
  s[1] = static_cast<float>(h / normalize) * 1.25f;
}

// 1-D gaussian of scipy.ndimage.gaussian_filter(sigma, radius), mode "reflect".
void GaussianReflect(const float* in, float* out, int h, int w, double sigma, int radius) {
  std::vector<double> k(static_cast<std::size_t>(2 * radius + 1));
  double sum = 0.0;
  for (int i = -radius; i <= radius; ++i) {
    k[static_cast<std::size_t>(i + radius)] = std::exp(-0.5 * i * i / (sigma * sigma));
    sum += k[static_cast<std::size_t>(i + radius)];
  }
  for (double& v : k) v /= sum;
  auto reflect = [](int i, int n) {
    while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1;
    return i;
  };
  std::vector<double> tmp(static_cast<std::size_t>(h) * w);
  for (int y = 0; y < h; ++y) {  // axis 0 (rows) first, like axes=(0, 1)
    for (int x = 0; x < w; ++x) {
      double s = 0.0;
      for (int j = -radius; j <= radius; ++j) s += k[static_cast<std::size_t>(j + radius)] * in[reflect(y + j, h) * w + x];
      tmp[static_cast<std::size_t>(y) * w + x] = static_cast<float>(s);
    }
  }
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      double s = 0.0;
      for (int j = -radius; j <= radius; ++j) s += k[static_cast<std::size_t>(j + radius)] * tmp[static_cast<std::size_t>(y) * w + reflect(x + j, w)];
      out[y * w + x] = static_cast<float>(s);
    }
  }
}

}  // namespace

struct VitPose::Impl {
  nn::Weights w;
  std::vector<std::string> names;
  int layers{12};
  int hidden{768};
  int heads{12};
  float eps{1e-12f};
  float mean[3]{0.485f, 0.456f, 0.406f};
  float stdv[3]{0.229f, 0.224f, 0.225f};
  float normalize{200.0f};
};

VitPose::VitPose() : impl_(std::make_unique<Impl>()) {}
VitPose::~VitPose() = default;

std::unique_ptr<VitPose> VitPose::Load(const std::filesystem::path& dir, std::string* error) {
  std::unique_ptr<VitPose> m(new VitPose());
  Impl& p = *m->impl_;
  if (!p.w.Load(dir / "vitpose_base_simple.safetensors", error)) return nullptr;
  std::ifstream in(dir / "vitpose_base_simple.json", std::ios::binary);
  if (!in) {
    if (error) *error = "missing " + (dir / "vitpose_base_simple.json").string();
    return nullptr;
  }
  std::stringstream ss;
  ss << in.rdbuf();
  try {
    const json::Value a = json::parse(ss.str());
    p.layers = static_cast<int>(a.member_size("layers", 12));
    p.hidden = static_cast<int>(a.member_size("hidden", 768));
    p.heads = static_cast<int>(a.member_size("heads", 12));
    p.eps = static_cast<float>(a.member_double("layer_norm_eps", 1e-12));
    for (const auto& n : a.find("keypoints")->items()) p.names.push_back(n.str());
    const json::Value* input = a.find("input");
    for (int c = 0; c < 3; ++c) {
      p.mean[c] = static_cast<float>(input->find("mean")->items()[static_cast<std::size_t>(c)].as_double());
      p.stdv[c] = static_cast<float>(input->find("std")->items()[static_cast<std::size_t>(c)].as_double());
    }
    p.normalize = static_cast<float>(input->member_double("normalize_factor", 200.0));
    if (input->member_size("height") != kInH || input->member_size("width") != kInW)
      throw std::runtime_error("input size must be 256 x 192");
  } catch (const std::exception& e) {
    if (error) *error = std::string("vitpose: ") + e.what();
    return nullptr;
  }
  return m;
}

const std::vector<std::string>& VitPose::KeypointNames() const { return impl_->names; }

std::vector<Keypoint> VitPose::Estimate(const RgbImage& image, const std::array<float, 4>& box,
                                        VitPoseTaps* taps) const {
  const Impl& p = *impl_;
  const nn::Weights& w = p.w;
  float c[2], s[2];
  CenterScale(box, p.normalize, c, s);

  // get_warp_matrix(0, center * 2, (191, 255), scale * 200) as float32, inverted in float64;
  // scipy affine_transform(order=1): bilinear, 0 outside [0, n-1], round half up to uint8.
  const double target[2] = {static_cast<double>(s[0] * 200.0f), static_cast<double>(s[1] * 200.0f)};
  const double sx = (kInW - 1.0) / target[0];
  const double sy = (kInH - 1.0) / target[1];
  const auto m00 = static_cast<float>(sx);
  // numpy 2 (NEP 50): -0.5 * float32 + 0.5 * float32 stays float32, then times the float64 scale.
  const float inner_x = -0.5f * (c[0] * 2.0f) + 0.5f * (s[0] * 200.0f);
  const float inner_y = -0.5f * (c[1] * 2.0f) + 0.5f * (s[1] * 200.0f);
  const auto m02 = static_cast<float>(sx * static_cast<double>(inner_x));
  const auto m11 = static_cast<float>(sy);
  const auto m12 = static_cast<float>(sy * static_cast<double>(inner_y));
  // np.linalg.inv (LAPACK getri on the upper-triangular matrix): the translation is
  // -(t * (1 / s)), not -t / s; the last bit decides x.5 roundings of the warp.
  const double i00 = 1.0 / m00, i02 = -(static_cast<double>(m02) * i00);
  const double i11 = 1.0 / m11, i12 = -(static_cast<double>(m12) * i11);
  Map x{3, kInH, kInW, std::vector<float>(static_cast<std::size_t>(3) * kInH * kInW)};
  const std::size_t plane = static_cast<std::size_t>(kInH) * kInW;
  const int H = image.height, W = image.width;
  for (int r = 0; r < kInH; ++r) {
    const double yi = i11 * r + i12;
    for (int col = 0; col < kInW; ++col) {
      const double xi = i00 * col + i02;
      double v[3] = {0, 0, 0};
      if (!(yi < 0 || xi < 0 || yi > H - 1 || xi > W - 1)) {
        const int y0 = static_cast<int>(std::floor(yi));
        const int x0 = static_cast<int>(std::floor(xi));
        const double fy = yi - y0, fx = xi - x0;
        const int y1 = std::min(y0 + 1, H - 1), x1 = std::min(x0 + 1, W - 1);
        for (int ch = 0; ch < 3; ++ch) {
          auto px = [&](int yy, int xx) {
            return static_cast<double>(image.rgb[(static_cast<std::size_t>(yy) * W + xx) * 3 + ch]);
          };
          const double val = px(y0, x0) * (1 - fy) * (1 - fx) + px(y0, x1) * (1 - fy) * fx +
                             px(y1, x0) * fy * (1 - fx) + px(y1, x1) * fy * fx;
          v[ch] = std::clamp(std::floor(val + 0.5), 0.0, 255.0);
        }
      }
      for (int ch = 0; ch < 3; ++ch) {
        // rescale_and_normalize fused: (x - mean / (1/255)) / (std / (1/255)).
        const auto mean = static_cast<float>(p.mean[ch] / (1.0 / 255.0));
        const auto sd = static_cast<float>(p.stdv[ch] / (1.0 / 255.0));
        x.d[ch * plane + static_cast<std::size_t>(r) * kInW + col] = (static_cast<float>(v[ch]) - mean) / sd;
      }
    }
  }
  if (taps != nullptr) taps->input = x.d;

  // Backbone.
  const int D = p.hidden;
  Map pe = nn::Conv(w, "backbone.embeddings.patch_embeddings.projection", x, 16, 2, Act::kNone);
  const int gh = pe.h, gw = pe.w, rows = gh * gw;
  std::vector<float> t = nn::ToTokens(pe);
  const auto& pos = w.Raw("backbone.embeddings.position_embeddings");
  for (int i = 0; i < rows; ++i) {
    for (int e = 0; e < D; ++e) {
      t[static_cast<std::size_t>(i) * D + e] += pos[static_cast<std::size_t>(i + 1) * D + e] + pos[static_cast<std::size_t>(e)];
    }
  }
  for (int l = 0; l < p.layers; ++l) {
    const std::string n = "backbone.encoder.layer." + std::to_string(l) + ".";
    std::vector<float> a(t);
    nn::LayerNorm(w, n + "layernorm_before", a.data(), rows, D, p.eps);
    const auto q = nn::Linear(w, n + "attention.attention.query", a.data(), rows);
    const auto k = nn::Linear(w, n + "attention.attention.key", a.data(), rows);
    const auto v = nn::Linear(w, n + "attention.attention.value", a.data(), rows);
    const auto o = nn::Attention(q.data(), k.data(), v.data(), rows, rows, D, p.heads);
    const auto ao = nn::Linear(w, n + "attention.output.dense", o.data(), rows);
    for (std::size_t i = 0; i < t.size(); ++i) t[i] += ao[i];
    a = t;
    nn::LayerNorm(w, n + "layernorm_after", a.data(), rows, D, p.eps);
    const auto f1 = nn::Linear(w, n + "mlp.fc1", a.data(), rows, Act::kGelu);
    const auto f2 = nn::Linear(w, n + "mlp.fc2", f1.data(), rows);
    for (std::size_t i = 0; i < t.size(); ++i) t[i] += f2[i];
  }
  nn::LayerNorm(w, "backbone.layernorm", t.data(), rows, D, p.eps);
  Map f = nn::FromTokens(t, D, gh, gw);
  nn::Activate(f.d.data(), f.d.size(), Act::kRelu);
  f = nn::UpsampleBilinear(f, 4);
  const Map hm = nn::Conv(w, "head.conv", f, 1, 1, Act::kNone);
  if (taps != nullptr) taps->heatmaps = hm.d;

  // post_process_pose_estimation: arg-max, DARK (kernel 11 -> radius 5, sigma 0.8), transform_preds.
  const int K = hm.c, HH = hm.h, WW = hm.w;
  std::vector<Keypoint> out(static_cast<std::size_t>(K));
  std::vector<float> blur(static_cast<std::size_t>(HH) * WW);
  for (int k = 0; k < K; ++k) {
    const float* m = hm.d.data() + static_cast<std::size_t>(k) * HH * WW;
    int arg = 0;
    float mx = m[0];
    for (int i = 1; i < HH * WW; ++i) {
      if (m[i] > mx) {  // np.argmax: first maximum
        mx = m[i];
        arg = i;
      }
    }
    double px = arg % WW, py = arg / WW;
    if (!(mx > 0.0f)) px = py = -1;
    GaussianReflect(m, blur.data(), HH, WW, 0.8, 5);
    auto L = [&](int yy, int xx) {  // log(clip(blur, 0.001, 50)) with edge padding
      yy = std::clamp(yy, 0, HH - 1);
      xx = std::clamp(xx, 0, WW - 1);
      return std::log(std::clamp(blur[static_cast<std::size_t>(yy) * WW + xx], 0.001f, 50.0f));
    };
    const int ix = static_cast<int>(px), iy = static_cast<int>(py);
    const double i_ = L(iy, ix), ix1 = L(iy, ix + 1), ix1_ = L(iy, ix - 1);
    const double iy1 = L(iy + 1, ix), iy1_ = L(iy - 1, ix);
    const double ix1y1 = L(iy + 1, ix + 1), ix1_y1_ = L(iy - 1, ix - 1);
    const double dx = 0.5 * (ix1 - ix1_), dy = 0.5 * (iy1 - iy1_);
    const double dxx = ix1 - 2 * i_ + ix1_, dyy = iy1 - 2 * i_ + iy1_;
    const double dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_);
    const double e = std::numeric_limits<float>::epsilon();
    const double a00 = dxx + e, a01 = dxy, a10 = dxy, a11 = dyy + e;
    const double det = a00 * a11 - a01 * a10;
    if (det != 0.0) {
      px -= (a11 * dx - a01 * dy) / det;
      py -= (-a10 * dx + a00 * dy) / det;
    }
    // transform_preds in float32 like numpy (coords float32, scale float32).
    const float s0 = s[0] * 200.0f, s1 = s[1] * 200.0f;
    const float scale_x = s0 / (static_cast<float>(WW) - 1.0f);
    const float scale_y = s1 / (static_cast<float>(HH) - 1.0f);
    const auto fx = static_cast<float>(px), fy = static_cast<float>(py);
    Keypoint kp;
    kp.x = fx * scale_x + c[0] - s0 * 0.5f;
    kp.y = fy * scale_y + c[1] - s1 * 0.5f;
    kp.score = mx;
    out[static_cast<std::size_t>(k)] = kp;
  }
  return out;
}

}  // namespace gufo::models::ccip
