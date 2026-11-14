// DWPose dw-ll_ucoco_384 (RTMPose-l whole body, 133 points, SimCC), CPU fp32.
// reference/convert_dwpose.py stores the ONNX node list (a fixed, small op
// set) and the weights; this file runs that list in order. Pre- and
// post-processing follow the film pipeline's posen_erkennung.wholebody.
// Written with AI assistance (Claude Code).
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <gufo/ext/json.hpp>

#include "ccip/cpu/kernels.hpp"
#include "ccip/cpu/nn.hpp"
#include "ccip/pose.hpp"

namespace gufo::models::ccip {

namespace {

struct Nd {
  std::vector<std::int64_t> shape;
  std::vector<float> d;
  [[nodiscard]] std::int64_t Numel() const {
    return std::accumulate(shape.begin(), shape.end(), std::int64_t{1}, std::multiplies<>());
  }
};

struct Node {
  std::string op, name;
  std::vector<std::string> in, out;
  json::Value attrs;
};

std::vector<std::int64_t> Ints(const json::Value& attrs, const char* key) {
  std::vector<std::int64_t> v;
  if (const json::Value* a = attrs.find(key); a != nullptr && a->is_array()) {
    for (const auto& x : a->items()) v.push_back(static_cast<std::int64_t>(x.as_double()));
  }
  return v;
}

std::vector<std::int64_t> Strides(const std::vector<std::int64_t>& shape) {
  std::vector<std::int64_t> s(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) s[static_cast<std::size_t>(i)] = s[static_cast<std::size_t>(i) + 1] * shape[static_cast<std::size_t>(i) + 1];
  return s;
}

// numpy broadcasting for binary element-wise ops.
Nd Broadcast(const Nd& a, const Nd& b, const std::function<float(float, float)>& f) {
  const std::size_t r = std::max(a.shape.size(), b.shape.size());
  std::vector<std::int64_t> sa(r, 1), sb(r, 1), so(r, 1);
  std::copy(a.shape.begin(), a.shape.end(), sa.begin() + static_cast<std::ptrdiff_t>(r - a.shape.size()));
  std::copy(b.shape.begin(), b.shape.end(), sb.begin() + static_cast<std::ptrdiff_t>(r - b.shape.size()));
  for (std::size_t i = 0; i < r; ++i) {
    if (sa[i] != sb[i] && sa[i] != 1 && sb[i] != 1) throw std::runtime_error("broadcast mismatch");
    so[i] = std::max(sa[i], sb[i]);
  }
  Nd o{so, {}};
  o.d.resize(static_cast<std::size_t>(o.Numel()));
  if (sa == sb) {
    for (std::size_t i = 0; i < o.d.size(); ++i) o.d[i] = f(a.d[i], b.d[i]);
    return o;
  }
  const auto ta = Strides(sa), tb = Strides(sb), tout = Strides(so);
  const auto n = static_cast<std::int64_t>(o.d.size());
#pragma omp parallel for schedule(static)
  for (std::int64_t i = 0; i < n; ++i) {
    std::int64_t ia = 0, ib = 0, rem = i;
    for (std::size_t k = 0; k < r; ++k) {
      const std::int64_t c = rem / tout[k];
      rem %= tout[k];
      if (sa[k] != 1) ia += c * ta[k];
      if (sb[k] != 1) ib += c * tb[k];
    }
    o.d[static_cast<std::size_t>(i)] = f(a.d[static_cast<std::size_t>(ia)], b.d[static_cast<std::size_t>(ib)]);
  }
  return o;
}

nn::Map ToMap(const Nd& x) {
  if (x.shape.size() != 4 || x.shape[0] != 1) throw std::runtime_error("expected 1 x C x H x W");
  return nn::Map{static_cast<int>(x.shape[1]), static_cast<int>(x.shape[2]), static_cast<int>(x.shape[3]), x.d};
}

Nd FromMap(nn::Map m) { return Nd{{1, m.c, m.h, m.w}, std::move(m.d)}; }

}  // namespace

struct DwPose::Impl {
  TensorFile w;
  std::vector<Node> nodes;
  std::map<std::string, Nd> consts;
  std::string input, out_x, out_y;
  double mean[3]{123.675, 116.28, 103.53};
  double stdv[3]{58.395, 57.12, 57.375};
  float split_ratio{2.0f};

  [[nodiscard]] Nd Run(Nd x) const {
    std::map<std::string, Nd> v;
    std::map<std::string, int> uses;
    for (const Node& n : nodes) {
      for (const auto& i : n.in) ++uses[i];
    }
    v[input] = std::move(x);
    auto get = [&](const std::string& name) -> const Nd& {
      if (auto it = v.find(name); it != v.end()) return it->second;
      if (auto it = consts.find(name); it != consts.end()) return it->second;
      throw std::runtime_error("dwpose: no value " + name);
    };
    for (const Node& n : nodes) {
      const std::string& op = n.op;
      std::vector<Nd> outs;
      if (op == "Conv") {
        const Nd& x0 = get(n.in[0]);
        const Nd& wt = get(n.in[1]);
        const Nd* b = n.in.size() > 2 && !n.in[2].empty() ? &get(n.in[2]) : nullptr;
        const auto pads = Ints(n.attrs, "pads");
        const auto strides = Ints(n.attrs, "strides");
        const auto group = static_cast<int>(n.attrs.member_size("group", 1));
        if (wt.shape[2] != wt.shape[3] || pads[0] != pads[2] || strides[0] != strides[1])
          throw std::runtime_error("dwpose: conv shape");
        outs.push_back(FromMap(nn::Conv(ToMap(x0), wt.d.data(), b ? b->d.data() : nullptr,
                                        static_cast<int>(wt.shape[0]), static_cast<int>(wt.shape[2]),
                                        static_cast<int>(strides[0]), static_cast<int>(pads[0]), group,
                                        nn::Act::kNone)));
      } else if (op == "Sigmoid" || op == "Relu" || op == "Sqrt" || op == "HardSigmoid" || op == "Identity") {
        Nd y = get(n.in[0]);
        const float alpha = static_cast<float>(n.attrs.member_double("alpha", 0.2));
        const float beta = static_cast<float>(n.attrs.member_double("beta", 0.5));
        for (float& t : y.d) {
          if (op == "Sigmoid") t = 1.0f / (1.0f + std::exp(-t));
          else if (op == "Relu") t = t > 0.0f ? t : 0.0f;
          else if (op == "Sqrt") t = std::sqrt(t);
          else if (op == "HardSigmoid") t = std::clamp(alpha * t + beta, 0.0f, 1.0f);
        }
        outs.push_back(std::move(y));
      } else if (op == "Mul" || op == "Add" || op == "Div") {
        const Nd& a = get(n.in[0]);
        const Nd& b = get(n.in[1]);
        if (op == "Mul") outs.push_back(Broadcast(a, b, [](float p, float q) { return p * q; }));
        else if (op == "Add") outs.push_back(Broadcast(a, b, [](float p, float q) { return p + q; }));
        else outs.push_back(Broadcast(a, b, [](float p, float q) { return p / q; }));
      } else if (op == "Clip") {
        Nd y = get(n.in[0]);
        const float lo = n.in.size() > 1 && !n.in[1].empty() ? get(n.in[1]).d[0] : -std::numeric_limits<float>::infinity();
        const float hi = n.in.size() > 2 && !n.in[2].empty() ? get(n.in[2]).d[0] : std::numeric_limits<float>::infinity();
        for (float& t : y.d) t = std::min(std::max(t, lo), hi);
        outs.push_back(std::move(y));
      } else if (op == "Concat") {
        const auto axis = static_cast<int>(n.attrs.member_double("axis", 1));
        if (axis != 1) throw std::runtime_error("dwpose: concat axis");
        nn::Map m = ToMap(get(n.in[0]));
        for (std::size_t i = 1; i < n.in.size(); ++i) m = nn::Concat(m, ToMap(get(n.in[i])));
        outs.push_back(FromMap(std::move(m)));
      } else if (op == "GlobalAveragePool") {
        const Nd& x0 = get(n.in[0]);
        const auto c = x0.shape[1];
        const auto hw = x0.shape[2] * x0.shape[3];
        Nd y{{1, c, 1, 1}, std::vector<float>(static_cast<std::size_t>(c))};
        for (std::int64_t i = 0; i < c; ++i) {
          double s = 0.0;
          for (std::int64_t j = 0; j < hw; ++j) s += x0.d[static_cast<std::size_t>(i * hw + j)];
          y.d[static_cast<std::size_t>(i)] = static_cast<float>(s / static_cast<double>(hw));
        }
        outs.push_back(std::move(y));
      } else if (op == "MaxPool") {
        const auto k = Ints(n.attrs, "kernel_shape");
        const auto pads = Ints(n.attrs, "pads");
        const auto strides = Ints(n.attrs, "strides");
        outs.push_back(FromMap(nn::MaxPool(ToMap(get(n.in[0])), static_cast<int>(k[0]), static_cast<int>(strides[0]),
                                           static_cast<int>(pads[0]))));
      } else if (op == "ReduceSum") {
        const Nd& x0 = get(n.in[0]);
        const auto axes = Ints(n.attrs, "axes");
        if (axes.size() != 1 || axes[0] != static_cast<std::int64_t>(x0.shape.size()) - 1)
          throw std::runtime_error("dwpose: reduce over the last axis only");
        const auto last = x0.shape.back();
        const auto rows = x0.Numel() / last;
        Nd y{x0.shape, std::vector<float>(static_cast<std::size_t>(rows))};
        y.shape.back() = 1;
        for (std::int64_t r = 0; r < rows; ++r) {
          float s = 0.0f;
          for (std::int64_t j = 0; j < last; ++j) s += x0.d[static_cast<std::size_t>(r * last + j)];
          y.d[static_cast<std::size_t>(r)] = s;
        }
        outs.push_back(std::move(y));
      } else if (op == "Reshape") {
        Nd y = get(n.in[0]);
        const Nd& s = get(n.in[1]);
        y.shape.clear();
        for (float t : s.d) y.shape.push_back(static_cast<std::int64_t>(t));
        outs.push_back(std::move(y));
      } else if (op == "Unsqueeze" || op == "Squeeze") {
        Nd y = get(n.in[0]);
        const auto axes = Ints(n.attrs, "axes");
        if (op == "Unsqueeze") {
          for (auto a : axes) y.shape.insert(y.shape.begin() + a, 1);
        } else {
          for (auto it = axes.rbegin(); it != axes.rend(); ++it) y.shape.erase(y.shape.begin() + *it);
        }
        outs.push_back(std::move(y));
      } else if (op == "Split") {
        const Nd& x0 = get(n.in[0]);
        const auto axis = static_cast<std::size_t>(n.attrs.member_double("axis", 0));
        const auto parts = Ints(n.attrs, "split");
        const auto st = Strides(x0.shape);
        const std::int64_t outer = x0.Numel() / (x0.shape[axis] * st[axis]);
        std::int64_t off = 0;
        for (auto p : parts) {
          Nd y{x0.shape, {}};
          y.shape[axis] = p;
          y.d.reserve(static_cast<std::size_t>(y.Numel()));
          for (std::int64_t o = 0; o < outer; ++o) {
            const auto begin = x0.d.begin() + static_cast<std::ptrdiff_t>(o * x0.shape[axis] * st[axis] + off * st[axis]);
            y.d.insert(y.d.end(), begin, begin + static_cast<std::ptrdiff_t>(p * st[axis]));
          }
          off += p;
          outs.push_back(std::move(y));
        }
      } else if (op == "Transpose") {
        const Nd& x0 = get(n.in[0]);
        const auto perm = Ints(n.attrs, "perm");
        if (perm != std::vector<std::int64_t>{0, 2, 1} || x0.shape.size() != 3) throw std::runtime_error("dwpose: transpose");
        Nd y{{x0.shape[0], x0.shape[2], x0.shape[1]}, std::vector<float>(x0.d.size())};
        for (std::int64_t b0 = 0; b0 < x0.shape[0]; ++b0) {
          for (std::int64_t i = 0; i < x0.shape[1]; ++i) {
            for (std::int64_t j = 0; j < x0.shape[2]; ++j) {
              y.d[static_cast<std::size_t>((b0 * x0.shape[2] + j) * x0.shape[1] + i)] =
                  x0.d[static_cast<std::size_t>((b0 * x0.shape[1] + i) * x0.shape[2] + j)];
            }
          }
        }
        outs.push_back(std::move(y));
      } else if (op == "MatMul") {
        const Nd& a = get(n.in[0]);
        const Nd& b = get(n.in[1]);
        const auto m = a.shape[a.shape.size() - 2], k = a.shape.back();
        const auto nn_ = b.shape.back();
        if (b.shape[b.shape.size() - 2] != k) throw std::runtime_error("dwpose: matmul shape");
        const auto batch = a.Numel() / (m * k);
        const auto bbatch = b.Numel() / (k * nn_);
        Nd y{a.shape, std::vector<float>(static_cast<std::size_t>(batch * m * nn_))};
        y.shape.back() = nn_;
        for (std::int64_t i = 0; i < batch; ++i) {
          const float* bp = b.d.data() + (bbatch == 1 ? 0 : i * k * nn_);
          cpu::Gemm(a.d.data() + i * m * k, bp, nullptr, y.d.data() + i * m * nn_, static_cast<int>(m),
                    static_cast<int>(nn_), static_cast<int>(k));
        }
        outs.push_back(std::move(y));
      } else {
        throw std::runtime_error("dwpose: op " + op);
      }
      if (outs.size() != n.out.size()) throw std::runtime_error("dwpose: outputs of " + n.name);
      for (std::size_t i = 0; i < outs.size(); ++i) v[n.out[i]] = std::move(outs[i]);
      for (const auto& i : n.in) {
        if (--uses[i] == 0 && i != out_x && i != out_y) v.erase(i);
      }
    }
    Nd both = get(out_x);
    const Nd& y = get(out_y);
    both.shape.push_back(-1);  // marker: x followed by y
    both.d.insert(both.d.end(), y.d.begin(), y.d.end());
    both.shape.push_back(y.shape.back());
    return both;
  }
};

DwPose::DwPose() : impl_(std::make_unique<Impl>()) {}
DwPose::~DwPose() = default;

std::unique_ptr<DwPose> DwPose::Load(const std::filesystem::path& dir, std::string* error) {
  std::unique_ptr<DwPose> m(new DwPose());
  Impl& p = *m->impl_;
  if (!p.w.Load(dir / "dwpose_wholebody.safetensors", error)) return nullptr;
  std::ifstream in(dir / "dwpose_wholebody.json", std::ios::binary);
  if (!in) {
    if (error) *error = "missing " + (dir / "dwpose_wholebody.json").string();
    return nullptr;
  }
  std::stringstream ss;
  ss << in.rdbuf();
  try {
    const json::Value a = json::parse(ss.str());
    const json::Value* input = a.find("input");
    p.input = input->member_str("name");
    for (int c = 0; c < 3; ++c) {
      p.mean[c] = input->find("mean")->items()[static_cast<std::size_t>(c)].as_double();
      p.stdv[c] = input->find("std")->items()[static_cast<std::size_t>(c)].as_double();
    }
    p.out_x = a.find("outputs")->items()[0].str();
    p.out_y = a.find("outputs")->items()[1].str();
    p.split_ratio = static_cast<float>(a.member_double("split_ratio", 2.0));
    for (const auto& n : a.find("nodes")->items()) {
      Node node;
      node.op = n.member_str("op");
      node.name = n.member_str("name");
      for (const auto& x : n.find("in")->items()) node.in.push_back(x.str());
      for (const auto& x : n.find("out")->items()) node.out.push_back(x.str());
      node.attrs = *n.find("attrs");
      p.nodes.push_back(std::move(node));
    }
    for (const auto& [name, t] : p.w.all()) {
      Nd c;
      c.shape = t.shape;
      c.d = t.f32;
      p.consts.emplace(name, std::move(c));
    }
    p.w = TensorFile{};  // the copies in consts are the only ones kept
  } catch (const std::exception& e) {
    if (error) *error = std::string("dwpose: ") + e.what();
    return nullptr;
  }
  return m;
}

std::vector<Keypoint> DwPose::Estimate(const RgbImage& image, const std::array<float, 4>& box,
                                       DwPoseTaps* taps) const {
  const Impl& p = *impl_;
  constexpr int kW = 288, kH = 384;
  // posen_erkennung.wholebody: centre, scale = box * 1.25 brought to 288:384 (float32/float64 as numpy).
  const float cx = (box[0] + box[2]) / 2.0f;
  const float cy = (box[1] + box[3]) / 2.0f;
  const float s0f = (box[2] - box[0]) * 1.25f;
  const float s1f = (box[3] - box[1]) * 1.25f;
  // numpy 2: float32 scalars with a Python float stay float32.
  const float aspect = static_cast<float>(kW) / static_cast<float>(kH);
  float s0 = s0f, s1 = s1f;
  if (s0f > s1f * aspect) {
    s1 = s0f / aspect;
  } else {
    s0 = s1f * aspect;
  }
  // _wb_warp: three point pairs (as float32) -> getAffineTransform (float64), then
  // warpAffine(float image, INTER_LINEAR, border 0) on the inverse mapping.
  const float src[3][2] = {{cx, cy},
                           {cx, cy - s0 * 0.5f},
                           {cx + s0 * 0.5f, cy - s0 * 0.5f}};
  const float dst[3][2] = {{kW * 0.5f, kH * 0.5f}, {kW * 0.5f, kH * 0.5f - kW * 0.5f}, {static_cast<float>(kW), kH * 0.5f - kW * 0.5f}};
  // Solve dst = M * [src, 1] (2 x 3) exactly like cv::getAffineTransform (6 x 6 system).
  double A[6][7] = {};
  for (int i = 0; i < 3; ++i) {
    A[i][0] = A[i + 3][3] = src[i][0];
    A[i][1] = A[i + 3][4] = src[i][1];
    A[i][2] = A[i + 3][5] = 1.0;
    A[i][6] = dst[i][0];
    A[i + 3][6] = dst[i][1];
  }
  for (int c = 0; c < 6; ++c) {  // Gaussian elimination with partial pivoting
    int piv = c;
    for (int r = c + 1; r < 6; ++r) {
      if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
    }
    std::swap(A[c], A[piv]);
    for (int r = 0; r < 6; ++r) {
      if (r == c || A[c][c] == 0.0) continue;
      const double f = A[r][c] / A[c][c];
      for (int k = c; k < 7; ++k) A[r][k] -= f * A[c][k];
    }
  }
  double M[6];
  for (int i = 0; i < 6; ++i) M[i] = A[i][6] / A[i][i];
  // Inverse of [[M0 M1 M2],[M3 M4 M5]].
  const double det = M[0] * M[4] - M[1] * M[3];
  const double i0 = M[4] / det, i1 = -M[1] / det, i3 = -M[3] / det, i4 = M[0] / det;
  const double i2 = -(i0 * M[2] + i1 * M[5]), i5 = -(i3 * M[2] + i4 * M[5]);
  const std::size_t plane = static_cast<std::size_t>(kW) * kH;
  Nd x{{1, 3, kH, kW}, std::vector<float>(plane * 3)};
  const int H = image.height, W = image.width;
  for (int r = 0; r < kH; ++r) {
    for (int c = 0; c < kW; ++c) {
      const double xs = i0 * c + i1 * r + i2;
      const double ys = i3 * c + i4 * r + i5;
      const int x0 = static_cast<int>(std::floor(xs));
      const int y0 = static_cast<int>(std::floor(ys));
      const double fx = xs - x0, fy = ys - y0;
      double v[3] = {0, 0, 0};
      for (int dy = 0; dy < 2; ++dy) {
        const int yy = y0 + dy;
        if (yy < 0 || yy >= H) continue;
        for (int dx = 0; dx < 2; ++dx) {
          const int xx = x0 + dx;
          if (xx < 0 || xx >= W) continue;
          const double wgt = (dy ? fy : 1 - fy) * (dx ? fx : 1 - fx);
          for (int ch = 0; ch < 3; ++ch) v[ch] += wgt * image.rgb[(static_cast<std::size_t>(yy) * W + xx) * 3 + ch];
        }
      }
      for (int ch = 0; ch < 3; ++ch) {
        const auto crop = static_cast<float>(v[ch]);
        x.d[ch * plane + static_cast<std::size_t>(r) * kW + c] =
            static_cast<float>((static_cast<double>(crop) - p.mean[ch]) / p.stdv[ch]);
      }
    }
  }
  if (taps != nullptr) taps->input = x.d;
  const Nd out = p.Run(std::move(x));
  // out.shape = [1, K, Lx, -1, Ly]; x data then y data.
  const auto K = out.shape[1];
  const auto Lx = out.shape[2];
  const auto Ly = out.shape[4];
  const float* sx = out.d.data();
  const float* sy = out.d.data() + K * Lx;
  if (taps != nullptr) {
    taps->simcc_x.assign(sx, sx + K * Lx);
    taps->simcc_y.assign(sy, sy + K * Ly);
  }
  std::vector<Keypoint> pts(static_cast<std::size_t>(K));
  for (std::int64_t k = 0; k < K; ++k) {
    const float* rx = sx + k * Lx;
    const float* ry = sy + k * Ly;
    const auto ax = std::max_element(rx, rx + Lx) - rx;  // first maximum, like np.argmax
    const auto ay = std::max_element(ry, ry + Ly) - ry;
    const float val = std::min(rx[ax], ry[ay]);
    // pts = argmax / split_ratio / (w, h) * scale + center - scale / 2 (float32 then float64)
    const float px = static_cast<float>(ax) / p.split_ratio;
    const float py = static_cast<float>(ay) / p.split_ratio;
    Keypoint kp;
    kp.x = static_cast<float>(static_cast<double>(px) / kW * s0 + cx - static_cast<double>(s0) / 2);
    kp.y = static_cast<float>(static_cast<double>(py) / kH * s1 + cy - static_cast<double>(s1) / 2);
    kp.score = val;
    pts[static_cast<std::size_t>(k)] = kp;
  }
  return pts;
}

}  // namespace gufo::models::ccip
