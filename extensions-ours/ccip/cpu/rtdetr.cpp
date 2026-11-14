// RT-DETR r50vd (transformers RTDetrForObjectDetection, eval), CPU fp32.
// Conv + BatchNorm pairs are folded by reference/convert_pose.py.
// Written with AI assistance (Claude Code).
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <gufo/ext/json.hpp>

#include "ccip/cpu/nn.hpp"
#include "ccip/pose.hpp"

namespace gufo::models::ccip {

using nn::Act;
using nn::Map;

struct RtDetr::Impl {
  nn::Weights w;
  std::vector<std::string> labels;
  int d{256};
  int enc_heads{8};
  int dec_layers{6};
  int dec_heads{8};
  int levels{3};
  int points{4};
  int queries{300};
  float eps{1e-5f};
  double temperature{10000.0};

  Map Bottleneck(const Map& x, const std::string& n, int stride) const {
    Map y = nn::Conv(w, n + "layer.0", x, 1, 0, Act::kRelu);
    y = nn::Conv(w, n + "layer.1", y, stride, 1, Act::kRelu);
    y = nn::Conv(w, n + "layer.2", y, 1, 0, Act::kNone);
    Map sc;
    if (stride == 2) {
      sc = nn::Conv(w, n + "shortcut.1", nn::AvgPoolCeil(x, 2, 2), 1, 0, Act::kNone);
    } else if (w.Has(n + "shortcut.weight")) {
      sc = nn::Conv(w, n + "shortcut", x, 1, 0, Act::kNone);
    } else {
      sc = x;
    }
    nn::AddInPlace(&y, sc);
    nn::Activate(y.d.data(), y.d.size(), Act::kRelu);
    return y;
  }

  Map Csp(const Map& x, const std::string& n) const {
    Map a = nn::Conv(w, n + ".conv1", x, 1, 0, Act::kSilu);
    for (int i = 0; i < 3; ++i) {
      const std::string b = n + ".bottlenecks." + std::to_string(i);
      Map y = nn::Conv(w, b + ".conv1", a, 1, 1, Act::kNone);
      nn::AddInPlace(&y, nn::Conv(w, b + ".conv2", a, 1, 0, Act::kNone));
      nn::Activate(y.d.data(), y.d.size(), Act::kSilu);
      a = std::move(y);
    }
    nn::AddInPlace(&a, nn::Conv(w, n + ".conv2", x, 1, 0, Act::kSilu));
    return a;  // conv3 is the identity (hidden == out channels)
  }

  // Self attention of RTDetrSelfAttention: q = k = x + pos, v = x.
  std::vector<float> SelfAttention(const std::string& n, const std::vector<float>& x,
                                   const std::vector<float>& pos, int rows, int heads) const {
    std::vector<float> qk(x);
    for (std::size_t i = 0; i < qk.size(); ++i) qk[i] += pos[i];
    const auto q = nn::Linear(w, n + ".q_proj", qk.data(), rows);
    const auto k = nn::Linear(w, n + ".k_proj", qk.data(), rows);
    const auto v = nn::Linear(w, n + ".v_proj", x.data(), rows);
    const auto o = nn::Attention(q.data(), k.data(), v.data(), rows, rows, d, heads);
    return nn::Linear(w, n + ".out_proj", o.data(), rows);
  }

  std::vector<float> Mlp(const std::string& n, const float* x, int rows, int layers) const {
    std::vector<float> h(x, x + static_cast<std::size_t>(rows) * w.Get(n + ".layers.0.weight").shape[1]);
    for (int i = 0; i < layers; ++i) {
      h = nn::Linear(w, n + ".layers." + std::to_string(i), h.data(), rows,
                     i < layers - 1 ? Act::kRelu : Act::kNone);
    }
    return h;
  }

  static void AddResidualNorm(const nn::Weights& w, const std::string& norm, std::vector<float>* x,
                              const std::vector<float>& y, int rows, int dim, float eps) {
    for (std::size_t i = 0; i < x->size(); ++i) (*x)[i] += y[i];
    nn::LayerNorm(w, norm, x->data(), rows, dim, eps);
  }

  void Aifi(Map* f) const {
    const int rows = f->h * f->w;
    std::vector<float> t = nn::ToTokens(*f);
    // build_2d_sinusoidal_position_embedding: [sin_h | cos_h | sin_w | cos_w], h outer (float64).
    const int pd = d / 4;
    std::vector<float> pos(static_cast<std::size_t>(rows) * d);
    for (int y = 0; y < f->h; ++y) {
      for (int x = 0; x < f->w; ++x) {
        float* p = &pos[static_cast<std::size_t>(y * f->w + x) * d];
        for (int i = 0; i < pd; ++i) {
          const double omega = 1.0 / std::pow(temperature, static_cast<double>(i) / pd);
          const double eh = y * omega;
          const double ew = x * omega;
          p[i] = static_cast<float>(std::sin(eh));
          p[pd + i] = static_cast<float>(std::cos(eh));
          p[2 * pd + i] = static_cast<float>(std::sin(ew));
          p[3 * pd + i] = static_cast<float>(std::cos(ew));
        }
      }
    }
    const std::string n = "encoder.encoder.0.layers.0.";
    AddResidualNorm(w, n + "self_attn_layer_norm", &t, SelfAttention(n + "self_attn", t, pos, rows, enc_heads),
                    rows, d, eps);
    const auto h = nn::Linear(w, n + "fc1", t.data(), rows, Act::kGelu);
    AddResidualNorm(w, n + "final_layer_norm", &t, nn::Linear(w, n + "fc2", h.data(), rows), rows, d, eps);
    *f = nn::FromTokens(t, d, f->h, f->w);
  }
};

RtDetr::RtDetr() : impl_(std::make_unique<Impl>()) {}
RtDetr::~RtDetr() = default;

std::unique_ptr<RtDetr> RtDetr::Load(const std::filesystem::path& dir, std::string* error) {
  std::unique_ptr<RtDetr> m(new RtDetr());
  Impl& p = *m->impl_;
  if (!p.w.Load(dir / "rtdetr_r50vd_coco_o365.safetensors", error)) return nullptr;
  std::ifstream in(dir / "rtdetr_r50vd_coco_o365.json", std::ios::binary);
  if (!in) {
    if (error) *error = "missing " + (dir / "rtdetr_r50vd_coco_o365.json").string();
    return nullptr;
  }
  std::stringstream ss;
  ss << in.rdbuf();
  try {
    const json::Value a = json::parse(ss.str());
    p.d = static_cast<int>(a.member_size("d_model", 256));
    p.enc_heads = static_cast<int>(a.member_size("encoder_heads", 8));
    p.dec_layers = static_cast<int>(a.member_size("decoder_layers", 6));
    p.dec_heads = static_cast<int>(a.member_size("decoder_heads", 8));
    p.levels = static_cast<int>(a.member_size("feature_levels", 3));
    p.points = static_cast<int>(a.member_size("decoder_points", 4));
    p.queries = static_cast<int>(a.member_size("num_queries", 300));
    p.eps = static_cast<float>(a.member_double("layer_norm_eps", 1e-5));
    p.temperature = a.member_double("pos_temperature", 10000.0);
    for (const auto& l : a.find("labels")->items()) p.labels.push_back(l.str());
    if (a.member_size("encoder_layers", 1) != 1) throw std::runtime_error("encoder_layers != 1");
  } catch (const std::exception& e) {
    if (error) *error = std::string("rtdetr: ") + e.what();
    return nullptr;
  }
  return m;
}

const std::vector<std::string>& RtDetr::Labels() const { return impl_->labels; }

std::vector<ObjectBox> RtDetr::Detect(const RgbImage& image, float threshold, RtDetrTaps* taps) const {
  const Impl& p = *impl_;
  const nn::Weights& w = p.w;
  const int d = p.d;
  // Preprocess: torchvision bilinear antialias resize to 640x640, x * (1/255).
  const RgbImage r = ResizeBilinearTorchAA(image, 640, 640);
  Map x{3, 640, 640, std::vector<float>(static_cast<std::size_t>(3) * 640 * 640)};
  const std::size_t plane = static_cast<std::size_t>(640) * 640;
  for (std::size_t i = 0; i < plane; ++i) {
    for (int c = 0; c < 3; ++c) x.d[c * plane + i] = static_cast<float>(r.rgb[i * 3 + c]) * (1.0f / 255.0f);
  }
  if (taps != nullptr) taps->input = x.d;

  // Backbone (ResNet50-vd).
  const std::string bb = "backbone.model.";
  Map h = nn::Conv(w, bb + "embedder.embedder.0", x, 2, 1, Act::kRelu);
  h = nn::Conv(w, bb + "embedder.embedder.1", h, 1, 1, Act::kRelu);
  h = nn::Conv(w, bb + "embedder.embedder.2", h, 1, 1, Act::kRelu);
  h = nn::MaxPool(h, 3, 2, 1);
  const int depths[4] = {3, 4, 6, 3};
  std::vector<Map> feats;
  for (int s = 0; s < 4; ++s) {
    for (int l = 0; l < depths[s]; ++l) {
      const std::string n = bb + "encoder.stages." + std::to_string(s) + ".layers." + std::to_string(l) + ".";
      h = p.Bottleneck(h, n, (s > 0 && l == 0) ? 2 : 1);
    }
    if (s >= 1) feats.push_back(h);
  }
  std::vector<Map> proj;
  for (int i = 0; i < 3; ++i) proj.push_back(nn::Conv(w, "encoder_input_proj." + std::to_string(i), feats[i], 1, 0, Act::kNone));
  feats.clear();

  // Hybrid encoder: AIFI on the last level, FPN top-down, PAN bottom-up.
  p.Aifi(&proj[2]);
  std::vector<Map> fpn{proj[2]};
  for (int i = 0; i < 2; ++i) {
    Map top = nn::Conv(w, "encoder.lateral_convs." + std::to_string(i), fpn.back(), 1, 0, Act::kSilu);
    fpn.back() = top;
    fpn.push_back(p.Csp(nn::Concat(nn::UpsampleNearest2(top), proj[1 - i]), "encoder.fpn_blocks." + std::to_string(i)));
  }
  std::reverse(fpn.begin(), fpn.end());
  std::vector<Map> pan{fpn[0]};
  for (int i = 0; i < 2; ++i) {
    Map down = nn::Conv(w, "encoder.downsample_convs." + std::to_string(i), pan.back(), 2, 1, Act::kSilu);
    pan.push_back(p.Csp(nn::Concat(down, fpn[i + 1]), "encoder.pan_blocks." + std::to_string(i)));
  }
  if (taps != nullptr) {
    taps->encoder.clear();
    for (const Map& m : pan) taps->encoder.push_back(m.d);
  }

  // Decoder inputs: flattened levels [8400, d].
  std::vector<std::array<int, 2>> shapes;
  std::vector<int> starts;
  std::vector<float> flat;
  int total = 0;
  for (int i = 0; i < p.levels; ++i) {
    const Map s = nn::Conv(w, "decoder_input_proj." + std::to_string(i), pan[static_cast<std::size_t>(i)], 1, 0, Act::kNone);
    shapes.push_back({s.h, s.w});
    starts.push_back(total);
    const auto t = nn::ToTokens(s);
    flat.insert(flat.end(), t.begin(), t.end());
    total += s.h * s.w;
  }
  pan.clear();
  fpn.clear();
  proj.clear();

  // Anchors and query selection.
  std::vector<float> anchors(static_cast<std::size_t>(total) * 4);
  std::vector<char> valid(static_cast<std::size_t>(total));
  {
    int idx = 0;
    for (int l = 0; l < p.levels; ++l) {
      const int hh = shapes[static_cast<std::size_t>(l)][0];
      const int ww = shapes[static_cast<std::size_t>(l)][1];
      for (int yy = 0; yy < hh; ++yy) {
        for (int xx = 0; xx < ww; ++xx, ++idx) {
          const float a[4] = {(static_cast<float>(xx) + 0.5f) / static_cast<float>(ww),
                              (static_cast<float>(yy) + 0.5f) / static_cast<float>(hh),
                              0.05f * static_cast<float>(1 << l), 0.05f * static_cast<float>(1 << l)};
          bool ok = true;
          for (float v : a) ok = ok && v > 1e-2f && v < 1.0f - 1e-2f;
          valid[static_cast<std::size_t>(idx)] = ok ? 1 : 0;
          for (int k = 0; k < 4; ++k) {
            anchors[static_cast<std::size_t>(idx) * 4 + k] =
                ok ? std::log(a[k] / (1.0f - a[k])) : std::numeric_limits<float>::max();
          }
        }
      }
    }
  }
  std::vector<float> memory(flat);
  for (int i = 0; i < total; ++i) {
    if (!valid[static_cast<std::size_t>(i)]) std::fill_n(memory.begin() + static_cast<std::ptrdiff_t>(i) * d, d, 0.0f);
  }
  std::vector<float> om = nn::Linear(w, "enc_output.0", memory.data(), total);
  nn::LayerNorm(w, "enc_output.1", om.data(), total, d, p.eps);
  memory.clear();
  const auto cls = nn::Linear(w, "enc_score_head", om.data(), total);
  const int classes = static_cast<int>(p.labels.size());
  std::vector<float> best(static_cast<std::size_t>(total));
  for (int i = 0; i < total; ++i) {
    best[static_cast<std::size_t>(i)] =
        *std::max_element(cls.begin() + static_cast<std::ptrdiff_t>(i) * classes,
                          cls.begin() + static_cast<std::ptrdiff_t>(i + 1) * classes);
  }
  std::vector<int> order(static_cast<std::size_t>(total));
  std::iota(order.begin(), order.end(), 0);
  const int nq = std::min(p.queries, total);
  std::partial_sort(order.begin(), order.begin() + nq, order.end(), [&](int a, int b) {
    return best[static_cast<std::size_t>(a)] > best[static_cast<std::size_t>(b)] ||
           (best[static_cast<std::size_t>(a)] == best[static_cast<std::size_t>(b)] && a < b);
  });
  std::vector<float> hsel(static_cast<std::size_t>(nq) * d);
  for (int q = 0; q < nq; ++q) {
    std::copy_n(om.begin() + static_cast<std::ptrdiff_t>(order[static_cast<std::size_t>(q)]) * d, d,
                hsel.begin() + static_cast<std::ptrdiff_t>(q) * d);
  }
  const auto coord = p.Mlp("enc_bbox_head", hsel.data(), nq, 3);
  std::vector<float> ref(static_cast<std::size_t>(nq) * 4);
  for (int q = 0; q < nq; ++q) {
    for (int k = 0; k < 4; ++k) {
      const float v = coord[static_cast<std::size_t>(q) * 4 + k] +
                      anchors[static_cast<std::size_t>(order[static_cast<std::size_t>(q)]) * 4 + k];
      ref[static_cast<std::size_t>(q) * 4 + k] = 1.0f / (1.0f + std::exp(-v));
    }
  }
  om.clear();

  // Decoder.
  std::vector<float> hq = std::move(hsel);
  std::vector<float> logits;
  const int hd = d / p.dec_heads;
  const int lp = p.levels * p.points;
  for (int l = 0; l < p.dec_layers; ++l) {
    const std::string n = "decoder.layers." + std::to_string(l) + ".";
    const auto pos = p.Mlp("decoder.query_pos_head", ref.data(), nq, 2);
    Impl::AddResidualNorm(w, n + "self_attn_layer_norm", &hq, p.SelfAttention(n + "self_attn", hq, pos, nq, p.dec_heads),
                          nq, d, p.eps);
    // Multi-scale deformable attention.
    std::vector<float> query(hq);
    for (std::size_t i = 0; i < query.size(); ++i) query[i] += pos[i];
    const auto value = nn::Linear(w, n + "encoder_attn.value_proj", flat.data(), total);
    const auto offsets = nn::Linear(w, n + "encoder_attn.sampling_offsets", query.data(), nq);
    auto attw = nn::Linear(w, n + "encoder_attn.attention_weights", query.data(), nq);
    nn::SoftmaxRows(attw.data(), nq * p.dec_heads, lp);
    std::vector<float> sampled(static_cast<std::size_t>(nq) * d, 0.0f);
#pragma omp parallel for schedule(static)
    for (int t = 0; t < nq * p.dec_heads; ++t) {
      const int q = t / p.dec_heads;
      const int hh = t % p.dec_heads;
      const float* rq = &ref[static_cast<std::size_t>(q) * 4];
      float* out = &sampled[static_cast<std::size_t>(q) * d + hh * hd];
      for (int lv = 0; lv < p.levels; ++lv) {
        const int H = shapes[static_cast<std::size_t>(lv)][0];
        const int W = shapes[static_cast<std::size_t>(lv)][1];
        for (int pt = 0; pt < p.points; ++pt) {
          const std::size_t oi = ((static_cast<std::size_t>(q) * p.dec_heads + hh) * p.levels + lv) * p.points + pt;
          const float lx = rq[0] + offsets[oi * 2] / static_cast<float>(p.points) * rq[2] * 0.5f;
          const float ly = rq[1] + offsets[oi * 2 + 1] / static_cast<float>(p.points) * rq[3] * 0.5f;
          const float aw = attw[(static_cast<std::size_t>(q) * p.dec_heads + hh) * lp + lv * p.points + pt];
          // grid_sample(bilinear, zeros, align_corners=False) on grid 2*loc-1.
          const float gx = 2.0f * lx - 1.0f;
          const float gy = 2.0f * ly - 1.0f;
          const float ix = ((gx + 1.0f) * static_cast<float>(W) - 1.0f) / 2.0f;
          const float iy = ((gy + 1.0f) * static_cast<float>(H) - 1.0f) / 2.0f;
          const int x0 = static_cast<int>(std::floor(ix));
          const int y0 = static_cast<int>(std::floor(iy));
          const float fx = ix - static_cast<float>(x0);
          const float fy = iy - static_cast<float>(y0);
          const int cx[2] = {x0, x0 + 1};
          const int cy[2] = {y0, y0 + 1};
          const float wx[2] = {1.0f - fx, fx};
          const float wy[2] = {1.0f - fy, fy};
          for (int a = 0; a < 2; ++a) {
            if (cy[a] < 0 || cy[a] >= H) continue;
            for (int b = 0; b < 2; ++b) {
              if (cx[b] < 0 || cx[b] >= W) continue;
              const float wgt = wy[a] * wx[b] * aw;
              const float* v = &value[(static_cast<std::size_t>(starts[static_cast<std::size_t>(lv)]) +
                                       static_cast<std::size_t>(cy[a]) * W + cx[b]) * d + hh * hd];
              for (int e = 0; e < hd; ++e) out[e] += wgt * v[e];
            }
          }
        }
      }
    }
    Impl::AddResidualNorm(w, n + "encoder_attn_layer_norm", &hq,
                          nn::Linear(w, n + "encoder_attn.output_proj", sampled.data(), nq), nq, d, p.eps);
    const auto f1 = nn::Linear(w, n + "fc1", hq.data(), nq, Act::kRelu);
    Impl::AddResidualNorm(w, n + "final_layer_norm", &hq, nn::Linear(w, n + "fc2", f1.data(), nq), nq, d, p.eps);
    const auto delta = p.Mlp("decoder.bbox_embed." + std::to_string(l), hq.data(), nq, 3);
    for (std::size_t i = 0; i < ref.size(); ++i) {
      const float r0 = std::clamp(ref[i], 0.0f, 1.0f);
      const float inv = std::log(std::max(r0, 1e-5f) / std::max(1.0f - r0, 1e-5f));
      ref[i] = 1.0f / (1.0f + std::exp(-(delta[i] + inv)));
    }
    if (l == p.dec_layers - 1) logits = nn::Linear(w, "decoder.class_embed." + std::to_string(l), hq.data(), nq);
  }
  if (taps != nullptr) {
    taps->logits = logits;
    taps->boxes = ref;
  }

  // post_process_object_detection: sigmoid, top nq over queries x classes, score > threshold.
  std::vector<float> scores(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) scores[i] = 1.0f / (1.0f + std::exp(-logits[i]));
  std::vector<int> idx(scores.size());
  std::iota(idx.begin(), idx.end(), 0);
  const int keep = std::min<int>(nq, static_cast<int>(idx.size()));
  std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(), [&](int a, int b) {
    return scores[static_cast<std::size_t>(a)] > scores[static_cast<std::size_t>(b)] ||
           (scores[static_cast<std::size_t>(a)] == scores[static_cast<std::size_t>(b)] && a < b);
  });
  std::vector<ObjectBox> outv;
  const auto W = static_cast<float>(image.width);
  const auto H = static_cast<float>(image.height);
  for (int i = 0; i < keep; ++i) {
    const int k = idx[static_cast<std::size_t>(i)];
    const float s = scores[static_cast<std::size_t>(k)];
    if (!(s > threshold)) continue;
    const int q = k / classes;
    const float* b = &ref[static_cast<std::size_t>(q) * 4];
    ObjectBox o;
    o.box = {(b[0] - 0.5f * b[2]) * W, (b[1] - 0.5f * b[3]) * H, (b[0] + 0.5f * b[2]) * W,
             (b[1] + 0.5f * b[3]) * H};
    o.label = k % classes;
    o.score = s;
    outv.push_back(o);
  }
  return outv;
}

}  // namespace gufo::models::ccip
