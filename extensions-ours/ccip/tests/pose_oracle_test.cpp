// "pose" task (RT-DETR persons + ViTPose keypoints) against the film
// pipeline's transformers reference (reference/make_pose_oracle.py).
// Env: GUFO_VISION_DIR (rtdetr_r50vd_coco_o365.* and vitpose_base_simple.*),
// GUFO_POSE_ORACLE (manifest.json + pose_*.safetensors), optional
// GUFO_CCIP_THREADS. Exit 77 (skip) without them.
// Pass: the RT-DETR input bit-exact; the same detections (count, label, box
// within 1 px, score within 1e-3); per person the 17 keypoints within 1 px
// and their scores within 1e-3; with DWPose data in the oracle, the 133
// whole-body points of each person within 1 px or one SimCC bin (the
// heatmap is quantised: a bin is 0.5 crop px = box x 1.25 / 576 image px;
// OpenCV 5's warpAffine and ours differ by ~2e-4 in the crop, which can tip
// an arg-max by one bin) and scores within 1e-3.
// Written with AI assistance (Claude Code).
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gufo/ext/json.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ccip/pose.hpp"
#include "ccip/tensors.hpp"

namespace ccip = gufo::models::ccip;

namespace {

const char* Env(const char* name) {
  const char* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? v : nullptr;
}

double Ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

double MaxDiff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return 1e30;
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
  return m;
}

}  // namespace

int main() {
  const char* model_dir = Env("GUFO_VISION_DIR");
  const char* oracle_dir = Env("GUFO_POSE_ORACLE");
  if (!model_dir || !oracle_dir) {
    std::printf("skip: set GUFO_VISION_DIR and GUFO_POSE_ORACLE\n");
    return 77;
  }
#ifdef _OPENMP
  if (const char* t = Env("GUFO_CCIP_THREADS")) omp_set_num_threads(std::atoi(t));
#endif
  const std::filesystem::path odir(oracle_dir);
  std::ifstream mf(odir / "manifest.json", std::ios::binary);
  if (!mf) {
    std::printf("missing manifest.json\n");
    return 1;
  }
  std::stringstream ss;
  ss << mf.rdbuf();
  const gufo::json::Value manifest = gufo::json::parse(ss.str());
  const float threshold = static_cast<float>(manifest.member_double("min_score", 0.35));
  const std::size_t max_persons = manifest.member_size("max_persons", 4);
  std::string error;
  const auto t0 = std::chrono::steady_clock::now();
  auto det = ccip::RtDetr::Load(model_dir, &error);
  if (!det) {
    std::printf("load rtdetr: %s\n", error.c_str());
    return 1;
  }
  auto pose = ccip::VitPose::Load(model_dir, &error);
  if (!pose) {
    std::printf("load vitpose: %s\n", error.c_str());
    return 1;
  }
  auto wb = ccip::DwPose::Load(model_dir, &error);
  if (!wb) {
    std::printf("load dwpose: %s\n", error.c_str());
    return 1;
  }
  double worst_wb = 0, worst_wbs = 0;
  std::printf("loaded in %.0f ms\n", Ms(t0, std::chrono::steady_clock::now()));
  bool ok = true;
  double worst_box = 0, worst_score = 0, worst_kp = 0, worst_kps = 0;
  for (const auto& item : manifest.find("images")->items()) {
    ccip::TensorFile f;
    if (!f.Load(odir / item.member_str("file"), &error)) {
      std::printf("oracle: %s\n", error.c_str());
      return 1;
    }
    const ccip::Tensor* t = f.Find("rgb");
    ccip::RgbImage img;
    img.height = static_cast<int>(t->shape[0]);
    img.width = static_cast<int>(t->shape[1]);
    img.rgb = t->u8;
    ccip::RtDetrTaps taps;
    const auto a = std::chrono::steady_clock::now();
    const std::vector<ccip::ObjectBox> got = det->Detect(img, threshold, &taps);
    const auto b = std::chrono::steady_clock::now();
    std::printf("\n%s: input diff %.2e", item.member_str("source").c_str(), MaxDiff(taps.input, f.F32("det_input")));
    if (MaxDiff(taps.input, f.F32("det_input")) > 0) ok = false;
    for (int l = 0; l < 3; ++l) {
      std::printf("  enc%d %.2e", l, MaxDiff(taps.encoder[static_cast<std::size_t>(l)], f.F32("det_encoder" + std::to_string(l))));
    }
    std::printf("  RT-DETR %.0f ms\n", Ms(a, b));
    const auto& want = item.find("detections")->items();
    double box_diff = 0, score_diff = 0;
    bool same = got.size() == want.size();
    for (std::size_t i = 0; same && i < got.size(); ++i) {
      if (got[i].label != static_cast<int>(want[i].member_size("label"))) same = false;
      const auto& wb = want[i].find("box")->items();
      for (int k = 0; k < 4; ++k) box_diff = std::max(box_diff, std::fabs(got[i].box[static_cast<std::size_t>(k)] - wb[static_cast<std::size_t>(k)].as_double()));
      score_diff = std::max(score_diff, std::fabs(got[i].score - want[i].member_double("score")));
    }
    const bool det_ok = same && box_diff <= 1.0 && score_diff <= 1e-3;
    std::printf("  detections %zu (reference %zu)  max box diff %.3f px  max score diff %.1e  %s\n", got.size(),
                want.size(), box_diff, score_diff, det_ok ? "ok" : "FAIL");
    ok = ok && det_ok;
    worst_box = std::max(worst_box, box_diff);
    worst_score = std::max(worst_score, score_diff);
    // Persons as posen_erkennung: label 0, by score, at most max_persons.
    std::vector<ccip::ObjectBox> persons;
    for (const auto& o : got) {
      if (o.label == 0) persons.push_back(o);
    }
    std::stable_sort(persons.begin(), persons.end(), [](const auto& x, const auto& y) { return x.score > y.score; });
    if (persons.size() > max_persons) persons.resize(max_persons);
    const auto& wp = item.find("persons")->items();
    if (persons.size() != wp.size()) {
      std::printf("  persons %zu (reference %zu) FAIL\n", persons.size(), wp.size());
      ok = false;
      continue;
    }
    for (std::size_t i = 0; i < persons.size(); ++i) {
      ccip::VitPoseTaps pt;
      const auto c = std::chrono::steady_clock::now();
      const auto kps = pose->Estimate(img, persons[i].box, &pt);
      const auto d = std::chrono::steady_clock::now();
      const auto& wk = wp[i].find("keypoints")->items();
      const auto& ws = wp[i].find("scores")->items();
      double kd = 0, sd = 0;
      for (std::size_t k = 0; k < kps.size(); ++k) {
        kd = std::max(kd, std::fabs(kps[k].x - wk[k].items()[0].as_double()));
        kd = std::max(kd, std::fabs(kps[k].y - wk[k].items()[1].as_double()));
        sd = std::max(sd, std::fabs(kps[k].score - ws[k].as_double()));
      }
      const bool kp_ok = kps.size() == wk.size() && kd <= 1.0 && sd <= 1e-3;
      // The same with the reference box: separates network error from box rounding.
      std::array<float, 4> rbox{};
      for (int k = 0; k < 4; ++k) rbox[static_cast<std::size_t>(k)] = static_cast<float>(wp[i].find("box")->items()[static_cast<std::size_t>(k)].as_double());
      ccip::VitPoseTaps rt;
      const auto rk = pose->Estimate(img, rbox, &rt);
      double rsd = 0;
      for (std::size_t k = 0; k < rk.size(); ++k) rsd = std::max(rsd, std::fabs(rk[k].score - ws[k].as_double()));
      const auto& ref_in = f.F32("pose_input");
      const std::size_t plane = rt.input.size();
      std::vector<float> ref_slice(ref_in.begin() + static_cast<std::ptrdiff_t>(i * plane),
                                   ref_in.begin() + static_cast<std::ptrdiff_t>((i + 1) * plane));
      std::vector<float> our_slice(pt.input);
      int flips = 0;
      for (std::size_t k = 0; k < plane; ++k) flips += std::fabs(our_slice[k] - ref_slice[k]) > 1e-3f ? 1 : 0;
      std::printf("  person %zu: keypoints max diff %.3f px, scores %.1e (reference box: scores %.1e, input %.1e; "
                  "our box: %d input values off), ViTPose %.0f ms  %s\n",
                  i, kd, sd, rsd, MaxDiff(rt.input, ref_slice), flips, Ms(c, d), kp_ok ? "ok" : "FAIL");
      ok = ok && kp_ok;
      worst_kp = std::max(worst_kp, kd);
      worst_kps = std::max(worst_kps, sd);
      if (const auto* wwb = wp[i].find("wholebody"); wwb != nullptr) {
        ccip::DwPoseTaps dt;
        const auto e = std::chrono::steady_clock::now();
        const auto pts = wb->Estimate(img, persons[i].box, &dt);
        const auto g = std::chrono::steady_clock::now();
        double wd = 0, ws2 = 0;
        int over = 0;
        int over_bin = 0;
        float bs0 = (persons[i].box[2] - persons[i].box[0]) * 1.25f;
        float bs1 = (persons[i].box[3] - persons[i].box[1]) * 1.25f;
        if (bs0 > bs1 * 0.75f) bs1 = bs0 / 0.75f; else bs0 = bs1 * 0.75f;
        const double bin = std::max(bs0 / 576.0, bs1 / 768.0);
        for (std::size_t k = 0; k < pts.size(); ++k) {
          const auto& r = wwb->items()[k].items();
          const double dd = std::max(std::fabs(pts[k].x - r[0].as_double()), std::fabs(pts[k].y - r[1].as_double()));
          if (dd > 1.0) ++over;
          if (dd > std::max(1.0, bin + 1e-3)) ++over_bin;
          wd = std::max(wd, dd);
          ws2 = std::max(ws2, std::fabs(pts[k].score - r[2].as_double()));
        }
        const bool wb_ok = pts.size() == wwb->size() && over_bin == 0 && ws2 <= 1e-3;
        std::printf("            wholebody: max diff %.3f px (%d of %zu over 1 px, bin %.2f px, %d over a bin), "
                    "scores %.1e, DWPose %.0f ms  %s\n",
                    wd, over, pts.size(), bin, over_bin, ws2, Ms(e, g), wb_ok ? "ok" : "FAIL");
        ok = ok && wb_ok;
        worst_wb = std::max(worst_wb, wd);
        worst_wbs = std::max(worst_wbs, ws2);
      }
    }
  }
  std::printf("\nworst: box %.3f px, det score %.1e, keypoint %.3f px, keypoint score %.1e, "
              "wholebody %.3f px, wholebody score %.1e\n%s\n",
              worst_box, worst_score, worst_kp, worst_kps, worst_wb, worst_wbs, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
