// /v1/vision/detect detectors against dghs-imgutils (reference/make_vision_oracle.py).
// Env: GUFO_VISION_DIR (dir with <model>.safetensors/.json for the four default
// detectors), GUFO_VISION_ORACLE (dir with manifest.json), optional
// GUFO_CCIP_THREADS. Exit 77 (skip) without them.
// Pass: the stretched 640x640 input bit-exact, MeanLuma equal to Pillow's,
// per task the same number of boxes, every box within 1 px and every score
// within 1e-3 of imgutils, on the image and on its gamma-0.5 copy (the film
// pipeline's dark-keyframe path).
// Written with AI assistance (Claude Code).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gufo/ext/json.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ccip/ccip.hpp"
#include "ccip/tensors.hpp"

namespace ccip = gufo::models::ccip;

namespace {

struct Task {
  const char* name;
  const char* stem;
  float conf;
  float iou;
};
// imgutils 0.19 defaults (detect/person.py, head.py, face.py, hand.py).
constexpr Task kTasks[] = {
    {"person", "person_detect_v1.1_m", 0.3f, 0.5f},
    {"head", "head_detect_v2.0_s", 0.4f, 0.7f},
    {"face", "face_detect_v1.4_s", 0.25f, 0.7f},
    {"hand", "hand_detect_v1.0_s", 0.35f, 0.7f},
};

const char* Env(const char* name) {
  const char* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? v : nullptr;
}

}  // namespace

int main() {
  const char* model_dir = Env("GUFO_VISION_DIR");
  const char* oracle_dir = Env("GUFO_VISION_ORACLE");
  if (!model_dir || !oracle_dir) {
    std::printf("skip: set GUFO_VISION_DIR and GUFO_VISION_ORACLE\n");
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
  std::printf("imgutils %s\n", manifest.member_str("imgutils").c_str());
  std::string error;
  std::vector<std::unique_ptr<ccip::YoloDetector>> nets;
  for (const Task& task : kTasks) {
    auto net = ccip::YoloDetector::Load(model_dir, task.stem, &error);
    if (!net) {
      std::printf("load %s: %s\n", task.stem, error.c_str());
      return 1;
    }
    nets.push_back(std::move(net));
  }
  bool ok = true;
  for (const auto& item : manifest.find("images")->items()) {
    ccip::TensorFile f;
    if (!f.Load(odir / item.member_str("file"), &error)) {
      std::printf("oracle: %s\n", error.c_str());
      return 1;
    }
    const ccip::Tensor* t = f.Find("rgb");
    if (t == nullptr || t->shape.size() != 3 || t->shape[2] != 3) {
      std::printf("rgb tensor missing\n");
      return 1;
    }
    ccip::RgbImage img;
    img.height = static_cast<int>(t->shape[0]);
    img.width = static_cast<int>(t->shape[1]);
    img.rgb = t->u8;
    const std::vector<float> x = ccip::StretchInput(img, 640, 640);
    const auto& ref_x = f.F32("input640");
    std::size_t differ = 0;
    for (std::size_t i = 0; i < x.size() && i < ref_x.size(); ++i) {
      if (x[i] != ref_x[i]) ++differ;
    }
    std::printf("\n%s (%dx%d): input %zu of %zu values differ\n",
                item.member_str("source").c_str(), img.width, img.height, differ, ref_x.size());
    if (differ != 0 || x.size() != ref_x.size()) ok = false;
    const double luma = ccip::MeanLuma(img);
    const double want_luma = item.member_double("mean_luma");
    std::printf("  mean luma %.6f (Pillow %.6f)\n", luma, want_luma);
    if (std::fabs(luma - want_luma) > 1e-9) ok = false;
    const ccip::RgbImage bright = ccip::ApplyGamma(img, 0.5);
    for (const char* variant : {"detections", "detections_gamma05"}) {
    const ccip::RgbImage& input = std::string(variant) == "detections" ? img : bright;
    const gufo::json::Value* want_all = item.find(variant);
    if (want_all == nullptr) continue;
    std::printf("  [%s]\n", variant);
    for (std::size_t k = 0; k < std::size(kTasks); ++k) {
      const Task& task = kTasks[k];
      const auto a = std::chrono::steady_clock::now();
      const std::vector<ccip::Detection> got = nets[k]->Detect(input, task.conf, task.iou);
      const auto b = std::chrono::steady_clock::now();
      const auto& want = want_all->find(task.name)->items();
      double max_px = 0;
      double max_score = 0;
      const bool same_count = got.size() == want.size();
      for (std::size_t i = 0; same_count && i < got.size(); ++i) {
        const auto& wb = want[i].find("box")->items();
        for (int c = 0; c < 4; ++c) {
          max_px = std::max(max_px, std::fabs(got[i].box[static_cast<std::size_t>(c)] -
                                              wb[static_cast<std::size_t>(c)].as_double()));
        }
        max_score = std::max(max_score,
                             std::fabs(got[i].score - want[i].member_double("score")));
      }
      const bool pass = same_count && max_px <= 1.0 && max_score <= 1e-3;
      std::printf("  %-6s %zu boxes (imgutils %zu)  max box diff %.0f px  max score diff %.1e  %.0f ms  %s\n",
                  task.name, got.size(), want.size(), max_px, max_score,
                  std::chrono::duration<double, std::milli>(b - a).count(), pass ? "ok" : "FAIL");
      ok = ok && pass;
    }
    }
  }
  std::printf("\n%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
