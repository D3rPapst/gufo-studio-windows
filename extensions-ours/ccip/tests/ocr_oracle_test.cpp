// /v1/vision/detect "ocr" against dghs-imgutils (reference/make_ocr_oracle.py).
// Env: GUFO_OCR_DIR (dir with ch_PP-OCRv4_det.* and ch_PP-OCRv4_rec.* from
// reference/convert_ocr.py), GUFO_OCR_ORACLE (dir with manifest.json), optional
// GUFO_CCIP_THREADS. Exit 77 (skip) without them.
// Checks per image:
//   - detector input bit-exact,
//   - post-processing on imgutils' own (onnxruntime) heat map: the same boxes
//     with identical corners and the same scores (1e-9),
//   - recognizer input images bit-exact (crop, 90 degree turn, resize),
//   - the native pipeline end to end: the same texts, rectangles within 2 px,
//     scores within 1e-3; native heat map vs onnxruntime is reported.
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
#include "ccip/ocr.hpp"
#include "ccip/tensors.hpp"

namespace ccip = gufo::models::ccip;
namespace json = gufo::json;

namespace {

const char* Env(const char* name) {
  const char* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? v : nullptr;
}

double Seconds(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

}  // namespace

int main() {
  const char* model_dir = Env("GUFO_OCR_DIR");
  const char* oracle_dir = Env("GUFO_OCR_ORACLE");
  if (!model_dir || !oracle_dir) {
    std::printf("skip: set GUFO_OCR_DIR and GUFO_OCR_ORACLE\n");
    return 77;
  }
  int threads = 8;
  if (const char* t = Env("GUFO_CCIP_THREADS")) threads = std::max(1, std::atoi(t));
  ccip::SetCpuThreads(threads);

  std::string error;
  const auto t_load = std::chrono::steady_clock::now();
  auto ocr = ccip::TextOcr::Load(model_dir, "ch_PP-OCRv4_det", "ch_PP-OCRv4_rec", &error);
  if (!ocr) {
    std::printf("FAIL load: %s\n", error.c_str());
    return 1;
  }
  std::printf("models loaded in %.2f s, %d threads\n",
              Seconds(t_load, std::chrono::steady_clock::now()), threads);

  const std::filesystem::path odir(oracle_dir);
  std::ifstream in(odir / "manifest.json", std::ios::binary);
  if (!in) {
    std::printf("FAIL: no manifest.json in %s\n", oracle_dir);
    return 1;
  }
  std::stringstream ss;
  ss << in.rdbuf();
  const json::Value manifest = json::parse(ss.str());
  std::printf("oracle: imgutils %s\n", manifest.member_str("imgutils", "?").c_str());

  const ccip::OcrOptions options;
  int failures = 0;
  double native_total = 0.0, imgutils_total = 0.0;
  int images = 0;
  double worst_heat = 0.0, worst_box = 0.0, worst_score = 0.0;
  for (const json::Value& entry : manifest.find("images")->items()) {
    const std::string source = entry.member_str("source");
    ccip::TensorFile tf;
    if (!tf.Load(odir / entry.member_str("file"), &error)) {
      std::printf("FAIL %s: %s\n", source.c_str(), error.c_str());
      return 1;
    }
    ccip::RgbImage image;
    image.width = static_cast<int>(entry.member_size("width"));
    image.height = static_cast<int>(entry.member_size("height"));
    image.rgb = tf.Find("rgb")->u8;
    const int w = image.width, h = image.height;

    // 1. Detector input.
    int pw = 0, ph = 0;
    const std::vector<float> det_input = ccip::OcrDetInput(image, &pw, &ph);
    const auto& ref_input = tf.F32("det_input", static_cast<std::int64_t>(3) * pw * ph);
    std::size_t input_diff = 0;
    for (std::size_t i = 0; i < det_input.size(); ++i) input_diff += det_input[i] != ref_input[i];
    if (input_diff != 0) {
      std::printf("FAIL %s: detector input differs in %zu values\n", source.c_str(), input_diff);
      ++failures;
    }

    // 2. Post-processing on the onnxruntime heat map.
    const auto& ref_heat = tf.F32("heat", static_cast<std::int64_t>(w) * h);
    const auto boxes = ccip::TextBoxesFromHeatmap(ref_heat.data(), w, h, options);
    const auto& ref_boxes = entry.find("boxes")->items();
    bool post_ok = boxes.size() == ref_boxes.size();
    for (std::size_t k = 0; post_ok && k < boxes.size(); ++k) {
      const auto& rp = ref_boxes[k].find("points")->items();
      for (int c = 0; c < 4; ++c)
        for (int d = 0; d < 2; ++d)
          if (boxes[k].points[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)] !=
              static_cast<int>(rp[static_cast<std::size_t>(c)].items()[static_cast<std::size_t>(d)].as_double()))
            post_ok = false;
      if (std::fabs(boxes[k].score - ref_boxes[k].member_double("score")) > 1e-9) post_ok = false;
    }
    if (!post_ok) {
      std::printf("FAIL %s: post-processing on the reference heat map: %zu boxes, imgutils %zu\n",
                  source.c_str(), boxes.size(), ref_boxes.size());
      for (std::size_t k = 0; k < std::max(boxes.size(), ref_boxes.size()); ++k) {
        std::printf("   %zu native:", k);
        if (k < boxes.size())
          for (const auto& p : boxes[k].points) std::printf(" (%d,%d)", p[0], p[1]);
        if (k < boxes.size()) std::printf(" %.9f", boxes[k].score);
        std::printf("\n   %zu imgutils:", k);
        if (k < ref_boxes.size()) {
          for (const auto& p : ref_boxes[k].find("points")->items())
            std::printf(" (%g,%g)", p.items()[0].as_double(), p.items()[1].as_double());
          std::printf(" %.9f", ref_boxes[k].member_double("score"));
        }
        std::printf("\n");
      }
      ++failures;
    }

    // 3. Recognizer input images.
    for (std::size_t k = 0; k < ref_boxes.size(); ++k) {
      const auto* ref_crop = tf.Find("crop_" + std::to_string(k));
      std::array<int, 4> rect{};
      const auto& rb = ref_boxes[k].find("box")->items();
      for (int c = 0; c < 4; ++c) rect[static_cast<std::size_t>(c)] = static_cast<int>(rb[static_cast<std::size_t>(c)].as_double());
      const ccip::RgbImage crop = ccip::OcrTextCrop(image, rect, options.rotation_threshold);
      if (ref_crop == nullptr) {
        if (crop.width != 0) {
          std::printf("FAIL %s: box %zu has a crop, imgutils none\n", source.c_str(), k);
          ++failures;
        }
        continue;
      }
      if (ref_crop->shape.size() != 3 || ref_crop->shape[0] != crop.height ||
          ref_crop->shape[1] != crop.width || ref_crop->u8 != crop.rgb) {
        std::size_t diff = 0;
        if (ref_crop->u8.size() == crop.rgb.size())
          for (std::size_t i = 0; i < crop.rgb.size(); ++i) diff += ref_crop->u8[i] != crop.rgb[i];
        std::printf("FAIL %s: recognizer input %zu differs (%dx%d vs %lldx%lld, %zu bytes)\n",
                    source.c_str(), k, crop.width, crop.height,
                    static_cast<long long>(ref_crop->shape[1]),
                    static_cast<long long>(ref_crop->shape[0]), diff);
        ++failures;
      }
    }

    // 4. Native heat map (reported) and the whole pipeline.
    const std::vector<float> heat = ocr->Heatmap(image);
    double max_heat = 0.0;
    std::size_t flips = 0;
    for (std::size_t i = 0; i < heat.size(); ++i) {
      max_heat = std::max(max_heat, static_cast<double>(std::fabs(heat[i] - ref_heat[i])));
      flips += (heat[i] >= 0.3f) != (ref_heat[i] >= 0.3f);
    }
    worst_heat = std::max(worst_heat, max_heat);
    const auto t0 = std::chrono::steady_clock::now();
    const auto results = ocr->Run(image, options);
    const double secs = Seconds(t0, std::chrono::steady_clock::now());
    native_total += secs;
    imgutils_total += entry.member_double("imgutils_seconds");
    ++images;
    const auto& ref = entry.find("ocr")->items();
    std::printf("%s %dx%d: %zu texts (imgutils %zu), native %.3f s (imgutils %.3f s), heat max |d| %.2e, "
                "%zu bitmap flips\n",
                source.c_str(), w, h, results.size(), ref.size(), secs,
                entry.member_double("imgutils_seconds"), max_heat, flips);
    if (results.size() != ref.size()) {
      std::printf("FAIL %s: %zu texts, imgutils %zu\n", source.c_str(), results.size(), ref.size());
      ++failures;
    }
    // Match each imgutils result with the native one that has the same text
    // and the nearest rectangle (orders can swap when scores are close).
    std::vector<bool> used(results.size(), false);
    for (const json::Value& r : ref) {
      const std::string text = r.member_str("text");
      const auto& rb = r.find("box")->items();
      int best = -1;
      double best_d = 1e9;
      for (std::size_t k = 0; k < results.size(); ++k) {
        if (used[k] || results[k].text != text) continue;
        double d = 0;
        for (int c = 0; c < 4; ++c)
          d = std::max(d, std::fabs(results[k].box[static_cast<std::size_t>(c)] - rb[static_cast<std::size_t>(c)].as_double()));
        if (d < best_d) {
          best_d = d;
          best = static_cast<int>(k);
        }
      }
      const double ref_score = r.member_double("score");
      if (best < 0) {
        std::printf("FAIL %s: text \"%s\" (score %.4f) not found natively\n", source.c_str(),
                    text.c_str(), ref_score);
        ++failures;
        continue;
      }
      used[static_cast<std::size_t>(best)] = true;
      const double ds = std::fabs(results[static_cast<std::size_t>(best)].score - ref_score);
      worst_box = std::max(worst_box, best_d);
      worst_score = std::max(worst_score, ds);
      const bool ok = best_d <= 2.0 && ds <= 1e-3;
      std::printf("  %s \"%s\" box d=%.0f px, score %.6f vs %.6f (d=%.2e)\n", ok ? "ok  " : "FAIL",
                  text.c_str(), best_d, results[static_cast<std::size_t>(best)].score, ref_score, ds);
      if (!ok) ++failures;
    }
    for (std::size_t k = 0; k < results.size(); ++k)
      if (!used[k])
        std::printf("  extra native text \"%s\" score %.4f\n", results[k].text.c_str(), results[k].score);
  }
  std::printf("summary: %d images, native %.3f s/image, imgutils %.3f s/image, worst heat |d| %.2e, "
              "worst box %.0f px, worst score %.2e\n",
              images, images ? native_total / images : 0.0, images ? imgutils_total / images : 0.0,
              worst_heat, worst_box, worst_score);
  if (failures != 0) {
    std::printf("FAIL: %d problem(s)\n", failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
