// "ocr" recognizer latin_v5 (PP-OCRv5 latin mobile) against onnxruntime
// (reference/make_ocr_latin_oracle.py). Env: GUFO_OCR_DIR (ch_PP-OCRv4_det.*,
// latin_PP-OCRv5_mobile_rec.* and, for the comparison, ch_PP-OCRv4_rec.* from
// reference/convert_ocr.py), GUFO_OCR_LATIN_ORACLE (dir with manifest.json).
// Exit 77 (skip) without them.
// Checks:
//   - the native graph (HardSwish, Unsqueeze, opset-14 Squeeze/Softmax) gives
//     onnxruntime's logits on every German line (max |diff| <= 2e-3) and the
//     same greedy text with repeats merged,
//   - the true text with umlauts and ß comes out on at least 29 of 30 lines,
//   - the whole pipeline (v4 detector + latin_v5, remove_duplicate) reads
//     all but at most one line of the rendered signs exactly; ch_v4 on the
//     same pages is the "before" number (it reads none: no umlauts, no ß).
// Written with AI assistance (Claude Code).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gufo/ext/json.hpp>

#include "ccip/ccip.hpp"
#include "ccip/cpu/ocr_net.hpp"
#include "ccip/ocr.hpp"
#include "ccip/tensors.hpp"

namespace ccip = gufo::models::ccip;
namespace json = gufo::json;

namespace {

const char* Env(const char* name) {
  const char* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? v : nullptr;
}

ccip::RgbImage Image(const ccip::Tensor& t) {
  ccip::RgbImage im;
  im.height = static_cast<int>(t.shape.at(0));
  im.width = static_cast<int>(t.shape.at(1));
  im.rgb = t.u8;
  return im;
}

// Lines of a page the pipeline read exactly (each found text used once).
int PageHits(const ccip::TextOcr& ocr, const ccip::RgbImage& page, const json::Value& lines,
             bool remove_duplicate, std::string* misses) {
  ccip::OcrOptions options;
  options.remove_duplicate = remove_duplicate;
  std::vector<ccip::OcrText> found = ocr.Run(page, options);
  int hits = 0;
  for (const json::Value& line : lines.items()) {
    auto it = std::find_if(found.begin(), found.end(),
                           [&](const ccip::OcrText& t) { return t.text == line.str(); });
    if (it != found.end()) {
      ++hits;
      found.erase(it);
    } else {
      *misses += " \"" + line.str() + "\"";
    }
  }
  if (hits != static_cast<int>(lines.size())) {
    *misses += " | read:";
    for (const ccip::OcrText& t : found) *misses += " \"" + t.text + "\"";
  }
  return hits;
}

}  // namespace

int main() {
  const char* model_dir = Env("GUFO_OCR_DIR");
  const char* oracle_dir = Env("GUFO_OCR_LATIN_ORACLE");
  if (!model_dir || !oracle_dir) {
    std::printf("skip: set GUFO_OCR_DIR and GUFO_OCR_LATIN_ORACLE\n");
    return 77;
  }
  ccip::SetCpuThreads(4);
  const std::string rec = "latin_PP-OCRv5_mobile_rec";
  std::string error;
  auto net = ccip::cpu::OcrNet::Load(model_dir, rec, &error);
  auto ocr = ccip::TextOcr::Load(model_dir, "ch_PP-OCRv4_det", rec, &error);
  if (!net || !ocr) {
    std::printf("FAIL load: %s\n", error.c_str());
    return 1;
  }
  const std::filesystem::path odir(oracle_dir);
  std::ifstream in(odir / "manifest.json", std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  const json::Value manifest = json::parse(ss.str());
  ccip::TensorFile tf;
  if (!tf.Load(odir / manifest.member_str("file"), &error)) {
    std::printf("FAIL oracle: %s\n", error.c_str());
    return 1;
  }
  int failures = 0, right = 0;
  double worst = 0.0;
  const auto& lines = manifest.find("lines")->items();
  for (std::size_t k = 0; k < lines.size(); ++k) {
    const ccip::RgbImage crop = Image(*tf.Find("crop_" + std::to_string(k)));
    const std::vector<float>& want = tf.F32("logits_" + std::to_string(k));
    ccip::cpu::NetTensor x;
    x.shape = {1, 3, crop.height, crop.width};
    const std::size_t plane = static_cast<std::size_t>(crop.width) * crop.height;
    x.f.resize(plane * 3);
    for (std::size_t i = 0; i < plane; ++i)
      for (std::size_t c = 0; c < 3; ++c)
        x.f[c * plane + i] = (static_cast<float>(crop.rgb[i * 3 + c]) / 255.0f - 0.5f) / 0.5f;
    const ccip::cpu::NetTensor y = net->Run(std::move(x));
    double diff = 0.0;
    if (y.f.size() != want.size()) {
      std::printf("FAIL line %zu: %zu logits, onnxruntime %zu\n", k, y.f.size(), want.size());
      ++failures;
      continue;
    }
    for (std::size_t i = 0; i < want.size(); ++i)
      diff = std::max(diff, static_cast<double>(std::fabs(y.f[i] - want[i])));
    worst = std::max(worst, diff);
    const auto [text, score] = ocr->Recognize(crop, true);
    const std::string truth = lines[k].member_str("text");
    const std::string ort_text = lines[k].member_str("ort_text");
    const bool same = text == ort_text && diff <= 2e-3;
    right += text == truth;
    std::printf("  %s \"%s\" -> \"%s\" (%.3f, logits d=%.1e)\n", same ? "ok  " : "FAIL",
                truth.c_str(), text.c_str(), score, diff);
    if (!same) ++failures;
  }
  std::printf("lines: %d of %zu exactly right, worst logit diff %.2e\n", right, lines.size(), worst);
  if (right + 1 < static_cast<int>(lines.size())) {
    std::printf("FAIL: fewer than %zu lines right\n", lines.size() - 1);
    ++failures;
  }
  // Whole pipeline on the signs, before (ch_v4) and after (latin_v5).
  std::unique_ptr<ccip::TextOcr> before;
  if (std::filesystem::exists(std::filesystem::path(model_dir) / "ch_PP-OCRv4_rec.dict.txt"))
    before = ccip::TextOcr::Load(model_dir, "ch_PP-OCRv4_det", "ch_PP-OCRv4_rec", &error);
  int hits = 0, hits_before = 0, total = 0;
  const auto& pages = manifest.find("pages")->items();
  for (std::size_t k = 0; k < pages.size(); ++k) {
    const ccip::RgbImage page = Image(*tf.Find("page_" + std::to_string(k)));
    const json::Value& page_lines = *pages[k].find("lines");
    std::string misses, misses_before;
    hits += PageHits(*ocr, page, page_lines, true, &misses);
    if (before) hits_before += PageHits(*before, page, page_lines, false, &misses_before);
    total += static_cast<int>(page_lines.size());
    if (!misses.empty()) std::printf("  page %zu latin_v5 missed:%s\n", k, misses.c_str());
    if (!misses_before.empty()) std::printf("  page %zu ch_v4 missed:%s\n", k, misses_before.c_str());
  }
  std::printf("pages: latin_v5 %d of %d lines", hits, total);
  if (before) std::printf(", ch_v4 %d of %d", hits_before, total);
  std::printf("\n");
  // The v4 detector misses one rendered line (Georgia 30 px on beige; ch_v4
  // finds no box there either): one line of slack, and far ahead of ch_v4.
  if (hits + 1 < total || (before && hits <= hits_before)) ++failures;
  if (failures == 0)
    std::printf("ocr_latin_test: ok\n");
  else
    std::printf("ocr_latin_test: %d failures\n", failures);
  return failures == 0 ? 0 : 1;
}
