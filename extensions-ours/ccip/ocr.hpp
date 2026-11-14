// Text recognition (OCR) for /v1/vision/detect, task "ocr": dghs-imgutils
// imgutils.ocr.ocr with its defaults, natively on the CPU (no Python, no
// ONNX runtime). Models: PaddleOCR PP-OCRv4 mobile from deepghs/paddleocr
// (det/ch_PP-OCRv4_det: DBNet with a PP-LCNetV3 backbone; rec/ch_PP-OCRv4_rec:
// SVTR-LCNet with a CTC head over 6625 symbols, Chinese + Latin), converted by
// reference/convert_ocr.py and run by cpu/ocr_net.cpp.
//
// Pipeline (imgutils 0.19.0, ocr/detect.py + ocr/recognize.py):
//  1. Detector input: the full image (no resize), padded right/bottom with 0
//     to multiples of 64, x/255, then normalised with the CLIP mean/std
//     (an imgutils choice, not PaddleOCR's ImageNet values).
//  2. Heat map cropped to the image; bitmap = heat >= 0.3; contours with
//     OpenCV findContours(RETR_LIST, CHAIN_APPROX_SIMPLE); per contour (at
//     most 1000) minAreaRect (short side >= 3), score = mean heat inside the
//     box (fillPoly mask), dropped below 0.7; the box grown by
//     area * 2.0 / perimeter with a Clipper round offset, minAreaRect again
//     (short side >= 5), corners rounded and clipped to [0, W] x [0, H].
//  3. Per box: Pillow crop of the axis-aligned rectangle, rotated by 90
//     degrees (Pillow rotate without expand) when height >= 1.5 * width,
//     resized to height 48 (Pillow BICUBIC), (x/255 - 0.5)/0.5; CTC greedy
//     decode (blank dropped, repeats kept like imgutils' default), text score
//     = mean of the per-step maxima.
//  4. score = det score * text score; results sorted by score, descending.
// The contour, rotated-rectangle, polygon-fill and offset steps follow the
// OpenCV 4.11 and Clipper 6.4.2 sources step by step (ocr_post.cpp), so the
// boxes match imgutils on the same heat map.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_CCIP_OCR_HPP_
#define GUFO_MODELS_CCIP_OCR_HPP_

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ccip/image_ops.hpp"

namespace gufo::models::ccip {

// imgutils.ocr.ocr defaults.
struct OcrOptions {
  double heat_threshold{0.3};  // compared in float32 like numpy
  double box_threshold{0.7};
  int max_candidates{1000};
  double unclip_ratio{2.0};
  double rotation_threshold{1.5};
  bool remove_duplicate{false};
};

using Quad = std::array<std::array<int, 2>, 4>;

// One detector box (imgutils _get_text_points): the four corners (image
// pixels, clipped to [0, W] x [0, H]) and the mean heat of the first box.
struct TextBox {
  Quad points{};
  double score{0};
};

struct OcrText {
  Quad points{};
  std::array<int, 4> box{};  // x0, y0, x1, y1 = min/max of the corners
  std::string text;           // UTF-8
  double det_score{0};
  double text_score{0};
  double score{0};            // det_score * text_score
};

// imgutils _boxes_from_bitmap on a heat map (width x height floats,
// row-major), in imgutils' order (OpenCV contour order).
[[nodiscard]] std::vector<TextBox> TextBoxesFromHeatmap(const float* heat, int width, int height,
                                                        const OcrOptions& options);

// Detector input: 1 x 3 x H64 x W64 floats (see step 1). *padded_width and
// *padded_height receive the padded size.
[[nodiscard]] std::vector<float> OcrDetInput(const RgbImage& image, int* padded_width,
                                             int* padded_height);

// The recognizer's input image for one rectangle (x1/y1 exclusive like
// Pillow's crop): crop, 90 degree turn when tall, resize to `height` rows.
// Empty (width 0) when the rectangle has no area.
[[nodiscard]] RgbImage OcrTextCrop(const RgbImage& image, const std::array<int, 4>& box,
                                   double rotation_threshold, int height = 48);

class TextOcr {
public:
  virtual ~TextOcr() = default;
  // `dir` holds <det>.{safetensors,json}, <rec>.{safetensors,json} and
  // <rec>.dict.txt from reference/convert_ocr.py.
  [[nodiscard]] static std::unique_ptr<TextOcr> Load(const std::filesystem::path& dir,
                                                     const std::string& det_stem,
                                                     const std::string& rec_stem,
                                                     std::string* error);
  // Detector heat map cropped to the image (height x width floats).
  [[nodiscard]] virtual std::vector<float> Heatmap(const RgbImage& image) const = 0;
  // Text and text score of a recognizer input (OcrTextCrop result).
  [[nodiscard]] virtual std::pair<std::string, double> Recognize(const RgbImage& crop,
                                                                 bool remove_duplicate) const = 0;
  // The whole imgutils.ocr.ocr. Thread-safe (no mutable state).
  [[nodiscard]] std::vector<OcrText> Run(const RgbImage& image, const OcrOptions& options) const;
};

}  // namespace gufo::models::ccip

#endif  // GUFO_MODELS_CCIP_OCR_HPP_
