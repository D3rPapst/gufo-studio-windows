// Image preprocessing for CCIP and the anime face detector, identical to the
// Python reference (deepghs imgutils / ccip_ref.py on Pillow):
//  - ResizeBilinear reproduces Pillow's Image.resize(..., BILINEAR) for RGB8
//    bit for bit: separable triangle filter whose support grows with the
//    downscale factor (antialiasing), fixed-point coefficients with 22
//    fraction bits, uint8 rounding after each pass; horizontal pass first
//    unless the height shrinks by more than twice the width's pixel loss
//    (Pillow 10+ rule; verified against Pillow 13.0.0).
//  - CcipInput: resize to 384x384, x/255, (x-0.5)/0.5, CHW float32.
//  - Letterbox: resize by min(640/w, 640/h) (Python round()), centred on a
//    (114,114,114) canvas, x/255, CHW float32.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_CCIP_IMAGE_OPS_HPP_
#define GUFO_MODELS_CCIP_IMAGE_OPS_HPP_

#include <array>
#include <cstdint>
#include <vector>

namespace gufo::models::ccip {

struct RgbImage {
  int width{0};
  int height{0};
  std::vector<std::uint8_t> rgb;  // packed RGB8, row-major
};

[[nodiscard]] RgbImage ResizeBilinear(const RgbImage& in, int out_width, int out_height);
// Pillow Image.resize(..., BICUBIC) (a = -0.5, support 2; the Pillow >= 7
// default of Image.resize), same fixed-point path as ResizeBilinear.
[[nodiscard]] RgbImage ResizeBicubic(const RgbImage& in, int out_width, int out_height);
// Pillow Image.resize(..., LANCZOS) (sinc windowed by sinc, a = 3, support
// 3), same fixed-point path: the film pipeline's 3x enlargement before OCR
// (erkennung_schrift.py).
[[nodiscard]] RgbImage ResizeLanczos(const RgbImage& in, int out_width, int out_height);

// Pillow ImageStat.Stat(image.convert("L")).mean[0]: L = (R*19595 +
// G*38470 + B*7471 + 0x8000) >> 16 per pixel, averaged.
[[nodiscard]] double MeanLuma(const RgbImage& image);
// image.point(lambda v: round(255 * (v / 255) ** gamma)) on every channel
// (Python round, half to even): the film pipeline's brightening of dark
// keyframes before detection (gamma 0.5 below mean luma 70).
[[nodiscard]] RgbImage ApplyGamma(const RgbImage& image, double gamma);

// imgutils generic YOLO input (allow_dynamic=False): stretched to
// width x height with ResizeBicubic (no letterbox), (x / 255.0) as float32,
// 3 x height x width.
[[nodiscard]] std::vector<float> StretchInput(const RgbImage& image, int width, int height);

// Pillow Image.crop for a box inside the image (x1/y1 exclusive); parts
// outside the image are black like Pillow's.
[[nodiscard]] RgbImage Crop(const RgbImage& in, int x0, int y0, int x1, int y1);

inline constexpr int kCcipSize = 384;
// 3 x 384 x 384, normalised to [-1, 1].
[[nodiscard]] std::vector<float> CcipInput(const RgbImage& image);

struct LetterboxInput {
  std::vector<float> chw;  // 3 x size x size, [0, 1]
  int size{640};
  double ratio{1.0};
  int dx{0};
  int dy{0};
};
[[nodiscard]] LetterboxInput Letterbox(const RgbImage& image, int size = 640);

// ccip_ref.head_crop: face box grown to a square of max(w, h) * grow around
// its centre, clamped to the image; integer bounds truncated like int().
[[nodiscard]] std::array<int, 4> HeadCropBox(const std::array<float, 4>& face, int width,
                                             int height, double grow = 1.6);

}  // namespace gufo::models::ccip

#endif  // GUFO_MODELS_CCIP_IMAGE_OPS_HPP_
