// DPTImageProcessorPil of transformers 5.18 for Depth Anything V2 (see
// depth_anything.hpp): fit 518 x 518 keeping the aspect ratio, multiples of
// 14 (Python round, half to even), Pillow BICUBIC on RGB8, then
// float32(v * (1 / 255.0)) and (x - mean) / std in float32 - bit for bit.
// Written with AI assistance (Claude Code).
#include <cmath>

#include "ccip/image_ops.hpp"
#include "src/models/vision_dsg/depth_anything.hpp"

namespace gufo::models::vision_dsg {
namespace {

constexpr int kSide = 518;
constexpr int kMultiple = 14;

int MultipleOf(double value) {
  const int x = static_cast<int>(std::nearbyint(value / kMultiple)) * kMultiple;
  return x < kMultiple ? kMultiple : x;
}

}  // namespace

void DepthInputSize(int width, int height, int* net_width, int* net_height) {
  double scale_h = static_cast<double>(kSide) / height;
  double scale_w = static_cast<double>(kSide) / width;
  if (std::abs(1 - scale_w) < std::abs(1 - scale_h))
    scale_h = scale_w;
  else
    scale_w = scale_h;
  *net_height = MultipleOf(scale_h * height);
  *net_width = MultipleOf(scale_w * width);
}

std::vector<float> DepthPixels(const std::uint8_t* rgb, int width, int height,
                               int* net_width, int* net_height) {
  if (rgb == nullptr || width < 1 || height < 1)
    return {};
  DepthInputSize(width, height, net_width, net_height);
  ccip::RgbImage image;
  image.width = width;
  image.height = height;
  image.rgb.assign(rgb, rgb + static_cast<std::size_t>(width) * height * 3);
  const ccip::RgbImage resized =
      (*net_width == width && *net_height == height)
          ? image
          : ccip::ResizeBicubic(image, *net_width, *net_height);
  constexpr float kMean[3] = {0.485F, 0.456F, 0.406F};
  constexpr float kStd[3] = {0.229F, 0.224F, 0.225F};
  const std::size_t plane = static_cast<std::size_t>(*net_width) * *net_height;
  std::vector<float> out(plane * 3);
  for (std::size_t i = 0; i < plane; ++i)
    for (int c = 0; c < 3; ++c) {
      const float v = static_cast<float>(resized.rgb[i * 3 + c] * (1.0 / 255.0));
      out[c * plane + i] = (v - kMean[c]) / kStd[c];
    }
  return out;
}

}  // namespace gufo::models::vision_dsg
