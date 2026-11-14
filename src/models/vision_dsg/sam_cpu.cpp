// SamImageProcessorPil of transformers 5.18 (see sam.hpp): longest side to
// 1024, Pillow BILINEAR, float32(v * (1 / 255.0)), (x - mean) / std, zero
// padding to 1024 x 1024 - bit for bit.
// Written with AI assistance (Claude Code).
#include <algorithm>

#include "ccip/image_ops.hpp"
#include "src/models/vision_dsg/sam.hpp"

namespace gufo::models::vision_dsg {

void SamInputSize(int width, int height, int* net_width, int* net_height) {
  const double scale = 1024 * 1.0 / std::max(width, height);
  *net_height = static_cast<int>(height * scale + 0.5);
  *net_width = static_cast<int>(width * scale + 0.5);
}

std::vector<float> SamPixels(const std::uint8_t* rgb, int width, int height,
                             int* net_width, int* net_height) {
  if (rgb == nullptr || width < 1 || height < 1)
    return {};
  SamInputSize(width, height, net_width, net_height);
  ccip::RgbImage image;
  image.width = width;
  image.height = height;
  image.rgb.assign(rgb, rgb + static_cast<std::size_t>(width) * height * 3);
  const ccip::RgbImage resized =
      (*net_width == width && *net_height == height)
          ? image
          : ccip::ResizeBilinear(image, *net_width, *net_height);
  constexpr float kMean[3] = {0.485F, 0.456F, 0.406F};
  constexpr float kStd[3] = {0.229F, 0.224F, 0.225F};
  constexpr int kSide = 1024;
  std::vector<float> out(static_cast<std::size_t>(3) * kSide * kSide, 0.0F);
  for (int y = 0; y < *net_height; ++y)
    for (int x = 0; x < *net_width; ++x)
      for (int c = 0; c < 3; ++c) {
        const std::size_t i = static_cast<std::size_t>(y) * *net_width + x;
        const float v = static_cast<float>(resized.rgb[i * 3 + c] * (1.0 / 255.0));
        out[(static_cast<std::size_t>(c) * kSide + y) * kSide + x] = (v - kMean[c]) / kStd[c];
      }
  return out;
}

}  // namespace gufo::models::vision_dsg
