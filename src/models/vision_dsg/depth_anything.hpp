// Depth Anything V2 (DINOv2 ViT-S/B/L + DPT head) on the GPU in FP32, read
// straight from a Hugging Face snapshot (config.json + model.safetensors of
// depth-anything/Depth-Anything-V2-{Small,Base,Large}-hf), as transformers
// 5.18 DepthAnythingForDepthEstimation with DPTImageProcessorPil:
//  - the image is scaled to fit 518 x 518 keeping the aspect ratio (the side
//    that changes least decides), both sides rounded half to even to a
//    multiple of 14, with Pillow's BICUBIC filter on RGB8, then x / 255 and
//    ImageNet mean / std;
//  - DINOv2 with bicubic-interpolated position embeddings, the
//    layer-normalised hidden states of four layers (config out_indices);
//  - DPT reassemble (1x1, transposed / strided convolutions), 3x3 convs,
//    four fusion stages (pre-activation residual units, bilinear
//    align_corners upsampling), head conv / bilinear to 14 x patches / conv /
//    ReLU / 1x1 / ReLU (relative depth, larger = nearer);
//  - `depth` is predicted_depth bicubic-interpolated to the image size
//    (align_corners false), as the film pipeline's hf.py does.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_VISION_DSG_DEPTH_ANYTHING_HPP_
#define GUFO_MODELS_VISION_DSG_DEPTH_ANYTHING_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace gufo::models::vision_dsg {

struct DepthResult {
  int width{0}, height{0};
  std::vector<float> depth;  // [height, width]
  int net_width{0}, net_height{0};
  std::vector<float> raw;  // predicted_depth [net_height, net_width]
};

/// Network input size for an image (DPT keep_aspect_ratio, multiple of 14).
void DepthInputSize(int width, int height, int* net_width, int* net_height);
/// pixel_values [3, net_height, net_width] as DPTImageProcessorPil makes them.
std::vector<float> DepthPixels(const std::uint8_t* rgb, int width, int height,
                               int* net_width, int* net_height);

class DepthAnything {
public:
  [[nodiscard]] static std::unique_ptr<DepthAnything> Load(
      const std::filesystem::path& dir, std::string* error);
  ~DepthAnything();
  DepthAnything(const DepthAnything&) = delete;
  DepthAnything& operator=(const DepthAnything&) = delete;

  /// `rgb` is packed RGB8 [height, width, 3].
  bool Run(const std::uint8_t* rgb, int width, int height, DepthResult* out,
           std::string* error);
  /// The network on given pixel_values [3, h, w] (h, w multiples of 14);
  /// fills `raw`, net size and, with width/height > 0, `depth`.
  bool RunPixels(const float* pixels, int net_height, int net_width,
                 int width, int height, DepthResult* out, std::string* error);

  /// "small", "base" or "large" (from hidden_size).
  [[nodiscard]] std::string size_name() const;
  [[nodiscard]] std::size_t weight_bytes() const;

private:
  struct Impl;
  explicit DepthAnything(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::vision_dsg

#endif  // GUFO_MODELS_VISION_DSG_DEPTH_ANYTHING_HPP_
