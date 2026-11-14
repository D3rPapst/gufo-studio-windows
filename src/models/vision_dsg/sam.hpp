// SAM (facebook/sam-vit-huge: ViT-H image encoder with windowed and global
// attention and decomposed relative positions, prompt encoder, two-way mask
// decoder) on the GPU in FP32, read straight from the Hugging Face snapshot,
// as transformers 5.18 SamModel with SamProcessor (SamImageProcessorPil):
//  - longest side to 1024 (int(x * scale + 0.5)), Pillow BILINEAR on RGB8,
//    x / 255, ImageNet mean / std, zero padding to 1024 x 1024;
//  - prompts in image pixels, scaled to the resized image like
//    SamProcessor; each box (or the one point set) is decoded on its own;
//  - post_process_masks: bilinear to 1024, crop, bilinear to the image size,
//    > 0.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_VISION_DSG_SAM_HPP_
#define GUFO_MODELS_VISION_DSG_SAM_HPP_

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace gufo::models::vision_dsg {

/// SamImageProcessorPil size and pixel_values [3, 1024, 1024] (padded).
void SamInputSize(int width, int height, int* net_width, int* net_height);
std::vector<float> SamPixels(const std::uint8_t* rgb, int width, int height,
                             int* net_width, int* net_height);

struct SamPrompt {
  std::vector<std::array<float, 4>> boxes;   // image pixels
  std::vector<std::array<float, 2>> points;  // image pixels
  std::vector<int> labels;                   // per point: 1 / 0
  bool multimask{false};
};

struct SamOutput {
  int width{0}, height{0};
  std::vector<std::vector<std::uint8_t>> masks;  // [height * width], 0 / 1
  std::vector<float> scores;                     // predicted IoU
  std::vector<std::vector<float>> low_res;       // [256 * 256] logits
};

class Sam {
public:
  [[nodiscard]] static std::unique_ptr<Sam> Load(const std::filesystem::path& dir,
                                                 std::string* error);
  ~Sam();
  Sam(const Sam&) = delete;
  Sam& operator=(const Sam&) = delete;

  bool Run(const std::uint8_t* rgb, int width, int height, const SamPrompt& prompt,
           SamOutput* out, std::string* error);
  [[nodiscard]] std::size_t weight_bytes() const;

private:
  struct Impl;
  explicit Sam(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::vision_dsg

#endif  // GUFO_MODELS_VISION_DSG_SAM_HPP_
