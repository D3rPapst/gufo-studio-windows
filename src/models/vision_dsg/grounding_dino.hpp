// Grounding DINO (IDEA-Research/grounding-dino-base: Swin-B, BERT-base,
// six fusion encoder and six deformable decoder layers) on the GPU in FP32,
// read straight from the Hugging Face snapshot, as transformers 5.18
// GroundingDinoForObjectDetection with GroundingDinoProcessor
// (GroundingDinoImageProcessorPil + BertTokenizer) and
// post_process_grounded_object_detection. One image, one prompt, no padding.
//  - image: shortest side 800 / longest 1333 (transformers'
//    get_size_with_aspect_ratio), Pillow BILINEAR on RGB8, x / 255, ImageNet
//    mean / std;
//  - text: "a. b. c." lower-cased BERT WordPiece with [CLS] / [SEP], the
//    phrase-block self-attention mask and per-phrase position ids;
//  - answer: queries whose best token probability is > box_threshold, label
//    = the decoded tokens (positions 1 .. 254) with probability >
//    text_threshold, boxes in image pixels (x0, y0, x1, y1).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_VISION_DSG_GROUNDING_DINO_HPP_
#define GUFO_MODELS_VISION_DSG_GROUNDING_DINO_HPP_

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gufo::models::vision_dsg {

/// BERT uncased WordPiece (BasicTokenizer + WordPiece of vocab.txt).
class BertTokenizer {
public:
  bool Load(const std::filesystem::path& vocab, std::string* error);
  /// [CLS] tokens [SEP] ids.
  [[nodiscard]] std::vector<int> Encode(const std::string& text) const;
  /// transformers decode (no special-token skipping, clean-up spaces).
  [[nodiscard]] std::string Decode(const std::vector<int>& ids) const;

private:
  std::map<std::string, int, std::less<>> vocab_;
  std::vector<std::string> tokens_;
};

/// generate_masks_with_special_tokens_and_transfer_map: mask [n * n] (1 =
/// may attend) and position ids.
void GroundingTextMasks(const std::vector<int>& ids, std::vector<float>* mask,
                        std::vector<int>* positions);

/// GroundingDinoImageProcessorPil size and pixel_values [3, h, w].
void GroundingInputSize(int width, int height, int* net_width, int* net_height);
std::vector<float> GroundingPixels(const std::uint8_t* rgb, int width,
                                   int height, int* net_width, int* net_height);

struct GroundingBox {
  std::array<float, 4> box{};
  float score{0};
  std::string label;
};

struct GroundingRaw {
  std::vector<float> logits;  // [900, ntokens]
  std::vector<float> boxes;   // [900, 4] cx, cy, w, h (sigmoid)
  std::vector<int> ids;
};

class GroundingDino {
public:
  [[nodiscard]] static std::unique_ptr<GroundingDino> Load(
      const std::filesystem::path& dir, std::string* error);
  ~GroundingDino();
  GroundingDino(const GroundingDino&) = delete;
  GroundingDino& operator=(const GroundingDino&) = delete;

  /// `text` as hf.py builds it ("person. sword.").
  bool Run(const std::uint8_t* rgb, int width, int height,
           const std::string& text, float box_threshold, float text_threshold,
           std::vector<GroundingBox>* boxes, GroundingRaw* raw,
           std::string* error);
  [[nodiscard]] std::size_t weight_bytes() const;

private:
  struct Impl;
  explicit GroundingDino(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::vision_dsg

#endif  // GUFO_MODELS_VISION_DSG_GROUNDING_DINO_HPP_
