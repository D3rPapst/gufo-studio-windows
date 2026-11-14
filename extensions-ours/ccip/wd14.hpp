// WD14 tagger (SmilingWolf wd-swinv2-tagger-v3, as dghs-imgutils loads it from
// deepghs/wd14_tagger_with_embeddings) natively on the CPU, fp32, OpenMP:
// /v1/vision/detect "wd14_tags". Weights from reference/convert_wd14.py.
//
// imgutils.tagging.get_wd14_tags (0.19.0) step by step:
//  1. The RGB image centred on a white max(w, h) square (offsets
//     (max - w) // 2, (max - h) // 2), Pillow BICUBIC to 448 x 448 unless the
//     square already has that size, float32 0..255 in BGR order (HWC).
//  2. SwinV2-B (timm swinv2_base_window8_256 at 448, window 14): 4x4 patch
//     embedding + LayerNorm; stages 2/2/18/2 at 128/256/512/1024 channels with
//     4/8/16/32 heads, patch merging before stages 1-3; post-norm blocks with
//     cosine window attention (q and k L2-normalised, logit scale, relative
//     position bias from the continuous-position MLP, constant-folded in the
//     export), shifted windows (shift 7, -100 mask) in every odd block of
//     stages 0-2, GELU (tanh) MLP; final LayerNorm, mean pool -> 1024
//     embedding, linear 10861 + sigmoid.
//  3. Tags in selected_tags.csv order: rating (category 9) always, general
//     (0) with p > general_threshold (0.35), character (4) with p >
//     character_threshold (0.85); thresholds compared in double like Python.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_CCIP_WD14_HPP_
#define GUFO_MODELS_CCIP_WD14_HPP_

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ccip/image_ops.hpp"

namespace gufo::models::ccip {

inline constexpr const char* kWd14Stem = "wd-swinv2-tagger-v3";
inline constexpr double kWd14GeneralThreshold = 0.35;
inline constexpr double kWd14CharacterThreshold = 0.85;

// (tag, probability) in selected_tags.csv order, like imgutils' dicts.
using TagList = std::vector<std::pair<std::string, float>>;

struct Wd14Tags {
  TagList rating;
  TagList general;
  TagList character;
};

class Wd14Tagger {
public:
  virtual ~Wd14Tagger() = default;
  // `dir` holds <stem>.safetensors + <stem>.json (tags included).
  [[nodiscard]] static std::unique_ptr<Wd14Tagger> Load(const std::filesystem::path& dir,
                                                        const std::string& stem,
                                                        std::string* error);
  [[nodiscard]] virtual int InputSize() const = 0;
  // Probabilities (classes) and the pooled embedding (features) for a
  // size x size x 3 BGR 0..255 input (step 1's output).
  virtual void Forward(const float* bgr_hwc, std::vector<float>* probs,
                       std::vector<float>* embedding) const = 0;
  // Thresholded tags from probabilities (step 3).
  [[nodiscard]] virtual Wd14Tags Select(const std::vector<float>& probs,
                                        double general_threshold,
                                        double character_threshold) const = 0;
  [[nodiscard]] Wd14Tags Tag(const RgbImage& image,
                             double general_threshold = kWd14GeneralThreshold,
                             double character_threshold = kWd14CharacterThreshold) const;
};

// Step 1 (imgutils _prepare_image_for_tagging) for an RGB image.
[[nodiscard]] std::vector<float> Wd14Input(const RgbImage& image, int size = 448);

// The film pipeline's person cut-out (erkennung.py): margin * max(w, h)
// added on every side, clipped to the image, then Pillow crop's rounding
// (Python round, half to even) of the float box. Returns x0, y0, x1, y1.
[[nodiscard]] std::array<int, 4> MarginCropBox(const std::array<double, 4>& box, double margin,
                                               int width, int height);

}  // namespace gufo::models::ccip

#endif  // GUFO_MODELS_CCIP_WD14_HPP_
