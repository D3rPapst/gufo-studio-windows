// MiniMax-H3-Image edits: the host-side half of the "image_conditioned"
// recipe of the release (diffusers modular_pipelines/minimax_h3_image):
// reference sizing for the VAE and the vision tower, the text-encoder
// presentation ("Picture i: " + vision block per reference, then the prompt)
// and the conditioning latent of every reference.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MINIMAX_H3_IMAGE_EDIT_HPP_
#define GUFO_MODELS_MINIMAX_H3_IMAGE_EDIT_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "src/models/minimax_h3/sampling.hpp"

namespace gufo::minimax_h3 {

// MiniMaxH3ImageEditResizeStep config: the VAE copy keeps its size while its
// short edge is within [1024, 2048], the vision copy is brought to 1024.
inline constexpr int kH3ImageReferenceMinimumShortEdge = 1024;
inline constexpr int kH3ImageReferenceMaximumShortEdge = 2048;
inline constexpr int kH3ImageVisionShortEdge = 1024;
// max_sequence_length: the prompt and every label are cut to it.
inline constexpr std::size_t kH3ImageEditMaximumTokens = 4096;
// Qwen2VLImageProcessorFast of the release (processor/preprocessor_config).
inline constexpr std::int64_t kH3ImageVisionMaximumPixels = 16777216;
inline constexpr std::int64_t kH3ImageVisionMinimumPixels = 65536;

/// bound_short_edge: a short edge within [minimum, maximum] keeps the size;
/// otherwise the image scales (ratio kept) so that its short edge lands on
/// the nearer bound, both sides rounded half-to-even to `multiple` (at least
/// one multiple). Returns (width, height).
[[nodiscard]] std::pair<int, int> BoundShortEdge(int width, int height,
                                                 int minimum, int maximum,
                                                 int multiple);

/// One uploaded reference, already upright (the host applies EXIF).
struct ImageEditInput {
  int width{0};
  int height{0};
  std::vector<std::uint8_t> rgb;  // interleaved RGB8, width * height * 3
};

/// A reference as the model consumes it.
struct ImageEditReference {
  // The VAE copy, cropped at the top left to whole 16-pixel VAE cells.
  int vae_width{0};
  int vae_height{0};
  std::vector<std::uint8_t> vae_rgb;
  // The conditioning latent: the VAE grid cropped to whole 2x2 patches.
  int latent_width{0};
  int latent_height{0};
  // The vision-tower copy (both sides multiples of 32).
  int vision_width{0};
  int vision_height{0};
  std::vector<std::uint8_t> vision_rgb;
};

/// MiniMaxH3ImageEditResizeStep plus the VAE encoder's crops: 1 to 16
/// references with at least 32 pixels per side, each resized (PIL Lanczos,
/// no crop) for the VAE and for the vision tower. A vision copy whose long
/// side is not a multiple of 32 (short edge exactly 1024) is rounded to the
/// nearest multiple as Qwen's smart_resize does.
[[nodiscard]] bool PlanImageEditReferences(
    std::span<const ImageEditInput> inputs,
    std::vector<ImageEditReference>* references, std::string* error = nullptr);

/// The DiT placement of a planned reference (one image frame).
[[nodiscard]] ReferencePlacement ImageEditPlacement(
    const ImageEditReference& reference) noexcept;

/// The "Picture i: " label of reference `index` (0-based); the prompt refers
/// to it as "<Picture i>".
[[nodiscard]] std::string ImageEditLabel(std::size_t index);

/// Where a vision block sits in the presentation and its merged grid.
struct ImageEditVisionSlot {
  std::size_t begin{0};  // first <|image_pad|> row
  std::uint32_t grid_height{0};
  std::uint32_t grid_width{0};
};

struct ImageEditPresentation {
  std::vector<std::uint32_t> tokens;
  // 1 text (labels, prompt), 0 image (a vision block with its start/end).
  std::vector<std::uint8_t> tags;
  std::vector<ImageEditVisionSlot> slots;
  std::size_t vision_tokens{0};
  bool prompt_truncated{false};
};

/// MiniMaxH3ImageEditTextEncoderStep's token layout: per reference its label
/// tokens (cut to 4096), <|vision_start|>, (h / 32) * (w / 32)
/// <|image_pad|>, <|vision_end|>; then the prompt tokens (cut to 4096).
/// `label_tokens` and `vision_sizes` ((width, height) of the vision copies)
/// are in reference order.
[[nodiscard]] bool BuildImageEditPresentation(
    std::span<const std::vector<std::uint32_t>> label_tokens,
    std::span<const std::pair<int, int>> vision_sizes,
    std::span<const std::uint32_t> prompt_tokens,
    ImageEditPresentation* presentation, std::string* error = nullptr);

/// latents_mean / latents_std of the image VAE (vae/config.json).
struct ImageVaeStatistics {
  std::array<float, 24> mean{};
  std::array<float, 24> deviation{};
};
[[nodiscard]] bool LoadImageVaeStatistics(const std::filesystem::path& model_root,
                                          ImageVaeStatistics* statistics,
                                          std::string* error = nullptr);

/// MiniMaxH3ImageEditVaeEncoderStep + EditPrepareLatentsStep for one
/// reference with given noise: `moments` are the 48 posterior channels over
/// moment_height x moment_width (means, then log variances clamped to
/// [-30, 20]); `posterior_noise` has 24 x moment_height x moment_width
/// values, `anchor_noise` 24 x latent_height x latent_width. The posterior
/// sample is rounded to float16, cropped at the top left to the even
/// latent_height x latent_width, normalized and held at t = 0.999:
/// 0.999 * x + (1 - 0.999) * noise in float32. Writes [24][h][w].
[[nodiscard]] bool BuildImageEditConditioningFromNoise(
    std::span<const float> moments, int moment_height, int moment_width,
    int latent_height, int latent_width, const ImageVaeStatistics& statistics,
    std::span<const float> posterior_noise, std::span<const float> anchor_noise,
    std::vector<float>* latent, std::string* error = nullptr);

/// The same with gufo's NormalRng streams: posterior noise under
/// `posterior_seed`, anchor noise under `anchor_seed` (channel-major fills).
[[nodiscard]] bool BuildImageEditConditioning(
    std::span<const float> moments, int moment_height, int moment_width,
    int latent_height, int latent_width, const ImageVaeStatistics& statistics,
    std::uint64_t posterior_seed, std::uint64_t anchor_seed,
    std::vector<float>* latent, std::string* error = nullptr);

/// Seeds of reference `index` (0-based) derived from the request seed; the
/// generated image keeps the plain seed (as text to image).
[[nodiscard]] std::uint64_t ImageEditPosteriorSeed(std::uint64_t seed,
                                                   std::size_t index) noexcept;
[[nodiscard]] std::uint64_t ImageEditAnchorSeed(std::uint64_t seed,
                                                std::size_t index) noexcept;

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_IMAGE_EDIT_HPP_
