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

#include "src/models/minimax_h3/prompt_encoder.hpp"
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

/// The vision-tower copy of a width x height reference: short edge 1024,
/// long side scaled with it and rounded to 32 (Qwen's smart_resize). The
/// presentation spends (width / 32) * (height / 32) image-pad tokens on it.
[[nodiscard]] std::pair<int, int> ImageEditVisionSize(int width, int height);

/// The release puts no limit on the whole presentation (only the prompt and
/// every label are cut to 4096 tokens). gufo's text encoder always takes
/// kPromptEncoderMaximumTokens (4096) tokens in all; more only when enough
/// device memory is free (ImageEditTokenLimit). Every reference costs at
/// least 1024 vision tokens (a square; its short edge is always brought to
/// 1024 px, so smaller images cost the same) plus its "Picture i: " label (5
/// tokens, 6 from Picture 10) and the two vision markers: within 4096 tokens
/// four references never fit, three squares leave about 1000 for the prompt.
inline constexpr std::size_t kH3ImageEditMinimumReferenceTokens = 1024 + 5 + 2;
/// References that always fit (the 4096-token base budget).
inline constexpr std::size_t kH3ImageEditMaximumImages = 3;
static_assert(kH3ImageEditMaximumImages * kH3ImageEditMinimumReferenceTokens <
                  kPromptEncoderMaximumTokens &&
              (kH3ImageEditMaximumImages + 1) *
                      kH3ImageEditMinimumReferenceTokens >
                  kPromptEncoderMaximumTokens);

// --- Memory-based token limit (2.2.17) ------------------------------------
//
// Beyond 4096 tokens an edit is accepted only when both phases fit in the
// device memory that is free when the request is checked, minus a reserve:
// the prompt child (PromptEncoderPhaseBytes) and the image DiT
// (ImageDitPhaseBytes: the resident core plus per-row buffers over
// [text | references | image]). The phases run one after the other, so the
// larger counts; the DiT one always dominates (about 37 GiB fixed).

/// DiT phase bytes that do not grow with the sequence: the resident 50-block
/// core (36,751 MiB of weights), heads, modulations, GEMM workspace and the
/// HIP runtime. Calibrated on gfx1151 21.11.2026 (2.2.15, 2048 x 2048): free
/// device memory dropped by 38.06 GiB at 6,877 rows and 38.66 GiB at 10,370
/// rows, 36.81 GiB each after ImageDitRowBytes; rounded up.
inline constexpr std::uint64_t kH3ImageDitFixedBytes = 37ULL << 30U;
/// Device memory kept free besides the estimate: hipMalloc failed on this
/// machine with 7.4 to 8.5 GiB still reported free, and other processes
/// (dwm.exe, another model) can take memory between the check and the
/// allocation. GUFO_H3_IMAGE_TOKEN_RESERVE_MIB overrides it.
inline constexpr std::uint64_t kH3ImageEditDefaultReserveBytes = 10ULL << 30U;

/// Rows of the image DiT's packed sequence.
struct ImageDitRows {
  std::size_t text{0};       // the whole text-encoder presentation
  std::size_t condition{0};  // reference latents (2x2 patches)
  std::size_t image{0};      // the generated image (2x2 patches)
  [[nodiscard]] std::size_t total() const noexcept {
    return text + condition + image;
  }
};

/// Device bytes of the DiT's per-row buffers (shared block workspace, rotary
/// tables, hidden state, row maps, the first block's GEMM check buffer and
/// the patch-projection scratch); mirrors MakeSharedWorkspaceLayout,
/// DitBlockSession::Create and AllocateIo. About 179 KiB per row (measured
/// slope 21.11.: 180 KiB) plus 21 KiB per row of the larger patch segment;
/// the cheaper native path from 32,768 rows is not credited.
[[nodiscard]] std::uint64_t ImageDitRowBytes(const ImageDitRows& rows) noexcept;

/// kH3ImageDitFixedBytes + ImageDitRowBytes.
[[nodiscard]] std::uint64_t ImageDitPhaseBytes(const ImageDitRows& rows) noexcept;

/// DiT rows of a width x height reference upload: its VAE copy (short edge
/// bounded to [1024, 2048], cropped to 16-px cells) in 2x2 latent patches;
/// 1,024 for any square up to 1024 px, 4,096 for a 2048 x 2048 one.
[[nodiscard]] std::size_t ImageEditReferenceDitRows(int width, int height);

/// What the memory-based limit is computed from.
struct ImageEditMemory {
  std::uint64_t free_bytes{0};  // hipMemGetInfo at the check; 0 = unknown
  std::uint64_t reserve_bytes{kH3ImageEditDefaultReserveBytes};
  // Hard cap (GUFO_H3_IMAGE_MAX_TOKENS), within [4096, ceiling].
  std::size_t token_cap{kPromptEncoderTokenCeiling};
};

/// The memory settings from GUFO_H3_IMAGE_TOKEN_RESERVE_MIB (MiB, a whole
/// number) and GUFO_H3_IMAGE_MAX_TOKENS (clamped to [4096, 32768]; 4096
/// switches the extension off). Null, empty or malformed values keep the
/// defaults.
[[nodiscard]] ImageEditMemory ImageEditMemoryFromSettings(
    std::uint64_t free_bytes, const char* reserve_mib, const char* max_tokens);

/// ImageEditMemoryFromSettings with the device's free memory now
/// (DeviceMemoryInfo; 0 without HIP) and the environment.
[[nodiscard]] ImageEditMemory CurrentImageEditMemory();

/// The most text-encoder tokens an edit may have when its DiT also carries
/// `condition_rows` reference rows and `image_rows` image rows: the largest
/// T in [4096, token_cap] whose prompt phase and DiT phase (T text rows)
/// both fit in free_bytes - reserve_bytes. Never below 4096, so with too
/// little (or unknown) free memory the limit stays exactly the old one.
[[nodiscard]] std::size_t ImageEditTokenLimit(const ImageEditMemory& memory,
                                              std::size_t condition_rows,
                                              std::size_t image_rows);

/// How many square references (1024 x 1024 or smaller: 1,031 tokens and
/// 1,024 DiT rows each, 1,032 from Picture 10) fit with a one-token prompt
/// and an image of `image_rows` rows; between kH3ImageEditMaximumImages and
/// kH3ImageMaximumReferences. With unlimited free memory (only the token
/// cap counts) this is what the provider reports as `max_images`.
[[nodiscard]] std::size_t ImageEditMaximumSquareReferences(
    const ImageEditMemory& memory, std::size_t image_rows);

/// What an edit presentation costs in text-encoder tokens.
struct ImageEditTokenBudget {
  std::size_t references{0};
  std::size_t vision_tokens{0};  // image-pad rows of all vision blocks
  std::size_t label_tokens{0};   // "Picture i: " labels plus 2 markers each
  std::size_t prompt_tokens{0};  // after the 4096 cut
  std::size_t total{0};
};

/// The budget of references of the given (width, height) upload sizes, the
/// token counts of their labels (in the same order) and of the prompt.
[[nodiscard]] ImageEditTokenBudget CountImageEditTokens(
    std::span<const std::pair<int, int>> image_sizes,
    std::span<const std::size_t> label_tokens, std::size_t prompt_tokens);

/// True when the budget fits `limit` tokens (ImageEditTokenLimit); otherwise
/// an English message with the numbers and what to change.
[[nodiscard]] bool CheckImageEditTokenBudget(
    const ImageEditTokenBudget& budget, std::size_t limit, std::string* error);

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
