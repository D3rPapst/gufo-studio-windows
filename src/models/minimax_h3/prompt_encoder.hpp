#ifndef GUFO_MODELS_MINIMAX_H3_PROMPT_ENCODER_HPP_
#define GUFO_MODELS_MINIMAX_H3_PROMPT_ENCODER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"

namespace gufo::minimax_h3 {

inline constexpr std::size_t kPromptEncoderLayers = 50;
inline constexpr std::size_t kPromptEncoderHiddenSize = 5120;
inline constexpr std::size_t kPromptEncoderIntermediateSize = 25600;
inline constexpr std::size_t kPromptEncoderQueryHeads = 64;
inline constexpr std::size_t kPromptEncoderKeyValueHeads = 8;
inline constexpr std::size_t kPromptEncoderHeadDimension = 128;
inline constexpr std::size_t kPromptEncoderVocabularySize = 151936;
/// The most tokens (text, vision blocks and their markers together) one
/// EncodePromptLayer50 call takes unless the caller raises
/// PromptEncoderOptions::maximum_tokens (video prompts always stay at it).
inline constexpr std::size_t kPromptEncoderMaximumTokens = 4096;
/// Hard ceiling of PromptEncoderOptions::maximum_tokens. The encoder sizes
/// its activations, rotary tables and attention scratch from the actual
/// prompt length; this only bounds what a caller may ask for (16 square
/// references and a 4096-token prompt need about 20,600).
inline constexpr std::size_t kPromptEncoderTokenCeiling = 32768;

inline constexpr std::uint32_t kVisionStartToken = 151652;
inline constexpr std::uint32_t kVisionEndToken = 151653;
inline constexpr std::uint32_t kImagePadToken = 151655;
inline constexpr std::size_t kVisionDeepstackLayers = 3;

/// One Qwen3-VL image block of the prompt: `rows` consecutive image-pad
/// tokens starting at `begin`, laid out row-major over the merged
/// grid_height x grid_width patch grid. Embeddings are BF16 rows of width
/// kPromptEncoderHiddenSize; deepstack features are added to the same rows
/// after decoder layers 0, 1 and 2.
struct PromptVisionBlock {
  std::size_t begin{0};
  std::uint32_t grid_height{0};
  std::uint32_t grid_width{0};
  std::vector<std::uint16_t> embedding;
  std::vector<std::vector<std::uint16_t>> deepstack;
};

struct PromptEncoderOptions {
  std::size_t layer_count{kPromptEncoderLayers};
  bool prefetch_layers{true};
  bool vision_inputs{false};
  bool deepstack_inputs{false};
  // Image blocks in prompt order. Non-empty switches the rotary positions
  // to Qwen3-VL's interleaved three-axis layout.
  std::span<const PromptVisionBlock> vision_blocks;
  // Longest prompt this call accepts, at most kPromptEncoderTokenCeiling.
  // MiniMax-H3-Image edits raise it from free device memory
  // (ImageEditTokenLimit); everything else keeps 4096.
  std::size_t maximum_tokens{kPromptEncoderMaximumTokens};
};

/// Device bytes of the grouped attention's score scratch for `tokens`
/// (ops::CausalGqaScratchBytes with the encoder's 64/8/128 heads): 0 while
/// the scores fit in shared memory, at most 256 MiB beyond.
[[nodiscard]] std::uint64_t PromptEncoderScoreScratchBytes(
    std::size_t tokens) noexcept;

/// Device bytes EncodePromptLayer50 allocates for its activations, rotary
/// tables and score scratch at `tokens` (Activations::Bytes()); about
/// 176.5 KiB per token plus the scratch.
[[nodiscard]] std::uint64_t PromptEncoderActivationBytes(
    std::size_t tokens) noexcept;

/// Estimated device memory of the whole prompt phase (prompt child) at
/// `tokens`: PromptEncoderActivationBytes plus a fixed
/// kPromptEncoderPhaseFixedBytes for the embedding table, two decoder layers
/// in flight, the GEMM workspace, the vision tower and the HIP runtime.
inline constexpr std::uint64_t kPromptEncoderPhaseFixedBytes = 6ULL << 30U;
[[nodiscard]] std::uint64_t PromptEncoderPhaseBytes(
    std::size_t tokens) noexcept;

/// Qwen3-VL rope positions (temporal, height, width) of every token: text
/// advances all three axes together, an image block spans its grid from the
/// current offset and the text after it resumes at max(grid) + 1.
[[nodiscard]] bool BuildPromptRopePositions(
    std::size_t tokens, std::span<const PromptVisionBlock> blocks,
    std::vector<std::array<std::uint32_t, 3>>* positions,
    std::string* error = nullptr);

struct PromptEncoderLayerTelemetry {
  std::size_t layer{0};
  std::uint64_t weight_bytes{0};
  double load_ms{0.0};
  double prefetch_wait_ms{0.0};
  double submit_ms{0.0};
  double gpu_ms{0.0};
  std::uint64_t dispatches{0};
};

struct PromptEncoderTelemetry {
  std::vector<PromptEncoderLayerTelemetry> layers;
  std::uint64_t embedding_weight_bytes{0};
  std::uint64_t activation_bytes{0};
  std::uint64_t peak_device_bytes{0};
  std::uint64_t peak_registered_host_bytes{0};
  std::uint64_t live_registered_host_bytes_after{0};
  std::uint64_t total_dispatches{0};
};

struct PromptEmbedding {
  std::size_t tokens{0};
  std::size_t width{0};
  std::vector<std::uint16_t> values;
};

[[nodiscard]] bool EncodePromptLayer50(const ModelInventory& inventory,
                                       std::span<const std::uint32_t> token_ids,
                                       const PromptEncoderOptions& options,
                                       const CancellationToken* cancellation,
                                       PromptEmbedding* output,
                                       PromptEncoderTelemetry* telemetry,
                                       std::string* error = nullptr);

[[nodiscard]] std::uint64_t PromptEncoderLiveRegisteredHostBytes() noexcept;

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_PROMPT_ENCODER_HPP_
