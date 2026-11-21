#include "src/models/minimax_h3/prompt_encoder.hpp"

#include <algorithm>
#include <cstdint>

namespace gufo::minimax_h3 {

bool BuildPromptRopePositions(
    std::size_t tokens, std::span<const PromptVisionBlock> blocks,
    std::vector<std::array<std::uint32_t, 3>>* positions, std::string* error) {
  positions->clear();
  positions->reserve(tokens);
  std::uint32_t offset = 0;
  std::size_t row = 0;
  for (const PromptVisionBlock& block : blocks) {
    const std::size_t rows =
        static_cast<std::size_t>(block.grid_height) * block.grid_width;
    if (block.begin < row || rows == 0 || block.begin + rows > tokens) {
      if (error != nullptr) {
        *error = "MiniMax H3 image blocks overlap or leave the prompt";
      }
      return false;
    }
    for (; row < block.begin; ++row) {
      positions->push_back({offset, offset, offset});
      ++offset;
    }
    for (std::uint32_t y = 0; y < block.grid_height; ++y) {
      for (std::uint32_t x = 0; x < block.grid_width; ++x) {
        positions->push_back({offset, offset + y, offset + x});
      }
    }
    row += rows;
    offset += std::max(block.grid_height, block.grid_width);
  }
  for (; row < tokens; ++row) {
    positions->push_back({offset, offset, offset});
    ++offset;
  }
  return true;
}

#if !defined(ENGINE_ENABLE_HIP)
bool EncodePromptLayer50(const ModelInventory&, std::span<const std::uint32_t>,
                         const PromptEncoderOptions&, const CancellationToken*,
                         PromptEmbedding*, PromptEncoderTelemetry*,
                         std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 prompt encoder requires ENGINE_ENABLE_HIP";
  }
  return false;
}

std::uint64_t PromptEncoderLiveRegisteredHostBytes() noexcept {
  return 0;
}
#endif

std::uint64_t PromptEncoderScoreScratchBytes(std::size_t tokens) noexcept {
  // Mirrors ops::CausalGqaScratchBytes (prompt_encoder_ops.hip.hpp) for the
  // encoder's grouped layout (64 query heads over 8 key/value heads, head
  // dimension 128); minimax_h3_prompt_encoder_hip_test compares the two.
  constexpr std::uint64_t kGrouped =
      kPromptEncoderQueryHeads / kPromptEncoderKeyValueHeads;
  constexpr std::uint64_t kSharedBytes = 48U << 10U;
  constexpr std::uint64_t kScratchBytes = 256ULL << 20U;
  const std::uint64_t sequence = tokens;
  const std::uint64_t shared =
      (sequence * kGrouped + kPromptEncoderHeadDimension * kGrouped + 128U) *
      sizeof(float);
  if (sequence == 0 || shared <= kSharedBytes) {
    return 0;
  }
  const std::uint64_t row_bytes =
      sequence * kPromptEncoderQueryHeads * sizeof(float);
  std::uint64_t rows = kScratchBytes / row_bytes;
  rows = rows == 0 ? 1 : std::min(rows, sequence);
  return rows * row_bytes;
}

std::uint64_t PromptEncoderActivationBytes(std::size_t tokens) noexcept {
  // Mirrors AllocateActivations (prompt_encoder.hip).
  constexpr std::uint64_t kBf16 = sizeof(std::uint16_t);
  constexpr std::uint64_t kAttention =
      kPromptEncoderQueryHeads * kPromptEncoderHeadDimension;
  constexpr std::uint64_t kKeyValue =
      kPromptEncoderKeyValueHeads * kPromptEncoderHeadDimension;
  constexpr std::uint64_t kPerToken =
      sizeof(std::uint32_t) +                         // ids
      4U * kPromptEncoderHiddenSize * kBf16 +         // hidden, norm, proj, mlp
      2U * kAttention * kBf16 +                       // query, attention
      2U * kKeyValue * kBf16 +                        // key, value
      2U * (kPromptEncoderHeadDimension / 2U) * sizeof(float) +  // rope
      2U * kPromptEncoderIntermediateSize * kBf16;    // gate, up
  return static_cast<std::uint64_t>(tokens) * kPerToken +
         PromptEncoderScoreScratchBytes(tokens);
}

std::uint64_t PromptEncoderPhaseBytes(std::size_t tokens) noexcept {
  return kPromptEncoderPhaseFixedBytes + PromptEncoderActivationBytes(tokens);
}

}  // namespace gufo::minimax_h3
