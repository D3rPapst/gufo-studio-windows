#pragma once

// gufo-studio-windows: eager stand-in for upstream's session state declarations
// (gufo/ext/state.hpp, src/mem/state_declaration.hpp, upstream #1333). This fork
// has no demand-mapped state memory (src/mem, upstream #1251/#1308/#1333), so
// the core cannot allocate declared state. The declaration types below match
// upstream's names and byte counts; compat/state_memory.hpp allocates every
// buffer whole when the session is created. Drop both files once the fork
// carries upstream's mem layer. KI-unterstuetzt (Claude Code).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace gufo::mem {

struct RegionShape {
  std::size_t row_bytes{0};
  std::size_t rows{0};
  std::size_t segments{1};
  /// Zero means rows * row_bytes: segments back to back.
  std::size_t segment_stride{0};
};

[[nodiscard]] inline std::size_t SegmentStride(
    const RegionShape& shape) noexcept {
  return shape.segment_stride != 0 ? shape.segment_stride
                                   : shape.rows * shape.row_bytes;
}

[[nodiscard]] inline std::size_t RegionSpanBytes(
    const RegionShape& shape) noexcept {
  if (shape.segments == 0)
    return 0;
  return (shape.segments - 1) * SegmentStride(shape) +
         shape.rows * shape.row_bytes;
}

enum class StateKind : std::uint8_t {
  kContext,
  kWhole,
  kFixed,
};

struct StateBuffer {
  StateKind kind{StateKind::kFixed};
  RegionShape shape{};
  std::size_t tokens_per_row{1};
};

using StateBufferId = std::size_t;

enum class StateBytes : std::uint8_t {
  kSize,
  kPages,
};

class StateDeclaration {
public:
  explicit StateDeclaration(std::size_t margin_tokens = 0) noexcept
      : margin_tokens_(margin_tokens) {}

  StateBufferId Context(const RegionShape& shape,
                        std::size_t tokens_per_row = 1) {
    CheckShape(shape);
    buffers_.push_back(
        {.kind = StateKind::kContext,
         .shape = shape,
         .tokens_per_row = std::max<std::size_t>(tokens_per_row, 1)});
    return buffers_.size() - 1;
  }
  StateBufferId Whole(const RegionShape& shape) {
    CheckShape(shape);
    buffers_.push_back({.kind = StateKind::kWhole, .shape = shape});
    return buffers_.size() - 1;
  }
  StateBufferId Fixed(std::size_t bytes) {
    buffers_.push_back(
        {.kind = StateKind::kFixed,
         .shape = {.row_bytes = std::max<std::size_t>(bytes, 1), .rows = 1}});
    return buffers_.size() - 1;
  }

  [[nodiscard]] const std::vector<StateBuffer>& buffers() const noexcept {
    return buffers_;
  }
  [[nodiscard]] const StateBuffer& buffer(StateBufferId id) const {
    return buffers_.at(id);
  }
  [[nodiscard]] std::size_t margin_tokens() const noexcept {
    return margin_tokens_;
  }
  [[nodiscard]] bool has_context() const noexcept {
    return std::any_of(buffers_.begin(), buffers_.end(), [](const auto& b) {
      return b.kind == StateKind::kContext;
    });
  }

  /// Without demand mapping a buffer always takes its whole span; kPages
  /// rounds it up to 2 MiB pages as upstream's region pool would.
  [[nodiscard]] std::size_t Bytes(StateBufferId id, StateBytes count) const {
    const StateBuffer& b = buffers_.at(id);
    if (b.kind == StateKind::kFixed)
      return b.shape.row_bytes;
    const std::size_t span = RegionSpanBytes(b.shape);
    if (count == StateBytes::kPages) {
      constexpr std::size_t kPage = std::size_t{2} << 20;
      return (span + kPage - 1) / kPage * kPage;
    }
    return span;
  }
  [[nodiscard]] std::size_t Bytes(StateBytes count) const noexcept {
    std::size_t bytes = 0;
    for (StateBufferId id = 0; id < buffers_.size(); ++id)
      bytes += Bytes(id, count);
    return bytes;
  }
  [[nodiscard]] std::size_t ContextBytes(StateBytes count) const noexcept {
    std::size_t bytes = 0;
    for (StateBufferId id = 0; id < buffers_.size(); ++id) {
      if (buffers_[id].kind == StateKind::kContext)
        bytes += Bytes(id, count);
    }
    return bytes;
  }

private:
  static void CheckShape(const RegionShape& shape) {
    if (shape.row_bytes == 0 || shape.rows == 0 || shape.segments == 0)
      throw std::invalid_argument("state buffer without rows");
    if (shape.segment_stride != 0 &&
        shape.rows > shape.segment_stride / shape.row_bytes)
      throw std::invalid_argument("state buffer segments overlap");
  }

  std::size_t margin_tokens_{0};
  std::vector<StateBuffer> buffers_;
};

}  // namespace gufo::mem
