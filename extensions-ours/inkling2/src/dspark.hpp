// gufo_windows addition (not part of headbouyJB/inkling2-gufo): DSpark drafter of Inkling-2-Mini on the GPU
// (unsloth dspark-Inkling-2-Mini-*.gguf, GGUF architecture `dflash`). After sebt3's engine/dspark.hpp and
// ciru-ai's vLLM draft: the target's features (residual stream entering visits 5 / 23 / 41 and the final
// normalised state, Engine::SetFeatureCapture) become context keys / values of a 3-layer block drafter; a draft is
// the anchor's target embedding plus 7 mask rows through the drafter (non-causal inside the block), then up to 7
// sequential steps of a recurrent Markov head over the target's lm_head. The head's correction
// lm_head . (hidden_out . latent + bias) is folded once into a [vocab][256] BF16 matrix (ciru-ai), so a step
// reads 100 MB instead of the 490 MB lm_head; the base logits of all steps come from one batched lm_head pass.
// Proposals only need to be good: the target verifies every token (Engine::VerifyChain / ChainKeep).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/engine.hpp"

namespace gufo::models::inkling2 {

class DSparkDrafter {
public:
  struct Options {
    int window_extra = 0;  // local layers see sliding_window (+ this) positions; ciru-ai uses + block (8)
  };
  /// Loads the drafter next to `target` (which must outlive it). Null and `error` on failure.
  static std::unique_ptr<DSparkDrafter> Create(Engine& target, const std::string& path, const Options& options,
                                               std::string* error);
  ~DSparkDrafter();
  DSparkDrafter(const DSparkDrafter&) = delete;
  DSparkDrafter& operator=(const DSparkDrafter&) = delete;

  /// A new sequence: no context before `pos` (after a reset or a cache restore at `pos`).
  void Reset(std::size_t pos);
  /// Context of positions [pos0, pos0 + n) from the target's last decode step (features rows 0..n-1).
  void Inject(std::size_t pos0, std::size_t n);
  /// Engine::FeatureSink target: small steps are staged (one device copy) and injected in one batch before the next
  /// draft; a step that starts before the staged end replaces the rejected tail; large steps (prefill) go in directly.
  void Stage(std::size_t pos0, std::size_t n, const float* device_rows);
  /// Injects the staged rows now (Draft does it itself).
  void Flush();
  /// Injects device rows right away (no staging; round-6 behaviour, for A/B checks).
  void InjectNow(std::size_t pos0, std::size_t n, const float* device_rows);
  /// Up to k (<= 7) tokens after `anchor` at position pos0; empty when the context of pos0 - 1 is missing.
  [[nodiscard]] std::vector<std::int32_t> Draft(std::int32_t anchor, std::size_t pos0, int k);
  /// Wall time of the last Draft call (synchronised), milliseconds.
  [[nodiscard]] double last_draft_ms() const;

private:
  DSparkDrafter() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::inkling2
