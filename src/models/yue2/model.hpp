// YuE2-3B (m-a-p/YuE2-3B `model.safetensors`, BF16): one 28-layer Qwen3-1.7B
// shaped backbone with two weight sets per layer ("mixture of
// transformers"): the AR set writes the ABC score and the 25 Hz semantic
// codes, the NAR set solves flow matching from noise to 64-channel VAE
// latents while attending to the AR keys and values of the song prefix.
// HIP only. Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_YUE2_MODEL_HPP_
#define GUFO_MODELS_YUE2_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/yue2/protocol.hpp"

namespace gufo::models::yue2 {

/// Result of one AR phase.
struct PhaseResult {
  std::vector<std::uint32_t> tokens;  // without the end token
  bool truncated{false};              // max_tokens reached without end
  double seconds{0};
  double prefill_seconds{0};
};

class Model {
public:
  [[nodiscard]] static std::unique_ptr<Model> Load(
      const std::filesystem::path& safetensors, std::string* error);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  /// Samples one phase continuing `prefix` (generate_tokens): with
  /// `guidance` != 1 the logits are uncond + g * (cond - uncond) with the
  /// `negative` prefix in a second KV cache. `progress` gets the token count
  /// (returning false cancels). `rng` is reset from `seed` per phase as the
  /// reference does.
  bool Generate(std::span<const std::uint32_t> prefix,
                std::span<const std::uint32_t> negative, float guidance,
                Phase phase, const PhaseSampling& sampling,
                std::uint64_t seed, bool legacy_off, PhaseResult* result,
                const std::function<bool(int)>& progress, std::string* error);

  /// FP32 logits of the phase slice after prefilling `prefix` (checks).
  bool PrefixLogits(std::span<const std::uint32_t> prefix, Phase phase,
                    std::vector<float>* logits, std::string* error);

  /// Prefills one acoustic chunk's AR tokens (prefix + codes + MUSIC_END)
  /// with the AR weights and keeps their keys and values for `frames` NAR
  /// frames.
  bool BeginAcoustic(std::span<const std::uint32_t> ar_tokens, int frames,
                     std::string* error);
  /// Flow velocity [frames, 64] of state [frames, 64] at the raw (logit)
  /// time `raw_t` for the current chunk.
  bool Velocity(std::span<const float> state, double raw_t,
                std::vector<float>* velocity, std::string* error);
  /// 32-step (or `steps`) midpoint ODE from `noise` [frames, 64] to latents
  /// (nar.CachedNAR.solve); `progress` gets the finished steps.
  bool Solve(std::span<const float> noise, int steps,
             std::vector<float>* latents,
             const std::function<bool(int)>& progress, std::string* error);

  [[nodiscard]] std::size_t weight_bytes() const;
  [[nodiscard]] std::size_t device_bytes() const;

private:
  struct Impl;
  explicit Model(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::yue2

#endif  // GUFO_MODELS_YUE2_MODEL_HPP_
