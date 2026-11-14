// YuE2 song pipeline: ABC score (AR), 25 Hz semantic codes (AR), flow
// matching to VAE latents (NAR, chunked as nar.song_chunks) and the
// YuE2-Vae Oobleck decoder (48 kHz stereo). HIP only.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_YUE2_PIPELINE_HPP_
#define GUFO_MODELS_YUE2_PIPELINE_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/yue2/protocol.hpp"

namespace gufo::models::yue2 {

struct SongRequest {
  std::string style;   // genre, instruments, voice, language, tempo
  std::string lyrics;  // "[Verse]" / "[Chorus]" sections, may be empty
  Cot cot{Cot::kFull};
  std::string abc;     // optional own score (cot full / melody)
  std::uint64_t seed{831001};
  float guidance{-1};  // < 0: the default (1.01 for off, else 1)
  /// Upper bound of the song length in seconds (0: the model ends the song,
  /// at most 9000 codes = 6 minutes). A shorter bound cuts the song.
  double max_seconds{0};
  int ode_steps{kOdeSteps};
};

struct SongResult {
  std::vector<float> audio;  // planar stereo [2][samples], 48 kHz, |x| <= 1
  std::string abc;           // the score (planned or given)
  bool abc_truncated{false};
  bool semantic_truncated{false};
  int abc_tokens{0};
  int semantic_tokens{0};  // = latent frames (25 / s)
  int chunks{0};
  double abc_seconds{0};
  double semantic_seconds{0};
  double nar_seconds{0};
  double vae_seconds{0};
  double total_seconds{0};
};

/// Fraction 0..1 and the phase ("abc", "semantic", "acoustic", "vae");
/// returning false cancels.
using SongProgress = std::function<bool(double, std::string_view)>;

class Model;

class SongPipeline {
public:
  /// `model_dir` holds YuE2-3B (`model.safetensors`, `qwen.tiktoken`),
  /// `vae_dir` YuE2-Vae (`model.safetensors`).
  [[nodiscard]] static std::unique_ptr<SongPipeline> Load(
      const std::filesystem::path& model_dir,
      const std::filesystem::path& vae_dir, std::string* error);
  ~SongPipeline();
  SongPipeline(const SongPipeline&) = delete;
  SongPipeline& operator=(const SongPipeline&) = delete;

  bool Generate(const SongRequest& request, SongResult* result,
                const SongProgress& progress, std::string* error);
  [[nodiscard]] std::size_t device_bytes() const;

private:
  struct Impl;
  explicit SongPipeline(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::yue2

#endif  // GUFO_MODELS_YUE2_PIPELINE_HPP_
