// Written with AI assistance (Claude Code).
#include "src/models/yue2/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

#include "src/models/ace_step/pipeline.hpp"
#include "src/models/ace_step/vae.hpp"
#include "src/models/yue2/model.hpp"
#include "src/models/yue2/tokenizer.hpp"

namespace gufo::models::yue2 {
namespace {

bool Fail(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
  return false;
}

double Since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       start)
      .count();
}

}  // namespace

struct SongPipeline::Impl {
  std::unique_ptr<TextTokenizer> tokenizer;
  std::unique_ptr<Model> model;
  std::unique_ptr<ace_step::VaeDecoder> vae;
};

SongPipeline::SongPipeline(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
SongPipeline::~SongPipeline() = default;

std::size_t SongPipeline::device_bytes() const {
  return impl_->model->device_bytes();
}

std::unique_ptr<SongPipeline> SongPipeline::Load(
    const std::filesystem::path& model_dir,
    const std::filesystem::path& vae_dir, std::string* error) {
  auto impl = std::make_unique<Impl>();
  impl->tokenizer = TextTokenizer::Load(model_dir / "qwen.tiktoken", error);
  if (!impl->tokenizer)
    return nullptr;
  impl->model = Model::Load(model_dir / "model.safetensors", error);
  if (!impl->model)
    return nullptr;
  impl->vae = ace_step::VaeDecoder::Load(vae_dir / "model.safetensors",
                                         ace_step::OobleckLayout::YuE2(),
                                         error);
  if (!impl->vae)
    return nullptr;
  return std::unique_ptr<SongPipeline>(new SongPipeline(std::move(impl)));
}

bool SongPipeline::Generate(const SongRequest& request, SongResult* result,
                            const SongProgress& progress, std::string* error) {
  Impl& m = *impl_;
  *result = SongResult{};
  const auto start = std::chrono::steady_clock::now();
  if (!request.abc.empty() && request.cot == Cot::kOff)
    return Fail(error, "an own ABC score needs cot full or melody");
  if (request.ode_steps < 1 || request.ode_steps > 256)
    return Fail(error, "ode_steps must be 1..256");
  const float guidance =
      request.guidance < 0 ? DefaultGuidance(request.cot) : request.guidance;
  if (!std::isfinite(guidance) || guidance > 20)
    return Fail(error, "guidance must be 0..20");
  const auto report = [&](double fraction, std::string_view phase) {
    return !progress || progress(std::clamp(fraction, 0.0, 1.0), phase);
  };
  std::vector<std::uint32_t> prefix;
  std::vector<std::uint32_t> abc_ids;
  try {
    // 1. Score (ABC): planned by the AR model or given.
    if (request.cot != Cot::kOff && !request.abc.empty()) {
      abc_ids = m.tokenizer->Encode(request.abc);
      result->abc = request.abc;
    } else if (request.cot != Cot::kOff) {
      const auto plan = TokenPrefix(*m.tokenizer, request.cot, request.style,
                                    request.lyrics, nullptr);
      const PhaseSampling sampling = DefaultAbcSampling();
      PhaseResult phase;
      if (!m.model->Generate(
              plan, {}, 1.0F, Phase::kAbc, sampling, request.seed, false,
              &phase,
              [&](int n) {
                return report(0.15 * n / sampling.max_tokens, "abc");
              },
              error))
        return false;
      abc_ids = phase.tokens;
      result->abc = m.tokenizer->Decode(abc_ids);
      result->abc_truncated = phase.truncated;
      result->abc_seconds = phase.seconds;
    }
    result->abc_tokens = static_cast<int>(abc_ids.size());
    prefix = TokenPrefix(*m.tokenizer, request.cot, request.style,
                         request.lyrics, &abc_ids);
    // 2. Semantic codes (25 Hz).
    PhaseSampling sampling = DefaultSemanticSampling();
    if (request.max_seconds > 0) {
      sampling.max_tokens = std::clamp(
          static_cast<int>(std::ceil(request.max_seconds * kFramesPerSecond)),
          1, sampling.max_tokens);
      sampling.min_tokens = std::min(sampling.min_tokens, sampling.max_tokens);
    }
    if (static_cast<int>(prefix.size()) + sampling.max_tokens > kContext)
      sampling.max_tokens = kContext - static_cast<int>(prefix.size());
    if (sampling.max_tokens < 1)
      return Fail(error, "lyrics and score leave no room in the context");
    sampling.min_tokens = std::min(sampling.min_tokens, sampling.max_tokens);
    std::vector<std::uint32_t> negative;
    if (guidance != 1.0F)
      negative = NegativePrefix(*m.tokenizer, request.cot, abc_ids);
    PhaseResult semantic;
    if (!m.model->Generate(
            prefix, negative, guidance, Phase::kSemantic, sampling,
            request.seed, request.cot == Cot::kOff, &semantic,
            [&](int n) {
              return report(0.15 + 0.5 * n / sampling.max_tokens, "semantic");
            },
            error))
      return false;
    result->semantic_truncated = semantic.truncated;
    result->semantic_tokens = static_cast<int>(semantic.tokens.size());
    result->semantic_seconds = semantic.seconds;
    const int frames = result->semantic_tokens;
    if (frames < 1)
      return Fail(error, "YuE2 produced no music codes");
    // 3. Flow matching, one AR prefill per chunk (nar.song_chunks).
    const auto nar_start = std::chrono::steady_clock::now();
    const std::vector<float> noise = ace_step::MusicNoise(
        request.seed, std::size_t(frames) * kLatentChannels);
    const auto ranges = ChunkRanges(frames, static_cast<int>(prefix.size()));
    result->chunks = static_cast<int>(ranges.size());
    std::vector<float> latents;
    latents.reserve(noise.size());
    for (std::size_t c = 0; c < ranges.size(); ++c) {
      const auto [a, b] = ranges[c];
      std::vector<std::uint32_t> ar = prefix;
      ar.insert(ar.end(), semantic.tokens.begin() + a,
                semantic.tokens.begin() + b);
      ar.push_back(kMusicEnd);
      std::vector<float> chunk;
      if (!m.model->BeginAcoustic(ar, b - a, error) ||
          !m.model->Solve(
              std::span<const float>(noise).subspan(
                  std::size_t(a) * kLatentChannels,
                  std::size_t(b - a) * kLatentChannels),
              request.ode_steps, &chunk,
              [&](int step) {
                const double done =
                    (c + double(step) / request.ode_steps) / ranges.size();
                return report(0.65 + 0.3 * done, "acoustic");
              },
              error))
        return false;
      latents.insert(latents.end(), chunk.begin(), chunk.end());
    }
    result->nar_seconds = Since(nar_start);
    // 4. VAE.
    if (!report(0.95, "vae"))
      return Fail(error, "cancelled");
    const auto vae_start = std::chrono::steady_clock::now();
    if (!m.vae->Decode(latents, frames, &result->audio, error))
      return false;
    for (float& x : result->audio)
      x = std::isfinite(x) ? std::clamp(x, -1.0F, 1.0F) : 0.0F;
    result->vae_seconds = Since(vae_start);
  } catch (const std::exception& e) {
    return Fail(error, e.what());
  }
  result->total_seconds = Since(start);
  report(1.0, "done");
  return true;
}

}  // namespace gufo::models::yue2
