// Written with AI assistance (Claude Code).
#include "src/models/wan22/pipeline_i2v22.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <system_error>
#include <utility>

#include "src/core/hip/admission_query.hpp"
#include "src/core/logging/logging.hpp"
#include "src/models/wan22/pipeline.hpp"
#include "src/models/wan22/text_encoder.hpp"
#include "src/models/wan22/tokenizer.hpp"
#include "src/models/wan22/transformer21.hpp"
#include "src/models/wan22/vace_context.hpp"

namespace gufo::models::wan22 {
namespace {

using Clock = std::chrono::steady_clock;

double Ms(Clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin)
      .count();
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

void TrackDevice(std::size_t* peak) {
  std::size_t free = 0, total = 0;
  if (hipMemGetInfo(&free, &total) == hipSuccess)
    *peak = std::max(*peak, total - std::min(free, total));
}

/// What an expert was loaded from (the LoRA is merged at load).
struct ExpertKey {
  std::filesystem::path dit, lora;
  float lora_strength{1.0F};
  bool operator==(const ExpertKey&) const = default;
};

/// Wan 2.2 I2V A14B experts kept in the worker over jobs. Never destroyed at
/// exit (the HIP runtime may be gone by then); ReleaseI2V22Experts frees it.
struct ResidentExperts {
  bool loaded{false};
  ExpertKey high_key, low_key;
  std::unique_ptr<Transformer21> high, low;
  void Reset() {
    high.reset();
    low.reset();
    high_key = {};
    low_key = {};
    loaded = false;
  }
};

std::mutex& ResidentMutex() {
  static std::mutex* mutex = new std::mutex;
  return *mutex;
}

ResidentExperts& Resident() {
  static ResidentExperts* experts = new ResidentExperts;
  return *experts;
}

std::uint64_t FileBytes(const std::filesystem::path& file) {
  std::error_code ec;
  if (file.empty())
    return 0;
  const auto bytes = std::filesystem::file_size(file, ec);
  return ec ? 0 : static_cast<std::uint64_t>(bytes);
}

std::uint64_t DirectoryBytes(const std::filesystem::path& directory) {
  std::error_code ec;
  std::uint64_t total = 0;
  for (std::filesystem::recursive_directory_iterator it(directory, ec), end;
       !ec && it != end; it.increment(ec))
    if (it->is_regular_file(ec))
      total += FileBytes(it->path());
  return total;
}

/// What this process can still allocate on the device, as the admission
/// check reads it (DXGI budget on a Windows iGPU, else hipMemGetInfo).
std::uint64_t DeviceFreeBytes() {
  if (const auto memory = gufo::hip::QueryAdmissionMemory())
    return memory->free_bytes;
  return 0;
}

std::string Gib(std::uint64_t bytes) {
  char text[32];
  std::snprintf(text, sizeof text, "%.1f",
                static_cast<double>(bytes) / static_cast<double>(1ULL << 30));
  return text;
}

void LogI2V(const std::string& message) {
  gufo::server::Logger::Info("video", "wan22-i2v-a14b " + message);
}

std::unique_ptr<Transformer21> LoadExpert(const ExpertKey& key,
                                          std::string* error) {
  auto dit = Transformer21::Load(key.dit, key.lora, key.lora_strength, error);
  if (!dit)
    return nullptr;
  // The A14B I2V experts: 36 input channels, no image cross-attention.
  if (dit->in_channels() != Transformer21::kInChannels || !dit->text_only()) {
    SetError(error, "Wan 2.2 I2V A14B needs an I2V A14B expert (36 input "
                    "channels, no CLIP image cross-attention), not " +
                        key.dit.string());
    return nullptr;
  }
  return dit;
}

}  // namespace

void ReleaseI2V22Experts() {
  const std::lock_guard<std::mutex> lock(ResidentMutex());
  Resident().Reset();
}

bool GenerateI2V22(const I2V22Request& request, const I2V21Callbacks& callbacks,
                   std::vector<float>* final_latents, DecodedVideo* video,
                   I2V22Telemetry* telemetry, std::string* error) {
  I2V22Telemetry local;
  I2V22Telemetry& tel = telemetry != nullptr ? *telemetry : local;
  const auto total_start = Clock::now();
  const int w = request.width, h = request.height, n = request.frames;
  const float high_guidance =
      request.high_guidance > 0.0F ? request.high_guidance : request.guidance;
  if (w % 16 != 0 || h % 16 != 0 || w < 16 || h < 16 || n < 1 ||
      (n - 1) % 4 != 0 || request.steps < 1 ||
      (request.sampler != "lcm" && request.sampler != "euler") ||
      !(request.guidance >= 1.0F) || !(high_guidance >= 1.0F) ||
      !(request.boundary >= 0.0 && request.boundary <= 1.0) ||
      request.high_dit_file.empty() || request.low_dit_file.empty() ||
      request.first_frame.size() != static_cast<std::size_t>(w) * h * 3) {
    SetError(error,
             "Wan 2.2 I2V A14B: width/height must be multiples of 16, frames "
             "4k+1, sampler lcm or euler, guidance >= 1, boundary 0..1, both "
             "experts set and the first frame at the video size");
    return false;
  }
  const auto dump = [&](const std::string& name,
                        const std::vector<float>& values,
                        std::vector<std::size_t> shape) {
    if (callbacks.dump)
      callbacks.dump(name, values, shape);
  };
  const auto progress = [&](const std::string& phase, int done, int total) {
    if (callbacks.progress)
      callbacks.progress(phase, done, total);
  };
  const auto cancelled = [&] {
    if (callbacks.keep_going && !callbacks.keep_going()) {
      SetError(error, "cancelled");
      return true;
    }
    return false;
  };
  const std::vector<double> sigmas = VaceSigmas(request.steps, request.shift);
  const int switch_at =
      VaceSwitchStep(sigmas, request.boundary, request.switch_step);
  tel.switch_step = switch_at;
  // CFG (negative prompt) when any step that runs uses guidance > 1.
  const bool cfg = (switch_at < request.steps && request.guidance > 1.0F) ||
                   (switch_at > 0 && high_guidance > 1.0F);
  const int lf = 1 + (n - 1) / 4, lh = h / 8, lw = w / 8;
  const std::size_t plane = static_cast<std::size_t>(lf) * lh * lw;
  const std::vector<std::size_t> shape16{16, static_cast<std::size_t>(lf),
                                         static_cast<std::size_t>(lh),
                                         static_cast<std::size_t>(lw)};

  // 1. Prompts (umT5, 512 rows: the states of the real tokens, then zeros).
  std::vector<float> text, negative;
  {
    auto start = Clock::now();
    const auto tokenizer = Tokenizer::Load(
        request.t5_root / "tokenizer" / "tokenizer.json", error);
    if (!tokenizer)
      return false;
    auto encoder = TextEncoder::Load(request.t5_root / "text_encoder", error);
    if (!encoder)
      return false;
    TrackDevice(&tel.peak_device_bytes);
    const auto encode = [&](const std::string& prompt,
                            std::vector<float>* out) {
      std::vector<float> states;
      if (!encoder->Encode(tokenizer->Encode(CleanPrompt(prompt)), &states,
                           error))
        return false;
      out->assign(TextEncoder::kMaxTokens * TextEncoder::kModelWidth, 0.0F);
      std::copy_n(states.begin(), std::min(states.size(), out->size()),
                  out->begin());
      return true;
    };
    if (!encode(request.prompt, &text) ||
        (cfg && !encode(request.negative_prompt, &negative)))
      return false;
    tel.text_ms = Ms(start);
    progress("prompt", 1, 1);
  }
  dump("text", text, {TextEncoder::kMaxTokens, TextEncoder::kModelWidth});
  if (cfg)
    dump("negative_text", negative,
         {TextEncoder::kMaxTokens, TextEncoder::kModelWidth});
  if (cancelled())
    return false;

  // 2. VAE condition: the start image, then grey (0 in [-1, 1]), encoded and
  // normalized; 4 mask channels on the first latent frame.
  std::vector<float> condition_channels;
  {
    auto start = Clock::now();
    auto encoder = VaeEncoder::Load(request.vae_root / "vae", error);
    if (!encoder)
      return false;
    const std::size_t frame = static_cast<std::size_t>(w) * h * 3;
    std::vector<float> input(frame * n, 0.0F);
    for (std::size_t i = 0; i < frame; ++i)
      input[i] = static_cast<float>(request.first_frame[i]) / 127.5F - 1.0F;
    std::vector<float> condition;
    if (!encoder->EncodeVideo(input, n, h, w, true, &condition, error))
      return false;
    TrackDevice(&tel.peak_device_bytes);
    dump("condition", condition, shape16);
    condition_channels = I2VConditionChannels(condition, lf, lh, lw);
    if (condition_channels.empty()) {
      SetError(error, "Wan 2.2 I2V A14B: unexpected VAE condition size");
      return false;
    }
    tel.condition_ms = Ms(start);
    progress("condition", 1, 1);
  }
  if (cancelled())
    return false;

  // 3. The high-noise expert, then the low-noise one, and sampling. With
  // experts_resident and room on the device both stay loaded over jobs;
  // otherwise one expert on the device at a time, the first released before
  // the second loads.
  std::vector<float> latents;
  std::unique_lock<std::mutex> resident_lock(ResidentMutex(),
                                             std::defer_lock);
  bool resident = false;
  const auto sample = [&]() -> bool {
    std::unique_ptr<Transformer21> sequential;
    Transformer21* high_expert = nullptr;  // resident experts
    Transformer21* low_expert = nullptr;
    Transformer21* dit = nullptr;
    bool on_high = false;
    float guidance = request.guidance;
    const ExpertKey high_key{request.high_dit_file, request.high_lora_file,
                             request.high_lora_strength};
    const ExpertKey low_key{request.low_dit_file, request.low_lora_file,
                            request.low_lora_strength};
    resident_lock.lock();
    ResidentExperts& kept = Resident();
    if (kept.loaded &&
        (!request.experts_resident ||
         !(kept.high_key == high_key && kept.low_key == low_key))) {
      kept.Reset();
      LogI2V("event=experts_released reason=" +
             std::string(request.experts_resident ? "other_files"
                                                  : "experts_resident_off"));
    }
    tel.experts_reused = kept.loaded;
    if (request.experts_resident && !kept.loaded) {
      // LoRAs are merged into the weights at load: no extra room.
      const std::uint64_t expert_bytes =
          FileBytes(high_key.dit) + FileBytes(low_key.dit);
      const std::uint64_t job_bytes =
          DirectoryBytes(request.t5_root / "text_encoder") +
          DirectoryBytes(request.vae_root / "vae");
      const std::uint64_t free_bytes = DeviceFreeBytes();
      const std::string numbers =
          " free_gib=" + Gib(free_bytes) + " experts_gib=" +
          Gib(expert_bytes) + " job_gib=" + Gib(job_bytes) +
          " margin_gib=" + Gib(kVaceExpertMarginBytes);
      if (KeepVaceExpertsResident(true, free_bytes, expert_bytes,
                                  job_bytes)) {
        const auto start = Clock::now();
        std::string load_error;
        kept.high = LoadExpert(high_key, &load_error);
        if (kept.high)
          kept.low = LoadExpert(low_key, &load_error);
        if (kept.high && kept.low) {
          kept.high_key = high_key;
          kept.low_key = low_key;
          kept.loaded = true;
          tel.dit_load_ms += Ms(start);
          TrackDevice(&tel.peak_device_bytes);
          LogI2V("event=experts_resident" + numbers);
        } else {
          kept.Reset();
          LogI2V("event=experts_sequential reason=load_failed" + numbers +
                 " error=\"" + load_error + "\"");
        }
      } else {
        LogI2V("event=experts_sequential reason=low_memory" + numbers);
      }
    }
    if (kept.loaded) {
      resident = true;
      high_expert = kept.high.get();
      low_expert = kept.low.get();
    } else {
      resident_lock.unlock();
    }
    tel.experts_resident = resident;
    // This job's prompts on an expert.
    const auto prepare = [&](Transformer21& expert, bool high) {
      const float expert_guidance = high ? high_guidance : request.guidance;
      const auto start = Clock::now();
      if (!expert.PrepareContext(0, text, TextEncoder::kMaxTokens, {},
                                 nullptr, nullptr, error) ||
          (expert_guidance > 1.0F &&
           !expert.PrepareContext(1, negative, TextEncoder::kMaxTokens, {},
                                  nullptr, nullptr, error)))
        return false;
      tel.context_ms += Ms(start);
      return true;
    };
    const auto use = [&](Transformer21* expert, bool high) {
      dit = expert;
      guidance = high ? high_guidance : request.guidance;
      on_high = high;
    };
    // Sequential: loads one expert (the other is released first).
    const auto load = [&](bool high) {
      sequential.reset();
      const auto start = Clock::now();
      sequential = LoadExpert(high ? high_key : low_key, error);
      if (!sequential)
        return false;
      tel.dit_load_ms += Ms(start);
      TrackDevice(&tel.peak_device_bytes);
      if (!prepare(*sequential, high))
        return false;
      use(sequential.get(), high);
      return true;
    };
    if (resident) {
      if (switch_at > 0 && !prepare(*high_expert, true))
        return false;
      if (switch_at < request.steps && !prepare(*low_expert, false))
        return false;
      use(switch_at > 0 ? high_expert : low_expert, switch_at > 0);
    } else if (!load(switch_at > 0)) {
      return false;
    }
    latents = GaussianNoise(16 * plane, request.seed);
    dump("noise0", latents, shape16);
    std::vector<float> input, velocity, uncond;
    for (int step = 0; step < request.steps; ++step) {
      if (cancelled())
        return false;
      if (on_high && step >= switch_at) {
        if (resident)
          use(low_expert, false);
        else if (!load(false))
          return false;
      }
      const auto step_start = Clock::now();
      if (!I2VModelInput(latents, condition_channels, &input)) {
        SetError(error, "Wan 2.2 I2V A14B: model input size");
        return false;
      }
      Transformer21Input in;
      in.latent = input;
      in.frames = lf;
      in.height = lh;
      in.width = lw;
      in.timestep = static_cast<float>(1000.0 * sigmas[step]);
      in.context_slot = 0;
      if (!dit->Forward(in, &velocity, nullptr, error))
        return false;
      if (guidance > 1.0F) {
        in.context_slot = 1;
        if (!dit->Forward(in, &uncond, nullptr, error))
          return false;
        for (std::size_t i = 0; i < velocity.size(); ++i)
          velocity[i] = uncond[i] + guidance * (velocity[i] - uncond[i]);
      }
      TrackDevice(&tel.peak_device_bytes);
      dump("step" + std::to_string(step) + "_velocity", velocity, shape16);
      const float sigma = static_cast<float>(sigmas[step]);
      const float next = static_cast<float>(sigmas[step + 1]);
      if (callbacks.preview.Wants(step + 1, request.steps)) {
        const auto preview_start = Clock::now();
        std::vector<float> x0(latents.size());
        for (std::size_t i = 0; i < latents.size(); ++i)
          x0[i] = latents[i] - sigma * velocity[i];
        callbacks.preview.publish({.step = step + 1,
                                   .steps = request.steps,
                                   .frames = lf,
                                   .height = lh,
                                   .width = lw,
                                   .channels = 16,
                                   .channel_major = true,
                                   .data = x0});
        tel.preview_ms += Ms(preview_start);
        ++tel.previews;
      }
      if (request.sampler == "euler") {
        for (std::size_t i = 0; i < latents.size(); ++i)
          latents[i] += (next - sigma) * velocity[i];
      } else {
        std::vector<float> fresh;
        if (next > 0.0F)
          fresh = GaussianNoise(latents.size(), request.seed + 1 + step);
        for (std::size_t i = 0; i < latents.size(); ++i) {
          const float x0 = latents[i] - sigma * velocity[i];
          latents[i] = next > 0.0F ? (1.0F - next) * x0 + next * fresh[i] : x0;
        }
      }
      tel.step_ms.push_back(Ms(step_start));
      progress("denoiser", step + 1, request.steps);
    }
    return true;
  };
  const bool sampled = sample();
  if (resident) {
    // A device error may leave the experts unusable: they load anew.
    if (!sampled && (error == nullptr || *error != "cancelled")) {
      Resident().Reset();
      LogI2V("event=experts_released reason=job_failed");
    }
    resident_lock.unlock();
  }
  if (!sampled)
    return false;
  dump("final", latents, shape16);
  if (final_latents != nullptr)
    *final_latents = latents;

  // 4. VAE decode.
  if (cancelled())
    return false;
  if (video != nullptr) {
    const auto start = Clock::now();
    auto decoder = VaeDecoder::Load(request.vae_root / "vae", error);
    if (!decoder ||
        !decoder->Decode(
            latents, lf, lh, lw, false, true, video,
            [&](int done, int total) { progress("video_vae", done, total); },
            nullptr, error))
      return false;
    TrackDevice(&tel.peak_device_bytes);
    tel.decode_ms = Ms(start);
  }
  tel.total_ms = Ms(total_start);
  return true;
}

}  // namespace gufo::models::wan22
