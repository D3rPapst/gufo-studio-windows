// Written with AI assistance (Claude Code).
#include "src/models/wan22/pipeline_vace.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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
#include "src/models/wan22/vace.hpp"
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

/// One transformer + its VACE module (the VACE blocks run on the DiT's
/// stream, so `vace` is destroyed first).
struct Expert {
  std::unique_ptr<Transformer21> dit;
  std::unique_ptr<Vace> vace;
};

/// What an expert was loaded from (LoRAs are merged at load).
struct ExpertKey {
  std::filesystem::path dit, vace, lora;
  float lora_strength{1.0F};
  bool operator==(const ExpertKey&) const = default;
};

/// Wan 2.2 Fun VACE experts kept in the worker over jobs. Never destroyed at
/// exit (the HIP runtime may be gone by then); ReleaseVaceExperts frees it.
struct ResidentExperts {
  bool loaded{false};
  /// Set once a job found no room beside the resident experts: this worker
  /// runs them one after the other from then on (Reset keeps it).
  bool refused{false};
  ExpertKey high_key, low_key;
  Expert high, low;
  void Reset() {
    for (Expert* expert : {&high, &low}) {
      expert->vace.reset();
      expert->dit.reset();
    }
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

/// What this process can still allocate on the device: the admission
/// reading, capped by the dedicated (DXGI local) budget left on a Windows
/// iGPU, whose shared segment the next job's umT5 cannot count on.
std::uint64_t DeviceFreeBytes() {
  if (const auto memory = gufo::hip::QueryAdmissionMemory())
    return VaceDeviceFreeBytes(
        memory->free_bytes, memory->local.has_value(),
        memory->local ? memory->local->budget_bytes : 0,
        memory->local ? memory->local->usage_bytes : 0);
  return 0;
}

std::string Gib(std::uint64_t bytes) {
  char text[32];
  std::snprintf(text, sizeof text, "%.1f",
                static_cast<double>(bytes) / static_cast<double>(1ULL << 30));
  return text;
}

void LogVace(const std::string& message) {
  gufo::server::Logger::Info("video", "wan22-fun-vace " + message);
}

/// What every job loads beside the experts (umT5, VAE).
std::uint64_t JobBytes(const Vace21Request& request) {
  return DirectoryBytes(request.t5_root / "text_encoder") +
         DirectoryBytes(request.vae_root / "vae");
}

/// Releases the resident experts for this job and keeps the worker on one
/// expert at a time from then on. True when experts were loaded.
bool ReleaseResidentExperts(const std::string& reason) {
  const std::lock_guard<std::mutex> lock(ResidentMutex());
  ResidentExperts& kept = Resident();
  if (!kept.loaded)
    return false;
  kept.refused = true;
  kept.Reset();
  LogVace("event=experts_released reason=" + reason);
  return true;
}

bool LoadExpert(const ExpertKey& key, Expert* expert, std::string* error) {
  expert->vace.reset();
  expert->dit.reset();
  expert->dit =
      Transformer21::Load(key.dit, key.lora, key.lora_strength, error);
  if (!expert->dit)
    return false;
  if (!expert->dit->text_only() ||
      expert->dit->in_channels() != 16) {  // not the Wan 2.2 I2V A14B
    SetError(error, "VACE needs a text-to-video 14B checkpoint, not " +
                        key.dit.string());
    expert->dit.reset();
    return false;
  }
  expert->vace = Vace::Load(key.vace, expert->dit->stream(), error);
  if (!expert->vace) {
    expert->dit.reset();
    return false;
  }
  return true;
}

}  // namespace

void ReleaseVaceExperts() {
  const std::lock_guard<std::mutex> lock(ResidentMutex());
  Resident().Reset();
}

bool GenerateVace21(const Vace21Request& request,
                    const I2V21Callbacks& callbacks,
                    std::vector<float>* final_latents, DecodedVideo* video,
                    Vace21Telemetry* telemetry, std::string* error) {
  Vace21Telemetry local;
  Vace21Telemetry& tel = telemetry != nullptr ? *telemetry : local;
  const auto total_start = Clock::now();
  const int w = request.width, h = request.height, n = request.frames;
  const std::size_t frame_pixels = static_cast<std::size_t>(w) * h;
  if (w % 16 != 0 || h % 16 != 0 || w < 16 || h < 16 || n < 1 ||
      (n - 1) % 4 != 0 || request.steps < 1 ||
      (request.sampler != "lcm" && request.sampler != "euler") ||
      !(request.guidance >= 1.0F) || request.control_frames < 0 ||
      request.control.size() != frame_pixels * 3 * request.control_frames ||
      request.mask_frames < 0 ||
      request.mask.size() != frame_pixels * request.mask_frames) {
    SetError(error,
             "Wan 2.1 VACE: width/height must be multiples of 16, frames "
             "4k+1, sampler lcm or euler, guidance >= 1, and the control "
             "video and mask must be at the video size");
    return false;
  }
  for (const auto& reference : request.references)
    if (reference.size() != frame_pixels * 3) {
      SetError(error, "Wan 2.1 VACE: reference images must be at the video "
                      "size");
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
  const bool two_experts = !request.high_dit_file.empty();
  const float high_guidance =
      request.high_guidance > 0.0F ? request.high_guidance : request.guidance;
  if (two_experts &&
      (request.high_vace_file.empty() || !(high_guidance >= 1.0F) ||
       !(request.boundary >= 0.0 && request.boundary <= 1.0))) {
    SetError(error, "Wan 2.2 Fun VACE: the high-noise expert needs its VACE "
                    "module, guidance >= 1 and a boundary in 0..1");
    return false;
  }
  const std::vector<double> sigmas = VaceSigmas(request.steps, request.shift);
  const int switch_at =
      two_experts
          ? VaceSwitchStep(sigmas, request.boundary, request.switch_step)
          : 0;
  tel.switch_step = switch_at;
  // CFG (negative prompt) when any step that runs uses guidance > 1.
  const bool cfg = (switch_at < request.steps && request.guidance > 1.0F) ||
                   (switch_at > 0 && high_guidance > 1.0F);
  const int refs = static_cast<int>(request.references.size());
  const int lf = 1 + (n - 1) / 4, lh = h / 8, lw = w / 8;
  const int frames = refs + lf;  // latent frames incl. references
  const std::size_t plane = static_cast<std::size_t>(lh) * lw;
  const auto shape16 = [&](int f) {
    return std::vector<std::size_t>{16, static_cast<std::size_t>(f),
                                    static_cast<std::size_t>(lh),
                                    static_cast<std::size_t>(lw)};
  };

  // 0. Resident experts from an earlier job: the umT5 and VAE of this job
  // load beside them, so they need room now, not only when the experts
  // loaded (the first job ran umT5 before them). No room: release them.
  if (two_experts) {
    bool fits = true;
    std::string numbers;
    {
      const std::lock_guard<std::mutex> lock(ResidentMutex());
      if (Resident().loaded) {
        const std::uint64_t job_bytes = JobBytes(request);
        const std::uint64_t free_bytes = DeviceFreeBytes();
        fits = VaceJobFitsBesideExperts(free_bytes, job_bytes);
        numbers = " free_gib=" + Gib(free_bytes) + " job_gib=" +
                  Gib(job_bytes) + " margin_gib=" +
                  Gib(kVaceExpertMarginBytes);
      }
    }
    if (!fits)
      (void)ReleaseResidentExperts("low_memory_before_job" + numbers);
  }

  // 1. Prompts (umT5, 512 rows: the states of the real tokens, then zeros).
  std::vector<float> text, negative;
  {
    auto start = Clock::now();
    const auto tokenizer = Tokenizer::Load(
        request.t5_root / "tokenizer" / "tokenizer.json", error);
    if (!tokenizer)
      return false;
    auto encoder = TextEncoder::Load(request.t5_root / "text_encoder", error);
    if (!encoder && two_experts &&
        ReleaseResidentExperts("text_encoder_load_failed")) {
      (void)hipGetLastError();  // the failed allocation, not sticky
      encoder = TextEncoder::Load(request.t5_root / "text_encoder", error);
    }
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

  // 2. VACE context: control video and mask (ComfyUI pads the video with
  // grey and the mask with ones), split, VAE-encoded; reference latents.
  std::vector<float> context;
  {
    auto start = Clock::now();
    std::vector<float> mask;
    if (request.mask_frames > 0) {
      mask.assign(frame_pixels * n, 1.0F);
      const std::size_t take =
          std::min(request.mask_frames, n) * frame_pixels;
      std::copy_n(request.mask.begin(), take, mask.begin());
    }
    std::vector<float> inactive_rgb, reactive_rgb;
    SplitVaceControl(request.control, std::min(request.control_frames, n),
                     mask, n, h, w, &inactive_rgb, &reactive_rgb);
    auto encoder = VaeEncoder::Load(request.vae_root / "vae", error);
    if (!encoder && two_experts &&
        ReleaseResidentExperts("vae_load_failed")) {
      (void)hipGetLastError();  // the failed allocation, not sticky
      encoder = VaeEncoder::Load(request.vae_root / "vae", error);
    }
    if (!encoder)
      return false;
    std::vector<float> inactive, reactive;
    if (!encoder->EncodeVideo(inactive_rgb, n, h, w, true, &inactive, error))
      return false;
    std::vector<float>().swap(inactive_rgb);
    if (!encoder->EncodeVideo(reactive_rgb, n, h, w, true, &reactive, error))
      return false;
    std::vector<float>().swap(reactive_rgb);
    std::vector<std::vector<float>> reference_latents(refs);
    for (int r = 0; r < refs; ++r) {
      std::vector<float> rgb(frame_pixels * 3);
      for (std::size_t i = 0; i < rgb.size(); ++i)
        rgb[i] = static_cast<float>(request.references[r][i]) / 127.5F - 1.0F;
      if (!encoder->EncodeVideo(rgb, 1, h, w, true, &reference_latents[r],
                                error))
        return false;
    }
    TrackDevice(&tel.peak_device_bytes);
    VaceContextInputs inputs;
    inputs.frames = n;
    inputs.height = h;
    inputs.width = w;
    inputs.inactive = inactive;
    inputs.reactive = reactive;
    inputs.mask = mask;
    for (const auto& latent : reference_latents)
      inputs.references.emplace_back(latent);
    context = BuildVaceContext(inputs);
    if (context.empty()) {
      SetError(error, "Wan 2.1 VACE: cannot build the VACE context");
      return false;
    }
    tel.condition_ms = Ms(start);
    progress("condition", 1, 1);
  }
  dump("vace_context", context,
       {static_cast<std::size_t>(Vace::kContextChannels),
        static_cast<std::size_t>(frames), static_cast<std::size_t>(lh),
        static_cast<std::size_t>(lw)});
  if (cancelled())
    return false;

  // 3. Transformer (T2V 14B; Wan 2.2: the high-noise expert first, then the
  // low-noise one) + VACE blocks and sampling. Wan 2.2 with
  // experts_resident and room on the device: both experts stay loaded over
  // jobs; otherwise one expert on the device at a time, the first released
  // before the second loads.
  std::vector<float> latents;
  std::unique_lock<std::mutex> resident_lock(ResidentMutex(),
                                             std::defer_lock);
  bool resident = false;
  const auto sample = [&]() -> bool {
    Expert sequential;
    Expert* high_expert = nullptr;  // resident experts
    Expert* low_expert = nullptr;
    Transformer21* dit = nullptr;
    Vace* vace = nullptr;
    bool on_high = false;
    float guidance = request.guidance;
    const ExpertKey high_key{request.high_dit_file, request.high_vace_file,
                             request.high_lora_file,
                             request.high_lora_strength};
    const ExpertKey low_key{request.dit_file, request.vace_file,
                            request.lora_file, request.lora_strength};
    if (two_experts) {
      resident_lock.lock();
      ResidentExperts& kept = Resident();
      if (kept.loaded && (!request.experts_resident ||
                          !(kept.high_key == high_key &&
                            kept.low_key == low_key))) {
        kept.Reset();
        LogVace("event=experts_released reason=" +
                std::string(request.experts_resident ? "other_files"
                                                     : "experts_resident_off"));
      }
      tel.experts_reused = kept.loaded;
      if (request.experts_resident && !kept.loaded && kept.refused) {
        LogVace("event=experts_sequential reason=low_memory_earlier_job");
      } else if (request.experts_resident && !kept.loaded) {
        // The FP8 linears keep their LoRA factors on the device (added
        // when each matrix is unpacked): they count with the experts.
        const std::uint64_t expert_bytes =
            FileBytes(high_key.dit) + FileBytes(high_key.vace) +
            FileBytes(high_key.lora) + FileBytes(low_key.dit) +
            FileBytes(low_key.vace) + FileBytes(low_key.lora);
        const std::uint64_t job_bytes = JobBytes(request);
        const std::uint64_t free_bytes = DeviceFreeBytes();
        const std::string numbers =
            " free_gib=" + Gib(free_bytes) + " experts_gib=" +
            Gib(expert_bytes) + " job_gib=" + Gib(job_bytes) +
            " margin_gib=" + Gib(kVaceExpertMarginBytes);
        if (KeepVaceExpertsResident(true, free_bytes, expert_bytes,
                                    job_bytes)) {
          const auto start = Clock::now();
          std::string load_error;
          if (LoadExpert(high_key, &kept.high, &load_error) &&
              LoadExpert(low_key, &kept.low, &load_error)) {
            kept.high_key = high_key;
            kept.low_key = low_key;
            kept.loaded = true;
            tel.dit_load_ms += Ms(start);
            TrackDevice(&tel.peak_device_bytes);
            LogVace("event=experts_resident" + numbers);
          } else {
            kept.Reset();
            LogVace("event=experts_sequential reason=load_failed" + numbers +
                    " error=\"" + load_error + "\"");
          }
        } else {
          LogVace("event=experts_sequential reason=low_memory" + numbers);
        }
      }
      if (kept.loaded) {
        resident = true;
        high_expert = &kept.high;
        low_expert = &kept.low;
      } else {
        resident_lock.unlock();
      }
    }
    tel.experts_resident = resident;
    // This job's VACE context and prompts on an expert.
    const auto prepare = [&](Expert& expert, bool high) {
      if (!expert.vace->SetContext(context, frames, lh, lw, error))
        return false;
      expert.vace->set_strength(request.strength);
      const float expert_guidance = high ? high_guidance : request.guidance;
      const auto start = Clock::now();
      if (!expert.dit->PrepareContext(0, text, TextEncoder::kMaxTokens, {},
                                      nullptr, nullptr, error) ||
          (expert_guidance > 1.0F &&
           !expert.dit->PrepareContext(1, negative, TextEncoder::kMaxTokens,
                                       {}, nullptr, nullptr, error)))
        return false;
      tel.context_ms += Ms(start);
      return true;
    };
    const auto use = [&](Expert& expert, bool high) {
      dit = expert.dit.get();
      vace = expert.vace.get();
      guidance = high ? high_guidance : request.guidance;
      on_high = high;
    };
    // Sequential: loads the high-noise expert or the low-noise / only model.
    const auto load = [&](bool high) {
      sequential.vace.reset();
      sequential.dit.reset();
      const auto start = Clock::now();
      if (!LoadExpert(high ? high_key : low_key, &sequential, error))
        return false;
      tel.dit_load_ms += Ms(start);
      TrackDevice(&tel.peak_device_bytes);
      if (!prepare(sequential, high))
        return false;
      use(sequential, high);
      return true;
    };
    if (resident) {
      if (switch_at > 0 && !prepare(*high_expert, true))
        return false;
      if (switch_at < request.steps && !prepare(*low_expert, false))
        return false;
      use(switch_at > 0 ? *high_expert : *low_expert, switch_at > 0);
      std::vector<float>().swap(context);
    } else {
      if (!load(switch_at > 0))
        return false;
      if (!two_experts || switch_at == 0 || switch_at >= request.steps)
        std::vector<float>().swap(context);  // no second expert to set up
    }
    latents = GaussianNoise(16 * frames * plane, request.seed);
    dump("noise0", latents, shape16(frames));
    std::vector<float> velocity, uncond;
    for (int step = 0; step < request.steps; ++step) {
      if (cancelled())
        return false;
      if (on_high && step >= switch_at) {
        if (resident) {
          use(*low_expert, false);
        } else {
          if (!load(false))
            return false;
          std::vector<float>().swap(context);
        }
      }
      const auto step_start = Clock::now();
      Transformer21Input in;
      in.latent = latents;
      in.frames = frames;
      in.height = lh;
      in.width = lw;
      in.timestep = static_cast<float>(1000.0 * sigmas[step]);
      in.hook = vace;
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
      dump("step" + std::to_string(step) + "_velocity", velocity,
           shape16(frames));
      const float sigma = static_cast<float>(sigmas[step]);
      const float next = static_cast<float>(sigmas[step + 1]);
      if (callbacks.preview.Wants(step + 1, request.steps)) {
        // x0 = x - sigma * v of the video frames (references left out).
        const auto preview_start = Clock::now();
        std::vector<float> x0(16 * lf * plane);
        for (int c = 0; c < 16; ++c)
          for (std::size_t i = 0; i < lf * plane; ++i) {
            const std::size_t at = (c * frames + refs) * plane + i;
            x0[c * lf * plane + i] = latents[at] - sigma * velocity[at];
          }
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
      LogVace("event=experts_released reason=job_failed");
    }
    resident_lock.unlock();
  }
  if (!sampled)
    return false;
  // TrimVideoLatent: drop the reference frames.
  if (refs > 0) {
    std::vector<float> trimmed(16 * lf * plane);
    for (int c = 0; c < 16; ++c)
      std::copy_n(latents.begin() + (c * frames + refs) * plane, lf * plane,
                  trimmed.begin() + c * lf * plane);
    latents = std::move(trimmed);
  }
  dump("final", latents, shape16(lf));
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
