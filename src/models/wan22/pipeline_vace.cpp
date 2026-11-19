// Written with AI assistance (Claude Code).
#include "src/models/wan22/pipeline_vace.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <list>
#include <memory>
#include <mutex>
#include <string>
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

/// The phases of one job beside the experts (vace_context.hpp).
struct JobNeed {
  std::uint64_t text_encoder{0}, vae{0}, sampling{0}, peak{0};
  std::string Numbers() const {
    return " umt5_gib=" + Gib(text_encoder) + " vae_gib=" + Gib(vae) +
           " sampling_gib=" + Gib(sampling) + " peak_gib=" + Gib(peak) +
           " reserve_gib=" + Gib(kVaceReserveBytes);
  }
};

/// Fun VACE A14B modules have at most 15 blocks; their text caches are small.
constexpr int kVaceBlocksBound = 16;

JobNeed JobPeak(const Vace21Request& request, int frames, int lh, int lw,
                bool cfg, int vace_blocks) {
  JobNeed need;
  need.text_encoder = VaceTextEncoderDeviceBytes(
      DirectoryBytes(request.t5_root / "text_encoder"));
  need.vae = DirectoryBytes(request.vae_root / "vae") + kVaceVaeWorkBytes;
  need.sampling = VaceSamplingBytes(frames, lh, lw, cfg ? 2 : 1, vace_blocks);
  need.peak = VaceJobPeakBytes(need.text_encoder, need.vae, need.sampling);
  return need;
}

/// event=<event> with the device memory now: hipMemGetInfo, the DXGI
/// dedicated budget and usage of this process (Windows iGPU) and what the
/// residency checks count on (free_gib).
void LogMemory(const std::string& event, const std::string& extra) {
  std::size_t hip_free = 0, hip_total = 0;
  (void)hipMemGetInfo(&hip_free, &hip_total);
  std::string line = "event=" + event + " hip_free_gib=" + Gib(hip_free) +
                     " hip_total_gib=" + Gib(hip_total);
  if (const auto memory = gufo::hip::QueryAdmissionMemory();
      memory && memory->local) {
    const auto& local = *memory->local;
    line += " dxgi_budget_gib=" + Gib(local.budget_bytes) +
            " dxgi_usage_gib=" + Gib(local.usage_bytes) + " dxgi_free_gib=" +
            Gib(local.budget_bytes > local.usage_bytes
                    ? local.budget_bytes - local.usage_bytes
                    : 0);
  }
  LogVace(line + " free_gib=" + Gib(DeviceFreeBytes()) + extra);
}

/// umT5 states of recent prompts (the real rows, FP32), per text encoder:
/// the same prompt (and the usual fixed negative prompt) skips umT5. The
/// encoder is deterministic, so a hit gives the states the first job had.
class PromptCache {
public:
  static constexpr std::size_t kEntries = 8;  // <= 8 x 8 MB host RAM

  bool Find(const std::string& key, std::vector<float>* states) {
    const std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it)
      if (it->first == key) {
        *states = it->second;
        entries_.splice(entries_.begin(), entries_, it);  // most recent
        return true;
      }
    return false;
  }
  void Put(const std::string& key, const std::vector<float>& states) {
    const std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it)
      if (it->first == key) {
        entries_.erase(it);
        break;
      }
    entries_.emplace_front(key, states);
    if (entries_.size() > kEntries)
      entries_.pop_back();
  }

private:
  std::mutex mutex_;
  std::list<std::pair<std::string, std::vector<float>>> entries_;
};

PromptCache& Prompts() {
  static PromptCache* cache = new PromptCache;  // never destroyed at exit
  return *cache;
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

  // Device memory before and after the job (event=job_memory_start /
  // job_memory_end), to measure what a job leaves behind.
  LogMemory("job_memory_start",
            std::string(" two_experts=") + (two_experts ? "1" : "0"));
  bool job_ok = false;
  struct EndLog {
    const Vace21Telemetry& tel;
    const bool& ok;
    ~EndLog() {
      LogMemory("job_memory_end",
                std::string(" ok=") + (ok ? "1" : "0") +
                    " experts_resident=" + (tel.experts_resident ? "1" : "0") +
                    " experts_reused=" + (tel.experts_reused ? "1" : "0") +
                    " peak_used_gib=" + Gib(tel.peak_device_bytes) +
                    " total_s=" + std::to_string(tel.total_ms / 1000.0));
    }
  } end_log{tel, job_ok};

  // 0. Resident experts from an earlier job: the umT5, VAE and sampling of
  // this job run beside them, so the largest of them needs room now (the
  // first job ran umT5 before the experts loaded). No room: release them.
  if (two_experts) {
    bool fits = true;
    std::string numbers;
    {
      const std::lock_guard<std::mutex> lock(ResidentMutex());
      ResidentExperts& kept = Resident();
      if (kept.loaded) {
        const JobNeed need =
            JobPeak(request, frames, lh, lw, cfg,
                    std::max(kept.high.vace->blocks(), kept.low.vace->blocks()));
        const std::uint64_t free_bytes = DeviceFreeBytes();
        fits = VaceJobFitsBesideResident(free_bytes, need.peak);
        numbers = " free_gib=" + Gib(free_bytes) + need.Numbers();
        LogVace(std::string("event=resident_check fits=") +
                (fits ? "1" : "0") + numbers);
      }
    }
    if (!fits)
      (void)ReleaseResidentExperts("low_memory_before_job" + numbers);
  }

  // 1. Prompts (umT5, 512 rows: the states of the real tokens, then zeros).
  // States of a recent identical prompt come from the cache; umT5 loads
  // only when one is missing and is freed right after encoding.
  std::vector<float> text, negative;
  {
    auto start = Clock::now();
    const std::string cache_root = request.t5_root.generic_string() + '|';
    const std::string text_key = cache_root + CleanPrompt(request.prompt);
    const std::string negative_key =
        cache_root + CleanPrompt(request.negative_prompt);
    std::vector<float> text_states, negative_states;
    const bool text_hit = Prompts().Find(text_key, &text_states);
    const bool negative_hit =
        !cfg || Prompts().Find(negative_key, &negative_states);
    if (!text_hit || !negative_hit) {
      const auto tokenizer = Tokenizer::Load(
          request.t5_root / "tokenizer" / "tokenizer.json", error);
      if (!tokenizer)
        return false;
      auto encoder =
          TextEncoder::Load(request.t5_root / "text_encoder", error);
      if (!encoder && two_experts &&
          ReleaseResidentExperts("text_encoder_load_failed")) {
        (void)hipGetLastError();  // the failed allocation, not sticky
        encoder = TextEncoder::Load(request.t5_root / "text_encoder", error);
      }
      if (!encoder)
        return false;
      TrackDevice(&tel.peak_device_bytes);
      const auto encode = [&](const std::string& prompt,
                              const std::string& key,
                              std::vector<float>* states) {
        if (!encoder->Encode(tokenizer->Encode(CleanPrompt(prompt)), states,
                             error))
          return false;
        Prompts().Put(key, *states);
        return true;
      };
      if ((!text_hit && !encode(request.prompt, text_key, &text_states)) ||
          (!negative_hit &&
           !encode(request.negative_prompt, negative_key, &negative_states)))
        return false;
    }  // umT5 freed here
    const auto pad = [](const std::vector<float>& states,
                        std::vector<float>* out) {
      out->assign(TextEncoder::kMaxTokens * TextEncoder::kModelWidth, 0.0F);
      std::copy_n(states.begin(), std::min(states.size(), out->size()),
                  out->begin());
    };
    pad(text_states, &text);
    if (cfg)
      pad(negative_states, &negative);
    LogVace(std::string("event=prompt_cache text=") +
            (text_hit ? "hit" : "miss") + " negative=" +
            (!cfg ? "unused" : negative_hit ? "hit" : "miss"));
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
        // Beside them: the largest phase of a job (umT5, VAE or one
        // expert sampling; each expert frees its job buffers after its
        // steps), not all of them at once.
        const JobNeed need =
            JobPeak(request, frames, lh, lw, cfg, kVaceBlocksBound);
        const std::uint64_t free_bytes = DeviceFreeBytes();
        const std::string numbers = " free_gib=" + Gib(free_bytes) +
                                    " experts_gib=" + Gib(expert_bytes) +
                                    need.Numbers();
        if (VaceExpertsFit(free_bytes, expert_bytes, need.peak)) {
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
      // Only the expert that runs first holds this job's buffers; the
      // low-noise one is set up at the switch, after the high-noise one
      // freed its own.
      if (!prepare(switch_at > 0 ? *high_expert : *low_expert,
                   switch_at > 0))
        return false;
      use(switch_at > 0 ? *high_expert : *low_expert, switch_at > 0);
      if (switch_at == 0 || switch_at >= request.steps)
        std::vector<float>().swap(context);  // no second expert to set up
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
          high_expert->vace->ReleaseJobBuffers();
          high_expert->dit->ReleaseJobBuffers();
          if (!prepare(*low_expert, false))
            return false;
          use(*low_expert, false);
          std::vector<float>().swap(context);
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
    } else {
      // The experts stay with their weights only: activations, context
      // slots and VACE buffers of this job go (2.2.6: ~12 GB left behind
      // blocked the next job's umT5).
      for (Expert* expert : {&Resident().high, &Resident().low}) {
        expert->vace->ReleaseJobBuffers();
        expert->dit->ReleaseJobBuffers();
      }
      LogMemory("experts_job_buffers_released", "");
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
  job_ok = true;
  return true;
}

}  // namespace gufo::models::wan22
