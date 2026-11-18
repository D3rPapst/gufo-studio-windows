// Written with AI assistance (Claude Code).
#include "src/models/wan22/pipeline_animate.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <utility>

#include "src/models/wan22/animate.hpp"
#include "src/models/wan22/animate_context.hpp"
#include "src/models/wan22/clip_vision.hpp"
#include "src/models/wan22/pipeline.hpp"
#include "src/models/wan22/text_encoder.hpp"
#include "src/models/wan22/tokenizer.hpp"
#include "src/models/wan22/transformer21.hpp"

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

bool Sized(std::span<const std::uint8_t> frames, int count,
           std::size_t frame_values) {
  return count >= 0 && frames.size() == frame_values * count;
}

}  // namespace

bool GenerateAnimate22(const Animate22Request& request,
                       const I2V21Callbacks& callbacks, DecodedVideo* video,
                       Animate22Telemetry* telemetry, std::string* error) {
  Animate22Telemetry local;
  Animate22Telemetry& tel = telemetry != nullptr ? *telemetry : local;
  const auto total_start = Clock::now();
  const int w = request.width, h = request.height, n = request.length;
  const std::size_t frame_values = static_cast<std::size_t>(w) * h * 3;
  const std::size_t face_values =
      static_cast<std::size_t>(kAnimateFaceSize) * kAnimateFaceSize * 3;
  if (w % 16 != 0 || h % 16 != 0 || w < 16 || h < 16 || n < 5 ||
      (n - 1) % 4 != 0 || request.frames < 1 || request.steps < 1 ||
      (request.sampler != "lcm" && request.sampler != "euler") ||
      !(request.guidance >= 1.0F) ||
      (!request.reference.empty() && request.reference.size() != frame_values) ||
      !Sized(request.pose, request.pose_frames, frame_values) ||
      !Sized(request.face, request.face_frames, face_values) ||
      !Sized(request.background, request.background_frames, frame_values) ||
      !Sized(request.continue_motion, request.continue_motion_frames,
             frame_values) ||
      request.mask_frames < 0 ||
      request.mask.size() !=
          static_cast<std::size_t>(w) * h * request.mask_frames) {
    SetError(error,
             "Wan 2.2 Animate: width/height must be multiples of 16, the "
             "chunk length 4k+1 (>= 5), sampler lcm or euler, guidance >= 1, "
             "and every input at the video size (face crops 512x512)");
    return false;
  }
  const bool chained = request.frames > n;
  std::vector<AnimateChunkSlot> slots;
  if (chained) {
    slots = PlanAnimateChunks(request.frames, n, request.continue_frames);
    if (slots.empty()) {
      SetError(error, "Wan 2.2 Animate: continue_frames must be 4k+1 and "
                      "shorter than the chunk length");
      return false;
    }
  } else {
    slots.push_back({});
  }
  const int chunks = static_cast<int>(slots.size());
  tel.chunks = chunks;
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
  const bool cfg = request.guidance > 1.0F;
  const int lf = 1 + (n - 1) / 4, lh = h / 8, lw = w / 8;
  const int frames = 1 + lf;  // latent frames incl. the reference
  const std::size_t lplane = static_cast<std::size_t>(lh) * lw;
  const std::size_t plane = frames * lplane;
  const auto shape = [&](int channels, int f) {
    return std::vector<std::size_t>{static_cast<std::size_t>(channels),
                                    static_cast<std::size_t>(f),
                                    static_cast<std::size_t>(lh),
                                    static_cast<std::size_t>(lw)};
  };

  // 1. Prompts (umT5, 512 rows: the real tokens' states, then zeros).
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

  // 2. Reference image: black without one (the node's zeros), CLIP states.
  std::vector<std::uint8_t> reference(request.reference.begin(),
                                      request.reference.end());
  if (reference.empty())
    reference.assign(frame_values, 0);
  std::vector<float> clip_states;
  {
    auto start = Clock::now();
    auto clip = ClipVision::Load(request.clip_root / "image_encoder", error);
    if (!clip)
      return false;
    const auto pixels = ClipVision::Preprocess(reference, w, h);
    if (!clip->Encode(pixels, &clip_states, error))
      return false;
    TrackDevice(&tel.peak_device_bytes);
    tel.clip_ms = Ms(start);
  }
  dump("clip_states", clip_states, {ClipVision::kTokens, ClipVision::kWidth});
  if (cancelled())
    return false;

  // 3. VAE (kept for every chunk), the reference latent.
  auto encoder = VaeEncoder::Load(request.vae_root / "vae", error);
  if (!encoder)
    return false;
  std::vector<float> reference_latent;
  {
    std::vector<float> rgb(frame_values);
    for (std::size_t i = 0; i < rgb.size(); ++i)
      rgb[i] = static_cast<float>(reference[i]) / 127.5F - 1.0F;
    if (!encoder->EncodeVideo(rgb, 1, h, w, true, &reference_latent, error))
      return false;
  }
  auto decoder = VaeDecoder::Load(request.vae_root / "vae", error);
  if (!decoder)
    return false;
  TrackDevice(&tel.peak_device_bytes);

  // 4. Transformer (Animate 14B: I2V 14B keys) + Animate additions.
  auto start = Clock::now();
  auto dit = Transformer21::Load(request.dit_file, request.lora_file,
                                 request.lora_strength, error);
  if (!dit)
    return false;
  if (dit->text_only() || dit->in_channels() != Transformer21::kInChannels) {
    SetError(error, "Wan 2.2 Animate needs the Animate 14B checkpoint (I2V "
                    "layout, 36 input channels), not " +
                        request.dit_file.string());
    return false;
  }
  auto animate = Animate::Load(request.dit_file, dit->stream(), error);
  if (!animate)
    return false;
  tel.dit_load_ms = Ms(start);
  TrackDevice(&tel.peak_device_bytes);
  start = Clock::now();
  if (!dit->PrepareContext(0, text, TextEncoder::kMaxTokens, clip_states,
                           nullptr, nullptr, error) ||
      (cfg && !dit->PrepareContext(1, negative, TextEncoder::kMaxTokens,
                                   clip_states, nullptr, nullptr, error)))
    return false;
  tel.context_ms = Ms(start);

  std::vector<double> sigmas;
  for (int i = 0; i < request.steps; ++i) {
    const double t = 1.0 - static_cast<double>(i) / request.steps;
    sigmas.push_back(request.shift * t / (1.0 + (request.shift - 1.0) * t));
  }
  sigmas.push_back(0.0);

  if (video != nullptr) {
    *video = {};
    video->height = h;
    video->width = w;
  }
  std::vector<std::uint8_t> previous;  // the last chunk's decoded frames
  int offset = request.video_frame_offset;
  int kept = 0;
  for (int k = 0; k < chunks; ++k) {
    const std::string tag = "chunk" + std::to_string(k) + "_";
    // 5. The chunk's conditioning (WanAnimateToVideo).
    start = Clock::now();
    AnimateChunkInputs inputs;
    inputs.length = n;
    inputs.height = h;
    inputs.width = w;
    inputs.video_frame_offset = offset;
    inputs.pose = request.pose;
    inputs.pose_frames = request.pose_frames;
    inputs.face = request.face;
    inputs.face_frames = request.face_frames;
    inputs.background = request.background;
    inputs.background_frames = request.background_frames;
    inputs.mask = request.mask;
    inputs.mask_frames = request.mask_frames;
    if (k == 0) {
      inputs.continue_motion = request.continue_motion;
      inputs.continue_frames = request.continue_motion_frames;
      inputs.continue_motion_max_frames = request.continue_motion_max_frames;
    } else {
      inputs.continue_motion = previous;
      inputs.continue_frames = n;
      inputs.continue_motion_max_frames = request.continue_frames;
    }
    AnimateChunkPlan plan;
    if (!BuildAnimateChunk(inputs, &plan)) {
      SetError(error, "Wan 2.2 Animate: cannot build the chunk conditioning");
      return false;
    }
    offset = plan.video_frame_offset;
    // Model input channels 16..35: mask (4) and [reference | condition]
    // latents (16).
    std::vector<float> input(36 * plane, 0.0F);
    std::copy(plan.mask.begin(), plan.mask.end(), input.begin() + 16 * plane);
    {
      std::vector<float> condition;
      if (!encoder->EncodeVideo(plan.condition, n, h, w, true, &condition,
                                error))
        return false;
      std::vector<float>().swap(plan.condition);
      for (int c = 0; c < 16; ++c) {
        float* target = input.data() + (20 + c) * plane;
        std::copy_n(reference_latent.begin() + c * lplane, lplane, target);
        std::copy_n(condition.begin() + c * lf * lplane, lf * lplane,
                    target + lplane);
      }
    }
    std::vector<float> pose_latent;
    if (!plan.pose.empty()) {
      if (!encoder->EncodeVideo(plan.pose, n, h, w, true, &pose_latent,
                                error))
        return false;
      std::vector<float>().swap(plan.pose);
    }
    TrackDevice(&tel.peak_device_bytes);
    tel.condition_ms += Ms(start);
    dump(tag + "condition",
         std::vector<float>(input.begin() + 16 * plane, input.end()),
         shape(20, frames));
    if (!pose_latent.empty())
      dump(tag + "pose", pose_latent, shape(16, lf));
    start = Clock::now();
    if (!animate->SetPose(pose_latent, pose_latent.empty() ? 0 : lf, lh, lw,
                          error) ||
        !animate->SetFace(0, plan.face, plan.face_count, frames, error) ||
        (cfg && !animate->SetUniformFace(1, -1.0F, plan.face_count, frames,
                                         error)))
      return false;
    std::vector<float>().swap(plan.face);
    tel.face_ms += Ms(start);
    if (plan.face_count > 0)
      dump(tag + "motion_tokens", animate->motion_tokens(0),
           {static_cast<std::size_t>(frames), Animate::kMotionTokens,
            Transformer21::kModelWidth});
    if (cancelled())
      return false;

    // 6. Sampling.
    std::vector<float> latents = GaussianNoise(16 * plane, request.seed);
    dump(tag + "noise0", latents, shape(16, frames));
    std::vector<float> velocity, uncond;
    for (int step = 0; step < request.steps; ++step) {
      if (cancelled())
        return false;
      const auto step_start = Clock::now();
      std::copy(latents.begin(), latents.end(), input.begin());
      Transformer21Input in;
      in.latent = input;
      in.frames = frames;
      in.height = lh;
      in.width = lw;
      in.timestep = static_cast<float>(1000.0 * sigmas[step]);
      in.hook = animate.get();
      in.context_slot = 0;
      if (!dit->Forward(in, &velocity, nullptr, error))
        return false;
      if (cfg) {
        in.context_slot = 1;
        if (!dit->Forward(in, &uncond, nullptr, error))
          return false;
        for (std::size_t i = 0; i < velocity.size(); ++i)
          velocity[i] = uncond[i] + request.guidance * (velocity[i] - uncond[i]);
      }
      TrackDevice(&tel.peak_device_bytes);
      dump(tag + "step" + std::to_string(step) + "_velocity", velocity,
           shape(16, frames));
      const float sigma = static_cast<float>(sigmas[step]);
      const float next = static_cast<float>(sigmas[step + 1]);
      if (callbacks.preview.Wants(k * request.steps + step + 1,
                                  chunks * request.steps)) {
        // x0 = x - sigma * v of the video frames (reference left out).
        const auto preview_start = Clock::now();
        std::vector<float> x0(16 * lf * lplane);
        for (int c = 0; c < 16; ++c)
          for (std::size_t i = 0; i < lf * lplane; ++i) {
            const std::size_t at = c * plane + lplane + i;
            x0[c * lf * lplane + i] = latents[at] - sigma * velocity[at];
          }
        callbacks.preview.publish({.step = k * request.steps + step + 1,
                                   .steps = chunks * request.steps,
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
      progress("denoiser", k * request.steps + step + 1,
               chunks * request.steps);
    }
    // TrimVideoLatent (trim_latent = 1): drop the reference frame.
    std::vector<float> trimmed(16 * lf * lplane);
    for (int c = 0; c < 16; ++c)
      std::copy_n(latents.begin() + c * plane + lplane, lf * lplane,
                  trimmed.begin() + c * lf * lplane);
    dump(tag + "final", trimmed, shape(16, lf));

    // 7. VAE decode; chained chunks drop their continue_motion frames
    // (trim_image) and the result is cut to the requested frames.
    if (cancelled())
      return false;
    start = Clock::now();
    DecodedVideo chunk;
    if (!decoder->Decode(
            trimmed, lf, lh, lw, false, true, &chunk,
            [&](int done, int total) { progress("video_vae", done, total); },
            nullptr, error))
      return false;
    TrackDevice(&tel.peak_device_bytes);
    tel.decode_ms += Ms(start);
    if (chunk.rgb8.size() != frame_values * n) {
      SetError(error, "Wan 2.2 Animate: unexpected decoded frame count");
      return false;
    }
    const int keep_from =
        (k > 0 || request.trim_continue) ? plan.trim_image : 0;
    const int take = std::min(n - keep_from, request.frames - kept);
    if (video != nullptr && take > 0) {
      video->rgb8.insert(video->rgb8.end(),
                         chunk.rgb8.begin() + keep_from * frame_values,
                         chunk.rgb8.begin() + (keep_from + take) * frame_values);
      video->frames += take;
    }
    kept += std::max(0, take);
    previous = std::move(chunk.rgb8);
  }
  tel.total_ms = Ms(total_start);
  return true;
}

}  // namespace gufo::models::wan22
