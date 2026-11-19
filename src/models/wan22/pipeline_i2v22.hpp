#ifndef GUFO_MODELS_WAN22_PIPELINE_I2V22_HPP_
#define GUFO_MODELS_WAN22_PIPELINE_I2V22_HPP_

// Written with AI assistance (Claude Code).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "src/models/wan22/i2v22_context.hpp"
#include "src/models/wan22/pipeline21.hpp"
#include "src/models/wan22/vae.hpp"

namespace gufo::models::wan22 {

/// Wan 2.2 I2V A14B (Wan-AI/Wan2.2-I2V-A14B) as ComfyUI runs WanImageToVideo
/// -> KSamplerAdvanced (high-noise expert, steps 0..k) -> KSamplerAdvanced
/// (low-noise expert, k..steps) -> VAEDecode:
///  - umT5 prompt (and negative prompt when a step uses guidance > 1);
///  - no CLIP vision: the A14B experts have no image cross-attention (a
///    clip_vision_output in the graph is ignored by ComfyUI as well);
///  - Wan 2.1 VAE condition: the start image, then grey (0 in [-1, 1]),
///    encoded and normalized, with 4 mask channels set on the first latent
///    frame (I2VConditionChannels) -> 36-channel model input;
///  - flow sampling over shifted sigmas (sigma = shift * t / (1 + (shift -
///    1) * t)), "euler" or "lcm" (x0 then fresh noise); guidance > 1 runs
///    classifier-free guidance with the negative prompt;
///  - the high-noise expert runs steps [0, switch), the low-noise one
///    [switch, steps): switch_step (KSamplerAdvanced end_at_step) or the
///    first step with 1000 * sigma < boundary * 1000 (0.900 for I2V);
///  - Wan 2.1 VAE decode.
/// Each expert is a Kijai FP8 scaled checkpoint (~15 GB) with an optional
/// LoRA of its own (lightx2v / Lightning I2V; its factors stay on the
/// device and are added at unpack): one expert may run with it and the
/// other without (high_lora_file / low_lora_file empty). With
/// `experts_resident` both load together and stay in the worker over jobs
/// (same files, LoRAs and strengths: no reload; another LoRA choice on
/// either expert reloads both) when the device has room
/// (VaceExpertsFit: the experts plus the largest job phase, umT5, VAE or one
/// expert sampling); otherwise one is on the device at a time, loaded from
/// disk every job. Resident experts free their job buffers (activations,
/// context slots) at the switch and after each job, before the VAE decode.
struct I2V22Request {
  std::filesystem::path vae_root;  // Diffusers root with vae/ (Wan 2.1 VAE)
  std::filesystem::path t5_root;   // Diffusers root with tokenizer/ and
                                   // text_encoder/ (umT5-XXL)
  std::filesystem::path high_dit_file;
  std::filesystem::path high_lora_file;  // optional
  float high_lora_strength{1.0F};
  std::filesystem::path low_dit_file;
  std::filesystem::path low_lora_file;  // optional
  float low_lora_strength{1.0F};
  std::string prompt;
  std::string negative_prompt;
  float guidance{1.0F};       // the low-noise expert's (and default) CFG
  float high_guidance{0.0F};  // 0: `guidance`
  std::string sampler{"euler"};  // "euler" or "lcm"
  int width{832};
  int height{480};
  int frames{81};
  int steps{4};
  float shift{static_cast<float>(kI2V22Shift)};
  std::uint64_t seed{42};
  double boundary{kI2V22Boundary};
  int switch_step{-1};  // -1: from boundary
  bool experts_resident{true};
  /// RGB8 [height, width, 3] at the video size.
  std::span<const std::uint8_t> first_frame;
};

struct I2V22Telemetry {
  double text_ms{0}, condition_ms{0}, dit_load_ms{0}, context_ms{0};
  double decode_ms{0}, total_ms{0};
  int switch_step{0};  // first low-noise step
  bool experts_resident{false};
  bool experts_reused{false};
  std::vector<double> step_ms;
  double preview_ms{0};
  int previews{0};
  std::size_t peak_device_bytes{0};
};

/// `callbacks.rendered_guide` is unused. Intermediates for `dump`: text,
/// negative_text, condition, noise0, step<i>_velocity, final.
bool GenerateI2V22(const I2V22Request& request, const I2V21Callbacks& callbacks,
                   std::vector<float>* final_latents, DecodedVideo* video,
                   I2V22Telemetry* telemetry, std::string* error);

/// Frees resident Wan 2.2 I2V A14B experts (worker shutdown, tests).
void ReleaseI2V22Experts();

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_PIPELINE_I2V22_HPP_
