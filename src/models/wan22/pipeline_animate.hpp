#ifndef GUFO_MODELS_WAN22_PIPELINE_ANIMATE_HPP_
#define GUFO_MODELS_WAN22_PIPELINE_ANIMATE_HPP_

// Written with AI assistance (Claude Code).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "src/models/wan22/pipeline21.hpp"
#include "src/models/wan22/vae.hpp"

namespace gufo::models::wan22 {

/// Wan 2.2 Animate 14B (pose-driven animation of a reference character, or
/// character replacement in a video) as ComfyUI runs WanAnimateToVideo ->
/// KSampler -> TrimVideoLatent -> VAEDecode on AnimateWanModel:
///  - umT5 prompt (+ negative for CFG), CLIP-H states of the reference
///    image (clip_vision_output; gufo always passes them);
///  - per chunk (animate_context.hpp): the reference latent in front of the
///    condition video (grey, continue_motion frames, background) with the
///    4-channel mask, pose latents into the pose patch embedding, face crops
///    through the motion / face encoders into the face adapter (CFG negative:
///    face pixels -1);
///  - flow sampling over shifted sigmas, "lcm" or "euler" (as VACE), the
///    reference latent trimmed, Wan 2.1 VAE decode;
///  - videos longer than `length` run as the chain of chunks ComfyUI's
///    long-video workflows wire (PlanAnimateChunks): each continues from
///    the last `continue_frames` decoded frames and drops them.
struct Animate22Request {
  std::filesystem::path vae_root;   // Diffusers Wan 2.1 root with vae/
  std::filesystem::path clip_root;  // Diffusers root with image_encoder/
  std::filesystem::path t5_root;    // Diffusers root with tokenizer/ and
                                    // text_encoder/ (umT5-XXL)
  std::filesystem::path dit_file;   // Kijai Wan2_2-Animate-14B FP8 file
  std::filesystem::path lora_file;  // optional (lightx2v I2V)
  float lora_strength{1.0F};
  /// Optional relight LoRA (Kijai's WanAnimate_relight_lora, ComfyUI's
  /// LoraLoaderModelOnly in replacement workflows: the character takes the
  /// light of the background), added on top of `lora_file`.
  std::filesystem::path relight_lora_file;
  float relight_lora_strength{1.0F};
  std::string prompt;
  std::string negative_prompt;
  float guidance{1.0F};
  std::string sampler{"lcm"};  // "lcm" or "euler"
  int width{832};
  int height{480};
  int frames{77};  // frames of the result (4k + 1)
  int length{77};  // frames per chunk (4k + 1, at most frames)
  int continue_frames{5};  // overlap of chained chunks (4k + 1 < length)
  int steps{4};
  float shift{5.0F};
  std::uint64_t seed{42};
  /// Reference image RGB8 [height, width, 3] at the video size; empty:
  /// black, as the node without reference_image.
  std::span<const std::uint8_t> reference;
  /// Videos RGB8 [n, height, width, 3] at the video size (face: [n, 512,
  /// 512, 3]), counted from their first frame; empty = not given.
  std::span<const std::uint8_t> pose;
  int pose_frames{0};
  std::span<const std::uint8_t> face;
  int face_frames{0};
  std::span<const std::uint8_t> background;
  int background_frames{0};
  /// Character mask [mask_frames, height, width] 0..1 (1 = generated).
  std::span<const float> mask;
  int mask_frames{0};
  /// The node's continue_motion (first chunk only) and seek.
  std::span<const std::uint8_t> continue_motion;
  int continue_motion_frames{0};
  int continue_motion_max_frames{5};
  int video_frame_offset{0};
  /// Drop the first chunk's trim_image frames (its continue_motion frames)
  /// from the result, as ImageFromBatch(batch_index = trim_image) does.
  bool trim_continue{false};
};

struct Animate22Telemetry {
  double text_ms{0}, clip_ms{0}, condition_ms{0}, face_ms{0};
  double dit_load_ms{0}, context_ms{0}, decode_ms{0}, total_ms{0};
  std::vector<double> step_ms;
  double preview_ms{0};
  int previews{0};
  int chunks{0};
  std::size_t peak_device_bytes{0};
};

/// `video` receives RGB8 frames (frames of the request). Intermediates for
/// `callbacks.dump`: text, negative_text, clip_states, chunk<k>_condition
/// (the 20 mask + latent channels), chunk<k>_pose, chunk<k>_motion_tokens,
/// chunk<k>_noise0, chunk<k>_step<i>_velocity, chunk<k>_final.
bool GenerateAnimate22(const Animate22Request& request,
                       const I2V21Callbacks& callbacks, DecodedVideo* video,
                       Animate22Telemetry* telemetry, std::string* error);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_PIPELINE_ANIMATE_HPP_
