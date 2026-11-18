#ifndef GUFO_MODELS_WAN22_ANIMATE_CONTEXT_HPP_
#define GUFO_MODELS_WAN22_ANIMATE_CONTEXT_HPP_

// Written with AI assistance (Claude Code).
// Host-side Wan 2.2 Animate conditioning (no device code; unit-tested in
// tests/models/wan22/animate_context_test.cpp).

#include <cstdint>
#include <span>
#include <vector>

namespace gufo::models::wan22 {

/// Side of the square face crops the motion encoder reads.
inline constexpr int kAnimateFaceSize = 512;

/// One WanAnimateToVideo call (ComfyUI comfy_extras/nodes_wan.py) on inputs
/// the runner has already fitted: videos RGB8 [frames, height, width, 3] at
/// the video size (the face video at 512 x 512), all counted from their
/// first frame (`video_frame_offset` seeks in them here).
struct AnimateChunkInputs {
  int length{77};  // pixel frames of the chunk, 4k + 1
  int height{480};
  int width{832};
  int video_frame_offset{0};
  int continue_motion_max_frames{5};
  std::span<const std::uint8_t> pose;
  int pose_frames{0};
  std::span<const std::uint8_t> face;  // [face_frames, 512, 512, 3]
  int face_frames{0};
  std::span<const std::uint8_t> background;
  int background_frames{0};
  /// Character mask [mask_frames, height, width] in 0..1 (1 = the character,
  /// generated; 0 = background kept). One frame stands for every frame.
  std::span<const float> mask;
  int mask_frames{0};
  /// Frames to continue from (the previous chunk's output); its last
  /// continue_motion_max_frames frames are used.
  std::span<const std::uint8_t> continue_motion;
  int continue_frames{0};
};

/// What WanAnimateToVideo builds, ready for the VAE and the transformer.
struct AnimateChunkPlan {
  int latent_frames{0};  // 1 + (length - 1) / 4 (video, without reference)
  /// The node's integer outputs.
  int trim_latent{1};          // the reference latent frame in front
  int trim_image{0};           // max(0, ref_motion_latent_length * 4 - 3)
  int video_frame_offset{0};   // offset of the next chunk (offset + length)
  int offset{0};               // the seek used here (after continue_motion)
  int continue_frames{0};      // continue_motion frames placed in front
  int ref_motion_latent_length{0};
  /// The condition video (continue_motion frames, the background where
  /// given, else grey) in [-1, 1], [length, height, width, 3], for
  /// VaeEncoder::EncodeVideo (normalized latents go behind the reference
  /// latent).
  std::vector<float> condition;
  /// The pose video (last frame repeated up to length) in [-1, 1],
  /// [length, height, width, 3]; empty without pose input.
  std::vector<float> pose;
  /// Face crops in [-1, 1], frame-major [face_count, 3, 512, 512] (no
  /// padding: the motion vectors of missing frames are zero); empty without
  /// face input.
  std::vector<float> face;
  int face_count{0};
  /// The model's 4 mask channels as the transformer reads them (ComfyUI's
  /// concat_mask after WAN21.concat_cond's 1 - mask): [4, 1 + latent_frames,
  /// height / 8, width / 8], 1 = known (reference frame, continue_motion
  /// frames, kept background), 0 = generated.
  std::vector<float> mask;
};

/// Builds one chunk; false on inconsistent sizes (length not 4k + 1, width
/// or height not multiples of 16, spans not matching their frame counts).
[[nodiscard]] bool BuildAnimateChunk(const AnimateChunkInputs& inputs,
                                     AnimateChunkPlan* plan);

/// One chunk of a long animation: the node's video_frame_offset input,
/// how many frames of the previous output it continues from, and from
/// which of its own output frames on the result is kept.
struct AnimateChunkSlot {
  int video_frame_offset{0};
  int continue_frames{0};
  int keep_from{0};
};

/// The chain of WanAnimateToVideo nodes ComfyUI's long-video workflows wire
/// (each node's video_frame_offset output into the next, continue_motion
/// from the previous decode, ImageFromBatch dropping trim_image frames)
/// covering `total_frames` frames with chunks of `length` (4k + 1) and
/// `continue_frames` (4k + 1, < length) frames of overlap. Empty on invalid
/// arguments. The kept frames of all chunks, cut to total_frames, are the
/// video.
[[nodiscard]] std::vector<AnimateChunkSlot> PlanAnimateChunks(
    int total_frames, int length, int continue_frames);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_ANIMATE_CONTEXT_HPP_
