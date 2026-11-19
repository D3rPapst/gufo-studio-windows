// MiniMax H3 Ref2VA references (ComfyUI's MiniMaxH3ReferenceToVideo):
// reference images, videos (with their soundtrack) and audio clips, sized
// and presented to the text encoder as "<Picture i>: ", "<Audio j>: " and
// "<Video k>: " blocks (videos at 2 fps in two-frame "<T.T seconds>"
// blocks), and encoded into the reference blocks of the packed sequence.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MINIMAX_H3_REFERENCES_HPP_
#define GUFO_MODELS_MINIMAX_H3_REFERENCES_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/models/minimax_h3/audio_vae.hpp"
#include "src/models/minimax_h3/control.hpp"
#include "src/models/minimax_h3/denoiser.hpp"
#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/sampling.hpp"
#include "src/models/minimax_h3/video_vae_encoder.hpp"

namespace gufo::minimax_h3 {

inline constexpr std::size_t kH3MaximumReferenceImages = 9;
inline constexpr std::size_t kH3MaximumReferenceVideos = 3;
inline constexpr std::size_t kH3MaximumReferenceAudios = 3;

/// A reference as decoded: an image (frames = 1) or a video (RGB8 frames at
/// 24 fps) with an optional soundtrack, or a standalone audio (no frames).
struct GenerationReference {
  ReferenceKind kind{ReferenceKind::kImage};
  ControlFrames frames;
  std::optional<AudioWaveform> audio;
};

/// One text-encoder presentation entry: a label, then a vision block when
/// `vision` >= 0 (index into ReferencePlan::vision_images).
struct PresentationEntry {
  std::string label;
  int vision{-1};
};

struct ReferencePlan {
  struct Item {
    const GenerationReference* source{nullptr};
    ReferencePlacement placement;
    int width{0};   // VAE canvas of an image or video
    int height{0};
    int frames{0};  // kept pixel frames (17k + 5 for a video, 1 for an image)
    std::vector<std::uint8_t> rgb;  // frames x height x width x 3
  };
  std::vector<Item> items;  // images, then videos, then audios
  // Vision blocks (images, two-frame video blocks); their pixels live in
  // `vision_pixels`.
  std::vector<std::vector<std::uint8_t>> vision_pixels;
  std::vector<KeyframeImage> vision_images;
  std::vector<PresentationEntry> presentation;
};

/// Sizes the references like ComfyUI: images scaled down (never up) to the
/// generation's pixel area ("match") or to a 2048 short edge ("max") on the
/// 32 grid; videos on the 768-short-edge canvas (at most 768 x 1344 and not
/// above their own size), cut to the video's frame count and down to
/// 17k + 5 frames (at least 5), sampled for the text encoder every 12th
/// frame. The presentation follows images, videos (a soundtrack's
/// "<Audio j>" before its "<Video k>"), then audios.
[[nodiscard]] bool PlanReferences(const std::vector<GenerationReference>& references,
                                  const GenerationGeometry& geometry,
                                  bool image_size_max, ReferencePlan* plan,
                                  std::string* error = nullptr);

/// Encodes the plan's pictures with the VisualVAE (posterior sample under
/// the keyframe seed, 0.999 anchor noise under `seed`) and the sounds with
/// the AudioVAE into the denoiser's reference blocks.
[[nodiscard]] bool EncodeReferences(const ModelInventory& inventory,
                                    const ReferencePlan& plan,
                                    std::uint64_t seed,
                                    const CancellationToken* cancellation,
                                    std::vector<ReferenceLatent>* latents,
                                    VideoVaeEncoderTelemetry* telemetry,
                                    std::string* error = nullptr);

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_REFERENCES_HPP_
