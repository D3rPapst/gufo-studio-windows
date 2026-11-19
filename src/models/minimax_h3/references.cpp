// MiniMax H3 Ref2VA references; see references.hpp.
// Written with AI assistance (Claude Code).
#include "src/models/minimax_h3/references.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <span>
#include <utility>

namespace gufo::minimax_h3 {
namespace {

constexpr int kCanvasMultiple = 32;
constexpr int kBaseShortEdge = 768;
constexpr double kMaximumPixels = 768.0 * 1344.0;
constexpr double kReferenceImageShortEdge = 2048.0;
constexpr int kQwenStride = 12;  // 24 fps sampled at 2 fps

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

// Python's round() (half to even) onto the canvas grid, at least one cell.
int Grid(double value) {
  return std::max(kCanvasMultiple,
                  static_cast<int>(std::nearbyint(value / kCanvasMultiple)) *
                      kCanvasMultiple);
}

// ComfyUI's adapt_canvas: 768 short edge, area cap 768 x 1344, per-axis
// round to 32.
std::pair<int, int> AdaptCanvas(int width, int height) {
  const double ratio = static_cast<double>(width) / height;
  double nominal_width = 0.0;
  double nominal_height = 0.0;
  if (ratio >= 1.0) {
    nominal_width = kBaseShortEdge * ratio;
    nominal_height = kBaseShortEdge;
  } else {
    nominal_width = kBaseShortEdge;
    nominal_height = kBaseShortEdge / ratio;
  }
  if (nominal_width * nominal_height > kMaximumPixels) {
    const double scale =
        std::sqrt(kMaximumPixels / (nominal_width * nominal_height));
    nominal_width *= scale;
    nominal_height *= scale;
  }
  return {Grid(nominal_width), Grid(nominal_height)};
}

std::span<const std::uint8_t> Frame(const ControlFrames& frames, int index) {
  const std::size_t bytes = static_cast<std::size_t>(frames.width) *
                            static_cast<std::size_t>(frames.height) * 3U;
  return std::span<const std::uint8_t>(frames.pixels)
      .subspan(bytes * static_cast<std::size_t>(index), bytes);
}

}  // namespace

bool PlanReferences(const std::vector<GenerationReference>& references,
                    const GenerationGeometry& geometry, bool image_size_max,
                    ReferencePlan* plan, std::string* error) {
  std::size_t images = 0;
  std::size_t videos = 0;
  std::size_t audios = 0;
  for (const GenerationReference& reference : references) {
    switch (reference.kind) {
      case ReferenceKind::kImage:
        ++images;
        break;
      case ReferenceKind::kVideo:
        ++videos;
        break;
      case ReferenceKind::kAudio:
        ++audios;
        break;
    }
  }
  if (images > kH3MaximumReferenceImages ||
      videos > kH3MaximumReferenceVideos ||
      audios > kH3MaximumReferenceAudios) {
    SetError(error,
             "MiniMax H3 Ref2VA takes at most 9 images, 3 videos and 3 audios");
    return false;
  }
  if (images + videos == 0) {
    SetError(error, "MiniMax H3 Ref2VA audio references cannot be the only "
                    "ones");
    return false;
  }
  *plan = ReferencePlan{};
  plan->vision_pixels.reserve(images + videos * 64U);
  // Vision images are built after all pixels are stored (spans into them).
  struct PendingVision {
    std::size_t first{0};
    bool pair{false};
    int width{0};
    int height{0};
  };
  std::vector<PendingVision> pending;
  int picture = 0;
  int audio_label = 0;
  int video_label = 0;
  const auto kinds = {ReferenceKind::kImage, ReferenceKind::kVideo,
                      ReferenceKind::kAudio};
  for (const ReferenceKind kind : kinds) {
    for (const GenerationReference& reference : references) {
      if (reference.kind != kind) {
        continue;
      }
      ReferencePlan::Item item;
      item.source = &reference;
      item.placement.kind = kind;
      if (kind == ReferenceKind::kImage) {
        const ControlFrames& image = reference.frames;
        if (image.frames < 1 || image.width < 1 || image.height < 1) {
          SetError(error, "MiniMax H3 reference image is empty");
          return false;
        }
        const double w = image.width;
        const double h = image.height;
        const double scale =
            image_size_max
                ? std::min(1.0, kReferenceImageShortEdge / std::min(w, h))
                : std::min(1.0, std::sqrt(static_cast<double>(geometry.width) *
                                          geometry.height / (w * h)));
        item.width = Grid(w * scale);
        item.height = Grid(h * scale);
        item.frames = 1;
        item.rgb = FitKeyframeToCanvas(Frame(image, 0), image.width,
                                       image.height, item.width, item.height);
        if (item.rgb.empty()) {
          SetError(error, "MiniMax H3 reference image is not valid RGB8");
          return false;
        }
        item.placement.latent_frames = 1;
        item.placement.latent_height = item.height / 16;
        item.placement.latent_width = item.width / 16;
        plan->presentation.push_back(
            {"<Picture " + std::to_string(++picture) + ">: ",
             static_cast<int>(pending.size())});
        pending.push_back({plan->vision_pixels.size(), false, item.width,
                           item.height});
        plan->vision_pixels.push_back(item.rgb);
      } else if (kind == ReferenceKind::kVideo) {
        const ControlFrames& video = reference.frames;
        auto [width, height] = AdaptCanvas(video.width, video.height);
        if (static_cast<double>(video.width) * video.height <
            static_cast<double>(width) * height) {
          width = Grid(video.width);
          height = Grid(video.height);
        }
        int kept = std::min(video.frames, geometry.temporal.frames);
        if (kept < 5) {
          SetError(error, "MiniMax H3 reference videos need at least 5 "
                          "frames (~0.2 s at 24 fps)");
          return false;
        }
        while (kept % 17 != 5) {
          --kept;
        }
        item.width = width;
        item.height = height;
        item.frames = kept;
        const std::size_t frame_bytes =
            static_cast<std::size_t>(width) * height * 3U;
        item.rgb.reserve(frame_bytes * static_cast<std::size_t>(kept));
        for (int frame = 0; frame < kept; ++frame) {
          const std::vector<std::uint8_t> fitted = FitKeyframeToCanvas(
              Frame(video, frame), video.width, video.height, width, height);
          if (fitted.size() != frame_bytes) {
            SetError(error, "MiniMax H3 reference video frame is invalid");
            return false;
          }
          item.rgb.insert(item.rgb.end(), fitted.begin(), fitted.end());
        }
        item.placement.latent_frames = VideoLatentFrameCount(kept);
        item.placement.latent_height = height / 16;
        item.placement.latent_width = width / 16;
        if (reference.audio.has_value()) {
          // Frames of the 40 Hz audio latent: the AudioVAE hop is 800 at
          // 32 kHz; EncodeReferences sets the exact count.
          item.placement.audio_latent_frames = -1;
          plan->presentation.push_back(
              {"<Audio " + std::to_string(++audio_label) + ">: ", -1});
        }
        plan->presentation.push_back(
            {"<Video " + std::to_string(++video_label) + ">: ", -1});
        // Qwen sees the video at 2 fps with timestamps, in two-frame blocks
        // (an odd count repeats the last sample).
        std::vector<int> samples;
        for (int frame = 0; frame < kept; frame += kQwenStride) {
          samples.push_back(frame);
        }
        std::vector<double> stamps;
        for (std::size_t index = 0; index < samples.size(); ++index) {
          stamps.push_back(static_cast<double>(index) / 2.0);
        }
        if (samples.size() % 2U == 1U) {
          samples.push_back(samples.back());
          stamps.push_back(stamps.back());
        }
        for (std::size_t index = 0; index < samples.size(); index += 2U) {
          char label[32];
          std::snprintf(label, sizeof(label), "<%.1f seconds>",
                        (stamps[index] + stamps[index + 1U]) / 2.0);
          plan->presentation.push_back(
              {label, static_cast<int>(pending.size())});
          const auto first = std::span<const std::uint8_t>(item.rgb).subspan(
              frame_bytes * static_cast<std::size_t>(samples[index]),
              frame_bytes);
          const auto second = std::span<const std::uint8_t>(item.rgb).subspan(
              frame_bytes * static_cast<std::size_t>(samples[index + 1U]),
              frame_bytes);
          pending.push_back({plan->vision_pixels.size(), true, width, height});
          plan->vision_pixels.emplace_back(first.begin(), first.end());
          plan->vision_pixels.emplace_back(second.begin(), second.end());
        }
      } else {
        if (!reference.audio.has_value()) {
          SetError(error, "MiniMax H3 reference audio is empty");
          return false;
        }
        item.placement.audio_latent_frames = -1;
        plan->presentation.push_back(
            {"<Audio " + std::to_string(++audio_label) + ">: ", -1});
      }
      plan->items.push_back(std::move(item));
    }
  }
  for (const PendingVision& vision : pending) {
    KeyframeImage image{.width = vision.width,
                        .height = vision.height,
                        .rgb = plan->vision_pixels[vision.first]};
    if (vision.pair) {
      image.second_rgb = plan->vision_pixels[vision.first + 1U];
    }
    plan->vision_images.push_back(image);
  }
  return true;
}

bool EncodeReferences(const ModelInventory& inventory,
                      const ReferencePlan& plan, std::uint64_t seed,
                      const CancellationToken* cancellation,
                      std::vector<ReferenceLatent>* latents,
                      VideoVaeEncoderTelemetry* telemetry,
                      std::string* error) {
  latents->clear();
  std::unique_ptr<AudioVaeEncoder> audio_encoder;
  AudioVaeEncoderTelemetry audio_telemetry;
  std::size_t index = 0;
  for (const ReferencePlan::Item& item : plan.items) {
    ++index;
    ReferenceLatent latent;
    latent.placement = item.placement;
    const std::uint64_t noise_seed =
        seed ^ (0xD1B54A32D192ED03ULL * static_cast<std::uint64_t>(index));
    if (item.placement.kind == ReferenceKind::kImage) {
      const KeyframeImage image{
          .width = item.width, .height = item.height, .rgb = item.rgb};
      std::vector<KeyframeMoments> moments;
      if (!EncodeKeyframeMoments(inventory, std::span(&image, 1), cancellation,
                                 &moments, telemetry, error) ||
          moments.size() != 1U ||
          !BuildKeyframeConditioning(inventory, moments.front(),
                                     kH3KeyframeEncodeSeed, noise_seed,
                                     &latent.video, error)) {
        return false;
      }
    } else if (item.placement.kind == ReferenceKind::kVideo) {
      std::vector<KeyframeMoments> moments;
      VideoVaeEncoderTelemetry clip;
      if (!EncodeGuideClipMoments(inventory, item.width, item.height, item.rgb,
                                  item.frames, cancellation, &moments, &clip,
                                  error) ||
          static_cast<int>(moments.size()) != item.placement.latent_frames ||
          !BuildGuideClipConditioning(inventory, moments,
                                      kH3KeyframeEncodeSeed, noise_seed,
                                      &latent.video, error)) {
        if (error != nullptr && error->empty()) {
          *error = "MiniMax H3 reference video latent differs";
        }
        return false;
      }
      if (telemetry != nullptr) {
        telemetry->encode_ms += clip.encode_ms;
        telemetry->load_ms += clip.load_ms;
        telemetry->tiles += clip.tiles;
      }
    }
    if (item.placement.audio_latent_frames != 0) {
      if (item.source == nullptr || !item.source->audio.has_value()) {
        SetError(error, "MiniMax H3 reference sound is missing");
        return false;
      }
      if (audio_encoder == nullptr) {
        audio_encoder = AudioVaeEncoder::Create(inventory, cancellation,
                                                &audio_telemetry, error);
        if (audio_encoder == nullptr) {
          return false;
        }
      }
      int frames = 0;
      if (!audio_encoder->Encode(*item.source->audio, cancellation,
                                 &latent.audio, &frames, &audio_telemetry,
                                 error) ||
          frames < 1) {
        if (error != nullptr && error->empty()) {
          *error = "MiniMax H3 reference audio encodes to no frames";
        }
        return false;
      }
      latent.placement.audio_latent_frames = frames;
    }
    latents->push_back(std::move(latent));
  }
  return true;
}

}  // namespace gufo::minimax_h3
