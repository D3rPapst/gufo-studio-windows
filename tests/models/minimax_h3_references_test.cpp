// MiniMax H3 Ref2VA reference planning (ComfyUI's MiniMaxH3ReferenceToVideo)
// with reference videos, the 24 fps resampling of reference videos and the
// multithreaded control frame fit. CPU only.
// Written with AI assistance (Claude Code).
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "src/models/minimax_h3/control.hpp"
#include "src/models/minimax_h3/references.hpp"
#include "src/models/minimax_h3/sampling.hpp"

namespace {

using gufo::minimax_h3::ControlFrames;
using gufo::minimax_h3::GenerationGeometry;
using gufo::minimax_h3::GenerationReference;
using gufo::minimax_h3::ReferenceKind;
using gufo::minimax_h3::ReferencePlan;

int failures = 0;

#define CHECK(condition)                                          \
  do {                                                            \
    if (!(condition)) {                                           \
      std::cerr << "FAIL " << __FILE__ << ':' << __LINE__ << ": " \
                << #condition << '\n';                            \
      ++failures;                                                 \
    }                                                             \
  } while (false)

GenerationGeometry Geometry(int width, int height, int frames) {
  std::string error;
  auto geometry = gufo::minimax_h3::ResolveGenerationGeometry(width, height,
                                                              frames, &error);
  if (!geometry.has_value()) {
    std::cerr << "unexpected geometry failure: " << error << '\n';
    ++failures;
    return {};
  }
  return *geometry;
}

// Frame f is filled with the byte f (so a frame's origin is visible).
ControlFrames Frames(int width, int height, int count) {
  ControlFrames frames;
  frames.width = width;
  frames.height = height;
  frames.frames = count;
  const std::size_t bytes = static_cast<std::size_t>(width) * height * 3U;
  frames.pixels.resize(bytes * static_cast<std::size_t>(count));
  for (int frame = 0; frame < count; ++frame) {
    std::fill_n(frames.pixels.begin() +
                    static_cast<std::ptrdiff_t>(bytes * frame),
                bytes, static_cast<std::uint8_t>(frame));
  }
  return frames;
}

void TestSourceFrames() {
  using gufo::minimax_h3::ReferenceSourceFrame;
  // 24 fps and an unknown rate keep the frames.
  for (int output = 0; output < 50; ++output) {
    CHECK(ReferenceSourceFrame(output, 24, 1) == output);
    CHECK(ReferenceSourceFrame(output, 0, 1) == output);
  }
  // 30 fps: FFmpeg's fps filter drops every fifth frame (source i lands on
  // round(0.8 i)).
  const std::vector<int> thirty = {0, 1, 3, 4, 5, 6, 8, 9};
  for (std::size_t output = 0; output < thirty.size(); ++output) {
    CHECK(ReferenceSourceFrame(static_cast<int>(output), 30, 1) ==
          thirty[output]);
  }
  // 12 fps repeats each frame.
  const std::vector<int> twelve = {0, 0, 1, 1, 2, 2};
  for (std::size_t output = 0; output < twelve.size(); ++output) {
    CHECK(ReferenceSourceFrame(static_cast<int>(output), 12, 1) ==
          twelve[output]);
  }
  // 16 fps: source 1 lands on round(1.5) = 2 (half away from zero).
  CHECK(ReferenceSourceFrame(1, 16, 1) == 0);
  CHECK(ReferenceSourceFrame(2, 16, 1) == 1);
  // NTSC 30000/1001.
  CHECK(ReferenceSourceFrame(2, 30000, 1001) == 3);
  CHECK(ReferenceSourceFrame(100, 30000, 1001) == 125);
}

void TestSecondsLabels() {
  using gufo::minimax_h3::ReferenceSecondsLabel;
  // Python's "%.1f": exact ties round half to even.
  CHECK(ReferenceSecondsLabel(0.25) == "<0.2 seconds>");
  CHECK(ReferenceSecondsLabel(0.75) == "<0.8 seconds>");
  CHECK(ReferenceSecondsLabel(1.25) == "<1.2 seconds>");
  CHECK(ReferenceSecondsLabel(2.0) == "<2.0 seconds>");
  CHECK(ReferenceSecondsLabel(13.75) == "<13.8 seconds>");
}

void TestVideoPlan() {
  const GenerationGeometry geometry = Geometry(256, 256, 56);
  std::vector<GenerationReference> references(3);
  // Request order audio, video, image: the plan sorts images, videos,
  // audios like the ComfyUI node.
  references[0].kind = ReferenceKind::kAudio;
  references[0].audio.emplace();
  references[1].kind = ReferenceKind::kVideo;
  references[1].frames = Frames(64, 64, 70);
  references[1].audio.emplace();
  references[2].kind = ReferenceKind::kImage;
  references[2].frames = Frames(64, 64, 1);
  ReferencePlan plan;
  std::string error;
  CHECK(gufo::minimax_h3::PlanReferences(references, geometry, false, &plan,
                                         &error));
  if (!error.empty()) {
    std::cerr << error << '\n';
  }
  CHECK(plan.items.size() == 3U);
  if (plan.items.size() != 3U) {
    return;
  }
  CHECK(plan.items[0].placement.kind == ReferenceKind::kImage);
  CHECK(plan.items[1].placement.kind == ReferenceKind::kVideo);
  CHECK(plan.items[2].placement.kind == ReferenceKind::kAudio);
  // The video: cut to the target's 56 frames (already 17k + 5), on its own
  // 64 x 64 canvas, 2 + 5 * 3 = 17 latent frames, its soundtrack pending.
  const ReferencePlan::Item& video = plan.items[1];
  CHECK(video.frames == 56);
  CHECK(video.width == 64 && video.height == 64);
  CHECK(video.placement.latent_frames == 17);
  CHECK(video.placement.latent_height == 4 && video.placement.latent_width == 4);
  CHECK(video.placement.audio_latent_frames == -1);
  CHECK(video.rgb.size() == 56U * 64U * 64U * 3U);
  // Qwen: frames 0, 12, 24, 36, 48 at 0, 0.5, 1, 1.5, 2 s, the odd last one
  // repeated: blocks at 0.25, 1.25 and 2.0 s.
  const std::vector<std::string> labels = {
      "<Picture 1>: ",   "<Audio 1>: ",     "<Video 1>: ",
      "<0.2 seconds>",   "<1.2 seconds>",   "<2.0 seconds>",
      "<Audio 2>: "};
  const std::vector<int> visions = {0, -1, -1, 1, 2, 3, -1};
  CHECK(plan.presentation.size() == labels.size());
  for (std::size_t index = 0;
       index < std::min(labels.size(), plan.presentation.size()); ++index) {
    CHECK(plan.presentation[index].label == labels[index]);
    CHECK(plan.presentation[index].vision == visions[index]);
  }
  CHECK(plan.vision_images.size() == 4U);
  if (plan.vision_images.size() == 4U) {
    CHECK(plan.vision_images[0].second_rgb.empty());
    const std::vector<std::pair<int, int>> pairs = {{0, 12}, {24, 36},
                                                    {48, 48}};
    for (std::size_t block = 0; block < pairs.size(); ++block) {
      const auto& image = plan.vision_images[block + 1U];
      CHECK(image.width == 64 && image.height == 64);
      CHECK(!image.rgb.empty() && !image.second_rgb.empty());
      if (!image.rgb.empty() && !image.second_rgb.empty()) {
        CHECK(image.rgb[0] == static_cast<std::uint8_t>(pairs[block].first));
        CHECK(image.second_rgb[0] ==
              static_cast<std::uint8_t>(pairs[block].second));
      }
    }
  }
  // A longer video is cut to the target; a 17k + 5 remainder is dropped.
  references.resize(2);
  references[0] = GenerationReference{};
  references[0].kind = ReferenceKind::kVideo;
  references[0].frames = Frames(64, 64, 30);  // 30 -> 22
  references[1].kind = ReferenceKind::kImage;
  references[1].frames = Frames(64, 64, 1);
  CHECK(gufo::minimax_h3::PlanReferences(references, geometry, false, &plan,
                                         &error));
  CHECK(plan.items.size() == 2U && plan.items[1].frames == 22 &&
        plan.items[1].placement.latent_frames == 7 &&
        plan.items[1].placement.audio_latent_frames == 0);
  // Fewer than five frames cannot form a clip.
  references[0].frames = Frames(64, 64, 4);
  CHECK(!gufo::minimax_h3::PlanReferences(references, geometry, false, &plan,
                                          &error));
  // Audio alone is refused.
  std::vector<GenerationReference> audio_only(1);
  audio_only[0].kind = ReferenceKind::kAudio;
  audio_only[0].audio.emplace();
  CHECK(!gufo::minimax_h3::PlanReferences(audio_only, geometry, false, &plan,
                                          &error));
}

void TestFitFramesThreads() {
  // The threaded fit gives every frame the same values as a single-frame
  // fit of that frame.
  ControlFrames input = Frames(48, 40, 9);
  for (std::size_t index = 0; index < input.pixels.size(); ++index) {
    input.pixels[index] =
        static_cast<std::uint8_t>((index * 2654435761U) >> 24U);
  }
  const std::vector<float> all =
      gufo::minimax_h3::FitFramesBilinearCenter(input, 3, 11, 64, 32);
  CHECK(all.size() == 11U * 64U * 32U * 3U);
  const std::size_t frame_values = 64U * 32U * 3U;
  const std::size_t source_bytes = 48U * 40U * 3U;
  for (int frame = 0; frame < 11; ++frame) {
    ControlFrames single;
    single.width = 48;
    single.height = 40;
    single.frames = 1;
    const int source = std::min(frame, 8);
    single.pixels.assign(
        input.pixels.begin() + static_cast<std::ptrdiff_t>(source_bytes * source),
        input.pixels.begin() +
            static_cast<std::ptrdiff_t>(source_bytes * (source + 1)));
    const std::vector<float> one =
        gufo::minimax_h3::FitFramesBilinearCenter(single, 3, 1, 64, 32);
    CHECK(std::equal(one.begin(), one.end(),
                     all.begin() + static_cast<std::ptrdiff_t>(
                                       frame_values * frame)));
  }
}

}  // namespace

int main() {
  TestSourceFrames();
  TestSecondsLabels();
  TestVideoPlan();
  TestFitFramesThreads();
  if (failures != 0) {
    std::cerr << failures << " MiniMax H3 reference checks failed\n";
    return 1;
  }
  std::cout << "MiniMax H3 reference checks passed\n";
  return 0;
}
