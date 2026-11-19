// MiniMax-H3-Image text-to-image contract on the host: canvas sizing
// (get_image_canvas_size), the packed [text | image] layout and its rotary
// grid (build_image_packed_sequence), the fixed PDD schedule and the AdaLN
// row map the denoiser derives from it. With GUFO_H3_IMAGE_ROOT also the
// release's own config files against the compiled constants.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/minimax_h3/image_generation.hpp"
#include "src/models/minimax_h3/json.hpp"
#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/sampling.hpp"

namespace {

namespace h3 = gufo::minimax_h3;

int failures = 0;

#define CHECK(condition)                                          \
  do {                                                            \
    if (!(condition)) {                                           \
      std::cerr << "FAIL " << __FILE__ << ':' << __LINE__ << ": " \
                << #condition << '\n';                            \
      ++failures;                                                 \
    }                                                             \
  } while (false)

void CheckCanvas(int width, int height, std::int64_t area, int canvas_width,
                 int canvas_height) {
  std::string error;
  auto canvas = h3::ResolveImageCanvas(width, height, area, &error);
  CHECK(canvas.has_value());
  if (!canvas) {
    std::cerr << error << '\n';
    return;
  }
  if (canvas->width != canvas_width || canvas->height != canvas_height) {
    std::cerr << width << 'x' << height << " -> " << canvas->width << 'x'
              << canvas->height << " expected " << canvas_width << 'x'
              << canvas_height << '\n';
  }
  CHECK(canvas->width == canvas_width);
  CHECK(canvas->height == canvas_height);
  CHECK(canvas->output_width == width && canvas->output_height == height);
}

void TestCanvas() {
  // Values from the reference get_image_canvas_size (python, round half to
  // even), default min_generation_area 2048 * 2048.
  CheckCanvas(4096, 4096, h3::kH3ImageDefaultMinimumArea, 4096, 4096);
  CheckCanvas(5472, 3072, h3::kH3ImageDefaultMinimumArea, 5472, 3072);
  CheckCanvas(4736, 3552, h3::kH3ImageDefaultMinimumArea, 4736, 3552);
  // 1024x1024 below 4 MP: scaled by 2.
  CheckCanvas(1024, 1024, h3::kH3ImageDefaultMinimumArea, 2048, 2048);
  // 1000x750: scale = sqrt(4194304 / 750000) = 2.36483...;
  // 2364.8 / 32 = 73.9 -> 74 * 32 = 2368; 1773.6 / 32 = 55.4 -> 1760.
  CheckCanvas(1000, 750, h3::kH3ImageDefaultMinimumArea, 2368, 1760);
  // min_generation_area 1024 turns the scaling off; sides still round.
  CheckCanvas(1000, 750, 1024, 992, 736);
  // Half to even: 48 / 32 = 1.5 -> 2 (64), 80 / 32 = 2.5 -> 2 (64).
  CheckCanvas(48, 80, 1024, 64, 64);
  // At least one multiple.
  CheckCanvas(8, 8, 1024, 32, 32);
  std::string error;
  CHECK(!h3::ResolveImageCanvas(1024, 1024, 512, &error).has_value());
  CHECK(!h3::ResolveImageCanvas(9000, 1024, 1024, &error).has_value());
  // 6272 x 3584 = 22.5 MP is above the supported area.
  CHECK(!h3::ResolveImageCanvas(6272, 3584, 1024, &error).has_value());
  CHECK(h3::ResolveImageCanvas(6272, 2688, 1024, &error).has_value());

  auto geometry = h3::ImageGenerationGeometry({.width = 4096, .height = 2048});
  CHECK(geometry.has_value());
  if (geometry) {
    CHECK(geometry->latent_width == 256 && geometry->latent_height == 128);
    CHECK(geometry->temporal.video_latent_frames == 1);
    CHECK(geometry->temporal.audio_latent_frames == 0);
  }
  CHECK(!h3::ImageGenerationGeometry({.width = 100, .height = 64})
             .has_value());
}

void TestLayout() {
  std::string error;
  // 128x64 px canvas: latent 8x4 -> 4x2 image rows after the 2x2 patch.
  auto layout = h3::BuildImageTextLayout(3, 4, 8, &error);
  CHECK(layout.has_value());
  if (!layout) {
    std::cerr << error << '\n';
    return;
  }
  CHECK(layout->rows() == 3 + 2 * 4);
  CHECK(layout->text_rows == 3);
  CHECK(layout->video_target_rows == 8);
  CHECK(layout->audio_target_rows == 0);
  CHECK(layout->segments.size() == 3);
  CHECK(layout->segments[1].kind == h3::SegmentKind::kAudioTarget);
  CHECK(layout->segments[1].begin == layout->segments[1].end);
  for (std::size_t row = 0; row < 3; ++row) {
    CHECK(layout->positions[row].temporal == static_cast<double>(row));
    CHECK(layout->positions[row].height == 0.0);
  }
  // _spatial_position_grid: sqrt_area = sqrt(4 * 8); h ratio = 4 / sqrt(32),
  // left = (1 - ratio) / 2, 2 rows, scaled by 32; the same for w with 4
  // columns. Image rows sit on rotary time = number of text rows.
  const double area = std::sqrt(32.0);
  const double h_ratio = 4.0 / area;
  const double w_ratio = 8.0 / area;
  const double h_left = (1.0 - h_ratio) / 2.0;
  const double w_left = (1.0 - w_ratio) / 2.0;
  for (std::size_t row = 0; row < 2; ++row) {
    for (std::size_t column = 0; column < 4; ++column) {
      const auto& position = layout->positions[3 + row * 4 + column];
      CHECK(position.temporal == 3.0);
      CHECK(std::fabs(position.height -
                      (h_left + h_ratio * static_cast<double>(row) / 2.0) *
                          32.0) < 1e-12);
      CHECK(std::fabs(position.width -
                      (w_left + w_ratio * static_cast<double>(column) / 4.0) *
                          32.0) < 1e-12);
    }
  }
  CHECK(!h3::BuildImageTextLayout(0, 4, 8, &error).has_value());
  CHECK(!h3::BuildImageTextLayout(3, 5, 8, &error).has_value());
}

void TestScheduleAndRowMap() {
  std::string error;
  auto schedule = h3::BuildProvidedSchedule(h3::kH3ImagePddSigmas, &error);
  CHECK(schedule.has_value());
  if (!schedule) {
    std::cerr << error << '\n';
    return;
  }
  CHECK(schedule->evaluations() == h3::kH3ImagePddSteps);
  CHECK(schedule->video == schedule->audio);
  const std::vector<float> rising = {1.0F, 0.5F, 0.6F, 0.0F};
  CHECK(!h3::BuildProvidedSchedule(rising, &error).has_value());
  const std::vector<float> open = {0.9F, 0.5F, 0.0F};
  CHECK(!h3::BuildProvidedSchedule(open, &error).has_value());

  auto rows = h3::BuildTimeRowSchedule(*schedule, &error);
  CHECK(rows.has_value());
  if (!rows) {
    return;
  }
  // One time row per step: text and image share the step's timestep.
  CHECK(rows->time_rows == static_cast<std::uint32_t>(h3::kH3ImagePddSteps));
  auto layout = h3::BuildImageTextLayout(2, 2, 2, &error);
  CHECK(layout.has_value());
  if (!layout) {
    return;
  }
  for (int step = 0; step < h3::kH3ImagePddSteps; ++step) {
    auto map = h3::BuildModulationRowMap(*rows, step, *layout, {}, &error);
    CHECK(map.has_value());
    if (!map) {
      continue;
    }
    // adaln_indices = timestep_index * 3 + tag: text tag 1, image tag 0.
    const std::uint32_t time = rows->video_rows[static_cast<std::size_t>(step)];
    CHECK(map->size() == 3);
    CHECK((*map)[0] == time * 3U + 1U);
    CHECK((*map)[1] == time * 3U + 1U);
    CHECK((*map)[2] == time * 3U + 0U);
  }
  auto plan = h3::BuildEulerPlan(*schedule, 1, &error);
  CHECK(plan.has_value());
  if (plan) {
    for (int step = 0; step < h3::kH3ImagePddSteps; ++step) {
      const auto index = static_cast<std::size_t>(step);
      CHECK((*plan)[index].evaluate);
      // 1 - (1 - sigma) in F32, as the scheduler round-trips its grid.
      CHECK(std::fabs((*plan)[index].video_sigma_from_timestep -
                      h3::kH3ImagePddSigmas[index]) < 1e-6F);
    }
  }
}

void TestLanczos() {
  // 8x6 pattern, resized with PIL Image.resize(..., Image.LANCZOS) (11.x).
  constexpr int kWidth = 8;
  constexpr int kHeight = 6;
  std::vector<std::uint8_t> source(kWidth * kHeight * 3);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      const std::size_t base = (static_cast<std::size_t>(y) * kWidth + x) * 3;
      source[base + 0] = static_cast<std::uint8_t>((x * 37 + y * 11) % 256);
      source[base + 1] = static_cast<std::uint8_t>((x * x * 7 + y * 29) % 256);
      source[base + 2] =
          static_cast<std::uint8_t>(((255 - x * 13 - y * 17) % 256 + 256) % 256);
    }
  }
  const std::vector<int> expected_5x4 = {
      14,  8,   247, 73,  31,  227, 128, 95,  206, 208, 183, 184, 91,  125, 164,
      30,  51,  222, 89,  72,  202, 142, 149, 181, 232, 156, 159, 117, 83,  139,
      47,  97,  195, 105, 118, 175, 168, 196, 154, 216, 68,  132, 88,  145, 112,
      63,  135, 170, 118, 177, 150, 199, 140, 129, 170, 60,  107, 17,  197, 87};
  std::vector<std::uint8_t> output;
  std::string error;
  CHECK(gufo::minimax_h3::ResizeRgb8Lanczos(source, kWidth, kHeight, 5, 4,
                                            &output, &error));
  CHECK(output.size() == expected_5x4.size());
  int worst = 0;
  for (std::size_t index = 0;
       index < std::min(output.size(), expected_5x4.size()); ++index) {
    worst = std::max(worst, std::abs(static_cast<int>(output[index]) -
                                     expected_5x4[index]));
  }
  // PIL accumulates in 22-bit fixed point; one step of 8-bit rounding.
  CHECK(worst <= 1);
  // Upscale 13x9: first row of the PIL result.
  const std::vector<int> expected_row = {0,   0,   255, 10,  0,   252, 38,  5,
                                         242, 62,  18,  234, 82,  33,  227, 106,
                                         57,  218, 129, 85,  211, 150, 115, 203,
                                         167, 145, 194, 220, 229, 187, 227, 255,
                                         179, 84,  162, 169, 0,   62,  164};
  CHECK(gufo::minimax_h3::ResizeRgb8Lanczos(source, kWidth, kHeight, 13, 9,
                                            &output, &error));
  CHECK(output.size() == 13U * 9U * 3U);
  worst = 0;
  for (std::size_t index = 0; index < expected_row.size(); ++index) {
    worst = std::max(worst, std::abs(static_cast<int>(output[index]) -
                                     expected_row[index]));
  }
  CHECK(worst <= 1);
  CHECK(gufo::minimax_h3::ResizeRgb8Lanczos(source, kWidth, kHeight, kWidth,
                                            kHeight, &output, &error));
  CHECK(output == source);
  CHECK(!gufo::minimax_h3::ResizeRgb8Lanczos(source, 7, kHeight, 5, 4, &output,
                                             &error));
}

void TestReleaseConfig(const std::filesystem::path& root) {
  const auto pdd = h3::json::ParseFile(root / "pdd_config.json", 1U << 20U);
  CHECK(pdd.Find("num_inference_steps")->AsUint64() ==
        static_cast<std::uint64_t>(h3::kH3ImagePddSteps));
  const auto& sigmas = pdd.Find("sigmas")->AsArray();
  CHECK(sigmas.size() == h3::kH3ImagePddSigmas.size());
  for (std::size_t index = 0; index < sigmas.size(); ++index) {
    CHECK(static_cast<float>(sigmas[index].AsDouble()) ==
          h3::kH3ImagePddSigmas[index]);
  }
  const auto scheduler = h3::json::ParseFile(
      root / "scheduler" / "scheduler_config.json", 1U << 20U);
  CHECK(!scheduler.Find("shift_provided_sigmas")->AsBool());
  CHECK(scheduler.Find("euler_update_form")->AsString() == "velocity");
  const auto transformer = h3::json::ParseFile(
      root / "transformer" / "config.json", 1U << 20U);
  CHECK(!transformer.Find("audio_enabled")->AsBool());
  CHECK(transformer.Find("hidden_size")->AsUint64() == 5376);
  CHECK(transformer.Find("num_layers")->AsUint64() == 50);
  CHECK(transformer.Find("latents_dim")->AsUint64() == 24);
  std::string error;
  auto inventory = h3::ModelInventory::InspectImage(root, "", &error);
  CHECK(inventory.has_value());
  if (!inventory) {
    std::cerr << error << '\n';
    return;
  }
  const auto* bias = inventory->FindTensor("pdd.video_out.bias");
  CHECK(bias != nullptr &&
        (bias->shape == std::vector<std::uint64_t>{6, 96}));
  CHECK(inventory->FindTensor("final_layer.audio_out.weight") == nullptr);
  CHECK(inventory->TensorsForPhase(h3::Phase::kPromptEncoder).empty());
  CHECK(!inventory->TensorsForPhase(h3::Phase::kAdaLnPrecompute).empty());
}

}  // namespace

int main() {
  TestCanvas();
  TestLayout();
  TestScheduleAndRowMap();
  TestLanczos();
  if (const char* root = std::getenv("GUFO_H3_IMAGE_ROOT")) {
    TestReleaseConfig(root);
  } else {
    std::cout << "SKIP MiniMax-H3-Image release config (GUFO_H3_IMAGE_ROOT "
                 "unset)\n";
  }
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "minimax_h3_image_test passed\n";
  return 0;
}
