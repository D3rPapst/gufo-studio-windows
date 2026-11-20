// MiniMax-H3-Image contract on the host: canvas sizing
// (get_image_canvas_size), the packed [text | image] layout and its rotary
// grid (build_image_packed_sequence), the fixed PDD schedule and the AdaLN
// row map the denoiser derives from it; for edits the reference sizing
// (bound_short_edge), the [text | references | image] layout, the
// "Picture i: " presentation and the reference conditioning latent, against
// values of the diffusers reference (c215605a, run on the CPU with torch).
// With GUFO_H3_IMAGE_ROOT also the release's own config files and tokenizer
// against the compiled constants.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/minimax_h3/image_edit.hpp"
#include "src/models/minimax_h3/image_generation.hpp"
#include "src/models/minimax_h3/json.hpp"
#include "src/models/minimax_h3/prompt_encoder.hpp"
#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/sampling.hpp"
#include "src/models/minimax_h3/tokenizer.hpp"

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

// --- Edits ------------------------------------------------------------------

void TestBoundShortEdge() {
  // bound_short_edge(w, h, 1024, 2048, 32) and (w, h, 1024, 1024, 32).
  struct Case {
    int width, height, vae_width, vae_height, vision_width, vision_height;
  };
  const Case cases[] = {
      {1500, 1100, 1500, 1100, 1408, 1024}, {4000, 3000, 2720, 2048, 1376, 1024},
      {800, 600, 1376, 1024, 1376, 1024},   {640, 2000, 1024, 3200, 1024, 3200},
      {1024, 1500, 1024, 1500, 1024, 1500}, {3000, 900, 3424, 1024, 3424, 1024},
      {33, 40, 1024, 1248, 1024, 1248},     {2048, 2049, 2048, 2049, 1024, 1024},
      {5000, 5000, 2048, 2048, 1024, 1024}};
  for (const Case& c : cases) {
    const auto vae = h3::BoundShortEdge(c.width, c.height, 1024, 2048, 32);
    const auto vision = h3::BoundShortEdge(c.width, c.height, 1024, 1024, 32);
    CHECK(vae.first == c.vae_width && vae.second == c.vae_height);
    CHECK(vision.first == c.vision_width && vision.second == c.vision_height);
    if (vae.first != c.vae_width || vae.second != c.vae_height ||
        vision.first != c.vision_width || vision.second != c.vision_height) {
      std::cerr << c.width << 'x' << c.height << " -> " << vae.first << 'x'
                << vae.second << " / " << vision.first << 'x' << vision.second
                << '\n';
    }
  }
}

void TestEditLayout() {
  // build_image_packed_sequence(tags [1, 0, 0, 1], references (4, 6) and
  // (6, 2), generated 4 x 4, patch (1, 2, 2), image tag 0).
  const double expected[17][3] = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {2.0, 0.0, 0.0},
      {3.0, 0.0, 0.0},
      {4.0, 2.9360547051563817, -3.5959179422654266},
      {4.0, 2.9360547051563817, 9.468027352578194},
      {4.0, 2.9360547051563817, 22.531972647421814},
      {4.0, 16.0, -3.5959179422654266},
      {4.0, 16.0, 9.468027352578194},
      {4.0, 16.0, 22.531972647421814},
      {5.0, -11.712812921102039, 6.7623956929659865},
      {5.0, 6.762395692965988, 6.7623956929659865},
      {5.0, 25.237604307034015, 6.7623956929659865},
      {6.0, 0.0, 0.0},
      {6.0, 0.0, 16.0},
      {6.0, 16.0, 0.0},
      {6.0, 16.0, 16.0}};
  const std::vector<h3::ReferencePlacement> references = {
      {.kind = h3::ReferenceKind::kImage, .latent_frames = 1,
       .latent_height = 4, .latent_width = 6},
      {.kind = h3::ReferenceKind::kImage, .latent_frames = 1,
       .latent_height = 6, .latent_width = 2}};
  std::string error;
  auto layout = h3::BuildImageEditLayout(4, references, 4, 4, &error);
  CHECK(layout.has_value());
  if (!layout) {
    std::cerr << error << '\n';
    return;
  }
  CHECK(layout->rows() == 17);
  CHECK(layout->text_rows == 4);
  CHECK(layout->condition_rows == 9);
  CHECK(layout->video_target_rows == 4);
  CHECK(layout->segments.size() == 4);
  CHECK(layout->segments[1].kind == h3::SegmentKind::kVideoCondition);
  CHECK(layout->segments[1].begin == 4 && layout->segments[1].end == 13);
  CHECK(layout->segments[2].kind == h3::SegmentKind::kAudioTarget);
  CHECK(layout->segments[3].begin == 13 && layout->segments[3].end == 17);
  double worst = 0.0;
  for (std::size_t row = 0; row < 17 && row < layout->rows(); ++row) {
    const auto& position = layout->positions[row];
    worst = std::max({worst, std::fabs(position.temporal - expected[row][0]),
                      std::fabs(position.height - expected[row][1]),
                      std::fabs(position.width - expected[row][2])});
  }
  CHECK(worst < 1e-12);

  // AdaLN rows: the text tags as given, the reference rows on the 0.999
  // condition time row with the image tag, the generated rows on the step.
  auto schedule = h3::BuildProvidedSchedule(h3::kH3ImagePddSigmas, &error);
  CHECK(schedule.has_value());
  if (!schedule) {
    return;
  }
  auto rows = h3::BuildTimeRowSchedule(*schedule, &error);
  CHECK(rows.has_value());
  if (!rows) {
    return;
  }
  h3::AppendConditionTimeRow(&*rows);
  CHECK(rows->condition_row ==
        static_cast<std::uint32_t>(h3::kH3ImagePddSteps));
  const std::vector<std::uint8_t> tags = {1, 0, 0, 1};
  auto map = h3::BuildModulationRowMap(*rows, 2, *layout, tags, &error);
  CHECK(map.has_value());
  if (map) {
    const std::uint32_t time = rows->video_rows[2];
    CHECK((*map)[0] == time * 3U + 1U);
    CHECK((*map)[1] == time * 3U + 0U);
    CHECK((*map)[3] == time * 3U + 1U);
    for (std::size_t row = 4; row < 13; ++row) {
      CHECK((*map)[row] == rows->condition_row * 3U);
    }
    for (std::size_t row = 13; row < 17; ++row) {
      CHECK((*map)[row] == time * 3U);
    }
  }

  // Without references the edit layout is the text-to-image layout.
  auto plain = h3::BuildImageEditLayout(4, {}, 4, 4, &error);
  auto text = h3::BuildImageTextLayout(4, 4, 4, &error);
  CHECK(plain && text &&
        h3::PackedLayoutSha256(*plain) == h3::PackedLayoutSha256(*text));
  const std::vector<h3::ReferencePlacement> odd = {
      {.kind = h3::ReferenceKind::kImage, .latent_frames = 1,
       .latent_height = 3, .latent_width = 4}};
  CHECK(!h3::BuildImageEditLayout(4, odd, 4, 4, &error).has_value());
  const std::vector<h3::ReferencePlacement> video = {
      {.kind = h3::ReferenceKind::kVideo, .latent_frames = 2,
       .latent_height = 4, .latent_width = 4}};
  CHECK(!h3::BuildImageEditLayout(4, video, 4, 4, &error).has_value());
  std::vector<h3::ReferencePlacement> many(
      h3::kH3ImageMaximumReferences + 1,
      {.kind = h3::ReferenceKind::kImage, .latent_frames = 1,
       .latent_height = 2, .latent_width = 2});
  CHECK(!h3::BuildImageEditLayout(4, many, 4, 4, &error).has_value());
}

void TestEditConditioning() {
  // Moments 48 x 3 x 3 (log variance clamp hit at channel 24, 0, 0) and
  // given noise -> the 24 x 2 x 2 conditioning latent of the reference:
  // posterior sample, float16, top-left crop to whole patches, release
  // statistics, scale_noise(0.999).
  constexpr int kHeight = 3;
  constexpr int kWidth = 3;
  std::vector<float> moments(48 * kHeight * kWidth);
  for (std::size_t index = 0; index < moments.size(); ++index) {
    moments[index] = static_cast<float>(std::sin(index * 0.37) * 2.5);
  }
  for (std::size_t index = 0; index < 24 * kHeight * kWidth; ++index) {
    moments[24 * kHeight * kWidth + index] =
        static_cast<float>(std::cos(index * 0.11) * 3.0 - 1.0);
  }
  moments[24 * kHeight * kWidth] = -40.0F;
  std::vector<float> posterior(24 * kHeight * kWidth);
  for (std::size_t index = 0; index < posterior.size(); ++index) {
    posterior[index] = static_cast<float>(std::sin(index * 0.73 + 0.2) * 1.3);
  }
  std::vector<float> anchor(24 * 2 * 2);
  for (std::size_t index = 0; index < anchor.size(); ++index) {
    anchor[index] = static_cast<float>(std::cos(index * 0.53 + 0.1) * 1.1);
  }
  // vae/config.json of the release (same as the H3 VisualVAE).
  const h3::ImageVaeStatistics statistics{
      .mean = {0.858090341091156F,   -0.9606591463088989F, 1.0661640167236328F,
               -0.5090325474739075F, -0.2727581858634949F, -1.3675414323806763F,
               -0.2553254961967468F, -0.26907554268836975F,
               -0.5376840829849243F, -0.0464097298681736F, 0.6657370328903198F,
               0.19690127670764923F, -0.5460608005523682F,
               -0.4035342037677765F, -0.23683024942874908F,
               0.25928452610969543F, -0.30133944749832153F, 0.211341992020607F,
               -1.1206848621368408F, 0.3581933379173279F,
               -0.04225143790245056F, 0.2604829967021942F,
               0.22864092886447906F, 0.7056031823158264F},
      .deviation = {1.2223774194717407F, 1.2767263650894165F,
                    1.6831774711608887F, 1.7549455165863037F,
                    1.5636216402053833F, 2.194143533706665F,
                    0.9653137922286987F, 1.0569885969161987F,
                    0.841948926448822F,  0.7729952931404114F,
                    1.8955937623977661F, 0.946841835975647F,
                    0.7996809482574463F, 0.44988900423049927F,
                    0.7197399735450745F, 0.6936293244361877F,
                    2.961095094680786F,  2.7694199085235596F,
                    3.0496184825897217F, 2.1088054180145264F,
                    3.276226282119751F,  3.1627357006073F,
                    2.2816812992095947F, 2.6127843856811523F}};
  const float expected[96] = {
      -0.70018822F,   2.33240628F,    2.94809389F,    1.38802624F,
      1.04196477F,    0.856979847F,   -0.731816292F,  -1.396384F,
      0.0930607021F,  0.597314417F,   0.87721175F,    0.711777329F,
      -0.381798267F,  -0.756725788F,  -1.13633001F,   -1.05463064F,
      1.44260299F,    1.79293442F,    1.59699273F,    1.14553559F,
      0.211707771F,   -0.0578301065F, -0.892787397F,  -1.03477275F,
      5.75510359F,    4.50162268F,    -0.721870542F,  -2.02617383F,
      -0.0515247062F, -1.64441526F,   -2.93288827F,   -1.91176093F,
      4.10927629F,    3.40766573F,    1.59938157F,    0.749788404F,
      -3.04623437F,   -2.87491393F,   -1.23030615F,   0.0441808626F,
      0.910247087F,   0.597095788F,   -0.283813417F,  -0.690696716F,
      -2.67045903F,   -2.58333683F,   -0.880728066F,  0.835550845F,
      1.58412266F,    -1.05563903F,   -1.77961826F,   0.0802657679F,
      -7.83522749F,   -7.76603317F,   2.18641829F,    7.66118145F,
      0.553826571F,   -0.574207008F,  -1.5232842F,    -1.89857316F,
      -1.90554023F,   -0.504888117F,  2.32003212F,    3.19571853F,
      0.147668764F,   -0.145218611F,  -0.556577384F,  -0.648090661F,
      -0.159623668F,  0.221999601F,   0.95336026F,    1.10426593F,
      -0.279758543F,  -0.229317337F,  0.34346813F,    0.375210643F,
      -0.658278227F,  0.846066058F,   2.58479953F,    2.06524754F,
      -0.629622877F,  -0.43939805F,   -0.382157028F,  -0.471995503F,
      0.484635144F,   0.726942956F,   0.738710761F,   0.531816661F,
      -1.00587201F,   -1.11112332F,   -0.986896932F,  -0.77050668F,
      0.669094026F,   0.797316909F,   0.508643806F,   0.112337284F};
  std::vector<float> latent;
  std::string error;
  CHECK(h3::BuildImageEditConditioningFromNoise(moments, kHeight, kWidth, 2, 2,
                                                statistics, posterior, anchor,
                                                &latent, &error));
  CHECK(latent.size() == 96);
  double worst = 0.0;
  for (std::size_t index = 0; index < 96 && index < latent.size(); ++index) {
    const double reference = expected[index];
    const double delta = static_cast<double>(latent[index]) - reference;
    worst = std::max(worst,
                     std::fabs(delta) / std::max(1.0, std::fabs(reference)));
  }
  if (worst >= 1e-5) {
    std::cerr << "conditioning worst relative difference " << worst << '\n';
  }
  CHECK(worst < 1e-5);
  // Odd or oversized crops are refused.
  CHECK(!h3::BuildImageEditConditioningFromNoise(moments, kHeight, kWidth, 3, 2,
                                                 statistics, posterior, anchor,
                                                 &latent, &error));
  // The seeded form is deterministic and keeps the shape.
  std::vector<float> first;
  std::vector<float> second;
  CHECK(h3::BuildImageEditConditioning(moments, kHeight, kWidth, 2, 2,
                                       statistics, 7, 8, &first, &error));
  CHECK(h3::BuildImageEditConditioning(moments, kHeight, kWidth, 2, 2,
                                       statistics, 7, 8, &second, &error));
  CHECK(first == second && first.size() == 96);
  CHECK(h3::ImageEditPosteriorSeed(5, 0) != h3::ImageEditAnchorSeed(5, 0));
  CHECK(h3::ImageEditPosteriorSeed(5, 0) != h3::ImageEditPosteriorSeed(5, 1));
}

void TestEditPresentation() {
  // "Picture 1: " and "Picture 2: " as the release tokenizer encodes them.
  const std::vector<std::vector<std::uint32_t>> labels = {
      {24669, 220, 16, 25, 220}, {24669, 220, 17, 25, 220}};
  const std::vector<std::pair<int, int>> sizes = {{96, 64}, {32, 32}};
  const std::vector<std::uint32_t> prompt = {9, 8, 7};
  h3::ImageEditPresentation presentation;
  std::string error;
  CHECK(h3::BuildImageEditPresentation(labels, sizes, prompt, &presentation,
                                       &error));
  // label(5) + start + 3*2 pads + end, label(5) + start + 1 pad + end, prompt.
  const std::vector<std::uint32_t> tokens = {
      24669, 220, 16, 25, 220, h3::kVisionStartToken, h3::kImagePadToken,
      h3::kImagePadToken, h3::kImagePadToken, h3::kImagePadToken,
      h3::kImagePadToken, h3::kImagePadToken, h3::kVisionEndToken,
      24669, 220, 17, 25, 220, h3::kVisionStartToken, h3::kImagePadToken,
      h3::kVisionEndToken, 9, 8, 7};
  const std::vector<std::uint8_t> tags = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                                          0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1};
  CHECK(presentation.tokens == tokens);
  CHECK(presentation.tags == tags);
  CHECK(presentation.slots.size() == 2);
  if (presentation.slots.size() == 2) {
    CHECK(presentation.slots[0].begin == 6);
    CHECK(presentation.slots[0].grid_height == 2 &&
          presentation.slots[0].grid_width == 3);
    CHECK(presentation.slots[1].begin == 19);
  }
  CHECK(presentation.vision_tokens == 7);
  CHECK(!presentation.prompt_truncated);
  CHECK(h3::ImageEditLabel(0) == "Picture 1: ");
  CHECK(h3::ImageEditLabel(15) == "Picture 16: ");
  // The prompt is cut to max_sequence_length.
  const std::vector<std::uint32_t> long_prompt(5000, 42);
  CHECK(h3::BuildImageEditPresentation(labels, sizes, long_prompt,
                                       &presentation, &error));
  CHECK(presentation.prompt_truncated);
  CHECK(presentation.tokens.size() == 21 + 4096);
  const std::vector<std::pair<int, int>> uneven = {{96, 48}, {32, 32}};
  CHECK(!h3::BuildImageEditPresentation(labels, uneven, prompt, &presentation,
                                        &error));
}

std::vector<std::uint8_t> Pattern(int width, int height) {
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3U);
  for (std::size_t index = 0; index < rgb.size(); ++index) {
    rgb[index] = static_cast<std::uint8_t>((index * 2654435761U) >> 24U);
  }
  return rgb;
}

void TestEditPlan() {
  std::vector<h3::ImageEditInput> inputs;
  // Kept size (short edge 1100 within [1024, 2048]): cropped to 1488 x 1088
  // for the VAE, latent 92 x 68; vision copy 1408 x 1024.
  inputs.push_back({.width = 1500, .height = 1100, .rgb = Pattern(1500, 1100)});
  // Small: scaled up to a 1024 short edge (1376 x 1024) for both.
  inputs.push_back({.width = 80, .height = 60, .rgb = Pattern(80, 60)});
  // Short edge exactly 1024 with a long side off the 32 grid: the vision
  // copy follows Qwen's rounding (1500 -> 1504).
  inputs.push_back({.width = 1500, .height = 1024, .rgb = Pattern(1500, 1024)});
  std::vector<h3::ImageEditReference> references;
  std::string error;
  CHECK(h3::PlanImageEditReferences(inputs, &references, &error));
  CHECK(references.size() == 3);
  if (references.size() != 3) {
    std::cerr << error << '\n';
    return;
  }
  const auto& kept = references[0];
  CHECK(kept.vae_width == 1488 && kept.vae_height == 1088);
  CHECK(kept.latent_width == 92 && kept.latent_height == 68);
  CHECK(kept.vae_rgb.size() == 1488U * 1088U * 3U);
  // The crop keeps the top-left pixels unchanged.
  CHECK(std::equal(kept.vae_rgb.begin(), kept.vae_rgb.begin() + 1488 * 3,
                   inputs[0].rgb.begin()));
  CHECK(std::equal(kept.vae_rgb.end() - 1488 * 3, kept.vae_rgb.end(),
                   inputs[0].rgb.begin() + 1087 * 1500 * 3));
  CHECK(kept.vision_width == 1408 && kept.vision_height == 1024);
  CHECK(kept.vision_rgb.size() == 1408U * 1024U * 3U);
  const auto& small = references[1];
  CHECK(small.vae_width == 1376 && small.vae_height == 1024);
  CHECK(small.latent_width == 86 && small.latent_height == 64);
  CHECK(small.vision_width == 1376 && small.vision_height == 1024);
  const auto& exact = references[2];
  CHECK(exact.vae_width == 1488 && exact.vae_height == 1024);
  CHECK(exact.latent_width == 92 && exact.latent_height == 64);
  CHECK(exact.vision_width == 1504 && exact.vision_height == 1024);
  const auto placement = h3::ImageEditPlacement(kept);
  CHECK(placement.kind == h3::ReferenceKind::kImage &&
        placement.latent_height == 68 && placement.latent_width == 92);

  std::vector<h3::ImageEditInput> tiny = {
      {.width = 31, .height = 64, .rgb = Pattern(31, 64)}};
  CHECK(!h3::PlanImageEditReferences(tiny, &references, &error));
  CHECK(!h3::PlanImageEditReferences({}, &references, &error));
  std::vector<h3::ImageEditInput> many(
      h3::kH3ImageMaximumReferences + 1,
      {.width = 32, .height = 32, .rgb = Pattern(32, 32)});
  CHECK(!h3::PlanImageEditReferences(many, &references, &error));
}

void TestReleaseTokenizer(const std::filesystem::path& root) {
  h3::Tokenizer tokenizer;
  std::string error;
  CHECK(h3::Tokenizer::Load(root / "tokenizer" / "tokenizer.json", &tokenizer,
                            &error));
  // transformers Qwen2TokenizerFast(..., add_special_tokens=False).
  const std::pair<std::size_t, std::vector<std::uint32_t>> labels[] = {
      {0, {24669, 220, 16, 25, 220}},
      {1, {24669, 220, 17, 25, 220}},
      {9, {24669, 220, 16, 15, 25, 220}},
      {15, {24669, 220, 16, 21, 25, 220}}};
  for (const auto& [index, expected] : labels) {
    std::vector<std::uint32_t> ids;
    CHECK(tokenizer.Encode(h3::ImageEditLabel(index), false, &ids, &error));
    CHECK(ids == expected);
  }
  std::uint32_t id = 0;
  CHECK(tokenizer.AddedTokenId("<|vision_start|>", &id) &&
        id == h3::kVisionStartToken);
  CHECK(tokenizer.AddedTokenId("<|image_pad|>", &id) &&
        id == h3::kImagePadToken);
  CHECK(tokenizer.AddedTokenId("<|vision_end|>", &id) &&
        id == h3::kVisionEndToken);
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
  TestBoundShortEdge();
  TestEditLayout();
  TestEditConditioning();
  TestEditPresentation();
  TestEditPlan();
  if (const char* root = std::getenv("GUFO_H3_IMAGE_ROOT")) {
    TestReleaseConfig(root);
    TestReleaseTokenizer(root);
    h3::ImageVaeStatistics statistics;
    std::string error;
    CHECK(h3::LoadImageVaeStatistics(root, &statistics, &error));
    CHECK(statistics.mean[0] == 0.858090341091156F);
    CHECK(statistics.deviation[23] == 2.6127843856811523F);
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
