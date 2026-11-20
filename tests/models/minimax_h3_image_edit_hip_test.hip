// MiniMax-H3-Image edit references on gfx1151: the image VAE encoder of the
// release (bit-identical to the H3 VisualVAE encoder) through
// EncodeKeyframeMoments on an InspectImage inventory, on a canvas whose sides
// are multiples of 16 but not of 32 (the edit's top-left crop), against the
// checkpoint-bundle oracle tools/h3/gufo-h3-vae-encoder-oracle.py --canvas N.
// Then the conditioning latent of that reference (BuildImageEditConditioning)
// is checked for shape and finiteness.
//
// GUFO_H3_IMAGE_ROOT: the MiniMax-H3-Image release; GUFO_H3_IMAGE_EDIT_ORACLE:
// the oracle directory; GUFO_H3_IMAGE_EDIT_CANVAS: its --canvas (default
// 336). Skips (77) without them. GPU only in a production pause.
// Written with AI assistance (Claude Code).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/minimax_h3/image_edit.hpp"
#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/video_vae_encoder.hpp"

namespace h3 = gufo::minimax_h3;

namespace {

std::vector<float> ReadFloats(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  input.seekg(0, std::ios::end);
  const auto bytes = static_cast<std::size_t>(input.tellg());
  input.seekg(0);
  std::vector<float> values(bytes / sizeof(float));
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
  return values;
}

}  // namespace

int main() {
  const char* root = std::getenv("GUFO_H3_IMAGE_ROOT");
  const char* oracle_dir = std::getenv("GUFO_H3_IMAGE_EDIT_ORACLE");
  if (root == nullptr || oracle_dir == nullptr) {
    std::cout << "SKIP: set GUFO_H3_IMAGE_ROOT and GUFO_H3_IMAGE_EDIT_ORACLE\n";
    return 77;
  }
  int canvas = 336;
  if (const char* value = std::getenv("GUFO_H3_IMAGE_EDIT_CANVAS")) {
    canvas = std::atoi(value);
  }
  if (canvas < 32 || canvas % 16 != 0) {
    std::cerr << "the oracle canvas must be a multiple of 16\n";
    return 1;
  }
  const std::filesystem::path oracle(oracle_dir);
  const std::string prefix =
      "video-vae-encoder-keyframe-" + std::to_string(canvas);
  gufo::core::Image png;
  try {
    png = gufo::core::DecodeImage(
        gufo::core::ReadImageFile(oracle / (prefix + ".png")));
  } catch (const std::exception& exception) {
    std::cerr << "oracle PNG: " << exception.what() << '\n';
    return 1;
  }
  const std::vector<float> expected =
      ReadFloats(oracle / (prefix + "-moments.f32"));
  const int latent = canvas / 16;
  if (static_cast<int>(png.width) != canvas ||
      static_cast<int>(png.height) != canvas ||
      expected.size() != static_cast<std::size_t>(48 * latent * latent)) {
    std::cerr << "oracle shapes differ from --canvas " << canvas << '\n';
    return 1;
  }

  std::string error;
  auto inventory = h3::ModelInventory::InspectImage(root, {}, &error);
  if (!inventory) {
    std::cerr << error << '\n';
    return 1;
  }
  const h3::KeyframeImage image{
      .width = canvas, .height = canvas, .rgb = png.pixels};
  std::vector<h3::KeyframeMoments> moments;
  h3::VideoVaeEncoderTelemetry telemetry;
  if (!h3::EncodeKeyframeMoments(*inventory, std::span(&image, 1), nullptr,
                                 &moments, &telemetry, &error) ||
      moments.size() != 1U) {
    std::cerr << error << '\n';
    return 1;
  }
  const auto& ours = moments.front().moments;
  if (ours.size() != expected.size()) {
    std::cerr << "moments shape differs\n";
    return 1;
  }
  double difference = 0.0;
  double reference = 0.0;
  double maximum = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const double delta = static_cast<double>(ours[index]) - expected[index];
    difference += delta * delta;
    reference += static_cast<double>(expected[index]) * expected[index];
    maximum = std::max(maximum, std::fabs(delta));
  }
  const double relative = std::sqrt(difference / reference);

  // The edit's conditioning: crop to whole 2x2 patches (21 -> 20).
  h3::ImageVaeStatistics statistics;
  std::vector<float> conditioning;
  const int even = latent / 2 * 2;
  const bool built =
      h3::LoadImageVaeStatistics(root, &statistics, &error) &&
      h3::BuildImageEditConditioning(ours, latent, latent, even, even,
                                     statistics, 1, 2, &conditioning, &error);
  const bool finite =
      built && conditioning.size() ==
                   static_cast<std::size_t>(24 * even * even) &&
      std::all_of(conditioning.begin(), conditioning.end(),
                  [](float value) { return std::isfinite(value); });
  std::cout << "MiniMax-H3-Image edit encoder HIP: canvas=" << canvas
            << " tiles=" << telemetry.tiles << " load_ms=" << telemetry.load_ms
            << " encode_ms=" << telemetry.encode_ms
            << " relative_l2=" << relative << " max_abs=" << maximum
            << " conditioning=" << (finite ? "ok" : error) << '\n';
  return relative < 1e-3 && finite ? 0 : 1;
}
