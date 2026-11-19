// MiniMax-H3-Image VAE decoder on gfx1151 against the diffusers oracle
// (tools/h3/gufo-h3-image-vae-oracle.py --blocks 24): GUFO_H3_IMAGE_ROOT is
// the release, GUFO_H3_IMAGE_VAE_ORACLE the oracle directory. Skips (77)
// without both. GPU only in a production pause.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/minimax_h3/image_vae.hpp"
#include "src/models/minimax_h3/json.hpp"
#include "src/models/minimax_h3/runtime.hpp"

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
  const char* oracle_dir = std::getenv("GUFO_H3_IMAGE_VAE_ORACLE");
  if (root == nullptr || oracle_dir == nullptr) {
    std::cout << "SKIP: set GUFO_H3_IMAGE_ROOT and GUFO_H3_IMAGE_VAE_ORACLE\n";
    return 77;
  }
  const std::filesystem::path oracle(oracle_dir);
  const auto meta = h3::json::ParseFile(oracle / "meta.json", 1U << 20U);
  if (meta.Find("blocks")->AsUint64() != h3::kH3ImageVaeBlocks) {
    std::cerr << "the HIP check needs a full 24-block oracle\n";
    return 1;
  }
  const auto& shape = meta.Find("latent")->AsArray();
  const int latent_height = static_cast<int>(shape[1].AsUint64());
  const int latent_width = static_cast<int>(shape[2].AsUint64());
  const std::vector<float> latent = ReadFloats(oracle / "latent.f32");
  const std::vector<float> expected = ReadFloats(oracle / "rgb.f32");

  std::string error;
  auto inventory = h3::ModelInventory::InspectImage(root, {}, &error);
  if (!inventory) {
    std::cerr << error << '\n';
    return 1;
  }
  h3::ImageVaeTelemetry telemetry;
  auto decoder = h3::ImageVaeDecoder::Create(
      *inventory,
      {.latent_height = latent_height, .latent_width = latent_width}, nullptr,
      &telemetry, &error);
  if (decoder == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  h3::DecodedImage image;
  if (!decoder->Decode(latent, nullptr, nullptr, nullptr, &image, &telemetry,
                       &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  if (image.normalized_rgb.size() != expected.size()) {
    std::cerr << "shape differs\n";
    return 1;
  }
  double difference = 0.0;
  double reference = 0.0;
  double maximum = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const double delta =
        static_cast<double>(image.normalized_rgb[index]) - expected[index];
    difference += delta * delta;
    reference += static_cast<double>(expected[index]) * expected[index];
    maximum = std::max(maximum, std::fabs(delta));
  }
  const double relative = std::sqrt(difference / reference);
  std::vector<std::uint8_t> ours;
  std::vector<std::uint8_t> theirs;
  h3::FinalizeImageVaeRgb(image.normalized_rgb, &ours);
  h3::FinalizeImageVaeRgb(expected, &theirs);
  std::size_t off = 0;
  for (std::size_t index = 0; index < ours.size(); ++index) {
    off += std::abs(static_cast<int>(ours[index]) - theirs[index]) > 2;
  }
  std::cout << "MiniMax-H3-Image VAE HIP: " << image.width << 'x'
            << image.height << " tiles=" << telemetry.tile_ms.size()
            << " load_ms=" << telemetry.load_ms
            << " decode_ms=" << telemetry.decode_ms
            << " relative_l2=" << relative << " max_abs=" << maximum
            << " bytes_off_by_more_than_2=" << off << '\n';
  return relative < 1e-3 && off == 0 ? 0 : 1;
}
