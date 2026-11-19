// MiniMax-H3-Image VAE decoder: tile plan, rotary table, patch unpacking,
// post-processing and config contract on synthetic data; with
// GUFO_H3_IMAGE_ROOT (the MiniMaxAI/MiniMax-H3-Image release) also the image
// inventory, and with GUFO_H3_IMAGE_VAE_ORACLE (a directory written by
// tools/h3/gufo-h3-image-vae-oracle.py) the F32 CPU decoder against the
// diffusers reference on the released weights.

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

void CheckAxis(const h3::VideoVaeTileAxis& axis, int extent,
               std::size_t tiles) {
  CHECK(axis.starts.size() == tiles);
  CHECK(axis.starts.front() == 0);
  CHECK(axis.starts.back() + axis.length == extent);
  for (std::size_t index = 0; index < axis.overlaps.size(); ++index) {
    CHECK(axis.overlaps[index] >= 64);
    CHECK(axis.overlaps[index] % 16 == 0);
  }
}

void TestPlan() {
  std::string error;
  auto single = h3::ResolveImageVaePlan(16, 16, &error);
  CHECK(single.has_value());
  if (single) {
    CHECK(single->width == 256 && single->height == 256);
    CheckAxis(single->x_axis, 256, 1);
    CHECK(single->patches == 256);
    CHECK(single->sequence == 261);
  }
  auto small = h3::ResolveImageVaePlan(6, 10, &error);
  CHECK(small.has_value());
  if (small) {
    // An image no larger than a tile is one tile of its own size.
    CHECK(small->y_axis.length == 96 && small->x_axis.length == 160);
    CHECK(small->patches == 60 && small->sequence == 65);
  }
  auto two = h3::ResolveImageVaePlan(20, 24, &error);
  CHECK(two.has_value());
  if (two) {
    CheckAxis(two->y_axis, 320, 2);
    CheckAxis(two->x_axis, 384, 2);
    CHECK(two->tile_latent_height == 16 && two->tile_latent_width == 16);
    // 320 = 2 * 256 - 192: the 128 px of slack widen the single overlap.
    CHECK(two->y_axis.overlaps[0] == 192);
    CHECK(two->x_axis.overlaps[0] == 128);
  }
  // 16 MP: 4096 px per axis needs 21 tiles (21 * 256 - 20 * 64 = 4096).
  auto large = h3::ResolveImageVaePlan(256, 256, &error);
  CHECK(large.has_value());
  if (large) {
    CheckAxis(large->x_axis, 4096, 21);
    CHECK(std::all_of(large->x_axis.overlaps.begin(),
                      large->x_axis.overlaps.end(),
                      [](int overlap) { return overlap == 64; }));
  }
  // 5472 x 3072 (342 x 192 latents).
  auto wide = h3::ResolveImageVaePlan(192, 342, &error);
  CHECK(wide.has_value());
  if (wide) {
    CheckAxis(wide->x_axis, 5472, 29);
    CheckAxis(wide->y_axis, 3072, 16);
  }
  CHECK(!h3::ResolveImageVaePlan(0, 4, &error).has_value());
}

void TestRope() {
  std::vector<float> cosine;
  std::vector<float> sine;
  h3::BuildImageVaeRope(2, 4, &cosine, &sine);
  CHECK(cosine.size() == (8U + 5U) * 16U);
  // Suffix rows sit at position 0.
  for (std::size_t index = 8U * 16U; index < cosine.size(); ++index) {
    CHECK(cosine[index] == 1.0F && sine[index] == 0.0F);
  }
  // Row 0 is (y, x) = (0, 0): h = 2 * 0.5 / 2 - 1 = -0.5, w = -0.75.
  // Frequency 0 of each axis has inv_freq 1.
  const float two_pi = 6.28318530717958647692F;
  CHECK(std::fabs(cosine[0] - std::cos(two_pi * -0.5F)) < 1e-6F);
  CHECK(std::fabs(sine[8] - std::sin(two_pi * -0.75F)) < 1e-6F);
  // Frequency 1 of the w axis: inv_freq = 100^-0.125.
  const float inverse = 1.0F / std::pow(100.0F, 0.125F);
  CHECK(std::fabs(sine[9] - std::sin(two_pi * -0.75F * inverse)) < 1e-6F);
  // Row 5 is (1, 1): h = 0.5, w = -0.25.
  CHECK(std::fabs(cosine[5 * 16] - std::cos(two_pi * 0.5F)) < 1e-6F);
  CHECK(std::fabs(sine[5 * 16 + 8] - std::sin(two_pi * -0.25F)) < 1e-6F);
}

void TestUnpack() {
  // Two patches side by side; component (c, py, px) holds c*1000+py*16+px
  // plus 10000 per patch.
  std::vector<float> projected(2 * 768);
  for (std::size_t patch = 0; patch < 2; ++patch) {
    for (std::size_t component = 0; component < 768; ++component) {
      const std::size_t channel = component / 256;
      const std::size_t py = (component / 16) % 16;
      const std::size_t px = component % 16;
      projected[patch * 768 + component] =
          static_cast<float>(patch * 10000 + channel * 1000 + py * 16 + px);
    }
  }
  std::vector<float> rgb(16 * 32 * 3);
  CHECK(h3::UnpackImageVaePatches(projected, 1, 2, rgb));
  const auto pixel = [&](int y, int x, int c) {
    return rgb[(static_cast<std::size_t>(y) * 32 + x) * 3 + c];
  };
  CHECK(pixel(0, 0, 0) == 0.0F);
  CHECK(pixel(3, 5, 2) == 2000.0F + 3 * 16 + 5);
  CHECK(pixel(15, 16, 1) == 10000.0F + 1000.0F + 15 * 16 + 0);
  CHECK(pixel(7, 31, 0) == 10000.0F + 7 * 16 + 15);
  std::vector<float> wrong(10);
  CHECK(!h3::UnpackImageVaePatches(projected, 1, 2, wrong));
}

void TestFinalize() {
  const std::vector<float> normalized = {0.0F, 0.0F, 0.0F, 100.0F, -100.0F,
                                         (1.0F - 0.406F) / 0.225F};
  std::vector<std::uint8_t> bytes;
  std::vector<float> unit;
  h3::FinalizeImageVaeRgb(normalized, &bytes, &unit);
  CHECK(bytes.size() == 6);
  // (0.485 * 255).to(uint8) truncates 123.675 to 123.
  CHECK(bytes[0] == 123);
  CHECK(bytes[1] == 116);  // 0.456 * 255 = 116.28
  CHECK(bytes[2] == 103);  // 0.406 * 255 = 103.53
  CHECK(bytes[3] == 255);
  CHECK(bytes[4] == 0);
  CHECK(bytes[5] == 255 || bytes[5] == 254);
  CHECK(std::fabs(unit[0] - 123.0F / 255.0F) < 1e-7F);
}

void TestConfig() {
  const auto directory =
      std::filesystem::temp_directory_path() / "gufo-h3-image-vae-test";
  std::filesystem::create_directories(directory);
  std::string mean = "[";
  std::string deviation = "[";
  for (int index = 0; index < 24; ++index) {
    mean += (index ? "," : "") + std::to_string(0.1 * index);
    deviation += (index ? "," : "") + std::to_string(1.0 + 0.1 * index);
  }
  mean += "]";
  deviation += "]";
  const auto write = [&](const std::string& name, const std::string& layers) {
    std::ofstream(directory / name)
        << R"({"latent_channels": 24, "latents_mean": )" << mean
        << R"(, "latents_std": )" << deviation
        << R"(, "vae_decoder_tiling": 1, "vae_tile_overlap_min": 64,
             "vae_tile_size": 256, "vit_decoder_2d": true,
             "vit_decoder_kwargs": {"dim_head": 64, "heads": 16,
             "num_layers": )"
        << layers << R"(, "rope_dim_ratio": 0.5, "rope_theta": 100.0}})";
  };
  write("good.json", "24");
  write("bad.json", "36");
  std::string error;
  auto good = h3::ReadImageVaeConfig(directory / "good.json", &error);
  CHECK(good.has_value());
  if (good) {
    CHECK(good->mean.size() == 24 && good->deviation.size() == 24);
    CHECK(std::fabs(good->deviation[23] - 3.3F) < 1e-6F);
  }
  CHECK(!h3::ReadImageVaeConfig(directory / "bad.json", &error).has_value());
  CHECK(error.find("geometry") != std::string::npos);
  std::filesystem::remove_all(directory);
}

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

void TestRelease(const std::filesystem::path& root) {
  std::string error;
  const char* text_encoder = std::getenv("GUFO_H3_TEXT_ENCODER");
  auto inventory = h3::ModelInventory::InspectImage(
      root, text_encoder == nullptr ? "" : text_encoder, &error);
  CHECK(inventory.has_value());
  if (!inventory) {
    std::cerr << error << '\n';
    return;
  }
  const auto* heads = inventory->FindTensor("pdd.video_out.weight");
  CHECK(heads != nullptr);
  if (heads != nullptr) {
    CHECK((heads->shape == std::vector<std::uint64_t>{6, 96, 5376}));
    CHECK(heads->phase == h3::Phase::kDitCore);
  }
  CHECK(inventory->FindTensor("blocks.49.mlp.fc2.weight") != nullptr);
  CHECK(inventory->FindTensor("audio_patch_proj.weight") == nullptr);
  CHECK(inventory->FindTensor("decoder.transformer_blocks.23.scale2") !=
        nullptr);
  if (text_encoder != nullptr) {
    CHECK(inventory->FindTensor("model.language_model.embed_tokens.weight") !=
          nullptr);
  }
  auto normalization =
      h3::ReadImageVaeConfig(root / "vae" / "config.json", &error);
  CHECK(normalization.has_value());

  const char* oracle_dir = std::getenv("GUFO_H3_IMAGE_VAE_ORACLE");
  if (oracle_dir == nullptr || !normalization) {
    std::cout << "SKIP image VAE oracle (GUFO_H3_IMAGE_VAE_ORACLE unset)\n";
    return;
  }
  const std::filesystem::path oracle(oracle_dir);
  const auto meta = h3::json::ParseFile(oracle / "meta.json", 1U << 20U);
  const auto blocks = meta.Find("blocks")->AsUint64();
  const auto& latent_shape = meta.Find("latent")->AsArray();
  const int latent_height = static_cast<int>(latent_shape[1].AsUint64());
  const int latent_width = static_cast<int>(latent_shape[2].AsUint64());
  const std::vector<float> latent = ReadFloats(oracle / "latent.f32");
  const std::vector<float> expected = ReadFloats(oracle / "rgb.f32");
  auto weights = h3::LoadImageVaeCpuWeights(*inventory, blocks, &error);
  CHECK(weights.has_value());
  if (!weights) {
    std::cerr << error << '\n';
    return;
  }
  std::vector<float> rgb;
  CHECK(h3::DecodeImageVaeCpu(*weights, *normalization, latent, latent_height,
                              latent_width, &rgb, &error));
  CHECK(rgb.size() == expected.size());
  if (rgb.size() != expected.size()) {
    std::cerr << error << '\n';
    return;
  }
  double difference = 0.0;
  double reference = 0.0;
  double maximum = 0.0;
  for (std::size_t index = 0; index < rgb.size(); ++index) {
    const double delta = static_cast<double>(rgb[index]) - expected[index];
    difference += delta * delta;
    reference += static_cast<double>(expected[index]) * expected[index];
    maximum = std::max(maximum, std::fabs(delta));
  }
  const double relative = std::sqrt(difference / reference);
  std::cout << "image VAE oracle: blocks=" << blocks << " relative_l2="
            << relative << " max_abs=" << maximum << '\n';
  CHECK(relative < 1e-4);
  CHECK(maximum < 1e-2);
  std::vector<std::uint8_t> ours;
  std::vector<std::uint8_t> theirs;
  h3::FinalizeImageVaeRgb(rgb, &ours);
  h3::FinalizeImageVaeRgb(expected, &theirs);
  std::size_t off_by_more = 0;
  for (std::size_t index = 0; index < ours.size(); ++index) {
    off_by_more += std::abs(static_cast<int>(ours[index]) - theirs[index]) > 1;
  }
  CHECK(off_by_more == 0);
}

}  // namespace

int main() {
  TestPlan();
  TestRope();
  TestUnpack();
  TestFinalize();
  TestConfig();
  if (const char* root = std::getenv("GUFO_H3_IMAGE_ROOT")) {
    TestRelease(root);
  } else {
    std::cout << "SKIP MiniMax-H3-Image release checks (GUFO_H3_IMAGE_ROOT "
                 "unset)\n";
  }
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "minimax_h3_image_vae_test passed\n";
  return 0;
}
