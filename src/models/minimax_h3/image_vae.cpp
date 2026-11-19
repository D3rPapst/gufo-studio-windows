#include "src/models/minimax_h3/image_vae.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "src/models/minimax_h3/json.hpp"

namespace gufo::minimax_h3 {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

// AutoencoderKLMiniMaxH3Image._split_tiles over `extent` pixels.
bool BuildImageTileAxis(int extent, VideoVaeTileAxis* axis,
                        std::string* error) {
  *axis = {};
  if (extent < 1 || extent % kH3ImageVaeSpatialRatio != 0) {
    SetError(error, "MiniMax-H3-Image VAE extent must be a multiple of 16");
    return false;
  }
  if (extent <= kH3ImageVaeTilePixels) {
    axis->length = extent;
    axis->starts = {0};
    return true;
  }
  int count = (extent + kH3ImageVaeTilePixels - 1) / kH3ImageVaeTilePixels;
  while (kH3ImageVaeTilePixels * count -
             kH3ImageVaeTileOverlapMinimum * (count - 1) <
         extent) {
    ++count;
  }
  axis->length = kH3ImageVaeTilePixels;
  axis->starts.assign(static_cast<std::size_t>(count), 0);
  axis->overlaps.assign(static_cast<std::size_t>(count - 1),
                        kH3ImageVaeTileOverlapMinimum);
  const int remaining = kH3ImageVaeTilePixels * count -
                        kH3ImageVaeTileOverlapMinimum * (count - 1) - extent;
  for (int unit = 0; unit < remaining / kH3ImageVaeSpatialRatio; ++unit) {
    axis->overlaps[static_cast<std::size_t>(unit % (count - 1))] +=
        kH3ImageVaeSpatialRatio;
  }
  for (int index = 1; index < count; ++index) {
    axis->starts[static_cast<std::size_t>(index)] =
        axis->starts[static_cast<std::size_t>(index - 1)] +
        kH3ImageVaeTilePixels -
        axis->overlaps[static_cast<std::size_t>(index - 1)];
  }
  return true;
}

bool SameSize(const std::vector<float>& values, std::size_t expected) {
  return values.size() == expected;
}

// y[rows][out] = x[rows][in] * W[out][in]^T (+ bias).
void LinearRows(const float* input, std::size_t rows, std::size_t in,
                const float* weight, const float* bias, std::size_t out,
                float* output) {
  for (std::size_t row = 0; row < rows; ++row) {
    const float* x = input + row * in;
    float* y = output + row * out;
    for (std::size_t column = 0; column < out; ++column) {
      const float* w = weight + column * in;
      double sum = 0.0;
      for (std::size_t k = 0; k < in; ++k) {
        sum += static_cast<double>(x[k]) * w[k];
      }
      y[column] = static_cast<float>(sum) + (bias != nullptr ? bias[column] : 0.0F);
    }
  }
}

void RmsNormRows(const float* input, std::size_t rows, std::size_t width,
                 const float* weight, float epsilon, float* output) {
  for (std::size_t row = 0; row < rows; ++row) {
    const float* x = input + row * width;
    double square = 0.0;
    for (std::size_t column = 0; column < width; ++column) {
      square += static_cast<double>(x[column]) * x[column];
    }
    const float inverse = 1.0F / std::sqrt(static_cast<float>(
                                     square / static_cast<double>(width)) +
                                 epsilon);
    for (std::size_t column = 0; column < width; ++column) {
      output[row * width + column] =
          x[column] * inverse * (weight != nullptr ? weight[column] : 1.0F);
    }
  }
}

void LayerNormRows(const float* input, std::size_t rows, std::size_t width,
                   const float* weight, const float* bias, float epsilon,
                   float* output) {
  for (std::size_t row = 0; row < rows; ++row) {
    const float* x = input + row * width;
    double sum = 0.0;
    for (std::size_t column = 0; column < width; ++column) {
      sum += x[column];
    }
    const double mean = sum / static_cast<double>(width);
    double square = 0.0;
    for (std::size_t column = 0; column < width; ++column) {
      const double centered = x[column] - mean;
      square += centered * centered;
    }
    const float inverse = 1.0F / std::sqrt(static_cast<float>(
                                     square / static_cast<double>(width)) +
                                 epsilon);
    for (std::size_t column = 0; column < width; ++column) {
      output[row * width + column] =
          static_cast<float>(x[column] - mean) * inverse * weight[column] +
          bias[column];
    }
  }
}

}  // namespace

std::optional<ImageVaePlan> ResolveImageVaePlan(int latent_height,
                                                int latent_width,
                                                std::string* error) {
  if (latent_height < 1 || latent_width < 1 ||
      latent_height > std::numeric_limits<int>::max() / kH3ImageVaeSpatialRatio ||
      latent_width > std::numeric_limits<int>::max() / kH3ImageVaeSpatialRatio) {
    SetError(error, "invalid MiniMax-H3-Image VAE latent size");
    return std::nullopt;
  }
  ImageVaePlan plan;
  plan.latent_height = latent_height;
  plan.latent_width = latent_width;
  plan.height = latent_height * kH3ImageVaeSpatialRatio;
  plan.width = latent_width * kH3ImageVaeSpatialRatio;
  if (!BuildImageTileAxis(plan.height, &plan.y_axis, error) ||
      !BuildImageTileAxis(plan.width, &plan.x_axis, error)) {
    return std::nullopt;
  }
  plan.tile_latent_height =
      static_cast<std::uint32_t>(plan.y_axis.length / kH3ImageVaeSpatialRatio);
  plan.tile_latent_width =
      static_cast<std::uint32_t>(plan.x_axis.length / kH3ImageVaeSpatialRatio);
  plan.patches = plan.tile_latent_height * plan.tile_latent_width;
  plan.sequence = plan.patches + static_cast<std::uint32_t>(kH3ImageVaeSuffixRows);
  return plan;
}

void BuildImageVaeRope(std::uint32_t tile_latent_height,
                       std::uint32_t tile_latent_width,
                       std::vector<float>* cosine, std::vector<float>* sine) {
  const std::size_t patches =
      static_cast<std::size_t>(tile_latent_height) * tile_latent_width;
  const std::size_t sequence = patches + kH3ImageVaeSuffixRows;
  cosine->assign(sequence * kH3ImageVaeRopeHalf, 1.0F);
  sine->assign(sequence * kH3ImageVaeRopeHalf, 0.0F);
  constexpr float kTwoPi = 6.28318530717958647692F;
  for (std::size_t row = 0; row < patches; ++row) {
    const std::uint32_t y = static_cast<std::uint32_t>(row / tile_latent_width);
    const std::uint32_t x = static_cast<std::uint32_t>(row % tile_latent_width);
    const float coordinates[2] = {
        2.0F * ((static_cast<float>(y) + 0.5F) /
                static_cast<float>(tile_latent_height)) -
            1.0F,
        2.0F * ((static_cast<float>(x) + 0.5F) /
                static_cast<float>(tile_latent_width)) -
            1.0F};
    for (std::size_t frequency = 0; frequency < kH3ImageVaeRopeHalf;
         ++frequency) {
      const std::size_t axis = frequency / kH3ImageVaeRopeFrequencies;
      const std::size_t index = frequency % kH3ImageVaeRopeFrequencies;
      const float inverse =
          1.0F / std::pow(kH3ImageVaeRopeTheta,
                          static_cast<float>(index) * 0.125F);
      const float angle = kTwoPi * coordinates[axis] * inverse;
      (*cosine)[row * kH3ImageVaeRopeHalf + frequency] = std::cos(angle);
      (*sine)[row * kH3ImageVaeRopeHalf + frequency] = std::sin(angle);
    }
  }
}

bool UnpackImageVaePatches(std::span<const float> projected,
                           std::uint32_t tile_latent_height,
                           std::uint32_t tile_latent_width,
                           std::span<float> rgb, std::string* error) {
  const std::size_t patches =
      static_cast<std::size_t>(tile_latent_height) * tile_latent_width;
  constexpr std::size_t kPatch = kH3ImageVaeSpatialRatio;
  const std::size_t width = tile_latent_width * kPatch;
  if (projected.size() < patches * kH3ImageVaeOutputPatch ||
      rgb.size() != patches * kPatch * kPatch * 3U) {
    SetError(error, "invalid MiniMax-H3-Image VAE unpack sizes");
    return false;
  }
  for (std::size_t patch = 0; patch < patches; ++patch) {
    const std::size_t patch_y = patch / tile_latent_width;
    const std::size_t patch_x = patch % tile_latent_width;
    const float* source = projected.data() + patch * kH3ImageVaeOutputPatch;
    for (std::size_t channel = 0; channel < 3; ++channel) {
      for (std::size_t py = 0; py < kPatch; ++py) {
        for (std::size_t px = 0; px < kPatch; ++px) {
          const std::size_t y = patch_y * kPatch + py;
          const std::size_t x = patch_x * kPatch + px;
          rgb[(y * width + x) * 3U + channel] =
              source[(channel * kPatch + py) * kPatch + px];
        }
      }
    }
  }
  return true;
}

void FinalizeImageVaeRgb(std::span<const float> normalized_rgb,
                         std::vector<std::uint8_t>* rgb8,
                         std::vector<float>* rgb01) {
  constexpr float kMean[3] = {0.485F, 0.456F, 0.406F};
  constexpr float kDeviation[3] = {0.229F, 0.224F, 0.225F};
  if (rgb8 != nullptr) {
    rgb8->resize(normalized_rgb.size());
  }
  if (rgb01 != nullptr) {
    rgb01->resize(normalized_rgb.size());
  }
  for (std::size_t index = 0; index < normalized_rgb.size(); ++index) {
    const std::size_t channel = index % 3U;
    float value = normalized_rgb[index] * kDeviation[channel] + kMean[channel];
    // NaN maps to 0, like the clamp of a torch tensor would not; the HIP path
    // rejects non-finite output before this step.
    value = std::isnan(value) ? 0.0F : std::clamp(value, 0.0F, 1.0F);
    const auto byte =
        static_cast<std::uint8_t>(std::clamp(value * 255.0F, 0.0F, 255.0F));
    if (rgb8 != nullptr) {
      (*rgb8)[index] = byte;
    }
    if (rgb01 != nullptr) {
      (*rgb01)[index] = static_cast<float>(byte) / 255.0F;
    }
  }
}

std::optional<ImageVaeLatentNormalization> ReadImageVaeConfig(
    const std::filesystem::path& vae_config, std::string* error) {
  try {
    const json::Value config = json::ParseFile(vae_config, 1U << 20U);
    const auto number = [&](const json::Value& object, std::string_view key) {
      const json::Value* value = object.Find(key);
      if (value == nullptr || !value->IsNumber()) {
        throw json::Error("missing number " + std::string(key));
      }
      return value->AsDouble();
    };
    const json::Value* is_2d = config.Find("vit_decoder_2d");
    const json::Value* kwargs = config.Find("vit_decoder_kwargs");
    if (is_2d == nullptr || !is_2d->IsBool() || !is_2d->AsBool() ||
        kwargs == nullptr || !kwargs->IsObject()) {
      throw json::Error("not a 2D ViT decoder (vit_decoder_2d)");
    }
    if (number(config, "latent_channels") != kH3ImageVaeLatentChannels ||
        number(*kwargs, "num_layers") != kH3ImageVaeBlocks ||
        number(*kwargs, "heads") != kH3ImageVaeHeads ||
        number(*kwargs, "dim_head") != kH3ImageVaeHeadDimension ||
        number(*kwargs, "rope_dim_ratio") != 0.5 ||
        number(*kwargs, "rope_theta") != kH3ImageVaeRopeTheta ||
        number(config, "vae_tile_size") != kH3ImageVaeTilePixels ||
        number(config, "vae_tile_overlap_min") !=
            kH3ImageVaeTileOverlapMinimum ||
        number(config, "vae_decoder_tiling") != 1) {
      throw json::Error("decoder geometry differs from the H3-Image release");
    }
    ImageVaeLatentNormalization normalization;
    for (const auto& [key, target] :
         {std::pair{"latents_mean", &normalization.mean},
          std::pair{"latents_std", &normalization.deviation}}) {
      const json::Value* values = config.Find(key);
      if (values == nullptr || !values->IsArray() ||
          values->AsArray().size() != kH3ImageVaeLatentChannels) {
        throw json::Error(std::string(key) + " must have 24 values");
      }
      for (const json::Value& value : values->AsArray()) {
        const float parsed = static_cast<float>(value.AsDouble());
        if (!std::isfinite(parsed)) {
          throw json::Error(std::string(key) + " is not finite");
        }
        target->push_back(parsed);
      }
    }
    for (const float deviation : normalization.deviation) {
      if (deviation <= 0.0F) {
        throw json::Error("latents_std must be positive");
      }
    }
    return normalization;
  } catch (const std::exception& exception) {
    SetError(error, "MiniMax-H3-Image VAE config " + vae_config.string() +
                        ": " + exception.what());
    return std::nullopt;
  }
}

std::optional<ImageVaeCpuWeights> LoadImageVaeCpuWeights(
    const ModelInventory& inventory, std::size_t block_count,
    std::string* error) {
  if (block_count > kH3ImageVaeBlocks) {
    SetError(error, "MiniMax-H3-Image VAE has only 24 blocks");
    return std::nullopt;
  }
  ImageVaeCpuWeights weights;
  const auto load = [&](const std::string& name,
                        std::initializer_list<std::uint64_t> shape,
                        std::vector<float>* output) {
    const TensorDescriptor* descriptor = inventory.FindTensor(name);
    if (descriptor == nullptr || descriptor->dtype != DType::kFloat32 ||
        descriptor->shape !=
            std::vector<std::uint64_t>(shape.begin(), shape.end())) {
      SetError(error, "MiniMax-H3-Image VAE tensor missing or different: " +
                          name);
      return false;
    }
    return ReadTensorF32(inventory, *descriptor, output, error);
  };
  constexpr std::uint64_t C = kH3ImageVaeLatentChannels;
  constexpr std::uint64_t H = kH3ImageVaeHidden;
  if (!load("post_quant_conv.weight", {C, C, 1, 1}, &weights.post_weight) ||
      !load("post_quant_conv.bias", {C}, &weights.post_bias) ||
      !load("decoder.x_embedder.weight", {H, C}, &weights.embed_weight) ||
      !load("decoder.x_embedder.bias", {H}, &weights.embed_bias) ||
      !load("decoder.register_tokens", {1, kH3ImageVaeRegisters, H},
            &weights.registers) ||
      !load("decoder.norm_out.weight", {H}, &weights.final_norm_weight) ||
      !load("decoder.norm_out.bias", {H}, &weights.final_norm_bias) ||
      !load("decoder.proj_out.weight", {kH3ImageVaeOutputPatch, H},
            &weights.output_weight) ||
      !load("decoder.proj_out.bias", {kH3ImageVaeOutputPatch},
            &weights.output_bias)) {
    return std::nullopt;
  }
  weights.blocks.resize(block_count);
  for (std::size_t block = 0; block < block_count; ++block) {
    const std::string prefix =
        "decoder.transformer_blocks." + std::to_string(block) + ".";
    ImageVaeCpuBlock& weight = weights.blocks[block];
    if (!load(prefix + "norm1.weight", {H}, &weight.norm1) ||
        !load(prefix + "attn.to_qkv.weight", {3U * H, H}, &weight.qkv_weight) ||
        !load(prefix + "attn.to_qkv.bias", {3U * H}, &weight.qkv_bias) ||
        !load(prefix + "attn.to_out.weight", {H, H}, &weight.output_weight) ||
        !load(prefix + "attn.to_out.bias", {H}, &weight.output_bias) ||
        !load(prefix + "scale1", {H}, &weight.scale1) ||
        !load(prefix + "norm2.weight", {H}, &weight.norm2) ||
        !load(prefix + "ff.w1.weight", {2U * kH3ImageVaeFeedForward, H},
              &weight.fc1_weight) ||
        !load(prefix + "ff.w1.bias", {2U * kH3ImageVaeFeedForward},
              &weight.fc1_bias) ||
        !load(prefix + "ff.w2.weight", {H, kH3ImageVaeFeedForward},
              &weight.fc2_weight) ||
        !load(prefix + "ff.w2.bias", {H}, &weight.fc2_bias) ||
        !load(prefix + "scale2", {H}, &weight.scale2)) {
      return std::nullopt;
    }
  }
  return weights;
}

bool DecodeImageVaeTileCpu(const ImageVaeCpuWeights& weights,
                           std::span<const float> latent,
                           std::uint32_t latent_height,
                           std::uint32_t latent_width, std::vector<float>* rgb,
                           std::string* error) {
  constexpr std::size_t C = kH3ImageVaeLatentChannels;
  constexpr std::size_t H = kH3ImageVaeHidden;
  constexpr std::size_t D = kH3ImageVaeHeadDimension;
  constexpr std::size_t kHeads = kH3ImageVaeHeads;
  constexpr std::size_t F = kH3ImageVaeFeedForward;
  constexpr std::size_t R = kH3ImageVaeRopeHalf;
  const std::size_t patches =
      static_cast<std::size_t>(latent_height) * latent_width;
  const std::size_t rows = patches + kH3ImageVaeSuffixRows;
  if (rgb == nullptr || patches == 0 || latent.size() != C * patches ||
      !SameSize(weights.post_weight, C * C) ||
      !SameSize(weights.embed_weight, H * C) ||
      !SameSize(weights.registers, kH3ImageVaeRegisters * H) ||
      !SameSize(weights.output_weight, kH3ImageVaeOutputPatch * H)) {
    SetError(error, "invalid MiniMax-H3-Image CPU VAE tile arguments");
    return false;
  }
  try {
    // [C][h][w] -> rows [h*w][C], then post_quant_conv (1x1) and x_embedder.
    std::vector<float> latent_rows(patches * C);
    for (std::size_t channel = 0; channel < C; ++channel) {
      for (std::size_t patch = 0; patch < patches; ++patch) {
        latent_rows[patch * C + channel] = latent[channel * patches + patch];
      }
    }
    std::vector<float> post(patches * C);
    LinearRows(latent_rows.data(), patches, C, weights.post_weight.data(),
               weights.post_bias.data(), C, post.data());
    std::vector<float> hidden(rows * H, 0.0F);
    LinearRows(post.data(), patches, C, weights.embed_weight.data(),
               weights.embed_bias.data(), H, hidden.data());
    std::copy(weights.registers.begin(), weights.registers.end(),
              hidden.begin() + static_cast<std::ptrdiff_t>(patches * H));

    std::vector<float> cosine;
    std::vector<float> sine;
    BuildImageVaeRope(latent_height, latent_width, &cosine, &sine);

    std::vector<float> norm(rows * H);
    std::vector<float> qkv(rows * 3U * H);
    std::vector<float> query(rows * H);
    std::vector<float> key(rows * H);
    std::vector<float> value(rows * H);
    std::vector<float> heads(rows * H);
    std::vector<float> branch(rows * H);
    std::vector<float> fc1(rows * 2U * F);
    std::vector<float> activated(rows * F);
    std::vector<double> scores(rows);
    const float scale = 1.0F / std::sqrt(static_cast<float>(D));
    for (const ImageVaeCpuBlock& block : weights.blocks) {
      RmsNormRows(hidden.data(), rows, H, block.norm1.data(),
                  kH3ImageVaeNormEpsilon, norm.data());
      LinearRows(norm.data(), rows, H, block.qkv_weight.data(),
                 block.qkv_bias.data(), 3U * H, qkv.data());
      // Per-head interleaved [q k v]; RMS q/k norm (no affine), then RoPE on
      // the first 2 * 16 channels (rotate_half convention).
      for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t head = 0; head < kHeads; ++head) {
          const float* base = qkv.data() + row * 3U * H + head * 3U * D;
          for (std::size_t part = 0; part < 2; ++part) {
            const float* source = base + part * D;
            float normalized[D];
            RmsNormRows(source, 1, D, nullptr, kH3ImageVaeNormEpsilon,
                        normalized);
            float* target =
                (part == 0 ? query.data() : key.data()) + row * H + head * D;
            for (std::size_t lane = 0; lane < D; ++lane) {
              float result = normalized[lane];
              if (lane < R) {
                const float c = cosine[row * R + lane];
                const float s = sine[row * R + lane];
                result = normalized[lane] * c - normalized[lane + R] * s;
              } else if (lane < 2U * R) {
                const float c = cosine[row * R + lane - R];
                const float s = sine[row * R + lane - R];
                result = normalized[lane] * c + normalized[lane - R] * s;
              }
              target[lane] = result;
            }
          }
          std::copy_n(base + 2U * D, D, value.data() + row * H + head * D);
        }
      }
      for (std::size_t head = 0; head < kHeads; ++head) {
        for (std::size_t row = 0; row < rows; ++row) {
          const float* q = query.data() + row * H + head * D;
          double maximum = -std::numeric_limits<double>::infinity();
          for (std::size_t other = 0; other < rows; ++other) {
            const float* k = key.data() + other * H + head * D;
            double dot = 0.0;
            for (std::size_t lane = 0; lane < D; ++lane) {
              dot += static_cast<double>(q[lane]) * k[lane];
            }
            scores[other] = dot * scale;
            maximum = std::max(maximum, scores[other]);
          }
          double sum = 0.0;
          for (std::size_t other = 0; other < rows; ++other) {
            scores[other] = std::exp(scores[other] - maximum);
            sum += scores[other];
          }
          float* out = heads.data() + row * H + head * D;
          for (std::size_t lane = 0; lane < D; ++lane) {
            double accumulated = 0.0;
            for (std::size_t other = 0; other < rows; ++other) {
              accumulated +=
                  scores[other] * value[other * H + head * D + lane];
            }
            out[lane] = static_cast<float>(accumulated / sum);
          }
        }
      }
      LinearRows(heads.data(), rows, H, block.output_weight.data(),
                 block.output_bias.data(), H, branch.data());
      for (std::size_t index = 0; index < rows * H; ++index) {
        hidden[index] += branch[index] * block.scale1[index % H];
      }
      RmsNormRows(hidden.data(), rows, H, block.norm2.data(),
                  kH3ImageVaeNormEpsilon, norm.data());
      LinearRows(norm.data(), rows, H, block.fc1_weight.data(),
                 block.fc1_bias.data(), 2U * F, fc1.data());
      // Native fused w1 is [gate; up]: silu(gate) * up.
      for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < F; ++column) {
          const float gate = fc1[row * 2U * F + column];
          const float up = fc1[row * 2U * F + F + column];
          activated[row * F + column] = gate / (1.0F + std::exp(-gate)) * up;
        }
      }
      LinearRows(activated.data(), rows, F, block.fc2_weight.data(),
                 block.fc2_bias.data(), H, branch.data());
      for (std::size_t index = 0; index < rows * H; ++index) {
        hidden[index] += branch[index] * block.scale2[index % H];
      }
    }
    // Only the patch rows reach the image.
    LayerNormRows(hidden.data(), patches, H, weights.final_norm_weight.data(),
                  weights.final_norm_bias.data(), kH3ImageVaeNormEpsilon,
                  norm.data());
    std::vector<float> projected(patches * kH3ImageVaeOutputPatch);
    LinearRows(norm.data(), patches, H, weights.output_weight.data(),
               weights.output_bias.data(), kH3ImageVaeOutputPatch,
               projected.data());
    rgb->assign(patches * kH3ImageVaeSpatialRatio * kH3ImageVaeSpatialRatio * 3U,
                0.0F);
    return UnpackImageVaePatches(projected, latent_height, latent_width, *rgb,
                                 error);
  } catch (const std::bad_alloc&) {
    SetError(error, "out of memory in the MiniMax-H3-Image CPU VAE");
    return false;
  }
}

bool DecodeImageVaeCpu(const ImageVaeCpuWeights& weights,
                       const ImageVaeLatentNormalization& normalization,
                       std::span<const float> normalized_latent,
                       int latent_height, int latent_width,
                       std::vector<float>* rgb, std::string* error) {
  auto plan = ResolveImageVaePlan(latent_height, latent_width, error);
  if (!plan) {
    return false;
  }
  constexpr std::size_t C = kH3ImageVaeLatentChannels;
  const std::size_t plane = static_cast<std::size_t>(latent_height) *
                            static_cast<std::size_t>(latent_width);
  if (rgb == nullptr || normalized_latent.size() != C * plane ||
      normalization.mean.size() != C || normalization.deviation.size() != C) {
    SetError(error, "invalid MiniMax-H3-Image CPU VAE arguments");
    return false;
  }
  const std::size_t tile_plane =
      static_cast<std::size_t>(plan->tile_latent_height) *
      plan->tile_latent_width;
  std::vector<std::vector<float>> tiles;
  std::vector<float> tile_latent(C * tile_plane);
  for (const int start_y : plan->y_axis.starts) {
    for (const int start_x : plan->x_axis.starts) {
      const int latent_y = start_y / kH3ImageVaeSpatialRatio;
      const int latent_x = start_x / kH3ImageVaeSpatialRatio;
      for (std::size_t channel = 0; channel < C; ++channel) {
        for (std::uint32_t y = 0; y < plan->tile_latent_height; ++y) {
          for (std::uint32_t x = 0; x < plan->tile_latent_width; ++x) {
            const std::size_t source =
                channel * plane +
                static_cast<std::size_t>(latent_y + static_cast<int>(y)) *
                    latent_width +
                static_cast<std::size_t>(latent_x + static_cast<int>(x));
            tile_latent[(channel * plan->tile_latent_height + y) *
                            plan->tile_latent_width +
                        x] = std::fma(normalized_latent[source],
                                      normalization.deviation[channel],
                                      normalization.mean[channel]);
          }
        }
      }
      tiles.emplace_back();
      if (!DecodeImageVaeTileCpu(weights, tile_latent, plan->tile_latent_height,
                                 plan->tile_latent_width, &tiles.back(),
                                 error)) {
        return false;
      }
    }
  }
  return StitchVideoVaeTiles(plan->y_axis, plan->x_axis, 1, tiles, rgb, error);
}

#if !defined(ENGINE_ENABLE_HIP)

struct ImageVaeDecoder::Impl {
  ImageVaePlan plan;
};

ImageVaeDecoder::ImageVaeDecoder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ImageVaeDecoder::~ImageVaeDecoder() = default;
ImageVaeDecoder::ImageVaeDecoder(ImageVaeDecoder&&) noexcept = default;
ImageVaeDecoder& ImageVaeDecoder::operator=(ImageVaeDecoder&&) noexcept =
    default;

std::unique_ptr<ImageVaeDecoder> ImageVaeDecoder::Create(
    const ModelInventory&, const ImageVaeOptions&, const CancellationToken*,
    ImageVaeTelemetry*, std::string* error) {
  SetError(error, "MiniMax-H3-Image VAE requires a HIP-enabled build");
  return nullptr;
}

bool ImageVaeDecoder::Decode(std::span<const float>, const CancellationToken*,
                             ImageVaeProgress, void*, DecodedImage*,
                             ImageVaeTelemetry*, std::string* error) {
  SetError(error, "MiniMax-H3-Image VAE requires a HIP-enabled build");
  return false;
}

const ImageVaePlan& ImageVaeDecoder::plan() const noexcept {
  static const ImageVaePlan empty;
  return empty;
}

#endif

}  // namespace gufo::minimax_h3
