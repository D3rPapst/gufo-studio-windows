// MiniMax-H3-Image edits; see image_edit.hpp.
// Written with AI assistance (Claude Code).
#include "src/models/minimax_h3/image_edit.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <exception>
#include <limits>

#include "src/models/minimax_h3/image_generation.hpp"
#include "src/models/minimax_h3/json.hpp"
#include "src/models/minimax_h3/prompt_encoder.hpp"

namespace gufo::minimax_h3 {
namespace {

constexpr int kLatentChannels = 24;
constexpr int kMomentChannels = 48;
constexpr int kVaeCell = 16;     // the VAE's spatial compression
constexpr int kVisionFactor = 32;  // Qwen patch 16, merged 2x2

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

// Python's round() (half to even) onto `multiple`, at least one multiple.
int RoundToMultiple(double value, int multiple) {
  return std::max(multiple,
                  static_cast<int>(std::nearbyint(value / multiple)) *
                      multiple);
}

// torch's `.to(torch.float16).float()`: nearest half, ties to even.
float RoundTripHalf(float value) {
  if (!std::isfinite(value)) {
    return value;
  }
  const float magnitude = std::fabs(value);
  if (magnitude >= 65520.0F) {
    return std::copysign(std::numeric_limits<float>::infinity(), value);
  }
  if (magnitude < 6.103515625e-05F) {
    const float step = 5.9604644775390625e-08F;
    return std::nearbyint(value / step) * step;
  }
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t lower = bits & 0x1FFFU;
  std::uint32_t upper = bits & ~0x1FFFU;
  if (lower > 0x1000U || (lower == 0x1000U && (upper & 0x2000U) != 0U)) {
    upper += 0x2000U;
  }
  return std::bit_cast<float>(upper);
}

// Qwen2-VL smart_resize with factor 32 and the release's pixel bounds.
std::pair<int, int> SmartResize(int width, int height) {
  int w_bar = RoundToMultiple(width, kVisionFactor);
  int h_bar = RoundToMultiple(height, kVisionFactor);
  const double area = static_cast<double>(width) * height;
  if (static_cast<std::int64_t>(w_bar) * h_bar > kH3ImageVisionMaximumPixels) {
    const double beta =
        std::sqrt(area / static_cast<double>(kH3ImageVisionMaximumPixels));
    h_bar = std::max(kVisionFactor,
                     static_cast<int>(std::floor(height / beta / kVisionFactor)) *
                         kVisionFactor);
    w_bar = std::max(kVisionFactor,
                     static_cast<int>(std::floor(width / beta / kVisionFactor)) *
                         kVisionFactor);
  } else if (static_cast<std::int64_t>(w_bar) * h_bar <
             kH3ImageVisionMinimumPixels) {
    const double beta =
        std::sqrt(static_cast<double>(kH3ImageVisionMinimumPixels) / area);
    h_bar = static_cast<int>(std::ceil(height * beta / kVisionFactor)) *
            kVisionFactor;
    w_bar = static_cast<int>(std::ceil(width * beta / kVisionFactor)) *
            kVisionFactor;
  }
  return {w_bar, h_bar};
}

bool Resize(const ImageEditInput& input, int width, int height,
            std::vector<std::uint8_t>* output, std::string* error) {
  if (width == input.width && height == input.height) {
    *output = input.rgb;
    return true;
  }
  return ResizeRgb8Lanczos(input.rgb, input.width, input.height, width, height,
                           output, error);
}

}  // namespace

std::pair<int, int> BoundShortEdge(int width, int height, int minimum,
                                   int maximum, int multiple) {
  const int short_edge = std::min(width, height);
  if (minimum <= short_edge && short_edge <= maximum) {
    return {width, height};
  }
  const double scale =
      static_cast<double>(short_edge < minimum ? minimum : maximum) /
      short_edge;
  return {RoundToMultiple(width * scale, multiple),
          RoundToMultiple(height * scale, multiple)};
}

bool PlanImageEditReferences(std::span<const ImageEditInput> inputs,
                             std::vector<ImageEditReference>* references,
                             std::string* error) {
  if (references == nullptr) {
    SetError(error, "invalid MiniMax-H3-Image reference plan output");
    return false;
  }
  references->clear();
  if (inputs.empty() || inputs.size() > kH3ImageMaximumReferences) {
    SetError(error, "MiniMax-H3-Image edits take 1 to " +
                        std::to_string(kH3ImageMaximumReferences) +
                        " reference images");
    return false;
  }
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const ImageEditInput& input = inputs[index];
    const std::string name = "reference image " + std::to_string(index + 1);
    if (input.width < kVisionFactor || input.height < kVisionFactor ||
        input.rgb.size() != static_cast<std::size_t>(input.width) *
                                static_cast<std::size_t>(input.height) * 3U) {
      SetError(error, "MiniMax-H3-Image " + name +
                          " needs at least 32 pixels on each side");
      return false;
    }
    if (std::max(input.width, input.height) >
        200 * std::min(input.width, input.height)) {
      SetError(error, "MiniMax-H3-Image " + name +
                          " is more elongated than 200:1");
      return false;
    }
    ImageEditReference reference;
    // The VAE copy: bounded short edge, then the encoder's top-left crop to
    // whole cells and the latent's crop to whole 2x2 patches.
    const auto [vae_width, vae_height] = BoundShortEdge(
        input.width, input.height, kH3ImageReferenceMinimumShortEdge,
        kH3ImageReferenceMaximumShortEdge, kVisionFactor);
    std::vector<std::uint8_t> resized;
    if (!Resize(input, vae_width, vae_height, &resized, error)) {
      return false;
    }
    reference.vae_width = vae_width / kVaeCell * kVaeCell;
    reference.vae_height = vae_height / kVaeCell * kVaeCell;
    reference.latent_width = reference.vae_width / kVaeCell / 2 * 2;
    reference.latent_height = reference.vae_height / kVaeCell / 2 * 2;
    if (static_cast<std::int64_t>(reference.vae_width) * reference.vae_height >
        kH3ImageMaximumPixels) {
      SetError(error, "MiniMax-H3-Image " + name +
                          " exceeds the supported area after sizing");
      return false;
    }
    if (reference.vae_width == vae_width &&
        reference.vae_height == vae_height) {
      reference.vae_rgb = std::move(resized);
    } else {
      reference.vae_rgb.resize(static_cast<std::size_t>(reference.vae_width) *
                               reference.vae_height * 3U);
      for (int y = 0; y < reference.vae_height; ++y) {
        std::copy_n(resized.begin() + static_cast<std::ptrdiff_t>(
                                          static_cast<std::size_t>(y) *
                                          vae_width * 3U),
                    static_cast<std::size_t>(reference.vae_width) * 3U,
                    reference.vae_rgb.begin() +
                        static_cast<std::ptrdiff_t>(
                            static_cast<std::size_t>(y) * reference.vae_width *
                            3U));
      }
    }
    // The vision copy: short edge 1024, then Qwen's own grid rounding.
    auto [vision_width, vision_height] = BoundShortEdge(
        input.width, input.height, kH3ImageVisionShortEdge,
        kH3ImageVisionShortEdge, kVisionFactor);
    const auto [grid_width, grid_height] =
        SmartResize(vision_width, vision_height);
    if (grid_width != vision_width || grid_height != vision_height) {
      vision_width = grid_width;
      vision_height = grid_height;
    }
    if (!Resize(input, vision_width, vision_height, &reference.vision_rgb,
                error)) {
      return false;
    }
    reference.vision_width = vision_width;
    reference.vision_height = vision_height;
    references->push_back(std::move(reference));
  }
  return true;
}

ReferencePlacement ImageEditPlacement(
    const ImageEditReference& reference) noexcept {
  return {.kind = ReferenceKind::kImage,
          .latent_frames = 1,
          .latent_height = reference.latent_height,
          .latent_width = reference.latent_width,
          .audio_latent_frames = 0};
}

std::string ImageEditLabel(std::size_t index) {
  // No angle brackets here, unlike the "<Picture i>" of the prompt (and
  // unlike the video model's Ref2VA labels).
  return "Picture " + std::to_string(index + 1) + ": ";
}

bool BuildImageEditPresentation(
    std::span<const std::vector<std::uint32_t>> label_tokens,
    std::span<const std::pair<int, int>> vision_sizes,
    std::span<const std::uint32_t> prompt_tokens,
    ImageEditPresentation* presentation, std::string* error) {
  if (presentation == nullptr || label_tokens.size() != vision_sizes.size() ||
      label_tokens.empty() || prompt_tokens.empty()) {
    SetError(error, "invalid MiniMax-H3-Image edit presentation arguments");
    return false;
  }
  ImageEditPresentation local;
  for (std::size_t index = 0; index < label_tokens.size(); ++index) {
    const auto& label = label_tokens[index];
    const auto [width, height] = vision_sizes[index];
    if (width < kVisionFactor || height < kVisionFactor ||
        width % kVisionFactor != 0 || height % kVisionFactor != 0) {
      SetError(error, "MiniMax-H3-Image vision copies need 32-pixel sides");
      return false;
    }
    const std::size_t kept = std::min(label.size(), kH3ImageEditMaximumTokens);
    local.tokens.insert(local.tokens.end(), label.begin(),
                        label.begin() + static_cast<std::ptrdiff_t>(kept));
    local.tags.insert(local.tags.end(), kept, 1);
    ImageEditVisionSlot slot{
        .begin = 0,
        .grid_height = static_cast<std::uint32_t>(height / kVisionFactor),
        .grid_width = static_cast<std::uint32_t>(width / kVisionFactor)};
    const std::size_t rows =
        static_cast<std::size_t>(slot.grid_height) * slot.grid_width;
    local.tokens.push_back(kVisionStartToken);
    slot.begin = local.tokens.size();
    local.tokens.insert(local.tokens.end(), rows, kImagePadToken);
    local.tokens.push_back(kVisionEndToken);
    local.tags.insert(local.tags.end(), rows + 2U, 0);
    local.vision_tokens += rows;
    local.slots.push_back(slot);
  }
  const std::size_t kept =
      std::min(prompt_tokens.size(), kH3ImageEditMaximumTokens);
  local.prompt_truncated = kept != prompt_tokens.size();
  local.tokens.insert(local.tokens.end(), prompt_tokens.begin(),
                      prompt_tokens.begin() + static_cast<std::ptrdiff_t>(kept));
  local.tags.insert(local.tags.end(), kept, 1);
  *presentation = std::move(local);
  return true;
}

bool LoadImageVaeStatistics(const std::filesystem::path& model_root,
                            ImageVaeStatistics* statistics,
                            std::string* error) {
  if (statistics == nullptr) {
    SetError(error, "invalid MiniMax-H3-Image VAE statistics output");
    return false;
  }
  try {
    const json::Value config =
        json::ParseFile(model_root / "vae" / "config.json", 1U << 20U);
    const json::Value* mean = config.Find("latents_mean");
    const json::Value* deviation = config.Find("latents_std");
    if (mean == nullptr || deviation == nullptr || !mean->IsArray() ||
        !deviation->IsArray() ||
        mean->AsArray().size() != statistics->mean.size() ||
        deviation->AsArray().size() != statistics->deviation.size()) {
      throw json::Error("latent normalization shape differs");
    }
    for (std::size_t index = 0; index < statistics->mean.size(); ++index) {
      statistics->mean[index] =
          static_cast<float>(mean->AsArray()[index].AsDouble());
      statistics->deviation[index] =
          static_cast<float>(deviation->AsArray()[index].AsDouble());
      if (!std::isfinite(statistics->mean[index]) ||
          !(statistics->deviation[index] > 0.0F)) {
        throw json::Error("latent normalization contains invalid values");
      }
    }
  } catch (const std::exception& exception) {
    SetError(error, "MiniMax-H3-Image VAE config failed: " +
                        std::string(exception.what()));
    return false;
  }
  return true;
}

bool BuildImageEditConditioningFromNoise(
    std::span<const float> moments, int moment_height, int moment_width,
    int latent_height, int latent_width, const ImageVaeStatistics& statistics,
    std::span<const float> posterior_noise, std::span<const float> anchor_noise,
    std::vector<float>* latent, std::string* error) {
  const std::size_t plane = static_cast<std::size_t>(moment_height) *
                            static_cast<std::size_t>(moment_width);
  const std::size_t target = static_cast<std::size_t>(latent_height) *
                             static_cast<std::size_t>(latent_width);
  if (latent == nullptr || moment_height < 1 || moment_width < 1 ||
      latent_height < 2 || latent_width < 2 || latent_height % 2 != 0 ||
      latent_width % 2 != 0 || latent_height > moment_height ||
      latent_width > moment_width ||
      moments.size() != plane * kMomentChannels ||
      posterior_noise.size() != plane * kLatentChannels ||
      anchor_noise.size() != target * kLatentChannels) {
    SetError(error, "invalid MiniMax-H3-Image reference conditioning inputs");
    return false;
  }
  latent->assign(target * kLatentChannels, 0.0F);
  const float t = kH3ImageReferenceNoiseAugmentation;
  for (int channel = 0; channel < kLatentChannels; ++channel) {
    const std::size_t mean_base = static_cast<std::size_t>(channel) * plane;
    const std::size_t variance_base =
        static_cast<std::size_t>(channel + kLatentChannels) * plane;
    const float mean = statistics.mean[static_cast<std::size_t>(channel)];
    const float deviation =
        statistics.deviation[static_cast<std::size_t>(channel)];
    for (int y = 0; y < latent_height; ++y) {
      for (int x = 0; x < latent_width; ++x) {
        const std::size_t source =
            static_cast<std::size_t>(y) * moment_width + x;
        const float mu = moments[mean_base + source];
        const float log_variance =
            std::clamp(moments[variance_base + source], -30.0F, 20.0F);
        const float sample = mu + std::exp(0.5F * log_variance) *
                                      posterior_noise[mean_base + source];
        const float normalized = (RoundTripHalf(sample) - mean) / deviation;
        const std::size_t output =
            static_cast<std::size_t>(channel) * target +
            static_cast<std::size_t>(y) * latent_width + x;
        (*latent)[output] =
            t * normalized + (1.0F - t) * anchor_noise[output];
      }
    }
  }
  return true;
}

bool BuildImageEditConditioning(std::span<const float> moments,
                                int moment_height, int moment_width,
                                int latent_height, int latent_width,
                                const ImageVaeStatistics& statistics,
                                std::uint64_t posterior_seed,
                                std::uint64_t anchor_seed,
                                std::vector<float>* latent,
                                std::string* error) {
  if (moment_height < 1 || moment_width < 1 || latent_height < 1 ||
      latent_width < 1) {
    SetError(error, "invalid MiniMax-H3-Image reference conditioning inputs");
    return false;
  }
  std::vector<float> posterior(static_cast<std::size_t>(kLatentChannels) *
                               moment_height * moment_width);
  std::vector<float> anchor(static_cast<std::size_t>(kLatentChannels) *
                            latent_height * latent_width);
  NormalRng posterior_rng(posterior_seed);
  posterior_rng.Fill(posterior);
  NormalRng anchor_rng(anchor_seed);
  anchor_rng.Fill(anchor);
  return BuildImageEditConditioningFromNoise(
      moments, moment_height, moment_width, latent_height, latent_width,
      statistics, posterior, anchor, latent, error);
}

std::uint64_t ImageEditPosteriorSeed(std::uint64_t seed,
                                     std::size_t index) noexcept {
  return seed ^ (0x9E3779B97F4A7C15ULL * (static_cast<std::uint64_t>(index) +
                                          1ULL));
}

std::uint64_t ImageEditAnchorSeed(std::uint64_t seed,
                                  std::size_t index) noexcept {
  return seed ^ (0xD1B54A32D192ED03ULL * (static_cast<std::uint64_t>(index) +
                                          1ULL));
}

}  // namespace gufo::minimax_h3
