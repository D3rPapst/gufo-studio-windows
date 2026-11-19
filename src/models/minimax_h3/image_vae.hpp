#ifndef GUFO_MODELS_MINIMAX_H3_IMAGE_VAE_HPP_
#define GUFO_MODELS_MINIMAX_H3_IMAGE_VAE_HPP_

// MiniMax-H3-Image VAE decoder: a non-causal 2D ViT (diffusers
// MiniMaxH3ImageViTDecoder2d). Every latent pixel is one token; four learned
// register tokens and one all-zero token follow the patch tokens (all at
// rotary position 0). 24 pre-norm blocks of width 1024 (16 heads x 64, RMS
// q/k norm, 2-axis RoPE over the first half of every head, layer-scaled
// residuals, SwiGLU 4096), a LayerNorm and a linear head that expands every
// token into a 16 x 16 RGB block in ImageNet-normalized space.
//
// The released VAE decodes in 256-pixel tiles with at least 64 pixels of
// overlap that are linearly blended (AutoencoderKLMiniMaxH3Image._decode); the
// tile geometry is part of the output contract because it changes the
// attention context.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/video_vae.hpp"

namespace gufo::minimax_h3 {

inline constexpr int kH3ImageVaeLatentChannels = 24;
inline constexpr int kH3ImageVaeSpatialRatio = 16;
inline constexpr int kH3ImageVaeTilePixels = 256;
inline constexpr int kH3ImageVaeTileOverlapMinimum = 64;
inline constexpr std::size_t kH3ImageVaeHidden = 1024;
inline constexpr std::size_t kH3ImageVaeBlocks = 24;
inline constexpr std::size_t kH3ImageVaeHeads = 16;
inline constexpr std::size_t kH3ImageVaeHeadDimension = 64;
inline constexpr std::size_t kH3ImageVaeFeedForward = 4096;
inline constexpr std::size_t kH3ImageVaeRegisters = 4;
// Registers plus the zero ("cls") token.
inline constexpr std::size_t kH3ImageVaeSuffixRows = kH3ImageVaeRegisters + 1;
// rope_dim_ratio 0.5 of 64 = 32 rotated channels per head, i.e. 16 angles:
// 8 frequencies for the h axis, then 8 for the w axis.
inline constexpr std::size_t kH3ImageVaeRopeHalf = 16;
inline constexpr std::size_t kH3ImageVaeRopeFrequencies = 8;
inline constexpr float kH3ImageVaeRopeTheta = 100.0F;
inline constexpr std::size_t kH3ImageVaeOutputPatch =
    3U * kH3ImageVaeSpatialRatio * kH3ImageVaeSpatialRatio;
inline constexpr float kH3ImageVaeNormEpsilon = 1.0e-5F;

/// Decoder tiling for one generated latent image.
struct ImageVaePlan {
  int width{0};
  int height{0};
  int latent_width{0};
  int latent_height{0};
  VideoVaeTileAxis y_axis;
  VideoVaeTileAxis x_axis;
  std::uint32_t tile_latent_height{0};
  std::uint32_t tile_latent_width{0};
  std::uint32_t patches{0};
  std::uint32_t sequence{0};
};

/// AutoencoderKLMiniMaxH3Image._split_tiles in pixel space, mapped back to
/// the latent grid. An extent of at most 256 pixels is one tile of its own
/// size.
[[nodiscard]] std::optional<ImageVaePlan> ResolveImageVaePlan(
    int latent_height, int latent_width, std::string* error = nullptr);

/// Rotary table of one tile: `cosine`/`sine` are [sequence][16]; patch rows
/// carry the length-normalized (h, w) coordinates in [-1, 1), the five suffix
/// rows position 0 (cos 1, sin 0).
void BuildImageVaeRope(std::uint32_t tile_latent_height,
                       std::uint32_t tile_latent_width,
                       std::vector<float>* cosine, std::vector<float>* sine);

/// Projected patch rows [height * width][768] (component order c, py, px)
/// to an interleaved RGB tile [height * 16][width * 16][3], still in
/// ImageNet-normalized space.
[[nodiscard]] bool UnpackImageVaePatches(std::span<const float> projected,
                                         std::uint32_t tile_latent_height,
                                         std::uint32_t tile_latent_width,
                                         std::span<float> rgb,
                                         std::string* error = nullptr);

/// The pipeline's post-processing (MiniMaxH3ImageDecodeStep): ImageNet
/// de-normalization, clamp to [0, 1] and truncation to 8 bits
/// ((x * 255).to(uint8)). Writes the 8-bit pixels and, when requested, the
/// same values as floats in [0, 1].
void FinalizeImageVaeRgb(std::span<const float> normalized_rgb,
                         std::vector<std::uint8_t>* rgb8,
                         std::vector<float>* rgb01 = nullptr);

struct ImageVaeLatentNormalization {
  std::vector<float> mean;
  std::vector<float> deviation;
};

/// `latents_mean` / `latents_std` of `<image root>/vae/config.json`, which
/// must also describe the 2D ViT decoder this file implements.
[[nodiscard]] std::optional<ImageVaeLatentNormalization>
ReadImageVaeConfig(const std::filesystem::path& vae_config,
                   std::string* error = nullptr);

// --- F32 CPU reference -----------------------------------------------------

struct ImageVaeCpuBlock {
  std::vector<float> norm1;
  std::vector<float> qkv_weight;
  std::vector<float> qkv_bias;
  std::vector<float> output_weight;
  std::vector<float> output_bias;
  std::vector<float> scale1;
  std::vector<float> norm2;
  std::vector<float> fc1_weight;
  std::vector<float> fc1_bias;
  std::vector<float> fc2_weight;
  std::vector<float> fc2_bias;
  std::vector<float> scale2;
};

struct ImageVaeCpuWeights {
  std::vector<float> post_weight;
  std::vector<float> post_bias;
  std::vector<float> embed_weight;
  std::vector<float> embed_bias;
  std::vector<float> registers;
  std::vector<ImageVaeCpuBlock> blocks;
  std::vector<float> final_norm_weight;
  std::vector<float> final_norm_bias;
  std::vector<float> output_weight;
  std::vector<float> output_bias;
};

/// Reads the decoder's F32 tensors from an H3-Image inventory, only blocks
/// [0, block_count) (a truncated decoder is the cheap oracle check).
[[nodiscard]] std::optional<ImageVaeCpuWeights> LoadImageVaeCpuWeights(
    const ModelInventory& inventory, std::size_t block_count,
    std::string* error = nullptr);

/// One decoder tile on the CPU: `latent` is the de-normalized latent tile
/// [24][height][width] (latent * std + mean, before post_quant_conv); writes
/// the ImageNet-normalized RGB tile [height * 16][width * 16][3].
[[nodiscard]] bool DecodeImageVaeTileCpu(const ImageVaeCpuWeights& weights,
                                         std::span<const float> latent,
                                         std::uint32_t latent_height,
                                         std::uint32_t latent_width,
                                         std::vector<float>* rgb,
                                         std::string* error = nullptr);

/// Whole image on the CPU, tiled and blended like the HIP decoder:
/// `normalized_latent` [24][latent_height][latent_width] as the DiT emits it.
/// Writes ImageNet-normalized interleaved RGB.
[[nodiscard]] bool DecodeImageVaeCpu(
    const ImageVaeCpuWeights& weights,
    const ImageVaeLatentNormalization& normalization,
    std::span<const float> normalized_latent, int latent_height,
    int latent_width, std::vector<float>* rgb, std::string* error = nullptr);

// --- HIP decoder -----------------------------------------------------------

struct ImageVaeOptions {
  int latent_height{0};
  int latent_width{0};
};

struct DecodedImage {
  int height{0};
  int width{0};
  // Interleaved RGB, row-major, ImageNet-normalized (before
  // FinalizeImageVaeRgb).
  std::vector<float> normalized_rgb;
};

using ImageVaeTelemetry = VideoVaeTelemetry;
using ImageVaeProgress = VideoVaeProgress;

class ImageVaeDecoder {
public:
  ~ImageVaeDecoder();

  ImageVaeDecoder(const ImageVaeDecoder&) = delete;
  ImageVaeDecoder& operator=(const ImageVaeDecoder&) = delete;
  ImageVaeDecoder(ImageVaeDecoder&&) noexcept;
  ImageVaeDecoder& operator=(ImageVaeDecoder&&) noexcept;

  [[nodiscard]] static std::unique_ptr<ImageVaeDecoder> Create(
      const ModelInventory& inventory, const ImageVaeOptions& options,
      const CancellationToken* cancellation, ImageVaeTelemetry* telemetry,
      std::string* error = nullptr);

  [[nodiscard]] bool Decode(std::span<const float> normalized_latent,
                            const CancellationToken* cancellation,
                            ImageVaeProgress progress, void* progress_opaque,
                            DecodedImage* output, ImageVaeTelemetry* telemetry,
                            std::string* error = nullptr);

  [[nodiscard]] const ImageVaePlan& plan() const noexcept;

private:
  struct Impl;
  explicit ImageVaeDecoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_IMAGE_VAE_HPP_
