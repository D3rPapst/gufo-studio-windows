#ifndef GUFO_MODELS_ACE_STEP_VAE_HPP_
#define GUFO_MODELS_ACE_STEP_VAE_HPP_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::ace_step {

constexpr int kSampleRate = 48000;
constexpr int kSamplesPerFrame = 1920;

/// Shape and tensor names of an Oobleck decoder file. ACE-Step 1.5 has five
/// upsampling blocks with Diffusers names (`decoder.block.N.conv_t1`);
/// YuE2-Vae has six blocks with stable-audio-tools names
/// (`decoder.layers.N.layers.M`). The product of the strides is the hop
/// (1920 = 25 frames/s at 48 kHz) for both.
struct OobleckLayout {
  std::vector<int> strides;  // decoder upsampling order
  bool stable_audio_names{false};

  [[nodiscard]] static OobleckLayout AceStep() {
    return {{10, 6, 4, 4, 2}, false};
  }
  [[nodiscard]] static OobleckLayout YuE2() {
    return {{6, 5, 4, 4, 2, 2}, true};
  }
  /// Output samples of a single-pass decode of `frames` latents: each
  /// transposed convolution (kernel 2s, stride s, padding ceil(s / 2)) maps
  /// L to L * s + s - 2 * ceil(s / 2), so an odd stride loses one sample
  /// (ACE-Step: frames * 1920; YuE2-Vae: frames * 1920 - 64).
  [[nodiscard]] std::size_t OutputSamples(std::size_t frames) const {
    std::size_t length = frames;
    for (const int s : strides)
      length = length * s + s - 2 * ((s + 1) / 2);
    return length;
  }
};

/// Decoder of the ACE-Step 1.5 AutoencoderOobleck (Diffusers `vae/` or the
/// ComfyUI `ace_1.5_vae` file): 64 latent channels at 25 frames/s to 48 kHz
/// stereo. Weight norm is folded at load; convolutions run as BF16 GEMMs
/// (im2col) with FP32 activations, Snake in FP32. Long inputs are decoded in
/// overlapping chunks wider than the receptive field, so the result matches
/// a single-pass decode.
class VaeDecoder {
public:
  [[nodiscard]] static std::unique_ptr<VaeDecoder> Load(
      const std::filesystem::path& safetensors, std::string* error);
  /// Any Oobleck decoder with 64 latent channels, stereo output and the
  /// given layout (OobleckLayout::YuE2() for YuE2-Vae `model.safetensors`).
  [[nodiscard]] static std::unique_ptr<VaeDecoder> Load(
      const std::filesystem::path& safetensors, const OobleckLayout& layout,
      std::string* error);
  ~VaeDecoder();
  VaeDecoder(const VaeDecoder&) = delete;
  VaeDecoder& operator=(const VaeDecoder&) = delete;

  /// latents [frames, 64] -> planar audio [2][layout.OutputSamples(frames)]
  /// (frames * 1920 for ACE-Step).
  bool Decode(std::span<const float> latents, int frames,
              std::vector<float>* audio, std::string* error);

private:
  struct Impl;
  explicit VaeDecoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

/// Encoder of the same Oobleck VAE (the `encoder.*` tensors of the file):
/// 48 kHz stereo to the 128-channel Gaussian parameters (64 means, 64
/// scales) at 25 frames/s, `OobleckDiagonalGaussianDistribution` with
/// std = softplus(scale) + 1e-4. Same numerics as the decoder; long inputs
/// are encoded in overlapping chunks (the official `tiled_encode` trims the
/// overlap the same way).
class VaeEncoder {
public:
  [[nodiscard]] static std::unique_ptr<VaeEncoder> Load(
      const std::filesystem::path& safetensors, std::string* error);
  ~VaeEncoder();
  VaeEncoder(const VaeEncoder&) = delete;
  VaeEncoder& operator=(const VaeEncoder&) = delete;

  /// Planar stereo audio [2][samples] at 48 kHz (samples a multiple of
  /// 1920; a remainder is dropped) -> latents [frames, 64]. With `sample`
  /// the latent is drawn as `latent_dist.sample()` does, with gufo's noise
  /// from `seed`; otherwise it is the mean (`mode()`).
  bool Encode(std::span<const float> planar, std::vector<float>* latents,
              int* frames, bool sample, std::uint64_t seed, std::string* error);

private:
  struct Impl;
  explicit VaeEncoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::ace_step

#endif  // GUFO_MODELS_ACE_STEP_VAE_HPP_
