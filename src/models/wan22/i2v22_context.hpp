#ifndef GUFO_MODELS_WAN22_I2V22_CONTEXT_HPP_
#define GUFO_MODELS_WAN22_I2V22_CONTEXT_HPP_

// Written with AI assistance (Claude Code).
// Host-side Wan 2.2 I2V A14B conditioning (no device code; unit-tested in
// tests/models/wan22/vace_context_test.cpp).

#include <cstddef>
#include <span>
#include <vector>

namespace gufo::models::wan22 {

/// Wan 2.2 I2V A14B (Wan-Video/Wan2.2 wan/configs/wan_i2v_A14B.py): the
/// high-noise expert runs while t >= boundary * 1000; sample_shift 5.
inline constexpr double kI2V22Boundary = 0.900;
inline constexpr double kI2V22Shift = 5.0;

/// The 20 conditioning channels of the 36-channel model input behind the 16
/// noisy latent channels, as ComfyUI's WanImageToVideo + WAN21.concat_cond
/// (and Wan 2.2 image2video.py) form them: 4 mask channels, 1 on the first
/// latent frame (the start image) and 0 elsewhere, then the 16 channels of
/// `condition`, the normalized VAE latent of [start image, grey 0.5 ...]
/// [16, latent_frames, height, width]. Returns [20, latent_frames, height,
/// width] channel-major, empty on a size mismatch.
[[nodiscard]] std::vector<float> I2VConditionChannels(
    std::span<const float> condition, int latent_frames, int height,
    int width);

/// The full model input [36, f, h, w]: `latents` [16, f, h, w] followed by
/// `condition_channels` (I2VConditionChannels). Writes into `input`
/// (resized); false on a size mismatch.
bool I2VModelInput(std::span<const float> latents,
                   std::span<const float> condition_channels,
                   std::vector<float>* input);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_I2V22_CONTEXT_HPP_
