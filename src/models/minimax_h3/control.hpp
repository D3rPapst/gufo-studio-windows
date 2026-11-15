// MiniMax H3 Fun-ControlNet-Union inputs (ComfyUI's
// MiniMaxH3FunControlNetApply): the control video, the inpaint mask and the
// source video behind it, fitted to the canvas, encoded by the VisualVAE and
// patchified into the control rows of the denoiser.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_MINIMAX_H3_CONTROL_HPP_
#define GUFO_MODELS_MINIMAX_H3_CONTROL_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/sampling.hpp"
#include "src/models/minimax_h3/video_vae_encoder.hpp"

namespace gufo::minimax_h3 {

/// Width of a control row: 49 latent channels (control video, inpaint
/// visibility, masked source) times the 1x2x2 patch.
inline constexpr std::size_t kH3ControlChannels = 49;
inline constexpr std::size_t kH3ControlColumns = kH3ControlChannels * 4U;

/// Frames at their own size, frame-major; RGB8 (3 bytes per pixel) for the
/// videos, one byte per pixel (255 = regenerate) for the mask.
struct ControlFrames {
  int width{0};
  int height{0};
  int frames{0};
  std::vector<std::uint8_t> pixels;

  [[nodiscard]] bool empty() const noexcept { return frames <= 0; }
};

struct GenerationControl {
  std::filesystem::path model;
  float strength{1.0F};
  float start_percent{0.0F};
  float end_percent{1.0F};
  ControlFrames video;
  ControlFrames mask;
  ControlFrames source;
};

/// ComfyUI's common_upscale(frames, width, height, "bilinear", "center") on
/// [0, 1] channel values: centre crop to the target aspect (Python's
/// round-half-even offsets), then torch bilinear (align_corners=False, no
/// antialias). `channels` interleaved values per pixel. Frame i of the
/// output reads input frame min(i, frames - 1).
[[nodiscard]] std::vector<float> FitFramesBilinearCenter(
    const ControlFrames& input, int channels, int frames, int width,
    int height);

/// The visibility (1 keep, 0 regenerate) of a fitted mask ([frames][h][w]
/// in [0, 1]) trilinearly resized onto the latent grid
/// ([latent_frames][latent_height][latent_width]) like
/// F.interpolate(mode="trilinear", align_corners=False).
[[nodiscard]] std::vector<float> ResizeTrilinear(std::span<const float> input,
                                                 int frames, int height,
                                                 int width, int out_frames,
                                                 int out_height,
                                                 int out_width);

/// The video sigma at which a ComfyUI percent of the schedule starts:
/// time_snr_shift(shift, 1 - percent), 1 at 0 and 0 at 1.
[[nodiscard]] float ControlPercentToSigma(float percent, float shift);

/// Builds the control rows: one row of kH3ControlColumns floats per target
/// video row (24 channels of the control video's latent, then the
/// visibility and the masked source latent when a mask is given, zero
/// elsewhere). `post_norm` puts the masked holes at the VAE's mid-gray
/// (Fun-ControlNet-Union 2.0) instead of black (v1).
[[nodiscard]] bool BuildControlRows(const ModelInventory& inventory,
                                    const GenerationControl& control,
                                    const GenerationGeometry& geometry,
                                    bool post_norm,
                                    const CancellationToken* cancellation,
                                    std::vector<float>* rows,
                                    VideoVaeEncoderTelemetry* telemetry,
                                    std::string* error = nullptr);

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_CONTROL_HPP_
