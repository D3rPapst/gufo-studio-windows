// MiniMax H3 Fun-ControlNet-Union inputs; see control.hpp.
// Written with AI assistance (Claude Code).
#include "src/models/minimax_h3/control.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <utility>

#include "src/models/minimax_h3/json.hpp"

namespace gufo::minimax_h3 {
namespace {

constexpr int kLatentChannels = 24;
constexpr int kSpatialCompression = 16;
// The VisualVAE's input normalization (ImageNet statistics): the pixel the
// VAE maps to zero.
constexpr std::array<float, 3> kImageNetMean = {0.485F, 0.456F, 0.406F};

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

struct Axis {
  std::vector<int> low;
  std::vector<int> high;
  std::vector<float> weight;  // of `high`
};

// torch's area_pixel_compute_source_index with align_corners=False.
Axis LinearAxis(int input, int output) {
  Axis axis;
  axis.low.resize(static_cast<std::size_t>(output));
  axis.high.resize(static_cast<std::size_t>(output));
  axis.weight.resize(static_cast<std::size_t>(output));
  const float scale = static_cast<float>(input) / static_cast<float>(output);
  for (int index = 0; index < output; ++index) {
    float source = scale * (static_cast<float>(index) + 0.5F) - 0.5F;
    if (source < 0.0F) {
      source = 0.0F;
    }
    const int low = std::min(static_cast<int>(source), input - 1);
    const int high = low < input - 1 ? low + 1 : low;
    axis.low[static_cast<std::size_t>(index)] = low;
    axis.high[static_cast<std::size_t>(index)] = high;
    axis.weight[static_cast<std::size_t>(index)] =
        source - static_cast<float>(low);
  }
  return axis;
}

// Python's round(): half to even.
int RoundHalfEven(double value) {
  return static_cast<int>(std::nearbyint(value));
}

std::vector<std::uint8_t> ToRgb8(std::span<const float> values) {
  std::vector<std::uint8_t> bytes(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    bytes[index] = static_cast<std::uint8_t>(
        std::lround(std::clamp(values[index], 0.0F, 1.0F) * 255.0F));
  }
  return bytes;
}

bool ReadLatentStatistics(const ModelInventory& inventory,
                          std::array<float, kLatentChannels>* mean,
                          std::array<float, kLatentChannels>* deviation,
                          std::string* error) {
  try {
    const json::Value config = json::ParseFile(
        inventory.model_root() / "FL2VA/video_vae/config.json", 1U << 20U);
    const json::Value* mean_value = config.Find("latents_mean");
    const json::Value* std_value = config.Find("latents_std");
    if (mean_value == nullptr || std_value == nullptr ||
        !mean_value->IsArray() || !std_value->IsArray() ||
        mean_value->AsArray().size() != mean->size() ||
        std_value->AsArray().size() != deviation->size()) {
      throw json::Error("latent normalization shape differs");
    }
    for (std::size_t index = 0; index < mean->size(); ++index) {
      (*mean)[index] =
          static_cast<float>(mean_value->AsArray()[index].AsDouble());
      (*deviation)[index] =
          static_cast<float>(std_value->AsArray()[index].AsDouble());
      if (!((*deviation)[index] > 0.0F)) {
        throw json::Error("latent normalization contains invalid values");
      }
    }
  } catch (const std::exception& exception) {
    SetError(error, "MiniMax H3 VisualVAE config failed: " +
                        std::string(exception.what()));
    return false;
  }
  return true;
}

// ComfyUI's VisualVAE encode of a clip: the posterior mean, normalized,
// [24][latent frames][h][w].
bool EncodeMean(const ModelInventory& inventory, std::span<const float> frames,
                int frame_count, int width, int height, int latent_frames,
                const std::array<float, kLatentChannels>& mean,
                const std::array<float, kLatentChannels>& deviation,
                const CancellationToken* cancellation,
                std::vector<float>* latent,
                VideoVaeEncoderTelemetry* telemetry, std::string* error) {
  const std::vector<std::uint8_t> rgb = ToRgb8(frames);
  std::vector<KeyframeMoments> moments;
  VideoVaeEncoderTelemetry local;
  if (!EncodeGuideClipMoments(inventory, width, height, rgb, frame_count,
                              cancellation, &moments, &local, error)) {
    return false;
  }
  if (telemetry != nullptr) {
    telemetry->load_ms += local.load_ms;
    telemetry->encode_ms += local.encode_ms;
    telemetry->tiles += local.tiles;
    telemetry->weight_bytes = std::max(telemetry->weight_bytes,
                                       local.weight_bytes);
    telemetry->scratch_bytes = std::max(telemetry->scratch_bytes,
                                        local.scratch_bytes);
  }
  if (static_cast<int>(moments.size()) != latent_frames) {
    SetError(error, "MiniMax H3 control latent has " +
                        std::to_string(moments.size()) + " frames, expected " +
                        std::to_string(latent_frames));
    return false;
  }
  const int latent_height = height / kSpatialCompression;
  const int latent_width = width / kSpatialCompression;
  const std::size_t plane = static_cast<std::size_t>(latent_height) *
                            static_cast<std::size_t>(latent_width);
  latent->assign(static_cast<std::size_t>(kLatentChannels) *
                     static_cast<std::size_t>(latent_frames) * plane,
                 0.0F);
  for (int frame = 0; frame < latent_frames; ++frame) {
    const KeyframeMoments& moment = moments[static_cast<std::size_t>(frame)];
    if (moment.latent_height != latent_height ||
        moment.latent_width != latent_width ||
        moment.moments.size() < plane * kLatentChannels) {
      SetError(error, "MiniMax H3 control latent shape differs");
      return false;
    }
    for (int channel = 0; channel < kLatentChannels; ++channel) {
      const float* source =
          moment.moments.data() + static_cast<std::size_t>(channel) * plane;
      float* target = latent->data() +
                      (static_cast<std::size_t>(channel) * latent_frames +
                       static_cast<std::size_t>(frame)) *
                          plane;
      for (std::size_t index = 0; index < plane; ++index) {
        target[index] =
            (source[index] - mean[static_cast<std::size_t>(channel)]) /
            deviation[static_cast<std::size_t>(channel)];
      }
    }
  }
  return true;
}

}  // namespace

std::vector<float> FitFramesBilinearCenter(const ControlFrames& input,
                                           int channels, int frames,
                                           int width, int height) {
  std::vector<float> output;
  if (input.empty() || input.width <= 0 || input.height <= 0 || frames <= 0 ||
      width <= 0 || height <= 0) {
    return output;
  }
  const double old_aspect =
      static_cast<double>(input.width) / static_cast<double>(input.height);
  const double new_aspect =
      static_cast<double>(width) / static_cast<double>(height);
  int x = 0;
  int y = 0;
  if (old_aspect > new_aspect) {
    x = RoundHalfEven((input.width - input.width * (new_aspect / old_aspect)) /
                      2.0);
  } else if (old_aspect < new_aspect) {
    y = RoundHalfEven(
        (input.height - input.height * (old_aspect / new_aspect)) / 2.0);
  }
  const int crop_width = input.width - 2 * x;
  const int crop_height = input.height - 2 * y;
  const Axis rows = LinearAxis(crop_height, height);
  const Axis columns = LinearAxis(crop_width, width);
  const std::size_t frame_values = static_cast<std::size_t>(input.width) *
                                   static_cast<std::size_t>(input.height) *
                                   static_cast<std::size_t>(channels);
  output.resize(static_cast<std::size_t>(frames) * height * width * channels);
  for (int frame = 0; frame < frames; ++frame) {
    const std::uint8_t* source =
        input.pixels.data() +
        frame_values * static_cast<std::size_t>(std::min(frame, input.frames - 1));
    float* target = output.data() + static_cast<std::size_t>(frame) * height *
                                        width * channels;
    const auto at = [&](int row, int column, int channel) {
      return static_cast<float>(
                 source[(static_cast<std::size_t>(row + y) * input.width +
                         static_cast<std::size_t>(column + x)) *
                            channels +
                        channel]) /
             255.0F;
    };
    for (int row = 0; row < height; ++row) {
      const int r0 = rows.low[static_cast<std::size_t>(row)];
      const int r1 = rows.high[static_cast<std::size_t>(row)];
      const float lr = rows.weight[static_cast<std::size_t>(row)];
      for (int column = 0; column < width; ++column) {
        const int c0 = columns.low[static_cast<std::size_t>(column)];
        const int c1 = columns.high[static_cast<std::size_t>(column)];
        const float lc = columns.weight[static_cast<std::size_t>(column)];
        for (int channel = 0; channel < channels; ++channel) {
          const float top = (1.0F - lc) * at(r0, c0, channel) +
                            lc * at(r0, c1, channel);
          const float bottom = (1.0F - lc) * at(r1, c0, channel) +
                               lc * at(r1, c1, channel);
          target[(static_cast<std::size_t>(row) * width + column) * channels +
                 channel] = (1.0F - lr) * top + lr * bottom;
        }
      }
    }
  }
  return output;
}

std::vector<float> ResizeTrilinear(std::span<const float> input, int frames,
                                   int height, int width, int out_frames,
                                   int out_height, int out_width) {
  std::vector<float> output(static_cast<std::size_t>(out_frames) * out_height *
                            out_width);
  const Axis t_axis = LinearAxis(frames, out_frames);
  const Axis h_axis = LinearAxis(height, out_height);
  const Axis w_axis = LinearAxis(width, out_width);
  const auto at = [&](int t, int h, int w) {
    return input[(static_cast<std::size_t>(t) * height + h) * width + w];
  };
  for (int t = 0; t < out_frames; ++t) {
    const int t0 = t_axis.low[static_cast<std::size_t>(t)];
    const int t1 = t_axis.high[static_cast<std::size_t>(t)];
    const float lt = t_axis.weight[static_cast<std::size_t>(t)];
    for (int h = 0; h < out_height; ++h) {
      const int h0 = h_axis.low[static_cast<std::size_t>(h)];
      const int h1 = h_axis.high[static_cast<std::size_t>(h)];
      const float lh = h_axis.weight[static_cast<std::size_t>(h)];
      for (int w = 0; w < out_width; ++w) {
        const int w0 = w_axis.low[static_cast<std::size_t>(w)];
        const int w1 = w_axis.high[static_cast<std::size_t>(w)];
        const float lw = w_axis.weight[static_cast<std::size_t>(w)];
        const auto plane = [&](int tt) {
          return (1.0F - lh) * ((1.0F - lw) * at(tt, h0, w0) + lw * at(tt, h0, w1)) +
                 lh * ((1.0F - lw) * at(tt, h1, w0) + lw * at(tt, h1, w1));
        };
        output[(static_cast<std::size_t>(t) * out_height + h) * out_width + w] =
            (1.0F - lt) * plane(t0) + lt * plane(t1);
      }
    }
  }
  return output;
}

float ControlPercentToSigma(float percent, float shift) {
  if (percent <= 0.0F) {
    return 1.0F;
  }
  if (percent >= 1.0F) {
    return 0.0F;
  }
  const float t = 1.0F - percent;
  return shift * t / (1.0F + (shift - 1.0F) * t);
}

bool BuildControlRows(const ModelInventory& inventory,
                      const GenerationControl& control,
                      const GenerationGeometry& geometry, bool post_norm,
                      const CancellationToken* cancellation,
                      std::vector<float>* rows,
                      VideoVaeEncoderTelemetry* telemetry,
                      std::string* error) {
  const int latent_frames = geometry.temporal.video_latent_frames;
  const int latent_height = geometry.latent_height;
  const int latent_width = geometry.latent_width;
  const int width = latent_width * kSpatialCompression;
  const int height = latent_height * kSpatialCompression;
  const int frame_count = std::max((latent_frames - 2) / 5, 0) * 17 + 5;
  if (control.video.empty() && control.mask.empty()) {
    SetError(error, "a MiniMax H3 control needs a control video or a mask");
    return false;
  }
  if (width != geometry.width || height != geometry.height ||
      latent_height % 2 != 0 || latent_width % 2 != 0) {
    SetError(error, "MiniMax H3 control canvas differs from the latent grid");
    return false;
  }
  std::array<float, kLatentChannels> mean{};
  std::array<float, kLatentChannels> deviation{};
  if (!ReadLatentStatistics(inventory, &mean, &deviation, error)) {
    return false;
  }
  const std::size_t plane = static_cast<std::size_t>(latent_height) *
                            static_cast<std::size_t>(latent_width);
  const std::size_t channel_values =
      static_cast<std::size_t>(latent_frames) * plane;
  std::vector<float> latent(kH3ControlChannels * channel_values, 0.0F);
  if (!control.video.empty()) {
    const std::vector<float> frames =
        FitFramesBilinearCenter(control.video, 3, frame_count, width, height);
    std::vector<float> hint;
    if (frames.empty() ||
        !EncodeMean(inventory, frames, frame_count, width, height,
                    latent_frames, mean, deviation, cancellation, &hint,
                    telemetry, error)) {
      if (error != nullptr && error->empty()) {
        *error = "MiniMax H3 control video is not a valid frame set";
      }
      return false;
    }
    std::copy(hint.begin(), hint.end(), latent.begin());
  }
  if (!control.mask.empty()) {
    // (mask > 0.5) before the resize, (resized > 0.5) after it.
    ControlFrames binary = control.mask;
    for (std::uint8_t& value : binary.pixels) {
      value = static_cast<float>(value) / 255.0F > 0.5F ? std::uint8_t{255}
                                                      : std::uint8_t{0};
    }
    const std::vector<float> fitted =
        FitFramesBilinearCenter(binary, 1, frame_count, width, height);
    if (fitted.empty()) {
      SetError(error, "MiniMax H3 control mask is not a valid frame set");
      return false;
    }
    std::vector<float> visibility(fitted.size());
    for (std::size_t index = 0; index < fitted.size(); ++index) {
      visibility[index] = fitted[index] > 0.5F ? 0.0F : 1.0F;
    }
    std::vector<float> source;
    if (!control.source.empty()) {
      source =
          FitFramesBilinearCenter(control.source, 3, frame_count, width, height);
    } else {
      source.assign(visibility.size() * 3U, 0.0F);
    }
    if (source.size() != visibility.size() * 3U) {
      SetError(error, "MiniMax H3 control source is not a valid frame set");
      return false;
    }
    for (std::size_t pixel = 0; pixel < visibility.size(); ++pixel) {
      for (std::size_t channel = 0; channel < 3; ++channel) {
        float& value = source[pixel * 3U + channel];
        value *= visibility[pixel];
        if (post_norm) {
          value += (1.0F - visibility[pixel]) * kImageNetMean[channel];
        }
      }
    }
    std::vector<float> masked;
    if (!EncodeMean(inventory, source, frame_count, width, height,
                    latent_frames, mean, deviation, cancellation, &masked,
                    telemetry, error)) {
      return false;
    }
    const std::vector<float> visibility_latent =
        ResizeTrilinear(visibility, frame_count, height, width, latent_frames,
                        latent_height, latent_width);
    std::copy(visibility_latent.begin(), visibility_latent.end(),
              latent.begin() + static_cast<std::ptrdiff_t>(
                                   kLatentChannels * channel_values));
    std::copy(masked.begin(), masked.end(),
              latent.begin() + static_cast<std::ptrdiff_t>(
                                   (kLatentChannels + 1) * channel_values));
  }
  rows->assign(latent.size(), 0.0F);
  if (!PatchifyVideo(latent, static_cast<int>(kH3ControlChannels),
                     latent_frames, latent_height, latent_width, *rows,
                     error)) {
    return false;
  }
  return true;
}

}  // namespace gufo::minimax_h3
