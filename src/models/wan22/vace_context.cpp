// Written with AI assistance (Claude Code).
#include "src/models/wan22/vace_context.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace gufo::models::wan22 {

void SplitVaceControl(std::span<const std::uint8_t> video, int video_frames,
                      std::span<const float> mask, int frames, int height,
                      int width, std::vector<float>* inactive,
                      std::vector<float>* reactive) {
  const std::size_t pixels = static_cast<std::size_t>(frames) * height * width;
  const std::size_t frame = static_cast<std::size_t>(height) * width;
  const std::size_t video_pixels =
      video.size() >= frame * 3 * std::max(video_frames, 0)
          ? frame * std::max(video_frames, 0)
          : 0;
  inactive->assign(pixels * 3, 0.0F);
  reactive->assign(pixels * 3, 0.0F);
  for (std::size_t p = 0; p < pixels; ++p) {
    const float m = mask.size() == pixels ? mask[p] : 1.0F;
    for (int c = 0; c < 3; ++c) {
      // ComfyUI: images in [0, 1], grey 0.5 without a control video; the
      // VAE sees 2x - 1.
      const float v = p < video_pixels
                          ? static_cast<float>(video[p * 3 + c]) / 255.0F - 0.5F
                          : 0.0F;
      (*inactive)[p * 3 + c] = 2.0F * (v * (1.0F - m) + 0.5F) - 1.0F;
      (*reactive)[p * 3 + c] = 2.0F * (v * m + 0.5F) - 1.0F;
    }
  }
}

std::vector<float> BuildVaceContext(const VaceContextInputs& in) {
  if (in.frames < 1 || (in.frames - 1) % 4 != 0 || in.height % 8 != 0 ||
      in.width % 8 != 0 || in.height < 8 || in.width < 8)
    return {};
  const int lf = 1 + (in.frames - 1) / 4, lh = in.height / 8,
            lw = in.width / 8;
  const int refs = static_cast<int>(in.references.size());
  const int frames = refs + lf;
  const std::size_t plane = static_cast<std::size_t>(lh) * lw;
  const std::size_t latent = static_cast<std::size_t>(16) * lf * plane;
  const std::size_t pixels =
      static_cast<std::size_t>(in.frames) * in.height * in.width;
  if (in.inactive.size() != latent || in.reactive.size() != latent ||
      (!in.mask.empty() && in.mask.size() != pixels))
    return {};
  for (const auto& ref : in.references)
    if (ref.size() != 16 * plane)
      return {};
  const std::size_t channel = static_cast<std::size_t>(frames) * plane;
  std::vector<float> out(static_cast<std::size_t>(96) * channel, 0.0F);
  // Reference frames: [latent | 0], mask 0.
  for (int r = 0; r < refs; ++r)
    for (int c = 0; c < 16; ++c)
      std::copy_n(in.references[r].begin() + c * plane, plane,
                  out.begin() + c * channel + r * plane);
  // Video frames: [inactive | reactive].
  for (int c = 0; c < 16; ++c)
    for (int f = 0; f < lf; ++f) {
      const std::size_t source = (static_cast<std::size_t>(c) * lf + f) * plane;
      const std::size_t target = c * channel + (refs + f) * plane;
      std::copy_n(in.inactive.begin() + source, plane, out.begin() + target);
      std::copy_n(in.reactive.begin() + source, plane,
                  out.begin() + 16 * channel + target);
    }
  // Mask: pixel unshuffle by 8 (channel kh * 8 + kw), nearest-exact in time
  // (source frame floor((f + 0.5) * frames / latent_frames)).
  for (int f = 0; f < lf; ++f) {
    const int source = std::min(
        static_cast<int>(std::floor((f + 0.5) * in.frames / lf)),
        in.frames - 1);
    for (int kh = 0; kh < 8; ++kh)
      for (int kw = 0; kw < 8; ++kw) {
        float* target = out.data() + (32 + kh * 8 + kw) * channel +
                        (refs + f) * plane;
        for (int y = 0; y < lh; ++y)
          for (int x = 0; x < lw; ++x)
            target[static_cast<std::size_t>(y) * lw + x] =
                in.mask.empty()
                    ? 1.0F
                    : in.mask[(static_cast<std::size_t>(source) * in.height +
                               y * 8 + kh) *
                                  in.width +
                              x * 8 + kw];
      }
  }
  return out;
}

std::vector<double> VaceSigmas(int steps, double shift) {
  std::vector<double> sigmas;
  if (steps < 1)
    return {0.0};
  for (int i = 0; i < steps; ++i) {
    const double t = 1.0 - static_cast<double>(i) / steps;
    sigmas.push_back(shift * t / (1.0 + (shift - 1.0) * t));
  }
  sigmas.push_back(0.0);
  return sigmas;
}

int VaceSwitchStep(const std::vector<double>& sigmas, double boundary,
                   int requested) {
  const int steps = std::max(static_cast<int>(sigmas.size()) - 1, 0);
  if (requested >= 0 && requested <= steps)
    return requested;
  for (int i = 0; i < steps; ++i)
    if (1000.0 * sigmas[i] < boundary * 1000.0)
      return i;
  return steps;
}

bool KeepVaceExpertsResident(bool requested, std::uint64_t free_bytes,
                             std::uint64_t expert_bytes,
                             std::uint64_t job_bytes) {
  if (!requested || expert_bytes == 0)
    return false;
  const std::uint64_t need = expert_bytes + job_bytes + kVaceExpertMarginBytes;
  return need >= expert_bytes && need <= free_bytes;  // no overflow
}

std::uint64_t VaceDeviceFreeBytes(std::uint64_t admission_free, bool has_local,
                                  std::uint64_t local_budget,
                                  std::uint64_t local_usage) {
  if (!has_local)
    return admission_free;
  const std::uint64_t local_left =
      local_budget > local_usage ? local_budget - local_usage : 0;
  return std::min(admission_free, local_left);
}

bool VaceJobFitsBesideExperts(std::uint64_t free_bytes,
                              std::uint64_t job_bytes) {
  const std::uint64_t need = job_bytes + kVaceExpertMarginBytes;
  return need >= job_bytes && need <= free_bytes;  // no overflow
}

std::uint64_t VaceTextEncoderDeviceBytes(std::uint64_t text_encoder_files) {
  return text_encoder_files > kUmt5EmbeddingBytes
             ? text_encoder_files - kUmt5EmbeddingBytes
             : text_encoder_files;
}

std::uint64_t Transformer21JobBytes(int frames, int latent_height,
                                   int latent_width, int slots) {
  if (frames <= 0 || latent_height <= 0 || latent_width <= 0)
    return 0;
  constexpr std::uint64_t W = 5120, F = 13824, kPatchIn = 144,
                          kPatchOut = 64, kRopePairs = 64, kChunk = 8192,
                          kLayers = 40, kText = 512;
  const auto padded = [](std::uint64_t rows) { return (rows + 31) & ~31ULL; };
  const std::uint64_t T = static_cast<std::uint64_t>(frames) *
                          (latent_height / 2) * (latent_width / 2);
  const std::uint64_t S = static_cast<std::uint64_t>(std::max(slots, 0));
  // Transformer21::Impl::Reserve.
  std::uint64_t bytes = T * kPatchIn * 4 + T * kPatchIn * 2 + T * W * 4 +
                        T * W * 2 + T * 3 * W * 2 + 3 * T * W * 2 +
                        padded(T) * W * 2 + 2 * T * W * 2 +
                        std::min(T, kChunk) * F * 2 + T * kRopePairs * 8 +
                        T * kPatchOut * 4;
  // Context slots: keys, packed values, embedded text.
  bytes += S * (kLayers * kText * W * 2 + kLayers * padded(kText) * W * 2 +
                kText * W * 2);
  return bytes;
}

std::uint64_t VaceSamplingBytes(int frames, int latent_height,
                                int latent_width, int slots,
                                int vace_blocks) {
  if (frames <= 0 || latent_height <= 0 || latent_width <= 0)
    return 0;
  constexpr std::uint64_t W = 5120, kText = 512;
  const auto padded = [](std::uint64_t rows) { return (rows + 31) & ~31ULL; };
  const std::uint64_t T = static_cast<std::uint64_t>(frames) *
                          (latent_height / 2) * (latent_width / 2);
  const std::uint64_t S = static_cast<std::uint64_t>(std::max(slots, 0));
  const std::uint64_t B = static_cast<std::uint64_t>(std::max(vace_blocks, 0));
  std::uint64_t bytes =
      Transformer21JobBytes(frames, latent_height, latent_width, slots);
  // VACE: context [96, frames, h, w] FP32, stream [T, W] FP32, text keys /
  // values per slot, text scratch.
  bytes += 96ULL * frames * latent_height * latent_width * 4 + T * W * 4 +
           S * (B * kText * W * 2 + B * padded(kText) * W * 2) +
           2 * kText * W * 2;
  return bytes;
}

std::uint64_t VaceJobPeakBytes(std::uint64_t text_encoder_bytes,
                               std::uint64_t vae_bytes,
                               std::uint64_t sampling_bytes) {
  return std::max({text_encoder_bytes, vae_bytes, sampling_bytes});
}

bool VaceExpertsFit(std::uint64_t free_bytes, std::uint64_t expert_bytes,
                    std::uint64_t peak_bytes) {
  if (expert_bytes == 0)
    return false;
  const std::uint64_t job = peak_bytes + kVaceReserveBytes;
  const std::uint64_t need = expert_bytes + job;
  return job >= peak_bytes && need >= expert_bytes &&
         need <= free_bytes;  // no overflow
}

bool VaceJobFitsBesideResident(std::uint64_t free_bytes,
                               std::uint64_t peak_bytes) {
  const std::uint64_t need = peak_bytes + kVaceReserveBytes;
  return need >= peak_bytes && need <= free_bytes;  // no overflow
}

}  // namespace gufo::models::wan22
