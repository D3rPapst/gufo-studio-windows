#ifndef GUFO_MODELS_WAN22_VACE_CONTEXT_HPP_
#define GUFO_MODELS_WAN22_VACE_CONTEXT_HPP_

// Written with AI assistance (Claude Code).
// Host-side VACE conditioning (no device code; unit-tested in
// tests/models/wan22_vace_context_test.cpp).

#include <cstdint>
#include <span>
#include <vector>

namespace gufo::models::wan22 {

/// Inputs of BuildVaceContext, all at the video size (width, height
/// multiples of 16, frames = 4k + 1) and already normalized latents from
/// VaeEncoder::EncodeVideo(..., normalize = true).
struct VaceContextInputs {
  int frames{0};  // pixel frames of the video
  int height{0};  // pixels
  int width{0};
  /// [16, latent_frames, height / 8, width / 8] each.
  std::span<const float> inactive;
  std::span<const float> reactive;
  /// Pixel mask [frames, height, width] in 0..1 (1 = generate); empty = all
  /// ones.
  std::span<const float> mask;
  /// Normalized latents of the reference images, [16, 1, h / 8, w / 8] each,
  /// placed in front of the video in this order.
  std::vector<std::span<const float>> references;
};

/// ComfyUI's WanVaceToVideo + WAN21_Vace.extra_conds: concatenates
/// [inactive | reactive] (16 + 16 channels; reference frames: [latent | 0]),
/// then the mask pixel-unshuffled by 8 (channel kh * 8 + kw) and resampled
/// to the latent frames with nearest-exact (zeros on the reference frames).
/// Returns [96, references + latent_frames, height / 8, width / 8], empty on
/// a size mismatch.
[[nodiscard]] std::vector<float> BuildVaceContext(
    const VaceContextInputs& inputs);

/// ComfyUI's control-video split before the VAE: with the control video v in
/// [0, 1] and the mask m, inactive = (v - 0.5) * (1 - m) + 0.5 and reactive
/// = (v - 0.5) * m + 0.5; written in [-1, 1] for VaeEncoder ([frames, height,
/// width, 3] each). `video` RGB8 [video_frames, height, width, 3] (its first
/// video_frames frames are read; later frames and an empty video are grey
/// 0.5 as ComfyUI pads), `mask` as in VaceContextInputs (empty = all ones).
void SplitVaceControl(std::span<const std::uint8_t> video, int video_frames,
                      std::span<const float> mask, int frames, int height,
                      int width, std::vector<float>* inactive,
                      std::vector<float>* reactive);

/// The flow sigmas of a VACE run: steps + 1 values, t = 1 - i / steps
/// shifted (sigma = shift * t / (1 + (shift - 1) * t)), the last one 0.
[[nodiscard]] std::vector<double> VaceSigmas(int steps, double shift);

/// Wan 2.2 A14B (two experts): the first step the low-noise expert runs.
/// `requested` in 0..steps is taken as given (ComfyUI's KSamplerAdvanced
/// end_at_step of the high-noise sampler); otherwise the first step whose
/// timestep 1000 * sigma is below boundary * 1000 (Wan 2.2 / VideoX-Fun:
/// the high-noise expert runs while t >= boundary * num_train_timesteps).
/// `sigmas` as VaceSigmas (steps + 1 values).
[[nodiscard]] int VaceSwitchStep(const std::vector<double>& sigmas,
                                 double boundary, int requested);

/// Room a resident pair of Wan 2.2 Fun VACE experts leaves on top of the
/// per-job loads (activations of both transformers, VAE tiles, HIP pools).
inline constexpr std::uint64_t kVaceExpertMarginBytes = std::uint64_t{6}
                                                        << 30;

/// Wan 2.2 I2V A14B (Fun VACE counts per phase, see VaceExpertsFit): whether
/// both experts (DiT + VACE module each) are loaded together and stay in the worker over jobs. `requested`
/// (gufo-vace22.json experts_resident, default true); `free_bytes` what the
/// device can still give with nothing of this model loaded (the admission
/// reading); `expert_bytes` the weights of both experts; `job_bytes` what
/// every job loads beside them (umT5, VAE). False: the experts load one after
/// the other per job (the high-noise one released before the low-noise one).
[[nodiscard]] bool KeepVaceExpertsResident(bool requested,
                                           std::uint64_t free_bytes,
                                           std::uint64_t expert_bytes,
                                           std::uint64_t job_bytes);

/// The free device memory the Wan 2.2 Fun VACE residency checks count on:
/// the admission reading `admission_free`, capped by the dedicated (DXGI
/// local) budget left when the OS reports one (`has_local`). On a Windows
/// iGPU (Strix Halo, 48 GB carve-out) hipMemGetInfo also counts the shared
/// segment (system RAM behind the commit limit), which the umT5 of the next
/// job cannot get once the experts fill the dedicated memory (2.2.6:
/// "Wan umT5 device allocation: out of memory" on the second job).
[[nodiscard]] std::uint64_t VaceDeviceFreeBytes(std::uint64_t admission_free,
                                                bool has_local,
                                                std::uint64_t local_budget,
                                                std::uint64_t local_usage);

/// Before a job on resident Fun VACE experts: whether the per-job loads
/// (`job_bytes`, umT5 + VAE) and the margin still fit in `free_bytes` (read
/// with the experts loaded). False: the experts are released for this job
/// and the worker runs them one after the other from then on.
[[nodiscard]] bool VaceJobFitsBesideExperts(std::uint64_t free_bytes,
                                            std::uint64_t job_bytes);

/// Wan 2.2 Fun VACE residency, counted per phase of a job instead of with
/// file sizes and a flat margin. A job loads umT5, then the VAE encoder,
/// then samples, then loads the VAE decoder: these never sit on the device
/// together, so beside resident experts only the largest of them needs room
/// (every expert frees its job buffers after its steps, see
/// Transformer21::ReleaseJobBuffers / Vace::ReleaseJobBuffers).

/// Room kept free on top of the largest phase: HIP / rocBLAS / hipBLASLt
/// workspaces, umT5 activations (~0.1 GB), allocator rounding.
inline constexpr std::uint64_t kVaceReserveBytes = std::uint64_t{3} << 29;

/// umT5-XXL token embedding [256384, 4096] BF16: read from its shard per
/// prompt, never on the device.
inline constexpr std::uint64_t kUmt5EmbeddingBytes =
    std::uint64_t{256384} * 4096 * 2;

/// The VAE encoder / decoder activations of one job beside their weights
/// (estimate for 832x480x81; the untiled causal Wan 2.1 VAE).
inline constexpr std::uint64_t kVaceVaeWorkBytes = std::uint64_t{4} << 30;

/// Device bytes umT5 holds while it encodes: its `text_encoder/` files
/// without the token embedding.
[[nodiscard]] std::uint64_t VaceTextEncoderDeviceBytes(
    std::uint64_t text_encoder_files);

/// One expert's job buffers while it samples a latent of `frames` (incl.
/// reference frames) x `latent_height` x `latent_width`: the Transformer21
/// activations of that token count (as its Reserve), `slots` prepared
/// context slots (1 without CFG, 2 with), and the VACE module's context,
/// context stream and text keys / values of `vace_blocks` blocks.
[[nodiscard]] std::uint64_t VaceSamplingBytes(int frames, int latent_height,
                                              int latent_width, int slots,
                                              int vace_blocks);

/// The largest phase of a job beside the experts: umT5, VAE (weights +
/// work) or one expert sampling.
[[nodiscard]] std::uint64_t VaceJobPeakBytes(std::uint64_t text_encoder_bytes,
                                             std::uint64_t vae_bytes,
                                             std::uint64_t sampling_bytes);

/// Both experts (`expert_bytes`, their files incl. LoRA: slightly more than
/// they take on the device) and the largest job phase `peak_bytes` plus
/// kVaceReserveBytes fit in `free_bytes` (read with nothing of this model
/// loaded).
[[nodiscard]] bool VaceExpertsFit(std::uint64_t free_bytes,
                                  std::uint64_t expert_bytes,
                                  std::uint64_t peak_bytes);

/// Before a job on resident experts (`free_bytes` read with them loaded and
/// their job buffers released): the largest job phase plus
/// kVaceReserveBytes fits.
[[nodiscard]] bool VaceJobFitsBesideResident(std::uint64_t free_bytes,
                                             std::uint64_t peak_bytes);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_VACE_CONTEXT_HPP_
