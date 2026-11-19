// Written with AI assistance (Claude Code).
// Host-side VACE conditioning against ComfyUI's WanVaceToVideo /
// WAN21_Vace.extra_conds layout (tools/models/vace/vace_reference.py
// --context prints the same numbers from the Python formulas), and the Wan
// 2.2 I2V A14B conditioning channels / expert switch (i2v22_context).
#include "src/models/wan22/vace_context.hpp"
#include "src/models/wan22/i2v22_context.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace wan22 = gufo::models::wan22;

namespace {

int failures = 0;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL wan22_vace_context_test: " << message << '\n';
    ++failures;
  }
}

bool Near(float a, float b) {
  return std::fabs(a - b) <= 1e-6F;
}

}  // namespace

int main() {
  // 5 pixel frames (2 latent frames) at 16x16 (latent 2x2), one reference.
  constexpr int kFrames = 5, kH = 16, kW = 16, kLf = 2, kLh = 2, kLw = 2;
  constexpr std::size_t kPlane = kLh * kLw;
  std::vector<float> inactive(16 * kLf * kPlane), reactive(inactive.size());
  for (std::size_t i = 0; i < inactive.size(); ++i) {
    inactive[i] = static_cast<float>(i);
    reactive[i] = -static_cast<float>(i);
  }
  std::vector<float> mask(static_cast<std::size_t>(kFrames) * kH * kW);
  for (int f = 0; f < kFrames; ++f)
    for (int y = 0; y < kH; ++y)
      for (int x = 0; x < kW; ++x)
        mask[(static_cast<std::size_t>(f) * kH + y) * kW + x] =
            static_cast<float>(f * 1000 + y * 16 + x);
  std::vector<float> reference(16 * kPlane);
  for (int c = 0; c < 16; ++c)
    for (std::size_t i = 0; i < kPlane; ++i)
      reference[c * kPlane + i] = 100.0F + static_cast<float>(c);

  wan22::VaceContextInputs inputs;
  inputs.frames = kFrames;
  inputs.height = kH;
  inputs.width = kW;
  inputs.inactive = inactive;
  inputs.reactive = reactive;
  inputs.mask = mask;
  inputs.references.emplace_back(reference);
  const std::vector<float> out = wan22::BuildVaceContext(inputs);
  constexpr int kOutFrames = 1 + kLf;
  constexpr std::size_t kChannel = kOutFrames * kPlane;
  Expect(out.size() == 96 * kChannel, "context size");
  if (out.size() == 96 * kChannel) {
    const auto at = [&](int c, int f, int y, int x) {
      return out[c * kChannel + f * kPlane + y * kLw + x];
    };
    // Reference frame: [latent | 0], mask 0.
    Expect(Near(at(3, 0, 1, 1), 103.0F), "reference latent");
    Expect(Near(at(20, 0, 0, 1), 0.0F), "reference reactive is zero");
    Expect(Near(at(40, 0, 1, 0), 0.0F), "reference mask is zero");
    // Video frames after it.
    Expect(Near(at(2, 1, 0, 1), inactive[(2 * kLf + 0) * kPlane + 1]),
           "inactive frame 0");
    Expect(Near(at(16 + 5, 2, 1, 0), reactive[(5 * kLf + 1) * kPlane + 2]),
           "reactive frame 1");
    // Mask channel kh * 8 + kw at latent frame f reads pixel frame
    // floor((f + 0.5) * 5 / 2) = 1, 3 at (y * 8 + kh, x * 8 + kw).
    const int kh = 3, kw = 6;
    Expect(Near(at(32 + kh * 8 + kw, 1, 1, 0),
                1000.0F * 1 + (8 + kh) * 16 + (0 + kw)),
           "mask unshuffle frame 0");
    Expect(Near(at(32 + kh * 8 + kw, 2, 0, 1),
                1000.0F * 3 + (0 + kh) * 16 + (8 + kw)),
           "mask unshuffle frame 1");
  }
  // No mask: ones; no references.
  inputs.mask = {};
  inputs.references.clear();
  const std::vector<float> plain = wan22::BuildVaceContext(inputs);
  Expect(plain.size() == 96 * kLf * kPlane, "plain size");
  if (!plain.empty())
    Expect(Near(plain[95 * kLf * kPlane + 3], 1.0F), "default mask ones");
  // Size mismatch.
  inputs.frames = 6;
  Expect(wan22::BuildVaceContext(inputs).empty(), "frames must be 4k+1");

  // Control split around grey (v - 0.5 with v in [0, 1]): inactive =
  // v' (1 - m) + 0.5, reactive = v' m + 0.5, in VAE range 2x - 1. One pixel,
  // grey padding for the second frame.
  const std::vector<std::uint8_t> video = {255, 0, 51};
  const std::vector<float> one_mask = {0.25F, 1.0F};
  std::vector<float> in, re;
  wan22::SplitVaceControl(video, 1, one_mask, 2, 1, 1, &in, &re);
  Expect(in.size() == 6 && re.size() == 6, "split sizes");
  if (in.size() == 6 && re.size() == 6) {
    // v = 0.5: inactive 2 * (0.5 * 0.75 + 0.5) - 1 = 0.75, reactive 0.25.
    Expect(Near(in[0], 0.75F) && Near(re[0], 0.25F), "split white");
    // v = -0.5: inactive -0.75, reactive -0.25.
    Expect(Near(in[1], -0.75F) && Near(re[1], -0.25F), "split black");
    // Padded grey frame: 0.
    Expect(Near(in[3], 0.0F) && Near(re[5], 0.0F), "split grey padding");
  }

  // Wan 2.2 Fun VACE expert switch. 4 steps, shift 5: sigmas 1, 0.9375,
  // 0.8333, 0.625, 0; boundary 0.875 -> high noise steps 0, 1 (as the
  // Lightning 2 + 2 split in ComfyUI).
  const std::vector<double> sigmas = wan22::VaceSigmas(4, 5.0);
  Expect(sigmas.size() == 5, "sigmas size");
  if (sigmas.size() == 5) {
    Expect(std::fabs(sigmas[0] - 1.0) < 1e-12, "sigma 0");
    Expect(std::fabs(sigmas[1] - 0.9375) < 1e-12, "sigma 1");
    Expect(std::fabs(sigmas[2] - 2.5 / 3.0) < 1e-12, "sigma 2");
    Expect(std::fabs(sigmas[3] - 0.625) < 1e-12, "sigma 3");
    Expect(sigmas[4] == 0.0, "sigma end");
  }
  Expect(wan22::VaceSwitchStep(sigmas, 0.875, -1) == 2, "switch from boundary");
  Expect(wan22::VaceSwitchStep(sigmas, 0.875, 3) == 3, "switch as requested");
  Expect(wan22::VaceSwitchStep(sigmas, 0.875, 0) == 0, "low noise only");
  Expect(wan22::VaceSwitchStep(sigmas, 0.875, 4) == 4, "high noise only");
  Expect(wan22::VaceSwitchStep(sigmas, 0.875, 9) == 2, "out of range: boundary");
  Expect(wan22::VaceSwitchStep(sigmas, 0.0, -1) == 4, "boundary 0: all high");
  Expect(wan22::VaceSwitchStep(sigmas, 1.0, -1) == 1,
         "boundary 1: only t = 1000 is high noise");
  // 20 steps, shift 12 (VideoX-Fun's 50-step default shape): t >= 875.
  const std::vector<double> long_sigmas = wan22::VaceSigmas(20, 12.0);
  const int switch_at = wan22::VaceSwitchStep(long_sigmas, 0.875, -1);
  Expect(switch_at > 0 && switch_at < 20 &&
             long_sigmas[switch_at - 1] >= 0.875 &&
             long_sigmas[switch_at] < 0.875,
         "switch at the boundary");

  // Resident experts: both (2 x ~15 GB FP8 + ~3 GB VACE module) stay when
  // they fit beside the per-job loads (umT5 + VAE) and the margin.
  constexpr std::uint64_t kGb = std::uint64_t{1} << 30;
  const std::uint64_t experts = 36 * kGb, job = 7 * kGb;
  Expect(wan22::KeepVaceExpertsResident(true, 52 * kGb, experts, job),
         "resident on the .198 (48 GB + shared)");
  Expect(wan22::KeepVaceExpertsResident(
             true, experts + job + wan22::kVaceExpertMarginBytes, experts, job),
         "resident exactly at the limit");
  Expect(!wan22::KeepVaceExpertsResident(
             true, experts + job + wan22::kVaceExpertMarginBytes - 1, experts,
             job),
         "one byte short: one after the other");
  Expect(!wan22::KeepVaceExpertsResident(true, 30 * kGb, experts, job),
         "too little free memory: one after the other");
  Expect(!wan22::KeepVaceExpertsResident(false, 200 * kGb, experts, job),
         "experts_resident false: one after the other");
  Expect(!wan22::KeepVaceExpertsResident(true, 200 * kGb, 0, job),
         "unknown expert size: one after the other");
  Expect(!wan22::KeepVaceExpertsResident(true, UINT64_MAX, UINT64_MAX - kGb,
                                         2 * kGb),
         "no overflow");
  // The .198 on 2.2.6: hipMemGetInfo 53.7 GiB (dedicated + shared), DXGI
  // local budget left ~46.6 GiB; the experts (DiT + VACE + LoRA) 35.3,
  // umT5 + VAE 11.1. Counted on the dedicated memory they run one after
  // the other (the second job's umT5 ran out of memory).
  Expect(wan22::VaceDeviceFreeBytes(53 * kGb, false, 0, 0) == 53 * kGb,
         "no DXGI budget: the admission reading");
  Expect(wan22::VaceDeviceFreeBytes(53 * kGb, true, 47 * kGb, kGb) ==
             46 * kGb,
         "dedicated budget left caps the reading");
  Expect(wan22::VaceDeviceFreeBytes(40 * kGb, true, 47 * kGb, kGb) ==
             40 * kGb,
         "a smaller admission reading stands");
  Expect(wan22::VaceDeviceFreeBytes(53 * kGb, true, kGb, 2 * kGb) == 0,
         "usage above the budget: nothing free");
  Expect(!wan22::KeepVaceExpertsResident(
             true, wan22::VaceDeviceFreeBytes(53 * kGb, true, 47 * kGb, kGb),
             35 * kGb, 11 * kGb),
         ".198 dedicated memory: one after the other");
  Expect(wan22::VaceJobFitsBesideExperts(
             11 * kGb + wan22::kVaceExpertMarginBytes, 11 * kGb),
         "job fits beside the experts exactly at the limit");
  Expect(!wan22::VaceJobFitsBesideExperts(
             11 * kGb + wan22::kVaceExpertMarginBytes - 1, 11 * kGb),
         "one byte short: release the experts");
  Expect(!wan22::VaceJobFitsBesideExperts(kGb / 2, 11 * kGb),
         "second job on the .198 (0.3 GB free): release the experts");
  Expect(!wan22::VaceJobFitsBesideExperts(UINT64_MAX, UINT64_MAX - kGb),
         "no overflow beside the experts");
  // Fun VACE per phase (.198 preset): experts 35.3 GiB of files, umT5
  // 10.6 GiB of files of which 2.0 GiB embedding stay on the host, VAE
  // 0.5 GiB, sampling 832x480x81 one expert ~5.2 GiB (no CFG).
  {
    const std::uint64_t umt5_files = 11361874539ULL;
    const std::uint64_t umt5 = wan22::VaceTextEncoderDeviceBytes(umt5_files);
    Expect(umt5 == umt5_files - wan22::kUmt5EmbeddingBytes,
           "umT5 on the device: without its token embedding");
    Expect(wan22::VaceTextEncoderDeviceBytes(kGb) == kGb,
           "umT5 smaller than its embedding: the files as given");
    const std::uint64_t sampling = wan22::VaceSamplingBytes(21, 60, 104, 1, 8);
    Expect(sampling > 5 * kGb && sampling < 11 * kGb / 2,
           "sampling 832x480x81 one slot: ~5.2 GiB");
    Expect(wan22::VaceSamplingBytes(21, 60, 104, 2, 8) > sampling,
           "CFG: a second context slot");
    Expect(wan22::VaceSamplingBytes(21, 90, 160, 1, 8) > umt5,
           "1280x720: sampling is the largest phase");
    Expect(wan22::VaceSamplingBytes(0, 60, 104, 1, 8) == 0,
           "no frames: nothing");
    const std::uint64_t vae = kGb / 2 + wan22::kVaceVaeWorkBytes;
    const std::uint64_t peak = wan22::VaceJobPeakBytes(umt5, vae, sampling);
    Expect(peak == umt5, "832x480: umT5 is the largest phase");
    const std::uint64_t experts = 37945510536ULL;  // 35.3 GiB
    const std::uint64_t free = wan22::VaceDeviceFreeBytes(
        53 * kGb, true, 47 * kGb + 6 * kGb / 10, kGb);  // 46.6 GiB
    Expect(wan22::VaceExpertsFit(free, experts, peak),
           ".198: both experts resident beside the largest phase");
    Expect(wan22::VaceExpertsFit(experts + peak + wan22::kVaceReserveBytes,
                                 experts, peak),
           "resident exactly at the limit");
    Expect(!wan22::VaceExpertsFit(
               experts + peak + wan22::kVaceReserveBytes - 1, experts, peak),
           "one byte short: one after the other");
    Expect(!wan22::VaceExpertsFit(200 * kGb, 0, peak),
           "unknown expert size: one after the other");
    Expect(!wan22::VaceExpertsFit(UINT64_MAX, UINT64_MAX - kGb, 2 * kGb),
           "no overflow (experts)");
    // Next job: experts loaded, their job buffers released (~12 GiB free).
    Expect(wan22::VaceJobFitsBesideResident(12 * kGb, peak),
           "second job: umT5 fits beside the resident experts");
    Expect(!wan22::VaceJobFitsBesideResident(kGb / 2, peak),
           "2.2.6 leftovers (0.3 GB free): release the experts");
    Expect(!wan22::VaceJobFitsBesideResident(UINT64_MAX, UINT64_MAX - 1),
           "no overflow (beside resident)");
  }

  // Wan 2.2 I2V A14B: boundary 0.900 (t >= 900 runs the high-noise
  // expert). 8 steps at shift 5: sigmas 1, .972, .9375, .893, ... -> the
  // low-noise expert from step 3 (with T2V's 0.875 it would be step 4).
  const std::vector<double> i2v_sigmas =
      wan22::VaceSigmas(8, wan22::kI2V22Shift);
  Expect(wan22::VaceSwitchStep(i2v_sigmas, wan22::kI2V22Boundary, -1) == 3,
         "I2V A14B switch at 0.900");
  Expect(wan22::VaceSwitchStep(i2v_sigmas, 0.875, -1) == 4,
         "T2V boundary switches later");
  Expect(wan22::VaceSwitchStep(wan22::VaceSigmas(4, 5.0), 0.9, -1) == 2,
         "I2V A14B 4 steps: two per expert");

  // WanImageToVideo + WAN21.concat_cond: 4 mask channels, 1 on the first
  // latent frame, then the 16 condition channels.
  {
    constexpr int kF = 3, kY = 2, kX = 2;
    constexpr std::size_t kP = kF * kY * kX, kFrame = kY * kX;
    std::vector<float> condition(16 * kP);
    for (std::size_t i = 0; i < condition.size(); ++i)
      condition[i] = static_cast<float>(i) * 0.5F;
    const std::vector<float> channels =
        wan22::I2VConditionChannels(condition, kF, kY, kX);
    Expect(channels.size() == 20 * kP, "I2V condition: 20 channels");
    bool mask_ok = channels.size() == 20 * kP;
    for (std::size_t c = 0; mask_ok && c < 4; ++c)
      for (std::size_t i = 0; i < kP; ++i)
        mask_ok =
            mask_ok && channels[c * kP + i] == (i < kFrame ? 1.0F : 0.0F);
    Expect(mask_ok, "I2V condition: mask 1 on the first latent frame only");
    bool copy_ok = channels.size() == 20 * kP;
    for (std::size_t i = 0; copy_ok && i < condition.size(); ++i)
      copy_ok = channels[4 * kP + i] == condition[i];
    Expect(copy_ok, "I2V condition: VAE latent behind the mask");
    Expect(wan22::I2VConditionChannels(condition, kF, kY, 3).empty(),
           "I2V condition: size mismatch");
    std::vector<float> latents(16 * kP, -1.0F), input;
    Expect(wan22::I2VModelInput(latents, channels, &input) &&
               input.size() == 36 * kP && input[0] == -1.0F &&
               input[16 * kP] == 1.0F && input[20 * kP + 1] == condition[1],
           "I2V model input: [16 noisy | 4 mask | 16 condition]");
    Expect(!wan22::I2VModelInput(latents, condition, &input),
           "I2V model input: size mismatch");
  }

  if (failures == 0)
    std::cout << "wan22_vace_context_test passed\n";
  return failures == 0 ? 0 : 1;
}
