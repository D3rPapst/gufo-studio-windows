#ifndef GUFO_MODELS_WAN22_ANIMATE_HPP_
#define GUFO_MODELS_WAN22_ANIMATE_HPP_

// Written with AI assistance (Claude Code).

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/wan22/transformer21.hpp"

namespace gufo::models::wan22 {

/// The Wan 2.2 Animate 14B additions to the Wan 2.1 I2V 14B transformer
/// (Kijai's `Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors` holds both;
/// Transformer21 loads the DiT keys, this class the rest), as ComfyUI's
/// AnimateWanModel (comfy/ldm/wan/model_animate.py) runs them:
///   x = patch_embedding(input); x[frames 1..] += pose_patch_embedding(pose)
///   motion = face_encoder(motion_encoder(face crops))   [frames, 5, 5120]
///   after DiT block i with i % 5 == 0:
///     x += face_adapter.fuser_blocks[i / 5](x, motion)
/// motion_encoder is LIA's appearance encoder (StyleGAN2 conv layers on
/// 512 x 512 crops, 4 + 1 equalized linears, QR-orthonormal directions to a
/// 512-vector per frame); face_encoder three causal 1-D convolutions (4
/// heads, stride 2, 2: 4x fewer frames) and a projection to 5120, plus a
/// learned padding token; each FaceBlock is a cross-attention of the DiT
/// tokens of latent frame f (q: layer norm, linear, per-head RMS norm) to
/// the 5 motion tokens of frame f, then a linear. Frame 0 (the reference
/// image latent) gets zero motion tokens.
/// Conv layers run as BF16 GEMMs (im2col) with FP32 activations; the face
/// encoder and the directions on the host in FP32; FaceBlock weights are
/// BF16 or Kijai's scaled FP8 (unpacked into the transformer's workspace
/// like its own).
class Animate : public Transformer21Hook {
public:
  static constexpr int kPoseChannels = 16;
  static constexpr int kMotionTokens = 5;  // 4 face heads + padding token
  static constexpr int kFaceSize = 512;

  /// Uses `stream` (the transformer's) for all device work.
  [[nodiscard]] static std::unique_ptr<Animate> Load(
      const std::filesystem::path& file, hipStream_t stream,
      std::string* error);
  ~Animate() override;
  Animate(const Animate&) = delete;
  Animate& operator=(const Animate&) = delete;

  /// Normalized pose latents [16, frames, height, width] (the VAE latent of
  /// the pose video; frames = video latent frames, without the reference).
  /// Empty clears the pose.
  bool SetPose(std::span<const float> latents, int frames, int height,
               int width, std::string* error);
  /// Face crops for context slot `slot`, frame-major [count, 3, 512, 512]
  /// in [-1, 1] (AnimateChunkPlan::face). `dit_frames` is the latent frame
  /// count of the transformer input (reference included). count 0 clears
  /// the slot (no face adapter on it).
  bool SetFace(int slot, std::span<const float> face, int count,
               int dit_frames, std::string* error);
  /// SetFace with `count` crops of constant `value` (the negative of
  /// ComfyUI's WanAnimateToVideo: face * 0 - 1), one encoder pass.
  bool SetUniformFace(int slot, float value, int count, int dit_frames,
                      std::string* error);
  /// The motion vectors of the last SetFace on `slot` ([count, 512], after
  /// the directions) and the face encoder output ([dit_frames, 5, 5120]),
  /// FP32, for oracle checks.
  [[nodiscard]] const std::vector<float>& motion(int slot) const;
  [[nodiscard]] const std::vector<float>& motion_tokens(int slot) const;
  [[nodiscard]] int face_blocks() const;
  [[nodiscard]] std::size_t weight_bytes() const;

  bool Begin(const float* input, int frames, int height, int width,
             const float* temb, hipStream_t stream,
             std::string* error) override;
  bool BeginBlocks(const Transformer21BlockState& state, hipStream_t stream,
                   std::string* error) override;
  bool AfterBlock(int layer, float* hidden, hipStream_t stream,
                  std::string* error) override;

private:
  struct Impl;
  explicit Animate(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_ANIMATE_HPP_
