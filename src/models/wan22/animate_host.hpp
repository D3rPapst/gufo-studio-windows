#ifndef GUFO_MODELS_WAN22_ANIMATE_HOST_HPP_
#define GUFO_MODELS_WAN22_ANIMATE_HOST_HPP_

// Written with AI assistance (Claude Code).
// Host (FP32 / FP64) parts of the Wan 2.2 Animate face path: the motion
// encoder's linear head with its QR directions and the face encoder
// (comfy/ldm/wan/model_animate.py Generator.get_motion after the conv net,
// FaceEncoder, AnimateWanModel.after_patch_embedding). No device code;
// unit-tested in tests/models/wan22/animate_context_test.cpp.

#include <cstddef>
#include <span>
#include <vector>

namespace gufo::models::wan22 {

/// LAPACK-style Householder QR (dgeqr2 + dorg2r, as torch.linalg.qr
/// "reduced" computes it) of a row-major [rows, columns] matrix, rows >=
/// columns: writes Q row-major [rows, columns].
void HouseholderQ(const std::vector<double>& matrix, int rows, int columns,
                  std::vector<double>* q);

/// Encoder.fc (equalized linears without activation: weights already scaled
/// by 1 / sqrt(in), row-major [out, in]) and Direction: the motion vector of
/// one frame is fc(feature) @ Q^T with Q = qr(direction + 1e-8 * eye).
struct MotionHead {
  std::vector<std::vector<float>> weights;  // 4 x [512, 512], [motion, 512]
  std::vector<std::vector<float>> biases;
  std::vector<float> q;  // [width, motion] row-major (HouseholderQ)
  int width{512};
  int motion{20};
};

/// Sets `head.q` from Direction.weight [width, motion] row-major.
void SetMotionDirections(std::span<const float> direction, MotionHead* head);

/// `feature` [width] (EncoderApp output) -> motion vector [width].
[[nodiscard]] std::vector<float> MotionVector(const MotionHead& head,
                                              std::span<const float> feature);

/// FaceEncoder weights (Conv1d [out, in, 3] row-major, biases, out_proj
/// [width, channels], padding token [width]).
struct FaceEncoderWeights {
  int in_dim{512};
  int heads{4};
  int channels{1024};
  int width{5120};
  std::vector<float> conv1_weight, conv1_bias;  // [heads * channels, in, 3]
  std::vector<float> conv2_weight, conv2_bias;  // [channels, channels, 3]
  std::vector<float> conv3_weight, conv3_bias;
  std::vector<float> out_weight, out_bias;      // [width, channels]
  std::vector<float> padding;                   // [width]
};

/// Frames after the face encoder's two stride-2 causal convolutions.
[[nodiscard]] int FaceEncoderFrames(int frames);

/// Motion vectors [frames, in_dim] -> the motion tokens the face adapter
/// reads, [dit_frames, heads + 1, width]: frame 0 zero (the reference
/// latent), then the encoder frames (each its `heads` tokens and the padding
/// token), zero frames after them or cut to dit_frames. Empty on a size
/// mismatch.
[[nodiscard]] std::vector<float> RunFaceEncoder(const FaceEncoderWeights& w,
                                                std::span<const float> motion,
                                                int frames, int dit_frames);

}  // namespace gufo::models::wan22

#endif  // GUFO_MODELS_WAN22_ANIMATE_HOST_HPP_
