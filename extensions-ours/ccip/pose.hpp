// Pose networks of the film pipeline's posen_erkennung.py, native CPU fp32:
//   RtDetr   RT-DETR r50vd (COCO + Objects365), Hugging Face transformers
//            semantics: torchvision resize to 640x640 (bilinear, antialias),
//            x/255, ResNet50-vd + hybrid encoder (AIFI + CCFM) + 6 decoder
//            layers with multi-scale deformable attention, 300 queries;
//            post_process_object_detection (sigmoid, top 300 of queries x
//            classes, score > threshold).
//   VitPose  ViTPose base (simple decoder), VitPoseImageProcessor: box ->
//            centre/scale (padding 1.25, 192:256), UDP warp to 192x256 with
//            scipy affine_transform semantics (bilinear, 0 outside), ImageNet
//            normalisation; 17 COCO heatmaps 48x64; post_process_pose_estimation
//            (arg-max, DARK refinement with an 11-tap gaussian, back to the image).
// Weights: reference/convert_pose.py. Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_CCIP_POSE_HPP_
#define GUFO_MODELS_CCIP_POSE_HPP_

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "ccip/image_ops.hpp"

namespace gufo::models::ccip {

struct ObjectBox {
  std::array<float, 4> box{};  // x0, y0, x1, y1 in image pixels (not clipped)
  int label{0};
  float score{0};
};

// Intermediate tensors for oracle tests.
struct RtDetrTaps {
  std::vector<float> input;                  // 3 x 640 x 640
  std::vector<std::vector<float>> encoder;   // 3 levels, 256 x h x w
  std::vector<float> logits;                 // 300 x classes (last decoder layer)
  std::vector<float> boxes;                  // 300 x 4 (cx, cy, w, h relative)
};

class RtDetr {
public:
  ~RtDetr();
  // `dir` holds rtdetr_r50vd_coco_o365.safetensors + .json.
  [[nodiscard]] static std::unique_ptr<RtDetr> Load(const std::filesystem::path& dir,
                                                    std::string* error);
  // All labels with score > threshold, in post_process_object_detection order
  // (score descending).
  [[nodiscard]] std::vector<ObjectBox> Detect(const RgbImage& image, float threshold,
                                              RtDetrTaps* taps = nullptr) const;
  [[nodiscard]] const std::vector<std::string>& Labels() const;

private:
  RtDetr();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct Keypoint {
  float x{0}, y{0}, score{0};
};

struct VitPoseTaps {
  std::vector<float> input;     // 3 x 256 x 192
  std::vector<float> heatmaps;  // 17 x 64 x 48
};

class VitPose {
public:
  ~VitPose();
  // `dir` holds vitpose_base_simple.safetensors + .json.
  [[nodiscard]] static std::unique_ptr<VitPose> Load(const std::filesystem::path& dir,
                                                     std::string* error);
  // 17 COCO keypoints of the person in box (x0, y0, x1, y1).
  [[nodiscard]] std::vector<Keypoint> Estimate(const RgbImage& image,
                                               const std::array<float, 4>& box,
                                               VitPoseTaps* taps = nullptr) const;
  [[nodiscard]] const std::vector<std::string>& KeypointNames() const;

private:
  VitPose();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct DwPoseTaps {
  std::vector<float> input;    // 3 x 384 x 288
  std::vector<float> simcc_x;  // 133 x 576
  std::vector<float> simcc_y;  // 133 x 768
};

// DWPose dw-ll_ucoco_384 (RTMPose-l, COCO-WholeBody: 0-16 body, 17-22 feet,
// 23-90 face (68), 91-111 left hand, 112-132 right hand), as the film
// pipeline's posen_erkennung.wholebody: box x 1.25 at 288:384, cv2
// warpAffine (bilinear, 0 outside) of the float image, (x - mean) / std,
// SimCC arg-max / 2, score = min(max_x, max_y). Weights:
// reference/convert_dwpose.py (the ONNX node list, run in order).
class DwPose {
public:
  ~DwPose();
  [[nodiscard]] static std::unique_ptr<DwPose> Load(const std::filesystem::path& dir,
                                                    std::string* error);
  // 133 points of the person in box (x0, y0, x1, y1).
  [[nodiscard]] std::vector<Keypoint> Estimate(const RgbImage& image,
                                               const std::array<float, 4>& box,
                                               DwPoseTaps* taps = nullptr) const;

private:
  DwPose();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::ccip

#endif  // GUFO_MODELS_CCIP_POSE_HPP_
