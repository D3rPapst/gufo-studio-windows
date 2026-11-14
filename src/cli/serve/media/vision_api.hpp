// `gufo serve vision`: depth, open-vocabulary boxes and masks on the GPU
//   POST /v1/vision/depth    Depth Anything V2 large | small, Distill-Any-Depth
//   POST /v1/vision/find     Grounding DINO base (or LLMDet): labels -> boxes
//   POST /v1/vision/segment  SAM ViT-H: boxes / points -> masks, or labels
//                            (find, then SAM on the boxes, as the film
//                            pipeline's hf.py segment())
// The router's own POST /v1/vision/detect (imgutils detectors, CPU) is a
// different service. Requests run one at a time in arrival order; models
// load on first use and stay. Fields and answers: docs/models/vision-dsg.md.
// Written with AI assistance (Claude Code).
#ifndef GUFO_SERVER_MEDIA_VISION_API_HPP_
#define GUFO_SERVER_MEDIA_VISION_API_HPP_

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/media/media_service.hpp"
#include "src/core/image.hpp"

namespace gufo::server {

struct VisionServeOptions {
  /// --depth-model NAME=DIR (repeatable; NAME large | small | base).
  std::vector<std::pair<std::string, std::filesystem::path>> depth;
  /// --find-model [NAME=]DIR (repeatable; Grounding DINO or MM-Grounding-DINO
  /// / LLMDet; without NAME: grounding-dino).
  std::vector<std::pair<std::string, std::filesystem::path>> find;
  std::filesystem::path segment;  // --segment-model DIR (SAM)
};

struct VisionBox {
  std::array<float, 4> box{};  // x0, y0, x1, y1 in image pixels
  float score{0};
  std::string label;
};

struct VisionFindRequest {
  std::string detector;  // a find model name ("" = the first)
  std::vector<std::string> labels;  // lower-cased, without trailing dots
  float box_threshold{0.3F};
  float text_threshold{0.25F};
};

struct VisionSegmentPrompt {
  std::vector<std::array<float, 4>> boxes;
  std::vector<std::array<float, 2>> points;
  std::vector<int> point_labels;  // 1 foreground, 0 background
  bool multimask{false};
};

struct VisionMask {
  int width{0}, height{0};
  std::vector<std::uint8_t> mask;  // 0 / 1, [height, width]
  float score{0};                  // predicted IoU
};

/// The model side, filled by LoadMediaService on HIP builds (tests inject
/// fakes). Each runner may load its model on first use.
struct VisionBackends {
  std::vector<std::string> depth_sizes;  // first = default
  std::vector<std::string> find_names;   // first = default
  std::function<bool(const std::string& size, const core::Image& image,
                     std::vector<float>* depth, std::string* error)>
      depth;
  std::function<bool(const core::Image& image, const VisionFindRequest& request,
                     std::vector<VisionBox>* boxes, std::string* error)>
      find;
  std::function<bool(const core::Image& image, const VisionSegmentPrompt& prompt,
                     std::vector<VisionMask>* masks, std::string* error)>
      segment;
  /// Per model name: repo, license, revision (for the answers' `models`).
  std::function<json::Value(const std::string& task, const std::string& size)>
      describe;
};

/// The GPU models behind `gufo serve vision` (vision_backends.cpp); false
/// with *error on builds without HIP. Models load on first use.
bool BuildVisionBackends(const VisionServeOptions& options,
                         VisionBackends* backends, std::string* error);

/// hf.py depth(): 2nd / 98th percentile over every 97th value of all maps
/// (numpy percentile, linear).
void DepthPercentiles(const std::vector<const std::vector<float>*>& maps,
                      double* lo, double* hi);
/// Grey PNG of a map scaled by (v - lo) / (hi - lo), clipped to [0, 1]:
/// 8 bit truncated like hf.py (`astype(uint8)` of x * 255), or 16 bit rounded.
std::vector<std::uint8_t> EncodeGreyPng(const std::vector<float>& map, int width,
                                        int height, double lo, double hi,
                                        int bits);

class VisionModelService final : public MediaService {
public:
  VisionModelService(std::string model_id, VisionBackends backends);

  [[nodiscard]] std::string_view capability() const override { return "vision"; }
  [[nodiscard]] std::string_view owned_by() const override {
    return "operator-supplied-vision";
  }
  [[nodiscard]] bool ready() const override { return true; }
  [[nodiscard]] std::string model_id() const override { return model_id_; }
  void Describe(json::Value* entry, bool ready) const override;
  [[nodiscard]] bool HandlesPath(std::string_view path) const override;
  [[nodiscard]] std::string_view unavailable_message() const override {
    return "vision models are not ready";
  }
  [[nodiscard]] std::string_view unavailable_code() const override {
    return "vision_unavailable";
  }
  HttpResponse Handle(const HttpRequest& request) override;

private:
  HttpResponse Depth(const json::Value& body);
  HttpResponse Find(const json::Value& body);
  HttpResponse Segment(const json::Value& body);

  std::string model_id_;
  VisionBackends backends_;
  std::mutex run_mutex_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_MEDIA_VISION_API_HPP_
