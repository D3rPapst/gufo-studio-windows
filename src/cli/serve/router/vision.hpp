#ifndef GUFO_CLI_SERVE_ROUTER_VISION_HPP_
#define GUFO_CLI_SERVE_ROUTER_VISION_HPP_

// Image recognition for the film pipeline (`POST /v1/vision/detect`): the
// detectors of dghs-imgutils (detect_person, detect_heads, detect_faces,
// detect_hands) with the same default models and thresholds, natively on the
// CPU (gufo_ccip YoloDetector, OpenMP), without Python or ONNX at run time.
//
//   POST /v1/vision/detect
//     {"image": "<data URL | base64 | https URL | /v1/jobs/<id>/files/<name>>",
//      "tasks": ["person", "head", "face", "hand"],
//      "thresholds": {"person": {"conf": 0.3, "iou": 0.5}}}   optional
//     ("bild" and "aufgaben" are accepted for image and tasks.)
//     "brighten_dark": true | {"below": 70, "gamma": 0.5}   optional: when
//     the mean luma (Pillow "L") is below `below`, the detectors run on a
//     copy brightened with v -> round(255 * (v/255)^gamma), as the film
//     pipeline does for dark keyframes; "brightened" in the answer says so.
//   -> {"object": "vision.detection", "width": W, "height": H,
//       "mean_luma": L, "brightened": bool,
//       "detections": {"person": [{"box": [x0, y0, x1, y1], "label":
//          "person", "score": 0.93}], ...},          one list per task
//       (score: the float32 value as imgutils returns it, unrounded)
//       "models": {"person": {"repo", "model", "conf", "iou"}, ...},
//       "seconds": compute time, "queue_seconds": wait before it}
//   Boxes are integer pixels of the input image, sorted by score (imgutils
//   order). "ocr" and "wd14_tags" are known tasks that this build does not
//   run yet: 501 "task_not_available". Unknown task or bad threshold: 400.
//   Models: <--vision-dir>/<model>.{safetensors,json} from
//   extensions-ours/ccip/reference/convert_weights.py, each loaded on first
//   use and kept (person m ~100 MB, the s models ~43 MB each, host RAM);
//   503 "vision_unavailable" when a file is missing.
//   Queue: one request computes at a time (FIFO, --vision-threads OpenMP
//   threads), so parallel callers wait instead of fighting for the cores;
//   more than --vision-queue waiting requests: 503 "busy" with Retry-After.
//   A caller that disconnects while waiting leaves the queue.
//   An Ablage path is read through the gate with the caller's own
//   Authorization (another user's job is the gate's 404), at most 20 MB.

#include <cstddef>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/core/json.hpp"

namespace gufo::router::vision {

/// One imgutils detector default (imgutils 0.19 detect/*.py).
struct TaskSpec {
  std::string task;   // "person", "head", "face", "hand"
  std::string repo;   // Hugging Face repository of the ONNX export
  std::string model;  // model folder there = file stem of the conversion
  float conf{0.25f};
  float iou{0.7f};
};

/// The four detector tasks in answer order.
const std::vector<TaskSpec>& DetectorTasks();
/// Tasks that are part of the API but not built yet (501).
const std::vector<std::string>& PlannedTasks();

struct Request {
  std::string image;
  std::vector<TaskSpec> tasks;  // with the request's thresholds applied
  /// "brighten_dark": detectors look at a gamma-brightened copy when the
  /// image's mean luma is below `dark_below` (film pipeline erkennung.py:
  /// 70 and gamma 0.5). Off by default (plain imgutils).
  bool brighten_dark{false};
  double dark_below{70.0};
  double dark_gamma{0.5};  // models::ccip::MeanLuma / ApplyGamma
};

/// Parses and checks a request body; throws std::invalid_argument with a
/// message for the 400 answer, and std::domain_error naming a planned task
/// (501). Tasks default to all four detectors; duplicates are dropped.
Request ParseRequest(const json::Value& body);

struct Options {
  std::filesystem::path model_dir;
  int threads = 8;
  /// Requests that may wait behind the running one.
  std::size_t max_waiting = 16;
  /// One request through the gate and the router (for Ablage paths), as
  /// mcp::Options::serve. Null: Ablage paths are refused (400).
  std::function<server::HttpResponse(const server::HttpRequest&)> serve;
};

struct VisionState;

class VisionService {
public:
  explicit VisionService(Options options);
  ~VisionService();
  /// Answers POST /v1/vision/detect, nullopt for other paths.
  std::optional<server::HttpResponse> Handle(const server::HttpRequest& request) const;
  /// Requests waiting for the CPU (status pages).
  [[nodiscard]] std::size_t Waiting() const;

private:
  Options options_;
  std::shared_ptr<VisionState> state_;
};

}  // namespace gufo::router::vision

#endif  // GUFO_CLI_SERVE_ROUTER_VISION_HPP_
