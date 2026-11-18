#ifndef GUFO_CLI_SERVE_ROUTER_VISION_HPP_
#define GUFO_CLI_SERVE_ROUTER_VISION_HPP_

// Image recognition for the film pipeline (`POST /v1/vision/detect`): the
// detectors of dghs-imgutils (detect_person, detect_heads, detect_faces,
// detect_hands), its text recognition (imgutils.ocr.ocr) and its WD14
// tagger (get_wd14_tags) with the same default models and thresholds,
// natively on the CPU (gufo_ccip YoloDetector, TextOcr and Wd14Tagger,
// OpenMP), without Python or ONNX at run time.
//
//   POST /v1/vision/detect
//     {"image": "<data URL | base64 | https URL | /v1/jobs/<id>/files/<name>>",
//      "tasks": ["person", "head", "face", "hand", "ocr", "wd14_tags"],
//      "thresholds": {"person": {"conf": 0.3, "iou": 0.5},
//                     "ocr": {"heat": 0.3, "box": 0.7}},       optional
//      "ocr": {"box_threshold": 0.5, "heat_threshold": 0.3,
//              "unclip_ratio": 2.0, "scale": 3}                 optional
//             or a list of up to 4 such passes,
//      "wd14_tags": {...}}                         optional, vision_wd14.hpp
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
//   "ocr" (not in the default task list) answers
//     [{"box": [x0, y0, x1, y1], "points": [[x, y] x 4], "text": "...",
//       "score": det * text, "det_score", "text_score"}, ...]
//   with models.ocr = {"repo", "model", "rec_model", "heat", "box",
//   "unclip_ratio", "scale"}, sorted by score (imgutils order). Boxes are
//   integer pixels of the input image; a pass with scale s > 1 reads a
//   Pillow-LANCZOS enlargement to s*W x s*H and answers its boxes divided
//   by s, unrounded (erkennung_schrift.py). "ocr" as a list of passes
//   answers one such list per pass (and models.ocr as a list). OCR and tags
//   always read the original, never the brightened copy.
//   "pose" / "pose_wholebody" (film pipeline posen_erkennung.erkenne):
//   RT-DETR r50vd finds the persons (label person, score > min_score 0.35,
//   at most max_persons 4 by score; "pose_options": {"min_score",
//   "max_persons"}), on the original image even with brighten_dark;
//   "pose" adds ViTPose base (17 COCO points), "pose_wholebody" DWPose
//   dw-ll_ucoco_384 (133 COCO-WholeBody points) per person:
//     "poses": [{"box": [x0, y0, x1, y1] (float), "score",
//                "keypoints": [[x, y, score] x 17],      with "pose"
//                "wholebody": [[x, y, score] x 133]}]    with "pose_wholebody"
//   in score order; the point names are in "models".
//   "pose_boxes": [[x0, y0, x1, y1], ...] (at most kMaxPoseBoxes) skips
//   RT-DETR: ViTPose / DWPose run in exactly these boxes, in this order
//   (the film pipeline's mundmessung.py keeps the first frame's box for
//   every frame); the poses then carry no "score".
//   "hand_fingers" (vision_fingers.hpp): the "hand" boxes cut out and each
//   hand checked for impossible anatomy by a vision chat model (default
//   qwen3.8-27b, "hand_fingers_options") through the router's own chat API:
//   detections.hand_fingers = [{"box", "score", "crop", "plausibel",
//   "sicher", "fingers_sichtbar", "verdeckt", "befunde", "befund",
//   "seconds", "waited_seconds"} or {..., "fehler"}]; not relayed to the pool
//   partner unless the models are missing here.
//   Unknown task or bad option: 400.
//   Models: <--vision-dir>/<model>.{safetensors,json} from
//   extensions-ours/ccip/reference/convert_weights.py, each loaded on first
//   use and kept (person m ~100 MB, the s models ~43 MB each, host RAM);
//   ocr: ch_PP-OCRv4_det.*, ch_PP-OCRv4_rec.* and ch_PP-OCRv4_rec.dict.txt
//   from extensions-ours/ccip/reference/convert_ocr.py (~16 MB);
//   503 "vision_unavailable" when a file is missing.
//   Queue and pool: cpu_pool.hpp. One request computes at a time in the
//   CPU queue shared with /v1/figuren "ccip" (FIFO, --vision-threads OpenMP
//   threads); more than --vision-queue waiting: 503 "busy" with Retry-After.
//   When this machine is busy or lacks the models, the pool partner gets
//   the request first (image inline as a data URL); X-Gufo-Ran-On: partner
//   marks its answer.
//   An Ablage path is read through the gate with the caller's own
//   Authorization or session cookie (another user's job is the gate's
//   404), at most 20 MB.
//   Limits per request (400 beyond): kMaxRequestBytes encoded over all
//   input pictures, kMaxRequestPixels decoded over all of them (decoded
//   only once the request has its CPU turn).

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/cpu_pool.hpp"
#include "src/cli/serve/router/vision_fingers.hpp"
#include "src/cli/serve/router/vision_wd14.hpp"
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

/// One "ocr" pass: imgutils.ocr.ocr keyword arguments (its defaults) and an
/// enlargement before it (erkennung_schrift.py: 3x Pillow LANCZOS, boxes
/// divided back by 3). `rec` names the text recognizer: "ch_v4" (imgutils'
/// PP-OCRv4, Chinese + Latin without umlauts) or "latin_v5" (PP-OCRv5 latin
/// mobile: German umlauts and ß, French, Polish, ...).
struct OcrPass {
  double heat_threshold{0.3};
  double box_threshold{0.7};
  double unclip_ratio{2.0};
  int scale{1};
  std::string rec{"ch_v4"};
};
inline constexpr std::size_t kMaxOcrPasses = 4;
inline constexpr int kMaxOcrScale = 4;
/// Largest OCR input, enlarged or not (3x of 1920x1080 is 18.7 megapixels).
/// imgutils.ocr has no limit (it pads to a multiple of 32 and runs DBNet on
/// the whole picture); the detector's float32 feature maps of a larger
/// input need several GB of host RAM, so gufo answers 400 instead.
inline constexpr std::int64_t kMaxOcrPixels = 24'000'000;
/// Encoded bytes of all input pictures of one request together (image and
/// wd14_tags.images; data URLs, https URLs and Ablage paths alike).
inline constexpr std::size_t kMaxRequestBytes = static_cast<std::size_t>(64) << 20;
/// Decoded pixels of all input pictures of one request together (RGB8:
/// about 192 MB). Pictures are decoded only once the request has its turn
/// in the CPU queue, so waiting requests hold encoded bytes only.
inline constexpr std::int64_t kMaxRequestPixels = 64'000'000;

struct Request {
  std::string image;
  std::vector<TaskSpec> tasks;  // with the request's thresholds applied
  /// "brighten_dark": detectors look at a gamma-brightened copy when the
  /// image's mean luma is below `dark_below` (film pipeline erkennung.py:
  /// 70 and gamma 0.5). Off by default (plain imgutils).
  bool brighten_dark{false};
  double dark_below{70.0};
  double dark_gamma{0.5};  // models::ccip::MeanLuma / ApplyGamma
  /// "wd14_tags" and its options (vision_wd14.hpp).
  Wd14Request wd14;
  /// "ocr" passes (imgutils.ocr.ocr), empty when not requested. `ocr_list`:
  /// the caller sent "ocr" as a list of passes, the answer has one list of
  /// texts per pass (else the one pass's list).
  std::vector<OcrPass> ocr;
  bool ocr_list{false};
  /// "pose" (ViTPose) and/or "pose_wholebody" (DWPose) on RT-DETR persons.
  bool pose{false};
  bool pose_wholebody{false};
  float pose_min_score{0.35f};
  int pose_max_persons{4};
  /// "pose_boxes": the persons' boxes given by the caller (RT-DETR skipped).
  std::vector<std::array<float, 4>> pose_boxes;
  /// "hand_fingers" and "hand_fingers_options" (vision_fingers.hpp); the
  /// "hand" detector then runs too.
  HandFingersRequest hand_fingers;
};
/// Most "pose_boxes" per request.
inline constexpr std::size_t kMaxPoseBoxes = 16;

/// The pose tasks ("pose", "pose_wholebody").
const std::vector<std::string>& PoseTasks();

/// Parses and checks a request body; throws std::invalid_argument with a
/// message for the 400 answer. Tasks default to all four detectors (ocr and
/// wd14_tags only on request); duplicates are dropped.
Request ParseRequest(const json::Value& body);

struct Options {
  std::filesystem::path model_dir;
  int threads = 8;
  /// Requests that may wait behind the running one (only for the own queue
  /// made when `queue` is null).
  std::size_t max_waiting = 4;
  /// The CPU queue shared with /v1/figuren "ccip" (cpu_pool.hpp); null: an
  /// own queue.
  std::shared_ptr<cpu::CpuQueue> queue;
  /// The pool partner (cpu_pool.hpp); null: never relayed.
  cpu::Relay relay;
  /// One request through the gate and the router (for Ablage paths and the
  /// "hand_fingers" chats), as mcp::Options::serve. Null: Ablage paths are
  /// refused (400), every hand_fingers hand answers "fehler".
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
