// POST /v1/vision/detect: imgutils detectors natively (see vision.hpp).
// Written with AI assistance (Claude Code).
#include "src/cli/serve/router/vision.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "ccip/ccip.hpp"
#include "ccip/pose.hpp"
#include "src/core/base64.hpp"
#include "src/core/image.hpp"

namespace gufo::router::vision {
namespace {

using server::HttpRequest;
using server::HttpResponse;

constexpr std::size_t kAblageBytesLimit = static_cast<std::size_t>(20) << 20;

HttpResponse JsonResponse(int status, std::string reason, std::string body) {
  HttpResponse response;
  response.status = status;
  response.reason = std::move(reason);
  response.body = std::move(body);
  response.headers = {{"Content-Type", "application/json"}};
  return response;
}

HttpResponse Error(int status, std::string reason, std::string message, std::string code) {
  json::Value detail = json::Value::object();
  detail["message"] = std::move(message);
  detail["type"] = status >= 500 ? "server_error" : "invalid_request_error";
  detail["code"] = std::move(code);
  json::Value body = json::Value::object();
  body["error"] = std::move(detail);
  return JsonResponse(status, std::move(reason), body.dump());
}

bool EqualsIgnoreCase(std::string_view a, std::string_view b) {
  if (a.size() != b.size())
    return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(a[i])) !=
        std::tolower(static_cast<unsigned char>(b[i])))
      return false;
  }
  return true;
}

bool IsAblagePath(std::string_view value) {
  return value.starts_with("/v1/jobs/") && value.find("/files/") != std::string_view::npos;
}

/// A threshold in (0, 1]; throws for anything else.
float Threshold(const json::Value& value, const std::string& what) {
  if (!value.is_number())
    throw std::invalid_argument(what + " must be a number");
  const double v = value.as_double();
  if (!(v > 0.0 && v <= 1.0))
    throw std::invalid_argument(what + " must be in (0, 1]");
  return static_cast<float>(v);
}

/// The image type of encoded bytes for a data URL (PNG, JPEG, WebP).
std::string SniffImageType(const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF)
    return "image/jpeg";
  if (bytes.size() >= 12 && bytes[0] == 'R' && bytes[1] == 'I' && bytes[2] == 'F' &&
      bytes[3] == 'F' && bytes[8] == 'W' && bytes[9] == 'E' && bytes[10] == 'B' && bytes[11] == 'P')
    return "image/webp";
  return "image/png";
}

/// The request body for the partner: every member as sent, the image
/// inline as a data URL (the partner cannot read this machine's Ablage).
std::string RelayBody(const json::Value& body, const std::vector<std::uint8_t>& bytes) {
  json::Value out = json::Value::object();
  for (const auto& [key, value] : body.members()) {
    if (key != "image" && key != "bild")
      out[key] = value;
  }
  out["image"] = "data:" + SniffImageType(bytes) + ";base64," +
                 core::EncodeBase64(std::string_view(
                     reinterpret_cast<const char*>(bytes.data()), bytes.size()));
  return out.dump();
}

/// Bytes of an Ablage path, read through the gate with the caller's token.
/// A refusal of the gate or the job store is returned in *failure.
std::vector<std::uint8_t> ReadAblage(const Options& options, const HttpRequest& outer,
                                     const std::string& path,
                                     std::optional<HttpResponse>* failure) {
  if (!options.serve)
    throw std::invalid_argument("Ablage paths are not available on this router");
  HttpRequest inner;
  inner.method = "GET";
  inner.path = path;
  inner.client_id = outer.client_id;
  inner.is_cancelled = outer.is_cancelled;
  inner.request_id = outer.request_id;
  for (const auto& [name, value] : outer.headers) {
    if (EqualsIgnoreCase(name, "Authorization"))
      inner.headers.emplace_back(name, value);
  }
  inner.headers.emplace_back("Accept", "*/*");
  HttpResponse response = options.serve(inner);
  std::string body;
  bool truncated = false;
  if (response.streaming_body) {
    response.streaming_body([&body, &truncated](std::string_view chunk) {
      if (body.size() + chunk.size() > kAblageBytesLimit) {
        truncated = true;
        return false;
      }
      body.append(chunk);
      return true;
    });
  } else {
    body = std::move(response.body);
  }
  if (response.status >= 400) {
    // The gate's or the job store's answer (404 for another user's job).
    HttpResponse relay;
    relay.status = response.status;
    relay.reason = response.reason;
    relay.body = std::move(body);
    for (const auto& [name, value] : response.headers) {
      if (EqualsIgnoreCase(name, "Content-Type"))
        relay.headers.emplace_back(name, value);
    }
    *failure = std::move(relay);
    return {};
  }
  if (truncated || body.size() > kAblageBytesLimit)
    throw std::invalid_argument("larger than 20 MB");
  return std::vector<std::uint8_t>(body.begin(), body.end());
}

}  // namespace

const std::vector<TaskSpec>& DetectorTasks() {
  static const std::vector<TaskSpec> tasks = {
      {"person", "deepghs/anime_person_detection", "person_detect_v1.1_m", 0.3f, 0.5f},
      {"head", "deepghs/anime_head_detection", "head_detect_v2.0_s", 0.4f, 0.7f},
      {"face", "deepghs/anime_face_detection", "face_detect_v1.4_s", 0.25f, 0.7f},
      {"hand", "deepghs/anime_hand_detection", "hand_detect_v1.0_s", 0.35f, 0.7f},
  };
  return tasks;
}

const std::vector<std::string>& PoseTasks() {
  static const std::vector<std::string> tasks = {"pose", "pose_wholebody"};
  return tasks;
}

const std::vector<std::string>& PlannedTasks() {
  static const std::vector<std::string> tasks = {"ocr", "wd14_tags"};
  return tasks;
}

Request ParseRequest(const json::Value& body) {
  if (!body.is_object())
    throw std::invalid_argument("body must be a JSON object");
  Request request;
  const json::Value* image = body.find("image");
  if (image == nullptr || image->is_null())
    image = body.find("bild");
  if (image == nullptr || !image->is_string() || image->str().empty())
    throw std::invalid_argument(
        "image must be a data URL, base64, an https URL or an Ablage path "
        "/v1/jobs/<id>/files/<name>");
  request.image = image->str();
  const json::Value* tasks = body.find("tasks");
  if (tasks == nullptr || tasks->is_null())
    tasks = body.find("aufgaben");
  std::vector<std::string> names;
  if (tasks == nullptr || tasks->is_null()) {
    for (const TaskSpec& spec : DetectorTasks())
      names.push_back(spec.task);
  } else {
    if (!tasks->is_array() || tasks->empty())
      throw std::invalid_argument("tasks must be a non-empty array of task names");
    for (const json::Value& item : tasks->items()) {
      if (!item.is_string())
        throw std::invalid_argument("tasks must be strings");
      if (std::find(names.begin(), names.end(), item.str()) == names.end())
        names.push_back(item.str());
    }
  }
  // Unknown names first (400), then planned ones (501).
  for (const std::string& name : names) {
    const auto& known = DetectorTasks();
    const auto& planned = PlannedTasks();
    const auto& poses = PoseTasks();
    if (std::none_of(known.begin(), known.end(),
                     [&](const TaskSpec& spec) { return spec.task == name; }) &&
        std::find(planned.begin(), planned.end(), name) == planned.end() &&
        std::find(poses.begin(), poses.end(), name) == poses.end())
      throw std::invalid_argument("unknown task " + name +
                                  " (person, head, face, hand, pose, pose_wholebody, ocr, wd14_tags)");
  }
  for (const std::string& name : names) {
    const auto& planned = PlannedTasks();
    if (std::find(planned.begin(), planned.end(), name) != planned.end())
      throw std::domain_error("task " + name + " is not available in this build");
  }
  // Answer order: the table order, whatever order the caller used.
  for (const TaskSpec& spec : DetectorTasks()) {
    if (std::find(names.begin(), names.end(), spec.task) != names.end())
      request.tasks.push_back(spec);
  }
  request.pose = std::find(names.begin(), names.end(), "pose") != names.end();
  request.pose_wholebody = std::find(names.begin(), names.end(), "pose_wholebody") != names.end();
  if (const json::Value* po = body.find("pose_options"); po != nullptr && !po->is_null()) {
    if (!po->is_object())
      throw std::invalid_argument("pose_options must be an object");
    for (const auto& [key, value] : po->members()) {
      if (key == "min_score")
        request.pose_min_score = Threshold(value, "pose_options.min_score");
      else if (key == "max_persons" && value.is_number() && value.as_double() >= 1 &&
               value.as_double() <= 32 && value.as_double() == std::floor(value.as_double()))
        request.pose_max_persons = static_cast<int>(value.as_double());
      else
        throw std::invalid_argument("pose_options: min_score in (0, 1], max_persons 1-32");
    }
  }
  if (const json::Value* dark = body.find("brighten_dark"); dark != nullptr && !dark->is_null()) {
    if (dark->is_bool()) {
      request.brighten_dark = dark->as_bool();
    } else if (dark->is_object()) {
      request.brighten_dark = true;
      for (const auto& [key, value] : dark->members()) {
        if (!value.is_number())
          throw std::invalid_argument("brighten_dark." + key + " must be a number");
        if (key == "below" && value.as_double() >= 0.0 && value.as_double() <= 256.0)
          request.dark_below = value.as_double();
        else if (key == "gamma" && value.as_double() > 0.0 && value.as_double() <= 4.0)
          request.dark_gamma = value.as_double();
        else
          throw std::invalid_argument("brighten_dark: below in [0, 256], gamma in (0, 4]");
      }
    } else {
      throw std::invalid_argument("brighten_dark must be true, false or {below, gamma}");
    }
  }
  if (const json::Value* thresholds = body.find("thresholds");
      thresholds != nullptr && !thresholds->is_null()) {
    if (!thresholds->is_object())
      throw std::invalid_argument("thresholds must be an object by task");
    for (const auto& [key_task, values] : thresholds->members()) {
      const std::string& task = key_task;
      auto spec = std::find_if(request.tasks.begin(), request.tasks.end(),
                               [&task](const TaskSpec& s) { return s.task == task; });
      if (spec == request.tasks.end())
        throw std::invalid_argument("thresholds." + task + ": task not requested");
      if (!values.is_object())
        throw std::invalid_argument("thresholds." + task + " must be an object");
      for (const auto& [key, value] : values.members()) {
        if (key == "conf")
          spec->conf = Threshold(value, "thresholds." + task + ".conf");
        else if (key == "iou")
          spec->iou = Threshold(value, "thresholds." + task + ".iou");
        else
          throw std::invalid_argument("thresholds." + task + "." + key + ": use conf or iou");
      }
    }
  }
  return request;
}

/// Loaded detectors (by model stem) and the CPU queue, shared by requests.
struct VisionState {
  std::mutex load_mutex;
  std::map<std::string, std::unique_ptr<models::ccip::YoloDetector>> detectors;
  std::unique_ptr<models::ccip::RtDetr> rtdetr;
  std::unique_ptr<models::ccip::VitPose> vitpose;
  std::unique_ptr<models::ccip::DwPose> dwpose;

  /// A pose network (stem = file stem), loaded on first use; false with *error set.
  template <typename Net>
  bool GetPose(const Options& options, std::unique_ptr<Net>* slot, const char* stem,
               const char* converter, std::string* error) {
    std::lock_guard<std::mutex> lock(load_mutex);
    if (*slot != nullptr)
      return true;
    if (options.model_dir.empty()) {
      *error = "image recognition is not configured: start the router with --vision-dir";
      return false;
    }
    for (const char* ext : {".safetensors", ".json"}) {
      const std::filesystem::path path = options.model_dir / (std::string(stem) + ext);
      std::error_code ec;
      if (!std::filesystem::is_regular_file(path, ec)) {
        *error = "model file missing: " + path.string() + " (convert with extensions-ours/ccip/reference/" +
                 converter + ")";
        return false;
      }
    }
    std::string load_error;
    auto net = Net::Load(options.model_dir, &load_error);
    if (net == nullptr) {
      *error = std::string(stem) + ": " + load_error;
      return false;
    }
    *slot = std::move(net);
    return true;
  }

  /// The detector of `spec`, loaded on first use; null with *error set.
  const models::ccip::YoloDetector* Get(const Options& options, const TaskSpec& spec,
                                        std::string* error) {
    std::lock_guard<std::mutex> lock(load_mutex);
    if (auto it = detectors.find(spec.model); it != detectors.end())
      return it->second.get();
    if (options.model_dir.empty()) {
      *error = "image recognition is not configured: start the router with --vision-dir";
      return nullptr;
    }
    for (const char* ext : {".safetensors", ".json"}) {
      const std::filesystem::path path = options.model_dir / (spec.model + ext);
      std::error_code ec;
      if (!std::filesystem::is_regular_file(path, ec)) {
        *error = "model file missing: " + path.string() + " (convert " + spec.repo + " " +
                 spec.model + "/model.onnx with extensions-ours/ccip/reference/convert_weights.py)";
        return nullptr;
      }
    }
    std::string load_error;
    auto net = models::ccip::YoloDetector::Load(options.model_dir, spec.model, &load_error);
    if (net == nullptr) {
      *error = spec.model + ": " + load_error;
      return nullptr;
    }
    const auto* loaded = net.get();
    detectors.emplace(spec.model, std::move(net));
    return loaded;
  }
};

VisionService::VisionService(Options options)
    : options_(std::move(options)), state_(std::make_shared<VisionState>()) {
  if (options_.queue == nullptr)
    options_.queue = std::make_shared<cpu::CpuQueue>(options_.max_waiting);
}

VisionService::~VisionService() = default;

std::size_t VisionService::Waiting() const { return options_.queue->Waiting(); }

std::optional<HttpResponse> VisionService::Handle(const HttpRequest& request) const {
  if (request.path != "/v1/vision/detect")
    return std::nullopt;
  if (request.method != "POST")
    return Error(405, "Method Not Allowed", "use POST", "method_not_allowed");
  Request parsed;
  json::Value body;
  try {
    body = json::parse(request.body);
    parsed = ParseRequest(body);
  } catch (const std::domain_error& error) {
    return Error(501, "Not Implemented", error.what(), "task_not_available");
  } catch (const std::exception& error) {
    return Error(400, "Bad Request", error.what(), "invalid_request");
  }
  // Models: a missing file is 503 unless the pool partner runs it.
  std::vector<const models::ccip::YoloDetector*> nets;
  std::string missing;
  for (const TaskSpec& spec : parsed.tasks) {
    const auto* net = state_->Get(options_, spec, &missing);
    if (net == nullptr)
      break;
    nets.push_back(net);
  }
  if (missing.empty() && (parsed.pose || parsed.pose_wholebody))
    state_->GetPose(options_, &state_->rtdetr, "rtdetr_r50vd_coco_o365", "convert_pose.py rtdetr", &missing);
  if (missing.empty() && parsed.pose)
    state_->GetPose(options_, &state_->vitpose, "vitpose_base_simple", "convert_pose.py vitpose", &missing);
  if (missing.empty() && parsed.pose_wholebody)
    state_->GetPose(options_, &state_->dwpose, "dwpose_wholebody", "convert_dwpose.py", &missing);
  // Input before the queue: reading an Ablage path goes through the gate
  // and must not hold the CPU turn.
  std::vector<std::uint8_t> bytes;
  try {
    if (IsAblagePath(parsed.image)) {
      std::optional<HttpResponse> failure;
      bytes = ReadAblage(options_, request, parsed.image, &failure);
      if (failure.has_value())
        return std::move(*failure);
    } else {
      const std::string url =
          parsed.image.starts_with("data:") || parsed.image.starts_with("https://")
              ? parsed.image
              : "data:image/png;base64," + parsed.image;
      core::ImageReadBudget budget;
      bytes = core::ReadImageUrl(url, budget);
    }
  } catch (const std::exception& error) {
    return Error(400, "Bad Request", std::string("image: ") + error.what(), "invalid_request");
  }
  // Pool (cpu_pool.hpp): the partner when this machine is busy or lacks
  // the models.
  cpu::CpuQueue& queue = *options_.queue;
  const cpu::Route route = cpu::Decide(request, static_cast<bool>(options_.relay),
                                       !queue.Idle(), !missing.empty());
  if (route == cpu::Route::kTryPartner || route == cpu::Route::kPartnerOnly) {
    const bool force = route == cpu::Route::kPartnerOnly;
    std::optional<HttpResponse> answer;
    if (options_.relay)
      answer = options_.relay(cpu::RelayRequest(request, RelayBody(body, bytes), force));
    if (answer.has_value() && (force || cpu::PartnerAnswered(*answer))) {
      answer->headers.emplace_back(cpu::kRanOnHeader, "partner");
      return std::move(*answer);
    }
    if (force)
      return Error(503, "Service Unavailable", "no pool partner can run it",
                   "remote_unavailable");
  }
  if (!missing.empty())
    return Error(503, "Service Unavailable", missing, "vision_unavailable");
  if (route == cpu::Route::kLocalIfIdle && !queue.Idle())
    return cpu::Busy("image recognition");
  models::ccip::RgbImage image;
  try {
    // RGB8 with the alpha channel dropped (imgutils puts RGBA on white).
    core::Image rgb =
        core::DecodeImage(std::span<const std::uint8_t>(bytes.data(), bytes.size()));
    image.width = static_cast<int>(rgb.width);
    image.height = static_cast<int>(rgb.height);
    image.rgb = std::move(rgb.pixels);
  } catch (const std::exception& error) {
    return Error(400, "Bad Request", std::string("image: ") + error.what(), "invalid_request");
  }
  bytes.clear();
  bytes.shrink_to_fit();
  const double mean_luma = models::ccip::MeanLuma(image);
  const bool brightened = parsed.brighten_dark && mean_luma < parsed.dark_below;
  // The detectors see the brightened copy; later tasks (OCR, tags) keep
  // the original, as in the pipeline's erkennung.py.
  models::ccip::RgbImage detect_image;
  if (brightened)
    detect_image = models::ccip::ApplyGamma(image, parsed.dark_gamma);
  const models::ccip::RgbImage& detector_input = brightened ? detect_image : image;

  const auto queued = std::chrono::steady_clock::now();
  switch (queue.Enter(request.is_cancelled)) {
    case cpu::CpuQueue::Result::kOk:
      break;
    case cpu::CpuQueue::Result::kFull:
      return cpu::Busy("image recognition");
    case cpu::CpuQueue::Result::kCancelled:
      return Error(499, "Client Closed Request", "cancelled while waiting", "cancelled");
  }
  const cpu::CpuQueue::Turn turn(queue);
  const auto started = std::chrono::steady_clock::now();
  json::Value detections = json::Value::object();
  json::Value used = json::Value::object();
  json::Value out_poses;
  try {
    models::ccip::SetCpuThreads(options_.threads);
    for (std::size_t i = 0; i < parsed.tasks.size(); ++i) {
      const TaskSpec& spec = parsed.tasks[i];
      const auto& labels = nets[i]->Labels();
      json::Value list = json::Value::array();
      for (const models::ccip::Detection& d : nets[i]->Detect(detector_input, spec.conf, spec.iou)) {
        json::Value box = json::Value::array();
        for (int v : d.box)
          box.push_back(v);
        json::Value entry = json::Value::object();
        entry["box"] = std::move(box);
        entry["label"] = labels[static_cast<std::size_t>(d.label)];
        entry["score"] = static_cast<double>(d.score);
        list.push_back(std::move(entry));
      }
      detections[spec.task] = std::move(list);
      json::Value model = json::Value::object();
      model["repo"] = spec.repo;
      model["model"] = spec.model;
      model["conf"] = std::round(static_cast<double>(spec.conf) * 1e6) / 1e6;
      model["iou"] = std::round(static_cast<double>(spec.iou) * 1e6) / 1e6;
      used[spec.task] = std::move(model);
    }
    if (parsed.pose || parsed.pose_wholebody) {
      // posen_erkennung.erkenne on the original image (not the brightened copy).
      std::vector<models::ccip::ObjectBox> persons;
      for (const auto& o : state_->rtdetr->Detect(image, parsed.pose_min_score)) {
        if (o.label == 0)
          persons.push_back(o);
      }
      std::stable_sort(persons.begin(), persons.end(),
                       [](const auto& a, const auto& b) { return a.score > b.score; });
      if (persons.size() > static_cast<std::size_t>(parsed.pose_max_persons))
        persons.resize(static_cast<std::size_t>(parsed.pose_max_persons));
      auto points = [](const std::vector<models::ccip::Keypoint>& kps) {
        json::Value list = json::Value::array();
        for (const auto& k : kps) {
          json::Value p = json::Value::array();
          p.push_back(static_cast<double>(k.x));
          p.push_back(static_cast<double>(k.y));
          p.push_back(static_cast<double>(k.score));
          list.push_back(std::move(p));
        }
        return list;
      };
      json::Value poses = json::Value::array();
      for (const auto& person : persons) {
        json::Value entry = json::Value::object();
        json::Value box = json::Value::array();
        for (float v : person.box)
          box.push_back(static_cast<double>(v));
        entry["box"] = std::move(box);
        entry["score"] = static_cast<double>(person.score);
        if (parsed.pose)
          entry["keypoints"] = points(state_->vitpose->Estimate(image, person.box));
        if (parsed.pose_wholebody)
          entry["wholebody"] = points(state_->dwpose->Estimate(image, person.box));
        poses.push_back(std::move(entry));
      }
      out_poses = std::move(poses);
      json::Value det = json::Value::object();
      det["repo"] = "PekingU/rtdetr_r50vd_coco_o365";
      det["model"] = "rtdetr_r50vd_coco_o365";
      det["min_score"] = std::round(static_cast<double>(parsed.pose_min_score) * 1e6) / 1e6;
      det["max_persons"] = parsed.pose_max_persons;
      used["pose_detector"] = std::move(det);
      if (parsed.pose) {
        json::Value m = json::Value::object();
        m["repo"] = "usyd-community/vitpose-base-simple";
        m["model"] = "vitpose_base_simple";
        json::Value names = json::Value::array();
        for (const auto& n : state_->vitpose->KeypointNames())
          names.push_back(n);
        m["keypoints"] = std::move(names);
        used["pose"] = std::move(m);
      }
      if (parsed.pose_wholebody) {
        json::Value m = json::Value::object();
        m["repo"] = "yzd-v/DWPose";
        m["model"] = "dw-ll_ucoco_384";
        m["layout"] = "COCO-WholeBody: 0-16 body, 17-22 feet, 23-90 face (68), 91-111 left hand, "
                      "112-132 right hand";
        used["pose_wholebody"] = std::move(m);
      }
    }
  } catch (const std::exception& error) {
    return Error(500, "Internal Server Error", std::string("detection failed: ") + error.what(),
                 "vision_failed");
  }
  const auto done = std::chrono::steady_clock::now();
  const auto seconds = [](std::chrono::steady_clock::time_point a,
                          std::chrono::steady_clock::time_point b) {
    return std::round(std::chrono::duration<double>(b - a).count() * 1000.0) / 1000.0;
  };
  json::Value out = json::Value::object();
  out["object"] = "vision.detection";
  out["width"] = image.width;
  out["height"] = image.height;
  out["mean_luma"] = std::round(mean_luma * 1000.0) / 1000.0;
  out["brightened"] = brightened;
  out["detections"] = std::move(detections);
  if (parsed.pose || parsed.pose_wholebody)
    out["poses"] = std::move(out_poses);
  out["models"] = std::move(used);
  out["seconds"] = seconds(started, done);
  out["queue_seconds"] = seconds(queued, started);
  return JsonResponse(200, "OK", out.dump());
}

}  // namespace gufo::router::vision
