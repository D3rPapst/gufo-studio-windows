// POST /v1/vision/detect: imgutils detectors, OCR and WD14 tags natively
// (see vision.hpp).
// Written with AI assistance (Claude Code).
#include "src/cli/serve/router/vision.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <map>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "ccip/ccip.hpp"
#include "ccip/ocr.hpp"
#include "ccip/wd14.hpp"
#include "ccip/pose.hpp"
#include "src/core/base64.hpp"
#include "src/core/image.hpp"

namespace gufo::router::vision {
namespace {

using server::HttpRequest;
using server::HttpResponse;

constexpr std::size_t kAblageBytesLimit = static_cast<std::size_t>(20) << 20;

// imgutils.ocr defaults (ocr/entry.py): PP-OCRv4 mobile from deepghs/paddleocr.
constexpr const char* kOcrTask = "ocr";
constexpr const char* kOcrRepo = "deepghs/paddleocr";
constexpr const char* kOcrDet = "ch_PP-OCRv4_det";
constexpr const char* kOcrRec = "ch_PP-OCRv4_rec";

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

/// The same check for the ocr thresholds, kept as double (imgutils compares
/// the box score against the Python float).
double OcrThreshold(const json::Value& value, const std::string& what) {
  if (!value.is_number())
    throw std::invalid_argument(what + " must be a number");
  const double v = value.as_double();
  if (!(v > 0.0 && v <= 1.0))
    throw std::invalid_argument(what + " must be in (0, 1]");
  return v;
}

/// The "ocr" member: one object of options or a list of passes (at most
/// kMaxOcrPasses). Without it one pass with `base`. Options without the
/// task requested are refused like wd14_tags'.
std::vector<OcrPass> ParseOcr(const json::Value& body, bool requested, const OcrPass& base,
                              bool* list) {
  *list = false;
  const json::Value* options = body.find(kOcrTask);
  const bool given = options != nullptr && !options->is_null();
  if (!requested) {
    if (given)
      throw std::invalid_argument("ocr options given, but the task is not requested");
    return {};
  }
  if (!given)
    return {base};
  std::vector<OcrPass> passes;
  const auto parse_one = [&](const json::Value& value, const std::string& where) {
    if (!value.is_object())
      throw std::invalid_argument(where + " must be an object of options");
    OcrPass pass = base;
    for (const auto& [key, option] : value.members()) {
      const std::string what = where + "." + key;
      if (key == "heat_threshold") {
        pass.heat_threshold = OcrThreshold(option, what);
      } else if (key == "box_threshold") {
        pass.box_threshold = OcrThreshold(option, what);
      } else if (key == "unclip_ratio") {
        if (!option.is_number() || !(option.as_double() > 0.0 && option.as_double() <= 10.0))
          throw std::invalid_argument(what + " must be a number in (0, 10]");
        pass.unclip_ratio = option.as_double();
      } else if (key == "scale") {
        const double s = option.is_number() ? option.as_double() : 0.0;
        if (!(s >= 1.0 && s <= kMaxOcrScale) || s != std::floor(s))
          throw std::invalid_argument(what + " must be a whole number from 1 to " +
                                      std::to_string(kMaxOcrScale));
        pass.scale = static_cast<int>(s);
      } else {
        throw std::invalid_argument(what +
                                    ": use heat_threshold, box_threshold, unclip_ratio or scale");
      }
    }
    passes.push_back(pass);
  };
  if (options->is_array()) {
    if (options->empty() || options->size() > kMaxOcrPasses)
      throw std::invalid_argument("ocr must list 1 to " + std::to_string(kMaxOcrPasses) +
                                  " passes");
    *list = true;
    for (std::size_t i = 0; i < options->size(); ++i)
      parse_one(options->items()[i], "ocr[" + std::to_string(i) + "]");
  } else {
    parse_one(*options, "ocr");
  }
  return passes;
}

/// The answer list of one ocr pass. At scale 1 integer pixels as imgutils
/// returns them; an enlarged pass divides its boxes and corners back by
/// the scale without rounding (erkennung_schrift.py).
json::Value OcrEntries(const std::vector<models::ccip::OcrText>& texts, int scale) {
  const auto coord = [scale](int v) -> json::Value {
    if (scale == 1)
      return json::Value(v);
    return json::Value(static_cast<double>(v) / scale);
  };
  json::Value list = json::Value::array();
  for (const models::ccip::OcrText& t : texts) {
    json::Value box = json::Value::array();
    for (int v : t.box)
      box.push_back(coord(v));
    json::Value points = json::Value::array();
    for (const auto& p : t.points) {
      json::Value xy = json::Value::array();
      xy.push_back(coord(p[0]));
      xy.push_back(coord(p[1]));
      points.push_back(std::move(xy));
    }
    json::Value entry = json::Value::object();
    entry["box"] = std::move(box);
    entry["points"] = std::move(points);
    entry["text"] = t.text;
    entry["score"] = t.score;
    entry["det_score"] = t.det_score;
    entry["text_score"] = t.text_score;
    list.push_back(std::move(entry));
  }
  return list;
}

/// models.ocr of one pass.
json::Value OcrModelInfo(const OcrPass& pass) {
  const auto round6 = [](double v) { return std::round(v * 1e6) / 1e6; };
  json::Value model = json::Value::object();
  model["repo"] = kOcrRepo;
  model["model"] = kOcrDet;
  model["rec_model"] = kOcrRec;
  model["heat"] = round6(pass.heat_threshold);
  model["box"] = round6(pass.box_threshold);
  model["unclip_ratio"] = round6(pass.unclip_ratio);
  model["scale"] = pass.scale;
  return model;
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

std::string DataUrl(const std::vector<std::uint8_t>& bytes) {
  return "data:" + SniffImageType(bytes) + ";base64," +
         core::EncodeBase64(
             std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}

/// The request body for the partner: every member as sent, the image (and
/// the further wd14_tags pictures) inline as data URLs (the partner cannot
/// read this machine's Ablage).
std::string RelayBody(const json::Value& body, const std::vector<std::uint8_t>& bytes,
                      const std::vector<std::vector<std::uint8_t>>& wd14_images) {
  json::Value out = json::Value::object();
  for (const auto& [key, value] : body.members()) {
    if (key == "image" || key == "bild")
      continue;
    if (key == kWd14Task && value.is_object() && !wd14_images.empty()) {
      json::Value options = json::Value::object();
      for (const auto& [name, option] : value.members()) {
        if (name != "images")
          options[name] = option;
      }
      json::Value images = json::Value::array();
      for (const auto& picture : wd14_images)
        images.push_back(DataUrl(picture));
      options["images"] = std::move(images);
      out[key] = std::move(options);
      continue;
    }
    out[key] = value;
  }
  out["image"] = DataUrl(bytes);
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

/// Bytes of one input picture: an Ablage path through the gate, else a
/// data URL, https URL or bare base64. A refusal of the gate is returned in
/// *failure; anything else unreadable throws. `budget` is shared by all
/// pictures of the request (kMaxRequestBytes); each read gets 15 s.
std::vector<std::uint8_t> ReadInput(const Options& options, const HttpRequest& outer,
                                    const std::string& source, core::ImageReadBudget& budget,
                                    std::optional<HttpResponse>* failure) {
  std::vector<std::uint8_t> bytes;
  if (IsAblagePath(source)) {
    bytes = ReadAblage(options, outer, source, failure);
  } else {
    const std::string url = source.starts_with("data:") || source.starts_with("https://")
                                ? source
                                : "data:image/png;base64," + source;
    budget.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    const std::size_t before = budget.remaining_bytes;
    bytes = core::ReadImageUrl(url, budget);
    if (budget.remaining_bytes != before)
      return bytes;  // ReadImageUrl counted them itself
  }
  if (bytes.size() > budget.remaining_bytes)
    throw std::invalid_argument("all pictures of the request together exceed " +
                                std::to_string(kMaxRequestBytes >> 20) + " MB");
  budget.remaining_bytes -= bytes.size();
  return bytes;
}

/// RGB8 pixels of encoded bytes, the alpha channel dropped (the pipeline
/// opens its pictures with .convert("RGB")). `pixels_left` is the request's
/// remaining kMaxRequestPixels.
models::ccip::RgbImage DecodeRgb(const std::vector<std::uint8_t>& bytes,
                                 std::int64_t* pixels_left) {
  core::Image rgb = core::DecodeImage(std::span<const std::uint8_t>(bytes.data(), bytes.size()));
  const auto pixels = static_cast<std::int64_t>(rgb.width) * static_cast<std::int64_t>(rgb.height);
  if (pixels > *pixels_left)
    throw std::invalid_argument("all pictures of the request together exceed " +
                                std::to_string(kMaxRequestPixels / 1000000) + " megapixels");
  *pixels_left -= pixels;
  models::ccip::RgbImage image;
  image.width = static_cast<int>(rgb.width);
  image.height = static_cast<int>(rgb.height);
  image.rgb = std::move(rgb.pixels);
  return image;
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
  for (const std::string& name : names) {
    const auto& known = DetectorTasks();
    const auto& poses = PoseTasks();
    if (name != kWd14Task && name != kOcrTask &&
        std::none_of(known.begin(), known.end(),
                     [&](const TaskSpec& spec) { return spec.task == name; }) &&
        std::find(poses.begin(), poses.end(), name) == poses.end())
      throw std::invalid_argument("unknown task " + name +
                                  " (person, head, face, hand, pose, pose_wholebody, ocr, wd14_tags)");
  }
  // wd14_tags options; regions "person" run the person detector too.
  request.wd14 =
      ParseWd14(body, std::find(names.begin(), names.end(), kWd14Task) != names.end());
  if (request.wd14.persons && std::find(names.begin(), names.end(), "person") == names.end())
    names.push_back("person");
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
  if (const json::Value* boxes = body.find("pose_boxes"); boxes != nullptr && !boxes->is_null()) {
    if (!request.pose && !request.pose_wholebody)
      throw std::invalid_argument("pose_boxes needs the task pose or pose_wholebody");
    if (!boxes->is_array() || boxes->empty() || boxes->size() > kMaxPoseBoxes)
      throw std::invalid_argument("pose_boxes must list 1 to " + std::to_string(kMaxPoseBoxes) +
                                  " boxes [x0, y0, x1, y1]");
    for (const json::Value& box : boxes->items()) {
      if (!box.is_array() || box.size() != 4)
        throw std::invalid_argument("pose_boxes: every box is [x0, y0, x1, y1]");
      std::array<float, 4> b{};
      for (std::size_t i = 0; i < 4; ++i) {
        const json::Value& v = box.items()[i];
        if (!v.is_number() || !(std::fabs(v.as_double()) <= 1e6))
          throw std::invalid_argument("pose_boxes: box values must be numbers within +-1e6");
        b[i] = static_cast<float>(v.as_double());
      }
      if (!(b[2] > b[0] && b[3] > b[1]))
        throw std::invalid_argument("pose_boxes: every box needs x1 > x0 and y1 > y0");
      request.pose_boxes.push_back(b);
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
  const bool ocr_requested = std::find(names.begin(), names.end(), kOcrTask) != names.end();
  OcrPass ocr_base;  // thresholds.ocr: the defaults of every pass
  if (const json::Value* thresholds = body.find("thresholds");
      thresholds != nullptr && !thresholds->is_null()) {
    if (!thresholds->is_object())
      throw std::invalid_argument("thresholds must be an object by task");
    for (const auto& [key_task, values] : thresholds->members()) {
      const std::string& task = key_task;
      if (task == kOcrTask) {
        if (!ocr_requested)
          throw std::invalid_argument("thresholds.ocr: task not requested");
        if (!values.is_object())
          throw std::invalid_argument("thresholds.ocr must be an object");
        for (const auto& [key, value] : values.members()) {
          if (key == "heat")
            ocr_base.heat_threshold = OcrThreshold(value, "thresholds.ocr.heat");
          else if (key == "box")
            ocr_base.box_threshold = OcrThreshold(value, "thresholds.ocr.box");
          else
            throw std::invalid_argument("thresholds.ocr." + key + ": use heat or box");
        }
        continue;
      }
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
  request.ocr = ParseOcr(body, ocr_requested, ocr_base, &request.ocr_list);
  return request;
}

/// Loaded detectors (by model stem) and the CPU queue, shared by requests.
struct VisionState {
  std::mutex load_mutex;
  std::map<std::string, std::unique_ptr<models::ccip::YoloDetector>> detectors;
  std::unique_ptr<models::ccip::Wd14Tagger> wd14;
  std::unique_ptr<models::ccip::TextOcr> ocr;
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

  /// The WD14 tagger (~460 MB host RAM), loaded on first use; null with
  /// *error set.
  const models::ccip::Wd14Tagger* GetWd14(const Options& options, std::string* error) {
    std::lock_guard<std::mutex> lock(load_mutex);
    if (wd14 != nullptr)
      return wd14.get();
    if (options.model_dir.empty()) {
      *error = "image recognition is not configured: start the router with --vision-dir";
      return nullptr;
    }
    const std::string stem = models::ccip::kWd14Stem;
    for (const char* ext : {".safetensors", ".json"}) {
      const std::filesystem::path path = options.model_dir / (stem + ext);
      std::error_code ec;
      if (!std::filesystem::is_regular_file(path, ec)) {
        *error = "model file missing: " + path.string() + " (convert " + kWd14Repo + " " +
                 kWd14Model + "/model.onnx and its selected_tags.csv with "
                 "extensions-ours/ccip/reference/convert_wd14.py)";
        return nullptr;
      }
    }
    std::string load_error;
    auto net = models::ccip::Wd14Tagger::Load(options.model_dir, stem, &load_error);
    if (net == nullptr) {
      *error = "wd14_tags: " + load_error;
      return nullptr;
    }
    wd14 = std::move(net);
    return wd14.get();
  }

  /// The text recognizer (detector + recognizer + dictionary), loaded on
  /// first use; null with *error set.
  const models::ccip::TextOcr* GetOcr(const Options& options, std::string* error) {
    std::lock_guard<std::mutex> lock(load_mutex);
    if (ocr != nullptr)
      return ocr.get();
    if (options.model_dir.empty()) {
      *error = "image recognition is not configured: start the router with --vision-dir";
      return nullptr;
    }
    const std::string det = kOcrDet, rec = kOcrRec;
    for (const std::string& name : {det + ".safetensors", det + ".json", rec + ".safetensors",
                                    rec + ".json", rec + ".dict.txt"}) {
      const std::filesystem::path path = options.model_dir / name;
      std::error_code ec;
      if (!std::filesystem::is_regular_file(path, ec)) {
        *error = "model file missing: " + path.string() + " (convert " + kOcrRepo +
                 " det/" + det + " and rec/" + rec +
                 " with extensions-ours/ccip/reference/convert_ocr.py)";
        return nullptr;
      }
    }
    std::string load_error;
    auto loaded = models::ccip::TextOcr::Load(options.model_dir, det, rec, &load_error);
    if (loaded == nullptr) {
      *error = "ocr: " + load_error;
      return nullptr;
    }
    ocr = std::move(loaded);
    return ocr.get();
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
  const models::ccip::Wd14Tagger* tagger = nullptr;
  if (parsed.wd14.enabled && missing.empty())
    tagger = state_->GetWd14(options_, &missing);
  const models::ccip::TextOcr* ocr = nullptr;
  if (!parsed.ocr.empty() && missing.empty())
    ocr = state_->GetOcr(options_, &missing);
  if (missing.empty() && (parsed.pose || parsed.pose_wholebody) && parsed.pose_boxes.empty())
    state_->GetPose(options_, &state_->rtdetr, "rtdetr_r50vd_coco_o365", "convert_pose.py rtdetr", &missing);
  if (missing.empty() && parsed.pose)
    state_->GetPose(options_, &state_->vitpose, "vitpose_base_simple", "convert_pose.py vitpose", &missing);
  if (missing.empty() && parsed.pose_wholebody)
    state_->GetPose(options_, &state_->dwpose, "dwpose_wholebody", "convert_dwpose.py", &missing);
  // Input before the queue: reading an Ablage path goes through the gate
  // and must not hold the CPU turn.
  std::vector<std::uint8_t> bytes;
  std::vector<std::vector<std::uint8_t>> wd14_bytes;
  core::ImageReadBudget budget;
  budget.remaining_bytes = kMaxRequestBytes;
  try {
    std::optional<HttpResponse> failure;
    bytes = ReadInput(options_, request, parsed.image, budget, &failure);
    if (failure.has_value())
      return std::move(*failure);
  } catch (const std::exception& error) {
    return Error(400, "Bad Request", std::string("image: ") + error.what(), "invalid_request");
  }
  for (std::size_t i = 0; i < parsed.wd14.images.size(); ++i) {
    try {
      std::optional<HttpResponse> failure;
      wd14_bytes.push_back(
          ReadInput(options_, request, parsed.wd14.images[i], budget, &failure));
      if (failure.has_value())
        return std::move(*failure);
    } catch (const std::exception& error) {
      return Error(400, "Bad Request",
                   "wd14_tags.images[" + std::to_string(i) + "]: " + error.what(),
                   "invalid_request");
    }
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
      answer = options_.relay(
          cpu::RelayRequest(request, RelayBody(body, bytes, wd14_bytes), force));
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
  // The CPU turn first, then decoding: a waiting request holds only its
  // encoded bytes (kMaxRequestBytes), never decoded pixels.
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
  models::ccip::RgbImage image;
  std::vector<models::ccip::RgbImage> wd14_images;
  std::int64_t pixels_left = kMaxRequestPixels;
  try {
    // RGB8 with the alpha channel dropped (imgutils puts RGBA on white).
    image = DecodeRgb(bytes, &pixels_left);
  } catch (const std::exception& error) {
    return Error(400, "Bad Request", std::string("image: ") + error.what(), "invalid_request");
  }
  bytes.clear();
  bytes.shrink_to_fit();
  for (std::size_t i = 0; i < wd14_bytes.size(); ++i) {
    try {
      wd14_images.push_back(DecodeRgb(wd14_bytes[i], &pixels_left));
    } catch (const std::exception& error) {
      return Error(400, "Bad Request",
                   "wd14_tags.images[" + std::to_string(i) + "]: " + error.what(),
                   "invalid_request");
    }
    wd14_bytes[i].clear();
    wd14_bytes[i].shrink_to_fit();
  }
  wd14_bytes.clear();
  // Explicit wd14 regions must leave a cut-out inside the image.
  for (std::size_t i = 0; i < parsed.wd14.boxes.size(); ++i) {
    const auto cut = models::ccip::MarginCropBox(parsed.wd14.boxes[i], parsed.wd14.margin,
                                                 image.width, image.height);
    if (cut[2] <= cut[0] || cut[3] <= cut[1])
      return Error(400, "Bad Request",
                   "wd14_tags.regions[" + std::to_string(i) + "] lies outside the image",
                   "invalid_request");
  }
  // Enlarged OCR passes stay within kMaxOcrPixels (host RAM).
  for (const OcrPass& pass : parsed.ocr) {
    const auto pixels = static_cast<std::int64_t>(image.width) * pass.scale *
                        static_cast<std::int64_t>(image.height) * pass.scale;
    if (pass.scale > 1 && pixels > kMaxOcrPixels)
      return Error(400, "Bad Request",
                   "ocr scale " + std::to_string(pass.scale) + ": " +
                       std::to_string(image.width * pass.scale) + "x" +
                       std::to_string(image.height * pass.scale) + " is more than " +
                       std::to_string(kMaxOcrPixels / 1000000) + " megapixels",
                   "invalid_request");
  }
  const double mean_luma = models::ccip::MeanLuma(image);
  const bool brightened = parsed.brighten_dark && mean_luma < parsed.dark_below;
  // The detectors see the brightened copy; later tasks (OCR, tags) keep
  // the original, as in the pipeline's erkennung.py.
  models::ccip::RgbImage detect_image;
  if (brightened)
    detect_image = models::ccip::ApplyGamma(image, parsed.dark_gamma);
  const models::ccip::RgbImage& detector_input = brightened ? detect_image : image;
  json::Value detections = json::Value::object();
  json::Value used = json::Value::object();
  std::vector<std::array<int, 4>> persons;  // for wd14_tags regions "person"
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
        if (spec.task == "person")
          persons.push_back(d.box);
      }
      detections[spec.task] = std::move(list);
      json::Value model = json::Value::object();
      model["repo"] = spec.repo;
      model["model"] = spec.model;
      model["conf"] = std::round(static_cast<double>(spec.conf) * 1e6) / 1e6;
      model["iou"] = std::round(static_cast<double>(spec.iou) * 1e6) / 1e6;
      used[spec.task] = std::move(model);
    }
    // Tags on the original pixels (never the brightened copy), like the
    // pipeline's erkennung.py.
    if (tagger != nullptr) {
      detections[kWd14Task] = RunWd14(*tagger, image, parsed.wd14, persons, wd14_images);
      used[kWd14Task] = Wd14ModelInfo(parsed.wd14);
    }
    // OCR on the original pixels too (erkennung.py / erkennung_schrift.py),
    // each pass on its own Pillow-LANCZOS enlargement when scale > 1.
    if (ocr != nullptr) {
      std::map<int, models::ccip::RgbImage> enlarged;
      json::Value lists = json::Value::array();
      json::Value infos = json::Value::array();
      for (const OcrPass& pass : parsed.ocr) {
        const models::ccip::RgbImage* input = &image;
        if (pass.scale > 1) {
          auto it = enlarged.find(pass.scale);
          if (it == enlarged.end())
            it = enlarged
                     .emplace(pass.scale,
                              models::ccip::ResizeLanczos(image, image.width * pass.scale,
                                                          image.height * pass.scale))
                     .first;
          input = &it->second;
        }
        models::ccip::OcrOptions ocr_options;
        ocr_options.heat_threshold = pass.heat_threshold;
        ocr_options.box_threshold = pass.box_threshold;
        ocr_options.unclip_ratio = pass.unclip_ratio;
        lists.push_back(OcrEntries(ocr->Run(*input, ocr_options), pass.scale));
        infos.push_back(OcrModelInfo(pass));
      }
      if (parsed.ocr_list) {
        detections[kOcrTask] = std::move(lists);
        used[kOcrTask] = std::move(infos);
      } else {
        detections[kOcrTask] = lists.items().front();
        used[kOcrTask] = infos.items().front();
      }
    }
    if (parsed.pose || parsed.pose_wholebody) {
      // posen_erkennung.erkenne on the original image (not the brightened copy).
      // pose_boxes: the caller's boxes, in its order, without RT-DETR.
      const bool given = !parsed.pose_boxes.empty();
      std::vector<models::ccip::ObjectBox> pose_persons;
      if (given) {
        for (const auto& box : parsed.pose_boxes) {
          models::ccip::ObjectBox o;
          o.box = box;
          pose_persons.push_back(o);
        }
      } else {
        for (const auto& o : state_->rtdetr->Detect(image, parsed.pose_min_score)) {
          if (o.label == 0)
            pose_persons.push_back(o);
        }
        std::stable_sort(pose_persons.begin(), pose_persons.end(),
                         [](const auto& a, const auto& b) { return a.score > b.score; });
        if (pose_persons.size() > static_cast<std::size_t>(parsed.pose_max_persons))
          pose_persons.resize(static_cast<std::size_t>(parsed.pose_max_persons));
      }
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
      for (const auto& person : pose_persons) {
        json::Value entry = json::Value::object();
        json::Value box = json::Value::array();
        for (float v : person.box)
          box.push_back(static_cast<double>(v));
        entry["box"] = std::move(box);
        if (!given)
          entry["score"] = static_cast<double>(person.score);
        if (parsed.pose)
          entry["keypoints"] = points(state_->vitpose->Estimate(image, person.box));
        if (parsed.pose_wholebody)
          entry["wholebody"] = points(state_->dwpose->Estimate(image, person.box));
        poses.push_back(std::move(entry));
      }
      out_poses = std::move(poses);
      json::Value det = json::Value::object();
      if (given) {
        det["model"] = "pose_boxes";
        det["boxes"] = static_cast<long long>(parsed.pose_boxes.size());
      } else {
        det["repo"] = "PekingU/rtdetr_r50vd_coco_o365";
        det["model"] = "rtdetr_r50vd_coco_o365";
        det["min_score"] = std::round(static_cast<double>(parsed.pose_min_score) * 1e6) / 1e6;
        det["max_persons"] = parsed.pose_max_persons;
      }
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
