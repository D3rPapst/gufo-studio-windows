// `gufo serve vision` HTTP side (see vision_api.hpp).
// Written with AI assistance (Claude Code).
#include "src/cli/serve/media/vision_api.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <map>
#include <utility>

#include <png.h>

#include "src/core/base64.hpp"
#include "src/core/json.hpp"

namespace gufo::server {
namespace {

constexpr std::size_t kMaxDepthImages = 32;
constexpr std::size_t kMaxPrompts = 64;

HttpResponse Json(int status, std::string reason, const json::Value& body) {
  HttpResponse response;
  response.status = status;
  response.reason = std::move(reason);
  response.body = body.dump();
  response.headers = {{"Content-Type", "application/json"}};
  return response;
}

HttpResponse Error(int status, std::string reason, std::string message,
                   std::string code) {
  json::Value root = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = std::move(message);
  error["type"] = status >= 500 ? "server_error" : "invalid_request_error";
  error["code"] = std::move(code);
  root["error"] = std::move(error);
  return Json(status, std::move(reason), root);
}

HttpResponse BadRequest(std::string message, std::string code = "invalid_request") {
  return Error(400, "Bad Request", std::move(message), std::move(code));
}

double Seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since)
      .count();
}

/// A data URL or https URL (core reader), else plain base64.
core::Image ReadImage(const std::string& text, core::ImageReadBudget& budget) {
  std::vector<std::uint8_t> bytes;
  if (text.starts_with("data:") || text.starts_with("https://")) {
    bytes = core::ReadImageUrl(text, budget);
  } else {
    core::Base64Options options;
    options.canonical = false;
    options.max_decoded = core::kMaxEncodedImageBytes;
    if (core::DecodeBase64(text, options, bytes) != core::Base64Status::kOk)
      throw std::invalid_argument(
          "image must be a data URL, an https URL or base64");
  }
  return core::DecodeImage(bytes);
}

std::string Lower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

/// hf.py: l.lower().strip(". ") per label.
std::string CleanLabel(const std::string& label) {
  std::string text = Lower(label);
  const auto begin = text.find_first_not_of(". ");
  if (begin == std::string::npos)
    return "";
  const auto end = text.find_last_not_of(". ");
  return text.substr(begin, end - begin + 1);
}

/// "labels": ["person", ...] or "text": "person. sword."
bool ParseLabels(const json::Value& body, std::vector<std::string>* raw,
                 std::vector<std::string>* clean, std::string* error) {
  if (const json::Value* labels = body.find("labels");
      labels != nullptr && !labels->is_null()) {
    if (!labels->is_array() || labels->empty()) {
      *error = "labels must be a non-empty array of English names";
      return false;
    }
    for (const json::Value& item : labels->items()) {
      if (!item.is_string()) {
        *error = "labels must be strings";
        return false;
      }
      raw->push_back(item.str());
    }
  } else if (const json::Value* text = body.find("text");
             text != nullptr && text->is_string()) {
    std::string piece;
    for (char c : text->str() + ".") {
      if (c == '.') {
        if (!CleanLabel(piece).empty())
          raw->push_back(piece);
        piece.clear();
      } else {
        piece += c;
      }
    }
  }
  for (const std::string& label : *raw) {
    const std::string c = CleanLabel(label);
    if (c.empty()) {
      *error = "empty label";
      return false;
    }
    clean->push_back(c);
  }
  if (clean->empty()) {
    *error = "labels (array) or text (\"a. b.\") is required";
    return false;
  }
  return true;
}

float Member(const json::Value& body, const char* name, float fallback) {
  const json::Value* v = body.find(name);
  return v != nullptr && v->is_number() ? static_cast<float>(v->as_double())
                                        : fallback;
}

json::Value BoxJson(const std::array<float, 4>& box) {
  json::Value out = json::Value::array();
  for (float v : box)
    out.push_back(static_cast<double>(v));
  return out;
}

void WritePng(png_structp png, png_bytep data, png_size_t size) {
  auto* out = static_cast<std::vector<std::uint8_t>*>(png_get_io_ptr(png));
  out->insert(out->end(), data, data + size);
}

void FlushPng(png_structp) {}

/// PNG from 8- or 16-bit samples (16 bit as big-endian pairs) of
/// `color_type` (PNG_COLOR_TYPE_GRAY: 1 sample per pixel, _RGBA: 4).
std::vector<std::uint8_t> Png(const std::vector<std::uint8_t>& samples,
                              int width, int height, int bits,
                              int color_type) {
  const int channels = color_type == PNG_COLOR_TYPE_RGBA ? 4 : 1;
  std::vector<std::uint8_t> out;
  png_structp png =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (png == nullptr)
    return {};
  png_infop info = png_create_info_struct(png);
  if (info == nullptr || setjmp(png_jmpbuf(png))) {
    png_destroy_write_struct(&png, info != nullptr ? &info : nullptr);
    return {};
  }
  png_set_write_fn(png, &out, WritePng, FlushPng);
  png_set_IHDR(png, info, static_cast<png_uint_32>(width),
               static_cast<png_uint_32>(height), bits, color_type,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  const std::size_t row =
      static_cast<std::size_t>(width) * channels * (bits / 8);
  for (int y = 0; y < height; ++y)
    png_write_row(png, const_cast<png_bytep>(samples.data() + row * y));
  png_write_end(png, nullptr);
  png_destroy_write_struct(&png, &info);
  return out;
}

/// Grey PNG from 8- or 16-bit samples (16 bit as big-endian pairs).
std::vector<std::uint8_t> GreyPng(const std::vector<std::uint8_t>& samples,
                                  int width, int height, int bits) {
  return Png(samples, width, height, bits, PNG_COLOR_TYPE_GRAY);
}

std::string MaskPng(const VisionMask& mask) {
  std::vector<std::uint8_t> samples(mask.mask.size());
  for (std::size_t i = 0; i < samples.size(); ++i)
    samples[i] = mask.mask[i] != 0 ? 255 : 0;
  return core::EncodeBase64(std::span<const std::uint8_t>(
      GreyPng(samples, mask.width, mask.height, 8)));
}

/// Segment "format": "rgba" (with "crop"): `target` gets "rgba_b64", the
/// input image as RGBA PNG with the mask as alpha (255 = region, 0 =
/// transparent), cut to the mask's bounding box with `crop` ("crop_box":
/// [x0, y0, x1, y1] in image pixels, x1 / y1 exclusive). An empty mask
/// with `crop`, or a mask of another size: null.
void AddRgba(const core::Image& image, const VisionMask& mask, bool crop,
             json::Value* target) {
  const int W = static_cast<int>(image.width), H = static_cast<int>(image.height);
  const auto none = [&] {
    (*target)["rgba_b64"] = json::Value();
    if (crop)
      (*target)["crop_box"] = json::Value();
  };
  if (mask.width != W || mask.height != H ||
      mask.mask.size() != static_cast<std::size_t>(W) * H ||
      image.pixels.size() != static_cast<std::size_t>(W) * H * 3) {
    none();
    return;
  }
  std::array<int, 4> box{0, 0, W, H};
  if (crop) {
    box = {W, H, 0, 0};
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        if (mask.mask[static_cast<std::size_t>(y) * W + x] != 0) {
          box[0] = std::min(box[0], x);
          box[1] = std::min(box[1], y);
          box[2] = std::max(box[2], x + 1);
          box[3] = std::max(box[3], y + 1);
        }
    if (box[2] <= box[0]) {
      none();
      return;
    }
  }
  const int w = box[2] - box[0], h = box[3] - box[1];
  std::vector<std::uint8_t> samples(static_cast<std::size_t>(w) * h * 4);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const std::size_t source =
          static_cast<std::size_t>(box[1] + y) * W + (box[0] + x);
      std::uint8_t* out = samples.data() + (static_cast<std::size_t>(y) * w + x) * 4;
      std::copy_n(image.pixels.data() + source * 3, 3, out);
      out[3] = mask.mask[source] != 0 ? 255 : 0;
    }
  (*target)["rgba_b64"] = core::EncodeBase64(std::span<const std::uint8_t>(
      Png(samples, w, h, 8, PNG_COLOR_TYPE_RGBA)));
  if (crop) {
    json::Value crop_box = json::Value::array();
    for (int v : box)
      crop_box.push_back(v);
    (*target)["crop_box"] = std::move(crop_box);
  }
}

json::Value ParseBodyOrError(const HttpRequest& request, HttpResponse* error) {
  try {
    std::string_view text = request.body;
    if (text.substr(0, 3) == "\xEF\xBB\xBF")
      text = text.substr(3);
    json::Value body = json::parse(text);
    if (!body.is_object()) {
      *error = BadRequest("request body must be a JSON object");
      return json::Value();
    }
    return body;
  } catch (const std::exception&) {
    *error = BadRequest("request body must be valid JSON", "parse_error");
    return json::Value();
  }
}

std::string DetectorName(const VisionBackends& backends, const VisionFindRequest& request) {
  if (!request.detector.empty() || backends.find_names.empty())
    return request.detector;
  return backends.find_names.front();
}

}  // namespace

void DepthPercentiles(const std::vector<const std::vector<float>*>& maps,
                      double* lo, double* hi) {
  std::vector<double> values;
  for (const std::vector<float>* map : maps)
    for (std::size_t i = 0; i < map->size(); i += 97)
      values.push_back((*map)[i]);
  if (values.empty()) {
    *lo = *hi = 0;
    return;
  }
  std::sort(values.begin(), values.end());
  const auto percentile = [&](double q) {
    const double rank = q / 100.0 * static_cast<double>(values.size() - 1);
    const auto below = static_cast<std::size_t>(std::floor(rank));
    const std::size_t above = std::min(below + 1, values.size() - 1);
    const double t = rank - static_cast<double>(below);
    return values[below] + (values[above] - values[below]) * t;
  };
  *lo = percentile(2);
  *hi = percentile(98);
}

std::vector<std::uint8_t> EncodeGreyPng(const std::vector<float>& map, int width,
                                        int height, double lo, double hi,
                                        int bits) {
  const double span = std::max(hi - lo, 1e-6);
  const std::size_t n = static_cast<std::size_t>(width) * height;
  std::vector<std::uint8_t> samples(n * (bits == 16 ? 2 : 1));
  for (std::size_t i = 0; i < n; ++i) {
    // float32 arithmetic as numpy does on the float32 map.
    const float v = std::clamp(static_cast<float>((map[i] - static_cast<float>(lo)) /
                                                  static_cast<float>(span)),
                               0.0F, 1.0F);
    if (bits == 16) {
      const auto s = static_cast<std::uint16_t>(std::lround(v * 65535.0F));
      samples[2 * i] = static_cast<std::uint8_t>(s >> 8);
      samples[2 * i + 1] = static_cast<std::uint8_t>(s & 0xFF);
    } else {
      samples[i] = static_cast<std::uint8_t>(v * 255.0F);
    }
  }
  return GreyPng(samples, width, height, bits);
}

VisionModelService::VisionModelService(std::string model_id, VisionBackends backends)
    : model_id_(std::move(model_id)), backends_(std::move(backends)) {}

void VisionModelService::Describe(json::Value* entry, bool) const {
  json::Value tasks = json::Value::array();
  if (backends_.depth)
    tasks.push_back("depth");
  if (backends_.find)
    tasks.push_back("find");
  if (backends_.segment)
    tasks.push_back("segment");
  (*entry)["tasks"] = std::move(tasks);
  json::Value sizes = json::Value::array();
  for (const std::string& size : backends_.depth_sizes)
    sizes.push_back(size);
  (*entry)["depth_models"] = std::move(sizes);
  json::Value finders = json::Value::array();
  for (const std::string& name : backends_.find_names)
    finders.push_back(name);
  (*entry)["find_models"] = std::move(finders);
}

bool VisionModelService::HandlesPath(std::string_view path) const {
  return path == "/v1/vision/depth" || path == "/v1/vision/find" ||
         path == "/v1/vision/segment";
}

HttpResponse VisionModelService::Handle(const HttpRequest& request) {
  if (request.method != "POST")
    return Error(405, "Method Not Allowed", "use POST", "method_not_allowed");
  HttpResponse error;
  const json::Value body = ParseBodyOrError(request, &error);
  if (!body.is_object())
    return error;
  if (request.path == "/v1/vision/depth")
    return Depth(body);
  if (request.path == "/v1/vision/find")
    return Find(body);
  return Segment(body);
}

HttpResponse VisionModelService::Depth(const json::Value& body) {
  if (!backends_.depth || backends_.depth_sizes.empty())
    return Error(501, "Not Implemented",
                 "this worker has no depth model (--depth-model)",
                 "task_not_available");
  // "depth_model" or "size": a --depth-model name (large, small, ...).
  const std::string size = body.member_str(
      "depth_model", body.member_str("size", backends_.depth_sizes.front()));
  if (std::find(backends_.depth_sizes.begin(), backends_.depth_sizes.end(),
                size) == backends_.depth_sizes.end())
    return BadRequest("size must be one of the loaded depth models", "unknown_size");
  const std::string format = body.member_str("format", "png16");
  if (format != "png16" && format != "png8" && format != "f32")
    return BadRequest("format must be png16, png8 or f32");
  const std::string normalize = body.member_str("normalize", "percentile");
  if (normalize != "percentile" && normalize != "minmax")
    return BadRequest("normalize must be percentile or minmax");
  std::vector<std::string> sources;
  if (const json::Value* images = body.find("images");
      images != nullptr && images->is_array()) {
    for (const json::Value& item : images->items())
      if (item.is_string())
        sources.push_back(item.str());
  } else if (body.member_str("image").size() > 0) {
    sources.push_back(body.member_str("image"));
  }
  if (sources.empty() || sources.size() > kMaxDepthImages)
    return BadRequest("image (string) or images (1 to 32 strings) is required");
  core::ImageReadBudget budget;
  std::vector<core::Image> images;
  try {
    for (const std::string& source : sources)
      images.push_back(ReadImage(source, budget));
  } catch (const std::exception& e) {
    return BadRequest(std::string("image: ") + e.what(), "invalid_media");
  }
  const auto started = std::chrono::steady_clock::now();
  std::vector<std::vector<float>> maps(images.size());
  {
    std::lock_guard lock(run_mutex_);
    for (std::size_t i = 0; i < images.size(); ++i) {
      std::string run_error;
      if (!backends_.depth(size, images[i], &maps[i], &run_error))
        return Error(500, "Internal Server Error", run_error, "depth_failed");
    }
  }
  double lo = 0, hi = 0;
  if (normalize == "percentile") {
    std::vector<const std::vector<float>*> refs;
    for (const auto& m : maps)
      refs.push_back(&m);
    DepthPercentiles(refs, &lo, &hi);
  } else {
    lo = INFINITY;
    hi = -INFINITY;
    for (const auto& m : maps)
      for (float v : m) {
        lo = std::min<double>(lo, v);
        hi = std::max<double>(hi, v);
      }
  }
  json::Value data = json::Value::array();
  for (std::size_t i = 0; i < images.size(); ++i) {
    const int w = static_cast<int>(images[i].width);
    const int h = static_cast<int>(images[i].height);
    json::Value item = json::Value::object();
    item["width"] = w;
    item["height"] = h;
    item["format"] = format;
    float mn = INFINITY, mx = -INFINITY;
    for (float v : maps[i]) {
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
    item["min"] = static_cast<double>(mn);
    item["max"] = static_cast<double>(mx);
    if (format == "f32") {
      item["b64_json"] = core::EncodeBase64(std::as_bytes(std::span(maps[i])));
    } else {
      item["b64_json"] = core::EncodeBase64(std::span<const std::uint8_t>(
          EncodeGreyPng(maps[i], w, h, lo, hi, format == "png8" ? 8 : 16)));
    }
    data.push_back(std::move(item));
  }
  json::Value out = json::Value::object();
  out["object"] = "vision.depth";
  out["model"] = model_id_;
  out["size"] = size;
  json::Value norm = json::Value::object();
  norm["method"] = normalize;
  norm["lo"] = lo;
  norm["hi"] = hi;
  out["normalization"] = std::move(norm);
  out["data"] = std::move(data);
  if (backends_.describe)
    out["source"] = backends_.describe("depth", size);
  out["seconds"] = Seconds(started);
  return Json(200, "OK", out);
}

HttpResponse VisionModelService::Find(const json::Value& body) {
  if (!backends_.find)
    return Error(501, "Not Implemented",
                 "this worker has no Grounding DINO model (--find-model)",
                 "task_not_available");
  std::vector<std::string> raw;
  VisionFindRequest request;
  std::string parse_error;
  if (!ParseLabels(body, &raw, &request.labels, &parse_error))
    return BadRequest(parse_error);
  request.box_threshold = Member(body, "box_threshold", 0.3F);
  request.text_threshold = Member(body, "text_threshold", 0.25F);
  request.detector = body.member_str("detector");
  if (!request.detector.empty() &&
      std::find(backends_.find_names.begin(), backends_.find_names.end(),
                request.detector) == backends_.find_names.end())
    return BadRequest("detector must be one of the loaded find models", "unknown_model");
  core::ImageReadBudget budget;
  core::Image image;
  try {
    image = ReadImage(body.member_str("image"), budget);
  } catch (const std::exception& e) {
    return BadRequest(std::string("image: ") + e.what(), "invalid_media");
  }
  const auto started = std::chrono::steady_clock::now();
  std::vector<VisionBox> boxes;
  {
    std::lock_guard lock(run_mutex_);
    std::string run_error;
    if (!backends_.find(image, request, &boxes, &run_error))
      return Error(500, "Internal Server Error", run_error, "find_failed");
  }
  std::stable_sort(boxes.begin(), boxes.end(),
                   [](const VisionBox& a, const VisionBox& b) { return a.score > b.score; });
  json::Value detections = json::Value::array();
  for (const VisionBox& box : boxes) {
    json::Value item = json::Value::object();
    item["box"] = BoxJson(box.box);
    item["score"] = static_cast<double>(box.score);
    item["label"] = box.label;
    detections.push_back(std::move(item));
  }
  json::Value out = json::Value::object();
  out["object"] = "vision.find";
  out["model"] = model_id_;
  out["width"] = static_cast<int>(image.width);
  out["height"] = static_cast<int>(image.height);
  out["detections"] = std::move(detections);
  if (backends_.describe)
    out["source"] = backends_.describe("find", DetectorName(backends_, request));
  out["seconds"] = Seconds(started);
  return Json(200, "OK", out);
}

HttpResponse VisionModelService::Segment(const json::Value& body) {
  if (!backends_.segment)
    return Error(501, "Not Implemented",
                 "this worker has no SAM model (--segment-model)",
                 "task_not_available");
  core::ImageReadBudget budget;
  core::Image image;
  try {
    image = ReadImage(body.member_str("image"), budget);
  } catch (const std::exception& e) {
    return BadRequest(std::string("image: ") + e.what(), "invalid_media");
  }
  const int W = static_cast<int>(image.width), H = static_cast<int>(image.height);
  // "format": "mask" (the grey PNG masks, default) or "rgba" (also the
  // image with the mask as alpha); "crop": cut that to the mask's box.
  const json::Value* format_value = body.find("format");
  if (format_value != nullptr && !format_value->is_null() &&
      (!format_value->is_string() ||
       (format_value->str() != "mask" && format_value->str() != "rgba")))
    return BadRequest("format must be mask (grey PNG, default) or rgba (also "
                      "the image with the mask as alpha, PNG)");
  const bool rgba = format_value != nullptr && format_value->is_string() &&
                    format_value->str() == "rgba";
  const json::Value* crop_value = body.find("crop");
  if (crop_value != nullptr && !crop_value->is_null() && !crop_value->is_bool())
    return BadRequest("crop must be a boolean");
  const bool crop = crop_value != nullptr && crop_value->is_bool() &&
                    crop_value->as_bool();
  if (crop && !rgba)
    return BadRequest("crop needs \"format\": \"rgba\"");
  const auto started = std::chrono::steady_clock::now();
  json::Value out = json::Value::object();
  out["object"] = "vision.segment";
  out["model"] = model_id_;
  out["width"] = W;
  out["height"] = H;
  const bool by_labels = body.contains("labels") || body.contains("text");
  VisionSegmentPrompt prompt;
  prompt.multimask = body.find("multimask") != nullptr && body.find("multimask")->as_bool();
  std::lock_guard lock(run_mutex_);
  if (by_labels) {
    // hf.py segment(): boxes of all labels, > max_area of the picture
    // dropped, at most `top` per label (matched as hf.py does), one union mask.
    if (!backends_.find)
      return Error(501, "Not Implemented",
                   "segment by labels needs a Grounding DINO model (--find-model)",
                   "task_not_available");
    std::vector<std::string> raw;
    VisionFindRequest request;
    std::string parse_error;
    if (!ParseLabels(body, &raw, &request.labels, &parse_error))
      return BadRequest(parse_error);
    request.box_threshold = Member(body, "box_threshold", 0.3F);
    request.text_threshold = Member(body, "text_threshold", 0.25F);
    request.detector = body.member_str("detector");
    if (!request.detector.empty() &&
        std::find(backends_.find_names.begin(), backends_.find_names.end(),
                  request.detector) == backends_.find_names.end())
      return BadRequest("detector must be one of the loaded find models", "unknown_model");
    const float max_area = Member(body, "max_area", 0.6F);
    const int top = static_cast<int>(Member(body, "top", 3.0F));
    std::vector<VisionBox> boxes;
    std::string run_error;
    if (!backends_.find(image, request, &boxes, &run_error))
      return Error(500, "Internal Server Error", run_error, "find_failed");
    std::stable_sort(boxes.begin(), boxes.end(),
                     [](const VisionBox& a, const VisionBox& b) { return a.score > b.score; });
    std::map<std::string, int> found;
    for (const std::string& label : raw)
      found[label] = 0;
    json::Value used = json::Value::array();
    for (const VisionBox& box : boxes) {
      const auto& b = box.box;
      if ((b[2] - b[0]) * (b[3] - b[1]) > max_area * W * H)
        continue;
      const std::string& name = box.label;
      std::string label = raw.front();
      for (const std::string& candidate : raw) {
        const std::string low = Lower(candidate);
        if (!name.empty() && (low.find(name) != std::string::npos ||
                              name.find(low) != std::string::npos)) {
          label = candidate;
          break;
        }
      }
      if (found[label] >= top)
        continue;
      ++found[label];
      prompt.boxes.push_back(b);
      json::Value item = json::Value::object();
      item["box"] = BoxJson(b);
      item["score"] = static_cast<double>(box.score);
      item["label"] = label;
      used.push_back(std::move(item));
    }
    json::Value counts = json::Value::object();
    for (const std::string& label : raw)
      counts[label] = found[label];
    out["found"] = std::move(counts);
    out["boxes"] = std::move(used);
    if (prompt.boxes.empty()) {
      out["mask"] = json::Value();
      if (rgba) {
        out["rgba_b64"] = json::Value();
        if (crop)
          out["crop_box"] = json::Value();
      }
      out["seconds"] = Seconds(started);
      return Json(200, "OK", out);
    }
    prompt.multimask = false;
    std::vector<VisionMask> masks;
    if (!backends_.segment(image, prompt, &masks, &run_error))
      return Error(500, "Internal Server Error", run_error, "segment_failed");
    VisionMask all;
    all.width = W;
    all.height = H;
    all.mask.assign(static_cast<std::size_t>(W) * H, 0);
    for (const VisionMask& m : masks)
      for (std::size_t i = 0; i < all.mask.size() && i < m.mask.size(); ++i)
        all.mask[i] |= m.mask[i];
    out["mask"] = MaskPng(all);
    if (rgba)
      AddRgba(image, all, crop, &out);
    if (backends_.describe) {
      json::Value source = json::Value::object();
      source["find"] = backends_.describe("find", DetectorName(backends_, request));
      source["segment"] = backends_.describe("segment", "");
      out["source"] = std::move(source);
    }
    out["seconds"] = Seconds(started);
    return Json(200, "OK", out);
  }
  if (const json::Value* boxes = body.find("boxes"); boxes != nullptr && boxes->is_array()) {
    for (const json::Value& b : boxes->items()) {
      if (!b.is_array() || b.size() != 4)
        return BadRequest("boxes must be [x0, y0, x1, y1] arrays");
      prompt.boxes.push_back({static_cast<float>(b.items()[0].as_double()),
                              static_cast<float>(b.items()[1].as_double()),
                              static_cast<float>(b.items()[2].as_double()),
                              static_cast<float>(b.items()[3].as_double())});
    }
  }
  if (const json::Value* points = body.find("points"); points != nullptr && points->is_array()) {
    for (const json::Value& p : points->items()) {
      if (!p.is_array() || p.size() != 2)
        return BadRequest("points must be [x, y] arrays");
      prompt.points.push_back({static_cast<float>(p.items()[0].as_double()),
                               static_cast<float>(p.items()[1].as_double())});
    }
    if (const json::Value* labels = body.find("point_labels");
        labels != nullptr && labels->is_array()) {
      for (const json::Value& l : labels->items())
        prompt.point_labels.push_back(static_cast<int>(l.as_double()));
    }
    if (prompt.point_labels.empty())
      prompt.point_labels.assign(prompt.points.size(), 1);
    if (prompt.point_labels.size() != prompt.points.size())
      return BadRequest("point_labels must match points");
  }
  if (prompt.boxes.empty() && prompt.points.empty())
    return BadRequest("labels, boxes or points is required");
  if (prompt.boxes.size() > kMaxPrompts || prompt.points.size() > kMaxPrompts)
    return BadRequest("at most 64 boxes and 64 points");
  if (!prompt.boxes.empty() && !prompt.points.empty())
    return BadRequest("send boxes or points, not both");
  std::vector<VisionMask> masks;
  std::string run_error;
  if (!backends_.segment(image, prompt, &masks, &run_error))
    return Error(500, "Internal Server Error", run_error, "segment_failed");
  json::Value list = json::Value::array();
  for (const VisionMask& m : masks) {
    json::Value item = json::Value::object();
    item["b64_json"] = MaskPng(m);
    item["score"] = static_cast<double>(m.score);
    if (rgba)
      AddRgba(image, m, crop, &item);
    list.push_back(std::move(item));
  }
  out["masks"] = std::move(list);
  if (backends_.describe)
    out["source"] = backends_.describe("segment", "");
  out["seconds"] = Seconds(started);
  return Json(200, "OK", out);
}

}  // namespace gufo::server
