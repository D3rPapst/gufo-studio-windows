#include "src/cli/serve/router/job_files.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <system_error>
#include <utility>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::router::job_files {
namespace {

namespace fs = std::filesystem;
using server::HttpRequest;
using server::HttpResponse;

/// job.json is a few hundred bytes; anything far larger is not ours.
constexpr std::uintmax_t kMaxJobJsonBytes = 1U << 20U;

bool IsLowerHex(std::string_view text) {
  return std::all_of(text.begin(), text.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

HttpResponse Error(int status, std::string reason, std::string message,
                   std::string code) {
  json::Value root = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = std::move(message);
  error["type"] = "invalid_request_error";
  error["code"] = std::move(code);
  root["error"] = std::move(error);
  HttpResponse response;
  response.status = status;
  response.reason = std::move(reason);
  response.body = root.dump();
  response.headers.emplace_back("Content-Type", "application/json");
  return response;
}

/// `candidate` lies below `root` (both canonical), never equal to it.
bool IsStrictDescendant(const fs::path& root, const fs::path& candidate) {
  auto r = root.begin();
  auto c = candidate.begin();
  for (; r != root.end() && c != candidate.end(); ++r, ++c) {
    if (*r != *c)
      return false;
  }
  return r == root.end() && c != candidate.end();
}

/// A plain file (no symlink, no junction) whose real path lies inside
/// `canonical_root`; its size in `bytes`.
bool IsSafeFile(const fs::path& canonical_root, const fs::path& file,
                std::uintmax_t* bytes) {
  std::error_code error;
  const fs::file_status status = fs::symlink_status(file, error);
  if (error || fs::is_symlink(status) || !fs::is_regular_file(status))
    return false;
  const fs::path real = fs::canonical(file, error);
  if (error || !IsStrictDescendant(canonical_root, real))
    return false;
  const std::uintmax_t size = fs::file_size(file, error);
  if (error)
    return false;
  if (bytes != nullptr)
    *bytes = size;
  return true;
}

bool ReadFile(const fs::path& file, std::uint64_t first, std::uint64_t count,
              std::string* out) {
  if (count > static_cast<std::uint64_t>(
                  std::numeric_limits<std::streamsize>::max()))
    return false;
  std::ifstream input(file, std::ios::binary);
  if (!input)
    return false;
  input.seekg(static_cast<std::streamoff>(first));
  out->assign(static_cast<std::size_t>(count), '\0');
  input.read(out->data(), static_cast<std::streamsize>(count));
  return input.gcount() == static_cast<std::streamsize>(count);
}

struct ByteRange {
  std::uint64_t first{0};
  std::uint64_t last{0};
};

bool ParseUnsigned(std::string_view text, std::uint64_t* result) {
  if (text.empty())
    return false;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), *result);
  return parsed.ec == std::errc() && parsed.ptr == text.data() + text.size();
}

/// The worker's single-range parser (video_api.cpp ParseRange).
std::optional<ByteRange> ParseRange(std::string_view header,
                                    std::uint64_t bytes) {
  if (!header.starts_with("bytes=") || bytes == 0)
    return std::nullopt;
  const std::string_view value = header.substr(6);
  if (value.find(',') != std::string_view::npos)
    return std::nullopt;
  const std::size_t dash = value.find('-');
  if (dash == std::string_view::npos)
    return std::nullopt;
  const std::string_view first_text = value.substr(0, dash);
  const std::string_view last_text = value.substr(dash + 1);
  if (first_text.empty()) {
    std::uint64_t suffix = 0;
    if (!ParseUnsigned(last_text, &suffix) || suffix == 0)
      return std::nullopt;
    suffix = std::min(suffix, bytes);
    return ByteRange{.first = bytes - suffix, .last = bytes - 1};
  }
  std::uint64_t first = 0;
  if (!ParseUnsigned(first_text, &first) || first >= bytes)
    return std::nullopt;
  std::uint64_t last = bytes - 1;
  if (!last_text.empty() && (!ParseUnsigned(last_text, &last) || last < first))
    return std::nullopt;
  return ByteRange{.first = first, .last = std::min(last, bytes - 1)};
}

/// Unix seconds of a job.json member; 0 for null, missing or not a whole
/// non-negative number.
long long Seconds(const json::Value& job, const char* key) {
  const json::Value* value = job.find(key);
  if (value == nullptr || !value->is_number())
    return 0;
  const double number = value->as_double();
  if (!std::isfinite(number) || number < 0 || std::floor(number) != number ||
      number > 9.0e15)
    return 0;
  return static_cast<long long>(number);
}

/// The status object `gufo serve video` answers (video_api.cpp
/// SnapshotValue), from the persisted job (video_jobs.cpp SnapshotJson).
json::Value VideoStatus(const json::Value& job) {
  json::Value root = json::Value::object();
  root["id"] = job.member_str("id");
  root["object"] = "video";
  root["schema"] = "gufo.video-api.v1";
  root["created_at"] = Seconds(job, "created_at");
  root["status"] = job.member_str("status");
  root["progress"] = static_cast<long long>(job.member_double("progress"));
  root["model"] = job.member_str("model");
  root["size"] = job.member_str("size");
  root["seconds"] = job.member_str("seconds");
  root["output_format"] = job.member_str("output_format");
  const long long completed_at = Seconds(job, "completed_at");
  root["completed_at"] =
      completed_at > 0 ? json::Value(completed_at) : json::Value();
  const long long expires_at = Seconds(job, "expires_at");
  root["expires_at"] = expires_at > 0 ? json::Value(expires_at) : json::Value();
  const json::Value* error = job.find("error");
  if (error != nullptr && error->is_object() &&
      !error->member_str("code").empty()) {
    json::Value copy = json::Value::object();
    copy["code"] = error->member_str("code");
    copy["message"] = error->member_str("message");
    root["error"] = std::move(copy);
  } else {
    root["error"] = json::Value();
  }
  return root;
}

HttpResponse Json(const json::Value& value, const std::string& id) {
  HttpResponse response;
  response.status = 200;
  response.reason = "OK";
  response.body = value.dump();
  response.headers.emplace_back("Content-Type", "application/json");
  response.log_details = "job=" + id + " from=job_files";
  return response;
}

HttpResponse VideoContent(const fs::path& canonical_root, const fs::path& dir,
                          const json::Value& job, const std::string& id,
                          const HttpRequest& request) {
  if (job.member_str("status") != "completed")
    return Error(409, "Conflict", "video content is not ready",
                 "video_not_ready");
  const std::string format = job.member_str("output_format");
  if (format != "mp4" && format != "ppm")
    return NotFound(Kind::kVideo);
  const fs::path file = dir / (format == "mp4" ? "output.mp4" : "output.ppm");
  const std::string media_type =
      format == "mp4" ? "video/mp4" : "image/x-portable-pixmap";
  std::uintmax_t size = 0;
  if (!IsSafeFile(canonical_root, file, &size))
    return NotFound(Kind::kVideo);
  const auto bytes = static_cast<std::uint64_t>(size);
  ByteRange range{.first = 0, .last = bytes == 0 ? 0 : bytes - 1};
  bool partial = false;
  if (const std::string header = request.header("range"); !header.empty()) {
    const auto parsed = ParseRange(header, bytes);
    if (!parsed.has_value()) {
      HttpResponse refused = Error(416, "Range Not Satisfiable",
                                   "invalid byte range", "invalid_range");
      refused.headers.emplace_back("Content-Range",
                                   "bytes */" + std::to_string(bytes));
      refused.headers.emplace_back("Accept-Ranges", "bytes");
      return refused;
    }
    range = *parsed;
    partial = true;
  }
  const std::uint64_t count = bytes == 0 ? 0 : range.last - range.first + 1;
  HttpResponse response;
  if (!ReadFile(file, range.first, count, &response.body))
    return Error(503, "Service Unavailable", "video content is unavailable",
                 "video_content_unavailable");
  response.status = partial ? 206 : 200;
  response.reason = partial ? "Partial Content" : "OK";
  response.headers.emplace_back("Content-Type", media_type);
  response.headers.emplace_back("Accept-Ranges", "bytes");
  response.headers.emplace_back(
      "Content-Disposition",
      "attachment; filename=\"" + id + (format == "mp4" ? ".mp4\"" : ".ppm\""));
  if (partial)
    response.headers.emplace_back(
        "Content-Range", "bytes " + std::to_string(range.first) + "-" +
                             std::to_string(range.last) + "/" +
                             std::to_string(bytes));
  response.log_details = "job=" + id + " from=job_files bytes=" +
                         std::to_string(count);
  return response;
}

HttpResponse ObjektFile(const fs::path& canonical_root, const fs::path& dir,
                        const json::Value& job, const std::string& id,
                        const std::string& part) {
  if (job.member_str("status") != "completed")
    return Error(409, "Conflict", "3D generation is not finished",
                 "objekt_not_ready");
  const bool content = part == "content";
  const fs::path file = dir / (content ? "model.glb" : "preview.png");
  std::uintmax_t size = 0;
  if (!IsSafeFile(canonical_root, file, &size))
    return Error(404, "Not Found", "result file is missing",
                 "objekt_file_missing");
  HttpResponse response;
  if (!ReadFile(file, 0, size, &response.body))
    return Error(404, "Not Found", "result file is missing",
                 "objekt_file_missing");
  response.status = 200;
  response.reason = "OK";
  response.headers.emplace_back("Content-Type",
                                content ? "model/gltf-binary" : "image/png");
  response.headers.emplace_back(
      "Content-Disposition", content
                                 ? "attachment; filename=\"" + id + ".glb\""
                                 : "inline; filename=\"" + id + ".png\"");
  response.log_details = "job=" + id + " from=job_files file=" + part;
  return response;
}

}  // namespace

bool IsVideoId(std::string_view id) {
  return id.size() == 30 && id.starts_with("video_") &&
         IsLowerHex(id.substr(6));
}

bool IsObjektId(std::string_view id) {
  return id.size() == 28 && id.starts_with("obj_") && IsLowerHex(id.substr(4));
}

std::optional<Target> ParseRead(std::string_view method,
                                std::string_view path) {
  if (method != "GET")
    return std::nullopt;
  Target target;
  std::string_view rest;
  if (path.starts_with("/v1/videos/")) {
    target.kind = Kind::kVideo;
    rest = path.substr(std::string_view("/v1/videos/").size());
  } else if (path.starts_with("/v1/3d/generations/")) {
    target.kind = Kind::kObjekt;
    rest = path.substr(std::string_view("/v1/3d/generations/").size());
  } else {
    return std::nullopt;
  }
  const std::size_t slash = rest.find('/');
  std::string_view id = rest.substr(0, slash);
  if (slash != std::string_view::npos) {
    const std::string_view part = rest.substr(slash + 1);
    if (part == "content" ||
        (part == "preview" && target.kind == Kind::kObjekt)) {
      target.part = std::string(part);
    } else {
      return std::nullopt;
    }
  }
  if (target.kind == Kind::kVideo ? !IsVideoId(id) : !IsObjektId(id))
    return std::nullopt;
  target.id = std::string(id);
  return target;
}

std::optional<fs::path> JobRootOf(const PresetModel& preset, Kind kind) {
  const std::vector<std::string>& argv = preset.argv;
  bool serves = kind == Kind::kVideo && preset.modality == "video";
  for (std::size_t i = 0; i + 1 < argv.size() && !serves; ++i) {
    if (argv[i] == "serve")
      serves = argv[i + 1] == (kind == Kind::kVideo ? "video" : "objekt");
  }
  if (!serves)
    return std::nullopt;
  std::optional<fs::path> root;
  for (std::size_t i = 0; i < argv.size(); ++i) {
    // The last one counts, as for the worker's own option parser.
    if (argv[i] == "--root" && i + 1 < argv.size())
      root = fs::path(argv[i + 1]);
    else if (argv[i].starts_with("--root="))
      root = fs::path(argv[i].substr(7));
  }
  if (root.has_value() && root->empty())
    return std::nullopt;
  return root;
}

std::optional<HttpResponse> ReadFinished(const fs::path& root,
                                         const Target& target,
                                         const HttpRequest& request,
                                         long long now) {
  // The id is checked again here: it becomes a path component.
  if (target.kind == Kind::kVideo ? !IsVideoId(target.id)
                                  : !IsObjektId(target.id))
    return std::nullopt;
  std::error_code error;
  const fs::path canonical_root = fs::canonical(root, error);
  if (error)
    return std::nullopt;
  const fs::path dir = canonical_root / target.id;
  const fs::file_status dir_status = fs::symlink_status(dir, error);
  if (error || fs::is_symlink(dir_status) || !fs::is_directory(dir_status))
    return std::nullopt;
  const fs::path job_file = dir / "job.json";
  std::uintmax_t job_bytes = 0;
  if (!IsSafeFile(canonical_root, job_file, &job_bytes) ||
      job_bytes > kMaxJobJsonBytes)
    return std::nullopt;
  std::string text;
  if (!ReadFile(job_file, 0, job_bytes, &text))
    return std::nullopt;
  json::Value job;
  try {
    job = json::parse(text);
  } catch (const std::exception&) {
    return std::nullopt;  // being written or damaged: the worker decides
  }
  if (!job.is_object() || job.member_str("id") != target.id)
    return std::nullopt;
  const std::string schema = job.member_str("schema");
  if (schema != (target.kind == Kind::kVideo ? "gufo.video-job.v1"
                                             : "gufo.3d-api.v1"))
    return std::nullopt;
  const std::string status = job.member_str("status");
  if (status != "completed" && status != "failed" && status != "cancelled")
    return std::nullopt;  // queued or running: only the worker knows more
  // Expired: the worker sweeps it on its next request and answers 404.
  if (const long long expires_at = Seconds(job, "expires_at");
      expires_at > 0 && expires_at <= now)
    return std::nullopt;
  if (target.kind == Kind::kVideo) {
    if (target.part.empty())
      return Json(VideoStatus(job), target.id);
    return VideoContent(canonical_root, dir, job, target.id, request);
  }
  if (target.part.empty())
    return Json(job, target.id);
  return ObjektFile(canonical_root, dir, job, target.id, target.part);
}

HttpResponse NotFound(Kind kind) {
  if (kind == Kind::kVideo)
    return Error(404, "Not Found", "video job not found", "video_not_found");
  return Error(404, "Not Found", "unknown 3D generation", "objekt_not_found");
}

}  // namespace gufo::router::job_files
