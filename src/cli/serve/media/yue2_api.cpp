// Written with AI assistance (Claude Code).
#include "src/cli/serve/media/yue2_api.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <exception>
#include <system_error>
#include <utility>
#include <vector>

#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/media/music_api.hpp"
#include "src/cli/serve/music_encode.hpp"
#include "src/core/json.hpp"
#include "src/models/ace_step/audio_io.hpp"
#include "src/models/ace_step/vae.hpp"
#include "src/models/yue2/protocol.hpp"
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/yue2/pipeline.hpp"
#endif

namespace gufo::models::yue2 {
#if !defined(ENGINE_ENABLE_HIP)
// Without HIP the service never loads a pipeline.
class SongPipeline {};
#endif
}  // namespace gufo::models::yue2

namespace gufo::server {
namespace {

namespace y = models::yue2;

std::int64_t Now() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

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

std::string Lower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

// YuE2 has no language field; the style text names the language
// ("..., German"). A code or name is appended unless the style has it.
std::string WithLanguage(std::string style, const std::string& language) {
  static const std::pair<const char*, const char*> kNames[] = {
      {"de", "German"},  {"en", "English"},  {"zh", "Chinese"},
      {"ja", "Japanese"}, {"ko", "Korean"},  {"fr", "French"},
      {"es", "Spanish"}, {"it", "Italian"},  {"pt", "Portuguese"},
      {"ru", "Russian"}, {"nl", "Dutch"},    {"pl", "Polish"}};
  if (language.empty())
    return style;
  std::string name = language;
  for (const auto& [code, full] : kNames)
    if (Lower(language) == code)
      name = full;
  if (Lower(style).find(Lower(name)) != std::string::npos)
    return style;
  if (!style.empty())
    style += ", ";
  return style + name;
}

}  // namespace

struct Yue2MusicService::Job {
  std::string id;
  y::SongRequest request;
  std::string status{"queued"};
  std::string phase;
  double progress{0};
  std::string error;
  MusicOutputFormat format;
  std::string output;
  std::string abc;
  bool truncated{false};
  double audio_seconds{0};
  double generation_seconds{0};
  int semantic_tokens{0};
  std::int64_t created_at{0};
  std::int64_t completed_at{0};
  bool done{false};
};

bool IsYue2ModelDir(const std::filesystem::path& dir) {
  std::error_code ec;
  return std::filesystem::is_regular_file(dir / "qwen.tiktoken", ec) &&
         std::filesystem::is_regular_file(dir / "model.safetensors", ec);
}

bool Yue2MusicService::HandlesPath(std::string_view path) const {
  return IsMusicApiPath(path);
}

Yue2MusicService::Yue2MusicService(const std::filesystem::path& model_dir,
                                   const std::filesystem::path& vae_dir,
                                   std::string model_id)
    : model_id_(model_id.empty() ? "yue2-3b" : std::move(model_id)) {
  if (!IsYue2ModelDir(model_dir)) {
    init_error_ = "not a YuE2-3B directory (qwen.tiktoken, "
                  "model.safetensors): " +
                  model_dir.string();
    return;
  }
#if defined(ENGINE_ENABLE_HIP)
  std::unique_ptr<y::SongPipeline> loaded =
      y::SongPipeline::Load(model_dir, vae_dir, &init_error_);
  pipeline_ = std::move(loaded);
  if (pipeline_)
    worker_ = std::thread([this] { Worker(); });
#else
  (void)vae_dir;
  init_error_ = "YuE2 music needs a ROCm HIP build of gufo";
#endif
}

Yue2MusicService::~Yue2MusicService() {
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
  }
  changed_.notify_all();
  if (worker_.joinable())
    worker_.join();
}

void Yue2MusicService::Describe(json::Value* entry, bool /*ready*/) const {
  (*entry)["license"] =
      "weights CC BY-NC 4.0 + MODEL_LICENSE creator permission";
}

void Yue2MusicService::ExpireLocked() {
  const std::int64_t now = Now();
  std::vector<std::pair<std::int64_t, std::string>> finished;
  for (const auto& [id, job] : jobs_)
    if (job->done)
      finished.emplace_back(job->completed_at, id);
  std::sort(finished.begin(), finished.end());
  const std::size_t surplus =
      finished.size() > kKeptJobs ? finished.size() - kKeptJobs : 0;
  for (std::size_t i = 0; i < finished.size(); ++i)
    if (i < surplus || now - finished[i].first > 3600)
      jobs_.erase(finished[i].second);
}

void Yue2MusicService::Worker() {
  for (;;) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock lock(mutex_);
      changed_.wait(lock, [&] { return stop_ || !queue_.empty(); });
      if (stop_)
        return;
      job = queue_.front();
      queue_.pop_front();
      job->status = "in_progress";
    }
    std::string error, output;
    bool ok = false;
#if defined(ENGINE_ENABLE_HIP)
    y::SongResult result;
    ok = pipeline_->Generate(
        job->request, &result,
        [this, job](double fraction, std::string_view phase) {
          std::lock_guard lock(mutex_);
          job->progress = fraction;
          job->phase = std::string(phase);
          return !stop_;
        },
        &error);
    if (ok) {
      const std::string wav = models::ace_step::EncodeStereoWav(
          result.audio, models::ace_step::kSampleRate);
      ok = EncodeMusic(wav, job->format, &output, &error);
    }
    const std::string outcome = ok ? "ok" : "failed: " + error;
    Logger::Info(
        "music", "job=" + job->id + " yue2 " + outcome +
                     " audio_s=" +
                     std::to_string(result.audio.size() / 2 / 48000.0) +
                     " total_s=" + std::to_string(result.total_seconds) +
                     " abc_tokens=" + std::to_string(result.abc_tokens) +
                     " codes=" + std::to_string(result.semantic_tokens));
#endif
    {
      std::lock_guard lock(mutex_);
#if defined(ENGINE_ENABLE_HIP)
      job->abc = result.abc;
      job->truncated = result.semantic_truncated;
      job->audio_seconds = result.audio.size() / 2 / 48000.0;
      job->generation_seconds = result.total_seconds;
      job->semantic_tokens = result.semantic_tokens;
#endif
      job->status = ok ? "completed" : "failed";
      job->error = error;
      job->output = std::move(output);
      job->progress = 1.0;
      job->completed_at = Now();
      job->done = true;
    }
    changed_.notify_all();
  }
}

HttpResponse Yue2MusicService::Handle(const HttpRequest& request) {
  if (request.path == kMusicApiPath) {
    if (request.method != "POST")
      return Error(405, "Method Not Allowed", "use POST", "method_not_allowed");
    return Submit(request);
  }
  if (request.method != "GET")
    return Error(405, "Method Not Allowed", "use GET", "method_not_allowed");
  std::string rest(request.path.substr(kMusicApiPath.size() + 1));
  bool content = false;
  if (const auto slash = rest.find('/'); slash != std::string::npos) {
    const std::string verb = rest.substr(slash);
    rest.resize(slash);
    if (verb != "/content")
      return Error(404, "Not Found", "unknown music path", "not_found");
    content = true;
  }
  return Status(rest, content);
}

HttpResponse Yue2MusicService::Submit(const HttpRequest& request) {
  json::Value body;
  try {
    const std::string_view text = request.body;
    body = json::parse(std::string(
        text.substr(0, 3) == "\xEF\xBB\xBF" ? text.substr(3) : text));
  } catch (const std::exception&) {
    return Error(400, "Bad Request", "request body must be valid JSON",
                 "parse_error");
  }
  if (!body.is_object())
    return Error(400, "Bad Request", "request body must be a JSON object",
                 "invalid_request");
  auto job = std::make_shared<Job>();
  y::SongRequest& r = job->request;
  std::string style = body.member_str("tags");
  if (style.empty())
    style = body.member_str("prompt", body.member_str("style"));
  if (style.empty())
    return Error(400, "Bad Request", "tags (style: genre, instruments, voice) "
                 "is required", "missing_prompt");
  r.style = WithLanguage(style, body.member_str("language"));
  r.lyrics = body.member_str("lyrics");
  if (r.lyrics.size() > 64000 || r.style.size() > 4000)
    return Error(400, "Bad Request", "lyrics or tags too long",
                 "invalid_request");
  if (const std::string cot = body.member_str("cot", "full");
      !y::ParseCot(cot, &r.cot))
    return Error(400, "Bad Request", "cot must be full, melody or off",
                 "invalid_cot");
  r.abc = body.member_str("abc");
  if (!r.abc.empty() && r.cot == y::Cot::kOff)
    return Error(400, "Bad Request", "abc needs cot full or melody",
                 "invalid_abc");
  if (const json::Value* seed = body.find("seed");
      seed != nullptr && seed->is_number())
    r.seed = static_cast<std::uint64_t>(seed->as_double());
  const double duration = body.member_double("duration", 0.0);
  if (!std::isfinite(duration) || duration < 0 || duration > 360)
    return Error(400, "Bad Request", "duration must be 0..360 seconds",
                 "invalid_duration");
  r.max_seconds = duration;
  if (const json::Value* g = body.find("guidance_scale");
      g != nullptr && g->is_number()) {
    if (g->as_double() < 0 || g->as_double() > 20)
      return Error(400, "Bad Request", "guidance_scale must be 0..20",
                   "invalid_guidance");
    r.guidance = static_cast<float>(g->as_double());
  }
  if (const json::Value* steps = body.find("steps");
      steps != nullptr && steps->is_number()) {
    if (steps->as_double() < 1 || steps->as_double() > 128)
      return Error(400, "Bad Request", "steps must be 1..128",
                   "invalid_steps");
    r.ode_steps = static_cast<int>(steps->as_double());
  }
  {
    std::string message, code;
    if (!ParseMusicOutputFormat(body, &job->format, &message, &code))
      return Error(400, "Bad Request", std::move(message), std::move(code));
  }
  const json::Value* async_value = body.find("async");
  const bool async = async_value != nullptr && async_value->as_bool();
  {
    std::lock_guard lock(mutex_);
    ExpireLocked();
    if (queue_.size() >= kQueueCapacity)
      return Error(429, "Too Many Requests", "music queue is full",
                   "queue_full");
    job->id = "music_" + std::to_string(next_id_++);
    job->created_at = Now();
    jobs_[job->id] = job;
    queue_.push_back(job);
  }
  changed_.notify_all();
  if (async)
    return Status(job->id, false);
  {
    std::unique_lock lock(mutex_);
    while (!job->done && !(request.is_cancelled && request.is_cancelled()))
      changed_.wait_for(lock, std::chrono::seconds(1));
    if (!job->done)
      return Error(499, "Client Closed Request", "request cancelled",
                   "cancelled");
  }
  return Status(job->id, true);
}

HttpResponse Yue2MusicService::Status(const std::string& id, bool content) {
  std::lock_guard lock(mutex_);
  const auto it = jobs_.find(id);
  if (it == jobs_.end())
    return Error(404, "Not Found", "unknown music job " + id, "not_found");
  const Job& job = *it->second;
  if (content) {
    if (job.status == "failed")
      return Error(500, "Internal Server Error", job.error,
                   "generation_failed");
    if (!job.done)
      return Error(409, "Conflict", "music job is not finished", "not_ready");
    HttpResponse response;
    response.body = job.output;
    response.headers = {
        {"Content-Type", job.format.content_type},
        {"Content-Disposition",
         "attachment; filename=\"" + job.format.filename + "\""},
        {"X-Gufo-Music-Id", job.id},
        {"X-Gufo-Seed", std::to_string(job.request.seed)},
        {"X-Gufo-Generation-Seconds", std::to_string(job.generation_seconds)}};
    return response;
  }
  json::Value body = json::Value::object();
  body["id"] = job.id;
  body["object"] = "audio.music";
  body["model"] = model_id_;
  body["status"] = job.status;
  body["phase"] = job.phase;
  body["progress"] = static_cast<double>(std::lround(job.progress * 100));
  body["created_at"] = static_cast<double>(job.created_at);
  if (job.done)
    body["completed_at"] = static_cast<double>(job.completed_at);
  body["seed"] = std::to_string(job.request.seed);
  body["cot"] = std::string(y::CotName(job.request.cot));
  body["tags"] = job.request.style;
  body["steps"] = static_cast<double>(job.request.ode_steps);
  if (job.done) {
    body["duration"] = job.audio_seconds;
    body["generation_seconds"] = job.generation_seconds;
    body["codes"] = static_cast<double>(job.semantic_tokens);
    body["truncated"] = job.truncated;
    body["abc"] = job.abc;
  }
  body["response_format"] = job.format.name;
  if (!job.error.empty())
    body["error"] = job.error;
  body["preview_url"] = nullptr;
  return Json(job.done || job.status != "queued" ? 200 : 202,
              job.done ? "OK" : "Accepted", body);
}

}  // namespace gufo::server
