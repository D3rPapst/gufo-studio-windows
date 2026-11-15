#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winsock2.h>
#else
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iterator>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/router/comfy_adapter.hpp"
#include "src/cli/serve/router/comfy_node_info.hpp"
#include "src/cli/serve/router/comfy_preview.hpp"
#include "src/cli/serve/router/jobs.hpp"
#include "src/cli/serve/router/pool.hpp"
#include "src/cli/serve/router/remote.hpp"
#include "src/cli/serve/websocket.hpp"

namespace gufo::router::comfy {

namespace fs = std::filesystem;
using server::HttpRequest;
using server::HttpResponse;
using server::Logger;

struct Adapter::Job {
  std::string id;
  long long number{0};
  json::Value prompt;
  json::Value extra_data = json::Value::object();
  std::string client_id;
  /// The signed-in user who sent the prompt (X-Gufo-User from the gate);
  /// empty for the router itself, loopback or auth = off.
  std::string user;
  std::vector<std::string> outputs;
  Plan plan;
  std::atomic<bool> cancel{false};
  /// Warnings found while running (worker thread only).
  std::vector<std::string> warnings;
  /// Live preview ids (2.1, TaskPreviewId): extra_data.gufo_preview_id or
  /// the prompt id; several = more than one image or video task.
  std::string preview_base;
  bool several_previews{false};
};

namespace {

/// A job stopped by /interrupt (ComfyUI's execution_interrupted).
struct Interrupted : std::runtime_error {
  Interrupted() : std::runtime_error("interrupted") {}
};

constexpr std::string_view kWorkDir = ".gufo-work";

fs::path U8Path(std::string_view text) {
  return fs::path(std::u8string(text.begin(), text.end()));
}

std::string PathU8(const fs::path& path) {
  const std::u8string text = path.u8string();
  return std::string(text.begin(), text.end());
}

long long NowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string NewUuid() {
  static std::mutex mutex;
  static std::mt19937_64 engine{std::random_device{}() ^
                                static_cast<std::uint64_t>(NowMs())};
  std::uint64_t hi = 0;
  std::uint64_t lo = 0;
  {
    const std::lock_guard<std::mutex> lock(mutex);
    hi = engine();
    lo = engine();
  }
  hi = (hi & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;
  lo = (lo & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;
  char text[40];
  std::snprintf(text, sizeof(text), "%08x-%04x-%04x-%04x-%012llx",
                static_cast<unsigned>(hi >> 32),
                static_cast<unsigned>((hi >> 16) & 0xFFFF),
                static_cast<unsigned>(hi & 0xFFFF),
                static_cast<unsigned>(lo >> 48),
                static_cast<unsigned long long>(lo & 0xFFFFFFFFFFFFULL));
  return text;
}

HttpResponse JsonResponse(int status, const json::Value& body) {
  return HttpResponse{.status = status,
                      .reason = status == 200   ? "OK"
                                : status == 400 ? "Bad Request"
                                : status == 404 ? "Not Found"
                                : status == 405 ? "Method Not Allowed"
                                                : "Error",
                      .body = body.dump(),
                      .headers = {{"Content-Type", "application/json"}}};
}

HttpResponse ErrorResponse(int status, std::string_view type,
                           std::string_view message) {
  json::Value error = json::Value::object();
  error["type"] = std::string(type);
  error["message"] = std::string(message);
  error["details"] = "";
  error["extra_info"] = json::Value::object();
  json::Value body = json::Value::object();
  body["error"] = std::move(error);
  body["node_errors"] = json::Value::object();
  return JsonResponse(status, body);
}

HttpResponse EmptyOk() {
  return HttpResponse{.status = 200, .reason = "OK", .body = ""};
}

std::string ReadAll(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot read " + PathU8(path));
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

void WriteAll(const fs::path& path, std::string_view bytes) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!out)
    throw std::runtime_error("cannot write " + PathU8(path));
}

/// Creates `path` and writes `bytes` in one step, only if no file of that
/// name exists yet (fopen "wbx": O_CREAT | O_EXCL, CREATE_NEW on
/// Windows). False when it exists; throws on any other error. Never
/// check-then-write: two uploads of one name could both see it free.
bool WriteNew(const fs::path& path, std::string_view bytes) {
  fs::create_directories(path.parent_path());
#ifdef _WIN32
  std::FILE* file = nullptr;
  if (_wfopen_s(&file, path.c_str(), L"wbx") != 0)
    file = nullptr;
#else
  std::FILE* file = std::fopen(path.c_str(), "wbx");
#endif
  if (file == nullptr) {
    std::error_code ec;
    if (fs::exists(path, ec))
      return false;
    throw std::runtime_error("cannot create " + PathU8(path));
  }
  const bool written =
      std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
  const bool closed = std::fclose(file) == 0;
  if (!written || !closed)
    throw std::runtime_error("cannot write " + PathU8(path));
  return true;
}

/// Serialises the name choice of uploads in this process, so the "same
/// content: reuse" check never reads a file another upload is still
/// writing. WriteNew keeps it safe across processes as well.
std::mutex& UploadMutex() {
  static std::mutex mutex;
  return mutex;
}

std::string ContentTypeFor(std::string_view name) {
  const std::string ext = [&] {
    const auto dot = name.rfind('.');
    std::string e(dot == std::string_view::npos ? "" : name.substr(dot + 1));
    for (char& c : e)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return e;
  }();
  if (ext == "png")
    return "image/png";
  if (ext == "jpg" || ext == "jpeg")
    return "image/jpeg";
  if (ext == "webp")
    return "image/webp";
  if (ext == "gif")
    return "image/gif";
  if (ext == "flac")
    return "audio/flac";
  if (ext == "wav")
    return "audio/wav";
  if (ext == "mp3")
    return "audio/mpeg";
  if (ext == "ogg")
    return "audio/ogg";
  if (ext == "mp4")
    return "video/mp4";
  if (ext == "webm")
    return "video/webm";
  if (ext == "json")
    return "application/json";
  return "application/octet-stream";
}

/// "image/png" or "image/jpeg" from the leading bytes.
std::string SniffImageType(std::string_view bytes) {
  return bytes.size() > 2 && static_cast<unsigned char>(bytes[0]) == 0xFF &&
                 static_cast<unsigned char>(bytes[1]) == 0xD8
             ? "image/jpeg"
             : "image/png";
}

/// ComfyUI reports subfolders with the platform's separator.
std::string ReportedSubfolder(std::string subfolder) {
#if defined(_WIN32)
  for (char& c : subfolder)
    if (c == '/')
      c = '\\';
#endif
  return subfolder;
}

std::string ErrorMessage(const FetchResult& result) {
  try {
    const json::Value body = json::parse(result.body);
    if (const json::Value* error = body.find("error")) {
      if (error->is_string())
        return error->str();
      const std::string message = error->member_str("message");
      if (!message.empty())
        return message;
    }
  } catch (const std::exception&) {
  }
  return result.body.substr(0, 300);
}

#if defined(_WIN32)
void AppendArgument(std::wstring& command, const std::wstring& argument) {
  if (!command.empty())
    command.push_back(L' ');
  if (!argument.empty() &&
      argument.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
    command += argument;
    return;
  }
  command.push_back(L'"');
  std::size_t backslashes = 0;
  for (const wchar_t c : argument) {
    if (c == L'\\') {
      ++backslashes;
      continue;
    }
    if (c == L'"') {
      command.append(backslashes * 2 + 1, L'\\');
    } else {
      command.append(backslashes, L'\\');
    }
    backslashes = 0;
    command.push_back(c);
  }
  command.append(backslashes * 2, L'\\');
  command.push_back(L'"');
}

int RunProcess(const fs::path& exe, const std::vector<fs::path>& args,
               std::chrono::minutes timeout, std::string* error) {
  std::wstring command;
  AppendArgument(command, exe.wstring());
  for (const fs::path& arg : args)
    AppendArgument(command, arg.wstring());
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION process{};
  if (!CreateProcessW(exe.wstring().c_str(), command.data(), nullptr, nullptr,
                      FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                      &process)) {
    *error = "cannot start " + PathU8(exe) + " (error " +
             std::to_string(GetLastError()) + ")";
    return -1;
  }
  CloseHandle(process.hThread);
  const DWORD waited = WaitForSingleObject(
      process.hProcess,
      static_cast<DWORD>(
          std::chrono::duration_cast<std::chrono::milliseconds>(timeout)
              .count()));
  if (waited != WAIT_OBJECT_0) {
    TerminateProcess(process.hProcess, 1);
    WaitForSingleObject(process.hProcess, 5000);
    CloseHandle(process.hProcess);
    *error = PathU8(exe) + " timed out";
    return -1;
  }
  DWORD code = 1;
  GetExitCodeProcess(process.hProcess, &code);
  CloseHandle(process.hProcess);
  return static_cast<int>(code);
}

void MemoryStats(std::uint64_t* total, std::uint64_t* free) {
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status)) {
    *total = status.ullTotalPhys;
    *free = status.ullAvailPhys;
  }
}
#else
int RunProcess(const fs::path& exe, const std::vector<fs::path>& args,
               std::chrono::minutes timeout, std::string* error) {
  std::vector<std::string> storage{exe.string()};
  for (const fs::path& arg : args)
    storage.push_back(arg.string());
  std::vector<char*> argv;
  for (std::string& item : storage)
    argv.push_back(item.data());
  argv.push_back(nullptr);
  const pid_t pid = fork();
  if (pid < 0) {
    *error = "fork failed";
    return -1;
  }
  if (pid == 0) {
    execv(argv[0], argv.data());
    _exit(127);
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    int status = 0;
    const pid_t done = waitpid(pid, &status, WNOHANG);
    if (done == pid)
      return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    if (std::chrono::steady_clock::now() > deadline) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      *error = exe.string() + " timed out";
      return -1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void MemoryStats(std::uint64_t* total, std::uint64_t* free) {
  std::ifstream in("/proc/meminfo");
  std::string key;
  std::uint64_t value = 0;
  std::string unit;
  while (in >> key >> value >> unit) {
    if (key == "MemTotal:")
      *total = value * 1024;
    else if (key == "MemAvailable:")
      *free = value * 1024;
  }
}
#endif

void CollectFiles(const Plan& plan, std::vector<FileRef>* files) {
  for (const Task& task : plan.tasks) {
    if (const auto* image = std::get_if<ImageTask>(&task.spec)) {
      files->insert(files->end(), image->references.begin(),
                    image->references.end());
      if (image->size_source == SizeSource::kCanvas)
        files->push_back(image->canvas);
      if (image->strength)
        files->push_back(image->init_image);
      for (const ImageRegion& region : image->regions) {
        files->insert(files->end(), region.references.begin(),
                      region.references.end());
        if (region.mask_file)
          files->push_back(*region.mask_file);
      }
      if (image->control && !image->control->image.name.empty())
        files->push_back(image->control->image);
    } else if (const auto* speech = std::get_if<SpeechTask>(&task.spec)) {
      if (speech->reference_audio)
        files->push_back(*speech->reference_audio);
    } else if (const auto* dialogue = std::get_if<DialogueTask>(&task.spec)) {
      for (const SpeechTask& line : dialogue->lines)
        if (line.reference_audio)
          files->push_back(*line.reference_audio);
    } else if (const auto* video = std::get_if<VideoTask>(&task.spec)) {
      if (video->first_frame)
        files->push_back(*video->first_frame);
      if (video->last_frame)
        files->push_back(*video->last_frame);
      for (const auto& [frame, ref] : video->keyframes)
        files->push_back(ref);
      if (video->audio_reference)
        files->push_back(*video->audio_reference);
      if (video->render_video)
        files->push_back(*video->render_video);
      if (video->source_video)
        files->push_back(*video->source_video);
      if (video->guide_video)
        files->push_back(*video->guide_video);
      if (video->control_video)
        files->push_back(*video->control_video);
      if (video->talk_video)
        files->push_back(*video->talk_video);
      if (video->vace_mask)
        files->push_back(*video->vace_mask);
      if (video->h3_control_mask)
        files->push_back(*video->h3_control_mask);
      if (video->h3_control_source)
        files->push_back(*video->h3_control_source);
      files->insert(files->end(), video->reference_images.begin(),
                    video->reference_images.end());
    } else if (const auto* foley = std::get_if<FoleyTask>(&task.spec)) {
      if (foley->video)
        files->push_back(*foley->video);
      if (foley->image)
        files->push_back(*foley->image);
    }
  }
}

}  // namespace

Adapter::Adapter(AdapterOptions options) : options_(std::move(options)) {
  for (const char* folder : {"input", "output", "temp"})
    fs::create_directories(options_.root / folder);
  LoadHistory();
}

Adapter::~Adapter() {
  Stop();
}

void Adapter::Start() {
  const std::size_t threads = std::max<std::size_t>(1, options_.parallel);
  for (std::size_t i = 0; i < threads; ++i)
    workers_.emplace_back([this] { WorkerMain(); });
}

void Adapter::Stop() {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    for (const auto& job : running_)
      job->cancel.store(true);
  }
  wake_.notify_all();
  for (std::thread& worker : workers_)
    if (worker.joinable())
      worker.join();
  workers_.clear();
}

void Adapter::SetPartner(std::string name, Fetch fetch) {
  partner_name_ = std::move(name);
  partner_fetch_ = std::move(fetch);
}

namespace {

/// GET /queue?pool=1 (or scope=pool) asks for the pool view; a request
/// another router relayed is always answered locally, so the two routers
/// cannot ask each other in a loop.
bool WantsPoolView(const HttpRequest& request) {
  if (!request.header(jobs::kRelayHeader).empty() ||
      !request.header(kForwardedHeader).empty())
    return false;
  const std::string pool = request.query_param("pool");
  return pool == "1" || pool == "true" ||
         request.query_param("scope") == "pool";
}

/// A queue item ([number, id, prompt, extra_data, outputs]) with
/// extra_data.gufo_router = `router` ("local" or the partner's URL).
json::Value TagQueueItem(const json::Value& item, const std::string& router) {
  if (!item.is_array())
    return item;
  json::Value out = json::Value::array();
  std::size_t index = 0;
  for (const json::Value& part : item.items()) {
    if (index++ == 3 && part.is_object()) {
      json::Value extra = part;
      extra["gufo_router"] = router;
      out.push_back(std::move(extra));
    } else {
      out.push_back(part);
    }
  }
  return out;
}

}  // namespace

std::optional<json::Value> Adapter::PartnerQueue(std::string* state) {
  if (!partner_fetch_) {
    *state = "none";
    return std::nullopt;
  }
  {
    const std::lock_guard<std::mutex> lock(partner_mutex_);
    if (std::chrono::steady_clock::now() < partner_retry_at_) {
      *state = "unreachable";
      return std::nullopt;
    }
  }
  HttpRequest ask;
  ask.method = "GET";
  ask.path = "/queue";
  ask.request_id = "comfy-pool-queue";
  ask.headers.emplace_back(jobs::kRelayHeader, "1");
  FetchResult answer;
  try {
    answer = partner_fetch_(ask);
  } catch (const std::exception&) {
    answer.status = 0;
  }
  if (answer.status == 200) {
    try {
      json::Value body = json::parse(answer.body);
      if (body.is_object()) {
        *state = partner_name_;
        return body;
      }
    } catch (const std::exception&) {
    }
  }
  if (answer.status == 404 || answer.status == 405) {
    *state = "no_comfy";  // the partner runs no --comfy-dir
    return std::nullopt;
  }
  *state = "unreachable";
  const std::lock_guard<std::mutex> lock(partner_mutex_);
  // One slow answer per 10 s at most while the partner is gone.
  partner_retry_at_ =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  return std::nullopt;
}

std::optional<HttpResponse> Adapter::Handle(const HttpRequest& request) {
  std::string_view path = request.path;
  if (path.starts_with("/api/"))
    path.remove_prefix(4);
  const std::string& method = request.method;
  const auto only =
      [&](std::string_view allowed) -> std::optional<HttpResponse> {
    if (method == allowed)
      return std::nullopt;
    return ErrorResponse(405, "method_not_allowed", "method is not allowed");
  };
  try {
    if (path == "/prompt") {
      if (method == "POST")
        return Submit(request);
      if (auto bad = only("GET"))
        return bad;
      return PromptGet(request);
    }
    if (path == "/queue")
      return method == "POST" ? QueuePost(request) : QueueGet(request);
    if (path == "/history")
      return method == "POST" ? HistoryPost(request) : HistoryGet(request, "");
    if (path.starts_with("/history/")) {
      if (auto bad = only("GET"))
        return bad;
      return HistoryGet(request, path.substr(9));
    }
    if (path == "/interrupt") {
      if (auto bad = only("POST"))
        return bad;
      return Interrupt(request);
    }
    if (path == "/free") {
      if (auto bad = only("POST"))
        return bad;
      return Free(request);
    }
    if (path == "/view") {
      if (auto bad = only("GET"))
        return bad;
      return View(request);
    }
    if (path == "/upload/image") {
      if (auto bad = only("POST"))
        return bad;
      return Upload(request);
    }
    if (path == "/upload/mask")
      return ErrorResponse(400, "unsupported",
                           "mask uploads are not supported by gufo");
    if (path == "/system_stats")
      return SystemStats();
    if (path == "/object_info")
      return ObjectInfo("");
    if (path.starts_with("/object_info/"))
      return ObjectInfo(path.substr(13));
    // ComfyUI's /embeddings is a GET list of textual-inversion names. A
    // POST to the bare /embeddings is the llama-server embeddings API and
    // routes to an embedding model by its JSON `model` instead.
    if (path == "/embeddings" && method != "GET" &&
        request.path == "/embeddings")
      return std::nullopt;
    if (path == "/embeddings" || path == "/extensions")
      return JsonResponse(200, json::Value::array());
    if (path == "/features")
      return JsonResponse(200, json::Value::object());
    if (path == "/ws")
      return Socket(request);
    if (path == "/filmstudio/progress") {
      if (auto bad = only("GET"))
        return bad;
      return FilmstudioProgress(request);
    }
    if (path == "/filmstudio/preview") {
      if (auto bad = only("GET"))
        return bad;
      return FilmstudioPreview(request);
    }
  } catch (const std::exception& error) {
    Logger::Warn("comfy", std::string("event=request_failed path=") +
                              request.path + " error=" + error.what());
    return ErrorResponse(500, "server_error", error.what());
  }
  return std::nullopt;
}

// The caller the gate names (auth.hpp kUserHeader/kRightsHeader): empty
// when it is no user (router, loopback, auth = off) or an admin; such a
// caller may stop or remove any prompt, a user only its own.
std::string RestrictedUser(const HttpRequest& request) {
  const std::string user = request.header("X-Gufo-User");
  if (user.empty())
    return {};
  const std::string rights = request.header("X-Gufo-Rights");
  if (rights == "rolle=admin" || rights.starts_with("rolle=admin;"))
    return {};
  return user;
}

HttpResponse Adapter::Submit(const HttpRequest& request) {
  json::Value body;
  try {
    body = json::parse(request.body);
  } catch (const std::exception& error) {
    return ErrorResponse(400, "invalid_prompt",
                         std::string("invalid JSON: ") + error.what());
  }
  const json::Value* prompt = body.find("prompt");
  if (prompt == nullptr || !prompt->is_object())
    return ErrorResponse(400, "no_prompt", "No prompt provided");
  auto job = std::make_shared<Job>();
  job->prompt = *prompt;
  job->client_id = body.member_str("client_id");
  job->user = request.header("X-Gufo-User");
  job->id = body.member_str("prompt_id");
  if (job->id.empty())
    job->id = NewUuid();
  if (const json::Value* extra = body.find("extra_data");
      extra != nullptr && extra->is_object())
    job->extra_data = *extra;
  // The live preview id the client may watch (GET /v1/previews/<id>/events)
  // from the moment /prompt answers: its own, or the prompt id.
  if (const json::Value* wanted = job->extra_data.find("gufo_preview_id");
      wanted != nullptr && !wanted->is_null()) {
    if (!wanted->is_string() || !IsPreviewId(wanted->str()))
      return ErrorResponse(400, "invalid_prompt",
                           "extra_data.gufo_preview_id must be 1..128 "
                           "characters [A-Za-z0-9._-]");
    job->preview_base = wanted->str();
  } else {
    job->preview_base = job->id;
  }
  if (!job->client_id.empty())
    job->extra_data["client_id"] = job->client_id;
  job->extra_data["create_time"] = NowMs();
  PlanError plan_error;
  if (!BuildPlan(job->prompt, options_.models, &job->plan, &plan_error,
                 [this](const FileRef& ref) { return ProbeSourceFps(ref); })) {
    Logger::Warn("comfy", "event=prompt_rejected type=" + plan_error.type +
                              " message=" + plan_error.message);
    json::Value error = json::Value::object();
    error["type"] = plan_error.type;
    error["message"] = plan_error.message;
    error["details"] = plan_error.details;
    error["extra_info"] = json::Value::object();
    json::Value reply = json::Value::object();
    reply["error"] = std::move(error);
    reply["node_errors"] = plan_error.node_errors;
    return JsonResponse(400, reply);
  }
  // ComfyUI rejects missing input files at validation, before queueing.
  std::vector<FileRef> files;
  CollectFiles(job->plan, &files);
  for (const FileRef& file : files) {
    std::error_code ec;
    const fs::path path = FilePath(file);
    if (path.empty() || !fs::is_regular_file(path, ec))
      return ErrorResponse(400, "prompt_outputs_failed_validation",
                           "Invalid file: " + file.name + " [" + file.folder +
                               "] (upload it with /upload/image first)");
  }
  for (const Task& task : job->plan.tasks)
    for (const OutputTarget& target : task.outputs)
      job->outputs.push_back(target.node_id);
  job->several_previews =
      std::count_if(job->plan.tasks.begin(), job->plan.tasks.end(),
                    [](const Task& task) {
                      return std::holds_alternative<ImageTask>(task.spec) ||
                             std::holds_alternative<VideoTask>(task.spec);
                    }) > 1;
  if (!RegisterPreviews(job))
    return ErrorResponse(400, "invalid_prompt",
                         "extra_data.gufo_preview_id is in use by another "
                         "user's prompt");
  json::Value reply = json::Value::object();
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_)
      return ErrorResponse(503, "server_error", "adapter is shutting down");
    job->number = next_number_++;
    if (const json::Value* number = body.find("number");
        number != nullptr && number->is_number())
      job->number = static_cast<long long>(number->as_double());
    if (body.find("front") != nullptr && body.find("front")->as_bool())
      pending_.push_front(job);
    else
      pending_.push_back(job);
    reply["prompt_id"] = job->id;
    reply["number"] = job->number;
    reply["node_errors"] = json::Value::object();
    EmitStatusLocked();
  }
  wake_.notify_all();
  std::string families;
  for (const Task& task : job->plan.tasks)
    families += (families.empty() ? "" : ",") + task.class_type;
  Logger::Info("comfy", "event=prompt_queued prompt_id=" + job->id +
                            " tasks=" + families);
  return JsonResponse(200, reply);
}

json::Value Adapter::QueueItemLocked(const Job& job) const {
  json::Value item = json::Value::array();
  item.push_back(job.number);
  item.push_back(job.id);
  item.push_back(job.prompt);
  item.push_back(job.extra_data);
  json::Value outputs = json::Value::array();
  for (const std::string& id : job.outputs)
    outputs.push_back(id);
  item.push_back(std::move(outputs));
  return item;
}

int Adapter::QueueRemainingLocked() const {
  return static_cast<int>(pending_.size() + running_.size());
}

HttpResponse Adapter::QueueGet(const HttpRequest& request) {
  const bool pool = WantsPoolView(request);
  // The partner first, without the lock (up to a few seconds).
  std::string partner_state;
  std::optional<json::Value> partner;
  if (pool)
    partner = PartnerQueue(&partner_state);
  json::Value running = json::Value::array();
  json::Value pending = json::Value::array();
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& job : running_)
      running.push_back(pool ? TagQueueItem(QueueItemLocked(*job), "local")
                             : QueueItemLocked(*job));
    for (const auto& job : pending_)
      pending.push_back(pool ? TagQueueItem(QueueItemLocked(*job), "local")
                             : QueueItemLocked(*job));
  }
  if (partner.has_value()) {
    if (const json::Value* items = partner->find("queue_running");
        items != nullptr && items->is_array())
      for (const json::Value& item : items->items())
        running.push_back(TagQueueItem(item, partner_name_));
    if (const json::Value* items = partner->find("queue_pending");
        items != nullptr && items->is_array())
      for (const json::Value& item : items->items())
        pending.push_back(TagQueueItem(item, partner_name_));
  }
  json::Value body = json::Value::object();
  body["queue_running"] = std::move(running);
  body["queue_pending"] = std::move(pending);
  if (pool)
    body["partner"] = partner_state;
  return JsonResponse(200, body);
}

HttpResponse Adapter::PromptGet(const HttpRequest& request) {
  const bool pool = WantsPoolView(request);
  std::string partner_state;
  std::optional<json::Value> partner;
  if (pool)
    partner = PartnerQueue(&partner_state);
  int remaining = 0;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    remaining = QueueRemainingLocked();
  }
  if (partner.has_value()) {
    for (const char* key : {"queue_running", "queue_pending"})
      if (const json::Value* items = partner->find(key);
          items != nullptr && items->is_array())
        remaining += static_cast<int>(items->size());
  }
  json::Value info = json::Value::object();
  info["queue_remaining"] = remaining;
  json::Value body = json::Value::object();
  body["exec_info"] = std::move(info);
  if (pool)
    body["partner"] = partner_state;
  return JsonResponse(200, body);
}

HttpResponse Adapter::QueuePost(const HttpRequest& request) {
  json::Value body = json::Value::object();
  if (!request.body.empty())
    body = json::parse(request.body);
  // A signed-in user clears or deletes only its own waiting prompts.
  const std::string user = RestrictedUser(request);
  const auto mine = [&](const std::shared_ptr<Job>& job) {
    return user.empty() || job->user == user;
  };
  const std::lock_guard<std::mutex> lock(mutex_);
  if (body.find("clear") != nullptr && body.find("clear")->as_bool())
    std::erase_if(pending_, mine);
  if (const json::Value* remove = body.find("delete");
      remove != nullptr && remove->is_array()) {
    for (const json::Value& id : remove->items())
      std::erase_if(pending_, [&](const auto& job) {
        return id.is_string() && job->id == id.str() && mine(job);
      });
  }
  EmitStatusLocked();
  return EmptyOk();
}

HttpResponse Adapter::HistoryGet(const HttpRequest& request,
                                 std::string_view id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  json::Value body = json::Value::object();
  if (!id.empty()) {
    if (const auto found = history_.find(std::string(id));
        found != history_.end())
      body[found->first] = found->second;
    return JsonResponse(200, body);
  }
  std::size_t limit = history_order_.size();
  if (const std::string max_items = request.query_param("max_items");
      !max_items.empty())
    limit = std::min<std::size_t>(
        limit, std::strtoull(max_items.c_str(), nullptr, 10));
  for (std::size_t i = history_order_.size() - limit; i < history_order_.size();
       ++i)
    body[history_order_[i]] = history_.at(history_order_[i]);
  return JsonResponse(200, body);
}

HttpResponse Adapter::HistoryPost(const HttpRequest& request) {
  json::Value body = json::Value::object();
  if (!request.body.empty())
    body = json::parse(request.body);
  // A signed-in user who is no admin clears or deletes only the entries of
  // its own prompts (like POST /queue); admin and loopback touch all.
  const std::string user = RestrictedUser(request);
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto mine = [&](const std::string& id) {
    if (user.empty())
      return true;
    const auto owner = history_users_.find(id);
    return owner != history_users_.end() && owner->second == user;
  };
  const auto erase = [&](const std::string& id) {
    history_.erase(id);
    history_users_.erase(id);
    std::erase(history_order_, id);
  };
  if (body.find("clear") != nullptr && body.find("clear")->as_bool()) {
    std::vector<std::string> ids;
    for (const std::string& id : history_order_)
      if (mine(id))
        ids.push_back(id);
    for (const std::string& id : ids)
      erase(id);
  }
  if (const json::Value* remove = body.find("delete");
      remove != nullptr && remove->is_array()) {
    for (const json::Value& id : remove->items())
      if (id.is_string() && history_.count(id.str()) != 0 && mine(id.str()))
        erase(id.str());
  }
  PersistLocked();
  return EmptyOk();
}

HttpResponse Adapter::Interrupt(const HttpRequest& request) {
  std::string wanted;
  if (!request.body.empty()) {
    try {
      wanted = json::parse(request.body).member_str("prompt_id");
    } catch (const std::exception&) {
    }
  }
  // Without a prompt id every running prompt stops (ComfyUI's global
  // interrupt); with one only that prompt. A signed-in user who is no
  // admin stops only its own prompts either way.
  const std::string user = RestrictedUser(request);
  const std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& job : running_) {
    if (!wanted.empty() && wanted != job->id)
      continue;
    if (!user.empty() && job->user != user)
      continue;
    job->cancel.store(true);
    Logger::Info("comfy", "event=interrupt prompt_id=" + job->id);
  }
  return EmptyOk();
}

HttpResponse Adapter::Free(const HttpRequest& request) {
  bool unload = true;
  if (!request.body.empty()) {
    try {
      const json::Value body = json::parse(request.body);
      unload = body.find("unload_models") != nullptr
                   ? body.find("unload_models")->as_bool()
                   : body.find("free_memory") != nullptr &&
                         body.find("free_memory")->as_bool();
    } catch (const std::exception&) {
    }
  }
  if (unload && options_.unload) {
    Logger::Info("comfy", "event=free unload=idle_workers");
    options_.unload();
  }
  return EmptyOk();
}

fs::path Adapter::FilePath(const FileRef& ref) const {
  if (ref.folder != "input" && ref.folder != "output" && ref.folder != "temp")
    return {};
  const auto relative = SafeRelative(ref.name);
  if (!relative || relative->empty())
    return {};
  return options_.root / ref.folder / U8Path(*relative);
}

std::string Adapter::ReadFileRef(const FileRef& ref) const {
  const fs::path path = FilePath(ref);
  std::error_code ec;
  if (path.empty() || !fs::is_regular_file(path, ec))
    throw std::runtime_error("file not found: " + ref.name + " [" + ref.folder +
                             "]");
  return ReadAll(path);
}

double Adapter::ProbeSourceFps(const FileRef& ref) const {
  if (options_.ffprobe.empty())
    return 0;
  const fs::path path = FilePath(ref);
  std::error_code ec;
  if (path.empty() || !fs::is_regular_file(path, ec))
    return 0;  // the missing file is reported after the plan
  std::string error;
  const double fps = jobs::ProbeFrameRate(options_.ffprobe, path, &error);
  if (fps > 0) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.3f", fps);
    Logger::Info("comfy", "event=source_fps file=" + ref.name + " fps=" + text);
  } else {
    Logger::Warn("comfy", "event=source_fps_unknown file=" + ref.name +
                              " error=" + error + " (24 fps assumed)");
  }
  return fps;
}

std::optional<std::string> Adapter::PromptOfFile(const std::string& filename,
                                                 const std::string& subfolder,
                                                 const std::string& type) {
  if (filename.empty())
    return std::nullopt;
  const std::string want_type = type.empty() ? std::string("output") : type;
  // History entries report the subfolder with the platform's separator;
  // a client may ask with either.
  const std::string want_sub = ReportedSubfolder(subfolder);
  const std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = history_order_.rbegin(); it != history_order_.rend(); ++it) {
    const auto entry = history_.find(*it);
    if (entry == history_.end())
      continue;
    const json::Value* outputs = entry->second.find("outputs");
    if (outputs == nullptr || !outputs->is_object())
      continue;
    for (const auto& [node, lists] : outputs->members()) {
      if (!lists.is_object())
        continue;
      for (const auto& [kind, files] : lists.members()) {
        if (!files.is_array())
          continue;
        for (const json::Value& file : files.items()) {
          if (file.is_object() && file.member_str("filename") == filename &&
              ReportedSubfolder(file.member_str("subfolder")) == want_sub &&
              file.member_str("type", "output") == want_type)
            return *it;
        }
      }
    }
  }
  return std::nullopt;
}

HttpResponse Adapter::View(const HttpRequest& request) {
  const std::string filename = request.query_param("filename");
  std::string type = request.query_param("type");
  if (type.empty())
    type = "output";
  const std::string subfolder = request.query_param("subfolder");
  if (filename.empty())
    return ErrorResponse(400, "invalid_request", "filename is required");
  FileRef ref{subfolder.empty() ? filename : subfolder + "/" + filename, type};
  const fs::path path = FilePath(ref);
  if (path.empty())
    return ErrorResponse(400, "invalid_request", "invalid file path");
  std::error_code ec;
  if (!fs::is_regular_file(path, ec))
    return ErrorResponse(404, "not_found", "file not found");
  return HttpResponse{
      .status = 200,
      .reason = "OK",
      .body = ReadAll(path),
      .headers = {{"Content-Type", ContentTypeFor(filename)},
                  {"Content-Disposition",
                   "filename=\"" + PathU8(path.filename()) + "\""},
                  {"Cache-Control", "no-cache"}}};
}

HttpResponse Adapter::Upload(const HttpRequest& request) {
  std::vector<MultipartPart> parts;
  std::string error;
  if (!ParseMultipart(request.header("content-type"), request.body, &parts,
                      &error))
    return ErrorResponse(400, "invalid_request", error);
  const MultipartPart* image = nullptr;
  std::string subfolder;
  std::string type = "input";
  bool overwrite = false;
  for (const MultipartPart& part : parts) {
    if (part.name == "image")
      image = &part;
    else if (part.name == "subfolder")
      subfolder = part.data;
    else if (part.name == "type")
      type = part.data;
    else if (part.name == "overwrite")
      overwrite = part.data == "true" || part.data == "1";
  }
  if (image == nullptr || image->filename.empty())
    return ErrorResponse(400, "invalid_request", "image file is required");
  if (type != "input" && type != "output" && type != "temp")
    return ErrorResponse(400, "invalid_request", "invalid type");
  const auto safe_sub = SafeRelative(subfolder);
  std::string base = image->filename;
  if (const auto slash = base.find_last_of("/\\"); slash != std::string::npos)
    base = base.substr(slash + 1);
  if (!safe_sub || !SafeRelative(base) || base.empty() || base == "." ||
      base == "..")
    return ErrorResponse(400, "invalid_request", "invalid file name");
  const fs::path folder = options_.root / type / U8Path(*safe_sub);
  std::string name = base;
  if (overwrite) {
    WriteAll(folder / U8Path(name), image->data);
  } else {
    // Name and write in one step (Regie #56-986, 15.11.): 20 parallel
    // uploads of "name.png" with different images all got "name.png"
    // back and the last writer won. Each try creates the file
    // exclusively; an existing one with the same bytes is reused, as
    // ComfyUI does, else the next "name (n).png" is tried.
    const auto dot = base.rfind('.');
    const std::string stem =
        dot == std::string::npos ? base : base.substr(0, dot);
    const std::string ext = dot == std::string::npos ? "" : base.substr(dot);
    const std::lock_guard<std::mutex> lock(UploadMutex());
    for (int i = 1;; ++i) {
      const fs::path path = folder / U8Path(name);
      if (WriteNew(path, image->data))
        break;
      if (ReadAll(path) == image->data)
        break;  // same content: reuse
      if (i > 10000)
        return ErrorResponse(500, "upload_failed",
                             "no free file name for " + base);
      name = stem + " (" + std::to_string(i) + ")" + ext;
    }
  }
  Logger::Info("comfy", "event=upload name=" + name +
                            " bytes=" + std::to_string(image->data.size()));
  json::Value body = json::Value::object();
  body["name"] = name;
  body["subfolder"] = *safe_sub;
  body["type"] = type;
  return JsonResponse(200, body);
}

HttpResponse Adapter::SystemStats() {
  std::uint64_t total = 0;
  std::uint64_t free = 0;
  MemoryStats(&total, &free);
  json::Value system = json::Value::object();
#if defined(_WIN32)
  system["os"] = "win32";
#else
  system["os"] = "posix";
#endif
  system["ram_total"] = total;
  system["ram_free"] = free;
  system["comfyui_version"] = "gufo-comfy-adapter";
  system["python_version"] = "";
  system["pytorch_version"] = "";
  system["embedded_python"] = false;
  system["argv"] = json::Value::array();
  json::Value device = json::Value::object();
  device["name"] = "gufo router (shared memory)";
  device["type"] = "gufo";
  device["index"] = 0;
  device["vram_total"] = total;
  device["vram_free"] = free;
  device["torch_vram_total"] = 0;
  device["torch_vram_free"] = 0;
  json::Value devices = json::Value::array();
  devices.push_back(std::move(device));
  json::Value body = json::Value::object();
  body["system"] = std::move(system);
  body["devices"] = std::move(devices);
  return JsonResponse(200, body);
}

HttpResponse Adapter::ObjectInfo(std::string_view class_type) {
  const auto files = [&](const char* folder) {
    json::Value list = json::Value::array();
    const fs::path base = options_.root / folder;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(base, ec), end; !ec && it != end;
         it.increment(ec)) {
      if (!it->is_regular_file(ec))
        continue;
      const std::u8string text =
          fs::relative(it->path(), base, ec).generic_u8string();
      list.push_back(std::string(text.begin(), text.end()));
    }
    return list;
  };
  json::Value body = json::Value::object();
  for (const std::string& cls : SupportedClassTypes()) {
    if (!class_type.empty() && cls != class_type)
      continue;
    if (auto info = NodeInfo(cls)) {
      body[cls] = std::move(*info);
      continue;
    }
    json::Value required = json::Value::object();
    if (cls == "LoadImage" || cls == "LoadAudio") {
      json::Value spec = json::Value::array();
      spec.push_back(files("input"));
      json::Value upload = json::Value::object();
      upload[cls == "LoadImage" ? "image_upload" : "audio_upload"] = true;
      spec.push_back(std::move(upload));
      required[cls == "LoadImage" ? "image" : "audio"] = std::move(spec);
    }
    json::Value input = json::Value::object();
    input["required"] = std::move(required);
    json::Value info = json::Value::object();
    info["input"] = std::move(input);
    info["output"] = json::Value::array();
    info["output_is_list"] = json::Value::array();
    info["output_name"] = json::Value::array();
    info["name"] = cls;
    info["display_name"] = cls;
    info["description"] = "served by gufo";
    info["python_module"] = "gufo";
    info["category"] = "gufo";
    info["output_node"] = IsOutputClass(cls);
    body[cls] = std::move(info);
  }
  return JsonResponse(200, body);
}

void Adapter::Emit(const std::string& client, std::string_view type,
                   json::Value data) {
  json::Value message = json::Value::object();
  message["type"] = std::string(type);
  const bool progress = type == "progress";
  const int value = progress ? static_cast<int>(data.member_double("value", 0)) : 0;
  const int max = progress ? static_cast<int>(data.member_double("max", 0)) : 0;
  const std::string prompt_id = progress ? data.member_str("prompt_id") : "";
  message["data"] = std::move(data);
  const std::lock_guard<std::mutex> lock(mutex_);
  if (progress) {
    if (const auto live = live_.find(prompt_id); live != live_.end()) {
      live->second.value = value;
      live->second.max = max;
      live->second.has_progress = true;
    }
  }
  events_.push_back(Event{next_event_++, client, message.dump()});
  while (events_.size() > 4096)
    events_.pop_front();
}

bool Adapter::WatchedLocked(const std::string& client) const {
  if (std::chrono::steady_clock::now() < http_watch_until_)
    return true;
  if (client.empty())
    return !sockets_.empty();
  const auto it = sockets_.find(client);
  return it != sockets_.end() && it->second > 0;
}

void Adapter::EmitPreview(const std::string& client,
                          const std::string& prompt_id,
                          const std::string& node_id, std::string jpeg) {
  const std::lock_guard<std::mutex> lock(mutex_);
  previews_[client] =
      Preview{next_preview_++, prompt_id, node_id, std::move(jpeg)};
  last_preview_ = previews_[client];
  if (live_.contains(prompt_id))
    prompt_previews_[prompt_id] = last_preview_;
}

void Adapter::RelayPreviews(const std::shared_ptr<Job>& job, const Task& task,
                            const std::string& preview_id,
                            const std::string& model, bool image_progress,
                            std::stop_token stop) {
  double seq = 0;
  while (!stop.stop_requested()) {
    // Short sleep slices: the end of the task stops the relay at once.
    const auto until = std::chrono::steady_clock::now() + options_.preview_poll;
    while (!stop.stop_requested() && std::chrono::steady_clock::now() < until)
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (stop.stop_requested())
      break;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (!WatchedLocked(job->client_id))
        continue;  // nobody looks: the worker renders no previews either
    }
    HttpRequest poll;
    poll.method = "GET";
    poll.path = "/v1/previews/" + preview_id;
    poll.query = "format=frame-json";
    if (!model.empty())
      poll.query += "&model=" + model;
    poll.request_id = "comfy-preview-" + job->id;
    FetchResult result;
    try {
      result = options_.fetch(poll);
    } catch (const std::exception&) {
      continue;
    }
    if (result.status != 200)
      continue;
    try {
      const json::Value parsed = json::parse(result.body);
      const double next = parsed.member_double("seq", 0);
      if (next <= seq || !parsed.contains("data"))
        continue;
      seq = next;
      auto jpeg = Base64Decode(parsed.member_str("data"));
      if (!jpeg || jpeg->empty())
        continue;
      EmitPreview(job->client_id, job->id, task.node_id, std::move(*jpeg));
      if (image_progress) {
        // Images have no other progress source: the preview's step.
        json::Value data = json::Value::object();
        data["value"] = static_cast<int>(parsed.member_double("step", 0));
        data["max"] = static_cast<int>(parsed.member_double("steps", 0));
        data["prompt_id"] = job->id;
        data["node"] = task.node_id;
        Emit(job->client_id, "progress", std::move(data));
      }
    } catch (const std::exception&) {
      // A malformed preview is skipped; the next poll tries again.
    }
  }
}

HttpResponse Adapter::FilmstudioProgress(const HttpRequest& request) {
  const double now = std::chrono::duration<double>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
  const std::string wanted = request.query_param("prompt_id");
  const std::lock_guard<std::mutex> lock(mutex_);
  http_watch_until_ = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  const auto describe = [this](const Live& live, json::Value* out) {
    json::Value& body = *out;
    body["prompt_id"] = live.prompt_id;
    body["node"] = live.node;
    body["class_type"] = live.class_type;
    if (live.has_progress) {
      body["value"] = live.value;
      body["max"] = live.max;
    } else {
      body["value"] = 0;
      body["max"] = 1;
    }
    body["nodes_done"] = live.nodes_done;
    body["nodes_total"] = live.nodes_total;
    if (const auto preview = prompt_previews_.find(live.prompt_id);
        preview != prompt_previews_.end() && !preview->second.jpeg.empty())
      body["preview"] = static_cast<double>(preview->second.seq);
    else
      body["preview"] = nullptr;
  };
  // Every running prompt in start order (2.1: several run at once); the
  // top-level fields describe the oldest one, or the one ?prompt_id= names.
  json::Value all = json::Value::array();
  const Live* top = nullptr;
  for (const auto& job : running_) {
    const auto live = live_.find(job->id);
    if (live == live_.end() || live->second.node.empty())
      continue;  // picked up, its first task has not started yet
    json::Value item = json::Value::object();
    describe(live->second, &item);
    all.push_back(std::move(item));
    if (wanted.empty() ? top == nullptr : wanted == job->id)
      top = &live->second;
  }
  json::Value body = json::Value::object();
  if (top == nullptr)
    body["prompt_id"] = nullptr;
  else
    describe(*top, &body);
  body["time"] = now;
  body["running"] = std::move(all);
  return JsonResponse(200, body);
}

HttpResponse Adapter::FilmstudioPreview(const HttpRequest& request) {
  const std::string wanted = request.query_param("prompt_id");
  const std::lock_guard<std::mutex> lock(mutex_);
  http_watch_until_ = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  const Preview* preview = &last_preview_;
  if (!wanted.empty()) {
    const auto found = prompt_previews_.find(wanted);
    preview = found == prompt_previews_.end() ? nullptr : &found->second;
  }
  if (preview == nullptr || preview->jpeg.empty())
    return HttpResponse{.status = 204, .reason = "No Content"};
  return HttpResponse{.status = 200,
                      .reason = "OK",
                      .body = preview->jpeg,
                      .headers = {{"Content-Type", "image/jpeg"},
                                  {"Cache-Control", "no-store"}}};
}

void Adapter::EmitStatusLocked() {
  // Every queue change ends here: keep the lock-free copy in step.
  queue_remaining_.store(QueueRemainingLocked());
  json::Value info = json::Value::object();
  info["queue_remaining"] = QueueRemainingLocked();
  json::Value status = json::Value::object();
  status["exec_info"] = std::move(info);
  json::Value data = json::Value::object();
  data["status"] = std::move(status);
  json::Value message = json::Value::object();
  message["type"] = "status";
  message["data"] = std::move(data);
  events_.push_back(Event{next_event_++, "", message.dump()});
  while (events_.size() > 4096)
    events_.pop_front();
}

HttpResponse Adapter::Socket(const HttpRequest& request) {
  if (!server::IsWebSocketUpgrade(request))
    return ErrorResponse(400, "invalid_request", "WebSocket upgrade required");
  std::string client = request.query_param("clientId");
  if (client.empty())
    client = NewUuid();
  return server::UpgradeWebSocket(
      request, [this, client](server::WebSocket& socket) {
        std::uint64_t last = 0;
        std::uint64_t last_preview = 0;
        std::string hello;
        // Clients announce {"type": "feature_flags", "data":
        // {"supports_preview_metadata": true}} as their first message.
        std::atomic<bool> metadata{false};
        std::thread reader([&socket, &metadata] {
          std::string text;
          while (socket.Receive(&text)) {
            try {
              const json::Value message = json::parse(text);
              if (message.member_str("type") != "feature_flags")
                continue;
              const json::Value* data = message.find("data");
              const json::Value* flag =
                  data != nullptr ? data->find("supports_preview_metadata")
                                  : nullptr;
              metadata = flag != nullptr && flag->is_bool() && flag->as_bool();
              json::Value features = json::Value::object();
              features["supports_preview_metadata"] = true;
              json::Value reply = json::Value::object();
              reply["type"] = "feature_flags";
              reply["data"] = std::move(features);
              (void)socket.SendText(reply.dump());
            } catch (const std::exception&) {
              // Not JSON or not for us; ComfyUI ignores it as well.
            }
          }
        });
        struct Session {
          Adapter& adapter;
          server::WebSocket& socket;
          std::thread& reader;
          const std::string& client;
          ~Session() {
            socket.Close();
            reader.join();
            const std::lock_guard<std::mutex> lock(adapter.mutex_);
            if (--adapter.sockets_[client] <= 0)
              adapter.sockets_.erase(client);
          }
        } session{*this, socket, reader, client};
        {
          const std::lock_guard<std::mutex> lock(mutex_);
          ++sockets_[client];
          last = next_event_ - 1;
          last_preview = next_preview_ - 1;
          json::Value info = json::Value::object();
          info["queue_remaining"] = QueueRemainingLocked();
          json::Value status = json::Value::object();
          status["exec_info"] = std::move(info);
          json::Value data = json::Value::object();
          data["status"] = std::move(status);
          data["sid"] = client;
          json::Value message = json::Value::object();
          message["type"] = "status";
          message["data"] = std::move(data);
          hello = message.dump();
        }
        if (!socket.SendText(hello))
          return;
        auto keepalive = std::chrono::steady_clock::now();
        for (;;) {
          std::vector<std::string> out;
          std::vector<Preview> previews;
          {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_)
              break;
            for (const Event& event : events_) {
              if (event.seq <= last)
                continue;
              last = event.seq;
              if (event.client.empty() || event.client == client)
                out.push_back(event.text);
            }
            std::uint64_t newest = last_preview;
            for (const std::string& key : {client, std::string()}) {
              const auto it = previews_.find(key);
              if (it != previews_.end() && it->second.seq > last_preview) {
                previews.push_back(it->second);
                newest = std::max(newest, it->second.seq);
              }
            }
            last_preview = newest;
          }
          if (socket.cancelled())
            break;
          bool ok = true;
          for (const std::string& text : out)
            ok = ok && socket.SendText(text);
          for (const Preview& preview : previews) {
            if (!metadata.load()) {
              ok = ok && socket.SendBinary(PreviewImageFrame(preview.jpeg));
              continue;
            }
            json::Value meta = json::Value::object();
            meta["node_id"] = preview.node_id;
            meta["prompt_id"] = preview.prompt_id;
            meta["display_node_id"] = preview.node_id;
            meta["parent_node_id"] = nullptr;
            meta["real_node_id"] = preview.node_id;
            meta["image_type"] = "image/jpeg";
            ok = ok && socket.SendBinary(
                           PreviewMetadataFrame(preview.jpeg, meta.dump()));
          }
          if (!previews.empty())
            keepalive = std::chrono::steady_clock::now();
          if (!ok)
            break;
          if (out.empty() && std::chrono::steady_clock::now() - keepalive >
                                 std::chrono::seconds(10)) {
            if (!socket.SendText(hello))
              break;
            keepalive = std::chrono::steady_clock::now();
          } else if (!out.empty()) {
            keepalive = std::chrono::steady_clock::now();
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      });
}

void Adapter::WorkerMain() {
  for (;;) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
      if (stopping_)
        return;
      job = pending_.front();
      pending_.pop_front();
      running_.push_back(job);
      live_[job->id].prompt_id = job->id;
      EmitStatusLocked();
    }
    Execute(job);
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      std::erase(running_, job);
      live_.erase(job->id);
      prompt_previews_.erase(job->id);
      EmitStatusLocked();
    }
  }
}

void Adapter::Execute(const std::shared_ptr<Job>& job) {
  const auto started = std::chrono::steady_clock::now();
  json::Value messages = json::Value::array();
  const auto message = [&](std::string_view type, json::Value data) {
    data["prompt_id"] = job->id;
    data["timestamp"] = NowMs();
    json::Value entry = json::Value::array();
    entry.push_back(std::string(type));
    entry.push_back(data);
    messages.push_back(std::move(entry));
    Emit(job->client_id, type, std::move(data));
  };
  message("execution_start", json::Value::object());
  json::Value cached = json::Value::object();
  cached["nodes"] = json::Value::array();
  message("execution_cached", std::move(cached));
  for (const std::string& warning : job->plan.warnings) {
    json::Value data = json::Value::object();
    data["message"] = warning;
    message("gufo_warning", std::move(data));
    Logger::Warn("comfy", "event=warning prompt_id=" + job->id + " " + warning);
  }
  json::Value outputs = json::Value::object();
  json::Value meta = json::Value::object();
  bool ok = true;
  json::Value executed = json::Value::array();
  for (const Task& task : job->plan.tasks) {
    json::Value executing = json::Value::object();
    executing["node"] = task.node_id;
    executing["display_node"] = task.node_id;
    executing["prompt_id"] = job->id;
    Emit(job->client_id, "executing", std::move(executing));
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      Live& live = live_[job->id];
      live.prompt_id = job->id;
      live.node = task.node_id;
      live.class_type = task.class_type;
      live.value = 0;
      live.max = 0;
      live.has_progress = false;
      live.nodes_done = static_cast<int>(executed.size());
      live.nodes_total = static_cast<int>(job->plan.tasks.size());
    }
    try {
      if (job->cancel.load())
        throw Interrupted();
      const json::Value ui = RunTask(job, task);
      for (const std::string& warning : job->warnings) {
        json::Value data = json::Value::object();
        data["message"] = warning;
        message("gufo_warning", std::move(data));
        Logger::Warn("comfy",
                     "event=warning prompt_id=" + job->id + " " + warning);
      }
      job->warnings.clear();
      for (const auto& [node, value] : ui.members()) {
        outputs[node] = value;
        json::Value node_meta = json::Value::object();
        node_meta["node_id"] = node;
        node_meta["display_node"] = node;
        node_meta["parent_node"] = nullptr;
        node_meta["real_node_id"] = node;
        meta[node] = std::move(node_meta);
        json::Value done = json::Value::object();
        done["node"] = node;
        done["display_node"] = node;
        done["output"] = value;
        done["prompt_id"] = job->id;
        Emit(job->client_id, "executed", std::move(done));
        executed.push_back(node);
      }
      executed.push_back(task.node_id);
    } catch (const Interrupted&) {
      json::Value data = json::Value::object();
      data["node_id"] = task.node_id;
      data["node_type"] = task.class_type;
      data["executed"] = executed;
      message("execution_interrupted", std::move(data));
      Logger::Warn("comfy", "event=prompt_interrupted prompt_id=" + job->id);
      ok = false;
      break;
    } catch (const std::exception& error) {
      json::Value data = json::Value::object();
      data["node_id"] = task.node_id;
      data["node_type"] = task.class_type;
      data["executed"] = executed;
      data["exception_message"] = std::string(error.what());
      data["exception_type"] = "RuntimeError";
      data["traceback"] = json::Value::array();
      data["current_inputs"] = json::Value::object();
      data["current_outputs"] = json::Value::object();
      message("execution_error", std::move(data));
      Logger::Warn("comfy", "event=prompt_failed prompt_id=" + job->id +
                                " node=" + task.node_id +
                                " error=" + error.what());
      ok = false;
      break;
    }
  }
  if (ok)
    message("execution_success", json::Value::object());
  json::Value done = json::Value::object();
  done["node"] = nullptr;
  done["prompt_id"] = job->id;
  Emit(job->client_id, "executing", std::move(done));
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    live_.erase(job->id);
    prompt_previews_.erase(job->id);
  }
  json::Value status = json::Value::object();
  status["status_str"] = ok ? "success" : "error";
  status["completed"] = ok;
  status["messages"] = std::move(messages);
  json::Value prompt_entry = json::Value::array();
  prompt_entry.push_back(job->number);
  prompt_entry.push_back(job->id);
  prompt_entry.push_back(job->prompt);
  prompt_entry.push_back(job->extra_data);
  json::Value outputs_to_execute = json::Value::array();
  for (const std::string& id : job->outputs)
    outputs_to_execute.push_back(id);
  prompt_entry.push_back(std::move(outputs_to_execute));
  json::Value entry = json::Value::object();
  entry["prompt"] = std::move(prompt_entry);
  entry["outputs"] = std::move(outputs);
  entry["status"] = std::move(status);
  entry["meta"] = std::move(meta);
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (history_.find(job->id) == history_.end())
      history_order_.push_back(job->id);
    history_[job->id] = std::move(entry);
    if (!job->user.empty())
      history_users_[job->id] = job->user;
    while (history_order_.size() > options_.history_limit) {
      history_.erase(history_order_.front());
      history_users_.erase(history_order_.front());
      history_order_.erase(history_order_.begin());
    }
    PersistLocked();
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  EndPreviews(*job);
  Logger::Info("comfy", "event=prompt_done prompt_id=" + job->id +
                            " status=" + (ok ? "success" : "error") +
                            " seconds=" + std::to_string(seconds));
}

json::Value Adapter::RunTask(const std::shared_ptr<Job>& job,
                             const Task& task) {
  if (const auto* image = std::get_if<ImageTask>(&task.spec))
    return RunImage(job, task, *image);
  if (const auto* speech = std::get_if<SpeechTask>(&task.spec))
    return RunSpeech(job, task, *speech);
  if (const auto* dialogue = std::get_if<DialogueTask>(&task.spec))
    return RunDialogue(job, task, *dialogue);
  if (const auto* music = std::get_if<MusicTask>(&task.spec))
    return RunMusic(job, task, *music);
  if (const auto* foley = std::get_if<FoleyTask>(&task.spec))
    return RunFoley(job, task, *foley);
  return RunVideo(job, task, std::get<VideoTask>(task.spec));
}

std::string Adapter::TaskPreview(const Job& job, const Task& task) const {
  return TaskPreviewId(job.preview_base, task.node_id, job.several_previews);
}

bool Adapter::RegisterPreviews(const std::shared_ptr<Job>& job) {
  const auto now = std::chrono::steady_clock::now();
  const std::lock_guard<std::mutex> lock(preview_mutex_);
  // Ended prompts keep their ids for ten minutes (late readers still find
  // the worker preview); at most 1024 entries. A prompt that is gone
  // without ending (deleted or cleared from the queue) counts as ended
  // from now on, so its ids leave as well.
  for (auto& [id, entry] : preview_ids_) {
    if (!entry.ended && entry.job.expired()) {
      entry.ended = true;
      entry.ended_at = now;
    }
  }
  std::erase_if(preview_ids_, [&](const auto& entry) {
    return entry.second.ended &&
           now - entry.second.ended_at > std::chrono::minutes(10);
  });
  while (preview_ids_.size() >= 1024) {
    auto oldest = preview_ids_.end();
    for (auto it = preview_ids_.begin(); it != preview_ids_.end(); ++it)
      if (it->second.ended &&
          (oldest == preview_ids_.end() ||
           it->second.ended_at < oldest->second.ended_at))
        oldest = it;
    if (oldest == preview_ids_.end())
      break;
    preview_ids_.erase(oldest);
  }
  // Another user's live prompt keeps its id: no redirect of its preview.
  for (const Task& task : job->plan.tasks) {
    if (!std::holds_alternative<ImageTask>(task.spec) &&
        !std::holds_alternative<VideoTask>(task.spec))
      continue;
    const auto it = preview_ids_.find(TaskPreview(*job, task));
    if (it != preview_ids_.end() && !it->second.ended &&
        it->second.user != job->user)
      return false;
  }
  for (const Task& task : job->plan.tasks) {
    if (!std::holds_alternative<ImageTask>(task.spec) &&
        !std::holds_alternative<VideoTask>(task.spec))
      continue;
    PreviewEntry entry;
    entry.job = job;
    entry.user = job->user;
    preview_ids_[TaskPreview(*job, task)] = std::move(entry);
  }
  return true;
}

void Adapter::SetPreviewTarget(const std::string& id, std::string target,
                               bool video, std::string model) {
  const std::lock_guard<std::mutex> lock(preview_mutex_);
  PreviewEntry& entry = preview_ids_[id];
  entry.target = std::move(target);
  entry.video = video;
  entry.model = std::move(model);
}

void Adapter::EndPreviews(const Job& job) {
  const auto now = std::chrono::steady_clock::now();
  const std::lock_guard<std::mutex> lock(preview_mutex_);
  for (const Task& task : job.plan.tasks) {
    if (!std::holds_alternative<ImageTask>(task.spec) &&
        !std::holds_alternative<VideoTask>(task.spec))
      continue;
    const auto it = preview_ids_.find(TaskPreview(job, task));
    if (it == preview_ids_.end() || it->second.ended)
      continue;
    // Another prompt may have taken the id since; only this job's entry.
    const auto owner = it->second.job.lock();
    if (owner != nullptr && owner.get() != &job)
      continue;
    if (it->second.target.empty()) {
      preview_ids_.erase(it);
      continue;
    }
    it->second.ended = true;
    it->second.ended_at = now;
  }
}

std::optional<Adapter::PreviewRoute> Adapter::FindPreview(
    std::string_view id, std::string_view user, bool* foreign) {
  if (foreign != nullptr)
    *foreign = false;
  const std::lock_guard<std::mutex> lock(preview_mutex_);
  const auto it = preview_ids_.find(id);
  if (it == preview_ids_.end())
    return std::nullopt;
  // A signed-in user (no admin) sees the previews of its own prompts and
  // of prompts without a user (the router's, loopback's).
  if (!user.empty() && !it->second.user.empty() && it->second.user != user) {
    if (foreign != nullptr)
      *foreign = true;
    return std::nullopt;
  }
  if (it->second.target.empty() &&
      (it->second.ended || it->second.job.expired())) {
    // Deleted from the queue (or ended) before the task started.
    preview_ids_.erase(it);
    return std::nullopt;
  }
  return PreviewRoute{it->second.target, it->second.video, it->second.model};
}

FetchResult Adapter::Call(const std::shared_ptr<Job>& job, std::string method,
                          std::string path, std::string body,
                          std::string content_type,
                          const std::string& preview_id) {
  HttpRequest request;
  request.method = std::move(method);
  request.path = std::move(path);
  request.body = std::move(body);
  if (!content_type.empty())
    request.headers.emplace_back("Content-Type", std::move(content_type));
  if (!preview_id.empty())
    request.headers.emplace_back(kPreviewIdHeader, preview_id);
  request.request_id = "comfy-" + job->id;
  // placement = pool: the graph's own LLM calls are not held back by the
  // busy rule its own queue entry triggers.
  request.headers.emplace_back(kOriginHeader, "comfy");
  request.is_cancelled = [job] { return job->cancel.load(); };
  FetchResult result = options_.fetch(request);
  if (job->cancel.load())
    throw Interrupted();
  return result;
}

json::Value Adapter::RunImage(const std::shared_ptr<Job>& job, const Task& task,
                              const ImageTask& spec) {
  std::vector<std::string> references;
  for (const FileRef& ref : spec.references)
    references.push_back(ReadFileRef(ref));
  int width = spec.width;
  int height = spec.height;
  switch (spec.size_source) {
    case SizeSource::kExplicit:
      // ComfyUI empty latents are multiples of 16 (or 8); gufo Qwen-Image
      // sizes are multiples of 32.
      width = std::max(32, static_cast<int>(std::lround(width / 32.0)) * 32);
      height = std::max(32, static_cast<int>(std::lround(height / 32.0)) * 32);
      break;
    case SizeSource::kSquare:
      width = height = spec.resolution;
      break;
    case SizeSource::kFirstReference: {
      const auto dims = ImageDimensions(references.front());
      if (!dims)
        throw std::runtime_error("cannot read the size of " +
                                 spec.references.front().name);
      std::tie(width, height) =
          ScaleToArea(dims->first, dims->second, spec.resolution);
      break;
    }
    case SizeSource::kCanvas: {
      const auto dims = ImageDimensions(ReadFileRef(spec.canvas));
      if (!dims)
        throw std::runtime_error("cannot read the size of " + spec.canvas.name);
      // gufo Qwen-Image sizes are multiples of 32.
      width =
          std::max(32, static_cast<int>(std::lround(dims->first / 32.0)) * 32);
      height =
          std::max(32, static_cast<int>(std::lround(dims->second / 32.0)) * 32);
      break;
    }
  }
  const std::string size = std::to_string(width) + "x" + std::to_string(height);
  std::string init_image, mask;
  if (spec.strength || !spec.mask.name.empty())
    init_image = ReadFileRef(spec.init_image);
  if (!spec.mask.name.empty())
    mask = ReadFileRef(spec.mask);
  // Qwen-Image ControlNet: the structure image, a file part on edits and a
  // data URL (gufo.control_image) on generations. Without a control image
  // (inpaint-only ZImageFunControlnet) only gufo.control_inpaint is sent.
  const std::string control =
      spec.control && !spec.control->image.name.empty()
          ? ReadFileRef(spec.control->image)
          : std::string();
  json::Value extension = ImageExtension(spec);
  // Regional prompts: gufo.regions with their references as
  // "region_image[]" parts (in region order) and mask files inline.
  std::vector<std::string> region_references;
  if (!spec.regions.empty()) {
    const json::Value fields = ImageRegionsField(spec, width, height);
    json::Value regions = json::Value::array();
    for (std::size_t i = 0; i < spec.regions.size(); ++i) {
      const ImageRegion& region = spec.regions[i];
      json::Value entry = fields.items()[i];
      if (region.mask_file) {
        const std::string bytes = ReadFileRef(*region.mask_file);
        entry["mask_image"] =
            "data:" + SniffImageType(bytes) + ";base64," + Base64Encode(bytes);
      }
      regions.push_back(std::move(entry));
      for (const FileRef& ref : region.references)
        region_references.push_back(ReadFileRef(ref));
    }
    extension["regions"] = std::move(regions);
    if (spec.base_strength != 1.0)
      extension["base_strength"] = spec.base_strength;
  }
  const json::Value loras = ImageLoras(spec);
  // Live preview under this id (TaskPreview: the prompt id or
  // extra_data.gufo_preview_id, known to the client since /prompt); the
  // worker renders it only while someone watches: the relay below for a
  // connected /ws client or /filmstudio poll, or a client's own
  // GET /v1/previews/<id>[/events].
  const std::string preview_id = TaskPreview(*job, task);
  const auto gufo_fields = [&](bool with_extension, bool with_preview) {
    json::Value fields =
        with_extension ? extension : json::Value(json::Value::object());
    if (with_preview) {
      json::Value preview = json::Value::object();
      preview["id"] = preview_id;
      fields["preview"] = std::move(preview);
    }
    return fields;
  };
  const auto send = [&](bool with_extension, bool with_preview) {
    const json::Value fields = gufo_fields(with_extension, with_preview);
    if (references.empty() && init_image.empty() &&
        region_references.empty()) {
      json::Value body = json::Value::object();
      body["model"] = spec.model;
      body["prompt"] = spec.prompt;
      body["size"] = size;
      body["steps"] = spec.steps;
      body["seed"] = spec.seed;
      body["n"] = spec.batch;
      body["response_format"] = "b64_json";
      if (!fields.empty())
        body["gufo"] = fields;
      if (!control.empty()) {
        if (!body.contains("gufo"))
          body["gufo"] = json::Value::object();
        body["gufo"]["control_image"] = "data:" + SniffImageType(control) +
                                        ";base64," + Base64Encode(control);
      }
      if (!loras.is_null())
        body["loras"] = loras;
      return Call(job, "POST", "/v1/images/generations", body.dump(),
                  "application/json", with_preview ? preview_id : "");
    }
    std::vector<MultipartPart> parts = {
        {"model", "", "", spec.model},
        {"prompt", "", "", spec.prompt},
        {"size", "", "", size},
        {"steps", "", "", std::to_string(spec.steps)},
        {"seed", "", "", std::to_string(spec.seed)},
        {"n", "", "", std::to_string(spec.batch)},
        {"response_format", "", "", "b64_json"}};
    if (!fields.empty())
      parts.push_back({"gufo", "", "application/json", fields.dump()});
    if (!loras.is_null())
      parts.push_back({"loras", "", "application/json", loras.dump()});
    if (!init_image.empty()) {
      const std::string type = SniffImageType(init_image);
      // The start image is the single "image" part, or "init_image" next
      // to "image[]" references or a mask (which would otherwise also make
      // a single "image" a reference).
      parts.push_back(
          {references.empty() && mask.empty() ? "image" : "init_image",
           std::string("init") + (type == "image/jpeg" ? ".jpg" : ".png"), type,
           init_image});
    }
    if (!mask.empty()) {
      const std::string type = SniffImageType(mask);
      parts.push_back({"mask", type == "image/jpeg" ? "mask.jpg" : "mask.png",
                       type, mask});
    }
    if (!control.empty()) {
      const std::string type = SniffImageType(control);
      parts.push_back({"control_image",
                       type == "image/jpeg" ? "control.jpg" : "control.png",
                       type, control});
    }
    for (std::size_t i = 0; i < references.size(); ++i) {
      const std::string type = SniffImageType(references[i]);
      parts.push_back({"image[]",
                       "reference-" + std::to_string(i + 1) +
                           (type == "image/jpeg" ? ".jpg" : ".png"),
                       type, references[i]});
    }
    for (std::size_t i = 0; i < region_references.size(); ++i) {
      const std::string type = SniffImageType(region_references[i]);
      parts.push_back({"region_image[]",
                       "region-reference-" + std::to_string(i + 1) +
                           (type == "image/jpeg" ? ".jpg" : ".png"),
                       type, region_references[i]});
    }
    const std::string boundary = "gufocomfy" + NewUuid();
    return Call(job, "POST", "/v1/images/edits",
                BuildMultipart(parts, boundary),
                "multipart/form-data; boundary=" + boundary,
                with_preview ? preview_id : "");
  };
  SetPreviewTarget(preview_id, preview_id, false, spec.model);
  std::jthread relay([&](std::stop_token stop) {
    RelayPreviews(job, task, preview_id, spec.model, true, stop);
  });
  FetchResult result = send(true, true);
  if (result.status == 400 &&
      ErrorMessage(result).find("preview") != std::string::npos) {
    // A worker without live previews renders as before.
    result = send(true, false);
  }
  relay.request_stop();
  relay.join();
  if (result.status == 400 && !extension.empty() && !spec.strength &&
      spec.mask.name.empty() && spec.regions.empty() && !spec.control &&
      ErrorMessage(result).find("gufo") != std::string::npos) {
    // An image worker without the gufo fields (negative prompt, guidance):
    // render without them and say so, as before.
    job->warnings.push_back("node " + task.node_id +
                            ": the image worker rejected " + extension.dump() +
                            " (" + ErrorMessage(result) +
                            "); rendered without negative prompt / guidance");
    result = send(false, false);
  }
  Logger::Info("comfy",
               "event=image prompt_id=" + job->id + " size=" + size +
                   " steps=" + std::to_string(spec.steps) +
                   " references=" + std::to_string(spec.references.size()) +
                   (spec.regions.empty()
                        ? std::string()
                        : " regions=" + std::to_string(spec.regions.size()) +
                              " region_references=" +
                              std::to_string(region_references.size())) +
                   (loras.is_null() ? "" : " loras=" + loras.dump()) +
                   " status=" + std::to_string(result.status));
  if (result.status != 200)
    throw std::runtime_error("gufo image request failed (HTTP " +
                             std::to_string(result.status) +
                             "): " + ErrorMessage(result));
  const json::Value parsed = json::parse(result.body);
  const json::Value* data = parsed.find("data");
  if (data == nullptr || !data->is_array() || data->empty())
    throw std::runtime_error("gufo image response has no data");
  std::vector<std::string> images;
  for (const json::Value& item : data->items()) {
    auto png = Base64Decode(item.member_str("b64_json"));
    if (!png || png->empty())
      throw std::runtime_error("gufo image response has no b64_json");
    images.push_back(std::move(*png));
  }
  json::Value ui = json::Value::object();
  for (const OutputTarget& target : task.outputs) {
    json::Value list = json::Value::array();
    for (const std::string& png : images) {
      json::Value saved = SaveBytes(target, "png", png);
      list.push_back(saved["images"].items().front());
    }
    json::Value node = json::Value::object();
    node["images"] = std::move(list);
    ui[target.node_id] = std::move(node);
  }
  return ui;
}

std::string Adapter::Speak(const std::shared_ptr<Job>& job,
                           const SpeechTask& spec) {
  json::Value body = json::Value::object();
  body["model"] = spec.model;
  body["input"] = spec.text;
  body["voice"] = spec.voice;
  if (!spec.language.empty())
    body["language"] = spec.language;
  if (!spec.instructions.empty())
    body["instructions"] = spec.instructions;
  if (spec.seed)
    body["seed"] = *spec.seed;
  body["response_format"] = "wav";
  for (const auto& [key, value] : spec.sampling.members())
    body[key] = value;
  if (spec.reference_audio) {
    std::string audio = ReadFileRef(*spec.reference_audio);
    if (!audio.starts_with("RIFF")) {
      if (options_.ffmpeg.empty())
        throw std::runtime_error(
            "voice reference " + spec.reference_audio->name +
            " is not WAV; start the router with --comfy-ffmpeg to convert it");
      const fs::path wav = WorkFile("wav");
      std::string error;
      if (!RunFfmpeg({"-i", FilePath(*spec.reference_audio), "-ac", "1", "-c:a",
                      "pcm_s16le", wav},
                     &error))
        throw std::runtime_error("cannot convert the voice reference: " +
                                 error);
      audio = ReadAll(wav);
      fs::remove(wav);
    }
    body["reference_audio"] = Base64Encode(audio);
    if (spec.embedding_only)
      body["voice_clone_mode"] = "speaker_embedding_only";
    else
      body["reference_text"] = spec.reference_text;
  }
  const FetchResult result =
      Call(job, "POST", "/v1/audio/speech", body.dump(), "application/json");
  Logger::Info("comfy", "event=speech prompt_id=" + job->id +
                            " model=" + spec.model +
                            " status=" + std::to_string(result.status) +
                            " bytes=" + std::to_string(result.body.size()));
  if (result.status != 200 || !result.body.starts_with("RIFF"))
    throw std::runtime_error("gufo speech request failed (HTTP " +
                             std::to_string(result.status) +
                             "): " + ErrorMessage(result));
  return result.body;
}

json::Value Adapter::SaveSpeech(const Task& task,
                                const std::vector<std::string>& wavs) {
  json::Value ui = json::Value::object();
  for (const OutputTarget& target : task.outputs) {
    json::Value list = json::Value::array();
    for (const std::string& wav : wavs) {
      json::Value saved;
      if (options_.ffmpeg.empty()) {
        if (target.class_type == "SaveAudioMP3" ||
            target.class_type == "SaveAudioOpus")
          throw std::runtime_error(
              target.class_type +
              " needs ffmpeg; start the router with --comfy-ffmpeg");
        saved = SaveBytes(target, "wav", wav);
      } else {
        const fs::path file = WorkFile("wav");
        WriteAll(file, wav);
        saved = SaveAudioTrack(target, file);
        fs::remove(file);
      }
      list.push_back(saved["audio"].items().front());
    }
    json::Value node = json::Value::object();
    node["audio"] = std::move(list);
    ui[target.node_id] = std::move(node);
  }
  return ui;
}

json::Value Adapter::RunSpeech(const std::shared_ptr<Job>& job,
                               const Task& task, const SpeechTask& spec) {
  return SaveSpeech(task, {Speak(job, spec)});
}

json::Value Adapter::RunDialogue(const std::shared_ptr<Job>& job,
                                 const Task& task, const DialogueTask& spec) {
  std::vector<std::string> lines;
  for (std::size_t i = 0; i < spec.lines.size(); ++i) {
    if (job->cancel.load())
      throw Interrupted();
    lines.push_back(Speak(job, spec.lines[i]));
    json::Value data = json::Value::object();
    data["value"] = static_cast<int>(i + 1);
    data["max"] = static_cast<int>(spec.lines.size());
    data["prompt_id"] = job->id;
    data["node"] = task.node_id;
    Emit(job->client_id, "progress", std::move(data));
  }
  if (!spec.merge)
    return SaveSpeech(task, lines);
  auto joined = JoinWavs(lines, spec.pause_seconds);
  if (!joined)
    throw std::runtime_error(
        "cannot join the dialogue lines (different WAV formats)");
  return SaveSpeech(task, {*joined});
}

json::Value Adapter::RunMusic(const std::shared_ptr<Job>& job,
                              const Task& task, const MusicTask& spec) {
  const FetchResult result = Call(job, "POST", "/v1/audio/music",
                                  MusicRequestBody(spec).dump(),
                                  "application/json");
  Logger::Info("comfy", "event=music prompt_id=" + job->id +
                            " model=" + spec.model +
                            " status=" + std::to_string(result.status) +
                            " bytes=" + std::to_string(result.body.size()));
  if (result.status != 200 || !result.body.starts_with("RIFF"))
    throw std::runtime_error("gufo music request failed (HTTP " +
                             std::to_string(result.status) +
                             "): " + ErrorMessage(result));
  return SaveSpeech(task, {result.body});
}

json::Value Adapter::RunFoley(const std::shared_ptr<Job>& job,
                              const Task& task, const FoleyTask& spec) {
  json::Value body = FoleyRequestBody(spec);
  if (spec.video) {
    const std::string bytes = ReadFileRef(*spec.video);
    std::string type = ContentTypeFor(spec.video->name);
    if (!type.starts_with("video/"))
      type = "video/mp4";
    body["video"] = "data:" + type + ";base64," + Base64Encode(bytes);
  } else if (spec.image) {
    const std::string bytes = ReadFileRef(*spec.image);
    body["image"] =
        "data:" + SniffImageType(bytes) + ";base64," + Base64Encode(bytes);
  }
  const FetchResult result = Call(job, "POST", "/v1/audio/foley", body.dump(),
                                  "application/json");
  Logger::Info("comfy", "event=foley prompt_id=" + job->id +
                            " model=" + spec.model +
                            " status=" + std::to_string(result.status) +
                            " bytes=" + std::to_string(result.body.size()));
  if (result.status != 200 || !result.body.starts_with("RIFF"))
    throw std::runtime_error("gufo foley request failed (HTTP " +
                             std::to_string(result.status) +
                             "): " + ErrorMessage(result));
  json::Value ui = json::Value::object();
  for (const OutputTarget& target : task.outputs) {
    if (target.kind != OutputKind::kVideo) {
      Task single = task;
      single.outputs = {target};
      json::Value saved = SaveSpeech(single, {result.body});
      ui[target.node_id] = saved[target.node_id];
      continue;
    }
    // VHS_VideoCombine / SaveVideo with the sampler's audio: the loaded
    // video with the generated track (H.264 + AAC, cut to the shorter).
    if (options_.ffmpeg.empty() || !spec.video)
      throw std::runtime_error(
          "a video output of MMAudioSampler needs the loaded video and "
          "--comfy-ffmpeg");
    const fs::path wav = WorkFile("wav");
    const fs::path mp4 = WorkFile("mp4");
    WriteAll(wav, result.body);
    std::string error;
    const bool muxed = RunFfmpeg(
        {"-i", FilePath(*spec.video), "-i", wav, "-map", "0:v:0", "-map",
         "1:a:0", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", mp4},
        &error);
    std::error_code ec;
    fs::remove(wav, ec);
    if (!muxed) {
      fs::remove(mp4, ec);
      throw std::runtime_error("cannot add the sound to the video: " + error);
    }
    const std::string bytes = ReadAll(mp4);
    fs::remove(mp4, ec);
    json::Value saved = SaveBytes(target, "mp4", bytes);
    if (target.class_type == "VHS_VideoCombine") {
      json::Value entry = saved["images"].items().front();
      entry["format"] = "video/h264-mp4";
      if (spec.force_rate > 0)
        entry["frame_rate"] = spec.force_rate;
      json::Value gifs = json::Value::array();
      gifs.push_back(std::move(entry));
      json::Value node = json::Value::object();
      node["gifs"] = std::move(gifs);
      ui[target.node_id] = std::move(node);
    } else {
      saved["animated"] = json::Value::array();
      saved["animated"].push_back(true);
      ui[target.node_id] = std::move(saved);
    }
  }
  return ui;
}

json::Value Adapter::RunVideo(const std::shared_ptr<Job>& job, const Task& task,
                              const VideoTask& spec) {
  json::Value body = VideoRequestBody(spec);
  // ComfyUI stretches H3 guide images to the canvas; gufo centre-crops.
  // Stretch here (ffmpeg) so both give the same picture.
  bool warned_stretch = false;
  const auto data_url = [&](const FileRef& ref) {
    std::string bytes = ReadFileRef(ref);
    const auto dims = ImageDimensions(bytes);
    if (spec.family == VideoFamily::kH3 && dims &&
        (dims->first != spec.width || dims->second != spec.height)) {
      if (options_.ffmpeg.empty()) {
        if (!warned_stretch)
          job->warnings.push_back(
              "node " + task.node_id +
              ": guide images are centre-cropped by gufo, not stretched "
              "(start the router with --comfy-ffmpeg)");
        warned_stretch = true;
      } else {
        const fs::path png = WorkFile("png");
        std::string error;
        if (!RunFfmpeg({"-i", FilePath(ref), "-vf",
                        "scale=" + std::to_string(spec.width) + ":" +
                            std::to_string(spec.height) + ":flags=bicubic",
                        "-frames:v", "1", png},
                       &error))
          throw std::runtime_error("cannot resize " + ref.name + ": " + error);
        bytes = ReadAll(png);
        fs::remove(png);
      }
    }
    return "data:" + SniffImageType(bytes) + ";base64," + Base64Encode(bytes);
  };
  if (spec.first_frame && spec.first_frame_index) {
    // One frame of a video (ImageFromBatch over LoadVideo): extracted as a
    // PNG by ffmpeg, frame n counted from 0 in decode order.
    const fs::path png = WorkFile("png");
    std::string error;
    if (!RunFfmpeg({"-i", FilePath(*spec.first_frame), "-vf",
                    "select=eq(n\\," +
                        std::to_string(*spec.first_frame_index) +
                        "),setpts=PTS-STARTPTS",
                    "-frames:v", "1", png},
                   &error) ||
        !fs::exists(png))
      throw std::runtime_error("cannot take frame " +
                               std::to_string(*spec.first_frame_index) +
                               " of " + spec.first_frame->name + ": " +
                               (error.empty() ? "no such frame" : error));
    const std::string bytes = ReadAll(png);
    fs::remove(png);
    body["first_frame"] =
        "data:image/png;base64," + Base64Encode(bytes);
  } else if (spec.first_frame) {
    body["first_frame"] = data_url(*spec.first_frame);
  }
  if (spec.last_frame)
    body["last_frame"] = data_url(*spec.last_frame);
  if (!spec.keyframes.empty()) {
    json::Value keyframes = json::Value::array();
    for (const auto& [frame, ref] : spec.keyframes) {
      json::Value keyframe = json::Value::object();
      keyframe["frame"] = frame;
      keyframe["image"] = data_url(ref);
      keyframes.push_back(std::move(keyframe));
    }
    body["gufo"]["keyframes"] = std::move(keyframes);
  }
  if (spec.render_video) {
    json::Value camera = json::Value::object();
    camera["render_video"] = "data:video/mp4;base64," +
                             Base64Encode(ReadFileRef(*spec.render_video));
    body["gufo"]["camera"] = std::move(camera);
  }
  if (spec.family == VideoFamily::kVace) {
    // VACE media into gufo.vace: videos as MP4 data URLs, images as they are
    // (gufo fits every input to the video size itself).
    json::Value& vace = body["gufo"]["vace"];
    if (spec.control_video)
      vace["control_video"] = "data:video/mp4;base64," +
                              Base64Encode(ReadFileRef(*spec.control_video));
    if (spec.vace_mask) {
      const std::string bytes = ReadFileRef(*spec.vace_mask);
      vace["mask"] = (ImageDimensions(bytes) ? "data:" + SniffImageType(bytes)
                                             : std::string("data:video/mp4")) +
                     ";base64," + Base64Encode(bytes);
    }
    if (!spec.reference_images.empty()) {
      json::Value references = json::Value::array();
      for (const FileRef& ref : spec.reference_images) {
        const std::string bytes = ReadFileRef(ref);
        references.push_back("data:" + SniffImageType(bytes) + ";base64," +
                             Base64Encode(bytes));
      }
      vace["reference_images"] = std::move(references);
    }
  }
  if (spec.source_video)
    body["source_video"] = "data:video/mp4;base64," +
                           Base64Encode(ReadFileRef(*spec.source_video));
  if (spec.h3_control) {
    // MiniMaxH3FunControlNetApply: gufo fits and encodes the videos itself.
    json::Value& control = body["gufo"]["control"];
    if (spec.control_video)
      control["video"] = "data:video/mp4;base64," +
                         Base64Encode(ReadFileRef(*spec.control_video));
    if (spec.h3_control_mask)
      control["mask"] = "data:video/mp4;base64," +
                        Base64Encode(ReadFileRef(*spec.h3_control_mask));
    if (spec.h3_control_source)
      control["source_video"] =
          "data:video/mp4;base64," +
          Base64Encode(ReadFileRef(*spec.h3_control_source));
  }
  if (spec.guide_video) {
    // MiniMaxH3AddGuide image from LoadVideo -> GetVideoComponents: gufo
    // decodes the clip and anchors its first 17k + 5 frames.
    json::Value guide = json::Value::object();
    guide["frame"] = spec.guide_video_frame;
    guide["video"] = "data:video/mp4;base64," +
                     Base64Encode(ReadFileRef(*spec.guide_video));
    body["gufo"]["guide_video"] = std::move(guide);
  }
  if (!spec.talk_audio.empty()) {
    // InfiniteTalk speakers and their masks (gufo.talk).
    json::Value audio = json::Value::array();
    for (const FileRef& ref : spec.talk_audio)
      audio.push_back(json::Value("data:" + ContentTypeFor(ref.name) +
                                  ";base64," +
                                  Base64Encode(ReadFileRef(ref))));
    body["gufo"]["talk"]["audio"] = std::move(audio);
  }
  if (spec.talk_video)
    body["gufo"]["talk"]["video"] = "data:video/mp4;base64," +
                                    Base64Encode(ReadFileRef(*spec.talk_video));
  if (!spec.talk_masks.empty()) {
    // gufo reads bright pixels as the speaker: extract the ComfyUI mask
    // channel (inverted alpha, red, green or blue) as a grey image.
    json::Value masks = json::Value::array();
    for (std::size_t i = 0; i < spec.talk_masks.size(); ++i) {
      const FileRef& ref = spec.talk_masks[i];
      const std::string& channel = spec.talk_mask_channels[i];
      std::string bytes;
      if (options_.ffmpeg.empty()) {
        bytes = ReadFileRef(ref);
        if (i == 0)
          job->warnings.push_back(
              "node " + task.node_id +
              ": speaker masks are sent as they are (bright = speaker); "
              "start the router with --comfy-ffmpeg to use their " +
              channel + " channel");
      } else {
        const std::string filter =
            channel == "inverted_alpha"
                ? "format=rgba,alphaextract,negate"
                : "format=gbrp,extractplanes=" + channel.substr(0, 1);
        const fs::path png = WorkFile("png");
        std::string error;
        if (!RunFfmpeg({"-i", FilePath(ref), "-vf", filter, "-frames:v", "1",
                        png},
                       &error))
          throw std::runtime_error("cannot read the mask " + ref.name + ": " +
                                   error);
        bytes = ReadAll(png);
        fs::remove(png);
      }
      masks.push_back(json::Value("data:" + SniffImageType(bytes) +
                                  ";base64," + Base64Encode(bytes)));
    }
    body["gufo"]["talk"]["speaker_masks"] = std::move(masks);
  }
  if (spec.audio_reference && spec.audio_reference_from_video) {
    // The anchor clip's soundtrack (GetVideoComponents audio) as FLAC.
    if (options_.ffmpeg.empty())
      throw std::runtime_error(
          "the H3 audio guide comes from a video; start the router with "
          "--comfy-ffmpeg to extract it");
    const fs::path flac = WorkFile("flac");
    std::string error;
    if (RunFfmpeg({"-i", FilePath(*spec.audio_reference), "-vn", "-map",
                   "0:a:0", "-ac", "2", "-c:a", "flac", flac},
                  &error)) {
      body["audio_reference"] =
          "data:audio/flac;base64," + Base64Encode(ReadAll(flac));
    } else {
      // ComfyUI's GetVideoComponents yields no audio for a silent clip, so
      // the guide anchors the picture only.
      job->warnings.push_back("node " + task.node_id + ": " +
                              spec.audio_reference->name +
                              " has no usable audio track, anchoring the "
                              "picture only (" +
                              error + ")");
    }
    std::error_code ignored;
    fs::remove(flac, ignored);
  } else if (spec.audio_reference) {
    const std::string audio = ReadFileRef(*spec.audio_reference);
    body["audio_reference"] =
        "data:" + ContentTypeFor(spec.audio_reference->name) + ";base64," +
        Base64Encode(audio);
  }
  if (spec.audio_source) {
    // H3 audio noise mask: the source audio of the masked audio latent.
    std::string url;
    if (spec.audio_source_from_video) {
      if (options_.ffmpeg.empty())
        throw std::runtime_error(
            "the H3 masked audio comes from a video; start the router with "
            "--comfy-ffmpeg to extract it");
      const fs::path flac = WorkFile("flac");
      std::string error;
      const bool extracted =
          RunFfmpeg({"-i", FilePath(*spec.audio_source), "-vn", "-map",
                     "0:a:0", "-ac", "2", "-c:a", "flac", flac},
                    &error);
      if (extracted)
        url = "data:audio/flac;base64," + Base64Encode(ReadAll(flac));
      std::error_code ignored;
      fs::remove(flac, ignored);
      if (!extracted)
        throw std::runtime_error(spec.audio_source->name +
                                 " has no usable audio track for the H3 "
                                 "audio mask: " +
                                 error);
    } else {
      url = "data:" + ContentTypeFor(spec.audio_source->name) + ";base64," +
            Base64Encode(ReadFileRef(*spec.audio_source));
    }
    body["gufo"]["audio_mask"]["audio"] = std::move(url);
  }
  const FetchResult created =
      Call(job, "POST", "/v1/videos", body.dump(), "application/json");
  if (created.status != 200 && created.status != 202 && created.status != 201)
    throw std::runtime_error("gufo video request failed (HTTP " +
                             std::to_string(created.status) +
                             "): " + ErrorMessage(created));
  const std::string video_id = json::parse(created.body).member_str("id");
  if (video_id.empty())
    throw std::runtime_error("gufo video response has no id");
  Logger::Info("comfy", "event=video_created prompt_id=" + job->id +
                            " video=" + video_id + " model=" + spec.model);
  // The task's preview id now leads to the video job's preview (2.1).
  SetPreviewTarget(TaskPreview(*job, task), video_id, true, "");
  // The middle frame of the worker's preview loop, while a /ws client looks.
  std::jthread relay([&](std::stop_token stop) {
    RelayPreviews(job, task, video_id, "", false, stop);
  });
  int last_progress = -1;
  for (;;) {
    if (job->cancel.load()) {
      HttpRequest cancel;
      cancel.method = "DELETE";
      cancel.path = "/v1/videos/" + video_id;
      (void)options_.fetch(cancel);
      throw Interrupted();
    }
    std::this_thread::sleep_for(options_.video_poll);
    HttpRequest poll;
    poll.method = "GET";
    poll.path = "/v1/videos/" + video_id;
    const FetchResult state = options_.fetch(poll);
    if (state.status != 200)
      throw std::runtime_error("gufo video status failed (HTTP " +
                               std::to_string(state.status) +
                               "): " + ErrorMessage(state));
    const json::Value parsed = json::parse(state.body);
    const std::string status = parsed.member_str("status");
    const int progress = static_cast<int>(parsed.member_double("progress", 0));
    if (progress != last_progress) {
      last_progress = progress;
      json::Value data = json::Value::object();
      data["value"] = progress;
      data["max"] = 100;
      data["prompt_id"] = job->id;
      data["node"] = task.node_id;
      Emit(job->client_id, "progress", std::move(data));
    }
    if (status == "completed")
      break;
    if (status == "failed" || status == "cancelled" || status == "expired") {
      const json::Value* error = parsed.find("error");
      throw std::runtime_error(
          "gufo video " + status +
          (error != nullptr ? ": " + error->member_str("message") : ""));
    }
  }
  const FetchResult content =
      Call(job, "GET", "/v1/videos/" + video_id + "/content", "", "");
  if (content.status != 200 || content.body.empty())
    throw std::runtime_error("gufo video content failed (HTTP " +
                             std::to_string(content.status) + ")");
  const fs::path mp4 = WorkFile("mp4");
  WriteAll(mp4, content.body);
  json::Value ui = json::Value::object();
  try {
    for (const OutputTarget& target : task.outputs) {
      if (target.kind == OutputKind::kVideo &&
          target.class_type == "VHS_VideoCombine") {
        // VideoHelperSuite reports its file under "gifs" with the format
        // and frame rate.
        json::Value saved = SaveBytes(target, "mp4", content.body);
        json::Value entry = saved["images"].items().front();
        entry["format"] = "video/h264-mp4";
        entry["frame_rate"] =
            spec.fps_num > 0
                ? static_cast<double>(spec.fps_num) / spec.fps_den
                : static_cast<double>(spec.fps);
        json::Value gifs = json::Value::array();
        gifs.push_back(std::move(entry));
        json::Value node = json::Value::object();
        node["gifs"] = std::move(gifs);
        ui[target.node_id] = std::move(node);
      } else if (target.kind == OutputKind::kVideo) {
        json::Value saved = SaveBytes(target, "mp4", content.body);
        saved["animated"] = json::Value::array();
        saved["animated"].push_back(true);
        ui[target.node_id] = std::move(saved);
      } else if (target.kind == OutputKind::kAudio) {
        ui[target.node_id] = SaveAudioTrack(target, mp4);
      } else {
        ui[target.node_id] = SaveVideoFrames(target, mp4);
      }
    }
  } catch (...) {
    std::error_code ec;
    fs::remove(mp4, ec);
    throw;
  }
  std::error_code ec;
  fs::remove(mp4, ec);
  return ui;
}

fs::path Adapter::WorkFile(std::string_view extension) {
  const fs::path dir = options_.root / "temp" / U8Path(kWorkDir);
  fs::create_directories(dir);
  return dir / U8Path("work-" + std::to_string(++work_counter_) + "-" +
                      NewUuid().substr(0, 8) + "." + std::string(extension));
}

bool Adapter::RunFfmpeg(const std::vector<fs::path>& args, std::string* error) {
  if (options_.ffmpeg.empty()) {
    *error = "no ffmpeg configured (--comfy-ffmpeg)";
    return false;
  }
  std::vector<fs::path> full = {"-hide_banner", "-loglevel", "error",
                                "-nostdin", "-y"};
  full.insert(full.end(), args.begin(), args.end());
  const int code =
      RunProcess(options_.ffmpeg, full, std::chrono::minutes(20), error);
  if (code != 0 && error->empty())
    *error = "ffmpeg exited with " + std::to_string(code);
  return code == 0;
}

namespace {

struct Destination {
  fs::path folder;
  std::string subfolder;
  std::string base;
  std::string type;
};

Destination DestinationFor(const fs::path& root, const OutputTarget& target) {
  const auto name = SplitPrefix(target.prefix);
  if (!name)
    throw std::runtime_error("filename_prefix '" + target.prefix +
                             "' leaves the output folder");
  Destination destination;
  destination.type = target.temp ? "temp" : "output";
  destination.subfolder = name->subfolder;
  destination.base = name->base;
  destination.folder = root / destination.type / U8Path(destination.subfolder);
  fs::create_directories(destination.folder);
  return destination;
}

int CounterIn(const Destination& destination) {
  std::vector<std::string> names;
  std::error_code ec;
  for (fs::directory_iterator it(destination.folder, ec), end; !ec && it != end;
       it.increment(ec))
    names.push_back(PathU8(it->path().filename()));
  return NextCounter(names, destination.base);
}

/// Serialises counter lookup + reservation of output names. Parallel prompts
/// (--comfy-parallel > 1) with the same filename_prefix would otherwise both
/// read the same counter and overwrite each other's file.
std::mutex& OutputMutex() {
  static std::mutex mutex;
  return mutex;
}

/// Creates `path` only if it does not exist yet (O_EXCL / CREATE_NEW).
/// Returns false when it exists; throws on any other error.
bool CreateExclusive(const fs::path& path) {
#ifdef _WIN32
  std::FILE* file = nullptr;
  if (_wfopen_s(&file, path.c_str(), L"wbx") != 0)
    file = nullptr;
#else
  std::FILE* file = std::fopen(path.c_str(), "wbx");
#endif
  if (file != nullptr) {
    std::fclose(file);
    return true;
  }
  std::error_code ec;
  if (fs::exists(path, ec))
    return false;
  throw std::runtime_error("cannot create " + PathU8(path));
}

/// Reserves the next free output name in `destination` by creating it empty
/// and exclusively; the caller then fills it.
std::string ReserveOutput(const Destination& destination, OutputKind kind,
                          std::string_view extension) {
  std::lock_guard<std::mutex> lock(OutputMutex());
  int counter = CounterIn(destination);
  for (int tries = 0; tries < 10000; ++tries, ++counter) {
    const std::string filename =
        OutputFilename(destination.base, counter, kind, extension);
    if (CreateExclusive(destination.folder / U8Path(filename)))
      return filename;
  }
  throw std::runtime_error("no free output name for '" + destination.base +
                           "'");
}

json::Value FileEntry(const Destination& destination,
                      const std::string& filename) {
  json::Value entry = json::Value::object();
  entry["filename"] = filename;
  entry["subfolder"] = ReportedSubfolder(destination.subfolder);
  entry["type"] = destination.type;
  return entry;
}

}  // namespace

json::Value Adapter::SaveBytes(const OutputTarget& target,
                               std::string_view extension,
                               const std::string& bytes) {
  const Destination destination = DestinationFor(options_.root, target);
  const std::string filename =
      ReserveOutput(destination, target.kind, extension);
  WriteAll(destination.folder / U8Path(filename), bytes);
  json::Value list = json::Value::array();
  list.push_back(FileEntry(destination, filename));
  json::Value ui = json::Value::object();
  ui[target.kind == OutputKind::kAudio ? "audio" : "images"] = std::move(list);
  return ui;
}

json::Value Adapter::SaveAudioTrack(const OutputTarget& target,
                                    const fs::path& media) {
  // SaveAudio/PreviewAudio write FLAC like ComfyUI; SaveAudioMP3 (VBR V0)
  // and SaveAudioOpus (128 kbit/s) use ComfyUI's default quality.
  const bool mp3 = target.class_type == "SaveAudioMP3";
  const bool opus = target.class_type == "SaveAudioOpus";
  const std::string ext = mp3 ? "mp3" : opus ? "opus" : "flac";
  const Destination destination = DestinationFor(options_.root, target);
  const std::string filename =
      ReserveOutput(destination, OutputKind::kAudio, ext);
  std::vector<fs::path> codec;
  if (mp3)
    codec = {"-c:a", "libmp3lame", "-q:a", "0"};
  else if (opus)
    codec = {"-c:a", "libopus", "-b:a", "128k"};
  else
    codec = {"-c:a", "flac"};
  std::vector<fs::path> args = {"-i", media, "-vn"};
  args.insert(args.end(), codec.begin(), codec.end());
  args.push_back(destination.folder / U8Path(filename));
  std::string error;
  if (!RunFfmpeg(args, &error)) {
    std::error_code ec;
    fs::remove(destination.folder / U8Path(filename), ec);  // the reservation
    throw std::runtime_error("cannot write the audio track: " + error);
  }
  json::Value list = json::Value::array();
  list.push_back(FileEntry(destination, filename));
  json::Value ui = json::Value::object();
  ui["audio"] = std::move(list);
  return ui;
}

json::Value Adapter::SaveVideoFrames(const OutputTarget& target,
                                     const fs::path& video) {
  const Destination destination = DestinationFor(options_.root, target);
  // ffmpeg writes into a private folder first: a parallel prompt saving into
  // the same output folder must neither get a number inside our range nor
  // have its frames counted as ours.
  const fs::path frames = WorkFile("frames");
  fs::create_directories(frames);
  struct RemoveFrames {
    fs::path dir;
    ~RemoveFrames() {
      std::error_code ignored;
      fs::remove_all(dir, ignored);
    }
  } cleanup{frames};
  std::string error;
  if (!RunFfmpeg({"-i", video, "-start_number", "1",
                  frames / U8Path("f_%05d.png")},
                 &error))
    throw std::runtime_error("cannot extract the video frames: " + error);
  json::Value list = json::Value::array();
  std::error_code ec;
  std::lock_guard<std::mutex> lock(OutputMutex());
  int counter = CounterIn(destination);
  for (int index = 1;; ++index) {
    char own[32];
    std::snprintf(own, sizeof(own), "f_%05d.png", index);
    const fs::path source = frames / own;
    if (!fs::exists(source, ec))
      break;
    std::string filename;
    for (int tries = 0;; ++tries, ++counter) {
      if (tries >= 10000)
        throw std::runtime_error("no free output name for '" +
                                 destination.base + "'");
      filename = OutputFilename(destination.base, counter, OutputKind::kImage,
                                "png");
      if (!fs::exists(destination.folder / U8Path(filename), ec))
        break;
    }
    ++counter;
    const fs::path target_file = destination.folder / U8Path(filename);
    fs::rename(source, target_file, ec);
    if (ec) {  // other volume: copy instead
      ec.clear();
      fs::copy_file(source, target_file, ec);
      if (ec)
        throw std::runtime_error("cannot store video frame " + filename +
                                 ": " + ec.message());
    }
    list.push_back(FileEntry(destination, filename));
  }
  json::Value ui = json::Value::object();
  ui["images"] = std::move(list);
  return ui;
}

void Adapter::PersistLocked() {
  json::Value all = json::Value::object();
  for (const std::string& id : history_order_) {
    const auto owner = history_users_.find(id);
    if (owner == history_users_.end()) {
      all.append_member(id, history_.at(id));
      continue;
    }
    json::Value entry = history_.at(id);
    entry["gufo_user"] = owner->second;
    all.append_member(id, std::move(entry));
  }
  const fs::path file = options_.root / "history.json";
  const fs::path temp = options_.root / "history.json.tmp";
  try {
    WriteAll(temp, all.dump());
    std::error_code ec;
    fs::rename(temp, file, ec);
    if (ec)
      Logger::Warn("comfy", "event=history_save_failed error=" + ec.message());
  } catch (const std::exception& error) {
    Logger::Warn("comfy", std::string("event=history_save_failed error=") +
                              error.what());
  }
}

void Adapter::LoadHistory() {
  const fs::path file = options_.root / "history.json";
  std::error_code ec;
  if (!fs::is_regular_file(file, ec))
    return;
  try {
    const json::Value all = json::parse(ReadAll(file));
    for (const auto& [id, saved] : all.members()) {
      history_order_.push_back(id);
      // The owner goes back to history_users_; GET /history shows the
      // entry as ComfyUI does, without it.
      json::Value entry = saved;
      if (const json::Value* owner = saved.find("gufo_user");
          owner != nullptr && owner->is_string() && saved.is_object()) {
        history_users_[id] = owner->str();
        entry = json::Value::object();
        for (const auto& [key, value] : saved.members())
          if (key != "gufo_user")
            entry.append_member(key, value);
      }
      history_[id] = entry;
      if (const json::Value* prompt = entry.find("prompt");
          prompt != nullptr && prompt->is_array() && !prompt->empty())
        next_number_ = std::max(
            next_number_,
            static_cast<long long>(prompt->items()[0].as_double()) + 1);
    }
  } catch (const std::exception& error) {
    Logger::Warn("comfy", std::string("event=history_load_failed error=") +
                              error.what());
  }
}

}  // namespace gufo::router::comfy
