#ifndef GUFO_CLI_SERVE_ROUTER_COMFY_ADAPTER_HPP_
#define GUFO_CLI_SERVE_ROUTER_COMFY_ADAPTER_HPP_

// ComfyUI-compatible front for the router (`gufo router --comfy-dir DIR`):
// POST /prompt takes an API-format workflow, comfy_graph maps it to gufo
// endpoints, and `AdapterOptions::parallel` worker threads run the jobs in
// order (up to that many at once, 2.1) through the router's own
// OpenAI-style API, so with placement = pool two prompts can compute on
// both machines at the same time. Results are served the ComfyUI way:
// /history/{id}, /view?filename=.., /queue, /upload/image, /free,
// /interrupt, /system_stats, /object_info and the /ws progress socket.
//
// Live previews: while a /ws client of the prompt is connected, the running
// image or video task's newest worker preview (/v1/previews) goes to it as
// a binary frame (type 4 with node metadata for clients that announced
// supports_preview_metadata, otherwise type 1), like a KSampler preview.
//
// Files live below DIR: input/ (uploads), output/ (Save* nodes), temp/
// (Preview* nodes and work files) and history.json (finished prompts).

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <stop_token>
#include <thread>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/comfy_graph.hpp"
#include "src/core/json.hpp"

namespace gufo::router::comfy {

struct FetchResult {
  int status{0};
  std::string body;
  std::string content_type;
};

/// Sends one request to the router's own model API and returns the whole
/// response. `request.is_cancelled` aborts the wait.
using Fetch = std::function<FetchResult(const server::HttpRequest&)>;

struct AdapterOptions {
  std::filesystem::path root;
  /// ffmpeg for FLAC output, PNG frames from videos and non-WAV voice
  /// references; empty = WAV output, no frame extraction.
  std::filesystem::path ffmpeg;
  /// ffprobe (next to ffmpeg): the frame rate of a LoadVideo file that is
  /// only interpolated / upscaled, so RIFE x5 of a 12 fps shot is 60 fps.
  /// Empty = 24 fps assumed.
  std::filesystem::path ffprobe;
  ModelIds models;
  Fetch fetch;
  /// POST /free: unloads idle workers. Optional.
  std::function<void()> unload;
  std::chrono::milliseconds video_poll{std::chrono::seconds(2)};
  /// Preview poll interval while a /ws client watches.
  std::chrono::milliseconds preview_poll{std::chrono::milliseconds(400)};
  std::size_t history_limit{2000};
  /// Prompts that run at the same time (worker threads, 2.1); 0 counts as 1.
  /// A second prompt's requests go through the router's pool like any
  /// other, so they compute on the partner while this machine's GPU is
  /// busy with the first one.
  std::size_t parallel{1};
};

class Adapter {
public:
  explicit Adapter(AdapterOptions options);
  ~Adapter();
  Adapter(const Adapter&) = delete;
  Adapter& operator=(const Adapter&) = delete;

  void Start();
  /// Cancels the running jobs and joins the workers.
  void Stop();

  /// The pool partner's ComfyUI front (placement = pool, 2.1): `GET
  /// /queue?pool=1` and `GET /prompt?pool=1` add its queue, read with
  /// `fetch` (a `GET /queue` to the partner router, which answers it
  /// locally). Call before Start(); `name` is the partner's
  /// `http://host:port`.
  void SetPartner(std::string name, Fetch fetch);

  /// The ComfyUI response for `request`, or nullopt when the path is not
  /// one of ComfyUI's (the router then routes it as usual).
  std::optional<server::HttpResponse> Handle(
      const server::HttpRequest& request);

  /// Prompts running or waiting (ComfyUI's queue_remaining). Lock-free, so
  /// the router may read it with its state mutex held (placement = pool:
  /// a non-empty queue keeps pool LLM requests off this machine).
  [[nodiscard]] int QueueRemaining() const { return queue_remaining_.load(); }

  /// The finished prompt whose outputs list this file, as `/view` names it
  /// (type "" = output); nullopt when no history entry does (an upload, a
  /// deleted entry). The access gate asks before it serves `/view` (2.0.5).
  [[nodiscard]] std::optional<std::string> PromptOfFile(
      const std::string& filename, const std::string& subfolder,
      const std::string& type);

  /// Where the live preview of a prompt's image or video task renders
  /// (2.1). Each such task has a stable preview id known at POST /prompt:
  /// extra_data.gufo_preview_id or else the prompt id, with "_" + node id
  /// appended when the prompt has more than one image or video task
  /// (TaskPreviewId). `target` is the worker preview id: the same id for an
  /// image, the video job id for a video; empty while the prompt waits or
  /// runs and the task's request is not sent yet.
  struct PreviewRoute {
    std::string target;
    bool video{false};
    /// The image model the task's request names ("" for a video).
    std::string model;
  };
  /// nullopt: no preview of this adapter, or its prompt ended (or left the
  /// queue) before the task started. Safe with the router's state mutex
  /// held (own mutex, nothing called under it).
  [[nodiscard]] std::optional<PreviewRoute> FindPreview(std::string_view id);

  struct Job;

private:
  server::HttpResponse Submit(const server::HttpRequest& request);
  server::HttpResponse QueueGet(const server::HttpRequest& request);
  server::HttpResponse PromptGet(const server::HttpRequest& request);
  /// The partner's /queue for a pool view: its body, or nullopt with
  /// *state "none" (no partner), "unreachable" or "no_comfy".
  std::optional<json::Value> PartnerQueue(std::string* state);
  server::HttpResponse QueuePost(const server::HttpRequest& request);
  server::HttpResponse HistoryGet(const server::HttpRequest& request,
                                  std::string_view id);
  server::HttpResponse HistoryPost(const server::HttpRequest& request);
  server::HttpResponse Interrupt(const server::HttpRequest& request);
  server::HttpResponse Free(const server::HttpRequest& request);
  server::HttpResponse View(const server::HttpRequest& request);
  server::HttpResponse Upload(const server::HttpRequest& request);
  server::HttpResponse SystemStats();
  server::HttpResponse ObjectInfo(std::string_view class_type);
  server::HttpResponse Socket(const server::HttpRequest& request);
  server::HttpResponse FilmstudioProgress(const server::HttpRequest& request);
  server::HttpResponse FilmstudioPreview(const server::HttpRequest& request);

  void WorkerMain();
  void Execute(const std::shared_ptr<Job>& job);
  json::Value RunTask(const std::shared_ptr<Job>& job, const Task& task);
  json::Value RunImage(const std::shared_ptr<Job>& job, const Task& task,
                       const ImageTask& spec);
  json::Value RunSpeech(const std::shared_ptr<Job>& job, const Task& task,
                        const SpeechTask& spec);
  json::Value RunDialogue(const std::shared_ptr<Job>& job, const Task& task,
                          const DialogueTask& spec);
  std::string Speak(const std::shared_ptr<Job>& job, const SpeechTask& spec);
  json::Value SaveSpeech(const Task& task,
                         const std::vector<std::string>& wavs);
  json::Value RunVideo(const std::shared_ptr<Job>& job, const Task& task,
                       const VideoTask& spec);  json::Value RunMusic(const std::shared_ptr<Job>& job, const Task& task,
                       const MusicTask& spec);
  json::Value RunFoley(const std::shared_ptr<Job>& job, const Task& task,
                       const FoleyTask& spec);

  std::filesystem::path FilePath(const FileRef& ref) const;
  std::string ReadFileRef(const FileRef& ref) const;
  /// Frame rate of a LoadVideo file (AdapterOptions::ffprobe); 0 = unknown.
  double ProbeSourceFps(const FileRef& ref) const;
  json::Value SaveBytes(const OutputTarget& target, std::string_view extension,
                        const std::string& bytes);
  json::Value SaveVideoFrames(const OutputTarget& target,
                              const std::filesystem::path& video);
  json::Value SaveAudioTrack(const OutputTarget& target,
                             const std::filesystem::path& media);
  std::filesystem::path WorkFile(std::string_view extension);
  bool RunFfmpeg(const std::vector<std::filesystem::path>& args,
                 std::string* error);
  FetchResult Call(const std::shared_ptr<Job>& job, std::string method,
                   std::string path, std::string body,
                   std::string content_type,
                   const std::string& preview_id = {});
  /// The preview id of `task` (Adapter::FindPreview).
  std::string TaskPreview(const Job& job, const Task& task) const;
  /// Registers the preview ids of a queued prompt (no target yet).
  void RegisterPreviews(const std::shared_ptr<Job>& job);
  /// The task's request is about to go out: its preview renders as `target`.
  void SetPreviewTarget(const std::string& id, std::string target, bool video,
                        std::string model);
  /// The prompt ended: ids without a target are forgotten at once, the
  /// others after a while.
  void EndPreviews(const Job& job);

  void Emit(const std::string& client, std::string_view type, json::Value data);
  /// Polls the worker preview `preview_id` until `stop`, while a socket of
  /// the job's client is open, and forwards new ones (EmitPreview).
  void RelayPreviews(const std::shared_ptr<Job>& job, const Task& task,
                     const std::string& preview_id, const std::string& model,
                     bool image_progress, std::stop_token stop);
  void EmitPreview(const std::string& client, const std::string& prompt_id,
                   const std::string& node_id, std::string jpeg);
  bool WatchedLocked(const std::string& client) const;
  void EmitStatusLocked();
  json::Value QueueItemLocked(const Job& job) const;
  int QueueRemainingLocked() const;
  void PersistLocked();
  void LoadHistory();

  AdapterOptions options_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<std::shared_ptr<Job>> pending_;
  /// Running prompts, oldest start first (at most options_.parallel).
  std::vector<std::shared_ptr<Job>> running_;
  /// QueueRemainingLocked() as of the last EmitStatusLocked().
  std::atomic<int> queue_remaining_{0};
  std::map<std::string, json::Value> history_;
  std::vector<std::string> history_order_;
  long long next_number_{0};
  bool stopping_{false};
  std::vector<std::thread> workers_;
  /// SetPartner(); read-only once Start() ran.
  std::string partner_name_;
  Fetch partner_fetch_;
  /// An unreachable partner is not asked again before this (pool view).
  std::mutex partner_mutex_;
  std::chrono::steady_clock::time_point partner_retry_at_{};

  struct Event {
    std::uint64_t seq;
    std::string client;
    std::string text;
  };
  std::deque<Event> events_;
  /// Open /ws sockets per client id.
  std::map<std::string, int, std::less<>> sockets_;
  /// Newest preview per client id ("" = every client).
  struct Preview {
    std::uint64_t seq{0};
    std::string prompt_id;
    std::string node_id;
    std::string jpeg;
  };
  std::map<std::string, Preview, std::less<>> previews_;
  /// Film studio polling (/filmstudio/progress, /filmstudio/preview, as the
  /// filmstudio_progress ComfyUI plugin): the running prompts for any client.
  struct Live {
    std::string prompt_id;
    std::string node;
    std::string class_type;
    int value{0};
    int max{0};
    bool has_progress{false};
    int nodes_done{0};
    int nodes_total{0};
  };
  /// Per running prompt id (Execute sets it, the prompt's end erases it).
  std::map<std::string, Live, std::less<>> live_;
  /// Newest preview of any client, for /filmstudio/preview.
  Preview last_preview_;
  /// Newest preview per running prompt (/filmstudio/*?prompt_id=).
  std::map<std::string, Preview, std::less<>> prompt_previews_;
  /// A /filmstudio poll counts as watching for 10 s: previews get rendered.
  std::chrono::steady_clock::time_point http_watch_until_{};
  std::uint64_t next_preview_{1};
  std::uint64_t next_event_{1};
  /// Preview ids of queued, running and recently finished prompts
  /// (FindPreview); guarded by preview_mutex_ alone.
  struct PreviewEntry {
    std::weak_ptr<Job> job;
    std::string target;
    bool video{false};
    std::string model;
    bool ended{false};
    std::chrono::steady_clock::time_point ended_at{};
  };
  std::mutex preview_mutex_;
  std::map<std::string, PreviewEntry, std::less<>> preview_ids_;
  std::atomic<std::uint64_t> work_counter_{0};
};

}  // namespace gufo::router::comfy

#endif  // GUFO_CLI_SERVE_ROUTER_COMFY_ADAPTER_HPP_
