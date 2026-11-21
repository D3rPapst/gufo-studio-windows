#ifndef GUFO_CLI_SERVE_ROUTER_COMFY_ADAPTER_HPP_
#define GUFO_CLI_SERVE_ROUTER_COMFY_ADAPTER_HPP_

// ComfyUI-compatible front for the router (`gufo router --comfy-dir DIR`):
// POST /prompt takes an API-format workflow, comfy_graph maps it to gufo
// endpoints, and `AdapterOptions::parallel` worker threads run the jobs in
// order (up to that many at once, 2.1; plus the pool partner's free share,
// AdapterOptions::partner_comfy, 2.2.6) through the router's own
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
#include <stdexcept>
#include <string>
#include <string_view>
#include <stop_token>
#include <thread>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/comfy_graph.hpp"
#include "src/cli/serve/router/comfy_parallel.hpp"
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
  /// Memory check before a prompt is accepted (2.2.3), asked only while
  /// no prompt runs or waits here: the router's message when the model of
  /// the prompt's first task cannot load now because another model that is
  /// busy or loading holds the memory (its admission would end in 503
  /// memory_busy), nullopt when it may load or the router cannot tell.
  /// POST /prompt then answers 503 with Retry-After
  /// (kMemoryBusyRetrySeconds) instead of queueing. Optional.
  std::function<std::optional<std::string>(const std::string& model)>
      memory_busy;
  std::chrono::milliseconds video_poll{std::chrono::seconds(2)};
  /// A video task whose POST /v1/videos gets "come back later"
  /// (RetryLaterOf: 429 video_queue_full, 503 overloaded) waits and sends
  /// it again every `video_busy_retry`, for at most `video_busy_limit` in a
  /// row (0 = until the prompt is interrupted), instead of failing the
  /// prompt (2.2.8; Regie 19.11.: four wan2.2 prompts against a worker
  /// queue of one).
  std::chrono::milliseconds video_busy_retry{std::chrono::seconds(10)};
  std::chrono::milliseconds video_busy_limit{std::chrono::hours(6)};
  /// A task's request that gets the router's passing 507 (IsMemoryNotFreeYet:
  /// "not enough free memory yet", memory held outside the router or still
  /// being released) or, since 2.2.12, 503 memory_busy (MemoryWaitOf) is
  /// sent again every `memory_free_retry` (a longer Retry-After of the
  /// answer wins), for at most `memory_free_limit` in a row (0 = until the
  /// prompt is interrupted), instead of failing the prompt. Any task kind
  /// and any request (creates, video content reads); nothing was started
  /// (Regie 19.11.: vace over the .56 -> .198, 507 after 14x memory_busy;
  /// a wan2.2 clip ended as GufoBusy when its content read had to reload
  /// the model beside a busy 27B). Only after the limit does memory_busy
  /// still end the prompt as GufoBusy.
  std::chrono::milliseconds memory_free_retry{std::chrono::seconds(30)};
  std::chrono::milliseconds memory_free_limit{std::chrono::hours(6)};
  /// How long one second of an answer's Retry-After (retry_after_s) lasts
  /// here; tests shorten it.
  std::chrono::milliseconds retry_after_unit{std::chrono::seconds(1)};
  /// A status poll of a running video that fails with a passing error
  /// (no answer, 429, 502, 503, 504) is tried again for this long before
  /// the prompt fails; the video job on the worker goes on meanwhile.
  std::chrono::milliseconds video_poll_grace{std::chrono::minutes(10)};
  /// 2.2.13 (Regie #56-1287): a cut or scaled video input (lossless work
  /// file) larger than this is encoded again visually lossless (libx264
  /// yuv444p, crf 12 and up) until it fits: it goes into the request as
  /// base64 (x4/3) beside the other inputs, and the workers' and fronts'
  /// max-request-bytes are 64-128 MiB. 0 = no limit.
  std::size_t video_input_budget{static_cast<std::size_t>(40) << 20};
  /// Preview poll interval while a /ws client watches.
  std::chrono::milliseconds preview_poll{std::chrono::milliseconds(400)};
  std::size_t history_limit{2000};
  /// Prompts that run at the same time (worker threads, 2.1); 0 counts as 1.
  /// A second prompt's requests go through the router's pool like any
  /// other, so they compute on the partner while this machine's GPU is
  /// busy with the first one.
  std::size_t parallel{1};
  /// The pool partner's ComfyUI numbers (GET /router/capabilities, 2.2.6):
  /// nullopt without a partner, while it cannot be reached or when it is
  /// older and sends no "comfy_parallel". With it, up to
  /// ComfyParallelLimit(parallel, partner) prompts run at once (the extra
  /// ones compute on the partner through the pool). Blocking; asked every
  /// `partner_poll` from the adapter's own thread, never under its mutex.
  /// Optional (unset = `parallel` alone).
  std::function<std::optional<PartnerComfy>()> partner_comfy;
  std::chrono::milliseconds partner_poll{std::chrono::seconds(5)};
  /// 2.2.6: whether a prompt beyond `parallel` may start now: the router's
  /// answer whether its first task's model finds a free machine (pool),
  /// so the extra prompt does not just wait in the pool. Asked outside the
  /// adapter's mutex, about every `model_free_poll` while the next prompt
  /// is held. Optional (unset = extra prompts start at once).
  std::function<bool(const std::string& model)> model_free;
  /// 2.2.13: whether a video job runs through the pool partner (the
  /// router's Registry::JobRemote). While a prompt only polls such a video
  /// it is parked (ParkedRunning): it keeps no GPU here busy. Lock-free
  /// enough to be asked from the poll loop; optional (unset = local).
  std::function<bool(const std::string& video_id)> video_remote;
  std::chrono::milliseconds model_free_poll{std::chrono::seconds(1)};
};

/// Seconds a client waits before it sends a prompt again after memory_busy
/// (the router's Retry-After, router::kMemoryBusyRetryAfterSeconds).
constexpr int kMemoryBusyRetrySeconds = 30;

/// A task's request answered 503 memory_busy: another model that is busy
/// holds the memory. The prompt ends with execution_error, exception_type
/// "GufoBusy", "code": "memory_busy" and "retry_after_s" (2.2.3).
struct MemoryBusyError : std::runtime_error {
  MemoryBusyError(const std::string& message, int retry_after_s)
      : std::runtime_error(message), retry_after_s(retry_after_s) {}
  int retry_after_s;
};

/// The memory_busy error of a router answer (503, error.code memory_busy),
/// nullopt for any other answer.
[[nodiscard]] std::optional<MemoryBusyError> MemoryBusyOf(
    const FetchResult& result);

/// The error code of an answer that means "full now, send it again later"
/// (2.2.8): 429 video_queue_full (the video worker's queue is full) or 503
/// overloaded (a router's connection limit); nullopt for anything else.
[[nodiscard]] std::optional<std::string> RetryLaterOf(
    const FetchResult& result);

/// The router's 507 insufficient_memory for a load that may fit later:
/// memory "held outside the router or still being released" (memory.cpp
/// PlanAdmission, Wait::kOutside, answered after the router's own 45 s
/// wait). Not the other 507s ("loading ... was stopped because memory ran
/// low", a partner's or worker's own refusals). A model larger than the
/// machine never gets this answer: the planner spawns it alone instead.
[[nodiscard]] inline bool IsMemoryNotFreeYet(int status, std::string_view code,
                                             std::string_view message) {
  return status == 507 && code == "insufficient_memory" &&
         message.find("not enough free memory yet") != std::string_view::npos;
}

/// IsMemoryNotFreeYet of an answer's body: the router's message, nullopt
/// for any other answer.
[[nodiscard]] std::optional<std::string> MemoryNotFreeOf(
    const FetchResult& result);

/// An answer after which the same request may simply be sent again later
/// (2.2.12): nothing was started, the memory may be free then.
struct MemoryWait {
  /// "memory_busy" (503), "insufficient_memory" (the passing 507) or, since
  /// 2.2.13, "pool_busy" (503: no machine now, nothing started).
  std::string code;
  std::string message;
  /// The answer's retry_after_s (memory_busy), 0 when it gives none.
  int retry_after_s{0};
};

/// 2.2.13: error.code of a JSON error body ("" when there is none).
[[nodiscard]] std::string ErrorCodeOf(std::string_view body);
/// error.message of a JSON error body ("" when there is none).
[[nodiscard]] std::string ErrorMessageOf(std::string_view body);

/// MemoryBusyOf or MemoryNotFreeOf of an answer, nullopt for anything else.
[[nodiscard]] std::optional<MemoryWait> MemoryWaitOf(const FetchResult& result);

/// A status poll failure that may pass (no answer, 429, 502, 503, 504).
[[nodiscard]] inline bool IsPassingPollFailure(int status) {
  return status == 0 || status == 429 || status == 502 || status == 503 ||
         status == 504;
}

/// The user a request acts for when it may touch only that user's
/// prompts: X-Gufo-User from the access gate, empty for an admin, the
/// router itself, loopback and auth = off.
[[nodiscard]] std::string RestrictedUser(const server::HttpRequest& request);

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

  /// 2.2.13 (pool busy rule, ComfyQueueGpuBusy): running prompts and the
  /// ones of them parked by the adapter (memory wait, polling a video that
  /// runs on the partner). Lock-free.
  [[nodiscard]] int RunningPrompts() const { return running_count_.load(); }
  [[nodiscard]] int ParkedRunning() const { return parked_count_.load(); }

  /// Prompts that may run at the same time now (own parallel + the
  /// partner's free share, 2.2.6) and the partner's share in it; lock-free.
  [[nodiscard]] std::size_t ParallelLimit() const { return limit_.load(); }
  [[nodiscard]] std::size_t PartnerShare() const {
    return partner_share_.load();
  }

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
  /// queue) before the task started, or it is another user's prompt and
  /// `user` (RestrictedUser of the reader) is set; then *foreign (when
  /// given) is true, so the router answers 404 instead of asking a worker
  /// for the id. Safe with the router's state mutex held (own mutex,
  /// nothing called under it).
  [[nodiscard]] std::optional<PreviewRoute> FindPreview(std::string_view id,
                                                        std::string_view user = {},
                                                        bool* foreign = nullptr);

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
                       const VideoTask& spec);
  /// Chained Wan I2V chunks (VideoTask::i2v_chunks): one /v1/videos job
  /// per chunk, each next one started from its i2v_starts frame (ffmpeg),
  /// the outputs' chain_frames joined from the chunks' videos.
  json::Value RunI2VChain(const std::shared_ptr<Job>& job, const Task& task,
                          const VideoTask& spec);
  /// 2.2.16 extra_data.gufo_seam (video_seam.hpp): chunk `part` (rendered
  /// from the start frame `start_png`) corrected towards that frame and
  /// written again as a lossless MP4; answers the new file (the caller
  /// removes both), or `part` with a gufo_warning when it cannot.
  std::filesystem::path SmoothSeam(const std::shared_ptr<Job>& job,
                                   const Task& task, const VideoTask& chunk,
                                   int index,
                                   const std::filesystem::path& start_png,
                                   const std::filesystem::path& part);
  /// The /v1/videos body of `spec` with its media as data URLs.
  json::Value VideoBody(const std::shared_ptr<Job>& job, const Task& task,
                        const VideoTask& spec);
  /// POSTs `payload`, waits for the video (progress `part` of `parts`) and
  /// returns its MP4 bytes. Throws on errors and interrupts.
  std::string RenderVideo(const std::shared_ptr<Job>& job, const Task& task,
                          const VideoTask& spec, const std::string& payload,
                          int part, int parts);
  /// Stores a video job's result (`bytes`, also written to `mp4`) for one
  /// output; returns its ui entry.
  json::Value SaveVideoOutput(const OutputTarget& target,
                              const VideoTask& spec, const std::string& bytes,
                              const std::filesystem::path& mp4);
  json::Value RunMusic(const std::shared_ptr<Job>& job, const Task& task,
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
  /// 2.2.13: `mp4` (a lossless work file) encoded again in place, visually
  /// lossless, when it is above video_input_budget (event=video_cut_reencode).
  void FitVideoToBudget(const std::filesystem::path& mp4,
                        std::string_view what);
  bool RunFfmpeg(const std::vector<std::filesystem::path>& args,
                 std::string* error);
  /// Frames built in the graph (VideoTask::control_runs / mask_runs) as a
  /// lossless 16 fps MP4 of width x height, at most `frames` frames (ffmpeg,
  /// libx264rgb); returns its bytes. Throws on errors.
  std::string RenderFrameRuns(const std::vector<FrameRun>& runs, int width,
                              int height, int frames, std::string_view what);
  /// A loaded video or still with the graph's ImageScale / ImageScaleBy
  /// steps applied (FitFilter, ffmpeg): a new work file the caller removes
  /// (a lossless MP4 with every frame, or a PNG). Throws on errors.
  std::filesystem::path ScaleToWorkFile(const FileRef& file,
                                        const std::vector<FrameFit>& fits,
                                        bool video, std::string_view what);
  /// A loaded video with FileRef::cuts (ImageFromBatch in the graph) as a
  /// FileRef without cuts: the file itself when the cut keeps every frame,
  /// else a lossless work file of the kept frames (libx264rgb, the file's
  /// frame rate, the sound trimmed to them) added to *made. `done` holds
  /// the cuts of this task already made. Throws on errors.
  FileRef CutVideo(const FileRef& file,
                   std::vector<std::pair<FileRef, FileRef>>* done,
                   std::vector<std::filesystem::path>* made);
  /// Every FileRef of `task` with cuts replaced by CutVideo's result.
  void CutVideoInputs(Task* task, std::vector<std::filesystem::path>* made);
  /// The bytes of `file`, scaled first when `fits` has steps.
  std::string ScaledBytes(const FileRef& file,
                          const std::vector<FrameFit>& fits, bool video,
                          std::string_view what);
  /// WanAnimateToVideo replacement: the pose (control_video / control_runs)
  /// inside the mask, the background (animate_background[_runs]) outside,
  /// as a lossless 16 fps MP4 of the video size (AnimateMergeFilter,
  /// ffmpeg); returns its bytes. Throws on errors.
  std::string AnimateReplaceControl(const VideoTask& spec);
  FetchResult Call(const std::shared_ptr<Job>& job, std::string method,
                   std::string path, std::string body,
                   std::string content_type,
                   const std::string& preview_id = {});
  /// The preview id of `task` (Adapter::FindPreview).
  std::string TaskPreview(const Job& job, const Task& task) const;
  /// Registers the preview ids of a queued prompt (no target yet); false
  /// (nothing registered) when one is taken by another user's live prompt.
  bool RegisterPreviews(const std::shared_ptr<Job>& job);
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
  /// Running prompts, oldest start first (at most limit_ when they started).
  std::vector<std::shared_ptr<Job>> running_;
  /// ComfyParallelLimit of the last partner poll; written under mutex_.
  std::atomic<std::size_t> limit_{1};
  std::atomic<std::size_t> partner_share_{0};
  /// Polls AdapterOptions::partner_comfy and adds workers when the limit
  /// grows (they are never removed: spare ones wait).
  std::thread limit_thread_;
  std::condition_variable limit_wake_;
  void LimitMain();
  /// QueueRemainingLocked() as of the last EmitStatusLocked().
  std::atomic<int> queue_remaining_{0};
  /// running_.size(), kept with it (2.2.13).
  std::atomic<int> running_count_{0};
  /// Running prompts parked right now (ParkGuard, 2.2.13).
  std::atomic<int> parked_count_{0};

 public:
  /// Counts a prompt as parked for its lifetime (RAII, cannot leak).
  class ParkGuard {
   public:
    explicit ParkGuard(std::atomic<int>& count) : count_(&count) {
      count_->fetch_add(1);
    }
    ~ParkGuard() { count_->fetch_sub(1); }
    ParkGuard(const ParkGuard&) = delete;
    ParkGuard& operator=(const ParkGuard&) = delete;

   private:
    std::atomic<int>* count_;
  };

 private:
  std::map<std::string, json::Value> history_;
  std::vector<std::string> history_order_;
  /// The signed-in user (X-Gufo-User) of each history entry that has one;
  /// POST /history clear/delete of a user who is no admin touch only these.
  /// Saved in history.json as the entry's "gufo_user", left out of GET.
  std::map<std::string, std::string> history_users_;
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
    /// The prompt's user (Job::user), kept after the prompt ended.
    std::string user;
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
