#ifndef GUFO_CLI_SERVE_ROUTER_MEMORY_HPP_
#define GUFO_CLI_SERVE_ROUTER_MEMORY_HPP_

// Memory-aware residency for router workers: live memory probes, learned
// per-model peaks, and the pure admission planner that decides whether a new
// worker may start beside the resident ones, which idle ones to evict first,
// or whether the request has to wait.

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::router {

constexpr std::int64_t kGiB = 1024LL * 1024 * 1024;

/// Live system-wide memory. GPU numbers are for the adapter the workers use
/// (dedicated = VRAM carve-out, shared = system RAM the GPU may map); shared
/// GPU allocations are backed by host RAM, so they also lower host_available.
struct SystemMemory {
  bool gpu_valid{false};
  bool host_valid{false};
  std::int64_t gpu_dedicated_total{0};
  std::int64_t gpu_dedicated_used{0};
  std::int64_t gpu_shared_total{0};
  std::int64_t gpu_shared_used{0};
  std::int64_t host_total{0};
  std::int64_t host_available{0};
  /// Windows commit charge: limit (RAM + page file, as it is sized right
  /// now) and what is still free under it. On this APU dedicated and shared
  /// GPU allocations are charged to commit as well, so a model costs its
  /// GPU and host need here. Invalid elsewhere (no commit check).
  bool commit_valid{false};
  std::int64_t commit_limit{0};
  std::int64_t commit_available{0};

  [[nodiscard]] std::int64_t DedicatedFree() const;
  [[nodiscard]] std::int64_t SharedFree() const;
};

/// One worker process. host_* is the working set (on this APU, VRAM
/// allocations are charged to private bytes, so commit is not host RAM).
struct ProcessMemory {
  bool valid{false};
  std::int64_t gpu_dedicated{0};
  std::int64_t gpu_shared{0};
  std::int64_t host_working_set{0};
  std::int64_t host_peak_working_set{0};
  /// Private bytes (commit charge) of the process; 0 = unknown.
  std::int64_t commit{0};

  [[nodiscard]] std::int64_t Gpu() const { return gpu_dedicated + gpu_shared; }
};

class MemoryProbe {
public:
  virtual ~MemoryProbe() = default;
  virtual SystemMemory System() = 0;
  /// Usage per pid; pids that cannot be read are missing from the map.
  virtual std::map<std::uint32_t, ProcessMemory> Processes(
      const std::vector<std::uint32_t>& pids) = 0;
  /// Adapter and counter diagnostics for --memory-report.
  virtual std::string Describe() { return {}; }
};

/// DXGI + PDH + psapi on Windows; host-only (/proc/meminfo) elsewhere.
std::unique_ptr<MemoryProbe> MakeSystemMemoryProbe();

/// Peak a model needs: total GPU (dedicated first, the driver spills the rest
/// into shared) and host working set.
struct MemoryNeed {
  std::int64_t gpu{0};
  std::int64_t host{0};
  /// Part of `gpu` the model itself keeps in shared memory (pinned host
  /// buffers such as an in-RAM prompt cache); it is already counted in
  /// `host`, so it is no spill. 0 for estimates (unknown).
  std::int64_t shared{0};
  /// Dedicated bytes the model was measured to hold (complete learned peak,
  /// without the safety factor). Only this part can spill into shared = host
  /// RAM when dedicated is short; the factor's headroom is guarded by the
  /// GPU total check and reserve_host. 0 = unknown: gpu - shared is used.
  std::int64_t dedicated{0};
};

struct MemoryPolicy {
  double factor{1.15};          // safety factor on hand-written estimates
  double learned_factor{1.10};  // on measured peaks (sampled every second)
  std::int64_t reserve_gpu{3 * kGiB};     // kept free on the GPU (desktop, dwm)
  std::int64_t reserve_host{4 * kGiB};    // kept free in host RAM
  std::int64_t abort_host{3 * kGiB / 2};  // watchdog: host below this
  std::int64_t abort_gpu{kGiB / 2};       // watchdog: GPU (ded+shared) below
  std::int64_t reserve_commit{3 * kGiB};  // kept free under the commit limit
  std::int64_t abort_commit{kGiB};        // watchdog: free commit below this
};

/// Commit a model needs: its GPU need plus its host need, without its own
/// shared buffers, which are in both (pinned host memory).
[[nodiscard]] std::int64_t CommitNeed(const MemoryNeed& need);

/// What the planner knows about one resident (loading or loaded) worker.
struct ResidentView {
  std::string id;
  bool loading{false};
  bool busy{false};
  bool exclusive{false};
  MemoryNeed need;        // already multiplied by the safety factor
  ProcessMemory current;  // last sample; invalid = nothing measured yet
  std::chrono::steady_clock::time_point last_active{};
};

struct AdmissionPlan {
  enum class Action { kSpawn, kEvict, kWait };
  /// What a kWait plan waits for (2.2.3): another model of ours that is
  /// busy or loading (kBusy), memory held outside the router (kOutside), or
  /// a stopped worker returning its memory (kRelease).
  enum class Wait { kNone, kBusy, kOutside, kRelease };
  Action action{Action::kWait};
  std::vector<std::string> evict;  // kEvict: stop these, then plan again
  bool alone{false};  // spawn although the estimate does not fit: no other
                      // worker is left that could be evicted
  std::string reason;
  Wait wait{Wait::kNone};
  /// kBusy: the models the load waits for, "id (busy)" / "id (loading)",
  /// comma separated.
  std::string blocked_by;
  /// kBusy: the same models by id only (2.2.4).
  std::vector<std::string> blocker_ids;
};

/// Pure admission decision for starting `id` (need already factored). Rules:
///  - memory still owed by residents (need minus current usage) is counted
///    as taken, so concurrent loads cannot double-book the same free bytes;
///  - fits: dedicated+shared free minus reserve_gpu >= need.gpu, and host
///    free minus reserve_host minus the part that spills past free
///    dedicated into shared (shared is host RAM) >= need.host, and (with
///    a valid commit reading) free commit minus reserve_commit >=
///    CommitNeed(need);
///  - otherwise idle, fully loaded residents are evicted least recently used
///    first until it fits; busy or loading residents are never evicted (wait);
///  - an exclusive candidate evicts every idle resident and waits for the
///    rest; an exclusive resident blocks every other model the same way;
///  - when nothing of ours is left to evict and it still does not fit
///    (memory held by other processes, or not yet returned by a stopped
///    worker), wait; only a need larger than the whole machine starts
///    alone. An idle resident without a memory sample is evicted first
///    instead (2.2.3): what it gives back is unknown, so its memory may be
///    the missing part;
///  - every kWait plan names its kind (AdmissionPlan::Wait) and, waiting
///    for busy or loading models, those models (blocked_by).
AdmissionPlan PlanAdmission(const SystemMemory& system,
                            const std::vector<ResidentView>& residents,
                            std::string_view id, const MemoryNeed& need,
                            bool exclusive, const MemoryPolicy& policy);

/// Bounds of the memory waits of one cold load (2.2.3). 0 = no bound of
/// its own (only the load timeout ends the wait).
struct MemoryWaitLimits {
  /// Waiting for another model that is busy or loading (Wait::kBusy):
  /// answered 503 memory_busy with Retry-After (--memory-wait-seconds).
  std::chrono::seconds busy{60};
  /// Waiting for memory held outside the router (Wait::kOutside): answered
  /// 507 insufficient_memory.
  std::chrono::seconds outside{45};
};

enum class MemoryWaitVerdict { kContinue, kGiveUpBusy, kGiveUpOutside };

/// Times the waits of one admission: each kind of wait counts from its
/// first plan in a row; any other plan (another kind, evict, spawn) resets
/// it. A release wait (a stopped worker returning memory, at most 30 s)
/// resets neither, so a load waiting for a busy model and then for the
/// memory of an evicted one is not bounded twice.
class MemoryWaitClock {
public:
  using Clock = std::chrono::steady_clock;
  MemoryWaitVerdict Next(const AdmissionPlan& plan, Clock::time_point now,
                         const MemoryWaitLimits& limits);

private:
  std::optional<Clock::time_point> busy_since_;
  std::optional<Clock::time_point> outside_since_;
};

/// Cold loads that wait for busy (or loading) models of ours (2.2.4).
/// While a load of Y waits for X, relayed pool requests of X (the partner's
/// work) are turned back, so X becomes idle and Y can load (18.11. 17:48:
/// wan2.1-vace-14b waited for the 27B on the .198 while the .56 kept
/// relaying 27B requests to it). A wait lingers kLinger after its last
/// plan: across the 503 memory_busy and the client's retry, the partner
/// work stays off X. Not thread-safe (Residency locks it).
class LoadWaitBook {
public:
  using Clock = std::chrono::steady_clock;
  /// Covers Retry-After (kMemoryBusyRetryAfterSeconds) and the slower
  /// retries of a job queue (90 s).
  static constexpr std::chrono::seconds kLinger{120};
  /// The plan of a cold load of `id`: a busy wait records (refreshes) its
  /// blockers, a spawn ends the wait, other plans (evict, release or
  /// outside waits) leave it as it is.
  void Note(std::string_view id, const AdmissionPlan& plan,
            Clock::time_point now);
  /// The load of `id` no longer waits (gave up for another reason).
  void Clear(std::string_view id);
  /// A load still waiting for `model`, or empty (the first by id).
  [[nodiscard]] std::string WaitingFor(std::string_view model,
                                       Clock::time_point now) const;

private:
  struct Entry {
    std::vector<std::string> blockers;
    Clock::time_point until{};
  };
  std::map<std::string, Entry, std::less<>> waits_;
};

/// Seconds a client should wait before it retries a memory_busy load.
constexpr int kMemoryBusyRetryAfterSeconds = 30;

/// Watchdog decision for one sample.
struct WatchdogAction {
  std::string abort_load;  // kill this loading worker (newest load)
  std::string evict;       // or stop this idle resident
  std::string reason;      // non-empty when memory is critical
};

/// `loading_order` lists loading ids oldest first.
WatchdogAction PlanWatchdog(const SystemMemory& system,
                            const std::vector<ResidentView>& residents,
                            const std::vector<std::string>& loading_order,
                            const MemoryPolicy& policy);

/// A lower bound gufo knows for a model's need, applied over estimates and
/// learned peaks so an old peak cannot plan it too small (nullopt: none).
/// wan2.2-fun-vace-a14b keeps both experts loaded (2.2.6): 2 x (14 GB FP8
/// DiT + 2.8 GB VACE module) + umT5 (10.6 GB) and the VAE during a job; a
/// peak learned while it loaded one expert at a time is far smaller.
[[nodiscard]] std::optional<MemoryNeed> BuiltInFloor(std::string_view id);

/// Estimates (preset keys or the JSON file) and learned peaks per model,
/// persisted as JSON. Thread-safe.
class MemoryBook {
public:
  /// A learned peak of one dedicated-memory layout (BIOS VRAM split).
  struct Learned {
    MemoryNeed need;
    std::string at;
    bool complete{false};
  };
  struct Record {
    bool has_estimate{false};
    MemoryNeed estimate;
    bool exclusive{false};
    bool has_learned{false};
    MemoryNeed learned;
    std::string learned_at;
    /// The peak comes from a worker that finished loading and served until a
    /// normal stop; from then on it replaces the estimate.
    bool learned_complete{false};
    /// Dedicated GPU memory (whole GiB) the learned peak was measured with;
    /// 0 = unknown (files written before the layout was recorded).
    int learned_layout_gb{0};
    /// Peaks learned with other dedicated sizes, by GiB (0 = unknown). Kept
    /// when the BIOS split changes and restored when it changes back.
    std::map<int, Learned> other_layouts;
    /// BuiltInFloor (not persisted): Need() is never below it.
    bool has_floor{false};
    MemoryNeed floor;
  };

  /// Empty path: nothing is read or written.
  explicit MemoryBook(std::filesystem::path file);

  /// Reads the file if it exists; returns false + error on a malformed file.
  bool Load(std::string* error);
  /// Writes when something changed since the last save (temp file + rename).
  bool Save(std::string* error);

  void SetEstimate(std::string_view id, const MemoryNeed& estimate);
  void SetExclusive(std::string_view id, bool exclusive);
  /// The least Need() returns for `id`, whatever was estimated or learned.
  void SetFloor(std::string_view id, const MemoryNeed& floor);
  /// Raises the learned peak; true when it grew by more than 256 MiB.
  bool Observe(std::string_view id, const ProcessMemory& usage);
  /// Raises the learned peak to at least `need` (watchdog abort).
  void RaiseLearned(std::string_view id, const MemoryNeed& need);
  /// A worker of `id` ran through load and use and was stopped normally.
  void MarkComplete(std::string_view id);
  /// The adapter's dedicated memory. When it differs from the size a peak was
  /// learned with, that peak moves to other_layouts and the model learns
  /// anew; until then Need() derives from the old peak (see there).
  void SetLayout(std::int64_t gpu_dedicated_total);

  /// Learned peak x learned_factor once a complete run was measured;
  /// before that max(estimate x factor, learned x learned_factor). Nothing
  /// learned with the current layout but with another: that peak x
  /// learned_factor, where a shared part that now fits into dedicated memory
  /// no longer counts as host RAM. A model with neither (and no floor) is
  /// exclusive. Never below SetFloor's need (gpu, host).
  [[nodiscard]] MemoryNeed Need(std::string_view id, double factor,
                                double learned_factor, bool* exclusive) const;
  [[nodiscard]] Record Get(std::string_view id) const;
  [[nodiscard]] const std::filesystem::path& file() const { return file_; }

private:
  std::filesystem::path file_;
  mutable std::mutex mutex_;
  std::map<std::string, Record, std::less<>> records_;
  bool dirty_{false};
  int layout_gb_{0};

  void AdoptLayoutLocked(Record& record);
};

[[nodiscard]] double ToGb(std::int64_t bytes);
[[nodiscard]] std::int64_t FromGb(double gb);

}  // namespace gufo::router

#endif  // GUFO_CLI_SERVE_ROUTER_MEMORY_HPP_
