#ifndef GUFO_CLI_SERVE_ROUTER_CPU_POOL_HPP_
#define GUFO_CLI_SERVE_ROUTER_CPU_POOL_HPP_

// The router's own CPU services (POST /v1/vision/detect and POST /v1/figuren
// "verfahren": "ccip") share one FIFO queue on this machine and hand work to
// the pool partner when this machine is busy. Written with AI assistance
// (Claude Code).
//
// Queue: one request computes at a time; the others wait in arrival order
// (no fight for the cores between the two services). A caller that
// disconnects while waiting leaves the line; more than `max_waiting`
// waiting requests are refused (503 busy).
//
// Pool (`Decide`): a request goes to the partner router only when
//  - it was not relayed here already (X-Gufo-Forwarded: loop guard),
//  - the caller did not pin it (X-Gufo-Pool: local),
//  - a partner is configured,
//  - and this machine is busy (something computes or waits) or cannot run
//    it (models missing).
// It is sent with X-Gufo-If-Idle: 1; the partner answers 503 busy at once
// when its own queue is not idle, and the request then queues here. A
// partner that cannot be reached, answers 404 (older router), 502, 503 or
// 504 counts as "not there": the request runs here. X-Gufo-Pool: remote
// sends it to the partner without the idle condition (503
// remote_unavailable when there is none). Every input image must be inline
// (data URL) in the relayed body: the partner cannot read this machine's
// Ablage.

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>

#include "src/cli/serve/http_server.hpp"

namespace gufo::router::cpu {

/// Set on a relayed CPU request: run only when the queue is idle, else
/// answer 503 busy at once.
inline constexpr const char* kIfIdleHeader = "X-Gufo-If-Idle";
/// Set on an answer the partner computed ("partner") so callers and logs
/// see where it ran.
inline constexpr const char* kRanOnHeader = "X-Gufo-Ran-On";

class CpuQueue {
public:
  enum class Result { kOk, kFull, kCancelled };

  explicit CpuQueue(std::size_t max_waiting = 4) : max_waiting_(max_waiting) {}

  /// Waits until this caller is first and nothing computes.
  Result Enter(const std::function<bool()>& cancelled);
  void Leave();
  /// Nothing computes and nobody waits.
  [[nodiscard]] bool Idle() const;
  [[nodiscard]] std::size_t Waiting() const;
  [[nodiscard]] bool Busy() const;

  /// Leaves the queue on every path out of a compute section.
  class Turn {
  public:
    explicit Turn(CpuQueue& queue) : queue_(queue) {}
    Turn(const Turn&) = delete;
    Turn& operator=(const Turn&) = delete;
    ~Turn() { queue_.Leave(); }

  private:
    CpuQueue& queue_;
  };

private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::uint64_t> line_;
  std::uint64_t next_{0};
  bool busy_{false};
  std::size_t max_waiting_;
};

/// Sends a request to the pool partner and collects the answer; nullopt
/// when there is no partner or it cannot be reached. The router fills it
/// (remote::RemoteTarget, X-Gufo-Forwarded, pool key).
using Relay = std::function<std::optional<server::HttpResponse>(const server::HttpRequest&)>;

enum class Route {
  kLocal,         // run here
  kLocalIfIdle,   // relayed here with X-Gufo-If-Idle: run only when idle
  kTryPartner,    // ask the partner first (if idle there), else run here
  kPartnerOnly,   // X-Gufo-Pool: remote
};

/// The decision above. `local_busy`: !queue.Idle(); `local_missing`: the
/// models of the request are not available here.
[[nodiscard]] Route Decide(const server::HttpRequest& request, bool have_partner,
                           bool local_busy, bool local_missing);

/// Whether a partner answer counts (returned to the caller) or means "not
/// there" (run here instead): 404, 502, 503 and 504 do not count.
[[nodiscard]] bool PartnerAnswered(const server::HttpResponse& answer);

/// 503 busy with Retry-After: 5 (queue full, or not idle for a relayed
/// request).
[[nodiscard]] server::HttpResponse Busy(const std::string& what);

/// The relayed copy of `request` with `body`: method and path kept,
/// Content-Type JSON, X-Gufo-If-Idle unless `force`; no Authorization or
/// cookies of the caller (the relay adds the partner key).
[[nodiscard]] server::HttpRequest RelayRequest(const server::HttpRequest& request,
                                               std::string body, bool force);

}  // namespace gufo::router::cpu

#endif  // GUFO_CLI_SERVE_ROUTER_CPU_POOL_HPP_
