#ifndef GUFO_CLI_SERVE_ROUTER_COMFY_PARALLEL_HPP_
#define GUFO_CLI_SERVE_ROUTER_COMFY_PARALLEL_HPP_

// How many ComfyUI prompts the adapter runs at the same time (2.2.6, Fabian
// 18.11.: "Parallel sollte je Rechner gelten"): its own --comfy-parallel
// plus the free share of the reachable pool partner. A prompt's requests go
// through the router's pool like any other, so a prompt beyond the own
// number computes on the partner. The partner tells its number in GET
// /router/capabilities ("comfy_parallel", "comfy_queue_remaining"); an
// older partner without the field, an unreachable or a draining one adds
// nothing. Pure code, no locks or sockets.

#include <algorithm>
#include <cstddef>
#include <optional>

namespace gufo::router::comfy {

/// What the partner's GET /router/capabilities says about its ComfyUI front.
struct PartnerComfy {
  /// Its own prompts at the same time (its --comfy-parallel, else
  /// --jobs-parallel, at least 1).
  std::size_t parallel{0};
  /// Its own prompts running or waiting now (its ComfyUI queue_remaining).
  std::size_t queue_remaining{0};
  /// It takes no new pool work (maintenance drain, 2.1.8).
  bool draining{false};
};

/// The partner's free share: its parallel minus its own queue, 0 while it
/// drains. Its own prompts make it "GPU busy" for the pool, so a slot it
/// uses itself is no slot for us.
[[nodiscard]] inline std::size_t PartnerComfyShare(
    const std::optional<PartnerComfy>& partner) {
  if (!partner.has_value() || partner->draining)
    return 0;
  return partner->parallel > partner->queue_remaining
             ? partner->parallel - partner->queue_remaining
             : 0;
}

/// Own parallel (0 counts as 1) + PartnerComfyShare. Never below the own
/// number: the partner going away lowers the limit back to it, running
/// prompts are not stopped.
[[nodiscard]] inline std::size_t ComfyParallelLimit(
    std::size_t own, const std::optional<PartnerComfy>& partner) {
  return std::max<std::size_t>(1, own) + PartnerComfyShare(partner);
}

}  // namespace gufo::router::comfy

#endif  // GUFO_CLI_SERVE_ROUTER_COMFY_PARALLEL_HPP_
