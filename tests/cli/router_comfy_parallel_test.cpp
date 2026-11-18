// 2.2.6: how many ComfyUI prompts the adapter runs at once: its own
// --comfy-parallel plus the free share of the reachable pool partner
// (comfy_parallel.hpp). Pure code, no server.

#include <chrono>
#include <cstddef>
#include <iostream>
#include <optional>

#include "src/cli/serve/router/comfy_parallel.hpp"
#include "src/cli/serve/router/pool.hpp"

namespace comfy = gufo::router::comfy;
namespace router = gufo::router;

namespace {

/// An available machine with the model loaded and nothing in flight.
router::PoolSide Free() {
  router::PoolSide side;
  side.available = true;
  side.warm = true;
  side.free = true;
  side.idle = true;
  side.empty = true;
  side.sticky = true;
  return side;
}

int failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond \
                << "\n";                                                   \
      ++failures;                                                          \
    }                                                                      \
  } while (0)

comfy::PartnerComfy Partner(std::size_t parallel, std::size_t queue = 0,
                            bool draining = false) {
  comfy::PartnerComfy partner;
  partner.parallel = parallel;
  partner.queue_remaining = queue;
  partner.draining = draining;
  return partner;
}

}  // namespace

int main() {
  // Fabian 18.11.: 2 on each machine -> 4 at once.
  CHECK(comfy::ComfyParallelLimit(2, Partner(2)) == 4);
  CHECK(comfy::PartnerComfyShare(Partner(2)) == 2);
  // Different numbers per machine add up.
  CHECK(comfy::ComfyParallelLimit(1, Partner(3)) == 4);
  CHECK(comfy::ComfyParallelLimit(3, Partner(1)) == 4);
  // No partner, unreachable or older without "comfy_parallel": own only.
  CHECK(comfy::ComfyParallelLimit(2, std::nullopt) == 2);
  CHECK(comfy::PartnerComfyShare(std::nullopt) == 0);
  // Own 0 counts as 1 (as AdapterOptions::parallel).
  CHECK(comfy::ComfyParallelLimit(0, std::nullopt) == 1);
  CHECK(comfy::ComfyParallelLimit(0, Partner(2)) == 3);
  // A draining partner takes no new pool work.
  CHECK(comfy::ComfyParallelLimit(2, Partner(2, 0, true)) == 2);
  // The partner's own prompts use its slots: only the rest is ours.
  CHECK(comfy::ComfyParallelLimit(2, Partner(2, 1)) == 3);
  CHECK(comfy::ComfyParallelLimit(2, Partner(2, 2)) == 2);
  // More queued there than it runs: nothing, never below our own.
  CHECK(comfy::ComfyParallelLimit(2, Partner(2, 7)) == 2);
  CHECK(comfy::PartnerComfyShare(Partner(2, 7)) == 0);
  // A partner that says 0 adds nothing.
  CHECK(comfy::ComfyParallelLimit(2, Partner(0)) == 2);

  // The adapter's requests (X-Gufo-Origin: comfy) wait without the queue
  // timeout: 7-27 min film clips on both machines must not end in 503.
  CHECK(router::ComfyOriginWait("comfy"));
  CHECK(!router::ComfyOriginWait(""));
  CHECK(!router::ComfyOriginWait("pages"));
  CHECK(router::PoolQueueTimeout(900, false) == std::chrono::seconds(900));
  CHECK(router::PoolQueueTimeout(900, true) >= std::chrono::hours(24));
  CHECK(router::PoolQueueTimeout(0, true) > std::chrono::seconds(27 * 60));

  // An extra prompt starts only when its model finds a machine.
  {
    router::PoolInput in;
    in.local = Free();
    in.remote = Free();
    CHECK(router::PoolCanStartSoon(in));
    in.local.free = false;  // a request of it runs here
    in.local.idle = false;
    CHECK(router::PoolCanStartSoon(in));  // the partner is free
    in.remote.free = false;
    in.remote.idle = false;
    CHECK(!router::PoolCanStartSoon(in));  // both busy: hold it back
    in.remote = router::PoolSide{};        // partner gone
    CHECK(!router::PoolCanStartSoon(in));
    in.local = Free();
    CHECK(router::PoolCanStartSoon(in));
    // GPU work on both machines (another image/video model busy there).
    in.local.gpu_busy = true;
    in.remote = Free();
    in.remote.gpu_busy = true;
    CHECK(!router::PoolCanStartSoon(in));
    // Cold on the partner while this machine had the model (hysteresis):
    // after kPoolSpillAfter it may load there, so it counts as free.
    in = router::PoolInput{};
    in.local = Free();
    in.local.free = false;
    in.local.idle = false;
    in.local.sticky = true;
    in.remote = Free();
    in.remote.warm = false;
    in.remote.sticky = false;
    CHECK(router::ChoosePool(in).choice == router::PoolChoice::kWait);
    CHECK(router::PoolCanStartSoon(in));
  }

  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "router_comfy_parallel_test: ok\n";
  return 0;
}
