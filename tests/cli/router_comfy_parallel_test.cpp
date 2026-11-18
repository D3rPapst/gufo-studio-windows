// 2.2.6: how many ComfyUI prompts the adapter runs at once: its own
// --comfy-parallel plus the free share of the reachable pool partner
// (comfy_parallel.hpp). Pure code, no server.

#include <cstddef>
#include <iostream>
#include <optional>

#include "src/cli/serve/router/comfy_parallel.hpp"

namespace comfy = gufo::router::comfy;

namespace {

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

  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "router_comfy_parallel_test: ok\n";
  return 0;
}
