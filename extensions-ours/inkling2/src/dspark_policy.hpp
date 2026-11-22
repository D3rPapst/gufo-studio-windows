// gufo_windows addition: speculation depth policy for DSpark (per request). Header only, no HIP.
//
// Decides per cycle between plain decoding (K = 0) and a draft of K tokens verified as one chain of K + 1.
// - Acceptance: per-token hit rate over the last `window` observations (cycles and shadow probes), with a weak
//   prior; one cycle of a hits out of K drafted gives a hits and (a < K) one miss.
// - Cost model: measured times (EMA) of plain decode, draft(K) and chain(T); expected tokens per cycle
//   E(K) = 1 + p + ... + p^K; rate(K) = E(K) / (draft(K) + chain(K + 1)).
// - Hysteresis: speculate while the best rate beats plain by `leave` (0.97); go back to speculating only when it
//   beats plain by `enter` (1.05).
// - In plain mode a shadow draft every `probe_every` tokens (cost: one draft, no verify) is compared with the
//   tokens plain decoding then produces; that keeps the estimate alive without paying for failed verifies.
#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <utility>
#include <vector>

namespace gufo::models::inkling2 {

struct DSparkPolicyParams {
  std::vector<int> depths{2, 3, 4, 5};  // candidate K (chain K + 1 <= 8)
  int window = 8;                       // observations (cycles / probes) in the acceptance estimate
  double enter = 1.05, leave = 0.97;    // hysteresis against plain decode
  int probe_every = 32;                 // plain tokens between shadow drafts
  int probe_k = 3;
  double start_p = 0.7;                 // prior hit rate (first cycles speculate with K = 3)
};

class DSparkPolicy {
public:
  using Params = DSparkPolicyParams;
  DSparkPolicy() : DSparkPolicy(Params{}) {}
  explicit DSparkPolicy(Params p) : p_(std::move(p)) {
    chain_ms_ = {{2, 30.5}, {3, 34.9}, {4, 39.9}, {5, 48.8}, {6, 55.1}, {7, 61.4}, {8, 67.0}};
    speculating_ = true;
  }

  /// K for the next cycle (0 = plain decode step).
  [[nodiscard]] int Choose() {
    const double plain = 1.0 / decode_ms_;
    double best = 0;
    int best_k = 0;
    for (int k : p_.depths) {
      const double r = Rate(k);
      if (r > best) { best = r; best_k = k; }
    }
    // stay at the current depth unless another one is clearly (3 %) better
    if (cur_k_ > 0 && std::find(p_.depths.begin(), p_.depths.end(), cur_k_) != p_.depths.end() &&
        Rate(cur_k_) * 1.03 >= best) {
      best_k = cur_k_;
      best = Rate(cur_k_);
    }
    speculating_ = speculating_ ? best > plain * p_.leave : best > plain * p_.enter;
    cur_k_ = speculating_ ? best_k : 0;
    return cur_k_;
  }
  /// In plain mode: true when a shadow draft is due now (then call Probe with its result later).
  [[nodiscard]] bool ProbeDue() {
    if (speculating_) return false;
    if (++since_probe_ < p_.probe_every) return false;
    since_probe_ = 0;
    return true;
  }
  [[nodiscard]] int probe_k() const { return p_.probe_k; }

  // Costs: the chain benchmark (T = 2..8) plus ONE learned overhead, and one learned draft time (+0.3 ms per
  // token). Round 9: learning per depth kept the unused depths at their optimistic start values, so the policy
  // drifted to K = 2 on code although K = 3 was faster.
  void Cycle(int k, int accepted, double draft_ms, double chain_ms) {
    Observe(k, accepted);
    Ema(&draft_base_, draft_ms - 0.3 * k);
    Ema(&chain_over_, chain_ms - Chain(k + 1));
  }
  void Probe(int k, int accepted, double draft_ms) {
    Observe(k, accepted);
    Ema(&draft_base_, draft_ms - 0.3 * k);
  }
  [[nodiscard]] double Rate(int k) const {  // expected tokens per ms at depth k
    const double hit = Hit();
    double e = 0, q = 1;
    for (int i = 0; i <= k; ++i) { e += q; q *= hit; }
    return e / (draft_base_ + 0.3 * k + Chain(k + 1) + chain_over_);
  }
  void Plain(double ms) { Ema(&decode_ms_, ms); }

  [[nodiscard]] double Hit() const {
    double hits = p_.start_p * 2, total = 2;  // prior worth two observations
    for (const auto& [h, m] : obs_) { hits += h; total += h + m; }
    return std::clamp(hits / total, 0.0, 0.99);
  }
  [[nodiscard]] bool speculating() const { return speculating_; }

private:
  void Observe(int k, int a) {
    obs_.push_back({a, a < k ? 1 : 0});
    while (static_cast<int>(obs_.size()) > p_.window) obs_.pop_front();
  }
  [[nodiscard]] double Chain(int t) const {
    auto it = chain_ms_.find(std::min(t, 8));
    return it == chain_ms_.end() ? 67.0 : it->second;
  }
  static void Ema(double* v, double x) { *v = 0.7 * *v + 0.3 * x; }

  Params p_;
  std::deque<std::pair<int, int>> obs_;
  std::map<int, double> chain_ms_;
  double draft_base_ = 6.5, chain_over_ = 2.0;
  int cur_k_ = 0;
  double decode_ms_ = 22.5;
  bool speculating_ = true;
  int since_probe_ = 0;
};

}  // namespace gufo::models::inkling2
