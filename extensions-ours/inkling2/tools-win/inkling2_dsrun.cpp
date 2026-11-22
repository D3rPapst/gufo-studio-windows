// gufo_windows addition: end-to-end DSpark speculative decoding on the GPU (target chain verify + GPU drafter),
// lossless check against plain greedy decoding, measured speed. Prototype of the runner policy.
//
//   inkling2_dsrun --model GGUF --drafter DSPARK.gguf --prompt FILE [--chat 1] [--gen 256] [--ctx 8192]
//                  [--policies 3,5,7,a] [--adaptive-set 0,3,5] [--window-extra 0]
//
// Prompt processing for both runs goes through chains of 8 tokens (decode kernels, feature capture on), so the
// plain reference and the speculative run start from bit-identical states. Plain: single-token decode (no capture).
// Speculative: per cycle draft K tokens after the bonus token, verify [bonus, d1..dK] as one chain, keep the
// accepted prefix (Engine::ChainKeep, conv histories from the chain's per-token log), inject the features of the
// kept rows into the drafter. Policy "a": per-token acceptance tracked as an EMA, K chosen from --adaptive-set by
// the measured cost model (draft ms + chain ms of K + 1 tokens vs plain decode ms), plain decode below break-even
// with a K=3 probe every 16 plain tokens.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>

#include "src/dspark.hpp"
#include "src/engine.hpp"
#include "text/tokenizer.hpp"

namespace {
using gufo::models::inkling2::DSparkDrafter;
using gufo::models::inkling2::Engine;
using gufo::models::inkling2::Session;
using Clock = std::chrono::steady_clock;

double Ms(Clock::time_point a) { return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); }

std::int32_t Argmax(const std::vector<float>& v, int n) {
  return static_cast<std::int32_t>(std::max_element(v.begin(), v.begin() + n) - v.begin());
}

std::vector<std::string> Split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) if (!t.empty()) out.push_back(t);
  return out;
}

// prompt through chains of up to 8 tokens (decode kernels); features of every position into the drafter
void FeedPrompt(Engine& e, Session& s, const std::vector<std::int32_t>& ids, DSparkDrafter* d) {
  std::size_t i = 0;
  std::vector<std::vector<float>> rows;
  while (i < ids.size()) {
    const std::size_t n = std::min<std::size_t>(8, ids.size() - i);
    if (n >= 2) e.VerifyChain(s, std::span<const std::int32_t>(ids.data() + i, n), &rows);
    else e.Forward(s, std::span<const std::int32_t>(ids.data() + i, 1));
    if (d) d->Inject(i, n);
    i += n;
  }
}

struct Cost {
  double decode_ms = 22.4;
  std::map<int, double> chain_ms{{2, 30.2}, {3, 34.8}, {4, 39.5}, {5, 44.0}, {6, 48.5}, {7, 53.0}, {8, 57.5}};
  std::map<int, double> draft_ms{{1, 6.0}, {3, 8.0}, {5, 10.0}, {7, 12.0}};
  static void Ema(double* v, double x) { *v = 0.8 * *v + 0.2 * x; }
  [[nodiscard]] double Draft(int k) const {
    auto it = draft_ms.lower_bound(k);
    return it == draft_ms.end() ? draft_ms.rbegin()->second : it->second;
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::string model, drafter_path, prompt_path, policies = "3,5,7,a", aset = "0,3,5";
  int gen = 256, chat = 1, window_extra = 0;
  unsigned ctx = 8192;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (k == "--model") model = next();
    else if (k == "--drafter") drafter_path = next();
    else if (k == "--prompt") prompt_path = next();
    else if (k == "--gen") gen = std::atoi(next().c_str());
    else if (k == "--chat") chat = std::atoi(next().c_str());
    else if (k == "--ctx") ctx = static_cast<unsigned>(std::strtoul(next().c_str(), nullptr, 10));
    else if (k == "--policies") policies = next();
    else if (k == "--adaptive-set") aset = next();
    else if (k == "--window-extra") window_extra = std::atoi(next().c_str());
    else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
  }
  std::string err, text;
  {
    std::ifstream f(prompt_path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    text = ss.str();
  }
  auto tok = inkling2::text::Tokenizer::CreateFromGguf(model, &err);
  if (!tok) { std::fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1; }
  std::vector<std::int32_t> ids;
  {
    const std::string full = chat ? "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>" +
                                        text + "<|end_message|>"
                                  : text;
    const auto e = tok->Encode(full, chat != 0);
    ids.assign(e.begin(), e.end());
  }
  gufo::core::GgufOpenOptions open;
  open.allow_unsized_tensors = true;
  std::shared_ptr<const gufo::core::GgufReader> reader = gufo::core::GgufReader::OpenFile(model, &err, open);
  if (!reader) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
  auto engine = Engine::Create(reader, ctx, &err);
  if (!engine) { std::fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
  if (!engine->batch_capable()) { std::fprintf(stderr, "engine not batch capable\n"); return 1; }
  const int V = static_cast<int>(engine->config().n_vocab_text);
  engine->SetFeatureCapture(true);
  const auto tl = Clock::now();
  DSparkDrafter::Options dopt;
  dopt.window_extra = window_extra;
  auto drafter = DSparkDrafter::Create(*engine, drafter_path, dopt, &err);
  if (!drafter) { std::fprintf(stderr, "drafter: %s\n", err.c_str()); return 1; }
  std::printf("prompt %zu tokens, gen %d, drafter loaded in %.1f s\n", ids.size(), gen, Ms(tl) / 1000.0);

  // ---- warm-up: capture the decode graphs (single token with / without features, chains of 2..8) outside the timing
  {
    auto w = engine->NewSession();
    std::vector<std::int32_t> t(8, ids.empty() ? 0 : ids[0]);
    std::vector<std::vector<float>> rows;
    for (int n = 2; n <= 8; ++n) { engine->Reset(*w); engine->VerifyChain(*w, std::span<const std::int32_t>(t.data(), n), &rows); }
    engine->Reset(*w);
    engine->Forward(*w, std::span<const std::int32_t>(t.data(), 1));
    engine->Forward(*w, std::span<const std::int32_t>(t.data(), 1));
    engine->SetFeatureCapture(false);
    engine->Forward(*w, std::span<const std::int32_t>(t.data(), 1));
    engine->SetFeatureCapture(true);
  }

  // ---- plain reference (prompt via chains, then single-token greedy decode without capture)
  std::vector<std::int32_t> ref;
  Cost cost;
  {
    auto s = engine->NewSession();
    FeedPrompt(*engine, *s, ids, nullptr);
    engine->SetFeatureCapture(false);
    std::int32_t nx = Argmax(s->logits(), V);
    const auto t0 = Clock::now();
    for (int g = 0; g < gen; ++g) {
      ref.push_back(nx);
      if (g + 1 == gen) break;
      engine->Forward(*s, std::span<const std::int32_t>(&nx, 1));
      nx = Argmax(s->logits(), V);
    }
    const double ms = Ms(t0);
    cost.decode_ms = ms / std::max(1, gen - 1);
    std::printf("plain: %d tokens, %.2f tok/s (%.2f ms/token)\n", gen, 1000.0 / cost.decode_ms, cost.decode_ms);
    engine->SetFeatureCapture(true);
    std::string txt = tok->Decode(std::span<const inkling2::text::TokenId>(reinterpret_cast<const inkling2::text::TokenId*>(ref.data()), ref.size()));
    for (char& c : txt) if (c == '\n') c = ' ';
    std::printf("reference text: %.240s\n", txt.c_str());
  }

  // ---- speculative runs
  const std::vector<int> adaptive_set = [&] {
    std::vector<int> v;
    for (const auto& x : Split(aset)) v.push_back(std::atoi(x.c_str()));
    return v;
  }();
  for (const auto& pol : Split(policies)) {
    const bool adaptive = pol == "a";
    const int fixed_k = adaptive ? 0 : std::min(7, std::atoi(pol.c_str()));
    auto s = engine->NewSession();
    drafter->Reset(0);
    FeedPrompt(*engine, *s, ids, drafter.get());
    std::vector<std::int32_t> out;
    std::int32_t nx = Argmax(s->logits(), V);
    std::size_t c = s->position();
    int cycles = 0, plain_steps = 0, drafted = 0, accepted = 0, since_probe = 0;
    double draft_total = 0, verify_total = 0, alpha = 0.6;
    std::map<int, int> k_used;
    const auto t0 = Clock::now();
    while (static_cast<int>(out.size()) < gen) {
      int K = fixed_k;
      if (adaptive) {
        double best = 1.0 / cost.decode_ms;  // tokens per ms of plain decode
        K = 0;
        for (int k : adaptive_set) {
          if (k <= 0) continue;
          double e = 0, p = 1;
          for (int i = 0; i <= k; ++i) { e += p; p *= alpha; }
          const double ms = cost.Draft(k) + cost.chain_ms[std::min(k + 1, 8)];
          if (e / ms > best) { best = e / ms; K = k; }
        }
        if (K == 0 && ++since_probe >= 16) { K = 3; since_probe = 0; }  // probe so the estimate can recover
      }
      K = std::min(K, gen - static_cast<int>(out.size()) - 1);
      std::vector<std::int32_t> d;
      if (K > 0) {
        d = drafter->Draft(nx, c, K);
        draft_total += drafter->last_draft_ms();
        Cost::Ema(&cost.draft_ms[K], drafter->last_draft_ms());
      }
      if (d.empty()) {
        out.push_back(nx);
        engine->Forward(*s, std::span<const std::int32_t>(&nx, 1));
        drafter->Inject(c, 1);
        ++c;
        nx = Argmax(s->logits(), V);
        ++plain_steps;
        continue;
      }
      std::vector<std::int32_t> step{nx};
      step.insert(step.end(), d.begin(), d.end());
      std::vector<std::vector<float>> rows;
      const auto tv = Clock::now();
      engine->VerifyChain(*s, step, &rows);
      const double vms = Ms(tv);
      verify_total += vms;
      Cost::Ema(&cost.chain_ms[static_cast<int>(step.size())], vms);
      std::size_t a = 0;
      while (a < d.size() && Argmax(rows[a], V) == d[a]) ++a;
      for (std::size_t i = 0; i < a; ++i) Cost::Ema(&alpha, 1.0);  // per-token acceptance: a hits, then one miss
      if (a < d.size()) Cost::Ema(&alpha, 0.0);
      if (a + 1 < step.size()) engine->ChainKeep(*s, a + 1, rows[a]);
      drafter->Inject(c, a + 1);
      for (std::size_t i = 0; i <= a; ++i) out.push_back(step[i]);
      c += a + 1;
      nx = Argmax(rows[a], V);
      ++cycles;
      ++k_used[static_cast<int>(d.size())];
      drafted += static_cast<int>(d.size());
      accepted += static_cast<int>(a);
    }
    const double ms = Ms(t0);
    out.resize(static_cast<std::size_t>(gen));
    int same = 0;
    while (same < gen && out[same] == ref[same]) ++same;
    std::printf("policy %s: %.2f tok/s (plain %.2f), identical to plain for %d of %d tokens%s; %d cycles + %d plain steps,"
                " drafted %d accepted %d (%.0f%%), %.2f tokens/cycle, draft %.1f ms/cycle, verify %.1f ms/cycle, alpha %.2f, K used:",
                pol.c_str(), gen * 1000.0 / ms, 1000.0 / cost.decode_ms, same, gen, same == gen ? " (lossless)" : " (MISMATCH)",
                cycles, plain_steps, drafted, accepted, drafted ? 100.0 * accepted / drafted : 0.0,
                cycles ? static_cast<double>(gen - plain_steps) / cycles : 0.0, cycles ? draft_total / cycles : 0.0,
                cycles ? verify_total / cycles : 0.0, alpha);
    for (const auto& [k, n] : k_used) std::printf(" %d:%d", k, n);
    std::printf("\n");
  }
  std::printf("chain cost (EMA ms):");
  for (const auto& [t, m] : cost.chain_ms) std::printf(" T%d=%.1f", t, m);
  std::printf("; draft cost (EMA ms):");
  for (const auto& [k, m] : cost.draft_ms) std::printf(" K%d=%.1f", k, m);
  std::printf("\n");
  return 0;
}
