// gufo_windows addition: end-to-end DSpark speculative decoding on the GPU (target chain verify + GPU drafter),
// lossless check against plain greedy decoding, measured speed. Prototype of the runner path.
//
//   inkling2_dsrun --model GGUF --drafter DSPARK.gguf --prompt FILE [--chat 1] [--gen 256] [--ctx 8192]
//                  [--policies 3,5,7,a] [--window-extra 0]
//
// Prompt: one prefill Forward (the drafter gets its features through Engine::SetFeatureSink). Plain reference:
// single-token greedy decode. Speculative: per cycle draft K tokens after the bonus token, verify
// [bonus, d1..dK] as one chain with GPU argmax (VerifyChainArgmax), keep the accepted prefix (ChainKeep: conv
// histories from the chain's per-token log, one logits row copied back); the sink stages the chain's feature rows
// and the drafter injects them before the next draft. Policy "a" = DSparkPolicy (src/dspark_policy.hpp).
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

#include <hip/hip_runtime.h>

#include <gufo/ext/gguf_reader.hpp>

#include "src/dspark.hpp"
#include "src/dspark_policy.hpp"
#include "src/engine.hpp"
#include "text/tokenizer.hpp"

namespace {
using gufo::models::inkling2::DSparkDrafter;
using gufo::models::inkling2::DSparkPolicy;
using gufo::models::inkling2::Engine;
using gufo::models::inkling2::Session;
using Clock = std::chrono::steady_clock;

double Ms(Clock::time_point a) { return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); }

std::int32_t Argmax(const std::vector<float>& v, int n) {
  return static_cast<std::int32_t>(std::max_element(v.begin(), v.begin() + n) - v.begin());
}

// prompt: one prefill Forward, chains of 8 (decode kernels) or single decode steps
void FeedPrompt(Engine& e, Session& s, const std::vector<std::int32_t>& ids, const std::string& path) {
  if (path == "prefill") { e.Forward(s, ids); return; }
  std::vector<std::int32_t> am;
  std::size_t i = 0;
  while (i < ids.size()) {
    const std::size_t n = path == "chain" ? std::min<std::size_t>(8, ids.size() - i) : 1;
    if (n >= 2) e.VerifyChainArgmax(s, std::span<const std::int32_t>(ids.data() + i, n), &am), e.ChainKeep(s, n);
    else e.Forward(s, std::span<const std::int32_t>(ids.data() + i, 1));
    i += n;
  }
}

double Cos(const float* a, const float* b, int n) {
  double ab = 0, aa = 0, bb = 0;
  for (int i = 0; i < n; ++i) { ab += double(a[i]) * b[i]; aa += double(a[i]) * a[i]; bb += double(b[i]) * b[i]; }
  return ab / std::sqrt(aa * bb + 1e-30);
}

std::vector<std::string> Split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) if (!t.empty()) out.push_back(t);
  return out;
}
}  // namespace

int main(int argc, char** argv) {
  std::string model, drafter_path, prompt_path, policies = "3,5,7,a", prompt_path_mode = "prefill";
  int gen = 256, chat = 1, window_extra = 0, stage = 0, feature_diff = 0;  // stage 0 = inject at once (round 8)
  double min_accept = 0;
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
    else if (k == "--window-extra") window_extra = std::atoi(next().c_str());
    else if (k == "--prompt-path") prompt_path_mode = next();
    else if (k == "--stage") stage = std::atoi(next().c_str());
    else if (k == "--feature-diff") feature_diff = std::atoi(next().c_str());
    else if (k == "--min-accept") min_accept = std::atof(next().c_str());
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
  const auto tl = Clock::now();
  DSparkDrafter::Options dopt;
  dopt.window_extra = window_extra;
  auto drafter = DSparkDrafter::Create(*engine, drafter_path, dopt, &err);
  if (!drafter) { std::fprintf(stderr, "drafter: %s\n", err.c_str()); return 1; }
  std::printf("prompt %zu tokens, gen %d, drafter loaded in %.1f s\n", ids.size(), gen, Ms(tl) / 1000.0);
  const Session* spec_session = nullptr;
  int exit_code = 0;
  auto sink = [&](const Session& s, std::size_t pos0, std::size_t n, const float* rows) {
    if (&s != spec_session) return;
    if (stage) drafter->Stage(pos0, n, rows);
    else drafter->InjectNow(pos0, n, rows);
  };

  // ---- warm-up: decode graphs with and without features, chains of 2..8, outside the timing
  {
    auto w = engine->NewSession();
    std::vector<std::int32_t> t(8, ids.empty() ? 0 : ids[0]), am;
    engine->SetFeatureSink(sink);
    for (int n = 2; n <= 8; ++n) { engine->Reset(*w); engine->VerifyChainArgmax(*w, std::span<const std::int32_t>(t.data(), n), &am); }
    engine->Reset(*w);
    engine->Forward(*w, std::span<const std::int32_t>(t.data(), 2));
    engine->Forward(*w, std::span<const std::int32_t>(t.data(), 1));
    engine->SetFeatureSink({});
    engine->Forward(*w, std::span<const std::int32_t>(t.data(), 1));
  }

  // ---- feature diff: the rows the drafter gets for the prompt from prefill / chains vs single decode steps
  if (feature_diff) {
    const int E = 2304;
    auto collect = [&](const std::string& path) {
      std::vector<float> host(ids.size() * 4 * E, 0.f);
      auto s = engine->NewSession();
      engine->SetFeatureSink([&](const Session& ss, std::size_t pos0, std::size_t n, const float* rows) {
        if (&ss != s.get() || pos0 + n > ids.size()) return;
        (void)hipMemcpy(host.data() + pos0 * 4 * E, rows, n * 4 * E * 4, hipMemcpyDeviceToHost);
      });
      FeedPrompt(*engine, *s, ids, path);
      engine->SetFeatureSink({});
      return host;
    };
    const auto dec = collect("decode"), pre = collect("prefill"), chn = collect("chain");
    for (const auto& [name, v] : {std::pair<const char*, const std::vector<float>*>{"prefill", &pre}, {"chain", &chn}}) {
      double mn[4] = {1, 1, 1, 1}, mean[4] = {0, 0, 0, 0};
      int worst[4] = {0, 0, 0, 0};
      for (std::size_t p = 0; p < ids.size(); ++p)
        for (int j = 0; j < 4; ++j) {
          const double c = Cos(v->data() + (p * 4 + j) * E, dec.data() + (p * 4 + j) * E, E);
          mean[j] += c / ids.size();
          if (c < mn[j]) { mn[j] = c; worst[j] = static_cast<int>(p); }
        }
      std::printf("feature diff %s vs decode (%zu positions): tap5 mean %.5f min %.5f@%d | tap23 %.5f %.5f@%d | tap41 %.5f %.5f@%d | final %.5f %.5f@%d\n",
                  name, ids.size(), mean[0], mn[0], worst[0], mean[1], mn[1], worst[1], mean[2], mn[2], worst[2], mean[3], mn[3], worst[3]);
    }
  }

  // ---- plain reference: prompt, then single-token greedy decode (no feature capture)
  std::vector<std::int32_t> ref;
  double decode_ms = 22.5;
  {
    auto s = engine->NewSession();
    FeedPrompt(*engine, *s, ids, prompt_path_mode);
    std::int32_t nx = Argmax(s->logits(), V);
    const auto t0 = Clock::now();
    for (int g = 0; g < gen; ++g) {
      ref.push_back(nx);
      if (g + 1 == gen) break;
      engine->Forward(*s, std::span<const std::int32_t>(&nx, 1));
      nx = Argmax(s->logits(), V);
    }
    decode_ms = Ms(t0) / std::max(1, gen - 1);
    std::printf("plain: %d tokens, %.2f tok/s (%.2f ms/token)\n", gen, 1000.0 / decode_ms, decode_ms);
    std::string txt = tok->Decode(std::span<const inkling2::text::TokenId>(reinterpret_cast<const inkling2::text::TokenId*>(ref.data()), ref.size()));
    for (char& c : txt) if (c == '\n') c = ' ';
    std::printf("reference text: %.240s\n", txt.c_str());
  }

  // ---- speculative runs
  engine->SetFeatureSink(sink);
  for (const auto& pol : Split(policies)) {
    const bool adaptive = pol == "a";
    const int fixed_k = adaptive ? 0 : std::min(7, std::atoi(pol.c_str()));
    DSparkPolicy policy;
    auto s = engine->NewSession();
    spec_session = s.get();
    drafter->Reset(0);
    FeedPrompt(*engine, *s, ids, prompt_path_mode);  // features through the sink
    std::vector<std::int32_t> out;
    std::int32_t nx = Argmax(s->logits(), V);
    int cycles = 0, plain_steps = 0, drafted = 0, accepted = 0, probes = 0;
    double draft_total = 0, verify_total = 0, keep_total = 0;
    std::map<int, int> k_used;
    // pending shadow draft (plain mode): drafted at position `probe_at` after anchor out[probe_idx]
    std::vector<std::int32_t> probe;
    std::size_t probe_idx = 0;
    double probe_ms = 0;
    const auto t0 = Clock::now();
    while (static_cast<int>(out.size()) < gen) {
      const std::size_t c = s->position();
      int K = adaptive ? policy.Choose() : fixed_k;
      K = std::min(K, gen - static_cast<int>(out.size()) - 1);
      std::vector<std::int32_t> d;
      if (K > 0) {
        d = drafter->Draft(nx, c, K);
        draft_total += drafter->last_draft_ms();
      }
      if (d.empty()) {  // plain step (with an optional shadow draft to keep the estimate alive)
        if (adaptive && probe.empty() && policy.ProbeDue()) {
          probe = drafter->Draft(nx, c, policy.probe_k());
          probe_ms = drafter->last_draft_ms();
          probe_idx = out.size();
          ++probes;
        }
        const auto tp = Clock::now();
        out.push_back(nx);
        engine->Forward(*s, std::span<const std::int32_t>(&nx, 1));
        nx = Argmax(s->logits(), V);
        policy.Plain(Ms(tp));
        ++plain_steps;
        if (!probe.empty() && out.size() + 1 >= probe_idx + 1 + probe.size()) {  // the probe's tokens are known now
          std::vector<std::int32_t> actual(out.begin() + static_cast<std::ptrdiff_t>(probe_idx) + 1, out.end());
          actual.push_back(nx);
          int a = 0;
          while (a < static_cast<int>(probe.size()) && a < static_cast<int>(actual.size()) && probe[a] == actual[a]) ++a;
          policy.Probe(static_cast<int>(probe.size()), a, probe_ms);
          probe.clear();
        }
        continue;
      }
      std::vector<std::int32_t> step{nx};
      step.insert(step.end(), d.begin(), d.end());
      std::vector<std::int32_t> am;
      const auto tv = Clock::now();
      engine->VerifyChainArgmax(*s, step, &am);
      std::size_t a = 0;
      while (a < d.size() && am[a] == d[a]) ++a;
      const auto tk = Clock::now();
      engine->ChainKeep(*s, a + 1);
      keep_total += Ms(tk);
      const double vms = Ms(tv);
      verify_total += vms;
      policy.Cycle(static_cast<int>(d.size()), static_cast<int>(a), drafter->last_draft_ms(), vms);
      for (std::size_t i = 0; i <= a; ++i) out.push_back(step[i]);
      nx = am[a];
      ++cycles;
      ++k_used[static_cast<int>(d.size())];
      drafted += static_cast<int>(d.size());
      accepted += static_cast<int>(a);
    }
    const double ms = Ms(t0);
    out.resize(static_cast<std::size_t>(gen));
    int same = 0;
    while (same < gen && out[same] == ref[same]) ++same;
    std::printf("policy %s: %.2f tok/s (plain %.2f), identical to plain for %d of %d tokens%s; %d cycles + %d plain steps"
                " (%d probes), drafted %d accepted %d (%.0f%%), %.2f tokens/cycle, per cycle: draft %.1f ms, verify+keep %.1f ms"
                " (keep %.2f ms); hit %.2f; K used:",
                pol.c_str(), gen * 1000.0 / ms, 1000.0 / decode_ms, same, gen, same == gen ? " (lossless)" : " (MISMATCH)",
                cycles, plain_steps, probes, drafted, accepted, drafted ? 100.0 * accepted / drafted : 0.0,
                cycles ? static_cast<double>(gen - plain_steps) / cycles : 0.0, cycles ? draft_total / cycles : 0.0,
                cycles ? verify_total / cycles : 0.0, cycles ? keep_total / cycles : 0.0, policy.Hit());
    for (const auto& [k, n] : k_used) std::printf(" %d:%d", k, n);
    std::printf(" [prompt %s, stage %d]\n", prompt_path_mode.c_str(), stage);
    if (!adaptive && min_accept > 0 && drafted > 0) {
      const double acc = 100.0 * accepted / drafted;
      std::printf("acceptance check K=%d: %.0f%% %s (minimum %.0f%%)\n", fixed_k, acc, acc >= min_accept ? "OK" : "FAIL", min_accept);
      if (acc < min_accept) exit_code = 4;
    }
  }
  engine->SetFeatureSink({});
  return exit_code;
}
