// gufo_windows addition (not part of headbouyJB/inkling2-gufo): A/B bench for lookup (n-gram)
// speculation on the Inkling 2 engine, plus the cost of a multi-token step (verify cost), which
// decides whether any speculation (lookup, DSpark) can pay off.
//
//   inkling2_spec --model GGUF --text FILE [--prompt 512] [--gen 256] [--k 0,1,2,3] [--ngram 3] [--ctx 8192]
//                 [--verify chain|prefill]
// --verify chain (default): Engine::VerifyChain, the batched decode graph with one session at consecutive
// positions (k <= 3); prefill: the multi-token Forward path (any k).
//
// 1 verify cost: after the prompt, time Forward() of T = 1..9 tokens (state restored after each).
// 2 greedy decode of --gen tokens for every k in --k (k = 0: plain single-token decode, the reference):
//   per cycle the last --ngram tokens are looked up in the history; a match proposes up to k tokens;
//   one Forward of [next, d1..dk] verifies them (logits of every row), accepted tokens are kept, on a
//   miss the conv history is restored and the accepted part is evaluated again (KV rows written past
//   the frontier are overwritten later; the window rings have ubatch rows of slack).
// Prints tok/s, acceptance and how many of the generated tokens equal the k = 0 run.
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include <gufo/ext/gguf_reader.hpp>

#include "src/engine.hpp"
#include "src/state_layout.hpp"
#include "text/tokenizer.hpp"

namespace {
using gufo::models::inkling2::Engine;
using gufo::models::inkling2::Session;
using gufo::models::inkling2::SessionStateAccess;
using Clock = std::chrono::steady_clock;

std::vector<int> ParseList(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) if (!t.empty()) out.push_back(std::atoi(t.c_str()));
  return out;
}

std::int32_t Argmax(const float* v, std::size_t n) {
  return static_cast<std::int32_t>(std::max_element(v, v + n) - v);
}

// Most recent earlier occurrence of the last `n` tokens of `h`; returns up to k following tokens.
std::vector<std::int32_t> Lookup(const std::vector<std::int32_t>& h, int n, int k) {
  std::vector<std::int32_t> d;
  if (k <= 0 || static_cast<int>(h.size()) <= n) return d;
  const std::size_t end = h.size() - static_cast<std::size_t>(n);
  for (std::size_t i = end; i-- > 0;) {
    if (std::equal(h.begin() + static_cast<std::ptrdiff_t>(i), h.begin() + static_cast<std::ptrdiff_t>(i + n),
                   h.begin() + static_cast<std::ptrdiff_t>(end))) {
      for (std::size_t j = i + n; j < h.size() && static_cast<int>(d.size()) < k; ++j) d.push_back(h[j]);
      break;
    }
  }
  return d;
}

struct Snapshot {
  void* conv = nullptr;
  std::size_t pos = 0;
  std::vector<float> logits;
};

void Save(const Session& s, Snapshot* snap) {
  SessionStateAccess::ConvSaveDevice(s, snap->conv);
  snap->pos = s.position();
  snap->logits = s.logits();
}
void Load(Session& s, const Snapshot& snap) {
  SessionStateAccess::ConvLoadDevice(s, snap.conv);
  SessionStateAccess::SetFrontier(s, snap.pos, snap.logits.data(), snap.logits.size());
}
}  // namespace

int main(int argc, char** argv) {
  std::string model, text, klist = "0,1,2,3", verify = "chain";
  int prompt_len = 512, gen = 256, ngram = 3;
  unsigned ctx = 8192;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (k == "--model") model = next();
    else if (k == "--text") text = next();
    else if (k == "--prompt") prompt_len = std::atoi(next().c_str());
    else if (k == "--gen") gen = std::atoi(next().c_str());
    else if (k == "--k") klist = next();
    else if (k == "--ngram") ngram = std::atoi(next().c_str());
    else if (k == "--verify") verify = next();
    else if (k == "--ctx") ctx = static_cast<unsigned>(std::strtoul(next().c_str(), nullptr, 10));
    else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
  }
  std::string err;
  auto tok = inkling2::text::Tokenizer::CreateFromGguf(model, &err);
  if (!tok) { std::fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1; }
  std::vector<std::int32_t> prompt;
  {
    std::ifstream f(text, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    const auto e = tok->Encode(ss.str().substr(0, 400000), false);
    prompt.assign(e.begin(), e.end());
  }
  if (static_cast<int>(prompt.size()) < prompt_len) { std::fprintf(stderr, "only %zu tokens\n", prompt.size()); return 1; }
  prompt.resize(static_cast<std::size_t>(prompt_len));

  gufo::core::GgufOpenOptions open;
  open.allow_unsized_tensors = true;
  std::shared_ptr<const gufo::core::GgufReader> reader = gufo::core::GgufReader::OpenFile(model, &err, open);
  if (!reader) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
  auto engine = Engine::Create(reader, ctx, &err);
  if (!engine) { std::fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
  const std::size_t V = engine->config().n_vocab_text;
  auto s = engine->NewSession();
  Snapshot base, cyc;
  if (hipMalloc(&base.conv, gufo::models::inkling2::kConvBytes) != hipSuccess ||
      hipMalloc(&cyc.conv, gufo::models::inkling2::kConvBytes) != hipSuccess) {
    std::fprintf(stderr, "hipMalloc failed\n");
    return 1;
  }

  // ---- 1 verify cost
  engine->Forward(*s, prompt);
  Save(*s, &base);
  {
    // warm-up (graph capture of the decode path, prefill buffers)
    for (int w = 0; w < 3; ++w) {
      const std::int32_t t = Argmax(s->logits().data(), V);
      engine->Forward(*s, std::span<const std::int32_t>(&t, 1));
    }
    Load(*s, base);
    std::vector<std::int32_t> toks(9, Argmax(base.logits.data(), V));
    for (int T = 1; T <= 9; ++T) {
      double total = 0;
      const int reps = 5;
      for (int r = 0; r < reps; ++r) {
        Load(*s, base);
        engine->Synchronize();
        const auto t0 = Clock::now();
        if (T == 1) {
          engine->Forward(*s, std::span<const std::int32_t>(toks.data(), 1));
        } else {
          engine->Forward(*s, std::span<const std::int32_t>(toks.data(), static_cast<std::size_t>(T)),
                          [](std::size_t, std::size_t, const float*) {});
        }
        engine->Synchronize();
        total += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      }
      std::printf("verify T=%d: %.2f ms per step (%s)\n", T, total / reps, T == 1 ? "decode path" : "multi-token path, all logits");
    }
    for (int T = 2; T <= 8 && engine->batch_capable(); ++T) {
      std::vector<std::vector<float>> rows;
      double total = 0;
      for (int r = 0; r < 6; ++r) {
        Load(*s, base);
        engine->Synchronize();
        const auto t0 = Clock::now();
        engine->VerifyChain(*s, std::span<const std::int32_t>(toks.data(), static_cast<std::size_t>(T)), &rows);
        if (r > 0) total += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      }
      std::printf("verify T=%d: %.2f ms per step (chain path, weights read once)\n", T, total / 5);
    }
    // chain correctness: 8 sequential decode steps vs one chain of the same 8 tokens
    if (engine->batch_capable()) {
      Load(*s, base);
      std::vector<std::int32_t> seq;
      std::vector<std::vector<float>> ref;
      std::int32_t t = Argmax(base.logits.data(), V);
      for (int i = 0; i < 8; ++i) {
        seq.push_back(t);
        engine->Forward(*s, std::span<const std::int32_t>(&t, 1));
        ref.push_back(s->logits());
        t = Argmax(s->logits().data(), V);
      }
      Load(*s, base);
      std::vector<std::vector<float>> rows;
      engine->VerifyChain(*s, seq, &rows);
      for (int i = 0; i < 8; ++i) {
        double ab = 0, aa = 0, bb = 0, md = 0;
        for (std::size_t j = 0; j < V; ++j) {
          md = std::max(md, std::abs(double(rows[i][j]) - ref[i][j]));
          ab += double(rows[i][j]) * ref[i][j];
          aa += double(rows[i][j]) * rows[i][j];
          bb += double(ref[i][j]) * ref[i][j];
        }
        std::printf("chain check row %d: argmax chain %d decode %d, cos %.6f, max|diff| %.3g%s\n", i,
                    Argmax(rows[i].data(), V), Argmax(ref[i].data(), V), ab / std::sqrt(aa * bb + 1e-30), md,
                    md == 0.0 ? " (bit-identical)" : "");
      }
      // the state after the chain must equal the state after the 4 decode steps: feed the same 5th token and
      // compare the logits of the step after it (Pause 7 check compared different steps)
      std::vector<float> after_seq;
      {
        Load(*s, base);
        for (int i = 0; i < 8; ++i) engine->Forward(*s, std::span<const std::int32_t>(&seq[i], 1));
        engine->Forward(*s, std::span<const std::int32_t>(&t, 1));
        after_seq = s->logits();
        Load(*s, base);
        std::vector<std::vector<float>> again;
        engine->VerifyChain(*s, seq, &again);
        engine->Forward(*s, std::span<const std::int32_t>(&t, 1));
      }
      double md = 0;
      for (std::size_t j = 0; j < V; ++j) md = std::max(md, std::abs(double(s->logits()[j]) - after_seq[j]));
      std::printf("chain check state: step after chain argmax %d, after 4 decodes %d, max|diff| %.3g%s\n",
                  Argmax(s->logits().data(), V), Argmax(after_seq.data(), V), md, md == 0.0 ? " (bit-identical)" : "");
    }
    const auto t0 = Clock::now();
    for (int r = 0; r < 20; ++r) { Save(*s, &cyc); Load(*s, cyc); }
    std::printf("conv save+restore: %.3f ms\n", std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / 20);
  }

  // ---- 2 greedy A/B
  std::vector<std::int32_t> reference;
  for (int k : ParseList(klist)) {
    Load(*s, base);
    std::vector<std::int32_t> hist = prompt, out;
    std::int32_t next = Argmax(s->logits().data(), V);
    int cycles = 0, drafted = 0, accepted = 0;
    engine->Synchronize();
    const auto t0 = Clock::now();
    while (static_cast<int>(out.size()) < gen) {
      // the draft continues AFTER `next` (Pause 6 bug: the lookup ran on the history without `next`, so every
      // draft guessed `next` itself and nothing was ever accepted)
      hist.push_back(next);
      const int kmax = verify == "chain" ? std::min(k, static_cast<int>(Engine::kMaxDecodeBatch) - 1) : k;
      std::vector<std::int32_t> d = Lookup(hist, ngram, std::min(kmax, gen - static_cast<int>(out.size()) - 1));
      hist.pop_back();
      ++cycles;
      if (d.empty()) {
        out.push_back(next);
        hist.push_back(next);
        engine->Forward(*s, std::span<const std::int32_t>(&next, 1));
        next = Argmax(s->logits().data(), V);
        continue;
      }
      Save(*s, &cyc);
      std::vector<std::int32_t> step{next};
      step.insert(step.end(), d.begin(), d.end());
      std::vector<std::int32_t> arg(step.size());
      if (verify == "chain") {
        std::vector<std::vector<float>> rows;
        engine->VerifyChain(*s, step, &rows);
        for (std::size_t r = 0; r < rows.size(); ++r) arg[r] = Argmax(rows[r].data(), V);
      } else {
        engine->Forward(*s, step, [&](std::size_t first, std::size_t count, const float* data) {
          for (std::size_t r = 0; r < count; ++r) arg[first + r] = Argmax(data + r * V, V);
        });
      }
      std::size_t a = 0;
      while (a < d.size() && arg[a] == d[a]) ++a;
      drafted += static_cast<int>(d.size());
      accepted += static_cast<int>(a);
      for (std::size_t i = 0; i <= a; ++i) { out.push_back(step[i]); hist.push_back(step[i]); }
      if (a < d.size()) {
        Load(*s, cyc);  // back to the frontier before `next`, then evaluate only the accepted part
        if (a == 0) {
          engine->Forward(*s, std::span<const std::int32_t>(step.data(), 1));
        } else if (verify == "chain") {
          std::vector<std::vector<float>> rows;
          engine->VerifyChain(*s, std::span<const std::int32_t>(step.data(), a + 1), &rows);
        } else {
          engine->Forward(*s, std::span<const std::int32_t>(step.data(), a + 1));
        }
      }
      next = arg[a];
    }
    engine->Synchronize();
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    out.resize(static_cast<std::size_t>(gen));
    if (k == 0) reference = out;
    int same = 0;
    for (std::size_t i = 0; i < out.size() && i < reference.size() && out[i] == reference[i]; ++i) ++same;
    std::printf("k=%d: %d tokens in %.2f s = %.2f tok/s, %d cycles, drafted %d accepted %d (%.0f%%), identical to k=0 for the first %d tokens\n",
                k, gen, secs, gen / secs, cycles, drafted, accepted, drafted ? 100.0 * accepted / drafted : 0.0, same);
  }
  (void)hipFree(base.conv);
  (void)hipFree(cyc.conv);
  return 0;
}
