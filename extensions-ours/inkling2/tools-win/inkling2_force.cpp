// gufo_windows addition (not part of headbouyJB/inkling2-gufo): teacher-forced check of the engine
// against llama.cpp logits dumped by D:\claude\gufo-arbeit\inkling2\dumper (logits_<k>.f32 = the
// logits after the prompt and k forced tokens, n_vocab floats).
//
//   inkling2_force --model GGUF --ids 1,2,3 --forced FILE --llama DIR [--ctx 8192] [--mode decode|prefill]
//
// decode : prompt in one Forward call, then one Forward call per forced token (the decode path).
// prefill: prompt + all forced tokens in one Forward call with a logits sink (the prefill path).
// Prints per step gufo argmax, forced token, llama argmax and the cosine of the logits, then a summary
// "teacher forced: argmax agreed A of N; first step with cos < 0.99: S; min cos C".
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>

#include "src/engine.hpp"

namespace {

std::vector<std::int32_t> ParseIds(const std::string& s) {
  std::vector<std::int32_t> out;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) {
    if (!t.empty()) out.push_back(static_cast<std::int32_t>(std::strtol(t.c_str(), nullptr, 10)));
  }
  return out;
}

std::vector<std::int32_t> ReadForced(const std::string& path) {
  std::vector<std::int32_t> out;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.find_first_not_of("0123456789\r") == std::string::npos)
      out.push_back(static_cast<std::int32_t>(std::strtol(line.c_str(), nullptr, 10)));
  }
  return out;
}

bool ReadLlama(const std::string& dir, std::size_t k, std::vector<float>* out) {
  std::ifstream f(dir + "/logits_" + std::to_string(k) + ".f32", std::ios::binary);
  if (!f) return false;
  f.seekg(0, std::ios::end);
  const auto bytes = static_cast<std::size_t>(f.tellg());
  f.seekg(0);
  out->resize(bytes / 4);
  f.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(bytes));
  return static_cast<bool>(f);
}

std::size_t Argmax(const float* v, std::size_t n) {
  return static_cast<std::size_t>(std::max_element(v, v + n) - v);
}

double Cos(const float* a, const float* b, std::size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (std::size_t i = 0; i < n; ++i) {
    ab += static_cast<double>(a[i]) * b[i];
    aa += static_cast<double>(a[i]) * a[i];
    bb += static_cast<double>(b[i]) * b[i];
  }
  return ab / std::sqrt(aa * bb + 1e-30);
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, ids_s, forced_path, llama, mode = "decode";
  unsigned ctx = 8192;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (k == "--model") model = next();
    else if (k == "--ids") ids_s = next();
    else if (k == "--forced") forced_path = next();
    else if (k == "--llama") llama = next();
    else if (k == "--ctx") ctx = static_cast<unsigned>(std::strtoul(next().c_str(), nullptr, 10));
    else if (k == "--mode") mode = next();
    else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
  }
  const auto prompt = ParseIds(ids_s);
  const auto forced = ReadForced(forced_path);
  if (model.empty() || prompt.empty() || forced.empty() || llama.empty()) {
    std::fprintf(stderr, "usage: inkling2_force --model GGUF --ids a,b,c --forced FILE --llama DIR [--ctx N] [--mode decode|prefill]\n");
    return 2;
  }
  std::string err;
  gufo::core::GgufOpenOptions open;
  open.allow_unsized_tensors = true;
  std::shared_ptr<const gufo::core::GgufReader> reader = gufo::core::GgufReader::OpenFile(model, &err, open);
  if (!reader) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
  const auto t0 = std::chrono::steady_clock::now();
  auto engine = gufo::models::inkling2::Engine::Create(reader, ctx, &err);
  if (!engine) { std::fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
  std::printf("load %.1f s, weights %.2f GB, mode %s\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
              engine->weight_bytes() / 1e9, mode.c_str());
  const std::size_t V = engine->config().n_vocab_text;
  auto session = engine->NewSession();

  // logits of step k (after prompt + k forced tokens), from the chosen path
  std::vector<std::vector<float>> mine(forced.size());
  if (mode == "prefill") {
    std::vector<std::int32_t> all = prompt;
    all.insert(all.end(), forced.begin(), forced.end() - 1);
    const std::size_t base = prompt.size() - 1;
    engine->Forward(*session, all, [&](std::size_t first, std::size_t count, const float* data) {
      for (std::size_t r = 0; r < count; ++r) {
        const std::size_t idx = first + r;
        if (idx >= base && idx - base < mine.size()) mine[idx - base].assign(data + r * V, data + (r + 1) * V);
      }
    });
  } else {
    engine->Forward(*session, prompt);
    for (std::size_t k = 0; k < forced.size(); ++k) {
      mine[k] = session->logits();
      if (k + 1 < forced.size()) engine->Forward(*session, std::span<const std::int32_t>(&forced[k], 1));
    }
  }

  int agree = 0, first_low = -1, steps = 0;
  double min_cos = 1.0;
  std::vector<float> ref;
  for (std::size_t k = 0; k < forced.size(); ++k) {
    if (!ReadLlama(llama, k, &ref) || mine[k].empty()) break;
    const std::size_t n = std::min(V, ref.size());
    const std::size_t g = Argmax(mine[k].data(), n), l = Argmax(ref.data(), n);
    const double c = Cos(mine[k].data(), ref.data(), n);
    ++steps;
    if (g == static_cast<std::size_t>(forced[k])) ++agree;
    if (c < 0.99 && first_low < 0) first_low = static_cast<int>(k);
    min_cos = std::min(min_cos, c);
    std::printf("k %3zu pos %5zu gufo %6zu forced %6d llama %6zu cos %.5f%s\n", k, prompt.size() + k, g, forced[k], l, c,
                g != static_cast<std::size_t>(forced[k]) ? "  DIFF" : "");
  }
  std::printf("teacher forced: argmax agreed %d of %d; first step with cos < 0.99: %d; min cos %.5f\n", agree, steps,
              first_low, min_cos);
  return 0;
}
