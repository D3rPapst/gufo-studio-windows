// gufo-yue2-check: compares the YuE2 port stage by stage with the FP32 CPU
// reference dumps of the official yue2_infer 0.1.6 code (.npy files and
// JSON written by the oracle script, see docs/internal/yue2.md).
//
//   gufo-yue2-check --oracle DIR --model YuE2-3B --vae YuE2-Vae
//       [--stages tok,vae,ar,nar]
// Written with AI assistance (Claude Code).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include <gufo/ext/json.hpp>
#include "src/models/ace_step/vae.hpp"
#include "src/models/yue2/model.hpp"
#include "src/models/yue2/protocol.hpp"
#include "src/models/yue2/tokenizer.hpp"

namespace y = gufo::models::yue2;
namespace ace = gufo::models::ace_step;
namespace json = gufo::json;

namespace {

int failures = 0;

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

// Minimal .npy reader for little-endian <f4 / <i4 C-order arrays.
template<typename T>
std::vector<T> ReadNpy(const std::filesystem::path& path,
                       std::vector<std::size_t>* shape = nullptr) {
  const std::string bytes = ReadText(path);
  if (bytes.size() < 10 || bytes.compare(1, 5, "NUMPY") != 0) {
    std::fprintf(stderr, "cannot read %s\n", path.string().c_str());
    ++failures;
    return {};
  }
  const int major = static_cast<unsigned char>(bytes[6]);
  std::size_t header_len = 0, offset = 0;
  if (major == 1) {
    header_len = static_cast<unsigned char>(bytes[8]) |
                 (static_cast<unsigned char>(bytes[9]) << 8U);
    offset = 10;
  } else {
    header_len = 0;
    for (int i = 0; i < 4; ++i)
      header_len |= std::size_t(static_cast<unsigned char>(bytes[8 + i]))
                    << (8U * i);
    offset = 12;
  }
  const std::string header = bytes.substr(offset, header_len);
  if (shape != nullptr) {
    shape->clear();
    const auto open = header.find("'shape': (");
    std::size_t i = open + 10;
    while (open != std::string::npos && i < header.size() &&
           header[i] != ')') {
      if (header[i] >= '0' && header[i] <= '9') {
        std::size_t value = 0;
        while (header[i] >= '0' && header[i] <= '9')
          value = value * 10 + (header[i++] - '0');
        shape->push_back(value);
      } else {
        ++i;
      }
    }
  }
  const std::size_t data = offset + header_len;
  std::vector<T> out((bytes.size() - data) / sizeof(T));
  std::memcpy(out.data(), bytes.data() + data, out.size() * sizeof(T));
  return out;
}

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty())
    return -2;
  double ab = 0, aa = 0, bb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  return ab / (std::sqrt(aa * bb) + 1e-30);
}

void Report(const char* name, const std::vector<float>& got,
            const std::vector<float>& want, double threshold,
            double max_rel = -1) {
  const double c = Cosine(got, want);
  double diff = 0, ref = 0, worst = 0;
  for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
    diff += (double(got[i]) - want[i]) * (double(got[i]) - want[i]);
    ref += double(want[i]) * want[i];
    worst = std::max(worst, std::fabs(double(got[i]) - want[i]));
  }
  const double rel = std::sqrt(diff / (ref + 1e-30));
  const double snr = 10 * std::log10((ref + 1e-30) / (diff + 1e-30));
  const bool ok = c >= threshold && (max_rel < 0 || rel <= max_rel) &&
                  got.size() == want.size();
  if (!ok)
    ++failures;
  std::printf("%-26s cos %.6f  rel %.3e  snr %.1f dB  max|d| %.4g  n %zu/%zu"
              "  %s\n",
              name, c, rel, snr, worst, got.size(), want.size(),
              ok ? "ok" : "FAIL");
  std::fflush(stdout);
}

std::vector<std::uint32_t> Ids(const json::Value& array) {
  std::vector<std::uint32_t> out;
  for (const auto& v : array.items())
    out.push_back(static_cast<std::uint32_t>(v.as_double()));
  return out;
}

double Seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since)
      .count();
}

// The oracle's slice of full-vocabulary logits.
std::vector<float> Slice(const std::vector<float>& full, y::Phase phase) {
  const y::VocabularySlice s = y::PhaseSlice(phase);
  if (full.size() < s.first + s.count)
    return {};
  std::vector<float> out(full.begin() + s.first,
                         full.begin() + s.first + s.count);
  if (phase == y::Phase::kAbc)  // the masked special ids are not sampled
    for (std::uint32_t id = y::kEod; id < y::kAbcEnd; ++id)
      out[id] = 0;
  return out;
}

void CompareTokens(const char* name, const std::vector<std::uint32_t>& got,
                   const std::vector<std::uint32_t>& want) {
  std::size_t same = 0;
  while (same < got.size() && same < want.size() && got[same] == want[same])
    ++same;
  const bool ok = same == want.size() && got.size() == want.size();
  if (!ok)
    ++failures;
  std::printf("%-26s identical %zu / %zu (got %zu)  %s\n", name, same,
              want.size(), got.size(), ok ? "ok" : "FAIL");
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i + 1 < argc; i += 2)
    args[argv[i]] = argv[i + 1];
  if (!args.count("--oracle") || !args.count("--model")) {
    std::fprintf(stderr,
                 "usage: gufo-yue2-check --oracle DIR --model YuE2-3B-dir "
                 "--vae YuE2-Vae-dir [--stages tok,vae,ar,nar]\n");
    return 2;
  }
  const std::filesystem::path oracle = args["--oracle"];
  const std::filesystem::path model_dir = args["--model"];
  const std::string stages =
      args.count("--stages") ? args["--stages"] : "tok,vae,ar,nar";
  const auto want = [&](const char* s) {
    return stages.find(s) != std::string::npos;
  };
  std::string error;
  auto tokenizer =
      y::TextTokenizer::Load(model_dir / "qwen.tiktoken", &error);
  if (!tokenizer) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }

  if (want("tok")) {
    const auto tokens = json::parse(ReadText(oracle / "tokens.json"));
    const auto& texts = tokens.find("texts")->items();
    const auto& ids = tokens.find("ids")->items();
    for (std::size_t i = 0; i < texts.size(); ++i)
      CompareTokens(("tokens text " + std::to_string(i)).c_str(),
                    tokenizer->Encode(texts[i].str()), Ids(ids[i]));
    const std::string style = texts[0].str(), lyrics = texts[1].str();
    CompareTokens("prefix full",
                  y::TokenPrefix(*tokenizer, y::Cot::kFull, style, lyrics,
                                 nullptr),
                  Ids(*tokens.find("prefix_full")));
    CompareTokens("prefix off",
                  y::TokenPrefix(*tokenizer, y::Cot::kOff, style, lyrics,
                                 nullptr),
                  Ids(*tokens.find("prefix_off")));
    CompareTokens("prefix melody",
                  y::TokenPrefix(*tokenizer, y::Cot::kMelody, style, lyrics,
                                 nullptr),
                  Ids(*tokens.find("prefix_melody")));
    const auto ids0 = Ids(ids[0]);
    const std::string round = tokenizer->Decode(ids0);
    if (round != texts[0].str()) {
      ++failures;
      std::printf("decode round trip FAIL\n");
    }
  }

  if (want("vae") && args.count("--vae")) {
    const auto start = std::chrono::steady_clock::now();
    auto vae = ace::VaeDecoder::Load(
        std::filesystem::path(args["--vae"]) / "model.safetensors",
        ace::OobleckLayout::YuE2(), &error);
    if (!vae) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    std::printf("vae loaded %.1f s\n", Seconds(start));
    const auto latents = ReadNpy<float>(oracle / "vae_latents_rand.npy");
    const int frames = static_cast<int>(latents.size() / 64);
    std::vector<float> audio;
    const auto t0 = std::chrono::steady_clock::now();
    if (!vae->Decode(latents, frames, &audio, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    const double seconds = Seconds(t0);
    std::printf("vae decode %d frames (%.1f s audio) in %.2f s\n", frames,
                frames / 25.0, seconds);
    Report("vae random latents", audio,
           ReadNpy<float>(oracle / "vae_audio_rand_full.npy"), 0.999);
    if (std::filesystem::exists(oracle / "vae_audio_nar_32.npy")) {
      const auto nar = ReadNpy<float>(oracle / "nar_latents_32.npy");
      if (vae->Decode(nar, static_cast<int>(nar.size() / 64), &audio, &error))
        Report("vae nar latents", audio,
               ReadNpy<float>(oracle / "vae_audio_nar_32.npy"), 0.999);
    }
  }

  std::unique_ptr<y::Model> model;
  if (want("ar") || want("nar")) {
    const auto start = std::chrono::steady_clock::now();
    model = y::Model::Load(model_dir / "model.safetensors", &error);
    if (!model) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    std::printf("model loaded %.1f s (%.2f GB)\n", Seconds(start),
                model->weight_bytes() / 1e9);
  }

  if (want("ar")) {
    const auto ar = json::parse(ReadText(oracle / "ar.json"));
    const auto abc_prefix = Ids(*ar.find("abc_prefix"));
    const auto sem_prefix = Ids(*ar.find("sem_prefix"));
    const auto abc_want = Ids(*ar.find("abc_greedy"));
    const auto sem_want = Ids(*ar.find("sem_greedy"));
    std::vector<float> logits;
    if (!model->PrefixLogits(abc_prefix, y::Phase::kAbc, &logits, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    for (std::uint32_t id = y::kEod; id < y::kAbcEnd; ++id)
      logits[id] = 0;
    Report("abc prefix logits", logits,
           Slice(ReadNpy<float>(oracle / "abc_prefix_logits.npy"),
                 y::Phase::kAbc),
           0.999);
    if (!model->PrefixLogits(sem_prefix, y::Phase::kSemantic, &logits,
                             &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    Report("semantic prefix logits", logits,
           Slice(ReadNpy<float>(oracle / "sem_prefix_logits.npy"),
                 y::Phase::kSemantic),
           0.999);
    // Greedy: temperature 0 with the phase's penalty and min_tokens.
    y::PhaseSampling abc = y::DefaultAbcSampling();
    abc.temperature = 0;
    abc.max_tokens = static_cast<int>(abc_want.size());
    y::PhaseResult result;
    auto t0 = std::chrono::steady_clock::now();
    if (!model->Generate(abc_prefix, {}, 1.0F, y::Phase::kAbc, abc, 0, false,
                         &result, {}, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    const double abc_seconds = Seconds(t0);
    auto got = result.tokens;
    if (!result.truncated)
      got.push_back(y::kAbcEnd);
    CompareTokens("abc greedy", got, abc_want);
    std::printf("abc %zu tokens in %.2f s (prefill %.2f s): %.1f tok/s\n",
                result.tokens.size(), abc_seconds, result.prefill_seconds,
                result.tokens.size() / (abc_seconds - result.prefill_seconds));
    y::PhaseSampling sem = y::DefaultSemanticSampling();
    sem.temperature = 0;
    sem.max_tokens = static_cast<int>(sem_want.size());
    sem.min_tokens = sem.max_tokens;
    t0 = std::chrono::steady_clock::now();
    if (!model->Generate(sem_prefix, {}, 1.0F, y::Phase::kSemantic, sem, 0,
                         false, &result, {}, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    const double sem_seconds = Seconds(t0);
    CompareTokens("semantic greedy", result.tokens, sem_want);
    std::printf("semantic %zu tokens in %.2f s (prefill %.2f s): %.1f tok/s\n",
                result.tokens.size(), sem_seconds, result.prefill_seconds,
                result.tokens.size() / (sem_seconds - result.prefill_seconds));
  }

  if (want("nar")) {
    const auto nar = json::parse(ReadText(oracle / "nar.json"));
    const auto ids = ReadNpy<std::int32_t>(oracle / "nar_ar_tokens.npy");
    std::vector<std::uint32_t> ar_tokens(ids.begin(), ids.end());
    const auto noise = ReadNpy<float>(oracle / "nar_noise.npy");
    const int frames = static_cast<int>(noise.size() / 64);
    auto t0 = std::chrono::steady_clock::now();
    if (!model->BeginAcoustic(ar_tokens, frames, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    std::printf("acoustic prefill %zu tokens, %d frames: %.2f s\n",
                ar_tokens.size(), frames, Seconds(t0));
    std::vector<float> v;
    t0 = std::chrono::steady_clock::now();
    if (!model->Velocity(noise, nar.find("raw_t1")->as_double(), &v, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    std::printf("velocity %.3f s\n", Seconds(t0));
    Report("nar velocity t=1", v,
           ReadNpy<float>(oracle / "nar_velocity_t1.npy"), 0.9999, 1e-2);
    if (!model->Velocity(ReadNpy<float>(oracle / "nar_state_t04.npy"),
                         nar.find("raw_t04")->as_double(), &v, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    Report("nar velocity t=0.4", v,
           ReadNpy<float>(oracle / "nar_velocity_t04.npy"), 0.9999, 1e-2);
    std::vector<float> latents;
    t0 = std::chrono::steady_clock::now();
    if (!model->Solve(noise, y::kOdeSteps, &latents, {}, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    std::printf("ode %d steps in %.2f s\n", y::kOdeSteps, Seconds(t0));
    Report("nar latents (32 steps)", latents,
           ReadNpy<float>(oracle / "nar_latents_32.npy"), 0.999, 5e-2);
  }
  std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
