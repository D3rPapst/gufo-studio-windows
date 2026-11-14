// gufo-yue2: one song with YuE2-3B.
//
//   gufo-yue2 --model YuE2-3B-dir --vae YuE2-Vae-dir --style "..."
//       [--lyrics "..." | --lyrics-file F] [--cot full|melody|off]
//       [--abc-file F] [--seed N] [--max-seconds S] [--guidance G]
//       [--steps 32] --out song.wav [--abc-out score.abc]
// Written with AI assistance (Claude Code).
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <string>

#include "src/models/ace_step/audio_io.hpp"
#include "src/models/yue2/pipeline.hpp"

namespace y = gufo::models::yue2;

namespace {

bool ReadFile(const std::string& path, std::string* out) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;
  out->assign(std::istreambuf_iterator<char>(in), {});
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i + 1 < argc; i += 2)
    args[argv[i]] = argv[i + 1];
  if (!args.count("--model") || !args.count("--vae") || !args.count("--out") ||
      !args.count("--style")) {
    std::fprintf(stderr,
                 "usage: gufo-yue2 --model DIR --vae DIR --style TEXT "
                 "[--lyrics TEXT | --lyrics-file F] [--cot full|melody|off] "
                 "[--abc-file F] [--seed N] [--max-seconds S] [--guidance G] "
                 "[--steps N] --out song.wav [--abc-out F]\n");
    return 2;
  }
  y::SongRequest request;
  request.style = args["--style"];
  request.lyrics = args["--lyrics"];
  if (args.count("--lyrics-file") &&
      !ReadFile(args["--lyrics-file"], &request.lyrics)) {
    std::fprintf(stderr, "cannot read %s\n", args["--lyrics-file"].c_str());
    return 2;
  }
  if (args.count("--abc-file") && !ReadFile(args["--abc-file"], &request.abc)) {
    std::fprintf(stderr, "cannot read %s\n", args["--abc-file"].c_str());
    return 2;
  }
  if (args.count("--cot") && !y::ParseCot(args["--cot"], &request.cot)) {
    std::fprintf(stderr, "--cot must be full, melody or off\n");
    return 2;
  }
  if (args.count("--seed"))
    request.seed = std::stoull(args["--seed"]);
  if (args.count("--max-seconds"))
    request.max_seconds = std::stod(args["--max-seconds"]);
  if (args.count("--guidance"))
    request.guidance = std::stof(args["--guidance"]);
  if (args.count("--steps"))
    request.ode_steps = std::stoi(args["--steps"]);
  std::string error;
  const auto load_start = std::chrono::steady_clock::now();
  auto pipeline = y::SongPipeline::Load(args["--model"], args["--vae"], &error);
  if (!pipeline) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::printf("loaded in %.1f s\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            load_start)
                  .count());
  int last = -1;
  y::SongResult result;
  if (!pipeline->Generate(
          request, &result,
          [&](double fraction, std::string_view phase) {
            const int percent = static_cast<int>(fraction * 100);
            if (percent / 5 != last / 5) {
              std::printf("  %3d %% %.*s\n", percent,
                          static_cast<int>(phase.size()), phase.data());
              std::fflush(stdout);
              last = percent;
            }
            return true;
          },
          &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  const double audio_seconds = result.audio.size() / 2 / 48000.0;
  std::printf(
      "song %.1f s: abc %d tokens %.1f s%s, semantic %d tokens %.1f s%s, "
      "acoustic %d chunk(s) %.1f s, vae %.1f s, total %.1f s (%.2f s audio "
      "per s), device %.2f GB\n",
      audio_seconds, result.abc_tokens, result.abc_seconds,
      result.abc_truncated ? " (cut)" : "", result.semantic_tokens,
      result.semantic_seconds, result.semantic_truncated ? " (cut)" : "",
      result.chunks, result.nar_seconds, result.vae_seconds,
      result.total_seconds, audio_seconds / result.total_seconds,
      pipeline->device_bytes() / 1e9);
  std::ofstream(args["--out"], std::ios::binary)
      << gufo::models::ace_step::EncodeStereoWav(result.audio, 48000);
  if (args.count("--abc-out"))
    std::ofstream(args["--abc-out"], std::ios::binary) << result.abc;
  return 0;
}
