// YuE2 prompting protocol ("yue2-native-v1", yue2/protocol.py of
// yue2_infer 0.1.6) and the phase sampler (yue2/sampling.py). CPU only.
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_YUE2_PROTOCOL_HPP_
#define GUFO_MODELS_YUE2_PROTOCOL_HPP_

#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gufo::models::yue2 {

class TextTokenizer;

inline constexpr std::uint32_t kEod = 151643;
inline constexpr std::uint32_t kAbcStart = 151847;
inline constexpr std::uint32_t kAbcEnd = 151848;
inline constexpr std::uint32_t kMusicStart = 151851;
inline constexpr std::uint32_t kMusicEnd = 151852;
inline constexpr std::uint32_t kCodecOffset = 151853;
inline constexpr std::uint32_t kCodecSize = 32768;
inline constexpr std::uint32_t kVocabulary = 184704;
inline constexpr int kContext = 24576;
inline constexpr int kFramesPerSecond = 25;
inline constexpr int kLatentChannels = 64;

/// Planning mode: "full" (chord-annotated ABC), "melody" (melody-only
/// ABC) or "off" (no score).
enum class Cot { kFull, kMelody, kOff };
[[nodiscard]] bool ParseCot(std::string_view text, Cot* cot);
[[nodiscard]] std::string_view CotName(Cot cot);

enum class Phase { kAbc, kSemantic };

/// Vocabulary slice a phase samples from: ABC = [0, ABC_END] (ordinary
/// text ids and ABC_END), semantic = [MUSIC_END, MUSIC_END + 32769) (the end
/// token directly precedes the 32768 codes).
struct VocabularySlice {
  std::uint32_t first{0};
  std::uint32_t count{0};
};
[[nodiscard]] VocabularySlice PhaseSlice(Phase phase);

/// Sampling of one phase (protocol.Sampling).
struct PhaseSampling {
  float temperature{1.0F};
  float top_p{0.95F};
  int top_k{100};
  float repetition_penalty{1.2F};
  int penalty_window{50};
  int min_tokens{200};
  int max_tokens{9000};
};
/// yue2_generation_config.json: abc 0.7 / 0.9 / 30 / 1.005 / 100 / 32 /
/// 4096, semantic 1.0 / 0.95 / 100 / 1.2 / 50 / 200 / 9000.
[[nodiscard]] PhaseSampling DefaultAbcSampling();
[[nodiscard]] PhaseSampling DefaultSemanticSampling();
inline constexpr int kOdeSteps = 32;

/// The request text: instruction, "[Tags]" style, "[Lyrics]" lyrics.
[[nodiscard]] std::string RequestText(Cot cot, std::string_view style,
                                      std::string_view lyrics);
/// token_prefixes: [EOD] + text, then ABC_START (cot full / melody without
/// `abc_ids`), ABC_START abc ABC_END MUSIC_START (with them) or ABC_START
/// ABC_END MUSIC_START (off).
[[nodiscard]] std::vector<std::uint32_t> TokenPrefix(
    const TextTokenizer& tokenizer, Cot cot, std::string_view style,
    std::string_view lyrics, const std::vector<std::uint32_t>* abc_ids);
/// negative_prefix of classifier-free guidance (instruction only).
[[nodiscard]] std::vector<std::uint32_t> NegativePrefix(
    const TextTokenizer& tokenizer, Cot cot,
    const std::vector<std::uint32_t>& abc_ids);
/// Default guidance: 1.01 for off, else 1 (no CFG).
[[nodiscard]] float DefaultGuidance(Cot cot);

/// chunk_ranges: acoustic chunks [a, b) of `frames` for a prefix length.
[[nodiscard]] std::vector<std::pair<int, int>> ChunkRanges(int frames,
                                                           int prefix_tokens);

/// One sampling step of `distribution` + multinomial over the phase slice:
/// `logits` are the slice's FP32 logits (count values, guidance already
/// applied), `history` the absolute ids sampled so far in this phase.
/// Masks (only the slice's allowed ids), suppresses the end token before
/// `min_tokens`, applies the windowed repetition penalty, temperature,
/// top_k and top_p (keeping at least one, or three for the legacy `off`
/// arithmetic), then draws with `rng` (argmax at temperature 0). Returns
/// the absolute token id.
[[nodiscard]] std::uint32_t SampleToken(std::span<const float> logits,
                                        Phase phase,
                                        const PhaseSampling& sampling,
                                        std::span<const std::uint32_t> history,
                                        int step, bool legacy_off,
                                        std::mt19937_64* rng);

}  // namespace gufo::models::yue2

#endif  // GUFO_MODELS_YUE2_PROTOCOL_HPP_
