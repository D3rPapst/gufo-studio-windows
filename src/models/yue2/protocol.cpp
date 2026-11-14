// Written with AI assistance (Claude Code).
#include "src/models/yue2/protocol.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>

#include "src/models/yue2/tokenizer.hpp"

namespace gufo::models::yue2 {
namespace {

constexpr float kMinusInfinity = -std::numeric_limits<float>::infinity();

std::string_view Instruction(Cot cot) {
  switch (cot) {
  case Cot::kOff:
    return "Generate music with codec tokens from the given conditions.";
  case Cot::kMelody:
    return "Generate a melody-only ABC transcription without chord symbols, "
           "then generate music with codec tokens from the given "
           "conditions.";
  case Cot::kFull:
    break;
  }
  return "Generate a chord-annotated ABC transcription, then generate music "
         "with codec tokens from the given conditions.";
}

float RoundBf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, 4);
  if ((bits & 0x7FFFFFFFU) > 0x7F800000U)
    return value;
  bits = (bits + 0x7FFFU + ((bits >> 16U) & 1U)) & 0xFFFF0000U;
  std::memcpy(&value, &bits, 4);
  return value;
}

}  // namespace

bool ParseCot(std::string_view text, Cot* cot) {
  if (text == "full")
    *cot = Cot::kFull;
  else if (text == "melody")
    *cot = Cot::kMelody;
  else if (text == "off")
    *cot = Cot::kOff;
  else
    return false;
  return true;
}

std::string_view CotName(Cot cot) {
  return cot == Cot::kFull ? "full" : cot == Cot::kMelody ? "melody" : "off";
}

VocabularySlice PhaseSlice(Phase phase) {
  if (phase == Phase::kAbc)
    return {0, kAbcEnd + 1};
  return {kMusicEnd, kCodecOffset + kCodecSize - kMusicEnd};
}

PhaseSampling DefaultAbcSampling() {
  return {0.7F, 0.9F, 30, 1.005F, 100, 32, 4096};
}

PhaseSampling DefaultSemanticSampling() {
  return {1.0F, 0.95F, 100, 1.2F, 50, 200, 9000};
}

std::string RequestText(Cot cot, std::string_view style,
                        std::string_view lyrics) {
  std::string text(Instruction(cot));
  text += "\n[Tags]\n";
  text += style;
  text += "\n[Lyrics]\n";
  text += lyrics;
  text += "\n";
  return text;
}

std::vector<std::uint32_t> TokenPrefix(
    const TextTokenizer& tokenizer, Cot cot, std::string_view style,
    std::string_view lyrics, const std::vector<std::uint32_t>* abc_ids) {
  std::vector<std::uint32_t> ids{kEod};
  const auto text = tokenizer.Encode(RequestText(cot, style, lyrics));
  ids.insert(ids.end(), text.begin(), text.end());
  ids.push_back(kAbcStart);
  if (cot == Cot::kOff) {
    ids.push_back(kAbcEnd);
    ids.push_back(kMusicStart);
  } else if (abc_ids != nullptr) {
    ids.insert(ids.end(), abc_ids->begin(), abc_ids->end());
    ids.push_back(kAbcEnd);
    ids.push_back(kMusicStart);
  }
  return ids;
}

std::vector<std::uint32_t> NegativePrefix(
    const TextTokenizer& tokenizer, Cot cot,
    const std::vector<std::uint32_t>& abc_ids) {
  std::vector<std::uint32_t> ids{kEod};
  const auto text = tokenizer.Encode(Instruction(cot));
  ids.insert(ids.end(), text.begin(), text.end());
  if (cot != Cot::kOff) {
    ids.push_back(kAbcStart);
    ids.insert(ids.end(), abc_ids.begin(), abc_ids.end());
    ids.push_back(kAbcEnd);
  }
  ids.push_back(kMusicStart);
  return ids;
}

float DefaultGuidance(Cot cot) {
  return cot == Cot::kOff ? 1.01F : 1.0F;
}

std::vector<std::pair<int, int>> ChunkRanges(int frames, int prefix_tokens) {
  const int size = std::min((kContext - prefix_tokens - 3) / 2, kContext);
  std::vector<std::pair<int, int>> ranges;
  if (frames < 1 || size < 1)
    return ranges;
  for (int a = 0; a < frames; a += size)
    ranges.emplace_back(a, std::min(a + size, frames));
  return ranges;
}

std::uint32_t SampleToken(std::span<const float> logits, Phase phase,
                          const PhaseSampling& sampling,
                          std::span<const std::uint32_t> history, int step,
                          bool legacy_off, std::mt19937_64* rng) {
  const VocabularySlice slice = PhaseSlice(phase);
  const std::size_t n = std::min<std::size_t>(logits.size(), slice.count);
  std::vector<float> scores(logits.begin(), logits.begin() + n);
  if (legacy_off)
    for (float& s : scores)
      s = RoundBf16(s);
  const std::uint32_t end = phase == Phase::kAbc ? kAbcEnd : kMusicEnd;
  if (phase == Phase::kAbc)
    for (std::uint32_t id = kEod; id < kAbcEnd && id < n; ++id)
      scores[id] = kMinusInfinity;
  if (step < sampling.min_tokens)
    scores[end - slice.first] = kMinusInfinity;
  // Windowed repetition penalty: penalty ** count of the id in the window.
  if (sampling.repetition_penalty != 1.0F && !history.empty()) {
    const std::size_t window = std::min<std::size_t>(
        history.size(), static_cast<std::size_t>(sampling.penalty_window));
    std::map<std::uint32_t, int> counts;
    for (std::size_t i = history.size() - window; i < history.size(); ++i)
      ++counts[history[i]];
    for (const auto& [id, count] : counts) {
      if (id < slice.first || id - slice.first >= n)
        continue;
      float& s = scores[id - slice.first];
      const float alpha = std::pow(sampling.repetition_penalty,
                                   static_cast<float>(count));
      s = s < 0 ? s * alpha : s / alpha;
    }
  }
  if (sampling.temperature == 0.0F) {
    const auto best = std::max_element(scores.begin(), scores.end());
    return slice.first + static_cast<std::uint32_t>(best - scores.begin());
  }
  if (sampling.temperature != 1.0F)
    for (float& s : scores)
      s /= sampling.temperature;
  // top_k: keep scores >= the k-th largest.
  const std::size_t k =
      std::min<std::size_t>(n, static_cast<std::size_t>(sampling.top_k));
  std::vector<std::uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0U);
  std::nth_element(order.begin(), order.begin() + (k - 1), order.end(),
                   [&](std::uint32_t a, std::uint32_t b) {
                     return scores[a] > scores[b];
                   });
  const float threshold = scores[order[k - 1]];
  order.clear();
  for (std::uint32_t i = 0; i < n; ++i)
    if (scores[i] >= threshold && scores[i] != kMinusInfinity)
      order.push_back(i);
  std::stable_sort(order.begin(), order.end(),
                   [&](std::uint32_t a, std::uint32_t b) {
                     return scores[a] > scores[b];
                   });
  if (order.empty())
    return end;
  const double top = scores[order[0]];
  std::vector<double> weights(order.size());
  double total = 0;
  for (std::size_t i = 0; i < order.size(); ++i) {
    weights[i] = std::exp(double(scores[order[i]]) - top);
    total += weights[i];
  }
  std::size_t keep = order.size();
  if (sampling.top_p < 1.0F) {
    // removed = cumsum(p) - p > top_p, never the first (legacy: three).
    const std::size_t always = legacy_off ? 3 : 1;
    double before = 0;
    for (std::size_t i = 0; i < order.size(); ++i) {
      if (i >= always && before > sampling.top_p) {
        keep = i;
        break;
      }
      before += weights[i] / total;
    }
  }
  double kept = 0;
  for (std::size_t i = 0; i < keep; ++i)
    kept += weights[i];
  double draw = std::uniform_real_distribution<double>(0.0, kept)(*rng);
  for (std::size_t i = 0; i < keep; ++i) {
    draw -= weights[i];
    if (draw <= 0)
      return slice.first + order[i];
  }
  return slice.first + order[keep - 1];
}

}  // namespace gufo::models::yue2
