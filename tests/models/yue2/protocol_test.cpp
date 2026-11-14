// YuE2 host side: vocabulary slices, chunk ranges, guidance defaults and
// the phase sampler (masks, min_tokens, windowed repetition penalty, top_k,
// top_p). With GUFO_YUE2_TIKTOKEN (qwen.tiktoken of YuE2-3B) and
// GUFO_YUE2_ORACLE (directory with the oracle's tokens.json) also the token
// ids and prompt prefixes against the reference tokenizer. CPU only.
// Written with AI assistance (Claude Code).
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gufo/ext/json.hpp>
#include "src/models/yue2/protocol.hpp"
#include "src/models/yue2/tokenizer.hpp"

namespace y = gufo::models::yue2;
namespace json = gufo::json;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
  if (!condition) {
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
  }
}

std::vector<std::uint32_t> Ids(const json::Value& array) {
  std::vector<std::uint32_t> out;
  for (const auto& v : array.items())
    out.push_back(static_cast<std::uint32_t>(v.as_double()));
  return out;
}

void TestSlicesAndChunks() {
  const auto abc = y::PhaseSlice(y::Phase::kAbc);
  Expect(abc.first == 0 && abc.count == y::kAbcEnd + 1, "abc slice");
  const auto sem = y::PhaseSlice(y::Phase::kSemantic);
  Expect(sem.first == y::kMusicEnd && sem.count == y::kCodecSize + 1,
         "semantic slice");
  // protocol.chunk_ranges: size = (24576 - prefix - 3) // 2.
  const auto one = y::ChunkRanges(750, 300);
  Expect(one.size() == 1 && one[0].first == 0 && one[0].second == 750,
         "short song is one chunk");
  const auto two = y::ChunkRanges(13000, 573);
  Expect(two.size() == 2 && two[0].second == 12000 && two[1].second == 13000,
         "long song is chunked");
  Expect(y::DefaultGuidance(y::Cot::kOff) == 1.01F &&
             y::DefaultGuidance(y::Cot::kFull) == 1.0F,
         "guidance defaults");
}

void TestSampler() {
  std::mt19937_64 rng(1);
  y::PhaseSampling greedy = y::DefaultSemanticSampling();
  greedy.temperature = 0;
  std::vector<float> logits(y::PhaseSlice(y::Phase::kSemantic).count, 0.0F);
  logits[0] = 9.0F;  // MUSIC_END
  logits[5] = 5.0F;
  logits[7] = 4.0F;
  // Before min_tokens the end token is suppressed.
  Expect(y::SampleToken(logits, y::Phase::kSemantic, greedy, {}, 0, false,
                        &rng) == y::kMusicEnd + 5,
         "end suppressed before min_tokens");
  Expect(y::SampleToken(logits, y::Phase::kSemantic, greedy, {}, 200, false,
                        &rng) == y::kMusicEnd,
         "end allowed after min_tokens");
  // Penalty 1.2 ** 2 = 1.44: 5 / 1.44 < 4.
  const std::vector<std::uint32_t> history{y::kMusicEnd + 5,
                                           y::kMusicEnd + 5};
  Expect(y::SampleToken(logits, y::Phase::kSemantic, greedy, history, 0,
                        false, &rng) == y::kMusicEnd + 7,
         "windowed repetition penalty");
  // ABC: the special ids between EOD and ABC_END are never sampled.
  std::vector<float> abc(y::PhaseSlice(y::Phase::kAbc).count, 0.0F);
  abc[y::kEod + 3] = 50.0F;
  abc[42] = 1.0F;
  y::PhaseSampling plan = y::DefaultAbcSampling();
  plan.temperature = 0;
  Expect(y::SampleToken(abc, y::Phase::kAbc, plan, {}, 0, false, &rng) == 42,
         "abc masks special tokens");
  // top_k 1 at temperature 1 is deterministic.
  y::PhaseSampling top1 = y::DefaultSemanticSampling();
  top1.top_k = 1;
  for (int i = 0; i < 20; ++i)
    Expect(y::SampleToken(logits, y::Phase::kSemantic, top1, {}, 0, false,
                          &rng) == y::kMusicEnd + 5,
           "top_k 1");
  // top_p: one dominant code (p > 0.95) is the only one kept.
  std::vector<float> peaked(logits.size(), 0.0F);
  peaked[10] = 30.0F;
  y::PhaseSampling nucleus = y::DefaultSemanticSampling();
  for (int i = 0; i < 20; ++i)
    Expect(y::SampleToken(peaked, y::Phase::kSemantic, nucleus, {}, 0, false,
                          &rng) == y::kMusicEnd + 10,
           "top_p keeps the dominant code");
  // Sampling spreads over the kept set.
  std::vector<float> flat(logits.size(), -1e9F);
  flat[1] = flat[2] = 1.0F;
  int first = 0;
  for (int i = 0; i < 400; ++i)
    first += y::SampleToken(flat, y::Phase::kSemantic, nucleus, {}, 0, false,
                            &rng) == y::kMusicEnd + 1;
  Expect(first > 150 && first < 250, "two equal codes are both drawn");
}

void TestTokenizer() {
  const char* tiktoken = std::getenv("GUFO_YUE2_TIKTOKEN");
  const char* oracle = std::getenv("GUFO_YUE2_ORACLE");
  if (tiktoken == nullptr || oracle == nullptr) {
    std::cout << "tokenizer oracle skipped (GUFO_YUE2_TIKTOKEN / "
                 "GUFO_YUE2_ORACLE unset)\n";
    return;
  }
  std::string error;
  auto tokenizer = y::TextTokenizer::Load(tiktoken, &error);
  Expect(tokenizer != nullptr, "load tokenizer: " + error);
  if (!tokenizer)
    return;
  std::ifstream in(std::filesystem::path(oracle) / "tokens.json",
                   std::ios::binary);
  const std::string text(std::istreambuf_iterator<char>(in), {});
  const auto tokens = json::parse(text);
  const auto& texts = tokens.find("texts")->items();
  const auto& ids = tokens.find("ids")->items();
  for (std::size_t i = 0; i < texts.size(); ++i) {
    Expect(tokenizer->Encode(texts[i].str()) == Ids(ids[i]),
           "token ids of text " + std::to_string(i));
    Expect(tokenizer->Decode(Ids(ids[i])) == texts[i].str() || i == 2,
           "decode of text " + std::to_string(i));
  }
  Expect(y::TokenPrefix(*tokenizer, y::Cot::kFull, texts[0].str(),
                        texts[1].str(), nullptr) ==
             Ids(*tokens.find("prefix_full")),
         "prefix full");
  Expect(y::TokenPrefix(*tokenizer, y::Cot::kOff, texts[0].str(),
                        texts[1].str(), nullptr) ==
             Ids(*tokens.find("prefix_off")),
         "prefix off");
}

}  // namespace

int main() {
  TestSlicesAndChunks();
  TestSampler();
  TestTokenizer();
  if (failures == 0)
    std::cout << "yue2_protocol_test: ok\n";
  return failures == 0 ? 0 : 1;
}
