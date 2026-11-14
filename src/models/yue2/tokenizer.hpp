// YuE2 text and ABC tokenizer: the checkpoint's `qwen.tiktoken` (151643
// byte-level BPE ranks, the Qwen2 / Qwen3 split regex, NFC), encoded like
// tiktoken's `encode_ordinary` (special-token text stays ordinary text).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_YUE2_TOKENIZER_HPP_
#define GUFO_MODELS_YUE2_TOKENIZER_HPP_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gufo::models::yue2 {

class TextTokenizer {
public:
  static constexpr std::uint32_t kOrdinaryTokens = 151643;

  /// Reads `qwen.tiktoken` (base64 token, space, rank per line).
  [[nodiscard]] static std::unique_ptr<TextTokenizer> Load(
      const std::filesystem::path& tiktoken, std::string* error);

  /// Ordinary token ids of NFC(`text`). Throws std::runtime_error on
  /// invalid UTF-8.
  [[nodiscard]] std::vector<std::uint32_t> Encode(std::string_view text) const;
  /// UTF-8 text of ordinary ids (other ids are skipped; invalid UTF-8 byte
  /// sequences become U+FFFD as tiktoken's errors="replace" does).
  [[nodiscard]] std::string Decode(std::span<const std::uint32_t> ids) const;

private:
  TextTokenizer() = default;
  void Merge(std::string_view piece, std::vector<std::uint32_t>* out) const;

  std::unordered_map<std::string, std::uint32_t> ranks_;
  std::vector<std::string> tokens_;
};

}  // namespace gufo::models::yue2

#endif  // GUFO_MODELS_YUE2_TOKENIZER_HPP_
