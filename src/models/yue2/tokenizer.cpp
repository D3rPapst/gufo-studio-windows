// Written with AI assistance (Claude Code).
#include "src/models/yue2/tokenizer.hpp"

#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/unistr.h>

#include <cstddef>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace gufo::models::yue2 {
namespace {

// The split regex of yue2/tokenization_yue2.py (Qwen2 / Qwen3).
constexpr const char* kPattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

int Base64Value(char c) {
  if (c >= 'A' && c <= 'Z')
    return c - 'A';
  if (c >= 'a' && c <= 'z')
    return c - 'a' + 26;
  if (c >= '0' && c <= '9')
    return c - '0' + 52;
  if (c == '+')
    return 62;
  if (c == '/')
    return 63;
  return -1;
}

bool DecodeBase64(std::string_view text, std::string* out) {
  out->clear();
  unsigned bits = 0;
  int count = 0;
  for (const char c : text) {
    if (c == '=')
      break;
    const int value = Base64Value(c);
    if (value < 0)
      return false;
    bits = (bits << 6U) | static_cast<unsigned>(value);
    count += 6;
    if (count >= 8) {
      count -= 8;
      out->push_back(static_cast<char>((bits >> count) & 0xFFU));
    }
  }
  return !out->empty();
}

}  // namespace

std::unique_ptr<TextTokenizer> TextTokenizer::Load(
    const std::filesystem::path& tiktoken, std::string* error) {
  std::ifstream in(tiktoken, std::ios::binary);
  if (!in) {
    if (error != nullptr)
      *error = "cannot read " + tiktoken.string();
    return nullptr;
  }
  const std::string data(std::istreambuf_iterator<char>(in), {});
  std::unique_ptr<TextTokenizer> t(new TextTokenizer());
  t->tokens_.resize(kOrdinaryTokens);
  std::size_t start = 0, count = 0;
  std::string bytes;
  while (start < data.size()) {
    std::size_t end = data.find('\n', start);
    if (end == std::string::npos)
      end = data.size();
    std::string_view line(data.data() + start, end - start);
    start = end + 1;
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
    if (line.empty())
      continue;
    const std::size_t space = line.find(' ');
    std::uint32_t rank = 0;
    bool ok = space != std::string_view::npos && space + 1 < line.size() &&
              DecodeBase64(line.substr(0, space), &bytes);
    for (std::size_t i = space + 1; ok && i < line.size(); ++i) {
      ok = line[i] >= '0' && line[i] <= '9' && rank < kOrdinaryTokens;
      rank = rank * 10 + static_cast<std::uint32_t>(line[i] - '0');
    }
    if (!ok || rank >= kOrdinaryTokens || !t->tokens_[rank].empty() ||
        !t->ranks_.emplace(bytes, rank).second) {
      if (error != nullptr)
        *error = "invalid qwen.tiktoken line " + std::to_string(count + 1);
      return nullptr;
    }
    t->tokens_[rank] = bytes;
    ++count;
  }
  if (count != kOrdinaryTokens) {
    if (error != nullptr)
      *error = "expected 151643 ranks in " + tiktoken.string() + ", found " +
               std::to_string(count);
    return nullptr;
  }
  return t;
}

// tiktoken's byte pair merge: repeatedly join the adjacent pair whose
// concatenation has the lowest rank (the leftmost on ties).
void TextTokenizer::Merge(std::string_view piece,
                          std::vector<std::uint32_t>* out) const {
  if (const auto whole = ranks_.find(std::string(piece));
      whole != ranks_.end()) {
    out->push_back(whole->second);
    return;
  }
  std::vector<std::size_t> bounds(piece.size() + 1);
  for (std::size_t i = 0; i <= piece.size(); ++i)
    bounds[i] = i;
  const auto rank_of = [&](std::size_t i) {
    const auto found = ranks_.find(
        std::string(piece.substr(bounds[i], bounds[i + 2] - bounds[i])));
    return found == ranks_.end() ? std::numeric_limits<std::uint32_t>::max()
                                 : found->second;
  };
  while (bounds.size() > 2) {
    std::uint32_t best = std::numeric_limits<std::uint32_t>::max();
    std::size_t at = 0;
    for (std::size_t i = 0; i + 2 < bounds.size(); ++i) {
      const std::uint32_t rank = rank_of(i);
      if (rank < best) {
        best = rank;
        at = i;
      }
    }
    if (best == std::numeric_limits<std::uint32_t>::max())
      break;
    bounds.erase(bounds.begin() + static_cast<std::ptrdiff_t>(at + 1));
  }
  for (std::size_t i = 0; i + 1 < bounds.size(); ++i) {
    const auto found = ranks_.find(
        std::string(piece.substr(bounds[i], bounds[i + 1] - bounds[i])));
    if (found == ranks_.end())
      throw std::runtime_error("YuE2 tokenizer: byte without a rank");
    out->push_back(found->second);
  }
}

std::vector<std::uint32_t> TextTokenizer::Encode(std::string_view text) const {
  std::vector<std::uint32_t> result;
  UErrorCode status = U_ZERO_ERROR;
  const auto pattern =
      std::unique_ptr<icu::RegexPattern>(icu::RegexPattern::compile(
          icu::UnicodeString::fromUTF8(kPattern), 0, status));
  if (U_FAILURE(status) || !pattern)
    throw std::runtime_error("YuE2 tokenizer: invalid regex");
  const auto* normalizer = icu::Normalizer2::getNFCInstance(status);
  if (U_FAILURE(status))
    throw std::runtime_error("YuE2 tokenizer: NFC initialization failed");
  icu::UnicodeString normalized;
  normalizer->normalize(
      icu::UnicodeString::fromUTF8(
          icu::StringPiece(text.data(), static_cast<int>(text.size()))),
      normalized, status);
  if (U_FAILURE(status))
    throw std::runtime_error("YuE2 tokenizer: NFC failed");
  const auto matcher =
      std::unique_ptr<icu::RegexMatcher>(pattern->matcher(normalized, status));
  if (U_FAILURE(status) || !matcher)
    throw std::runtime_error("YuE2 tokenizer: Unicode processing failed");
  int consumed = 0;
  std::string piece;
  while (matcher->find(status)) {
    if (matcher->start(status) != consumed)
      throw std::runtime_error("YuE2 tokenizer left unmatched text");
    piece.clear();
    matcher->group(status).toUTF8String(piece);
    Merge(piece, &result);
    consumed = matcher->end(status);
  }
  if (U_FAILURE(status) || consumed != normalized.length())
    throw std::runtime_error("YuE2 tokenization failed");
  return result;
}

std::string TextTokenizer::Decode(std::span<const std::uint32_t> ids) const {
  std::string bytes;
  for (const std::uint32_t id : ids)
    if (id < tokens_.size())
      bytes += tokens_[id];
  // Replace invalid UTF-8 with U+FFFD.
  std::string out;
  out.reserve(bytes.size());
  const auto* p = reinterpret_cast<const unsigned char*>(bytes.data());
  const std::size_t n = bytes.size();
  for (std::size_t i = 0; i < n;) {
    const unsigned char c = p[i];
    std::size_t length = c < 0x80 ? 1 : (c >> 5U) == 6 ? 2 : (c >> 4U) == 14 ? 3
                                     : (c >> 3U) == 30 ? 4 : 0;
    bool valid = length != 0 && i + length <= n;
    for (std::size_t k = 1; valid && k < length; ++k)
      valid = (p[i + k] & 0xC0U) == 0x80U;
    if (valid && length == 2)
      valid = c >= 0xC2;
    if (valid && length == 3) {
      const unsigned cp = ((c & 0x0FU) << 12U) | ((p[i + 1] & 0x3FU) << 6U) |
                          (p[i + 2] & 0x3FU);
      valid = cp >= 0x800 && (cp < 0xD800 || cp > 0xDFFF);
    }
    if (valid && length == 4) {
      const unsigned cp = ((c & 0x07U) << 18U) | ((p[i + 1] & 0x3FU) << 12U) |
                          ((p[i + 2] & 0x3FU) << 6U) | (p[i + 3] & 0x3FU);
      valid = cp >= 0x10000 && cp <= 0x10FFFF;
    }
    if (valid) {
      out.append(bytes, i, length);
      i += length;
    } else {
      out += "\xEF\xBF\xBD";
      ++i;
    }
  }
  return out;
}

}  // namespace gufo::models::yue2
