// CPU side of Grounding DINO (see grounding_dino.hpp): BERT tokenizer, the
// phrase masks and the image processor.
// Written with AI assistance (Claude Code).
#include <algorithm>
#include <cmath>
#include <fstream>

#include "ccip/image_ops.hpp"
#include "src/models/vision_dsg/grounding_dino.hpp"

namespace gufo::models::vision_dsg {
namespace {

constexpr int kCls = 101, kSep = 102, kUnk = 100;

bool IsPunctuation(unsigned char c) {
  return (c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) ||
         (c >= 123 && c <= 126);
}

void ReplaceAll(std::string* text, const std::string& from, const std::string& to) {
  std::size_t at = 0;
  while ((at = text->find(from, at)) != std::string::npos) {
    text->replace(at, from.size(), to);
    at += to.size();
  }
}

}  // namespace

bool BertTokenizer::Load(const std::filesystem::path& vocab, std::string* error) {
  std::ifstream in(vocab, std::ios::binary);
  if (!in) {
    *error = "cannot read " + vocab.string();
    return false;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    vocab_.emplace(line, static_cast<int>(tokens_.size()));
    tokens_.push_back(line);
  }
  if (tokens_.size() <= static_cast<std::size_t>(kSep)) {
    *error = "vocab.txt is too short";
    return false;
  }
  return true;
}

std::vector<int> BertTokenizer::Encode(const std::string& text) const {
  // BasicTokenizer: control characters dropped, whitespace splits, ASCII
  // lower-casing, every ASCII punctuation character its own word.
  std::vector<std::string> words;
  std::string word;
  const auto flush = [&] {
    if (!word.empty())
      words.push_back(word);
    word.clear();
  };
  for (unsigned char c : text) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      flush();
    } else if (c < 32 || c == 127) {
      continue;
    } else if (IsPunctuation(c)) {
      flush();
      words.emplace_back(1, static_cast<char>(c));
    } else {
      word += static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    }
  }
  flush();
  std::vector<int> ids = {kCls};
  for (const std::string& w : words) {
    if (w.size() > 100) {
      ids.push_back(kUnk);
      continue;
    }
    std::vector<int> pieces;
    std::size_t start = 0;
    bool bad = false;
    while (start < w.size()) {
      std::size_t end = w.size();
      int found = -1;
      while (start < end) {
        std::string sub = w.substr(start, end - start);
        if (start > 0)
          sub = "##" + sub;
        if (const auto it = vocab_.find(sub); it != vocab_.end()) {
          found = it->second;
          break;
        }
        --end;
      }
      if (found < 0) {
        bad = true;
        break;
      }
      pieces.push_back(found);
      start = end;
    }
    if (bad)
      ids.push_back(kUnk);
    else
      ids.insert(ids.end(), pieces.begin(), pieces.end());
  }
  ids.push_back(kSep);
  return ids;
}

std::string BertTokenizer::Decode(const std::vector<int>& ids) const {
  std::string text;
  for (int id : ids) {
    if (id < 0 || static_cast<std::size_t>(id) >= tokens_.size())
      continue;
    const std::string& token = tokens_[id];
    if (token.rfind("##", 0) == 0 && !text.empty()) {
      text += token.substr(2);
    } else {
      if (!text.empty())
        text += ' ';
      text += token;
    }
  }
  for (const auto& [from, to] :
       std::vector<std::pair<std::string, std::string>>{
           {" .", "."}, {" ?", "?"}, {" !", "!"}, {" ,", ","}, {" ' ", "'"},
           {" n't", "n't"}, {" 'm", "'m"}, {" 's", "'s"}, {" 've", "'ve"},
           {" 're", "'re"}})
    ReplaceAll(&text, from, to);
  return text;
}

void GroundingTextMasks(const std::vector<int>& ids, std::vector<float>* mask,
                        std::vector<int>* positions) {
  const int n = static_cast<int>(ids.size());
  const auto special = [&](int i) {
    const int id = ids[i];
    return id == 101 || id == 102 || id == 1012 || id == 1029;
  };
  std::vector<int> prev(n), next(n);
  int last = -1;
  for (int i = 0; i < n; ++i) {
    if (special(i))
      last = i;
    prev[i] = last;
  }
  int upcoming = n;
  for (int i = n - 1; i >= 0; --i) {
    if (special(i))
      upcoming = i;
    next[i] = upcoming;
  }
  std::vector<bool> valid(n);
  for (int i = 0; i < n; ++i)
    valid[i] = next[i] != 0 && next[i] != n - 1 && next[i] != n;
  mask->assign(static_cast<std::size_t>(n) * n, 0.0F);
  positions->assign(n, 0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j)
      if (i == j || (next[i] == next[j] && valid[j]))
        (*mask)[static_cast<std::size_t>(i) * n + j] = 1.0F;
    (*positions)[i] = valid[i] ? std::max(i - prev[i] - 1, 0) : 0;
  }
}

void GroundingInputSize(int width, int height, int* net_width, int* net_height) {
  constexpr double kShort = 800, kLong = 1333;
  double size = kShort;
  double raw = -1;
  const double lo = std::min(width, height), hi = std::max(width, height);
  if (hi / lo * size > kLong) {
    raw = kLong * lo / hi;
    size = std::nearbyint(raw);
  }
  const int s = static_cast<int>(size);
  if ((height <= width && height == s) || (width <= height && width == s)) {
    *net_height = height;
    *net_width = width;
  } else if (width < height) {
    *net_width = s;
    *net_height = static_cast<int>(raw >= 0 ? raw * height / width
                                            : size * height / width);
  } else {
    *net_height = s;
    *net_width = static_cast<int>(raw >= 0 ? raw * width / height
                                           : size * width / height);
  }
}

std::vector<float> GroundingPixels(const std::uint8_t* rgb, int width,
                                   int height, int* net_width,
                                   int* net_height) {
  if (rgb == nullptr || width < 1 || height < 1)
    return {};
  GroundingInputSize(width, height, net_width, net_height);
  ccip::RgbImage image;
  image.width = width;
  image.height = height;
  image.rgb.assign(rgb, rgb + static_cast<std::size_t>(width) * height * 3);
  const ccip::RgbImage resized =
      (*net_width == width && *net_height == height)
          ? image
          : ccip::ResizeBilinear(image, *net_width, *net_height);
  constexpr float kMean[3] = {0.485F, 0.456F, 0.406F};
  constexpr float kStd[3] = {0.229F, 0.224F, 0.225F};
  const std::size_t plane = static_cast<std::size_t>(*net_width) * *net_height;
  std::vector<float> out(plane * 3);
  for (std::size_t i = 0; i < plane; ++i)
    for (int c = 0; c < 3; ++c) {
      const float v = static_cast<float>(resized.rgb[i * 3 + c] * (1.0 / 255.0));
      out[c * plane + i] = (v - kMean[c]) / kStd[c];
    }
  return out;
}

}  // namespace gufo::models::vision_dsg
