// imgutils.ocr.ocr natively (see ocr.hpp). Written with AI assistance (Claude Code).
#include "ccip/ocr.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string_view>

#include "ccip/cpu/ocr_net.hpp"

namespace gufo::models::ccip {
namespace {

constexpr int kAlign = 64;
// imgutils ocr/detect.py _normalize (CLIP statistics).
constexpr double kMean[3] = {0.48145466, 0.4578275, 0.40821073};
constexpr double kStd[3] = {0.26862954, 0.26130258, 0.27577711};

// Python str.strip() for the whitespace that can occur in dict.txt (ASCII
// and the Unicode spaces Python strips; the dictionary has one U+3000 line).
bool StripOne(std::string& s, bool front) {
  static const std::string_view kSpaces[] = {
      "\t", "\n", "\v", "\f", "\r", " ", "\x1c", "\x1d", "\x1e", "\x1f",
      "\xc2\x85", "\xc2\xa0", "\xe1\x9a\x80", "\xe2\x80\x80", "\xe2\x80\x81", "\xe2\x80\x82",
      "\xe2\x80\x83", "\xe2\x80\x84", "\xe2\x80\x85", "\xe2\x80\x86", "\xe2\x80\x87",
      "\xe2\x80\x88", "\xe2\x80\x89", "\xe2\x80\x8a", "\xe2\x80\xa8", "\xe2\x80\xa9",
      "\xe2\x80\xaf", "\xe2\x81\x9f", "\xe3\x80\x80"};
  for (std::string_view sp : kSpaces) {
    if (s.size() < sp.size()) continue;
    if (front && std::string_view(s).substr(0, sp.size()) == sp) {
      s.erase(0, sp.size());
      return true;
    }
    if (!front && std::string_view(s).substr(s.size() - sp.size()) == sp) {
      s.erase(s.size() - sp.size());
      return true;
    }
  }
  return false;
}

std::string PyStrip(std::string s) {
  while (!s.empty() && StripOne(s, true)) {
  }
  while (!s.empty() && StripOne(s, false)) {
  }
  return s;
}

// numpy's pairwise float32 summation (add.reduce on a contiguous float32 array).
float PairwiseSum(const float* a, std::size_t n) {
  if (n < 8) {
    float res = 0.f;
    for (std::size_t i = 0; i < n; i++) res += a[i];
    return res;
  }
  if (n <= 128) {
    float r[8];
    for (int k = 0; k < 8; ++k) r[k] = a[k];
    std::size_t i = 8;
    for (; i < n - (n % 8); i += 8)
      for (int k = 0; k < 8; ++k) r[k] += a[i + static_cast<std::size_t>(k)];
    float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[i];
    return res;
  }
  std::size_t n2 = n / 2;
  n2 -= n2 % 8;
  return PairwiseSum(a, n2) + PairwiseSum(a + n2, n - n2);
}

// Pillow Image.rotate(90) without expand: same size, nearest neighbour
// affine map around the centre, uncovered pixels black (a plain transpose
// when the image is square).
RgbImage Rotate90(const RgbImage& in) {
  RgbImage out;
  out.width = in.width;
  out.height = in.height;
  out.rgb.assign(in.rgb.size(), 0);
  const double cx = in.width / 2.0, cy = in.height / 2.0;
  for (int y = 0; y < out.height; ++y) {
    for (int x = 0; x < out.width; ++x) {
      const double xs = cx + cy - 0.5 - y;  // matrix (0, -1, cx + cy; 1, 0, cy - cx)
      const double ys = cy - cx + 0.5 + x;
      const int xi = xs < 0.0 ? -1 : static_cast<int>(xs);
      const int yi = ys < 0.0 ? -1 : static_cast<int>(ys);
      if (xi < 0 || xi >= in.width || yi < 0 || yi >= in.height) continue;
      std::memcpy(&out.rgb[(static_cast<std::size_t>(y) * out.width + x) * 3],
                  &in.rgb[(static_cast<std::size_t>(yi) * in.width + xi) * 3], 3);
    }
  }
  return out;
}

class CpuTextOcr final : public TextOcr {
public:
  bool Init(const std::filesystem::path& dir, const std::string& det_stem,
            const std::string& rec_stem, std::string* error) {
    det_ = cpu::OcrNet::Load(dir, det_stem, error);
    if (det_ == nullptr) return false;
    rec_ = cpu::OcrNet::Load(dir, rec_stem, error);
    if (rec_ == nullptr) return false;
    const auto dict_path = dir / (rec_stem + ".dict.txt");
    std::ifstream in(dict_path, std::ios::binary);
    if (!in) {
      if (error) *error = "missing " + dict_path.string();
      return false;
    }
    const std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    // ['<blank>', *[line.strip() for line in f], ' '] (Python text mode).
    vocab_.push_back("<blank>");
    std::size_t pos = 0;
    while (pos < raw.size()) {
      std::size_t end = raw.find('\n', pos);
      if (end == std::string::npos) end = raw.size();
      vocab_.push_back(PyStrip(raw.substr(pos, end - pos)));
      pos = end + 1;
    }
    vocab_.push_back(" ");
    return true;
  }

  std::vector<float> Heatmap(const RgbImage& image) const override {
    int pw = 0, ph = 0;
    cpu::NetTensor x;
    x.f = OcrDetInput(image, &pw, &ph);
    x.shape = {1, 3, ph, pw};
    const cpu::NetTensor y = det_->Run(std::move(x));
    if (y.shape.size() != 4 || y.shape[2] != ph || y.shape[3] != pw)
      throw std::runtime_error("unexpected detector output shape");
    std::vector<float> heat(static_cast<std::size_t>(image.width) * image.height);
    for (int r = 0; r < image.height; ++r)
      std::memcpy(&heat[static_cast<std::size_t>(r) * image.width],
                  &y.f[static_cast<std::size_t>(r) * pw], sizeof(float) * image.width);
    return heat;
  }

  std::pair<std::string, double> Recognize(const RgbImage& crop,
                                           bool remove_duplicate) const override {
    if (crop.width <= 0 || crop.height <= 0) return {std::string(), 0.0};
    cpu::NetTensor x;
    x.shape = {1, 3, crop.height, crop.width};
    const std::size_t plane = static_cast<std::size_t>(crop.width) * crop.height;
    x.f.resize(plane * 3);
    for (std::size_t i = 0; i < plane; ++i)
      for (int c = 0; c < 3; ++c)
        x.f[static_cast<std::size_t>(c) * plane + i] =
            (static_cast<float>(crop.rgb[i * 3 + static_cast<std::size_t>(c)]) / 255.0f - 0.5f) / 0.5f;
    const cpu::NetTensor y = rec_->Run(std::move(x));
    if (y.shape.size() != 3 || y.shape[2] != static_cast<std::int64_t>(vocab_.size()))
      throw std::runtime_error("unexpected recognizer output shape");
    const std::int64_t steps = y.shape[1], classes = y.shape[2];
    std::string text;
    std::vector<float> confs;
    std::int64_t prev = -1;
    for (std::int64_t t = 0; t < steps; ++t) {
      const float* row = y.f.data() + t * classes;
      std::int64_t best = 0;
      for (std::int64_t c = 1; c < classes; ++c)
        if (row[c] > row[best]) best = c;
      const bool keep = best != 0 && !(remove_duplicate && t > 0 && best == prev);
      prev = best;
      if (!keep) continue;
      text += vocab_[static_cast<std::size_t>(best)];
      confs.push_back(row[best]);
    }
    if (confs.empty()) return {text, 0.0};
    const float mean = PairwiseSum(confs.data(), confs.size()) / static_cast<float>(confs.size());
    return {text, static_cast<double>(mean)};
  }

private:
  std::unique_ptr<cpu::OcrNet> det_;
  std::unique_ptr<cpu::OcrNet> rec_;
  std::vector<std::string> vocab_;
};

}  // namespace

std::vector<float> OcrDetInput(const RgbImage& image, int* padded_width, int* padded_height) {
  int w = image.width, h = image.height;
  if (w % kAlign != 0) w += kAlign - w % kAlign;
  if (h % kAlign != 0) h += kAlign - h % kAlign;
  *padded_width = w;
  *padded_height = h;
  const std::size_t plane = static_cast<std::size_t>(w) * h;
  std::vector<float> x(plane * 3);
  for (int c = 0; c < 3; ++c) {
    // np.pad with 0 before the (float64) normalisation.
    const float pad = static_cast<float>((0.0 - kMean[c]) / kStd[c]);
    float* dst = x.data() + static_cast<std::size_t>(c) * plane;
    for (int y = 0; y < h; ++y) {
      for (int xx = 0; xx < w; ++xx) {
        float v = pad;
        if (y < image.height && xx < image.width) {
          const float u = static_cast<float>(
                              image.rgb[(static_cast<std::size_t>(y) * image.width + xx) * 3 +
                                        static_cast<std::size_t>(c)]) /
                          255.0f;
          v = static_cast<float>((static_cast<double>(u) - kMean[c]) / kStd[c]);
        }
        dst[static_cast<std::size_t>(y) * w + xx] = v;
      }
    }
  }
  return x;
}

RgbImage OcrTextCrop(const RgbImage& image, const std::array<int, 4>& box,
                     double rotation_threshold, int height) {
  const int bw = box[2] - box[0];
  const int bh = box[3] - box[1];
  if (bw <= 0 || bh <= 0) return RgbImage{};
  RgbImage area = Crop(image, box[0], box[1], box[2], box[3]);
  if (bh >= bw * rotation_threshold) area = Rotate90(area);
  // r = 48 / h; int(round(h * r)), int(round(w * r)) (Python round: half even).
  const double r = static_cast<double>(height) / area.height;
  const int new_height = static_cast<int>(std::nearbyint(area.height * r));
  const int new_width = static_cast<int>(std::nearbyint(area.width * r));
  if (new_width <= 0 || new_height <= 0) return RgbImage{};
  if (new_width == area.width && new_height == area.height) return area;
  return ResizeBicubic(area, new_width, new_height);
}

std::unique_ptr<TextOcr> TextOcr::Load(const std::filesystem::path& dir,
                                       const std::string& det_stem, const std::string& rec_stem,
                                       std::string* error) {
  auto ocr = std::make_unique<CpuTextOcr>();
  if (!ocr->Init(dir, det_stem, rec_stem, error)) return nullptr;
  return ocr;
}

std::vector<OcrText> TextOcr::Run(const RgbImage& image, const OcrOptions& options) const {
  std::vector<OcrText> out;
  if (image.width <= 0 || image.height <= 0) return out;
  const std::vector<float> heat = Heatmap(image);
  for (const TextBox& tb : TextBoxesFromHeatmap(heat.data(), image.width, image.height, options)) {
    OcrText t;
    t.points = tb.points;
    t.box = {tb.points[0][0], tb.points[0][1], tb.points[0][0], tb.points[0][1]};
    for (const auto& p : tb.points) {
      t.box[0] = std::min(t.box[0], p[0]);
      t.box[1] = std::min(t.box[1], p[1]);
      t.box[2] = std::max(t.box[2], p[0]);
      t.box[3] = std::max(t.box[3], p[1]);
    }
    const RgbImage crop = OcrTextCrop(image, t.box, options.rotation_threshold);
    auto [text, text_score] = Recognize(crop, options.remove_duplicate);
    t.text = std::move(text);
    t.det_score = tb.score;
    t.text_score = text_score;
    t.score = tb.score * text_score;
    out.push_back(std::move(t));
  }
  std::stable_sort(out.begin(), out.end(),
                   [](const OcrText& a, const OcrText& b) { return a.score > b.score; });
  return out;
}

}  // namespace gufo::models::ccip
