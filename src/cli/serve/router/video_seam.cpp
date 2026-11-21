#include "src/cli/serve/router/video_seam.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <system_error>
#include <utility>

namespace gufo::router::seam {

namespace fs = std::filesystem;

namespace {

double Round2(double value) { return std::round(value * 100.0) / 100.0; }

std::uint8_t Clamp8(long value) {
  return static_cast<std::uint8_t>(std::clamp(value, 0L, 255L));
}

}  // namespace

bool ParseOptions(const json::Value* value, const std::string& where,
                  Options* out, std::string* error) {
  *out = Options{};
  if (value == nullptr || value->is_null())
    return true;
  if (value->is_bool()) {
    if (value->as_bool()) {
      out->color_match = true;
      out->blend_frames = kDefaultBlendFrames;
    }
    return true;
  }
  if (!value->is_object()) {
    *error = where + " must be true, false or {\"color_match\": bool, "
                     "\"blend_frames\": 0.." +
             std::to_string(kMaxBlendFrames) + "}";
    return false;
  }
  for (const auto& [key, member] : value->members()) {
    if (key == "color_match") {
      if (member.is_null())
        continue;
      if (!member.is_bool()) {
        *error = where + ".color_match must be true or false";
        return false;
      }
      out->color_match = member.as_bool();
    } else if (key == "blend_frames") {
      if (member.is_null())
        continue;
      const double frames = member.as_double(-1);
      if (!member.is_number() || std::floor(frames) != frames || frames < 0 ||
          frames > kMaxBlendFrames) {
        *error = where + ".blend_frames must be a whole number 0.." +
                 std::to_string(kMaxBlendFrames);
        return false;
      }
      out->blend_frames = static_cast<int>(frames);
    } else {
      *error = where + " may contain only color_match and blend_frames, not " +
               key;
      return false;
    }
  }
  return true;
}

json::Value OptionsJson(const Options& options) {
  json::Value value = json::Value::object();
  value["color_match"] = options.color_match;
  value["blend_frames"] = options.blend_frames;
  return value;
}

double MeanAbsDiff(const std::uint8_t* a, const std::uint8_t* b,
                   std::size_t bytes) {
  if (bytes == 0)
    return 0;
  unsigned long long sum = 0;
  for (std::size_t i = 0; i < bytes; ++i)
    sum += static_cast<unsigned>(a[i] > b[i] ? a[i] - b[i] : b[i] - a[i]);
  return static_cast<double>(sum) / static_cast<double>(bytes);
}

double Median(std::vector<double> values) {
  if (values.empty())
    return 0;
  const auto middle = values.begin() + static_cast<long>(values.size() / 2);
  std::nth_element(values.begin(), middle, values.end());
  return *middle;
}

ColorTransform FitColorTransform(const std::uint8_t* reference,
                                 const std::uint8_t* frame,
                                 std::size_t pixels) {
  ColorTransform transform;
  if (pixels == 0)
    return transform;
  for (int c = 0; c < 3; ++c) {
    double sum_r = 0, sum_f = 0, sq_r = 0, sq_f = 0;
    for (std::size_t p = 0; p < pixels; ++p) {
      const double r = reference[p * 3 + c];
      const double f = frame[p * 3 + c];
      sum_r += r;
      sum_f += f;
      sq_r += r * r;
      sq_f += f * f;
    }
    const double n = static_cast<double>(pixels);
    const double mean_r = sum_r / n;
    const double mean_f = sum_f / n;
    const double std_r = std::sqrt(std::max(0.0, sq_r / n - mean_r * mean_r));
    const double std_f = std::sqrt(std::max(0.0, sq_f / n - mean_f * mean_f));
    double gain = 1;
    if (std_r >= 1 && std_f >= 1)
      gain = std::clamp(std_r / std_f, 0.75, 4.0 / 3.0);
    transform.gain[c] = static_cast<float>(gain);
    transform.offset[c] = static_cast<float>(mean_r - gain * mean_f);
  }
  return transform;
}

void ApplyColorTransform(const ColorTransform& transform, std::uint8_t* frame,
                         std::size_t pixels) {
  std::uint8_t table[3][256];
  for (int c = 0; c < 3; ++c)
    for (int x = 0; x < 256; ++x)
      table[c][x] = Clamp8(std::lround(static_cast<double>(x) *
                                           transform.gain[c] +
                                       transform.offset[c]));
  for (std::size_t p = 0; p < pixels; ++p)
    for (int c = 0; c < 3; ++c)
      frame[p * 3 + c] = table[c][frame[p * 3 + c]];
}

std::string FormatStats(const Stats& stats) {
  char text[320];
  std::snprintf(text, sizeof text,
                "frames=%d drift=%.2f->%.2f seam1=%.2f->%.2f typical=%.2f "
                "gain=%.3f,%.3f,%.3f offset=%.1f,%.1f,%.1f",
                stats.frames, stats.drift_before, stats.drift_after,
                stats.seam1_before, stats.seam1_after, stats.typical,
                stats.transform.gain[0], stats.transform.gain[1],
                stats.transform.gain[2], stats.transform.offset[0],
                stats.transform.offset[1], stats.transform.offset[2]);
  return text;
}

json::Value StatsJson(const Stats& stats) {
  json::Value value = json::Value::object();
  value["frames"] = stats.frames;
  value["drift_before"] = Round2(stats.drift_before);
  value["drift_after"] = Round2(stats.drift_after);
  value["seam1_before"] = Round2(stats.seam1_before);
  value["seam1_after"] = Round2(stats.seam1_after);
  value["typical"] = Round2(stats.typical);
  json::Value gain = json::Value::array();
  json::Value offset = json::Value::array();
  for (int c = 0; c < 3; ++c) {
    gain.push_back(std::round(stats.transform.gain[c] * 1000.0) / 1000.0);
    offset.push_back(std::round(stats.transform.offset[c] * 10.0) / 10.0);
  }
  value["gain"] = std::move(gain);
  value["offset"] = std::move(offset);
  return value;
}

Corrector::Corrector(const Options& options, int width, int height,
                     std::vector<std::uint8_t> reference)
    : options_(options),
      pixels_(static_cast<std::size_t>(std::max(0, width)) *
              static_cast<std::size_t>(std::max(0, height))),
      reference_(std::move(reference)) {
  options_.blend_frames = std::clamp(options_.blend_frames, 0, kMaxBlendFrames);
  reference_.resize(pixels_ * 3);
}

void Corrector::Process(std::uint8_t* frame) {
  const std::size_t bytes = reference_.size();
  const int k = index_++;
  ++stats_.frames;
  // Measures on the frame as the model made it.
  if (k == 0)
    stats_.drift_before = MeanAbsDiff(reference_.data(), frame, bytes);
  if (k == 1)
    stats_.seam1_before = MeanAbsDiff(reference_.data(), frame, bytes);
  if (k >= 2 && static_cast<int>(pairs_.size()) < kTypicalPairs) {
    pairs_.push_back(MeanAbsDiff(previous_.data(), frame, bytes));
    stats_.typical = Median(pairs_);
  }
  if (k >= 1 && static_cast<int>(pairs_.size()) < kTypicalPairs)
    previous_.assign(frame, frame + bytes);
  else if (k >= 1)
    previous_.clear();

  if (options_.color_match) {
    if (k == 0)
      stats_.transform =
          FitColorTransform(reference_.data(), frame, pixels_);
    ApplyColorTransform(stats_.transform, frame, pixels_);
  }
  const int n = options_.blend_frames;
  if (n > 0) {
    if (k == 0) {
      residual_.resize(bytes);
      for (std::size_t i = 0; i < bytes; ++i)
        residual_[i] = static_cast<std::int16_t>(
            static_cast<int>(reference_[i]) - static_cast<int>(frame[i]));
    }
    if (k < n) {
      const float weight =
          static_cast<float>(n - k) / static_cast<float>(n);
      for (std::size_t i = 0; i < bytes; ++i)
        frame[i] = Clamp8(static_cast<long>(frame[i]) +
                          std::lround(residual_[i] * weight));
    } else if (!residual_.empty()) {
      residual_.clear();
      residual_.shrink_to_fit();
    }
  }
  if (k == 0)
    stats_.drift_after = MeanAbsDiff(reference_.data(), frame, bytes);
  if (k == 1)
    stats_.seam1_after = MeanAbsDiff(reference_.data(), frame, bytes);
}

bool ReadRawImage(const fs::path& file, int width, int height,
                  std::vector<std::uint8_t>* image, std::string* error) {
  const std::size_t want = static_cast<std::size_t>(std::max(0, width)) *
                           static_cast<std::size_t>(std::max(0, height)) * 3;
  std::error_code ec;
  const auto size = fs::file_size(file, ec);
  if (ec || want == 0 || size != want) {
    *error = "the start frame is " + (ec ? std::string("unreadable") :
                                           std::to_string(size) + " bytes") +
             ", not " + std::to_string(width) + "x" + std::to_string(height) +
             " RGB";
    return false;
  }
  std::ifstream in(file, std::ios::binary);
  image->resize(want);
  if (!in.read(reinterpret_cast<char*>(image->data()),
               static_cast<std::streamsize>(want))) {
    *error = "cannot read the start frame";
    return false;
  }
  return true;
}

bool LastRawFrame(const fs::path& file, int width, int height,
                  std::vector<std::uint8_t>* frame, std::string* error) {
  const std::size_t frame_bytes =
      static_cast<std::size_t>(std::max(0, width)) *
      static_cast<std::size_t>(std::max(0, height)) * 3;
  std::error_code ec;
  const auto size = fs::file_size(file, ec);
  if (ec || frame_bytes == 0 || size < frame_bytes ||
      size % frame_bytes != 0) {
    *error = "the decoded video is no whole number of " +
             std::to_string(width) + "x" + std::to_string(height) +
             " RGB frames";
    return false;
  }
  std::ifstream in(file, std::ios::binary);
  in.seekg(static_cast<std::streamoff>(size - frame_bytes));
  frame->resize(frame_bytes);
  if (!in.read(reinterpret_cast<char*>(frame->data()),
               static_cast<std::streamsize>(frame_bytes))) {
    *error = "cannot read the last frame";
    return false;
  }
  return true;
}

bool CorrectRawFile(const fs::path& file, int width, int height,
                    const std::vector<std::uint8_t>& reference,
                    const Options& options, Stats* stats,
                    std::vector<std::uint8_t>* last_frame,
                    std::string* error) {
  const std::size_t frame_bytes =
      static_cast<std::size_t>(std::max(0, width)) *
      static_cast<std::size_t>(std::max(0, height)) * 3;
  if (frame_bytes == 0 || reference.size() != frame_bytes) {
    *error = "the start frame is not " + std::to_string(width) + "x" +
             std::to_string(height) + " RGB";
    return false;
  }
  std::error_code ec;
  const auto size = fs::file_size(file, ec);
  if (ec || size < frame_bytes || size % frame_bytes != 0) {
    *error = "the decoded video is no whole number of " +
             std::to_string(width) + "x" + std::to_string(height) +
             " RGB frames";
    return false;
  }
  const std::size_t frames = static_cast<std::size_t>(size / frame_bytes);
  std::fstream io(file, std::ios::in | std::ios::out | std::ios::binary);
  if (!io) {
    *error = "cannot open the decoded video";
    return false;
  }
  Corrector corrector(options, width, height, reference);
  std::vector<std::uint8_t> frame(frame_bytes);
  for (std::size_t i = 0; i < frames; ++i) {
    const auto at = static_cast<std::streamoff>(i * frame_bytes);
    io.seekg(at);
    if (!io.read(reinterpret_cast<char*>(frame.data()),
                 static_cast<std::streamsize>(frame_bytes))) {
      *error = "cannot read frame " + std::to_string(i);
      return false;
    }
    corrector.Process(frame.data());
    io.seekp(at);
    if (!io.write(reinterpret_cast<const char*>(frame.data()),
                  static_cast<std::streamsize>(frame_bytes))) {
      *error = "cannot write frame " + std::to_string(i);
      return false;
    }
  }
  io.flush();
  if (!io) {
    *error = "cannot write the decoded video";
    return false;
  }
  if (stats != nullptr)
    *stats = corrector.stats();
  if (last_frame != nullptr)
    *last_frame = std::move(frame);
  return true;
}

}  // namespace gufo::router::seam
