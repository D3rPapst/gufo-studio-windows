// Unit tests for the seam smoothing of chained videos (2.2.16,
// src/cli/serve/router/video_seam.hpp): option parsing, the seam measure,
// the colour transfer, the residual blend and the raw-file pass. CPU only,
// no ffmpeg.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "src/cli/serve/router/video_seam.hpp"
#include "src/core/json.hpp"

namespace seam = gufo::router::seam;
namespace json = gufo::json;
namespace fs = std::filesystem;

namespace {

constexpr int kW = 16;
constexpr int kH = 8;
constexpr std::size_t kBytes = kW * kH * 3;

bool Near(double a, double b, double tolerance) {
  return std::fabs(a - b) <= tolerance;
}

/// A frame with a soft pattern that moves by `shift` pixels to the right.
std::vector<std::uint8_t> Pattern(int shift) {
  std::vector<std::uint8_t> frame(kBytes);
  for (int y = 0; y < kH; ++y)
    for (int x = 0; x < kW; ++x)
      for (int c = 0; c < 3; ++c)
        frame[(y * kW + x) * 3 + c] = static_cast<std::uint8_t>(
            40 + 7 * ((x + shift) % kW) + 5 * y + 20 * c);
  return frame;
}

std::vector<std::uint8_t> Offset(std::vector<std::uint8_t> frame, int add) {
  for (auto& byte : frame)
    byte = static_cast<std::uint8_t>(std::clamp(byte + add, 0, 255));
  return frame;
}

seam::Options Parse(const std::string& text, bool ok = true) {
  seam::Options options;
  options.blend_frames = 99;  // must be reset
  std::string error;
  const json::Value value = json::parse(text);
  const bool parsed =
      seam::ParseOptions(&value, "extra_data.gufo_seam", &options, &error);
  if (parsed != ok)
    std::cerr << "parse " << text << ": " << error << "\n";
  assert(parsed == ok);
  if (!ok)
    assert(error.find("extra_data.gufo_seam") != std::string::npos);
  return options;
}

void TestParse() {
  seam::Options off;
  std::string error;
  assert(seam::ParseOptions(nullptr, "x", &off, &error));
  assert(!off.Enabled());
  assert(!Parse("null").Enabled());
  assert(!Parse("false").Enabled());
  const seam::Options both = Parse("true");
  assert(both.color_match && both.blend_frames == seam::kDefaultBlendFrames);
  const seam::Options color = Parse(R"({"color_match": true})");
  assert(color.color_match && color.blend_frames == 0 && color.Enabled());
  const seam::Options blend = Parse(R"({"blend_frames": 4})");
  assert(!blend.color_match && blend.blend_frames == 4 && blend.Enabled());
  assert(!Parse(R"({"color_match": false, "blend_frames": 0})").Enabled());
  assert(Parse(R"({"blend_frames": 32})").blend_frames == 32);
  Parse(R"({"blend_frames": 33})", false);
  Parse(R"({"blend_frames": -1})", false);
  Parse(R"({"blend_frames": 2.5})", false);
  Parse(R"({"blend_frames": "8"})", false);
  Parse(R"({"color_match": 1})", false);
  Parse(R"({"overlap": 4})", false);
  Parse(R"("on")", false);
  Parse("8", false);
  const json::Value echoed = seam::OptionsJson(both);
  assert(echoed.find("color_match")->as_bool());
  assert(echoed.member_double("blend_frames") == seam::kDefaultBlendFrames);
}

void TestMeasure() {
  const std::uint8_t a[6] = {0, 10, 20, 30, 40, 255};
  const std::uint8_t b[6] = {3, 7, 20, 30, 50, 0};
  // |diffs| = 3 3 0 0 10 255 -> 271 / 6
  assert(Near(seam::MeanAbsDiff(a, b, 6), 271.0 / 6.0, 1e-9));
  assert(seam::MeanAbsDiff(a, b, 0) == 0);
  assert(seam::Median({}) == 0);
  assert(seam::Median({5, 1, 3}) == 3);
  assert(seam::Median({4, 1, 3, 2}) == 3);
}

void TestColorTransform() {
  const std::vector<std::uint8_t> reference = Pattern(0);
  // The model's frame: lower contrast and brighter.
  std::vector<std::uint8_t> frame(kBytes);
  for (std::size_t i = 0; i < kBytes; ++i)
    frame[i] = static_cast<std::uint8_t>(std::lround(reference[i] * 0.9 + 14));
  const seam::ColorTransform t =
      seam::FitColorTransform(reference.data(), frame.data(), kW * kH);
  for (int c = 0; c < 3; ++c) {
    assert(Near(t.gain[c], 1 / 0.9, 0.02));
    assert(Near(t.offset[c], -14 / 0.9, 1.5));
  }
  seam::ApplyColorTransform(t, frame.data(), kW * kH);
  assert(seam::MeanAbsDiff(reference.data(), frame.data(), kBytes) < 0.75);

  // The gain stays in 0.75 .. 4/3; a flat channel keeps gain 1.
  std::vector<std::uint8_t> flat(kBytes, 100);
  const seam::ColorTransform f =
      seam::FitColorTransform(reference.data(), flat.data(), kW * kH);
  for (int c = 0; c < 3; ++c)
    assert(f.gain[c] == 1.0f);
  std::vector<std::uint8_t> wild(kBytes);
  for (std::size_t i = 0; i < kBytes; ++i)
    wild[i] = static_cast<std::uint8_t>((i % 2) ? 255 : 0);
  const seam::ColorTransform w =
      seam::FitColorTransform(reference.data(), wild.data(), kW * kH);
  for (int c = 0; c < 3; ++c)
    assert(Near(w.gain[c], 0.75, 1e-6));
}

/// A part of `count` frames that continues the reference (moving pattern)
/// with a constant brightness error `error`.
std::vector<std::vector<std::uint8_t>> Part(int count, int error) {
  std::vector<std::vector<std::uint8_t>> frames;
  for (int k = 0; k < count; ++k)
    frames.push_back(Offset(Pattern(k), error));
  return frames;
}

void TestOffKeepsFramesAndMeasures() {
  const std::vector<std::uint8_t> reference = Pattern(0);
  auto frames = Part(20, 9);
  const auto original = frames;
  seam::Corrector corrector(seam::Options{}, kW, kH, reference);
  for (auto& frame : frames)
    corrector.Process(frame.data());
  assert(frames == original);
  const seam::Stats& s = corrector.stats();
  assert(s.frames == 20);
  assert(Near(s.drift_before, 9, 1e-9));
  assert(s.drift_after == s.drift_before);
  assert(s.seam1_before > s.drift_before);  // drift + one frame of motion
  assert(s.seam1_after == s.seam1_before);
  // typical = median neighbour change of frames 1..17 (motion only).
  std::vector<double> pairs;
  for (int k = 1; k < 17; ++k)
    pairs.push_back(seam::MeanAbsDiff(original[k].data(),
                                      original[k + 1].data(), kBytes));
  assert(Near(s.typical, seam::Median(pairs), 1e-9));
  assert(s.typical > 0);
}

void TestColorMatchRemovesConstantError() {
  const std::vector<std::uint8_t> reference = Pattern(0);
  auto frames = Part(12, 12);
  seam::Options options;
  options.color_match = true;
  seam::Corrector corrector(options, kW, kH, reference);
  for (auto& frame : frames)
    corrector.Process(frame.data());
  const seam::Stats& s = corrector.stats();
  assert(Near(s.drift_before, 12, 1e-9));
  assert(s.drift_after < 0.6);
  // Every frame of the part is corrected the same way, not only the first.
  for (int k = 0; k < 12; ++k)
    assert(seam::MeanAbsDiff(frames[k].data(), Pattern(k).data(), kBytes) <
           0.6);
  assert(s.seam1_after < s.seam1_before);
}

void TestBlendFadesResidual() {
  const std::vector<std::uint8_t> reference = Pattern(0);
  // Frame 0 = reference + a per-pixel error (not a colour shift).
  std::vector<std::vector<std::uint8_t>> frames;
  for (int k = 0; k < 10; ++k) {
    std::vector<std::uint8_t> frame = reference;
    for (std::size_t i = 0; i < kBytes; ++i)
      frame[i] = static_cast<std::uint8_t>(frame[i] + ((i % 3 == 0) ? 8 : 0));
    frames.push_back(frame);
  }
  const auto original = frames;
  seam::Options options;
  options.blend_frames = 4;
  seam::Corrector corrector(options, kW, kH, reference);
  for (auto& frame : frames)
    corrector.Process(frame.data());
  // Frame 0 becomes the start frame exactly; frame k carries (4-k)/4 of the
  // residual; from frame 4 on the part is as the model made it.
  assert(frames[0] == reference);
  assert(corrector.stats().drift_after == 0);
  for (std::size_t i = 0; i < kBytes; i += 3) {
    assert(frames[1][i] == reference[i] + 2);
    assert(frames[2][i] == reference[i] + 4);
    assert(frames[3][i] == reference[i] + 6);
  }
  for (int k = 4; k < 10; ++k)
    assert(frames[k] == original[k]);
  assert(Near(corrector.stats().seam1_after, 2.0 / 3.0, 1e-9));
}

void TestBothClampAndChain() {
  // Colour match first, then the blend on what is left: frame 0 equals the
  // start frame; values stay in 0..255.
  std::vector<std::uint8_t> reference(kBytes);
  for (std::size_t i = 0; i < kBytes; ++i)
    reference[i] = static_cast<std::uint8_t>((i * 37) % 256);
  std::vector<std::vector<std::uint8_t>> frames;
  for (int k = 0; k < 6; ++k)
    frames.push_back(Offset(reference, 30 + k));
  seam::Options options;
  options.color_match = true;
  options.blend_frames = 2;
  seam::Corrector corrector(options, kW, kH, reference);
  for (auto& frame : frames)
    corrector.Process(frame.data());
  assert(frames[0] == reference);
  assert(corrector.stats().drift_before > 20);
  assert(corrector.stats().drift_after == 0);
}

void WriteFile(const fs::path& path,
               const std::vector<std::vector<std::uint8_t>>& frames) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  for (const auto& frame : frames)
    out.write(reinterpret_cast<const char*>(frame.data()),
              static_cast<std::streamsize>(frame.size()));
}

std::vector<std::uint8_t> ReadFile(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void TestRawFile() {
  const fs::path dir = fs::temp_directory_path() /
                       ("gufo-seam-test-" + std::to_string(std::rand()));
  fs::create_directories(dir);
  const std::vector<std::uint8_t> reference = Pattern(0);
  const auto frames = Part(10, 10);
  const fs::path raw = dir / "part.rgb";
  WriteFile(raw, frames);

  // The same result as the corrector on frames in memory.
  seam::Options options;
  options.color_match = true;
  options.blend_frames = 3;
  auto expected = frames;
  seam::Corrector corrector(options, kW, kH, reference);
  for (auto& frame : expected)
    corrector.Process(frame.data());

  seam::Stats stats;
  std::vector<std::uint8_t> last;
  std::string error;
  assert(seam::CorrectRawFile(raw, kW, kH, reference, options, &stats, &last,
                              &error));
  std::vector<std::uint8_t> joined;
  for (const auto& frame : expected)
    joined.insert(joined.end(), frame.begin(), frame.end());
  const std::vector<std::uint8_t> written = ReadFile(raw);
  assert(written == joined);

  assert(last == expected.back());
  assert(stats.frames == 10);
  assert(stats.drift_before == corrector.stats().drift_before);

  std::vector<std::uint8_t> tail;
  assert(seam::LastRawFrame(raw, kW, kH, &tail, &error));
  assert(tail == expected.back());

  const fs::path still = dir / "start.rgb";
  WriteFile(still, {reference});
  std::vector<std::uint8_t> image;
  assert(seam::ReadRawImage(still, kW, kH, &image, &error));
  assert(image == reference);
  assert(!seam::ReadRawImage(still, kW + 1, kH, &image, &error));
  assert(error.find("17x8") != std::string::npos);

  // Not a whole number of frames, or a start frame of another size.
  const fs::path broken = dir / "broken.rgb";
  {
    std::ofstream out(broken, std::ios::binary);
    out.write(reinterpret_cast<const char*>(reference.data()), 100);
  }
  error.clear();
  assert(!seam::CorrectRawFile(broken, kW, kH, reference, options, nullptr,
                               nullptr, &error));
  assert(error.find("whole number") != std::string::npos);
  assert(!seam::LastRawFrame(broken, kW, kH, &tail, &error));
  error.clear();
  assert(!seam::CorrectRawFile(raw, kW, kH, std::vector<std::uint8_t>(5),
                               options, nullptr, nullptr, &error));
  assert(error.find("start frame") != std::string::npos);

  std::error_code ec;
  fs::remove_all(dir, ec);
}

void TestStatsText() {
  seam::Stats stats;
  stats.frames = 81;
  stats.drift_before = 3.7;
  stats.seam1_before = 4.1;
  stats.seam1_after = 1.62;
  stats.typical = 1.6;
  const std::string text = seam::FormatStats(stats);
  assert(text.find("frames=81 drift=3.70->0.00 seam1=4.10->1.62 "
                   "typical=1.60") == 0);
  const json::Value value = seam::StatsJson(stats);
  assert(value.member_double("drift_before") == 3.7);
  assert(value.member_double("seam1_after") == 1.62);
  assert(value.find("gain")->size() == 3);
}

}  // namespace

int main() {
  TestParse();
  TestMeasure();
  TestColorTransform();
  TestOffKeepsFramesAndMeasures();
  TestColorMatchRemovesConstantError();
  TestBlendFadesResidual();
  TestBothClampAndChain();
  TestRawFile();
  TestStatsText();
  std::cout << "router_video_seam_test: ok\n";
  return 0;
}
