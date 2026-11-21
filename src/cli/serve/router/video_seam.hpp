#ifndef GUFO_CLI_SERVE_ROUTER_VIDEO_SEAM_HPP_
#define GUFO_CLI_SERVE_ROUTER_VIDEO_SEAM_HPP_

// Seam smoothing for chained videos (2.2.16): a part that was rendered from
// a start frame (Wan 2.1 / 2.2 I2V "extend": the next part starts from a
// frame of the part before) begins close to that frame, but the model's
// first frames differ from it in colour, contrast and sharpness, which shows
// as a jump where the parts meet. Off by default (ComfyUI's ImageBatch joins
// the parts as they are); two independent tools, both on RGB24 frames:
//
//   color_match    one per-channel gain / offset for the whole part, fitted
//                  so that its first frame has the mean and standard
//                  deviation of the start frame (Reinhard transfer in RGB);
//                  keeps the part's own changes over time.
//   blend_frames N the remaining per-pixel difference between the start
//                  frame and the part's first frame (after color_match) is
//                  added back with weight (N - k) / N on frame k < N: frame 0
//                  becomes the start frame, the difference fades out over N
//                  frames instead of showing at once.
//
// The seam measure (logged before / after, also for GPU checks):
// D(a, b) = mean over all pixels and the three channels of |a - b| on the
// 0..255 scale. drift = D(start, part[0]), seam1 = D(start, part[1]) (the
// visible seam when the repeated start frame is dropped, ComfyUI
// ImageFromBatch(batch_index 1)), typical = median of D(part[k], part[k+1])
// for k = 1 .. 16 (the part's normal frame-to-frame change).
//
// No I/O but plain files here, so everything is unit-testable without
// ffmpeg (router_video_seam_test).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::router::seam {

inline constexpr int kDefaultBlendFrames = 8;
inline constexpr int kMaxBlendFrames = 32;
/// Neighbour pairs of a part used for `typical`.
inline constexpr int kTypicalPairs = 16;

struct Options {
  bool color_match{false};
  /// 0 = no blend; 1..kMaxBlendFrames.
  int blend_frames{0};
  bool Enabled() const { return color_match || blend_frames > 0; }
};

/// Reads the option value (ComfyUI extra_data.gufo_seam, the "seam" member
/// of POST /v1/videos/concat): absent / null / false = off, true = both
/// tools (color_match + kDefaultBlendFrames), or an object
/// {"color_match": bool, "blend_frames": 0..32} (missing members off).
/// False with *error (an English sentence naming `where`) on a bad value.
bool ParseOptions(const json::Value* value, const std::string& where,
                  Options* out, std::string* error);

/// The options as JSON ({"color_match": .., "blend_frames": ..}).
json::Value OptionsJson(const Options& options);

/// D(a, b): mean |a - b| over `bytes` bytes, 0..255.
double MeanAbsDiff(const std::uint8_t* a, const std::uint8_t* b,
                   std::size_t bytes);

/// Median of the values (0 for none); the upper middle for an even count.
double Median(std::vector<double> values);

/// y = x * gain + offset per channel, rounded and clamped to 0..255.
struct ColorTransform {
  float gain[3]{1, 1, 1};
  float offset[3]{0, 0, 0};
};

/// The transform that gives `frame` the per-channel mean and standard
/// deviation of `reference` (both RGB24, `pixels` pixels). The gain stays in
/// 0.75..1.333 (a flat channel keeps gain 1).
ColorTransform FitColorTransform(const std::uint8_t* reference,
                                 const std::uint8_t* frame,
                                 std::size_t pixels);

void ApplyColorTransform(const ColorTransform& transform, std::uint8_t* frame,
                         std::size_t pixels);

/// What one part's correction did (all D values 0..255).
struct Stats {
  int frames{0};
  double drift_before{0};
  double drift_after{0};
  /// D(start, part[1]); 0 with fewer than two frames.
  double seam1_before{0};
  double seam1_after{0};
  /// Median neighbour difference of the uncorrected part (k >= 1).
  double typical{0};
  ColorTransform transform;
};

/// "frames=81 drift=3.70->0.00 seam1=4.10->1.62 typical=1.60
/// gain=1.010,0.990,1.000 offset=-1.2,0.4,0.0".
std::string FormatStats(const Stats& stats);

json::Value StatsJson(const Stats& stats);

/// Corrects the frames of one part in order (frame 0 first), in place.
class Corrector {
 public:
  /// `reference` = the start frame the part was rendered from (RGB24,
  /// width x height).
  Corrector(const Options& options, int width, int height,
            std::vector<std::uint8_t> reference);

  std::size_t FrameBytes() const { return reference_.size(); }
  /// The next frame of the part.
  void Process(std::uint8_t* frame);
  const Stats& stats() const { return stats_; }

 private:
  Options options_;
  std::size_t pixels_;
  std::vector<std::uint8_t> reference_;
  /// reference - corrected frame 0, per byte (blend only).
  std::vector<std::int16_t> residual_;
  std::vector<std::uint8_t> previous_;  // uncorrected frame k-1
  std::vector<double> pairs_;
  int index_{0};
  Stats stats_;
};

/// Corrects an RGB24 raw video file (width x height frames back to back,
/// ffmpeg -f rawvideo -pix_fmt rgb24) in place. `last_frame` (optional)
/// receives its last frame after correction. False with *error when the
/// file cannot be read or written, its size is no whole number of frames,
/// or the reference has another size.
bool CorrectRawFile(const std::filesystem::path& file, int width, int height,
                    const std::vector<std::uint8_t>& reference,
                    const Options& options, Stats* stats,
                    std::vector<std::uint8_t>* last_frame, std::string* error);

/// The last frame of an RGB24 raw video file (false with *error when it
/// has no whole frame).
bool LastRawFrame(const std::filesystem::path& file, int width, int height,
                  std::vector<std::uint8_t>* frame, std::string* error);

/// Reads a whole RGB24 raw image of width x height (false with *error on
/// another size).
bool ReadRawImage(const std::filesystem::path& file, int width, int height,
                  std::vector<std::uint8_t>* image, std::string* error);

}  // namespace gufo::router::seam

#endif  // GUFO_CLI_SERVE_ROUTER_VIDEO_SEAM_HPP_
