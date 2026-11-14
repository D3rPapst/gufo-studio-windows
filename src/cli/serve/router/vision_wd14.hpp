#ifndef GUFO_CLI_SERVE_ROUTER_VISION_WD14_HPP_
#define GUFO_CLI_SERVE_ROUTER_VISION_WD14_HPP_

// The "wd14_tags" task of POST /v1/vision/detect: Danbooru tags of the
// WD14 SwinV2 v3 tagger as dghs-imgutils get_wd14_tags returns them
// (gufo_ccip Wd14Tagger, CPU). Options in "wd14_tags": {...}:
//   general_threshold   (0.35)  general tags with p > threshold
//   character_threshold (0.85)  character tags with p > threshold
//   regions: [[x0, y0, x1, y1], ...]  tag these boxes of the image instead
//            of the whole image, or "person" (also use_detections:
//            "person"): the person boxes of the same request (detect_person,
//            on the brightened copy with brighten_dark); the person task is
//            then run and answered even when not listed; at most
//            kMaxWd14Regions boxes (each is one SwinV2 run), more: 400
//   margin  (0.08)  added on every side of a region: margin * max(w, h),
//           clipped to the image, Pillow's rounding (the film pipeline's
//           erkennung.py cut-outs); the pixels come from the original
//   images: [...]   further pictures (data URL, base64, https, Ablage path),
//           each tagged as a whole (the pipeline's reference pictures)
// Answer detections.wd14_tags: one entry per tagged picture,
//   {"source": "image" | "region" | "person" | "images", "index": n,
//    "box": [x0, y0, x1, y1] (the cut-out, for region/person),
//    "width", "height" (of the tagged picture),
//    "rating": {tag: p}, "general": {tag: p}, "character": {tag: p}}
// p is the float32 probability, unrounded; tags in selected_tags.csv order.
// Written with AI assistance (Claude Code).

#include <array>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "ccip/wd14.hpp"
#include "src/core/json.hpp"

namespace gufo::router::vision {

inline constexpr const char* kWd14Task = "wd14_tags";
inline constexpr const char* kWd14Repo = "deepghs/wd14_tagger_with_embeddings";
inline constexpr const char* kWd14Model = "SmilingWolf/wd-swinv2-tagger-v3";
/// Explicit regions per request (each is one tagger run on the CPU queue).
inline constexpr std::size_t kWd14MaxRegions = 64;

struct Wd14Request {
  bool enabled{false};
  double general_threshold{models::ccip::kWd14GeneralThreshold};
  double character_threshold{models::ccip::kWd14CharacterThreshold};
  bool persons{false};                      // regions = person detections
  std::vector<std::array<double, 4>> boxes;  // explicit regions
  double margin{0.08};
  std::vector<std::string> images;  // further pictures, tagged whole
};

/// Reads the "wd14_tags" options of `body` (an object; absent: defaults).
/// Throws std::invalid_argument for the 400 answer. Options without the
/// task requested are refused.
Wd14Request ParseWd14(const json::Value& body, bool requested);

/// One answer entry for the tags of one picture.
json::Value Wd14Entry(const models::ccip::Wd14Tags& tags, const std::string& source, int index,
                      const std::array<int, 4>* box, int width, int height);

/// models.wd14_tags of the answer.
json::Value Wd14ModelInfo(const Wd14Request& request);

/// Tags `image` as the request asks (whole, explicit regions or the
/// `persons` boxes) and every picture of `extra`; the entries in order.
/// Throws std::invalid_argument when a region lies outside the image and
/// Wd14Cancelled when `cancelled` says so between two pictures.
struct Wd14Cancelled {};
json::Value RunWd14(const models::ccip::Wd14Tagger& net, const models::ccip::RgbImage& image,
                    const Wd14Request& request,
                    const std::vector<std::array<int, 4>>& persons,
                    const std::vector<models::ccip::RgbImage>& extra,
                    const std::function<bool()>& cancelled = {});

}  // namespace gufo::router::vision

#endif  // GUFO_CLI_SERVE_ROUTER_VISION_WD14_HPP_
