#ifndef GUFO_CLI_SERVE_ROUTER_VISION_FINGERS_HPP_
#define GUFO_CLI_SERVE_ROUTER_VISION_FINGERS_HPP_

// The "hand_fingers" task of POST /v1/vision/detect: the film direction's
// hand check ("Arm ohne Hand", wrong number of fingers). The boxes of the
// "hand" detector (same request, same thresholds; the hand task then runs
// and is answered even when not listed) are cut out with a margin, and a
// vision chat model counts the visible fingers in each cut-out through the
// router's own /v1/chat/completions (gate and Dispatch, under the caller's
// own Authorization / Cookie, so the caller needs the text area and the
// chat counts like any other). Options in "hand_fingers_options": {...}:
//   model            ("qwen3.8-27b")  the chat model asked
//   max_hands        (6, 1-16)        hands asked, in score order; the rest
//                                     are answered with "fehler"
//   thinking         (false)          chat_template_kwargs.enable_thinking
//   timeout_seconds  (120, 5-600)     per hand; then "fehler"
//   margin           (0.25, 0-1)      added on every side: margin * the box
//                                     side (width left/right, height
//                                     top/bottom), clipped to the image
//   max_tokens       (256, 32-4096)
// The cut-out is enlarged (Pillow LANCZOS) to at least kFingerMinSide on its
// short side, shrunk to at most kFingerMaxSide on its long side, and sent as
// a PNG data URL with response_format json_schema
// {fingers: int 0-10, sicher: bool, befund: string}.
// Answer detections.hand_fingers: one entry per hand box, in score order,
//   {"box": [x0, y0, x1, y1], "score", "crop": [x0, y0, x1, y1],
//    "fingers": n, "sicher": bool, "befund": "...", "seconds": s}
// or, when that one chat failed (refused, timed out, no JSON),
//   {"box", "score", "crop", "fehler": "...", "seconds": s}
// (never a 500 for the whole request). models.hand_fingers names the model.
// The CPU turn is released before the chats: they run on the GPU and may
// take minutes when the model must be loaded first.
// Written with AI assistance (Claude Code).

#include <array>
#include <chrono>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "ccip/image_ops.hpp"
#include "src/cli/serve/http_server.hpp"
#include "src/core/json.hpp"

namespace gufo::router::vision {

inline constexpr const char* kHandFingersTask = "hand_fingers";
inline constexpr const char* kHandFingersOptions = "hand_fingers_options";
inline constexpr int kMaxFingerHands = 16;
inline constexpr int kFingerMinSide = 224;
inline constexpr int kFingerMaxSide = 1024;

struct HandFingersRequest {
  bool enabled{false};
  std::string model{"qwen3.8-27b"};
  int max_hands{6};
  bool thinking{false};
  int timeout_seconds{120};
  double margin{0.25};
  int max_tokens{256};
};

/// The "hand_fingers_options" member; throws std::invalid_argument for the
/// 400 answer (also for options without the task requested).
HandFingersRequest ParseHandFingers(const json::Value& body, bool requested);

/// One hand box of the detector (integer pixels, score as answered).
struct HandBox {
  std::array<int, 4> box{};
  double score{0};
};

/// The model's verdict on one cut-out.
struct FingerCount {
  int fingers{0};
  bool sicher{false};
  std::string befund;
};

/// The cut-out of one box: margin * width added left and right, margin *
/// height above and below, rounded outwards, clipped to the image.
std::array<int, 4> FingerCropBox(const std::array<int, 4>& box, double margin, int width,
                                 int height);

/// The /v1/chat/completions body for one cut-out (a data URL).
json::Value FingerChatBody(const HandFingersRequest& options, const std::string& data_url);

/// Reads the verdict out of a /v1/chat/completions answer body: the
/// message content as JSON (a code fence, a <think> block or text around
/// the object are tolerated), fingers a whole number 0-10, sicher a bool,
/// befund a string. Throws std::runtime_error with the reason.
FingerCount ParseFingerAnswer(std::string_view chat_body);

/// detections.hand_fingers for `hands` (in this order) on `image`: one chat
/// per hand through `serve`, the caller's Authorization / Cookie copied
/// from `outer`; a failed chat gives that hand "fehler". Stops asking when
/// `outer` is cancelled (the rest: "fehler": "cancelled").
json::Value CountFingers(const HandFingersRequest& options, const models::ccip::RgbImage& image,
                         const std::vector<HandBox>& hands,
                         const std::function<server::HttpResponse(const server::HttpRequest&)>& serve,
                         const server::HttpRequest& outer);

/// models.hand_fingers.
json::Value HandFingersModelInfo(const HandFingersRequest& options);

}  // namespace gufo::router::vision

#endif  // GUFO_CLI_SERVE_ROUTER_VISION_FINGERS_HPP_
