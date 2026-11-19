#ifndef GUFO_CLI_SERVE_ROUTER_VISION_FINGERS_HPP_
#define GUFO_CLI_SERVE_ROUTER_VISION_FINGERS_HPP_

// The "hand_fingers" task of POST /v1/vision/detect: the film direction's
// hand check. It is an anatomy check, not a finger count: "Ist diese Hand so
// moeglich?" A thumb hidden behind the other fingers, fingers behind an
// object or outside the frame are no error; only clearly impossible things
// are reported. The boxes of the "hand" detector (same request, same
// thresholds; the hand task then runs and is answered even when not listed)
// are cut out with a margin, and a vision chat model judges each cut-out
// through the router's own /v1/chat/completions (gate and Dispatch, under
// the caller's own Authorization / Cookie, so the caller needs the text area
// and the chat counts like any other). The hands of one image are asked one
// after the other, never at the same time. Options in
// "hand_fingers_options": {...}:
//   model            ("qwen3.8-27b")  the chat model asked
//   max_hands        (6, 1-16)        hands asked, in score order; the rest
//                                     are answered with "fehler"
//   thinking         (true)           chat_template_kwargs.enable_thinking;
//                                     without it the 27B missed a
//                                     see-through ghost arm and saw six
//                                     fingers on a good hand (prompt test
//                                     2.2.6); with it ~40-170 s per hand
//   timeout_seconds  (300, 5-600)     per hand, counted from the first
//                                     generated token (thinking included,
//                                     not the role chunk sent while the
//                                     chat still waits in the worker's
//                                     queue); then "fehler"
//   wait_seconds     (600, 0-3600)    per hand: how long to wait for the
//                                     model before it starts (queue, load,
//                                     503 memory_busy is retried after its
//                                     Retry-After); then "fehler", and the
//                                     later hands are not asked
//   margin           (0.25, 0-1)      added on every side: margin * the box
//                                     side (width left/right, height
//                                     top/bottom), clipped to the image
//   max_tokens       (4096, 32-8192)  thinking included
// The cut-out is enlarged (Pillow LANCZOS) to at least kFingerMinSide on its
// short side, shrunk to at most kFingerMaxSide on its long side, and sent as
// a PNG data URL, streamed, with response_format json_schema (strict)
// {plausibel: bool, sicher: bool, fingers_sichtbar: int 0-10,
//  verdeckt: [kFingerNames], befunde: [{art: kBefundArten, stelle: string}],
//  befund: string}.
// Answer detections.hand_fingers: one entry per hand box, in score order,
//   {"box": [x0, y0, x1, y1], "score", "crop": [x0, y0, x1, y1],
//    "plausibel": bool, "sicher": bool, "fingers_sichtbar": n,
//    "verdeckt": [...], "befunde": [{"art", "stelle"}], "befund": "...",
//    "seconds": s, "waited_seconds": s}
// or, when that one chat failed (refused, timed out, no JSON),
//   {"box", "score", "crop", "fehler": "...", "seconds": s}
// (never a 500 for the whole request). seconds is the whole time of the
// hand, waited_seconds the part before the first token. models.hand_fingers
// names the model. The CPU turn is released before the chats: they run on
// the GPU and may wait minutes when the model must be loaded first.
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
/// verdeckt: the fingers that can be hidden.
inline constexpr std::array<const char*, 5> kFingerNames = {
    "daumen", "zeigefinger", "mittelfinger", "ringfinger", "kleiner_finger"};
/// befunde[].art: the impossible things a hand is checked for.
inline constexpr std::array<const char*, 12> kBefundArten = {
    "daumen_falsche_seite",        "daumen_aus_handflaeche",
    "finger_verschmolzen",         "finger_verknotet",
    "zu_viele_finger",             "zu_wenige_finger_unerklaert",
    "gelenk_falsche_richtung",     "hand_passt_nicht_zum_arm",
    "zweite_hand_am_arm",          "arm_ohne_hand",
    "handflaeche_widerspricht_arm", "sonstiges"};

struct HandFingersRequest {
  bool enabled{false};
  std::string model{"qwen3.8-27b"};
  int max_hands{6};
  bool thinking{true};
  int timeout_seconds{300};
  int wait_seconds{600};
  double margin{0.25};
  int max_tokens{4096};
};

/// The "hand_fingers_options" member; throws std::invalid_argument for the
/// 400 answer (also for options without the task requested).
HandFingersRequest ParseHandFingers(const json::Value& body, bool requested);

/// One hand box of the detector (integer pixels, score as answered).
struct HandBox {
  std::array<int, 4> box{};
  double score{0};
};

/// One impossible thing the model saw (befunde[]).
struct HandFinding {
  std::string art;     // one of kBefundArten
  std::string stelle;  // where, in the model's words
};

/// The model's verdict on one cut-out.
struct HandVerdict {
  bool plausibel{true};
  bool sicher{false};
  int fingers_sichtbar{0};
  std::vector<std::string> verdeckt;  // kFingerNames
  std::vector<HandFinding> befunde;
  std::string befund;
};

/// The cut-out of one box: margin * width added left and right, margin *
/// height above and below, rounded outwards, clipped to the image.
std::array<int, 4> FingerCropBox(const std::array<int, 4>& box, double margin, int width,
                                 int height);

/// The /v1/chat/completions body for one cut-out (a data URL).
json::Value FingerChatBody(const HandFingersRequest& options, const std::string& data_url);

/// Reads the verdict out of a /v1/chat/completions answer body, plain JSON
/// or a stream ("data: {...}" chunks, the delta contents joined): the
/// message content as JSON (a code fence, a <think> block or text around
/// the object are tolerated) with every field of the schema. Throws
/// std::runtime_error with the reason.
HandVerdict ParseFingerAnswer(std::string_view chat_body);

/// Whether the stream text `body`, from byte `*scanned` on, holds a whole
/// "data:" line that shows the generation running: a non-empty delta
/// content, reasoning_content or tool_calls, a finish_reason or an error.
/// The role chunk ({"delta": {"role": "assistant"}}) and prompt_progress
/// chunks do not count: a gufo worker sends the role chunk already when the
/// request has waited stream_start_delay (5 s) in its queue behind another
/// chat, long before the model works on it. `*scanned` moves past the whole
/// lines looked at (an unfinished last line is looked at again later).
bool GenerationStarted(std::string_view body, std::size_t* scanned);

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
