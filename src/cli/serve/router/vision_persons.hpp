#ifndef GUFO_CLI_SERVE_ROUTER_VISION_PERSONS_HPP_
#define GUFO_CLI_SERVE_ROUTER_VISION_PERSONS_HPP_

// The "person_count" task of POST /v1/vision/detect: the film direction's
// person check. A vision chat model (vision_chat.hpp) looks at the whole
// picture, lists every person it sees and reports what is impossible about
// the persons: a ghost arm or any body part that belongs to no person
// (extra_limb), the same person twice (duplicate_person), more or fewer
// persons than expected (count_mismatch), else sonstiges. Hidden body parts
// (behind a desk, another person, cut off by the frame) are NO finding: the
// lesson of hand_fingers (thumb hidden). The "person" detector runs too
// (same request, same thresholds); its count is answered as
// detektor_personen for comparison, the model does not see it.
// Options in "person_count_options": {...}:
//   model            ("qwen3.8-27b")  the chat model asked
//   thinking         (true)           chat_template_kwargs.enable_thinking
//                                     (hand_fingers: without it the 27B
//                                     missed the see-through ghost arm)
//   expected         (none, 0-20)     persons the scene should show
//                                     ("erwartet" accepted too); without
//                                     it there is no count_mismatch
//   references       ([], up to 4)    reference pictures of the film's
//                                     characters ("referenzen" accepted):
//                                     [{"name": "Bernd", "image": "<data
//                                     URL | base64 | https URL | Ablage
//                                     path>"}] or plain image strings
//                                     (named "Figur 1", ...); the model
//                                     names each person after them
//   timeout_seconds  (300, 5-600)     counted from the first generated token
//   wait_seconds     (600, 0-3600)    until the model starts (queue, load,
//                                     503 memory_busy retried)
//   max_tokens       (4096, 32-8192)  thinking included
// The picture is sent at most kPersonMaxSide on its long side, references
// at most kReferenceMaxSide, as PNG data URLs, each reference after a text
// part "Referenz <name>:", streamed, with response_format json_schema
// (strict)
//   {personen: [{stelle: string, wer: string}],
//    befunde: [{art: kPersonBefundArten, stelle: string,
//               sicherheit: kSicherheiten}],
//    sicher: bool, befund: string}.
// Answer detections.person_count (one object for the picture):
//   {"personen": n (the listed persons), "erwartet": m (when given),
//    "detektor_personen": k, "liste": [{"stelle", "wer"}],
//    "befunde": [{"art", "stelle", "sicherheit", "quelle": "modell"|"gufo"}],
//    "sicher": bool, "befund": "...", "urteil": "ok"|"pruefen"|"fehler",
//    "seconds": s, "waited_seconds": s}
// or, when the chat failed, {"erwartet", "detektor_personen",
// "fehler": "...", "seconds"} (never a 500 for the whole request).
// gufo adds count_mismatch itself (quelle "gufo", sicherheit hoch, or
// mittel when the model is unsure) when expected is given and the listed
// persons differ from it and the model did not report one; without
// expected a count_mismatch of the model is dropped. urteil: "fehler" when
// a finding has sicherheit hoch, "pruefen" when there are only mittel /
// niedrig findings or sicher is false, else "ok".
// Written with AI assistance (Claude Code).

#include <array>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "ccip/image_ops.hpp"
#include "src/cli/serve/http_server.hpp"
#include "src/core/json.hpp"

namespace gufo::router::vision {

inline constexpr const char* kPersonCountTask = "person_count";
inline constexpr const char* kPersonCountOptions = "person_count_options";
inline constexpr int kMaxPersonReferences = 4;
inline constexpr int kMaxExpectedPersons = 20;
/// Most persons a model answer may list (more: the answer is refused).
inline constexpr int kMaxListedPersons = 32;
inline constexpr int kPersonMaxSide = 1536;
inline constexpr int kReferenceMaxSide = 768;
/// befunde[].art.
inline constexpr std::array<const char*, 4> kPersonBefundArten = {
    "extra_limb", "duplicate_person", "count_mismatch", "sonstiges"};
/// befunde[].sicherheit, surest first.
inline constexpr std::array<const char*, 3> kSicherheiten = {"hoch", "mittel", "niedrig"};

struct PersonReference {
  std::string name;
  std::string image;  // as sent: data URL, base64, https URL or Ablage path
};

struct PersonCountRequest {
  bool enabled{false};
  std::string model{"qwen3.8-27b"};
  bool thinking{true};
  // 2.2.13: 600 s / 8192 tokens, as hand_fingers.
  int timeout_seconds{600};
  int wait_seconds{600};
  int max_tokens{8192};
  int expected{-1};  // -1: not given
  std::vector<PersonReference> references;
};

/// The "person_count_options" member; throws std::invalid_argument for the
/// 400 answer (also for options without the task requested).
PersonCountRequest ParsePersonCount(const json::Value& body, bool requested);

/// One person the model listed.
struct PersonSeen {
  std::string stelle;  // where, in the model's words
  std::string wer;     // reference name or a short description
};

/// One finding (befunde[]).
struct PersonFinding {
  std::string art;         // kPersonBefundArten
  std::string stelle;
  std::string sicherheit;  // kSicherheiten
  std::string quelle{"modell"};
};

/// The model's verdict on the picture.
struct PersonVerdict {
  std::vector<PersonSeen> personen;
  std::vector<PersonFinding> befunde;
  bool sicher{false};
  std::string befund;
};

/// The question for the picture (the expected count and the reference
/// names worked in).
std::string PersonPrompt(const PersonCountRequest& options);

/// The /v1/chat/completions body: text "Szene:", the picture, a text part
/// and picture per reference (in the order of options.references), the
/// question last.
json::Value PersonChatBody(const PersonCountRequest& options, const std::string& scene_url,
                           const std::vector<std::string>& reference_urls);

/// Reads the verdict out of a /v1/chat/completions answer body (plain or
/// streamed, see ChatAnswerObject). Throws std::runtime_error.
PersonVerdict ParsePersonAnswer(std::string_view chat_body);

/// gufo's own part: count_mismatch against options.expected (added, or a
/// model's dropped without expected).
void CheckPersonCount(const PersonCountRequest& options, PersonVerdict& verdict);

/// "ok", "pruefen" or "fehler" (see the top of this file).
std::string PersonUrteil(const PersonVerdict& verdict);

/// detections.person_count for `image`: one chat through `serve`, the
/// caller's Authorization / Cookie copied from `outer`; a failed chat
/// gives "fehler". `references` are the decoded pictures of
/// options.references, in order; `detector_persons` the count of the
/// "person" detector.
json::Value CheckPersons(const PersonCountRequest& options, const models::ccip::RgbImage& image,
                         const std::vector<models::ccip::RgbImage>& references,
                         int detector_persons,
                         const std::function<server::HttpResponse(const server::HttpRequest&)>& serve,
                         const server::HttpRequest& outer);

/// models.person_count.
json::Value PersonCountModelInfo(const PersonCountRequest& options);

}  // namespace gufo::router::vision

#endif  // GUFO_CLI_SERVE_ROUTER_VISION_PERSONS_HPP_
