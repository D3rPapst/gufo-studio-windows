// The "hand_fingers" task of POST /v1/vision/detect (see vision_fingers.hpp).
// Written with AI assistance (Claude Code).
#include "src/cli/serve/router/vision_fingers.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <utility>

#include "src/cli/serve/router/vision_chat.hpp"

namespace gufo::router::vision {
namespace {

using server::HttpRequest;
using server::HttpResponse;

/// The anatomy question for one cut-out (prompt-tested on the film
/// direction's hand set, see docs/internal/server.md).
constexpr const char* kFingerPrompt =
    "Ist diese Hand so möglich? This is a cut-out of an AI-generated image around a hand. "
    "Check the anatomy, do not just count fingers. First look at the whole cut-out, not "
    "only the fingers: is there a second, faint, see-through figure, arm or hand (a ghost "
    "double: wall, desk or background show through it, or legs show through a desk), or "
    "an extra arm or hand that belongs to no body? That is impossible: "
    "zweite_hand_am_arm. Rules: fingers_sichtbar = the fingers you can actually see "
    "(thumb included). Hidden or cut-off fingers are NOT an error: a thumb behind the "
    "other fingers, under the palm or on the far side of the hand, fingers behind an "
    "object, curled into a fist or outside the cut-out are normal anatomy; list them in "
    "verdeckt. A hand lying flat or gripping often shows only four fingers: that is "
    "normal. befunde lists only things that are clearly impossible for a real human body "
    "and clearly visible, each with stelle (where, in German): daumen_falsche_seite "
    "(thumb on the wrong side for this hand and view), daumen_aus_handflaeche (thumb "
    "growing out of the middle of the palm), finger_verschmolzen (fingers merged into one "
    "shape), finger_verknotet (fingers tangled into each other), zu_viele_finger (more "
    "than five fingers on one hand), zu_wenige_finger_unerklaert (fewer than five and the "
    "missing ones cannot be hidden), gelenk_falsche_richtung (a finger bent backwards at "
    "a joint), hand_passt_nicht_zum_arm (the hand does not fit the arm or wrist), "
    "zweite_hand_am_arm (an extra hand or arm, also a see-through ghost arm or ghost "
    "figure), arm_ohne_hand (an arm that ends without a hand, not cut off by the frame or "
    "an object), handflaeche_widerspricht_arm (palm shown where the arm position needs "
    "the back of the hand), sonstiges. Shadows, dark lighting, drawing style and wrinkles "
    "are no finding. In doubt report nothing: a normal hand has befunde = []. plausibel = "
    "false only when befunde is not empty. sicher = false when fingers are hidden, the "
    "hand is a fist, overlaps another hand, is blurred, very small or you are unsure; "
    "then plausibel stays true unless something impossible is clearly visible. befund: "
    "one short German sentence, for example \"Hand normal, Daumen verdeckt\" or "
    "\"durchscheinender zweiter Arm rechts\". Answer only with the JSON object.";

int WholeNumber(const json::Value& value, const std::string& what, int low, int high) {
  if (!value.is_number() || value.as_double() != std::floor(value.as_double()) ||
      value.as_double() < low || value.as_double() > high)
    throw std::invalid_argument(what + " must be a whole number from " + std::to_string(low) +
                                " to " + std::to_string(high));
  return static_cast<int>(value.as_double());
}

json::Value BoxValue(const std::array<int, 4>& box) {
  json::Value list = json::Value::array();
  for (int v : box)
    list.push_back(v);
  return list;
}

/// The cut-out as a PNG data URL, at least kFingerMinSide on the short side
/// and at most kFingerMaxSide on the long side.
std::string CropDataUrl(const models::ccip::RgbImage& frame, const std::array<int, 4>& crop) {
  return ImageDataUrl(models::ccip::Crop(frame, crop[0], crop[1], crop[2], crop[3]),
                      kFingerMinSide, kFingerMaxSide);
}

template <std::size_t N>
bool OneOf(const std::string& name, const std::array<const char*, N>& names) {
  return std::any_of(names.begin(), names.end(),
                     [&name](const char* known) { return name == known; });
}

}  // namespace

HandFingersRequest ParseHandFingers(const json::Value& body, bool requested) {
  HandFingersRequest request;
  const json::Value* options = body.find(kHandFingersOptions);
  const bool given = options != nullptr && !options->is_null();
  if (!requested) {
    if (given)
      throw std::invalid_argument(
          "hand_fingers_options given, but the task hand_fingers is not requested");
    return request;
  }
  request.enabled = true;
  if (!given)
    return request;
  if (!options->is_object())
    throw std::invalid_argument("hand_fingers_options must be an object");
  for (const auto& [key, value] : options->members()) {
    const std::string what = std::string(kHandFingersOptions) + "." + key;
    if (key == "model") {
      if (!value.is_string() || value.str().empty() || value.str().size() > 128)
        throw std::invalid_argument(what + " must be a model id");
      request.model = value.str();
    } else if (key == "max_hands") {
      request.max_hands = WholeNumber(value, what, 1, kMaxFingerHands);
    } else if (key == "thinking") {
      if (!value.is_bool())
        throw std::invalid_argument(what + " must be true or false");
      request.thinking = value.as_bool();
    } else if (key == "timeout_seconds") {
      request.timeout_seconds = WholeNumber(value, what, 5, 600);
    } else if (key == "wait_seconds") {
      request.wait_seconds = WholeNumber(value, what, 0, 3600);
    } else if (key == "margin") {
      if (!value.is_number() || !(value.as_double() >= 0.0 && value.as_double() <= 1.0))
        throw std::invalid_argument(what + " must be a number in [0, 1]");
      request.margin = value.as_double();
    } else if (key == "max_tokens") {
      request.max_tokens = WholeNumber(value, what, 32, 8192);
    } else {
      throw std::invalid_argument(
          what +
          ": use model, max_hands, thinking, timeout_seconds, wait_seconds, margin or max_tokens");
    }
  }
  return request;
}

std::array<int, 4> FingerCropBox(const std::array<int, 4>& box, double margin, int width,
                                 int height) {
  const double w = box[2] - box[0];
  const double h = box[3] - box[1];
  std::array<int, 4> out{
      static_cast<int>(std::floor(box[0] - margin * w)),
      static_cast<int>(std::floor(box[1] - margin * h)),
      static_cast<int>(std::ceil(box[2] + margin * w)),
      static_cast<int>(std::ceil(box[3] + margin * h)),
  };
  out[0] = std::clamp(out[0], 0, width);
  out[1] = std::clamp(out[1], 0, height);
  out[2] = std::clamp(out[2], 0, width);
  out[3] = std::clamp(out[3], 0, height);
  return out;
}

json::Value FingerChatBody(const HandFingersRequest& options, const std::string& data_url) {
  json::Value content = json::Value::array();
  content.push_back(ImagePart(data_url));
  content.push_back(TextPart(kFingerPrompt));

  // The order of the properties is the order the model writes them: what
  // it sees first, the verdict after it.
  json::Value fingers = json::Value::object();
  fingers["type"] = "integer";
  fingers["minimum"] = 0;
  fingers["maximum"] = 10;
  json::Value verdeckt = json::Value::object();
  verdeckt["type"] = "array";
  verdeckt["items"] = EnumSchema(kFingerNames);
  json::Value art = EnumSchema(kBefundArten);
  json::Value finding_properties = json::Value::object();
  finding_properties["art"] = std::move(art);
  finding_properties["stelle"] = TypeSchema("string");
  json::Value finding = ObjectSchema(std::move(finding_properties));
  json::Value befunde = json::Value::object();
  befunde["type"] = "array";
  befunde["items"] = std::move(finding);
  json::Value properties = json::Value::object();
  properties["fingers_sichtbar"] = std::move(fingers);
  properties["verdeckt"] = std::move(verdeckt);
  properties["befunde"] = std::move(befunde);
  properties["plausibel"] = TypeSchema("boolean");
  properties["sicher"] = TypeSchema("boolean");
  properties["befund"] = TypeSchema("string");
  return VisionChatBody(options.model, std::move(content), "hand_fingers",
                        ObjectSchema(std::move(properties)), options.max_tokens, options.thinking);
}

HandVerdict ParseFingerAnswer(std::string_view chat_body) {
  const json::Value verdict = ChatAnswerObject(chat_body);
  const auto flag = [&verdict](const char* name) {
    const json::Value* value = verdict.find(name);
    if (value == nullptr || !value->is_bool())
      throw std::runtime_error(std::string(name) + " missing or not true/false");
    return value->as_bool();
  };
  const auto text_of = [](const json::Value* value, const std::string& name) {
    if (value == nullptr || !value->is_string())
      throw std::runtime_error(name + " missing or not a string");
    return value->str();
  };
  HandVerdict out;
  out.plausibel = flag("plausibel");
  out.sicher = flag("sicher");
  const json::Value* fingers = verdict.find("fingers_sichtbar");
  if (fingers == nullptr || !fingers->is_number() ||
      fingers->as_double() != std::floor(fingers->as_double()) || fingers->as_double() < 0 ||
      fingers->as_double() > 10)
    throw std::runtime_error("fingers_sichtbar missing or not a whole number 0-10");
  out.fingers_sichtbar = static_cast<int>(fingers->as_double());
  const json::Value* verdeckt = verdict.find("verdeckt");
  if (verdeckt == nullptr || !verdeckt->is_array())
    throw std::runtime_error("verdeckt missing or not a list");
  for (const json::Value& item : verdeckt->items()) {
    const std::string name = text_of(&item, "verdeckt[]");
    if (!OneOf(name, kFingerNames))
      throw std::runtime_error("verdeckt: unknown finger '" + name.substr(0, 40) + "'");
    if (std::find(out.verdeckt.begin(), out.verdeckt.end(), name) == out.verdeckt.end())
      out.verdeckt.push_back(name);
  }
  const json::Value* befunde = verdict.find("befunde");
  if (befunde == nullptr || !befunde->is_array())
    throw std::runtime_error("befunde missing or not a list");
  for (const json::Value& item : befunde->items()) {
    if (!item.is_object())
      throw std::runtime_error("befunde[] is no object");
    HandFinding finding;
    finding.art = text_of(item.find("art"), "befunde[].art");
    if (!OneOf(finding.art, kBefundArten))
      throw std::runtime_error("befunde[].art: unknown '" + finding.art.substr(0, 40) + "'");
    finding.stelle = text_of(item.find("stelle"), "befunde[].stelle");
    out.befunde.push_back(std::move(finding));
  }
  out.befund = text_of(verdict.find("befund"), "befund");
  return out;
}

namespace {

double Seconds(std::chrono::steady_clock::duration d) {
  return std::round(std::chrono::duration<double>(d).count() * 1000.0) / 1000.0;
}

void AddVerdict(const HandVerdict& verdict, json::Value& entry) {
  entry["plausibel"] = verdict.plausibel;
  entry["sicher"] = verdict.sicher;
  entry["fingers_sichtbar"] = verdict.fingers_sichtbar;
  json::Value verdeckt = json::Value::array();
  for (const std::string& name : verdict.verdeckt)
    verdeckt.push_back(name);
  entry["verdeckt"] = std::move(verdeckt);
  json::Value befunde = json::Value::array();
  for (const HandFinding& finding : verdict.befunde) {
    json::Value item = json::Value::object();
    item["art"] = finding.art;
    item["stelle"] = finding.stelle;
    befunde.push_back(std::move(item));
  }
  entry["befunde"] = std::move(befunde);
  entry["befund"] = verdict.befund;
}

}  // namespace

json::Value CountFingers(const HandFingersRequest& options, const models::ccip::RgbImage& image,
                         const std::vector<HandBox>& hands,
                         const std::function<HttpResponse(const HttpRequest&)>& serve,
                         const HttpRequest& outer) {
  using Clock = std::chrono::steady_clock;
  json::Value list = json::Value::array();
  // Set when a hand waited wait_seconds for the model in vain: the later
  // hands would only wait again.
  std::string model_gone;
  // One hand after the other: the hands of one image never compete for
  // the model.
  for (std::size_t i = 0; i < hands.size(); ++i) {
    const HandBox& hand = hands[i];
    const std::array<int, 4> crop =
        FingerCropBox(hand.box, options.margin, image.width, image.height);
    json::Value entry = json::Value::object();
    entry["box"] = BoxValue(hand.box);
    entry["score"] = hand.score;
    entry["crop"] = BoxValue(crop);
    const auto started = Clock::now();
    if (i >= static_cast<std::size_t>(options.max_hands)) {
      entry["fehler"] = "not checked: more than max_hands " + std::to_string(options.max_hands);
      list.push_back(std::move(entry));
      continue;
    }
    if (outer.is_cancelled && outer.is_cancelled()) {
      entry["fehler"] = "cancelled";
      list.push_back(std::move(entry));
      continue;
    }
    if (!serve) {
      entry["fehler"] = "no chat on this router";
      list.push_back(std::move(entry));
      continue;
    }
    if (!model_gone.empty()) {
      entry["fehler"] = "not checked: " + model_gone;
      list.push_back(std::move(entry));
      continue;
    }
    try {
      if (crop[2] - crop[0] < 2 || crop[3] - crop[1] < 2)
        throw std::runtime_error("hand box too small");
      const ChatLimits limits{options.model, options.wait_seconds, options.timeout_seconds};
      double waited = -1;
      std::string answer;
      try {
        answer = AskVisionChat(FingerChatBody(options, CropDataUrl(image, crop)).dump(), limits,
                               serve, outer, &model_gone, &waited);
      } catch (const std::exception&) {
        if (waited >= 0)
          entry["waited_seconds"] = waited;
        throw;
      }
      if (waited >= 0)
        entry["waited_seconds"] = waited;
      AddVerdict(ParseFingerAnswer(answer), entry);
    } catch (const std::exception& error) {
      entry["fehler"] = error.what();
    }
    entry["seconds"] = Seconds(Clock::now() - started);
    list.push_back(std::move(entry));
  }
  return list;
}

json::Value HandFingersModelInfo(const HandFingersRequest& options) {
  json::Value info = json::Value::object();
  info["model"] = options.model;
  info["detector"] = "hand";
  info["thinking"] = options.thinking;
  info["max_hands"] = options.max_hands;
  info["margin"] = std::round(options.margin * 1e6) / 1e6;
  info["timeout_seconds"] = options.timeout_seconds;
  info["wait_seconds"] = options.wait_seconds;
  return info;
}

}  // namespace gufo::router::vision
