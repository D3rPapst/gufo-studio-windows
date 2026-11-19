// The "hand_fingers" task of POST /v1/vision/detect (see vision_fingers.hpp).
// Written with AI assistance (Claude Code).
#include "src/cli/serve/router/vision_fingers.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <exception>
#include <atomic>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

#include "src/cli/serve/router/rgba_image.hpp"
#include "src/core/base64.hpp"

namespace gufo::router::vision {
namespace {

using server::HttpRequest;
using server::HttpResponse;

/// Largest chat answer read back (a verdict is a few hundred bytes).
constexpr std::size_t kChatAnswerLimit = static_cast<std::size_t>(1) << 20;

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

bool EqualsIgnoreCase(std::string_view a, std::string_view b) {
  if (a.size() != b.size())
    return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(a[i])) !=
        std::tolower(static_cast<unsigned char>(b[i])))
      return false;
  }
  return true;
}

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
  models::ccip::RgbImage cut = models::ccip::Crop(frame, crop[0], crop[1], crop[2], crop[3]);
  const int short_side = std::min(cut.width, cut.height);
  const int long_side = std::max(cut.width, cut.height);
  double scale = 1.0;
  if (short_side < kFingerMinSide)
    scale = static_cast<double>(kFingerMinSide) / short_side;
  if (long_side * scale > kFingerMaxSide)
    scale = static_cast<double>(kFingerMaxSide) / long_side;
  if (scale != 1.0) {
    const int w = std::max(1, static_cast<int>(std::lround(cut.width * scale)));
    const int h = std::max(1, static_cast<int>(std::lround(cut.height * scale)));
    cut = models::ccip::ResizeLanczos(cut, w, h);
  }
  image::RgbaImage rgba;
  rgba.width = cut.width;
  rgba.height = cut.height;
  rgba.rgba.resize(static_cast<std::size_t>(cut.width) * cut.height * 4);
  for (std::size_t i = 0, n = static_cast<std::size_t>(cut.width) * cut.height; i < n; ++i) {
    rgba.rgba[i * 4 + 0] = cut.rgb[i * 3 + 0];
    rgba.rgba[i * 4 + 1] = cut.rgb[i * 3 + 1];
    rgba.rgba[i * 4 + 2] = cut.rgb[i * 3 + 2];
    rgba.rgba[i * 4 + 3] = 255;
  }
  const std::vector<std::uint8_t> png = image::EncodePng(rgba);
  return "data:image/png;base64," +
         core::EncodeBase64(
             std::string_view(reinterpret_cast<const char*>(png.data()), png.size()));
}

/// The error text of a refused chat: the server's error.message, else the
/// start of the body.
std::string ChatError(int status, const std::string& body) {
  std::string message;
  try {
    const json::Value parsed = json::parse(body);
    if (const json::Value* error = parsed.find("error"); error != nullptr) {
      if (error->is_string())
        message = error->str();
      else if (error->is_object())
        message = error->member_str("message");
    }
  } catch (const std::exception&) {
  }
  if (message.empty())
    message = body.substr(0, 200);
  return "chat " + std::to_string(status) + (message.empty() ? "" : ": " + message);
}

json::Value TypeSchema(const char* type) {
  json::Value schema = json::Value::object();
  schema["type"] = type;
  return schema;
}

template <std::size_t N>
json::Value EnumSchema(const std::array<const char*, N>& names) {
  json::Value list = json::Value::array();
  for (const char* name : names)
    list.push_back(name);
  json::Value schema = TypeSchema("string");
  schema["enum"] = std::move(list);
  return schema;
}

/// A strict object schema: every property required, no others.
json::Value ObjectSchema(json::Value properties) {
  json::Value required = json::Value::array();
  for (const auto& [name, value] : properties.members())
    required.push_back(name);
  json::Value schema = json::Value::object();
  schema["type"] = "object";
  schema["properties"] = std::move(properties);
  schema["required"] = std::move(required);
  schema["additionalProperties"] = false;
  return schema;
}

template <std::size_t N>
bool OneOf(const std::string& name, const std::array<const char*, N>& names) {
  return std::any_of(names.begin(), names.end(),
                     [&name](const char* known) { return name == known; });
}

/// The message text of a chat answer and its finish_reason: a plain
/// chat.completion, or a stream whose "data:" chunks carry the deltas.
std::string AnswerText(std::string_view chat_body, std::string* finish_reason) {
  std::size_t first = 0;
  while (first < chat_body.size() && std::isspace(static_cast<unsigned char>(chat_body[first])))
    ++first;
  const bool stream = chat_body.substr(first, 1) != "{";
  if (!stream) {
    json::Value parsed;
    try {
      parsed = json::parse(std::string(chat_body));
    } catch (const std::exception&) {
      throw std::runtime_error("chat answer is not JSON");
    }
    const json::Value* choices = parsed.find("choices");
    if (choices == nullptr || !choices->is_array() || choices->empty())
      throw std::runtime_error("chat answer has no choices");
    *finish_reason = choices->items()[0].member_str("finish_reason");
    const json::Value* message = choices->items()[0].find("message");
    const json::Value* content = message != nullptr ? message->find("content") : nullptr;
    return content != nullptr && content->is_string() ? content->str() : std::string();
  }
  std::string text;
  bool any = false;
  std::size_t at = 0;
  while (at < chat_body.size()) {
    std::size_t end = chat_body.find('\n', at);
    if (end == std::string_view::npos)
      end = chat_body.size();
    std::string_view line = chat_body.substr(at, end - at);
    at = end + 1;
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
    if (!line.starts_with("data:"))
      continue;  // comments (": ping"), event names, blank lines
    line.remove_prefix(5);
    while (!line.empty() && line.front() == ' ')
      line.remove_prefix(1);
    if (line == "[DONE]")
      continue;
    json::Value chunk;
    try {
      chunk = json::parse(std::string(line));
    } catch (const std::exception&) {
      throw std::runtime_error("stream chunk is not JSON: " + std::string(line.substr(0, 120)));
    }
    if (const json::Value* error = chunk.find("error"); error != nullptr) {
      const std::string message = error->is_string()   ? error->str()
                                  : error->is_object() ? error->member_str("message")
                                                       : std::string();
      throw std::runtime_error("chat stream error" + (message.empty() ? "" : ": " + message));
    }
    const json::Value* choices = chunk.find("choices");
    if (choices == nullptr || !choices->is_array() || choices->empty())
      continue;  // usage-only chunk
    any = true;
    const json::Value& choice = choices->items()[0];
    if (const std::string reason = choice.member_str("finish_reason"); !reason.empty())
      *finish_reason = reason;
    const json::Value* delta = choice.find("delta");
    const json::Value* content = delta != nullptr ? delta->find("content") : nullptr;
    if (content != nullptr && content->is_string())
      text += content->str();
  }
  if (!any)
    throw std::runtime_error("chat answer has no choices");
  return text;
}

/// The JSON object in a model text: without a <think> block and code
/// fences, from the first '{' to the last '}'.
std::string ObjectText(std::string text) {
  if (const auto end = text.find("</think>"); end != std::string::npos)
    text.erase(0, end + 8);
  const auto open = text.find('{');
  const auto close = text.rfind('}');
  if (open == std::string::npos || close == std::string::npos || close < open)
    throw std::runtime_error("no JSON object in the answer: " + text.substr(0, 120));
  return text.substr(open, close - open + 1);
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
  json::Value image_url = json::Value::object();
  image_url["url"] = data_url;
  json::Value image_part = json::Value::object();
  image_part["type"] = "image_url";
  image_part["image_url"] = std::move(image_url);
  json::Value text_part = json::Value::object();
  text_part["type"] = "text";
  text_part["text"] = kFingerPrompt;
  json::Value content = json::Value::array();
  content.push_back(std::move(image_part));
  content.push_back(std::move(text_part));
  json::Value message = json::Value::object();
  message["role"] = "user";
  message["content"] = std::move(content);
  json::Value messages = json::Value::array();
  messages.push_back(std::move(message));

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
  json::Value schema = ObjectSchema(std::move(properties));
  json::Value json_schema = json::Value::object();
  json_schema["name"] = "hand_fingers";
  json_schema["strict"] = true;
  json_schema["schema"] = std::move(schema);
  json::Value format = json::Value::object();
  format["type"] = "json_schema";
  format["json_schema"] = std::move(json_schema);

  json::Value body = json::Value::object();
  body["model"] = options.model;
  body["messages"] = std::move(messages);
  body["max_tokens"] = options.max_tokens;
  body["temperature"] = 0;
  // Streamed: the first chunk marks the start of the generation, from
  // which timeout_seconds counts (CountFingers).
  body["stream"] = true;
  body["response_format"] = std::move(format);
  json::Value kwargs = json::Value::object();
  kwargs["enable_thinking"] = options.thinking;
  body["chat_template_kwargs"] = std::move(kwargs);
  return body;
}

HandVerdict ParseFingerAnswer(std::string_view chat_body) {
  std::string finish_reason;
  const std::string text = AnswerText(chat_body, &finish_reason);
  if (text.empty())
    throw std::runtime_error("chat answer has no text" +
                             (finish_reason.empty() ? std::string()
                                                    : " (finish_reason " + finish_reason + ")"));
  const std::string cut = finish_reason == "length" ? " (cut off at max_tokens)" : "";
  const std::string object_text = ObjectText(text);
  json::Value verdict;
  try {
    verdict = json::parse(object_text);
  } catch (const std::exception&) {
    throw std::runtime_error("answer is no valid JSON" + cut + ": " + text.substr(0, 120));
  }
  if (!verdict.is_object())
    throw std::runtime_error("answer is no JSON object");
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

bool GenerationStarted(std::string_view body, std::size_t* scanned) {
  while (*scanned < body.size()) {
    const std::size_t end = body.find('\n', *scanned);
    if (end == std::string_view::npos)
      return false;  // the line is not whole yet
    std::string_view line = body.substr(*scanned, end - *scanned);
    *scanned = end + 1;
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
    if (!line.starts_with("data:"))
      continue;  // ": ping", event names, blank lines
    line.remove_prefix(5);
    while (!line.empty() && line.front() == ' ')
      line.remove_prefix(1);
    if (line == "[DONE]")
      return true;
    json::Value chunk;
    try {
      chunk = json::parse(std::string(line));
    } catch (const std::exception&) {
      return true;  // not ours to judge: ParseFingerAnswer reports it
    }
    if (!chunk.is_object())
      return true;
    if (chunk.find("error") != nullptr)
      return true;
    const json::Value* choices = chunk.find("choices");
    if (choices == nullptr || !choices->is_array() || choices->empty())
      continue;  // usage-only chunk
    const json::Value& choice = choices->items()[0];
    if (!choice.member_str("finish_reason").empty())
      return true;
    const json::Value* delta = choice.find("delta");
    if (delta == nullptr || !delta->is_object())
      continue;
    for (const char* name : {"content", "reasoning_content"}) {
      const json::Value* text = delta->find(name);
      if (text != nullptr && text->is_string() && !text->str().empty())
        return true;
    }
    const json::Value* calls = delta->find("tool_calls");
    if (calls != nullptr && calls->is_array() && !calls->empty())
      return true;
    // The role chunk, prompt_progress: the chat may still be queued.
  }
  return false;
}

namespace {

/// The pause a 503 memory_busy asks for (Retry-After, else
/// error.retry_after_s; 1-30 s), nullopt for any other answer (not retried).
std::optional<int> RetryAfter(const HttpResponse& response, const std::string& body) {
  if (response.status != 503)
    return std::nullopt;
  bool memory_busy = false;
  int seconds = 0;
  try {
    const json::Value parsed = json::parse(body);
    if (const json::Value* error = parsed.find("error"); error != nullptr && error->is_object()) {
      memory_busy = error->member_str("code") == "memory_busy";
      if (const json::Value* after = error->find("retry_after_s");
          after != nullptr && after->is_number())
        seconds = static_cast<int>(after->as_double());
    }
  } catch (const std::exception&) {
  }
  if (!memory_busy)
    return std::nullopt;
  for (const auto& [name, value] : response.headers) {
    if (EqualsIgnoreCase(name, "Retry-After")) {
      try {
        seconds = std::stoi(value);
      } catch (const std::exception&) {
      }
    }
  }
  return std::clamp(seconds, 1, 30);
}

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
    // Two clocks: until the first generated token (router and worker
    // queue, load, prefill) the hand may wait wait_seconds; from it on the
    // generation has timeout_seconds.
    const auto wait_deadline = started + std::chrono::seconds(options.wait_seconds);
    const auto generation_limit = std::chrono::seconds(options.timeout_seconds);
    const auto first_chunk = std::make_shared<std::atomic<Clock::rep>>(0);
    const auto over = [wait_deadline, generation_limit, first_chunk] {
      const Clock::rep first = first_chunk->load();
      const auto now = Clock::now();
      if (first == 0)
        return now > wait_deadline;
      return now > Clock::time_point(Clock::duration(first)) + generation_limit;
    };
    try {
      if (crop[2] - crop[0] < 2 || crop[3] - crop[1] < 2)
        throw std::runtime_error("hand box too small");
      HttpRequest inner;
      inner.method = "POST";
      inner.path = "/v1/chat/completions";
      inner.body = FingerChatBody(options, CropDataUrl(image, crop)).dump();
      inner.client_id = outer.client_id;
      inner.request_id = outer.request_id;
      const auto outer_cancelled = outer.is_cancelled;
      inner.is_cancelled = [outer_cancelled, over] {
        return (outer_cancelled && outer_cancelled()) || over();
      };
      for (const auto& [name, value] : outer.headers) {
        if (EqualsIgnoreCase(name, "Authorization") || EqualsIgnoreCase(name, "Cookie"))
          inner.headers.emplace_back(name, value);
      }
      inner.headers.emplace_back("Content-Type", "application/json");
      inner.headers.emplace_back("Accept", "text/event-stream, application/json");
      HttpResponse response;
      std::string body;
      for (;;) {
        response = serve(inner);
        body.clear();
        if (response.streaming_body) {
          // The generation clock starts with the first generated token,
          // not with the first "data:" line: a gufo worker sends the role
          // chunk already after 5 s in its queue (stream_start_delay), and
          // a busy 27B then spent the whole timeout_seconds in that queue
          // (2.2.6-2.2.10: "timeout after 300 s of generation").
          std::size_t scanned = 0;
          response.streaming_body([&](std::string_view chunk) {
            if (body.size() + chunk.size() > kChatAnswerLimit)
              return false;
            body.append(chunk);
            if (first_chunk->load() == 0 && GenerationStarted(body, &scanned))
              first_chunk->store(Clock::now().time_since_epoch().count());
            return !over();
          });
        } else {
          body = std::move(response.body);
        }
        // A 503 memory_busy (another model holds the memory): wait as it
        // asks while the wait clock allows, then ask again.
        const std::optional<int> retry =
            first_chunk->load() == 0 ? RetryAfter(response, body) : std::nullopt;
        if (!retry.has_value())
          break;
        const auto resume = Clock::now() + std::chrono::seconds(*retry);
        if (resume > wait_deadline) {
          model_gone = "model " + options.model + " not free within wait_seconds " +
                       std::to_string(options.wait_seconds) + " (" +
                       ChatError(response.status, body) + ")";
          throw std::runtime_error(model_gone);
        }
        while (Clock::now() < resume && !(outer_cancelled && outer_cancelled()))
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (outer_cancelled && outer_cancelled())
          throw std::runtime_error("cancelled");
      }
      if (const Clock::rep first = first_chunk->load(); first != 0)
        entry["waited_seconds"] = Seconds(Clock::time_point(Clock::duration(first)) - started);
      if (over()) {
        if (first_chunk->load() == 0) {
          model_gone = "model " + options.model + " did not start within wait_seconds " +
                       std::to_string(options.wait_seconds);
          throw std::runtime_error(model_gone);
        }
        throw std::runtime_error("timeout after " + std::to_string(options.timeout_seconds) +
                                 " s of generation");
      }
      if (response.status < 200 || response.status >= 300)
        throw std::runtime_error(ChatError(response.status, body));
      AddVerdict(ParseFingerAnswer(body), entry);
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
