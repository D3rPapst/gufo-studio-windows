// The "hand_fingers" task of POST /v1/vision/detect (see vision_fingers.hpp).
// Written with AI assistance (Claude Code).
#include "src/cli/serve/router/vision_fingers.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>

#include "src/cli/serve/router/rgba_image.hpp"
#include "src/core/base64.hpp"

namespace gufo::router::vision {
namespace {

using server::HttpRequest;
using server::HttpResponse;

/// Largest chat answer read back (a verdict is a few hundred bytes).
constexpr std::size_t kChatAnswerLimit = static_cast<std::size_t>(1) << 20;

constexpr const char* kFingerPrompt =
    "Zähle die sichtbaren Finger dieser Hand. Count the visible fingers of this hand. "
    "The thumb counts as a finger. Count only fingers you can actually see; do not add "
    "hidden ones. If no hand is visible (for example only an arm or a sleeve), fingers = 0. "
    "sicher = false when the hand is covered, a fist, blurred, cut off at the edge or you "
    "are unsure. befund: one short German sentence, for example \"fünf Finger, normal\", "
    "\"sechs Finger\", \"Arm ohne Hand\", \"Faust, Finger nicht zählbar\". "
    "Answer only with the JSON object.";

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
    } else if (key == "margin") {
      if (!value.is_number() || !(value.as_double() >= 0.0 && value.as_double() <= 1.0))
        throw std::invalid_argument(what + " must be a number in [0, 1]");
      request.margin = value.as_double();
    } else if (key == "max_tokens") {
      request.max_tokens = WholeNumber(value, what, 32, 4096);
    } else {
      throw std::invalid_argument(
          what + ": use model, max_hands, thinking, timeout_seconds, margin or max_tokens");
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

  json::Value fingers = json::Value::object();
  fingers["type"] = "integer";
  fingers["minimum"] = 0;
  fingers["maximum"] = 10;
  json::Value sicher = json::Value::object();
  sicher["type"] = "boolean";
  json::Value befund = json::Value::object();
  befund["type"] = "string";
  json::Value properties = json::Value::object();
  properties["fingers"] = std::move(fingers);
  properties["sicher"] = std::move(sicher);
  properties["befund"] = std::move(befund);
  json::Value required = json::Value::array();
  required.push_back("fingers");
  required.push_back("sicher");
  required.push_back("befund");
  json::Value schema = json::Value::object();
  schema["type"] = "object";
  schema["properties"] = std::move(properties);
  schema["required"] = std::move(required);
  schema["additionalProperties"] = false;
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
  body["response_format"] = std::move(format);
  json::Value kwargs = json::Value::object();
  kwargs["enable_thinking"] = options.thinking;
  body["chat_template_kwargs"] = std::move(kwargs);
  return body;
}

FingerCount ParseFingerAnswer(std::string_view chat_body) {
  json::Value parsed;
  try {
    parsed = json::parse(std::string(chat_body));
  } catch (const std::exception&) {
    throw std::runtime_error("chat answer is not JSON");
  }
  const json::Value* choices = parsed.find("choices");
  if (choices == nullptr || !choices->is_array() || choices->empty())
    throw std::runtime_error("chat answer has no choices");
  const json::Value* message = choices->items()[0].find("message");
  const json::Value* content = message != nullptr ? message->find("content") : nullptr;
  if (content == nullptr || !content->is_string() || content->str().empty()) {
    const std::string reason = choices->items()[0].member_str("finish_reason");
    throw std::runtime_error("chat answer has no text" +
                             (reason.empty() ? std::string() : " (finish_reason " + reason + ")"));
  }
  const std::string object_text = ObjectText(content->str());
  json::Value verdict;
  try {
    verdict = json::parse(object_text);
  } catch (const std::exception&) {
    throw std::runtime_error("answer is no valid JSON: " + content->str().substr(0, 120));
  }
  if (!verdict.is_object())
    throw std::runtime_error("answer is no JSON object");
  const json::Value* fingers = verdict.find("fingers");
  if (fingers == nullptr || !fingers->is_number() ||
      fingers->as_double() != std::floor(fingers->as_double()) || fingers->as_double() < 0 ||
      fingers->as_double() > 10)
    throw std::runtime_error("fingers missing or not a whole number 0-10");
  const json::Value* sicher = verdict.find("sicher");
  if (sicher == nullptr || !sicher->is_bool())
    throw std::runtime_error("sicher missing or not true/false");
  const json::Value* befund = verdict.find("befund");
  if (befund == nullptr || !befund->is_string())
    throw std::runtime_error("befund missing or not a string");
  FingerCount count;
  count.fingers = static_cast<int>(fingers->as_double());
  count.sicher = sicher->as_bool();
  count.befund = befund->str();
  return count;
}

json::Value CountFingers(const HandFingersRequest& options, const models::ccip::RgbImage& image,
                         const std::vector<HandBox>& hands,
                         const std::function<HttpResponse(const HttpRequest&)>& serve,
                         const HttpRequest& outer) {
  json::Value list = json::Value::array();
  for (std::size_t i = 0; i < hands.size(); ++i) {
    const HandBox& hand = hands[i];
    const std::array<int, 4> crop =
        FingerCropBox(hand.box, options.margin, image.width, image.height);
    json::Value entry = json::Value::object();
    entry["box"] = BoxValue(hand.box);
    entry["score"] = hand.score;
    entry["crop"] = BoxValue(crop);
    const auto started = std::chrono::steady_clock::now();
    const auto finish = [&](json::Value& e) {
      e["seconds"] =
          std::round(std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                         .count() *
                     1000.0) /
          1000.0;
    };
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
    try {
      if (crop[2] - crop[0] < 2 || crop[3] - crop[1] < 2)
        throw std::runtime_error("hand box too small");
      HttpRequest inner;
      inner.method = "POST";
      inner.path = "/v1/chat/completions";
      inner.body = FingerChatBody(options, CropDataUrl(image, crop)).dump();
      inner.client_id = outer.client_id;
      inner.request_id = outer.request_id;
      const auto deadline = started + std::chrono::seconds(options.timeout_seconds);
      const auto outer_cancelled = outer.is_cancelled;
      inner.is_cancelled = [outer_cancelled, deadline] {
        return (outer_cancelled && outer_cancelled()) ||
               std::chrono::steady_clock::now() > deadline;
      };
      for (const auto& [name, value] : outer.headers) {
        if (EqualsIgnoreCase(name, "Authorization") || EqualsIgnoreCase(name, "Cookie"))
          inner.headers.emplace_back(name, value);
      }
      inner.headers.emplace_back("Content-Type", "application/json");
      inner.headers.emplace_back("Accept", "application/json");
      HttpResponse response = serve(inner);
      std::string body;
      if (response.streaming_body) {
        response.streaming_body([&body](std::string_view chunk) {
          if (body.size() + chunk.size() > kChatAnswerLimit)
            return false;
          body.append(chunk);
          return true;
        });
      } else {
        body = std::move(response.body);
      }
      if (std::chrono::steady_clock::now() > deadline)
        throw std::runtime_error("timeout after " + std::to_string(options.timeout_seconds) +
                                 " s");
      if (response.status < 200 || response.status >= 300)
        throw std::runtime_error(ChatError(response.status, body));
      const FingerCount count = ParseFingerAnswer(body);
      entry["fingers"] = count.fingers;
      entry["sicher"] = count.sicher;
      entry["befund"] = count.befund;
    } catch (const std::exception& error) {
      entry["fehler"] = error.what();
    }
    finish(entry);
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
  return info;
}

}  // namespace gufo::router::vision
