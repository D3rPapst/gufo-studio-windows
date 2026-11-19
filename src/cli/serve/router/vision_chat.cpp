// The vision chat of the model checks (see vision_chat.hpp).
// Written with AI assistance (Claude Code).
#include "src/cli/serve/router/vision_chat.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/router/rgba_image.hpp"
#include "src/core/base64.hpp"

namespace gufo::router::vision {
namespace {

using server::HttpRequest;
using server::HttpResponse;

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

}  // namespace

std::string ImageDataUrl(const models::ccip::RgbImage& image, int min_short, int max_long) {
  models::ccip::RgbImage cut = image;
  const int short_side = std::min(cut.width, cut.height);
  const int long_side = std::max(cut.width, cut.height);
  double scale = 1.0;
  if (min_short > 0 && short_side < min_short)
    scale = static_cast<double>(min_short) / short_side;
  if (max_long > 0 && long_side * scale > max_long)
    scale = static_cast<double>(max_long) / long_side;
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

json::Value TypeSchema(const char* type) {
  json::Value schema = json::Value::object();
  schema["type"] = type;
  return schema;
}

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

json::Value TextPart(const std::string& text) {
  json::Value part = json::Value::object();
  part["type"] = "text";
  part["text"] = text;
  return part;
}

json::Value ImagePart(const std::string& data_url) {
  json::Value image_url = json::Value::object();
  image_url["url"] = data_url;
  json::Value part = json::Value::object();
  part["type"] = "image_url";
  part["image_url"] = std::move(image_url);
  return part;
}

json::Value VisionChatBody(const std::string& model, json::Value content, const char* name,
                           json::Value schema, int max_tokens, bool thinking) {
  json::Value message = json::Value::object();
  message["role"] = "user";
  message["content"] = std::move(content);
  json::Value messages = json::Value::array();
  messages.push_back(std::move(message));
  json::Value json_schema = json::Value::object();
  json_schema["name"] = name;
  json_schema["strict"] = true;
  json_schema["schema"] = std::move(schema);
  json::Value format = json::Value::object();
  format["type"] = "json_schema";
  format["json_schema"] = std::move(json_schema);
  json::Value body = json::Value::object();
  body["model"] = model;
  body["messages"] = std::move(messages);
  body["max_tokens"] = max_tokens;
  body["temperature"] = 0;
  // Streamed: the first chunk marks the start of the generation, from
  // which timeout_seconds counts (AskVisionChat).
  body["stream"] = true;
  body["response_format"] = std::move(format);
  json::Value kwargs = json::Value::object();
  kwargs["enable_thinking"] = thinking;
  body["chat_template_kwargs"] = std::move(kwargs);
  return body;
}

json::Value ChatAnswerObject(std::string_view chat_body) {
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
  return verdict;
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
      return true;  // not ours to judge: ChatAnswerObject reports it
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

double RoundedSeconds(double seconds) {
  return std::round(seconds * 1000.0) / 1000.0;
}

std::string AskVisionChat(const std::string& body, const ChatLimits& limits,
                          const std::function<HttpResponse(const HttpRequest&)>& serve,
                          const HttpRequest& outer, std::string* model_gone,
                          double* waited_seconds) {
  using Clock = std::chrono::steady_clock;
  *waited_seconds = -1;
  const auto started = Clock::now();
  const auto wait_deadline = started + std::chrono::seconds(limits.wait_seconds);
  const auto generation_limit = std::chrono::seconds(limits.timeout_seconds);
  const auto first_chunk = std::make_shared<std::atomic<Clock::rep>>(0);
  const auto over = [wait_deadline, generation_limit, first_chunk] {
    const Clock::rep first = first_chunk->load();
    const auto now = Clock::now();
    if (first == 0)
      return now > wait_deadline;
    return now > Clock::time_point(Clock::duration(first)) + generation_limit;
  };
  HttpRequest inner;
  inner.method = "POST";
  inner.path = "/v1/chat/completions";
  inner.body = body;
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
  std::string answer;
  for (;;) {
    response = serve(inner);
    answer.clear();
    if (response.streaming_body) {
      // The generation clock starts with the first generated token, not
      // with the first "data:" line: a gufo worker sends the role chunk
      // already after 5 s in its queue (stream_start_delay), and a busy 27B
      // then spent the whole timeout_seconds in that queue (2.2.6-2.2.10:
      // "timeout after 300 s of generation").
      std::size_t scanned = 0;
      response.streaming_body([&](std::string_view chunk) {
        if (answer.size() + chunk.size() > kChatAnswerLimit)
          return false;
        answer.append(chunk);
        if (first_chunk->load() == 0 && GenerationStarted(answer, &scanned))
          first_chunk->store(Clock::now().time_since_epoch().count());
        return !over();
      });
    } else {
      answer = std::move(response.body);
    }
    // A 503 memory_busy (another model holds the memory): wait as it asks
    // while the wait clock allows, then ask again.
    const std::optional<int> retry =
        first_chunk->load() == 0 ? RetryAfter(response, answer) : std::nullopt;
    if (!retry.has_value())
      break;
    const auto resume = Clock::now() + std::chrono::seconds(*retry);
    if (resume > wait_deadline) {
      *model_gone = "model " + limits.model + " not free within wait_seconds " +
                    std::to_string(limits.wait_seconds) + " (" +
                    ChatError(response.status, answer) + ")";
      throw std::runtime_error(*model_gone);
    }
    while (Clock::now() < resume && !(outer_cancelled && outer_cancelled()))
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (outer_cancelled && outer_cancelled())
      throw std::runtime_error("cancelled");
  }
  if (const Clock::rep first = first_chunk->load(); first != 0)
    *waited_seconds = RoundedSeconds(
        std::chrono::duration<double>(Clock::time_point(Clock::duration(first)) - started)
            .count());
  if (over()) {
    if (first_chunk->load() == 0) {
      *model_gone = "model " + limits.model + " did not start within wait_seconds " +
                    std::to_string(limits.wait_seconds);
      throw std::runtime_error(*model_gone);
    }
    throw std::runtime_error("timeout after " + std::to_string(limits.timeout_seconds) +
                             " s of generation");
  }
  if (response.status < 200 || response.status >= 300)
    throw std::runtime_error(ChatError(response.status, answer));
  return answer;
}

}  // namespace gufo::router::vision
