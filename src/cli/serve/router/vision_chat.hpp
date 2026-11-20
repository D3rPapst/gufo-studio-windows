#ifndef GUFO_CLI_SERVE_ROUTER_VISION_CHAT_HPP_
#define GUFO_CLI_SERVE_ROUTER_VISION_CHAT_HPP_

// The vision chat shared by the model checks of POST /v1/vision/detect
// ("hand_fingers", vision_fingers.hpp; "person_count", vision_persons.hpp):
// one picture question to a chat model through the router's own
// /v1/chat/completions (gate and Dispatch, under the caller's own
// Authorization / Cookie), streamed, with a strict json_schema answer.
// Two clocks: until the first generated token (router and worker queue,
// load, prefill) the chat may wait wait_seconds, a 503 memory_busy is asked
// again after its Retry-After; from that token on (GenerationStarted) the
// generation has timeout_seconds.
// Written with AI assistance (Claude Code).

#include <array>
#include <functional>
#include <string>
#include <string_view>

#include "ccip/image_ops.hpp"
#include "src/cli/serve/http_server.hpp"
#include "src/core/json.hpp"

namespace gufo::router::vision {

/// Largest chat answer read back (a verdict is a few hundred bytes).
inline constexpr std::size_t kChatAnswerLimit = static_cast<std::size_t>(1) << 20;

/// `image` as a PNG data URL, enlarged (Pillow LANCZOS) to at least
/// `min_short` on its short side and shrunk to at most `max_long` on its
/// long side (0: no limit).
std::string ImageDataUrl(const models::ccip::RgbImage& image, int min_short, int max_long);

/// {"type": type}.
json::Value TypeSchema(const char* type);

/// A string schema with these names as its enum.
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
json::Value ObjectSchema(json::Value properties);

/// The /v1/chat/completions body: one user message with `content` (the
/// parts as sent), response_format json_schema strict named `name`,
/// temperature 0, streamed, chat_template_kwargs.enable_thinking.
json::Value VisionChatBody(const std::string& model, json::Value content, const char* name,
                           json::Value schema, int max_tokens, bool thinking);

/// {"type": "text", "text": text} and {"type": "image_url", ...}.
json::Value TextPart(const std::string& text);
json::Value ImagePart(const std::string& data_url);

/// The JSON object a model answered: the message content of a
/// /v1/chat/completions answer body, plain JSON or a stream ("data: {...}"
/// chunks, the delta contents joined), a code fence, a <think> block or
/// text around the object tolerated. Throws std::runtime_error with the
/// reason (no text, no object, invalid JSON, cut off at max_tokens).
json::Value ChatAnswerObject(std::string_view chat_body);

/// Whether the stream text `body`, from byte `*scanned` on, holds a whole
/// "data:" line that shows the generation running: a non-empty delta
/// content, reasoning_content or tool_calls, a finish_reason or an error.
/// The role chunk ({"delta": {"role": "assistant"}}) and prompt_progress
/// chunks do not count: a gufo worker sends the role chunk already when the
/// request has waited stream_start_delay (5 s) in its queue behind another
/// chat, long before the model works on it. `*scanned` moves past the whole
/// lines looked at (an unfinished last line is looked at again later).
bool GenerationStarted(std::string_view body, std::size_t* scanned);

/// The clocks of one chat (see the top of this file).
struct ChatLimits {
  std::string model;
  int wait_seconds{600};
  int timeout_seconds{300};
};

/// Asks one chat with `body` through `serve`, the caller's Authorization /
/// Cookie / X-Gufo-Pool copied from `outer`, and returns the answer body (2xx). Throws
/// std::runtime_error: refused ("chat <status>: <message>"), cancelled,
/// timed out. When the model did not start (or stayed busy) within
/// wait_seconds, *model_gone is set as well (the caller asks no further
/// chats). *waited_seconds is the time until the first chunk, -1 without
/// one.
std::string AskVisionChat(const std::string& body, const ChatLimits& limits,
                          const std::function<server::HttpResponse(const server::HttpRequest&)>& serve,
                          const server::HttpRequest& outer, std::string* model_gone,
                          double* waited_seconds);

/// Seconds rounded to milliseconds.
double RoundedSeconds(double seconds);

}  // namespace gufo::router::vision

#endif  // GUFO_CLI_SERVE_ROUTER_VISION_CHAT_HPP_
