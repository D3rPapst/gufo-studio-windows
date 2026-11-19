// 2.2.3: how the ComfyUI adapter reports a model that cannot load because
// another, busy model holds the memory (the router's 503 memory_busy):
//  - 2.2.12: a task's request that gets it waits (Retry-After) and is sent
//    again, like the passing 507; only after memory_free_limit does the
//    prompt end in /history/<id> as execution_error with exception_type
//    "GufoBusy", "code": "memory_busy", "retry_after_s" and error {code,
//    message, retry_after_s};
//  - as 503 with Retry-After on POST /prompt when the adapter's own queue
//    is empty and the router's memory check says the first model must wait.
// The router is a fake fetch; no model, no GPU.

#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/comfy_adapter.hpp"
#include "src/core/json.hpp"

namespace comfy = gufo::router::comfy;
namespace json = gufo::json;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;

namespace {

int failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond \
                << "\n";                                                   \
      ++failures;                                                          \
    }                                                                      \
  } while (0)

constexpr const char* kQwenText = R"({
 "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
 "2": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
 "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
 "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["1", 0], "prompt": "an office", "negative_prompt": "", "vae": ["2", 0], "resolution": 1024}},
 "6": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["5", 2], "seed": 1, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
 "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "busy/test"}}
})";

constexpr const char* kBusyBody =
    R"x({"error":{"message":"cannot load 'qwen-image-2.1' now: memory is in use by minimax-h3 (busy); retry in 30 s","type":"server_error","code":"memory_busy","retry_after_s":30,"blocked_by":"minimax-h3 (busy)","model":"qwen-image-2.1"}})x";

HttpResponse Post(comfy::Adapter& adapter, const std::string& body) {
  HttpRequest request;
  request.method = "POST";
  request.path = "/prompt";
  request.body = body;
  request.headers.emplace_back("Content-Type", "application/json");
  const auto response = adapter.Handle(request);
  return response ? *response : HttpResponse{.status = 0};
}

std::optional<json::Value> History(comfy::Adapter& adapter,
                                   const std::string& id) {
  HttpRequest request;
  request.method = "GET";
  request.path = "/history/" + id;
  for (int i = 0; i < 500; ++i) {
    const auto response = adapter.Handle(request);
    if (response && response->status == 200) {
      const json::Value body = json::parse(response->body);
      if (const json::Value* entry = body.find(id))
        return *entry;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return std::nullopt;
}

std::string Header(const HttpResponse& response, const std::string& name) {
  for (const auto& [key, value] : response.headers)
    if (key == name)
      return value;
  return {};
}

void TestMemoryBusyOf() {
  comfy::FetchResult busy{.status = 503, .body = kBusyBody};
  const auto error = comfy::MemoryBusyOf(busy);
  CHECK(error.has_value() && error->retry_after_s == 30 &&
        std::string(error->what()).find("minimax-h3 (busy)") !=
            std::string::npos);
  // Other 503s (pool_busy, shutting down) and other statuses are not it.
  CHECK(!comfy::MemoryBusyOf(
      {.status = 503,
       .body = R"({"error":{"message":"x","code":"pool_busy"}})"}));
  CHECK(!comfy::MemoryBusyOf({.status = 507, .body = kBusyBody}));
  CHECK(!comfy::MemoryBusyOf({.status = 503, .body = "not json"}));
  // A missing or silly retry_after_s falls back to 30.
  const auto fallback = comfy::MemoryBusyOf(
      {.status = 503,
       .body = R"({"error":{"message":"m","code":"memory_busy","retry_after_s":-4}})"});
  CHECK(fallback && fallback->retry_after_s == comfy::kMemoryBusyRetrySeconds);
}

void TestHistoryMark(const std::filesystem::path& root) {
  std::atomic<int> image_posts{0};
  comfy::AdapterOptions options;
  options.root = root / "history";
  // 2.2.12: memory_busy is waited for; past the limit it is GufoBusy.
  options.memory_free_retry = std::chrono::milliseconds(10);
  options.memory_free_limit = std::chrono::milliseconds(80);
  options.retry_after_unit = std::chrono::milliseconds(1);
  options.fetch = [&](const HttpRequest& request) {
    if (request.path == "/v1/images/generations" ||
        request.path == "/v1/images/edits") {
      ++image_posts;
      return comfy::FetchResult{.status = 503,
                                .body = kBusyBody,
                                .content_type = "application/json"};
    }
    return comfy::FetchResult{.status = 404, .body = "{}"};
  };
  // The pre-check says "may load" (the model became busy only later).
  options.memory_busy = [](const std::string&) -> std::optional<std::string> {
    return std::nullopt;
  };
  comfy::Adapter adapter(std::move(options));
  adapter.Start();
  const HttpResponse accepted =
      Post(adapter, std::string(R"({"prompt":)") + kQwenText + "}");
  CHECK(accepted.status == 200);
  const std::string id = json::parse(accepted.body).member_str("prompt_id");
  const auto entry = History(adapter, id);
  CHECK(entry.has_value());
  if (entry) {
    const json::Value* status = entry->find("status");
    CHECK(status != nullptr && status->member_str("status_str") == "error");
    const json::Value* error = nullptr;
    for (const json::Value& message : status->find("messages")->items())
      if (message.items()[0].str() == "execution_error")
        error = &message.items()[1];
    CHECK(error != nullptr);
    if (error) {
      CHECK(error->member_str("exception_type") == "GufoBusy");
      CHECK(error->member_str("code") == "memory_busy");
      CHECK(error->member_double("retry_after_s") == 30);
      CHECK(error->member_str("exception_message").find("minimax-h3 (busy)") !=
            std::string::npos);
      const json::Value* detail = error->find("error");
      CHECK(detail != nullptr && detail->member_str("code") == "memory_busy" &&
            detail->member_double("retry_after_s") == 30);
      CHECK(error->member_str("node_id") == "6");
    }
  }
  CHECK(image_posts.load() >= 2);  // waited and asked again first
  adapter.Stop();
}

void TestSubmitRefused(const std::filesystem::path& root) {
  std::atomic<int> fetches{0};
  std::string asked;
  comfy::AdapterOptions options;
  options.root = root / "submit";
  options.fetch = [&](const HttpRequest&) {
    ++fetches;
    return comfy::FetchResult{.status = 500, .body = "{}"};
  };
  options.memory_busy = [&](const std::string& model) {
    asked = model;
    return std::optional<std::string>(
        "cannot load 'qwen-image-2.1' now: memory is in use by minimax-h3 "
        "(busy); retry in 30 s");
  };
  comfy::Adapter adapter(std::move(options));
  adapter.Start();
  const HttpResponse refused =
      Post(adapter, std::string(R"({"prompt":)") + kQwenText + "}");
  CHECK(refused.status == 503);
  CHECK(Header(refused, "Retry-After") == "30");
  CHECK(asked == "qwen-image-2.1");
  const json::Value body = json::parse(refused.body);
  const json::Value* error = body.find("error");
  CHECK(error != nullptr && error->member_str("code") == "memory_busy" &&
        error->member_str("type") == "memory_busy" &&
        error->member_double("retry_after_s") == 30 &&
        error->member_str("message").find("minimax-h3") != std::string::npos);
  CHECK(body.find("node_errors") != nullptr);
  CHECK(adapter.QueueRemaining() == 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  CHECK(fetches.load() == 0);  // nothing was queued or sent
  adapter.Stop();
}

// 2.2.8: a video task whose POST /v1/videos meets a full worker queue
// (429 video_queue_full) or a full router (503 overloaded) waits and asks
// again; a status poll that gets a passing error goes on polling.
constexpr const char* kH3Text = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "strength_model": 1.0}},
 "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "type": "minimax", "device": "default"}},
 "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
 "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
 "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "a quiet office at night"}},
 "7": {"class_type": "EmptyMiniMaxH3LatentAV", "inputs": {"width": 896, "height": 512, "length": 124}},
 "8": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["6", 0], "latent_image": ["7", 0], "seed": 5, "steps": 8, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
 "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
 "10": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["8", 0], "vae": ["5", 0]}},
 "11": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "h3/f"}},
 "12": {"class_type": "SaveAudio", "inputs": {"audio": ["10", 0], "filename_prefix": "h3/ton"}}
})";

constexpr const char* kQueueFullBody =
    R"({"error":{"message":"video queue is full","type":"rate_limit_error","code":"video_queue_full"}})";
constexpr const char* kOverloadedBody =
    R"({"error":{"message":"connection limit reached","type":"server_error","code":"overloaded"}})";

void TestRetryLaterOf() {
  CHECK(comfy::RetryLaterOf({.status = 429, .body = kQueueFullBody}) ==
        std::optional<std::string>("video_queue_full"));
  CHECK(comfy::RetryLaterOf({.status = 503, .body = kOverloadedBody}) ==
        std::optional<std::string>("overloaded"));
  // memory_busy keeps its own path (GufoBusy); other codes are failures.
  CHECK(!comfy::RetryLaterOf({.status = 503, .body = kBusyBody}));
  CHECK(!comfy::RetryLaterOf({.status = 429, .body = kOverloadedBody}));
  CHECK(!comfy::RetryLaterOf({.status = 503, .body = kQueueFullBody}));
  CHECK(!comfy::RetryLaterOf({.status = 429, .body = "not json"}));
  CHECK(!comfy::RetryLaterOf({.status = 202, .body = R"({"id":"v"})"}));
  CHECK(comfy::IsPassingPollFailure(503) && comfy::IsPassingPollFailure(0) &&
        comfy::IsPassingPollFailure(429) && !comfy::IsPassingPollFailure(404) &&
        !comfy::IsPassingPollFailure(500));
}

// The router's two 507 insufficient_memory answers (router.cpp AdmitLocked):
// the passing one (memory.cpp PlanAdmission, Wait::kOutside) and the load
// stopped by low memory.
constexpr const char* kNotFreeBody =
    R"({"error":{"message":"cannot load 'wan2.1-vace-14b': not enough free memory yet (held outside the router or still being released); need 46.0 GB gpu + 2.0 GB host; free gpu 47.1+7.7 GB, host 5.7 GB, commit 2.9 GB","type":"server_error","code":"insufficient_memory"}})";
constexpr const char* kExclusiveNotFreeBody =
    R"({"error":{"message":"cannot load 'x': exclusive, not enough free memory yet (held outside the router or still being released); free gpu 1.0 GB","type":"server_error","code":"insufficient_memory"}})";
constexpr const char* kRanLowBody =
    R"({"error":{"message":"loading 'x' was stopped because memory ran low","type":"server_error","code":"insufficient_memory"}})";

void TestMemoryNotFreeOf() {
  const auto message =
      comfy::MemoryNotFreeOf({.status = 507, .body = kNotFreeBody});
  CHECK(message && message->find("46.0 GB gpu") != std::string::npos);
  CHECK(comfy::MemoryNotFreeOf({.status = 507, .body = kExclusiveNotFreeBody}));
  CHECK(!comfy::MemoryNotFreeOf({.status = 507, .body = kRanLowBody}));
  CHECK(!comfy::MemoryNotFreeOf({.status = 503, .body = kNotFreeBody}));
  CHECK(!comfy::MemoryNotFreeOf({.status = 507, .body = "not json"}));
  CHECK(!comfy::MemoryNotFreeOf(
      {.status = 507,
       .body = R"({"error":{"message":"not enough free memory yet","code":"other"}})"}));
  CHECK(comfy::IsMemoryNotFreeYet(507, "insufficient_memory",
                                  "x: not enough free memory yet (held"));
  CHECK(!comfy::IsMemoryNotFreeYet(
      507, "insufficient_memory",
      "loading 'x' was stopped because memory ran low"));
  // Neither a video "come back later" nor memory_busy.
  CHECK(!comfy::RetryLaterOf({.status = 507, .body = kNotFreeBody}));
  CHECK(!comfy::MemoryBusyOf({.status = 507, .body = kNotFreeBody}));
}

std::string ErrorText(const json::Value& entry) {
  const json::Value* status = entry.find("status");
  if (status == nullptr || status->member_str("status_str") != "error")
    return {};
  for (const json::Value& message : status->find("messages")->items())
    if (message.items()[0].str() == "execution_error")
      return message.items()[1].member_str("exception_message");
  return {};
}

void TestVideoWaits(const std::filesystem::path& root) {
  std::atomic<int> posts{0};
  std::atomic<int> polls{0};
  comfy::AdapterOptions options;
  options.root = root / "video-wait";
  options.video_busy_retry = std::chrono::milliseconds(20);
  options.video_poll = std::chrono::milliseconds(10);
  options.fetch = [&](const HttpRequest& request) {
    if (request.method == "POST" && request.path == "/v1/videos") {
      const int n = ++posts;
      if (n == 1)
        return comfy::FetchResult{.status = 429, .body = kQueueFullBody};
      if (n == 2)
        return comfy::FetchResult{.status = 503, .body = kOverloadedBody};
      return comfy::FetchResult{.status = 202,
                                .body = R"({"id":"video_wait1","status":"queued"})",
                                .content_type = "application/json"};
    }
    if (request.method == "GET" && request.path == "/v1/videos/video_wait1") {
      const int n = ++polls;
      if (n == 1)
        return comfy::FetchResult{.status = 503, .body = kOverloadedBody};
      return comfy::FetchResult{
          .status = 200,
          .body = R"({"id":"video_wait1","status":"failed","progress":3,"error":{"message":"fake end"}})"};
    }
    return comfy::FetchResult{.status = 404, .body = "{}"};
  };
  comfy::Adapter adapter(std::move(options));
  adapter.Start();
  const HttpResponse accepted =
      Post(adapter, std::string(R"({"prompt":)") + kH3Text + "}");
  CHECK(accepted.status == 200);
  const std::string id = json::parse(accepted.body).member_str("prompt_id");
  const auto entry = History(adapter, id);
  CHECK(entry.has_value());
  if (entry) {
    // It got past the full queue and the overloaded poll: the worker's own
    // end ("failed") is what ends the prompt.
    const std::string error = ErrorText(*entry);
    CHECK(error.find("fake end") != std::string::npos);
    CHECK(error.find("429") == std::string::npos);
  }
  CHECK(posts.load() == 3);
  CHECK(polls.load() == 2);
  adapter.Stop();
}

void TestVideoWaitLimit(const std::filesystem::path& root) {
  std::atomic<int> posts{0};
  comfy::AdapterOptions options;
  options.root = root / "video-limit";
  options.video_busy_retry = std::chrono::milliseconds(10);
  options.video_busy_limit = std::chrono::milliseconds(60);
  options.fetch = [&](const HttpRequest& request) {
    if (request.method == "POST" && request.path == "/v1/videos") {
      ++posts;
      return comfy::FetchResult{.status = 429, .body = kQueueFullBody};
    }
    return comfy::FetchResult{.status = 404, .body = "{}"};
  };
  comfy::Adapter adapter(std::move(options));
  adapter.Start();
  const HttpResponse accepted =
      Post(adapter, std::string(R"({"prompt":)") + kH3Text + "}");
  CHECK(accepted.status == 200);
  const std::string id = json::parse(accepted.body).member_str("prompt_id");
  const auto entry = History(adapter, id);
  CHECK(entry.has_value());
  if (entry)
    CHECK(ErrorText(*entry).find("HTTP 429") != std::string::npos);
  CHECK(posts.load() >= 2);
  adapter.Stop();
}

void TestVideoWaitInterrupted(const std::filesystem::path& root) {
  std::atomic<int> posts{0};
  comfy::AdapterOptions options;
  options.root = root / "video-interrupt";
  options.video_busy_retry = std::chrono::milliseconds(20);
  options.video_busy_limit = std::chrono::milliseconds(0);  // no limit
  options.fetch = [&](const HttpRequest& request) {
    if (request.method == "POST" && request.path == "/v1/videos") {
      ++posts;
      return comfy::FetchResult{.status = 429, .body = kQueueFullBody};
    }
    return comfy::FetchResult{.status = 404, .body = "{}"};
  };
  comfy::Adapter adapter(std::move(options));
  adapter.Start();
  const HttpResponse accepted =
      Post(adapter, std::string(R"({"prompt":)") + kH3Text + "}");
  CHECK(accepted.status == 200);
  const std::string id = json::parse(accepted.body).member_str("prompt_id");
  for (int i = 0; i < 200 && posts.load() < 3; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  CHECK(posts.load() >= 3);  // still waiting, not failed
  CHECK(adapter.QueueRemaining() == 1);
  HttpRequest interrupt;
  interrupt.method = "POST";
  interrupt.path = "/interrupt";
  interrupt.body = "{}";
  interrupt.headers.emplace_back("Content-Type", "application/json");
  (void)adapter.Handle(interrupt);
  const auto entry = History(adapter, id);
  CHECK(entry.has_value());
  if (entry) {
    bool interrupted = false;
    for (const json::Value& message :
         entry->find("status")->find("messages")->items())
      interrupted = interrupted ||
                    message.items()[0].str() == "execution_interrupted";
    CHECK(interrupted);
  }
  adapter.Stop();
}

// A video task (vace in the Regie's case) and an image task wait through
// the passing 507 and go on; another 507 fails the prompt at once.
void TestMemoryNotFreeWait(const std::filesystem::path& root) {
  std::atomic<int> video_posts{0};
  comfy::AdapterOptions options;
  options.root = root / "not-free-video";
  options.memory_free_retry = std::chrono::milliseconds(20);
  options.video_poll = std::chrono::milliseconds(10);
  options.fetch = [&](const HttpRequest& request) {
    if (request.method == "POST" && request.path == "/v1/videos") {
      const int n = ++video_posts;
      if (n <= 3)
        return comfy::FetchResult{.status = 507, .body = kNotFreeBody};
      return comfy::FetchResult{.status = 202,
                                .body = R"({"id":"video_nf","status":"queued"})",
                                .content_type = "application/json"};
    }
    if (request.method == "GET" && request.path == "/v1/videos/video_nf")
      return comfy::FetchResult{
          .status = 200,
          .body = R"({"id":"video_nf","status":"failed","progress":3,"error":{"message":"fake end"}})"};
    return comfy::FetchResult{.status = 404, .body = "{}"};
  };
  comfy::Adapter adapter(std::move(options));
  adapter.Start();
  const HttpResponse accepted =
      Post(adapter, std::string(R"({"prompt":)") + kH3Text + "}");
  CHECK(accepted.status == 200);
  const std::string id = json::parse(accepted.body).member_str("prompt_id");
  const auto entry = History(adapter, id);
  CHECK(entry.has_value());
  if (entry) {
    const std::string error = ErrorText(*entry);
    CHECK(error.find("fake end") != std::string::npos);
    CHECK(error.find("507") == std::string::npos);
  }
  CHECK(video_posts.load() == 4);
  adapter.Stop();
}

void TestMemoryNotFreeImage(const std::filesystem::path& root) {
  std::atomic<int> posts{0};
  comfy::AdapterOptions options;
  options.root = root / "not-free-image";
  options.memory_free_retry = std::chrono::milliseconds(10);
  options.fetch = [&](const HttpRequest& request) {
    if (request.path == "/v1/images/generations" ||
        request.path == "/v1/images/edits") {
      const int n = ++posts;
      if (n == 1)
        return comfy::FetchResult{.status = 507, .body = kNotFreeBody};
      // Then a hard failure: the prompt ends with it (not with the 507).
      return comfy::FetchResult{
          .status = 500,
          .body = R"({"error":{"message":"fake image end","code":"x"}})"};
    }
    return comfy::FetchResult{.status = 404, .body = "{}"};
  };
  comfy::Adapter adapter(std::move(options));
  adapter.Start();
  const HttpResponse accepted =
      Post(adapter, std::string(R"({"prompt":)") + kQwenText + "}");
  CHECK(accepted.status == 200);
  const std::string id = json::parse(accepted.body).member_str("prompt_id");
  const auto entry = History(adapter, id);
  CHECK(entry.has_value());
  if (entry)
    CHECK(ErrorText(*entry).find("fake image end") != std::string::npos);
  CHECK(posts.load() == 2);
  adapter.Stop();
}

void TestMemoryRanLowFails(const std::filesystem::path& root) {
  std::atomic<int> posts{0};
  comfy::AdapterOptions options;
  options.root = root / "ran-low";
  options.memory_free_retry = std::chrono::milliseconds(10);
  options.fetch = [&](const HttpRequest& request) {
    if (request.method == "POST" && request.path == "/v1/videos") {
      ++posts;
      return comfy::FetchResult{.status = 507, .body = kRanLowBody};
    }
    return comfy::FetchResult{.status = 404, .body = "{}"};
  };
  comfy::Adapter adapter(std::move(options));
  adapter.Start();
  const HttpResponse accepted =
      Post(adapter, std::string(R"({"prompt":)") + kH3Text + "}");
  CHECK(accepted.status == 200);
  const std::string id = json::parse(accepted.body).member_str("prompt_id");
  const auto entry = History(adapter, id);
  CHECK(entry.has_value());
  if (entry)
    CHECK(ErrorText(*entry).find("HTTP 507") != std::string::npos);
  CHECK(posts.load() == 1);
  adapter.Stop();
}

void TestMemoryNotFreeLimit(const std::filesystem::path& root) {
  std::atomic<int> posts{0};
  comfy::AdapterOptions options;
  options.root = root / "not-free-limit";
  options.memory_free_retry = std::chrono::milliseconds(10);
  options.memory_free_limit = std::chrono::milliseconds(60);
  options.fetch = [&](const HttpRequest& request) {
    if (request.method == "POST" && request.path == "/v1/videos") {
      ++posts;
      return comfy::FetchResult{.status = 507, .body = kNotFreeBody};
    }
    return comfy::FetchResult{.status = 404, .body = "{}"};
  };
  comfy::Adapter adapter(std::move(options));
  adapter.Start();
  const HttpResponse accepted =
      Post(adapter, std::string(R"({"prompt":)") + kH3Text + "}");
  CHECK(accepted.status == 200);
  const std::string id = json::parse(accepted.body).member_str("prompt_id");
  const auto entry = History(adapter, id);
  CHECK(entry.has_value());
  if (entry)
    CHECK(ErrorText(*entry).find("HTTP 507") != std::string::npos);
  CHECK(posts.load() >= 2);
  adapter.Stop();
}

// 2.2.12 (19.11., wan2.2-i2v over the pool): memory_busy on the video
// create, on a status poll and on the content read is waited for (with the
// answer's Retry-After), and the clip goes on. Before, the first one ended
// the prompt as GufoBusy.
void TestMemoryBusyWait(const std::filesystem::path& root) {
  std::atomic<int> posts{0};
  std::atomic<int> polls{0};
  std::atomic<int> contents{0};
  comfy::AdapterOptions options;
  options.root = root / "busy-wait";
  options.memory_free_retry = std::chrono::milliseconds(5);
  options.retry_after_unit = std::chrono::milliseconds(1);  // 30 s -> 30 ms
  options.video_poll = std::chrono::milliseconds(5);
  options.video_poll_grace = std::chrono::milliseconds(1);
  options.fetch = [&](const HttpRequest& request) {
    if (request.method == "POST" && request.path == "/v1/videos") {
      if (++posts <= 2)
        return comfy::FetchResult{.status = 503, .body = kBusyBody};
      return comfy::FetchResult{.status = 202,
                                .body = R"({"id":"video_mb","status":"queued"})",
                                .content_type = "application/json"};
    }
    if (request.method == "GET" && request.path == "/v1/videos/video_mb") {
      // Longer than video_poll_grace: memory waits do not count to it.
      if (++polls <= 4) {
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        return comfy::FetchResult{.status = 503, .body = kBusyBody};
      }
      return comfy::FetchResult{
          .status = 200,
          .body = R"({"id":"video_mb","status":"completed","progress":100})"};
    }
    if (request.method == "GET" &&
        request.path == "/v1/videos/video_mb/content") {
      if (++contents == 1)
        return comfy::FetchResult{.status = 503, .body = kBusyBody};
      return comfy::FetchResult{.status = 200,
                                .body = "not really an mp4",
                                .content_type = "video/mp4"};
    }
    return comfy::FetchResult{.status = 404, .body = "{}"};
  };
  comfy::Adapter adapter(std::move(options));
  adapter.Start();
  const HttpResponse accepted =
      Post(adapter, std::string(R"({"prompt":)") + kH3Text + "}");
  CHECK(accepted.status == 200);
  const std::string id = json::parse(accepted.body).member_str("prompt_id");
  const auto entry = History(adapter, id);
  CHECK(entry.has_value());
  if (entry) {
    // It got the content; what ends this fake prompt is saving frames
    // without ffmpeg, never memory_busy.
    bool busy = false;
    for (const json::Value& message :
         entry->find("status")->find("messages")->items())
      busy = busy || (message.items()[0].str() == "execution_error" &&
                      message.items()[1].member_str("exception_type") ==
                          "GufoBusy");
    CHECK(!busy);
    CHECK(ErrorText(*entry).find("memory") == std::string::npos);
  }
  CHECK(posts.load() == 3);
  CHECK(polls.load() == 5);
  CHECK(contents.load() == 2);
  adapter.Stop();
}

// memory_busy waits until /interrupt, like the full video queue.
void TestMemoryBusyInterrupted(const std::filesystem::path& root) {
  std::atomic<int> posts{0};
  comfy::AdapterOptions options;
  options.root = root / "busy-interrupt";
  options.memory_free_retry = std::chrono::milliseconds(5);
  options.memory_free_limit = std::chrono::milliseconds(0);  // no limit
  options.retry_after_unit = std::chrono::milliseconds(1);
  options.fetch = [&](const HttpRequest& request) {
    if (request.method == "POST" && request.path == "/v1/videos") {
      ++posts;
      return comfy::FetchResult{.status = 503, .body = kBusyBody};
    }
    return comfy::FetchResult{.status = 404, .body = "{}"};
  };
  comfy::Adapter adapter(std::move(options));
  adapter.Start();
  const HttpResponse accepted =
      Post(adapter, std::string(R"({"prompt":)") + kH3Text + "}");
  CHECK(accepted.status == 200);
  const std::string id = json::parse(accepted.body).member_str("prompt_id");
  for (int i = 0; i < 300 && posts.load() < 4; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  CHECK(posts.load() >= 4);  // still waiting, not failed
  CHECK(adapter.QueueRemaining() == 1);
  HttpRequest interrupt;
  interrupt.method = "POST";
  interrupt.path = "/interrupt";
  interrupt.body = "{}";
  interrupt.headers.emplace_back("Content-Type", "application/json");
  (void)adapter.Handle(interrupt);
  const auto entry = History(adapter, id);
  CHECK(entry.has_value());
  if (entry) {
    bool interrupted = false;
    for (const json::Value& message :
         entry->find("status")->find("messages")->items())
      interrupted = interrupted ||
                    message.items()[0].str() == "execution_interrupted";
    CHECK(interrupted);
  }
  adapter.Stop();
}

void TestMemoryWaitOf() {
  const auto busy = comfy::MemoryWaitOf({.status = 503, .body = kBusyBody});
  CHECK(busy && busy->code == "memory_busy" && busy->retry_after_s == 30);
  const auto not_free =
      comfy::MemoryWaitOf({.status = 507, .body = kNotFreeBody});
  CHECK(not_free && not_free->code == "insufficient_memory" &&
        not_free->retry_after_s == 0);
  CHECK(!comfy::MemoryWaitOf({.status = 507, .body = kRanLowBody}));
  CHECK(!comfy::MemoryWaitOf({.status = 503, .body = kOverloadedBody}));
}

}  // namespace

int main() {
  const auto root = std::filesystem::temp_directory_path() /
                    "gufo-comfy-busy-test";
  std::filesystem::remove_all(root);
  TestMemoryBusyOf();
  TestHistoryMark(root);
  TestSubmitRefused(root);
  TestRetryLaterOf();
  TestVideoWaits(root);
  TestVideoWaitLimit(root);
  TestVideoWaitInterrupted(root);
  TestMemoryNotFreeOf();
  TestMemoryNotFreeWait(root);
  TestMemoryNotFreeImage(root);
  TestMemoryRanLowFails(root);
  TestMemoryNotFreeLimit(root);
  TestMemoryWaitOf();
  TestMemoryBusyWait(root);
  TestMemoryBusyInterrupted(root);
  std::filesystem::remove_all(root);
  if (failures != 0) {
    std::cerr << failures << " comfy memory_busy checks failed.\n";
    return 1;
  }
  std::cout << "All comfy memory_busy tests passed.\n";
  return 0;
}
