// 2.2.3: how the ComfyUI adapter reports a model that cannot load because
// another, busy model holds the memory (the router's 503 memory_busy):
//  - in /history/<id> as execution_error with exception_type "GufoBusy",
//    "code": "memory_busy", "retry_after_s" and error {code, message,
//    retry_after_s} when a task's request gets it;
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
  CHECK(image_posts.load() == 1);  // no retry inside the adapter
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

}  // namespace

int main() {
  const auto root = std::filesystem::temp_directory_path() /
                    "gufo-comfy-busy-test";
  std::filesystem::remove_all(root);
  TestMemoryBusyOf();
  TestHistoryMark(root);
  TestSubmitRefused(root);
  std::filesystem::remove_all(root);
  if (failures != 0) {
    std::cerr << failures << " comfy memory_busy checks failed.\n";
    return 1;
  }
  std::cout << "All comfy memory_busy tests passed.\n";
  return 0;
}
