// Maintenance drain (2.1.8, POST /router/pool/drain) over real sockets:
// two `gufo router` processes (argv[1]) serve the pool model "twin" with
// each other as partner, and each one's local twin is this test binary
// (--fake-worker --where a|b, the answer names the machine).
//
// 15.11.: pause-run.ps1 on the .198 waits for 30 quiet seconds, but the
// .56 kept sending pool requests (qwen-image) and the pause hung for over
// 40 min. With drain on, router A refuses new work the partner relays
// (503 pool_busy) and reports "draining"; partner B then runs the request
// itself, and A's own requests run as before. The state lives in memory
// only: a restarted router starts with it off.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/proxy.hpp"
#include "src/cli/serve/router/workers.hpp"
#include "src/core/json.hpp"
#include "src/core/platform/net.hpp"

namespace {

namespace net = gufo::net;
namespace json = gufo::json;
using gufo::router::ProxyToWorker;
using gufo::router::UpstreamTarget;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;

int failures = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond \
                << "\n";                                                  \
      ++failures;                                                         \
    }                                                                     \
  } while (0)

std::string Answer(int status, std::string_view reason, std::string_view body) {
  return "HTTP/1.1 " + std::to_string(status) + " " + std::string(reason) +
         "\r\nContent-Type: application/json\r\nContent-Length: " +
         std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" +
         std::string(body);
}

std::string ReadRequest(net::Socket client) {
  std::string data;
  char buffer[4096];
  std::size_t want = std::string::npos;
  while (want == std::string::npos || data.size() < want) {
    if (net::PollReadable(client, 10000) <= 0)
      break;
    const auto count = net::Receive(client, buffer, sizeof(buffer));
    if (count <= 0)
      break;
    data.append(buffer, static_cast<std::size_t>(count));
    const auto end = data.find("\r\n\r\n");
    if (want == std::string::npos && end != std::string::npos) {
      std::size_t body = 0;
      for (const char* name : {"Content-Length: ", "content-length: "}) {
        const auto at = data.find(name);
        if (at != std::string::npos && at < end)
          body = std::stoul(data.substr(at + 16));
      }
      want = end + 4 + body;
    }
  }
  return data;
}

/// The local twin: every GET 200, every POST a chat answer naming the
/// machine ("where": a|b). Ends after 60 s without a connection.
int ServeFakeWorker(std::span<char*> args) {
  int port = 0;
  std::string where = "?";
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (std::string_view(args[i]) == "--port")
      port = std::atoi(args[i + 1]);
    if (std::string_view(args[i]) == "--where")
      where = args[i + 1];
  }
  if (port <= 0)
    return 2;
  const net::Socket listener = net::CreateStreamSocket(AF_INET);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<unsigned short>(port));
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (listener == net::kInvalidSocket ||
      ::bind(listener, reinterpret_cast<const sockaddr*>(&address),
             sizeof(address)) != 0 ||
      ::listen(listener, 16) != 0)
    return 3;
  auto last = std::chrono::steady_clock::now();
  for (;;) {
    if (net::PollReadable(listener, 100) <= 0) {
      if (std::chrono::steady_clock::now() - last > std::chrono::seconds(60))
        std::_Exit(0);
      continue;
    }
    last = std::chrono::steady_clock::now();
    const net::Socket client = net::AcceptClientSocket(listener);
    if (client == net::kInvalidSocket)
      continue;
    const std::string request = ReadRequest(client);
    const std::string text =
        std::string_view(request).starts_with("POST ")
            ? Answer(200, "OK",
                     "{\"object\":\"chat.completion\",\"where\":\"" + where +
                         "\"}")
            : Answer(200, "OK", "{\"status\":\"ok\"}");
    std::string_view rest = text;
    while (!rest.empty()) {
      const auto sent = net::SendNoSignal(client, rest.data(), rest.size());
      if (sent <= 0)
        break;
      rest.remove_prefix(static_cast<std::size_t>(sent));
    }
    net::CloseSocket(client);
  }
}

struct Reply {
  int status{0};
  std::string body;
  std::string retry_after;
};

Reply Ask(int port, const std::string& method, const std::string& path,
          const std::string& body = "",
          std::vector<std::pair<std::string, std::string>> headers = {}) {
  UpstreamTarget target;
  target.host = "127.0.0.1";
  target.port = port;
  target.connect_timeout = std::chrono::seconds(2);
  target.head_timeout = std::chrono::seconds(60);
  HttpRequest request;
  request.method = method;
  request.path = path;
  request.body = body;
  if (!body.empty())
    request.headers.emplace_back("Content-Type", "application/json");
  for (auto& header : headers)
    request.headers.push_back(std::move(header));
  HttpResponse response = ProxyToWorker(target, request);
  Reply reply;
  reply.status = response.status;
  reply.body = response.body;
  if (response.streaming_body) {
    response.streaming_body([&reply](std::string_view chunk) {
      reply.body.append(chunk);
      return true;
    });
  }
  for (const auto& [name, value] : response.headers)
    if (name == "Retry-After" || name == "retry-after")
      reply.retry_after = value;
  return reply;
}

json::Value Parse(const Reply& reply) {
  try {
    return json::parse(reply.body);
  } catch (const std::exception&) {
    return json::Value::object();
  }
}

bool Bool(const json::Value& object, const char* key, bool def) {
  const json::Value* value = object.find(key);
  return value != nullptr && value->is_bool() ? value->as_bool() : def;
}

std::string ErrorCode(const Reply& reply) {
  const json::Value body = Parse(reply);
  const json::Value* error = body.find("error");
  return error != nullptr && error->is_object() ? error->member_str("code")
                                                : std::string();
}

const std::string kChat =
    "{\"model\":\"twin\",\"messages\":[{\"role\":\"user\",\"content\":"
    "\"x\"}]}";

/// One router of the pair as a child process.
struct Router {
  int port{0};
  std::unique_ptr<gufo::router::Worker> worker;
  bool Gone() {
    int status = 0;
    if (worker == nullptr || !worker->PollExit(&status))
      return false;
    std::cerr << "router on " << port << " exited with status " << status
              << "\n";
    return true;
  }
};

bool StartRouter(Router* router, const std::string& gufo_exe,
                 const std::filesystem::path& preset, int port) {
  gufo::router::WorkerSpec spec;
  spec.model_id = "router";
  spec.port = port;
  spec.command = gufo_exe;
  spec.args = {"router",       "--models-preset", preset.string(),
               "--models-max", "0",               "--remote-allow",
               "127.0.0.1"};
  std::string error;
  router->port = port;
  router->worker = gufo::router::Worker::Spawn(spec, gufo_exe, &error);
  if (router->worker == nullptr) {
    std::cerr << "cannot start " << gufo_exe << ": " << error << "\n";
    return false;
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (std::chrono::steady_clock::now() < deadline && !router->Gone()) {
    if (router->worker->Healthy())
      return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  std::cerr << "router on " << port << " did not come up\n";
  return false;
}

void WritePreset(const std::filesystem::path& path,
                 const std::filesystem::path& self, const char* where,
                 int partner_port) {
  std::ofstream out(path, std::ios::binary);
  out << "[external/twin]\ncommand = " << self.string()
      << "\nargs = --fake-worker --where " << where << "\n\n"
      << "[remote/twin]\nurl = http://127.0.0.1:" << partner_port
      << "\nqueue-timeout-seconds = 20\nplacement = pool\n";
}

}  // namespace

int main(int argc, char** argv) {
  net::EnsureSocketRuntimeStarted();
  const std::span<char*> args(argv, static_cast<std::size_t>(argc));
  for (char* arg : args)
    if (std::string_view(arg) == "--fake-worker")
      return ServeFakeWorker(args);
  if (argc < 2) {
    std::cerr << "usage: router_pool_drain_test <path to gufo>\n";
    return 2;
  }
  const std::string gufo_exe = argv[1];
  std::string error;
  const int port_a = gufo::router::ReserveLoopbackPort(&error);
  const int port_b = gufo::router::ReserveLoopbackPort(&error);
  if (port_a <= 0 || port_b <= 0 || port_a == port_b) {
    std::cerr << "no loopback ports: " << error << "\n";
    return 1;
  }
  const std::filesystem::path self =
      std::filesystem::absolute(std::filesystem::path(argv[0]));
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() /
      ("gufo_pool_drain_" + std::to_string(port_a));
  std::filesystem::create_directories(dir);
  const std::filesystem::path preset_a = dir / "a.preset";
  const std::filesystem::path preset_b = dir / "b.preset";
  WritePreset(preset_a, self, "a", port_b);
  WritePreset(preset_b, self, "b", port_a);

  Router a;
  Router b;
  if (!StartRouter(&a, gufo_exe, preset_a, port_a) ||
      !StartRouter(&b, gufo_exe, preset_b, port_b)) {
    std::_Exit(1);
  }

  // A fresh router is not draining.
  Reply r = Ask(a.port, "GET", "/router/pool/drain");
  CHECK(r.status == 200);
  CHECK(!Bool(Parse(r), "an", true));

  // A has the twin warm; B's probe (every 5 s) sees it warm and free.
  r = Ask(a.port, "POST", "/models/load", "{\"model\":\"twin\"}");
  CHECK(r.status == 200);
  {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(20);
    bool loaded = false;
    while (!loaded && std::chrono::steady_clock::now() < deadline) {
      loaded = Ask(a.port, "GET", "/v1/models").body.find(
                   "\"value\":\"loaded\"") != std::string::npos;
      if (!loaded)
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    CHECK(loaded);
  }
  std::this_thread::sleep_for(std::chrono::seconds(11));

  // Before: B, cold itself, sends the chat to A (warm and free there).
  r = Ask(b.port, "POST", "/v1/chat/completions", kChat);
  CHECK(r.status == 200);
  CHECK(r.body.find("\"where\":\"a\"") != std::string::npos);
  if (r.body.find("\"where\":\"a\"") == std::string::npos)
    std::cerr << "before drain, B answered " << r.status << ": " << r.body
              << "\n";

  // Drain on (loopback: admin).
  r = Ask(a.port, "POST", "/router/pool/drain",
          "{\"an\":true,\"grund\":\"pause test\"}");
  CHECK(r.status == 200);
  {
    const json::Value state = Parse(r);
    CHECK(Bool(state, "an", false));
    CHECK(state.member_str("grund") == "pause test");
  }

  // B at once (its probe still shows A warm and free): B relays, A
  // refuses with 503 pool_busy, B runs it itself; the client sees 200.
  r = Ask(b.port, "POST", "/v1/chat/completions", kChat);
  CHECK(r.status == 200);
  CHECK(r.body.find("\"where\":\"b\"") != std::string::npos);
  if (r.body.find("\"where\":\"b\"") == std::string::npos)
    std::cerr << "while draining, B answered " << r.status << ": " << r.body
              << "\n";
  const long long refused_after_b = static_cast<long long>(
      Parse(Ask(a.port, "GET", "/router/pool/drain"))
          .member_double("abgelehnt", -1));
  // 1 = B relayed and was refused; 0 = B's probe had already seen it.
  CHECK(refused_after_b == 0 || refused_after_b == 1);

  // What the partner gets for a relayed POST: 503 pool_busy, Retry-After.
  r = Ask(a.port, "POST", "/v1/chat/completions", kChat,
          {{"X-Gufo-Forwarded", "r-partner"}});
  CHECK(r.status == 503);
  CHECK(ErrorCode(r) == "pool_busy");
  CHECK(r.retry_after == "30");
  CHECK(static_cast<long long>(
            Parse(Ask(a.port, "GET", "/router/pool/drain"))
                .member_double("abgelehnt", -1)) == refused_after_b + 1);
  // Job reads and the pool's own paths still pass for the partner.
  r = Ask(a.port, "GET", "/v1/models", "", {{"X-Gufo-Forwarded", "r-p"}});
  CHECK(r.status == 200);

  // A reports the drain to the partner's probe and capabilities.
  r = Ask(a.port, "GET", "/v1/models");
  CHECK(r.body.find("\"draining\":true") != std::string::npos);
  {
    const json::Value caps = Parse(Ask(a.port, "GET", "/router/capabilities"));
    CHECK(Bool(caps, "draining", false));
    const json::Value* models = caps.find("models");
    const json::Value* twin =
        models != nullptr ? models->find("twin") : nullptr;
    CHECK(twin != nullptr && !Bool(*twin, "available", true));
    CHECK(twin != nullptr && twin->member_str("unavailable") == "draining");
  }
  {
    const json::Value status = Parse(Ask(a.port, "GET", "/router/status"));
    const json::Value* drain = status.find("pool_drain");
    CHECK(drain != nullptr && Bool(*drain, "an", false));
  }

  // A's own requests run as before.
  r = Ask(a.port, "POST", "/v1/chat/completions", kChat);
  CHECK(r.status == 200);
  CHECK(r.body.find("\"where\":\"a\"") != std::string::npos);

  // Bad input, wrong method.
  r = Ask(a.port, "POST", "/router/pool/drain", "{\"an\":\"ja\"}");
  CHECK(r.status == 400);
  CHECK(ErrorCode(r) == "invalid_drain");
  r = Ask(a.port, "PUT", "/router/pool/drain", "{\"an\":false}");
  CHECK(r.status == 405);

  // Drain off: relayed work is taken again.
  r = Ask(a.port, "POST", "/router/pool/drain", "{\"an\":false}");
  CHECK(r.status == 200);
  CHECK(!Bool(Parse(r), "an", true));
  r = Ask(a.port, "POST", "/v1/chat/completions", kChat,
          {{"X-Gufo-Forwarded", "r-partner"}});
  CHECK(r.status == 200);
  CHECK(r.body.find("\"where\":\"a\"") != std::string::npos);

  // Drain on again, then a restart of A: it starts with drain off.
  r = Ask(a.port, "POST", "/router/pool/drain", "{\"an\":true}");
  CHECK(Bool(Parse(r), "an", false));
  CHECK(!a.Gone() && !b.Gone());
  a.worker->Stop(std::chrono::seconds(5));
  a.worker.reset();
  Router again;
  if (StartRouter(&again, gufo_exe, preset_a, port_a)) {
    r = Ask(again.port, "GET", "/router/pool/drain");
    CHECK(r.status == 200);
    CHECK(!Bool(Parse(r), "an", true));
    CHECK(!Bool(Parse(Ask(again.port, "GET", "/router/capabilities")),
                "draining", true));
    again.worker->Stop(std::chrono::seconds(5));
  } else {
    CHECK(false);
  }
  b.worker->Stop(std::chrono::seconds(5));
  std::error_code ignored;
  std::filesystem::remove_all(dir, ignored);
  if (failures != 0) {
    std::cerr << failures << " router pool drain check(s) failed\n";
    std::cerr.flush();
    std::_Exit(1);
  }
  std::cout << "All router pool drain tests passed.\n";
  std::cout.flush();
  std::_Exit(0);
}
