// How the router relay ends a streamed worker answer (15.11.: two qwen
// streams ended mid-answer on the .56 although the router logged
// "completed" and the client got a clean chunked end without [DONE]).
//
// A real router front (HttpServer with the router dispatcher) relays to an
// in-process fake worker. The client reads the raw front bytes, so a clean
// chunked end ("0\r\n\r\n") and an error event are visible as such.
//
// Usage: router_proxy_stream_end_test [--stress <tokens per stream>]
// Portable (Winsock and POSIX). Prints one line per check; exits 1 when one
// failed.
// Written with AI assistance (Claude Code).

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/proxy.hpp"
#include "src/core/platform/net.hpp"

namespace {

namespace net = gufo::net;
using gufo::router::ProxyToWorker;
using gufo::router::UpstreamTarget;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;
using gufo::server::HttpServer;
using gufo::server::HttpServerOptions;

int failures = 0;

void Check(bool ok, std::string_view what, std::string_view detail = {}) {
  std::cout << (ok ? "PASS " : "FAIL ") << what;
  if (!ok && !detail.empty())
    std::cout << "  [" << detail << "]";
  std::cout << std::endl;
  if (!ok)
    ++failures;
}

void SendAll(net::Socket fd, std::string_view data) {
  while (!data.empty()) {
    const auto count = net::SendNoSignal(fd, data.data(), data.size());
    if (count <= 0)
      return;  // the relay may have closed already
    data.remove_prefix(static_cast<std::size_t>(count));
  }
}

std::string Hex(std::size_t value) {
  static const char* digits = "0123456789abcdef";
  std::string out;
  do {
    out.insert(out.begin(), digits[value % 16]);
    value /= 16;
  } while (value != 0);
  return out;
}

/// One chunk the way `gufo serve` writes it: size line, data and CRLF as
/// three separate sends (SendChunk in http_server.cpp).
void SendWorkerChunk(net::Socket fd, std::string_view data) {
  SendAll(fd, Hex(data.size()) + "\r\n");
  SendAll(fd, data);
  SendAll(fd, "\r\n");
}

std::string Token(int index) {
  return "data: {\"choices\":[{\"delta\":{\"content\":\"t" +
         std::to_string(index) + " \"},\"finish_reason\":null}]}\n\n";
}

const char* kSseHead =
    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
    "Cache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n"
    "Connection: close\r\n\r\n";

/// In-process fake worker: accepts connections on an ephemeral loopback
/// port, reads one request each and hands the socket to `responder`.
class FakeWorker {
public:
  explicit FakeWorker(std::function<void(net::Socket)> responder,
                      int connections = 1)
      : responder_(std::move(responder)) {
    listen_ = net::CreateStreamSocket(AF_INET);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = 0;
    ::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
    if (::bind(listen_, reinterpret_cast<const sockaddr*>(&address),
               sizeof(address)) != 0 ||
        ::listen(listen_, 16) != 0) {
      std::cerr << "fake worker: bind/listen failed\n";
      std::exit(2);
    }
    net::SocklenT length = sizeof(address);
    ::getsockname(listen_, reinterpret_cast<sockaddr*>(&address), &length);
    port_ = ntohs(address.sin_port);
    for (int i = 0; i < connections; ++i)
      threads_.emplace_back([this] { ServeOne(); });
  }
  ~FakeWorker() {
    for (auto& thread : threads_)
      thread.join();
    net::CloseSocket(listen_);
  }
  FakeWorker(const FakeWorker&) = delete;
  FakeWorker& operator=(const FakeWorker&) = delete;
  [[nodiscard]] int port() const { return port_; }

private:
  void ServeOne() {
    const net::Socket fd = net::AcceptClientSocket(listen_);
    if (fd == net::kInvalidSocket)
      return;
    const int on = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY,
                 reinterpret_cast<const char*>(&on), sizeof(on));
    if (ReadRequest(fd))
      responder_(fd);
    net::CloseSocket(fd);
  }
  static bool ReadRequest(net::Socket fd) {
    std::string data;
    char buffer[4096];
    std::size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
      const auto count = net::Receive(fd, buffer, sizeof(buffer));
      if (count <= 0)
        return false;
      data.append(buffer, static_cast<std::size_t>(count));
      header_end = data.find("\r\n\r\n");
    }
    std::size_t length = 0;
    const auto at = data.find("Content-Length: ");
    if (at != std::string::npos && at < header_end)
      length = std::strtoul(data.c_str() + at + 16, nullptr, 10);
    while (data.size() < header_end + 4 + length) {
      const auto count = net::Receive(fd, buffer, sizeof(buffer));
      if (count <= 0)
        return false;
      data.append(buffer, static_cast<std::size_t>(count));
    }
    return true;
  }

  std::function<void(net::Socket)> responder_;
  net::Socket listen_{net::kInvalidSocket};
  int port_{0};
  std::vector<std::thread> threads_;
};

/// The router front: `gufo router`'s HttpServer with a dispatcher that
/// relays every request to the fake worker, as the router does for a loaded
/// local model.
class Front {
public:
  explicit Front(int worker_port) {
    HttpServerOptions options;
    options.dispatcher = [worker_port](const HttpRequest& request) {
      UpstreamTarget target;
      target.port = worker_port;
      return ProxyToWorker(target, request);
    };
    options.graceful_shutdown = true;
    server_ = std::make_unique<HttpServer>("127.0.0.1", 0, nullptr, nullptr,
                                           nullptr, nullptr, options);
    std::string error;
    if (!server_->start(&error)) {
      std::cerr << "front: " << error << '\n';
      std::exit(2);
    }
    thread_ = std::thread([this] { server_->run(false); });
  }
  ~Front() {
    server_->stop();
    thread_.join();
  }
  Front(const Front&) = delete;
  Front& operator=(const Front&) = delete;

  /// One streamed chat request; returns the raw response bytes up to the
  /// connection close.
  [[nodiscard]] std::string Post() const {
    const net::Socket fd = net::CreateStreamSocket(AF_INET);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<unsigned short>(server_->port()));
    ::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
    (void)net::SetSocketTimeouts(fd, 20'000);
    if (::connect(fd, reinterpret_cast<const sockaddr*>(&address),
                  sizeof(address)) != 0) {
      net::CloseSocket(fd);
      return {};
    }
    const std::string body =
        R"({"model":"fake","stream":true,"messages":[]})";
    SendAll(fd, "POST /v1/chat/completions HTTP/1.1\r\nHost: front\r\n"
                "Content-Type: application/json\r\nContent-Length: " +
                    std::to_string(body.size()) + "\r\n\r\n" + body);
    std::string response;
    char buffer[8192];
    for (;;) {
      const auto count = net::Receive(fd, buffer, sizeof(buffer));
      if (count <= 0)
        break;
      response.append(buffer, static_cast<std::size_t>(count));
    }
    net::CloseSocket(fd);
    return response;
  }

private:
  std::unique_ptr<HttpServer> server_;
  std::thread thread_;
};

/// The client's view of a chunked front response.
struct Decoded {
  std::string head;
  std::string body;
  bool clean_end{false};  // final 0-size chunk received
};

Decoded Decode(const std::string& raw) {
  Decoded out;
  const auto head_end = raw.find("\r\n\r\n");
  if (head_end == std::string::npos)
    return out;
  out.head = raw.substr(0, head_end);
  std::size_t at = head_end + 4;
  for (;;) {
    const auto line_end = raw.find("\r\n", at);
    if (line_end == std::string::npos)
      return out;
    const std::size_t size =
        std::strtoul(raw.substr(at, line_end - at).c_str(), nullptr, 16);
    if (size == 0) {
      out.clean_end = raw.compare(line_end, 4, "\r\n\r\n") == 0;
      return out;
    }
    if (raw.size() < line_end + 2 + size + 2)
      return out;
    out.body.append(raw, line_end + 2, size);
    at = line_end + 2 + size + 2;
  }
}

std::string Shorten(std::string text) {
  if (text.size() > 160)
    text = "..." + text.substr(text.size() - 160);
  for (char& c : text)
    if (c == '\n' || c == '\r')
      c = '|';
  return text;
}

void TestCleanStream() {
  FakeWorker worker([](net::Socket fd) {
    SendAll(fd, kSseHead);
    for (int i = 0; i < 3; ++i)
      SendWorkerChunk(fd, Token(i));
    SendWorkerChunk(fd, "data: [DONE]\n\n");
    SendAll(fd, "0\r\n\r\n");
  });
  Front front(worker.port());
  const Decoded seen = Decode(front.Post());
  const std::string want =
      Token(0) + Token(1) + Token(2) + "data: [DONE]\n\n";
  Check(seen.clean_end && seen.body == want,
        "complete worker stream: relayed exactly, clean chunked end",
        Shorten(seen.body));
}

void TestWorkerClosesMidStream() {
  FakeWorker worker([](net::Socket fd) {
    SendAll(fd, kSseHead);
    SendWorkerChunk(fd, Token(0));
    SendWorkerChunk(fd, Token(1));
    // The worker connection ends without [DONE] and without the final chunk.
  });
  Front front(worker.port());
  const Decoded seen = Decode(front.Post());
  Check(seen.body.starts_with(Token(0) + Token(1)),
        "worker closed mid-stream: tokens before the break are relayed");
  Check(!seen.clean_end,
        "worker closed mid-stream: client gets NO clean chunked end",
        "front sent 0\\r\\n\\r\\n");
  Check(seen.body.find("\"upstream_closed\"") != std::string::npos,
        "worker closed mid-stream: client gets an SSE error event",
        Shorten(seen.body));
}

void TestWorkerResetMidStream() {
  FakeWorker worker([](net::Socket fd) {
    SendAll(fd, kSseHead);
    SendWorkerChunk(fd, Token(0));
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    // Abortive close: RST instead of FIN (recv fails with a reset error).
    const linger hard{1, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_LINGER,
                 reinterpret_cast<const char*>(&hard), sizeof(hard));
  });
  Front front(worker.port());
  const Decoded seen = Decode(front.Post());
  Check(!seen.clean_end, "worker reset mid-stream: NO clean chunked end");
  Check(seen.body.find("\"upstream_") != std::string::npos,
        "worker reset mid-stream: SSE error event", Shorten(seen.body));
}

void TestCorruptFraming() {
  FakeWorker worker([](net::Socket fd) {
    SendAll(fd, kSseHead);
    SendWorkerChunk(fd, Token(0));
    // A chunk boundary that does not line up: data where a size belongs.
    SendAll(fd, "data: lost-framing\r\n\r\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  });
  Front front(worker.port());
  const Decoded seen = Decode(front.Post());
  Check(!seen.clean_end, "corrupt chunk framing: NO clean chunked end");
  Check(seen.body.find("\"upstream_bad_framing\"") != std::string::npos,
        "corrupt chunk framing: SSE error event names it", Shorten(seen.body));
}

void TestContentLengthShort() {
  FakeWorker worker([](net::Socket fd) {
    SendAll(fd,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            "Content-Length: 100\r\n\r\n{\"partial\":");
  });
  UpstreamTarget target;
  target.port = worker.port();
  HttpRequest request;
  request.method = "POST";
  request.path = "/v1/chat/completions";
  request.body = "{}";
  const HttpResponse response = ProxyToWorker(target, request);
  std::string collected;
  response.streaming_body([&](std::string_view chunk) {
    collected.append(chunk);
    return true;
  });
  const bool marked = response.stream_log != nullptr &&
                      response.stream_log->error_code == "upstream_closed";
  Check(marked, "short Content-Length body: relay marks upstream_closed");
  Check(collected == "{\"partial\":",
        "short Content-Length body: no SSE event mixed into JSON",
        Shorten(collected));
}

void TestClientCancelled() {
  FakeWorker worker([](net::Socket fd) {
    SendAll(fd, kSseHead);
    SendWorkerChunk(fd, Token(0));
    char byte = 0;
    (void)net::Receive(fd, &byte, 1);  // until the relay closes
  });
  UpstreamTarget target;
  target.port = worker.port();
  HttpRequest request;
  request.method = "POST";
  request.path = "/v1/chat/completions";
  request.body = "{}";
  std::atomic<bool> gone{false};
  request.is_cancelled = [&gone] { return gone.load(); };
  const HttpResponse response = ProxyToWorker(target, request);
  std::string collected;
  response.streaming_body([&](std::string_view chunk) {
    collected.append(chunk);
    gone = true;
    return true;
  });
  const bool marked = response.stream_log != nullptr &&
                      response.stream_log->error_code == "client_disconnected";
  Check(marked, "client cancelled: relay marks client_disconnected");
  Check(collected == Token(0), "client cancelled: nothing written afterwards",
        Shorten(collected));
}

/// Many streams at once, chunks split into three sends with gaps around the
/// relay's 50 ms poll slice: every byte must arrive in order and every
/// stream must end cleanly.
void TestStress(int tokens, int streams) {
  FakeWorker worker(
      [tokens](net::Socket fd) {
        std::mt19937 random(static_cast<unsigned>(fd) * 7919u);
        std::uniform_int_distribution<int> gap(0, 110);
        SendAll(fd, kSseHead);
        for (int i = 0; i < tokens; ++i) {
          const std::string data = Token(i);
          SendAll(fd, Hex(data.size()) + "\r\n");
          if (i % 7 == 3)
            std::this_thread::sleep_for(std::chrono::milliseconds(gap(random)));
          const auto half = data.size() / 2;
          SendAll(fd, std::string_view(data).substr(0, half));
          if (i % 5 == 1)
            std::this_thread::sleep_for(std::chrono::milliseconds(gap(random)));
          SendAll(fd, std::string_view(data).substr(half));
          SendAll(fd, "\r\n");
          const int wait = gap(random);
          if (wait > 20)
            std::this_thread::sleep_for(std::chrono::milliseconds(wait - 20));
        }
        SendWorkerChunk(fd, "data: [DONE]\n\n");
        SendAll(fd, "0\r\n\r\n");
      },
      streams);
  Front front(worker.port());
  std::string want;
  for (int i = 0; i < tokens; ++i)
    want += Token(i);
  want += "data: [DONE]\n\n";
  std::vector<std::thread> clients;
  std::atomic<int> intact{0};
  std::atomic<int> clean{0};
  for (int s = 0; s < streams; ++s) {
    clients.emplace_back([&] {
      const Decoded seen = Decode(front.Post());
      if (seen.body == want)
        ++intact;
      if (seen.clean_end)
        ++clean;
    });
  }
  for (auto& client : clients)
    client.join();
  Check(intact.load() == streams && clean.load() == streams,
        "stress " + std::to_string(streams) + "x" + std::to_string(tokens) +
            " tokens: every stream intact and clean",
        "intact " + std::to_string(intact.load()) + ", clean " +
            std::to_string(clean.load()));
}

}  // namespace

// 2.2.13 (Regie #56-1287): a worker whose --max-request-bytes the body
// exceeds answers 413 with size and limit after reading the body away; the
// relay hands that 413 on instead of "502 receive failed (10054)". With
// body_limit the router's own calls get a larger limit.
void TestWorkerRefusesLargeBody() {
  HttpServerOptions worker_options;
  worker_options.max_request_body_bytes = 8192;
  worker_options.body_limit = [](const HttpRequest& head) {
    return head.header("X-Big") == "1" ? std::size_t{8} << 20
                                       : std::size_t{8192};
  };
  worker_options.dispatcher = [](const HttpRequest& request) {
    HttpResponse response;
    response.body = std::to_string(request.body.size());
    return response;
  };
  auto worker = std::make_unique<HttpServer>("127.0.0.1", 0, nullptr, nullptr,
                                             nullptr, nullptr, worker_options);
  std::string error;
  if (!worker->start(&error)) {
    std::cerr << "worker: " << error << std::endl;
    std::exit(2);
  }
  std::thread run([&] { worker->run(false); });
  const std::string body(3 << 20, 'x');  // 3 MiB, as a base64 video
  UpstreamTarget target;
  target.port = worker->port();
  HttpRequest request;
  request.method = "POST";
  request.path = "/v1/videos";
  request.body = body;
  request.headers.emplace_back("Content-Type", "application/json");
  HttpResponse refused = ProxyToWorker(target, request);
  std::string refused_body = refused.body;
  if (refused.streaming_body)
    refused.streaming_body([&](std::string_view chunk) {
      refused_body.append(chunk);
      return true;
    });
  Check(refused.status == 413, "worker 413 reaches the client as 413",
        std::to_string(refused.status) + " " + refused_body);
  Check(refused_body.find("3145728 bytes, limit 8192") != std::string::npos,
        "413 names size and limit", refused_body);
  request.headers.emplace_back("X-Big", "1");
  HttpResponse accepted = ProxyToWorker(target, request);
  std::string accepted_body = accepted.body;
  if (accepted.streaming_body)
    accepted.streaming_body([&](std::string_view chunk) {
      accepted_body.append(chunk);
      return true;
    });
  Check(accepted.status == 200 && accepted_body == std::to_string(body.size()),
        "body_limit lets the larger call through",
        std::to_string(accepted.status) + " " + accepted_body);
  worker->stop();
  run.join();
}

int main(int argc, char** argv) {
  net::EnsureSocketRuntimeStarted();
  int stress_tokens = 60;
  for (int i = 1; i + 1 < argc; ++i)
    if (std::string_view(argv[i]) == "--stress")
      stress_tokens = std::atoi(argv[i + 1]);
  TestCleanStream();
  TestWorkerClosesMidStream();
  TestWorkerResetMidStream();
  TestCorruptFraming();
  TestContentLengthShort();
  TestClientCancelled();
  TestWorkerRefusesLargeBody();
  TestStress(stress_tokens, 6);
  std::cout << (failures == 0 ? "All router relay end tests passed."
                              : "FAILED: " + std::to_string(failures))
            << std::endl;
  return failures == 0 ? 0 : 1;
}
