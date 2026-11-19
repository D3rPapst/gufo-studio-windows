// Finished video/3D jobs answered from the worker's job folder (2.2.13).
//
// Part 1 (always): router/job_files.cpp on a temporary job folder: id
// patterns, the preset's `root`, status/content/preview answers, Range,
// failed and expired jobs, unsafe names.
//
// Part 2 (with the path to `gufo` as argv[1]): `gufo router` as a child
// process with users (auth = lan, loopback must log in). Alice creates a
// video on a fake worker (this binary, --fake-worker), the worker is
// unloaded, then the job folder gets a finished job.json and a small MP4.
// Status and /content come from the folder; the worker is never started
// again and never sees a job read. Bob (another account) gets 404, the
// admin may read it, ids of the wrong shape are 404.
//
// Regression 19.11. (2.2.11, .198): a /content read of a finished
// wan2.2-i2v-a14b clip reloaded the evicted 54.8 GB model and the client
// got memory_busy.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/router/auth.hpp"
#include "src/cli/serve/router/job_files.hpp"
#include "src/cli/serve/router/preset.hpp"
#include "src/cli/serve/router/proxy.hpp"
#include "src/cli/serve/router/workers.hpp"
#include "src/core/platform/net.hpp"

namespace {

namespace fs = std::filesystem;
namespace net = gufo::net;
namespace jf = gufo::router::job_files;
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

constexpr const char* kVideoId = "video_0123456789abcdef01234567";
constexpr const char* kObjektId = "obj_0123456789abcdef01234567";
const std::string kMp4("\x00\x00\x00\x18" "ftypmp42 fake clip bytes", 28);

long long Now() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

void WriteFile(const fs::path& path, std::string_view text) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::string ReadText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::string VideoJobJson(const std::string& status, long long expires_at) {
  const long long now = Now();
  std::string json = "{\"schema\":\"gufo.video-job.v1\",\"id\":\"" +
                     std::string(kVideoId) + "\",\"status\":\"" + status +
                     "\",\"progress\":" + (status == "completed" ? "100" : "1") +
                     ",\"created_at\":" + std::to_string(now - 100) +
                     ",\"completed_at\":" + std::to_string(now - 10) +
                     ",\"expires_at\":" + std::to_string(expires_at) +
                     ",\"model\":\"clip\",\"size\":\"896x512\",\"seconds\":"
                     "\"2\",\"output_format\":\"mp4\",\"error\":";
  json += status == "failed" ? "{\"code\":\"generation_failed\",\"message\":"
                               "\"video generation failed\"}}"
                             : "null}";
  return json + "\n";
}

int Count(const fs::path& file) {
  int lines = 0;
  for (char c : ReadText(file))
    lines += c == '\n' ? 1 : 0;
  return lines;
}

std::string Header(const HttpResponse& response, std::string_view name) {
  for (const auto& [key, value] : response.headers) {
    if (key.size() != name.size())
      continue;
    bool same = true;
    for (std::size_t i = 0; i < key.size(); ++i)
      same = same && std::tolower(static_cast<unsigned char>(key[i])) ==
                         std::tolower(static_cast<unsigned char>(name[i]));
    if (same)
      return value;
  }
  return "";
}

// ---------------------------------------------------------------- part 1

void UnitTests(const fs::path& base) {
  // Id patterns and paths.
  CHECK(jf::IsVideoId(kVideoId));
  CHECK(!jf::IsVideoId("video_0123456789ABCDEF01234567"));
  CHECK(!jf::IsVideoId("video_0123456789abcdef0123456"));
  CHECK(!jf::IsVideoId("video_..\\..\\..\\windows\\win.ini"));
  CHECK(jf::IsObjektId(kObjektId));
  CHECK(!jf::IsObjektId("obj_0123456789abcdef0123456g"));
  CHECK(jf::ParseRead("GET", std::string("/v1/videos/") + kVideoId)
            .has_value());
  const auto content =
      jf::ParseRead("GET", std::string("/v1/videos/") + kVideoId + "/content");
  CHECK(content.has_value() && content->part == "content" &&
        content->kind == jf::Kind::kVideo);
  CHECK(!jf::ParseRead("DELETE", std::string("/v1/videos/") + kVideoId));
  CHECK(!jf::ParseRead("GET", std::string("/v1/videos/") + kVideoId +
                                  "/preview"));
  CHECK(!jf::ParseRead("GET", "/v1/videos/video_../content"));
  CHECK(!jf::ParseRead("GET", "/v1/videos/../../x/content"));
  CHECK(!jf::ParseRead("GET", "/v1/videos/upscale"));
  const auto preview = jf::ParseRead(
      "GET", std::string("/v1/3d/generations/") + kObjektId + "/preview");
  CHECK(preview.has_value() && preview->kind == jf::Kind::kObjekt &&
        preview->part == "preview");
  CHECK(!jf::ParseRead("GET", std::string("/v1/3d/generations/") + kObjektId +
                                  "/content/x"));

  // The preset's root: only `serve video` / `serve objekt`.
  const fs::path dir = base / "unit";
  fs::create_directories(dir);
  const fs::path exe = dir / "fake.exe";
  WriteFile(exe, "x");
  const fs::path preset_file = dir / "unit.preset";
  WriteFile(preset_file,
            "[external/clip]\ncommand = " + exe.string() +
                "\nargs = serve video\nroot = " + (dir / "jobs").string() +
                "\n\n[external/figur]\ncommand = " + exe.string() +
                "\nargs = serve objekt\nroot = " + (dir / "obj").string() +
                "\n\n[external/hoch]\ncommand = " + exe.string() +
                "\nargs = serve upscale\nroot = " + (dir / "up").string() +
                "\n\n[external/ohne]\ncommand = " + exe.string() +
                "\nargs = serve video\n");
  std::vector<gufo::router::PresetModel> presets;
  std::string error;
  CHECK(gufo::router::LoadPreset(preset_file, {}, &presets, &error));
  if (presets.size() == 4) {
    const auto video_root = jf::JobRootOf(presets[0], jf::Kind::kVideo);
    CHECK(video_root.has_value() && *video_root == dir / "jobs");
    CHECK(!jf::JobRootOf(presets[0], jf::Kind::kObjekt).has_value());
    const auto objekt_root = jf::JobRootOf(presets[1], jf::Kind::kObjekt);
    CHECK(objekt_root.has_value() && *objekt_root == dir / "obj");
    CHECK(!jf::JobRootOf(presets[2], jf::Kind::kVideo).has_value());
    CHECK(!jf::JobRootOf(presets[3], jf::Kind::kVideo).has_value());
  } else {
    std::cerr << "preset: " << error << " (" << presets.size() << ")\n";
    ++failures;
  }

  // Reads.
  const fs::path root = dir / "jobs";
  const fs::path job = root / kVideoId;
  HttpRequest get;
  get.method = "GET";
  jf::Target status{jf::Kind::kVideo, kVideoId, ""};
  jf::Target clip{jf::Kind::kVideo, kVideoId, "content"};
  CHECK(!jf::ReadFinished(root, status, get, Now()).has_value());  // no folder
  WriteFile(job / "job.json",
            "{\"schema\":\"gufo.video-job.v1\",\"id\":\"" +
                std::string(kVideoId) +
                "\",\"status\":\"in_progress\",\"progress\":40}");
  CHECK(!jf::ReadFinished(root, status, get, Now()).has_value());  // running
  WriteFile(job / "job.json", VideoJobJson("completed", Now() + 3600));
  WriteFile(job / "output.mp4", kMp4);
  auto answer = jf::ReadFinished(root, status, get, Now());
  CHECK(answer.has_value() && answer->status == 200);
  if (answer.has_value()) {
    CHECK(answer->body.find("\"object\":\"video\"") != std::string::npos);
    CHECK(answer->body.find("\"schema\":\"gufo.video-api.v1\"") !=
          std::string::npos);
    CHECK(answer->body.find("\"status\":\"completed\"") != std::string::npos);
    CHECK(answer->body.find("\"error\":null") != std::string::npos);
  }
  answer = jf::ReadFinished(root, clip, get, Now());
  CHECK(answer.has_value() && answer->status == 200 && answer->body == kMp4);
  if (answer.has_value()) {
    CHECK(Header(*answer, "Content-Type") == "video/mp4");
    CHECK(Header(*answer, "Accept-Ranges") == "bytes");
  }
  HttpRequest ranged = get;
  ranged.headers.emplace_back("Range", "bytes=4-7");
  answer = jf::ReadFinished(root, clip, ranged, Now());
  CHECK(answer.has_value() && answer->status == 206 && answer->body == "ftyp");
  if (answer.has_value())
    CHECK(Header(*answer, "Content-Range") ==
          "bytes 4-7/" + std::to_string(kMp4.size()));
  HttpRequest bad_range = get;
  bad_range.headers.emplace_back("Range", "bytes=999-");
  answer = jf::ReadFinished(root, clip, bad_range, Now());
  CHECK(answer.has_value() && answer->status == 416);
  // The id inside job.json must match the folder.
  WriteFile(job / "job.json",
            "{\"schema\":\"gufo.video-job.v1\",\"id\":\"video_"
            "ffffffffffffffffffffffff\",\"status\":\"completed\"}");
  CHECK(!jf::ReadFinished(root, status, get, Now()).has_value());
  // Clip gone (removed by hand, or the disk was cleaned): 404 like a
  // reloaded worker.
  WriteFile(job / "job.json", VideoJobJson("completed", Now() + 3600));
  fs::remove(job / "output.mp4");
  answer = jf::ReadFinished(root, clip, get, Now());
  CHECK(answer.has_value() && answer->status == 404 &&
        answer->body.find("video_not_found") != std::string::npos);
  // Failed: status answers, content is 409 like the worker.
  WriteFile(job / "job.json", VideoJobJson("failed", Now() + 3600));
  answer = jf::ReadFinished(root, status, get, Now());
  CHECK(answer.has_value() && answer->status == 200 &&
        answer->body.find("generation_failed") != std::string::npos);
  answer = jf::ReadFinished(root, clip, get, Now());
  CHECK(answer.has_value() && answer->status == 409);
  // Expired: not answered here (the caller gives 404 or asks the worker).
  WriteFile(job / "job.json", VideoJobJson("completed", Now() - 5));
  WriteFile(job / "output.mp4", kMp4);
  CHECK(!jf::ReadFinished(root, clip, get, Now()).has_value());
  // An id of the wrong shape is never used as a path.
  CHECK(!jf::ReadFinished(root, {jf::Kind::kVideo, "..", "content"}, get, Now())
             .has_value());
  CHECK(!jf::ReadFinished(root, {jf::Kind::kVideo, "video_x", ""}, get, Now())
             .has_value());

  // 3D: the stored job object, model.glb and preview.png.
  const fs::path obj_root = dir / "obj";
  const fs::path obj = obj_root / kObjektId;
  WriteFile(obj / "job.json",
            "{\"id\":\"" + std::string(kObjektId) +
                "\",\"object\":\"3d.generation\",\"schema\":\"gufo.3d-api.v1\","
                "\"model\":\"figur\",\"status\":\"completed\",\"progress\":100,"
                "\"expires_at\":" +
                std::to_string(Now() + 3600) + ",\"data\":[]}");
  WriteFile(obj / "model.glb", "glTF-bytes");
  WriteFile(obj / "preview.png", "png-bytes");
  answer = jf::ReadFinished(obj_root, {jf::Kind::kObjekt, kObjektId, ""}, get,
                            Now());
  CHECK(answer.has_value() && answer->status == 200 &&
        answer->body.find("3d.generation") != std::string::npos);
  answer = jf::ReadFinished(obj_root, {jf::Kind::kObjekt, kObjektId, "content"},
                            get, Now());
  CHECK(answer.has_value() && answer->status == 200 &&
        answer->body == "glTF-bytes");
  if (answer.has_value())
    CHECK(Header(*answer, "Content-Type") == "model/gltf-binary");
  answer = jf::ReadFinished(obj_root, {jf::Kind::kObjekt, kObjektId, "preview"},
                            get, Now());
  CHECK(answer.has_value() && answer->status == 200 &&
        answer->body == "png-bytes");
  fs::remove(obj / "preview.png");
  answer = jf::ReadFinished(obj_root, {jf::Kind::kObjekt, kObjektId, "preview"},
                            get, Now());
  CHECK(answer.has_value() && answer->status == 404 &&
        answer->body.find("objekt_file_missing") != std::string::npos);
}

// ---------------------------------------------------------------- part 2

/// One loopback listener; each connection gets the raw request text and is
/// closed after `serve` answered it.
class Listener {
public:
  using Serve = std::function<std::string(const std::string& request)>;

  Listener(int port, Serve serve) : serve_(std::move(serve)) {
    listener_ = net::CreateStreamSocket(AF_INET);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<unsigned short>(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (listener_ == net::kInvalidSocket ||
        ::bind(listener_, reinterpret_cast<const sockaddr*>(&address),
               sizeof(address)) != 0 ||
        ::listen(listener_, 16) != 0) {
      std::cerr << "cannot listen on 127.0.0.1:" << port << "\n";
      std::_Exit(3);
    }
  }
  Listener(const Listener&) = delete;
  Listener& operator=(const Listener&) = delete;
  ~Listener() { net::CloseSocket(listener_); }

  /// Serves until the process ends; ends it after `idle_exit` without a
  /// connection (the router is gone).
  void Run(std::chrono::seconds idle_exit) {
    auto last = std::chrono::steady_clock::now();
    for (;;) {
      if (net::PollReadable(listener_, 100) <= 0) {
        if (std::chrono::steady_clock::now() - last > idle_exit)
          std::_Exit(0);
        continue;
      }
      last = std::chrono::steady_clock::now();
      const net::Socket client = net::AcceptClientSocket(listener_);
      if (client == net::kInvalidSocket)
        continue;
      const std::string text = serve_(ReadRequest(client));
      std::string_view answer = text;
      while (!answer.empty()) {
        const auto sent =
            net::SendNoSignal(client, answer.data(), answer.size());
        if (sent <= 0)
          break;
        answer.remove_prefix(static_cast<std::size_t>(sent));
      }
      net::CloseSocket(client);
    }
  }

private:
  static std::string ReadRequest(net::Socket client) {
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

  Serve serve_;
  net::Socket listener_{net::kInvalidSocket};
};

std::string Answer(int status, std::string_view reason, std::string_view body) {
  return "HTTP/1.1 " + std::to_string(status) + " " + std::string(reason) +
         "\r\nContent-Type: application/json\r\nContent-Length: " +
         std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" +
         std::string(body);
}

/// The video worker: notes its start in --starts, every request but
/// /health in --hits; a create answers the fixed job id.
int ServeFakeWorker(std::span<char*> args) {
  int port = 0;
  fs::path starts;
  fs::path hits;
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    const std::string_view arg(args[i]);
    if (arg == "--port")
      port = std::atoi(args[i + 1]);
    else if (arg == "--starts")
      starts = args[i + 1];
    else if (arg == "--hits")
      hits = args[i + 1];
  }
  if (port <= 0 || starts.empty() || hits.empty())
    return 2;
  {
    std::ofstream out(starts, std::ios::binary | std::ios::app);
    out << "start\n";
  }
  Listener worker(port, [hits](const std::string& request) {
    const std::string line = request.substr(0, request.find("\r\n"));
    if (line.find(" /health") == std::string::npos) {
      std::ofstream out(hits, std::ios::binary | std::ios::app);
      out << line << "\n";
    }
    if (line.starts_with("POST /v1/videos"))
      return Answer(202, "Accepted",
                    "{\"id\":\"" + std::string(kVideoId) +
                        "\",\"object\":\"video\",\"status\":\"queued\","
                        "\"model\":\"clip\"}");
    return Answer(200, "OK", "{\"status\":\"ok\"}");
  });
  worker.Run(std::chrono::seconds(60));
  return 0;
}

HttpResponse Ask(int port, const std::string& token, const std::string& method,
                 const std::string& path, const std::string& body = "",
                 const std::string& range = "") {
  UpstreamTarget target;
  target.host = "127.0.0.1";
  target.port = port;
  target.connect_timeout = std::chrono::seconds(2);
  target.head_timeout = std::chrono::seconds(60);
  HttpRequest request;
  request.method = method;
  request.path = path;
  request.body = body;
  // ProxyToWorker drops a client Authorization; extra_headers carry it.
  if (!token.empty())
    target.extra_headers.emplace_back("Authorization", "Bearer " + token);
  if (!body.empty())
    request.headers.emplace_back("Content-Type", "application/json");
  if (!range.empty())
    request.headers.emplace_back("Range", range);
  HttpResponse response = ProxyToWorker(target, request);
  if (response.streaming_body) {
    std::string streamed = response.body;
    response.streaming_body([&streamed](std::string_view chunk) {
      streamed.append(chunk);
      return true;
    });
    response.streaming_body = nullptr;
    response.body = std::move(streamed);
  }
  return response;
}

int EndToEnd(const std::string& gufo_exe, const fs::path& self,
             const fs::path& dir) {
  const fs::path root = dir / "video-jobs";
  const fs::path starts = dir / "starts.txt";
  const fs::path hits = dir / "hits.txt";
  const fs::path users = dir / "nutzer.json";
  fs::create_directories(root);

  // Users: alice and bob (nutzer), chef (admin), each with a token.
  std::string alice;
  std::string bob;
  std::string chef;
  {
    gufo::router::auth::UserStore store(users, 1000);
    std::string error;
    CHECK(store.Load(&error));
    for (const char* name : {"alice", "bob", "chef"}) {
      gufo::router::auth::User user;
      user.name = name;
      user.rolle = std::string_view(name) == "chef" ? "admin" : "nutzer";
      // The router counts an admin only with a password.
      CHECK(store.Create(user, "Testpasswort-123", &error));
      if (!error.empty())
        std::cerr << "user " << name << ": " << error << "\n";
    }
    alice = store.CreateToken("alice", "test", 0, &error);
    bob = store.CreateToken("bob", "test", 0, &error);
    chef = store.CreateToken("chef", "test", 0, &error);
    CHECK(!alice.empty() && !bob.empty() && !chef.empty());
  }

  const fs::path preset = dir / "content.preset";
  WriteFile(preset, "[router]\nauth = lan\nauth-loopback = login\nusers-file = " +
                        users.string() + "\n\n[external/clip]\ncommand = " +
                        self.string() +
                        "\nargs = serve video --fake-worker\nroot = " +
                        root.string() + "\nstarts = " + starts.string() +
                        "\nhits = " + hits.string() + "\n");

  std::string error;
  const int port = gufo::router::ReserveLoopbackPort(&error);
  if (port <= 0) {
    std::cerr << "no loopback port: " << error << "\n";
    return 1;
  }
  gufo::router::WorkerSpec spec;
  spec.model_id = "router";
  spec.port = port;
  spec.command = gufo_exe;
  spec.args = {"router", "--models-preset", preset.string(), "--models-max",
               "0"};
  std::unique_ptr<gufo::router::Worker> router =
      gufo::router::Worker::Spawn(spec, gufo_exe, &error);
  if (router == nullptr) {
    std::cerr << "cannot start " << gufo_exe << ": " << error << "\n";
    return 1;
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  bool healthy = false;
  while (!healthy && std::chrono::steady_clock::now() < deadline) {
    int exit_status = 0;
    if (router->PollExit(&exit_status)) {
      std::cerr << "router exited with status " << exit_status << "\n";
      return 1;
    }
    healthy = router->Healthy();
    if (!healthy)
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  CHECK(healthy);
  if (!healthy)
    return 1;

  const std::string status_path = std::string("/v1/videos/") + kVideoId;
  const std::string content_path = status_path + "/content";

  // Alice's clip: created on the worker (the router records the job and
  // the gate its owner).
  HttpResponse created =
      Ask(port, alice, "POST", "/v1/videos",
          "{\"model\":\"clip\",\"prompt\":\"ein Hund am Strand\"}");
  CHECK(created.status == 202);
  if (created.status != 202)
    std::cerr << "create " << created.status << ": " << created.body << "\n";
  CHECK(Count(starts) == 1);

  // The worker finished it (as gufo serve video writes it).
  WriteFile(root / kVideoId / "job.json", VideoJobJson("completed", Now() + 3600));
  WriteFile(root / kVideoId / "output.mp4", kMp4);

  // Status from the folder while the worker still runs: it never sees the
  // read, and the finished job no longer keeps the model busy.
  HttpResponse status = Ask(port, alice, "GET", status_path);
  CHECK(status.status == 200 &&
        status.body.find("\"status\":\"completed\"") != std::string::npos);
  CHECK(ReadText(hits).find("GET /v1/videos") == std::string::npos);

  // Unload the worker (as an eviction would).
  HttpResponse unloaded =
      Ask(port, chef, "POST", "/models/unload", "{\"model\":\"clip\"}");
  CHECK(unloaded.status == 200);
  if (unloaded.status != 200)
    std::cerr << "unload " << unloaded.status << ": " << unloaded.body << "\n";

  // /content and status without the worker: no new start.
  HttpResponse content = Ask(port, alice, "GET", content_path);
  CHECK(content.status == 200 && content.body == kMp4);
  CHECK(Header(content, "Content-Type") == "video/mp4");
  HttpResponse part = Ask(port, alice, "GET", content_path, "", "bytes=0-3");
  CHECK(part.status == 206 && part.body == kMp4.substr(0, 4));
  status = Ask(port, alice, "GET", status_path);
  CHECK(status.status == 200 &&
        status.body.find("\"status\":\"completed\"") != std::string::npos);
  // The admin may read it as well.
  HttpResponse admin = Ask(port, chef, "GET", content_path);
  CHECK(admin.status == 200 && admin.body == kMp4);

  // Another account: 404 (the gate's owner check, as for a relayed read).
  HttpResponse foreign = Ask(port, bob, "GET", content_path);
  CHECK(foreign.status == 404 && foreign.body.find("ftyp") == std::string::npos);
  foreign = Ask(port, bob, "GET", status_path);
  CHECK(foreign.status == 404);
  // No token: refused before anything else.
  HttpResponse anonymous = Ask(port, "", "GET", content_path);
  CHECK(anonymous.status == 401 || anonymous.status == 403);
  CHECK(anonymous.body.find("ftyp") == std::string::npos);

  // Ids of the wrong shape and unknown ids: 404, never a path.
  for (const std::string& path :
       {std::string("/v1/videos/video_0123456789ABCDEF01234567/content"),
        std::string("/v1/videos/video_..%5C..%5Cnutzer.json/content"),
        std::string("/v1/videos/..%2F..%2Fnutzer.json/content"),
        std::string("/v1/videos/video_ffffffffffffffffffffffff/content")}) {
    HttpResponse bad = Ask(port, alice, "GET", path);
    CHECK(bad.status == 404);
    CHECK(bad.body.find("ftyp") == std::string::npos);
    CHECK(bad.body.find("\"rolle\"") == std::string::npos);
    if (bad.status != 404)
      std::cerr << path << " -> " << bad.status << ": " << bad.body << "\n";
  }

  // Failed job: status answers, content is 409; expired: 404. Still no
  // worker start.
  WriteFile(root / kVideoId / "job.json", VideoJobJson("failed", Now() + 3600));
  CHECK(Ask(port, alice, "GET", content_path).status == 409);
  WriteFile(root / kVideoId / "job.json", VideoJobJson("completed", Now() - 5));
  CHECK(Ask(port, alice, "GET", content_path).status == 404);

  CHECK(Count(starts) == 1);
  CHECK(ReadText(hits).find("GET /v1/videos") == std::string::npos);
  int exit_status = 0;
  CHECK(!router->PollExit(&exit_status));

  router->Stop(std::chrono::seconds(5));
  router.reset();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  net::EnsureSocketRuntimeStarted();
  const std::span<char*> args(argv, static_cast<std::size_t>(argc));
  for (char* arg : args)
    if (std::string_view(arg) == "--fake-worker")
      return ServeFakeWorker(args);

  const fs::path base = fs::temp_directory_path() /
                        ("gufo_job_files_" + std::to_string(Now()) + "_" +
                         std::to_string(std::rand()));
  fs::create_directories(base);
  UnitTests(base);
  if (argc >= 2) {
    const fs::path self = fs::absolute(fs::path(argv[0]));
    const fs::path dir = base / "e2e";
    fs::create_directories(dir);
    if (EndToEnd(argv[1], self, dir) != 0)
      ++failures;
  } else {
    std::cout << "(no gufo path: end-to-end part skipped)\n";
  }
  std::error_code ignored;
  fs::remove_all(base, ignored);
  if (failures != 0) {
    std::cerr << failures << " job files check(s) failed\n";
    std::cerr.flush();
    std::_Exit(1);
  }
  std::cout << "All router job files tests passed.\n";
  std::cout.flush();
  std::_Exit(0);
}
