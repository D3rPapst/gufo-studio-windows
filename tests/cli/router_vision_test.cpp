// POST /v1/vision/detect through VisionService: request parsing (tasks,
// aliases, thresholds), the error answers without models (405, 400, 501,
// 503) and, with GUFO_VISION_DIR (converted detectors) and GUFO_VISION_FRAME
// (a PNG keyframe), real detections: the frame as a data URL and as an
// Ablage path through a scripted gate (its 404 is relayed), and two parallel
// requests that must queue one after the other.
// Written with AI assistance (Claude Code).
#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/router/vision.hpp"
#include "src/core/base64.hpp"
#include "src/core/json.hpp"

namespace json = gufo::json;
namespace vision = gufo::router::vision;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;

namespace {

HttpRequest Post(const json::Value& body) {
  HttpRequest request;
  request.method = "POST";
  request.path = "/v1/vision/detect";
  request.body = body.dump();
  return request;
}

json::Value Body(std::string image, std::vector<std::string> tasks) {
  json::Value body = json::Value::object();
  body["image"] = std::move(image);
  if (!tasks.empty()) {
    json::Value list = json::Value::array();
    for (auto& t : tasks)
      list.push_back(std::move(t));
    body["tasks"] = std::move(list);
  }
  return body;
}

template <typename F>
bool Throws(F&& f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

void TestParse() {
  // Defaults: all four detectors in table order with imgutils thresholds.
  vision::Request r = vision::ParseRequest(Body("data:image/png;base64,AAAA", {}));
  assert(r.tasks.size() == 4);
  assert(r.tasks[0].task == "person" && r.tasks[0].conf == 0.3f && r.tasks[0].iou == 0.5f);
  assert(r.tasks[1].task == "head" && r.tasks[1].model == "head_detect_v2.0_s");
  assert(r.tasks[2].task == "face" && r.tasks[2].model == "face_detect_v1.4_s");
  assert(r.tasks[3].task == "hand" && r.tasks[3].conf == 0.35f);
  // Caller order and duplicates do not matter; aliases bild/aufgaben.
  json::Value alias = json::Value::object();
  alias["bild"] = "AAAA";
  json::Value list = json::Value::array();
  list.push_back("hand");
  list.push_back("face");
  list.push_back("hand");
  alias["aufgaben"] = std::move(list);
  r = vision::ParseRequest(alias);
  assert(r.image == "AAAA");
  assert(r.tasks.size() == 2 && r.tasks[0].task == "face" && r.tasks[1].task == "hand");
  // Thresholds per task.
  json::Value body = Body("AAAA", {"person"});
  json::Value th = json::Value::object();
  json::Value person = json::Value::object();
  person["conf"] = 0.5;
  person["iou"] = 0.6;
  th["person"] = std::move(person);
  body["thresholds"] = std::move(th);
  r = vision::ParseRequest(body);
  assert(r.tasks.size() == 1 && r.tasks[0].conf == 0.5f && r.tasks[0].iou == 0.6f);
  // Errors.
  assert(Throws([] { (void)vision::ParseRequest(json::Value::array()); }));
  assert(Throws([] { (void)vision::ParseRequest(Body("", {})); }));
  assert(Throws([] { (void)vision::ParseRequest(Body("AAAA", {"cat"})); }));
  assert(Throws([&] {
    json::Value bad = Body("AAAA", {"face"});
    json::Value t = json::Value::object();
    json::Value f = json::Value::object();
    f["conf"] = 1.5;
    t["face"] = std::move(f);
    bad["thresholds"] = std::move(t);
    (void)vision::ParseRequest(bad);
  }));
  assert(Throws([&] {
    json::Value bad = Body("AAAA", {"face"});
    json::Value t = json::Value::object();
    t["hand"] = json::Value::object();
    bad["thresholds"] = std::move(t);
    (void)vision::ParseRequest(bad);
  }));
  bool planned = false;
  try {
    (void)vision::ParseRequest(Body("AAAA", {"face", "ocr"}));
  } catch (const std::domain_error&) {
    planned = true;
  }
  assert(planned);
}

void TestErrors() {
  const vision::VisionService service(vision::Options{});
  HttpRequest other;
  other.method = "POST";
  other.path = "/v1/figuren";
  assert(!service.Handle(other).has_value());
  HttpRequest get;
  get.method = "GET";
  get.path = "/v1/vision/detect";
  assert(service.Handle(get)->status == 405);
  HttpRequest broken;
  broken.method = "POST";
  broken.path = "/v1/vision/detect";
  broken.body = "{";
  assert(service.Handle(broken)->status == 400);
  auto response = service.Handle(Post(Body("AAAA", {"wd14_tags"})));
  assert(response->status == 501);
  assert(response->body.find("task_not_available") != std::string::npos);
  response = service.Handle(Post(Body("AAAA", {"person"})));
  assert(response->status == 503);
  assert(response->body.find("vision_unavailable") != std::string::npos);
  vision::Options missing;
  missing.model_dir = "Z:/does-not-exist/vision";
  const vision::VisionService service_missing(missing);
  response = service_missing.Handle(Post(Body("AAAA", {"face"})));
  assert(response->status == 503);
  assert(response->body.find("face_detect_v1.4_s.safetensors") != std::string::npos);
}

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void TestReal(const char* dir, const char* frame) {
  const std::string png = ReadFile(frame);
  assert(!png.empty());
  const std::string data_url =
      "data:image/png;base64," +
      gufo::core::EncodeBase64(std::string_view(png));
  vision::Options options;
  options.model_dir = dir;
  int gate_calls = 0;
  options.serve = [&](const HttpRequest& inner) {
    ++gate_calls;
    assert(inner.method == "GET");
    assert(inner.header("Authorization") == "Bearer test");
    HttpResponse response;
    if (inner.path == "/v1/jobs/abc/files/frame.png") {
      response.body = png;
      response.headers = {{"Content-Type", "image/png"}};
    } else {
      response.status = 404;
      response.reason = "Not Found";
      response.body = R"({"error":{"code":"not_found","message":"no such job"}})";
    }
    return response;
  };
  const vision::VisionService service(options);
  auto response = service.Handle(Post(Body(data_url, {})));
  std::cout << response->body.substr(0, 600) << '\n';
  assert(response->status == 200);
  const json::Value answer = json::parse(response->body);
  for (const char* task : {"person", "head", "face", "hand"})
    assert(answer.find("detections")->find(task) != nullptr);
  const std::size_t persons = answer.find("detections")->find("person")->size();
  // Ablage path through the scripted gate: same answer.
  HttpRequest ablage = Post(Body("/v1/jobs/abc/files/frame.png", {"person"}));
  ablage.headers.emplace_back("Authorization", "Bearer test");
  response = service.Handle(ablage);
  assert(response->status == 200);
  assert(json::parse(response->body).find("detections")->find("person")->size() == persons);
  HttpRequest foreign = Post(Body("/v1/jobs/other/files/frame.png", {"person"}));
  foreign.headers.emplace_back("Authorization", "Bearer test");
  response = service.Handle(foreign);
  assert(response->status == 404);
  assert(gate_calls == 2);
  // Two at once: one waits for the other.
  auto a = std::async(std::launch::async, [&] { return service.Handle(Post(Body(data_url, {}))); });
  auto b = std::async(std::launch::async, [&] { return service.Handle(Post(Body(data_url, {}))); });
  const json::Value ja = json::parse(a.get()->body);
  const json::Value jb = json::parse(b.get()->body);
  const double waited = std::max(ja.member_double("queue_seconds"), jb.member_double("queue_seconds"));
  const double ran = std::min(ja.member_double("seconds"), jb.member_double("seconds"));
  std::cout << "queued " << waited << " s, ran " << ran << " s\n";
  assert(waited >= 0.5 * ran);
}

}  // namespace

int main() {
  TestParse();
  TestErrors();
  const char* dir = std::getenv("GUFO_VISION_DIR");
  const char* frame = std::getenv("GUFO_VISION_FRAME");
  if (dir != nullptr && *dir != '\0' && frame != nullptr && *frame != '\0')
    TestReal(dir, frame);
  else
    std::cout << "real detections skipped (set GUFO_VISION_DIR and GUFO_VISION_FRAME)\n";
  std::cout << "router_vision_test: ok\n";
  return 0;
}
