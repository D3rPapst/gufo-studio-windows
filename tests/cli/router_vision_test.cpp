// POST /v1/vision/detect through VisionService: request parsing (tasks,
// aliases, thresholds), the error answers without models (405, 400, 501,
// 503) and, with GUFO_VISION_DIR (converted detectors) and GUFO_VISION_FRAME
// (a PNG keyframe), real detections: the frame as a data URL and as an
// Ablage path through a scripted gate (its 404 is relayed), and two parallel
// requests that must queue one after the other. Pool (cpu_pool.hpp) with a
// scripted partner: Decide, relay when busy or without models, the Ablage
// image inlined for the partner, X-Gufo-Pool local/remote, the partner's
// 503 falling back here, and /v1/figuren "ccip" over the same queue/relay.
// Written with AI assistance (Claude Code).
#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cstdint>
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

#include "ccip/image_ops.hpp"
#include "src/cli/serve/router/cpu_pool.hpp"
#include "src/cli/serve/router/figuren.hpp"
#include "src/cli/serve/router/vision.hpp"
#include "src/core/base64.hpp"
#include "src/core/json.hpp"

namespace json = gufo::json;
namespace cpu = gufo::router::cpu;
namespace figuren = gufo::router::figuren;
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
  // brighten_dark: off by default, true = 70 / 0.5, object overrides.
  assert(!vision::ParseRequest(Body("AAAA", {})).brighten_dark);
  json::Value dark = Body("AAAA", {"head"});
  dark["brighten_dark"] = true;
  r = vision::ParseRequest(dark);
  assert(r.brighten_dark && r.dark_below == 70.0 && r.dark_gamma == 0.5);
  json::Value custom = json::Value::object();
  custom["below"] = 50;
  custom["gamma"] = 0.7;
  dark["brighten_dark"] = std::move(custom);
  r = vision::ParseRequest(dark);
  assert(r.brighten_dark && r.dark_below == 50.0 && r.dark_gamma == 0.7);
  assert(Throws([&] {
    json::Value bad = Body("AAAA", {"head"});
    bad["brighten_dark"] = "ja";
    (void)vision::ParseRequest(bad);
  }));
  // Pose tasks: flags, options, no detector tasks.
  r = vision::ParseRequest(Body("AAAA", {"pose_wholebody"}));
  assert(r.pose_wholebody && !r.pose && r.tasks.empty());
  assert(r.pose_min_score == 0.35f && r.pose_max_persons == 4);
  json::Value po = Body("AAAA", {"pose", "face"});
  json::Value opt = json::Value::object();
  opt["min_score"] = 0.5;
  opt["max_persons"] = 2;
  po["pose_options"] = std::move(opt);
  r = vision::ParseRequest(po);
  assert(r.pose && !r.pose_wholebody && r.tasks.size() == 1 && r.pose_min_score == 0.5f &&
         r.pose_max_persons == 2);
  assert(Throws([&] {
    json::Value bad = Body("AAAA", {"pose"});
    json::Value o = json::Value::object();
    o["max_persons"] = 0;
    bad["pose_options"] = std::move(o);
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

// Pillow values: (255,0,0) -> L 76, (10,200,30) -> L 124, mean 100.0;
// round(255 * (v/255) ** 0.5) for 0, 1, 2, 64, 128, 254, 255.
void TestLumaGamma() {
  gufo::models::ccip::RgbImage two;
  two.width = 2;
  two.height = 1;
  two.rgb = {255, 0, 0, 10, 200, 30};
  assert(gufo::models::ccip::MeanLuma(two) == 100.0);
  gufo::models::ccip::RgbImage ramp;
  ramp.width = 7;
  ramp.height = 1;
  for (int v : {0, 1, 2, 64, 128, 254, 255})
    for (int c = 0; c < 3; ++c)
      ramp.rgb.push_back(static_cast<std::uint8_t>(v));
  const auto bright = gufo::models::ccip::ApplyGamma(ramp, 0.5);
  const int want[] = {0, 16, 23, 128, 181, 254, 255};
  for (int i = 0; i < 7; ++i)
    assert(bright.rgb[static_cast<std::size_t>(i) * 3] == want[i]);
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
  response = service_missing.Handle(Post(Body("AAAA", {"pose"})));
  assert(response->status == 503);
  assert(response->body.find("rtdetr_r50vd_coco_o365.safetensors") != std::string::npos);
}

HttpResponse JsonAnswer(int status, std::string body) {
  HttpResponse response;
  response.status = status;
  response.body = std::move(body);
  response.headers = {{"Content-Type", "application/json"}};
  return response;
}

void TestPool() {
  // Decide.
  HttpRequest plain = Post(Body("AAAA", {"face"}));
  assert(cpu::Decide(plain, true, false, false) == cpu::Route::kLocal);
  assert(cpu::Decide(plain, true, true, false) == cpu::Route::kTryPartner);
  assert(cpu::Decide(plain, true, false, true) == cpu::Route::kTryPartner);
  assert(cpu::Decide(plain, false, true, true) == cpu::Route::kLocal);
  HttpRequest pinned = plain;
  pinned.headers.emplace_back("X-Gufo-Pool", "local");
  assert(cpu::Decide(pinned, true, true, true) == cpu::Route::kLocal);
  HttpRequest remote = plain;
  remote.headers.emplace_back("X-Gufo-Pool", "Remote");
  assert(cpu::Decide(remote, true, false, false) == cpu::Route::kPartnerOnly);
  HttpRequest relayed = plain;
  relayed.headers.emplace_back("X-Gufo-Forwarded", "abc");
  assert(cpu::Decide(relayed, true, true, true) == cpu::Route::kLocal);
  relayed.headers.emplace_back(cpu::kIfIdleHeader, "1");
  assert(cpu::Decide(relayed, true, true, true) == cpu::Route::kLocalIfIdle);
  // The relayed request: no credentials, If-Idle unless forced.
  HttpRequest with_token = plain;
  with_token.headers.emplace_back("Authorization", "Bearer user");
  const HttpRequest copy = cpu::RelayRequest(with_token, "{}", false);
  assert(copy.header("Authorization").empty() && copy.header(cpu::kIfIdleHeader) == "1");
  assert(cpu::RelayRequest(with_token, "{}", true).header(cpu::kIfIdleHeader).empty());

  // No models here: the partner gets it; the Ablage image travels inline.
  struct Partner {
    int calls{0};
    int status{200};
    std::string last_body;
    std::string last_if_idle;
  };
  auto partner = std::make_shared<Partner>();
  vision::Options options;
  options.model_dir = "Z:/does-not-exist/vision";
  options.serve = [](const HttpRequest& inner) {
    HttpResponse file;
    assert(inner.path == "/v1/jobs/j1/files/a.png");
    file.body = std::string("\x89PNG\r\n\x1a\nrest", 12);
    return file;
  };
  options.relay = [partner](const HttpRequest& request) -> std::optional<HttpResponse> {
    ++partner->calls;
    partner->last_body = request.body;
    partner->last_if_idle = request.header(cpu::kIfIdleHeader);
    if (partner->status == 0)
      return std::nullopt;
    return JsonAnswer(partner->status, R"({"object":"vision.detection","detections":{}})");
  };
  auto queue = std::make_shared<cpu::CpuQueue>(4);
  options.queue = queue;
  const vision::VisionService service(options);
  auto response = service.Handle(Post(Body("/v1/jobs/j1/files/a.png", {"face"})));
  assert(response->status == 200 && partner->calls == 1);
  bool ran_on_partner = false;
  for (const auto& [name, value] : response->headers)
    ran_on_partner = ran_on_partner || (name == cpu::kRanOnHeader && value == "partner");
  assert(ran_on_partner);
  const json::Value sent = json::parse(partner->last_body);
  assert(sent.member_str("image").starts_with("data:image/png;base64,"));
  assert(sent.find("tasks") != nullptr && partner->last_if_idle == "1");
  // Partner busy (503) or gone: back here, where the models are missing.
  partner->status = 503;
  response = service.Handle(Post(Body("AAAA", {"face"})));
  assert(response->status == 503 && response->body.find("vision_unavailable") != std::string::npos);
  partner->status = 0;
  response = service.Handle(Post(Body("AAAA", {"face"})));
  assert(response->status == 503 && response->body.find("vision_unavailable") != std::string::npos);
  // Pinned here: never relayed.
  const int before = partner->calls;
  HttpRequest local = Post(Body("AAAA", {"face"}));
  local.headers.emplace_back("X-Gufo-Pool", "local");
  response = service.Handle(local);
  assert(response->status == 503 && partner->calls == before);
  // X-Gufo-Pool: remote without a partner: 503 remote_unavailable.
  HttpRequest forced = Post(Body("AAAA", {"face"}));
  forced.headers.emplace_back("X-Gufo-Pool", "remote");
  response = service.Handle(forced);
  assert(response->status == 503 && response->body.find("remote_unavailable") != std::string::npos);
  assert(partner->last_if_idle.empty());

  // /v1/figuren "ccip" over the same queue and relay: no CCIP models here,
  // so the partner gets the body unchanged.
  partner->status = 200;
  figuren::Options fig;
  fig.queue = queue;
  fig.relay = options.relay;
  const figuren::FigurenService figures(fig);
  json::Value ref = json::Value::object();
  ref["name"] = "lars";
  ref["kopf"] = "AAAA";
  json::Value refs = json::Value::array();
  refs.push_back(std::move(ref));
  json::Value ccip = json::Value::object();
  ccip["verfahren"] = "ccip";
  ccip["bild"] = "AAAA";
  ccip["referenzen"] = std::move(refs);
  HttpRequest fr;
  fr.method = "POST";
  fr.path = "/v1/figuren";
  fr.body = ccip.dump();
  const int calls = partner->calls;
  response = figures.Handle(fr);
  assert(response->status == 200 && partner->calls == calls + 1);
  assert(partner->last_body == fr.body);
  // Busy queue on a relayed request with If-Idle: 503 busy at once.
  fr.headers.emplace_back("X-Gufo-Forwarded", "abc");
  fr.headers.emplace_back(cpu::kIfIdleHeader, "1");
  response = figures.Handle(fr);  // no models here: 503 ccip_unavailable first
  assert(response->status == 503);
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
  assert(answer.find("brightened") != nullptr && !answer.find("brightened")->as_bool());
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
  // Pose (when converted): persons with 17 and 133 points.
  if (std::filesystem::exists(std::filesystem::path(dir) / "rtdetr_r50vd_coco_o365.safetensors")) {
    auto pr = service.Handle(Post(Body(data_url, {"pose", "pose_wholebody"})));
    assert(pr->status == 200);
    const json::Value pj = json::parse(pr->body);
    const json::Value* poses = pj.find("poses");
    assert(poses != nullptr && poses->is_array() && poses->size() >= 1 && poses->size() <= 4);
    for (const auto& p : poses->items()) {
      assert(p.find("keypoints")->size() == 17 && p.find("wholebody")->size() == 133);
    }
    std::cout << "pose: " << poses->size() << " persons, " << pj.member_double("seconds") << " s\n";
  }
  // Relayed here with If-Idle while the queue is busy: 503 busy at once.
  {
    auto queue = std::make_shared<cpu::CpuQueue>(4);
    vision::Options busy_options = options;
    busy_options.queue = queue;
    const vision::VisionService busy_service(busy_options);
    assert(queue->Enter({}) == cpu::CpuQueue::Result::kOk);
    HttpRequest relayed = Post(Body(data_url, {"face"}));
    relayed.headers.emplace_back("X-Gufo-Forwarded", "abc");
    relayed.headers.emplace_back(cpu::kIfIdleHeader, "1");
    auto busy = busy_service.Handle(relayed);
    assert(busy->status == 503 && busy->body.find("busy") != std::string::npos);
    queue->Leave();
    busy = busy_service.Handle(relayed);
    assert(busy->status == 200);
  }
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
  TestLumaGamma();
  TestErrors();
  TestPool();
  const char* dir = std::getenv("GUFO_VISION_DIR");
  const char* frame = std::getenv("GUFO_VISION_FRAME");
  if (dir != nullptr && *dir != '\0' && frame != nullptr && *frame != '\0')
    TestReal(dir, frame);
  else
    std::cout << "real detections skipped (set GUFO_VISION_DIR and GUFO_VISION_FRAME)\n";
  std::cout << "router_vision_test: ok\n";
  return 0;
}
