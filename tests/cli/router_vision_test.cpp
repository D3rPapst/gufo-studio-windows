// POST /v1/vision/detect through VisionService: request parsing (tasks,
// aliases, thresholds), the error answers without models (405, 400, 503),
// the ocr passes (thresholds, box_threshold, scale, lists) and wd14_tags
// options and, with GUFO_VISION_DIR (converted detectors, PP-OCRv4 and
// tagger) and GUFO_VISION_FRAME (a PNG keyframe), real detections, texts
// (one plain and one 3x LANCZOS pass, the original pixels under
// brighten_dark) and tags (whole frame, person cut-outs, explicit regions,
// further pictures): the frame as a data URL and as an
// Ablage path through a scripted gate (its 404 is relayed), and two parallel
// requests that must queue one after the other. Pool (cpu_pool.hpp) with a
// scripted partner: Decide, relay when busy or without models, the Ablage
// image inlined for the partner (with the ocr options and thresholds), the
// ocr models missing here, X-Gufo-Pool local/remote, the partner's 503
// falling back here, and /v1/figuren "ccip" over the same queue/relay.
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
#include "ccip/wd14.hpp"
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
  // wd14_tags: defaults, options, regions "person" adds the person task.
  r = vision::ParseRequest(Body("AAAA", {"wd14_tags"}));
  assert(r.tasks.empty() && r.wd14.enabled && r.wd14.general_threshold == 0.35 &&
         r.wd14.character_threshold == 0.85 && !r.wd14.persons && r.wd14.boxes.empty());
  json::Value tags = Body("AAAA", {"wd14_tags", "face"});
  json::Value wd = json::Value::object();
  wd["general_threshold"] = 0.5;
  wd["use_detections"] = "person";
  wd["margin"] = 0.1;
  json::Value pics = json::Value::array();
  pics.push_back("BBBB");
  wd["images"] = std::move(pics);
  tags["wd14_tags"] = std::move(wd);
  r = vision::ParseRequest(tags);
  assert(r.wd14.persons && r.wd14.general_threshold == 0.5 && r.wd14.margin == 0.1);
  assert(r.wd14.images.size() == 1 && r.wd14.images[0] == "BBBB");
  assert(r.tasks.size() == 2 && r.tasks[0].task == "person" && r.tasks[1].task == "face");
  json::Value boxes = Body("AAAA", {"wd14_tags"});
  json::Value wb = json::Value::object();
  wb["regions"] = json::parse("[[10, 20, 110, 220.5]]");
  boxes["wd14_tags"] = std::move(wb);
  r = vision::ParseRequest(boxes);
  assert(r.wd14.boxes.size() == 1 && r.wd14.boxes[0][3] == 220.5 && r.tasks.empty());
  for (const char* bad_options :
       {R"({"regions": [[10, 20, 5, 30]]})", R"({"regions": "face"})",
        R"({"general_threshold": 2})", R"({"colour": 1})", R"({"images": [1]})",
        R"({"regions": "person", "use_detections": "person"})"}) {
    assert(Throws([&] {
      json::Value bad = Body("AAAA", {"wd14_tags"});
      bad["wd14_tags"] = json::parse(bad_options);
      (void)vision::ParseRequest(bad);
    }));
  }
  assert(Throws([&] {
    json::Value bad = Body("AAAA", {"face"});
    bad["wd14_tags"] = json::Value::object();
    (void)vision::ParseRequest(bad);
  }));
  // Every task of the API runs: detectors, ocr and wd14_tags together.
  r = vision::ParseRequest(Body("AAAA", {"face", "wd14_tags", "ocr"}));
  assert(r.tasks.size() == 1 && r.wd14.enabled && r.ocr.size() == 1);
  // ocr: opt-in, one pass with imgutils defaults, kept as double.
  r = vision::ParseRequest(Body("AAAA", {"ocr"}));
  assert(r.ocr.size() == 1 && !r.ocr_list && r.tasks.empty());
  assert(r.ocr[0].heat_threshold == 0.3 && r.ocr[0].box_threshold == 0.7 &&
         r.ocr[0].unclip_ratio == 2.0 && r.ocr[0].scale == 1);
  r = vision::ParseRequest(Body("AAAA", {}));
  assert(r.ocr.empty());
  json::Value ocr_body = Body("AAAA", {"ocr", "face"});
  json::Value ocr_th = json::Value::object();
  json::Value ocr_values = json::Value::object();
  ocr_values["heat"] = 0.4;
  ocr_values["box"] = 0.6;
  ocr_th["ocr"] = std::move(ocr_values);
  ocr_body["thresholds"] = std::move(ocr_th);
  r = vision::ParseRequest(ocr_body);
  assert(r.ocr.size() == 1 && r.tasks.size() == 1 && r.ocr[0].heat_threshold == 0.4 &&
         r.ocr[0].box_threshold == 0.6);
  // "ocr" options: one object, or a list of passes (erkennung_schrift.py:
  // box_threshold 0.5 on the frame and on a 3x enlargement); thresholds.ocr
  // is the default of every pass.
  ocr_body["ocr"] = json::parse(R"({"box_threshold": 0.5, "unclip_ratio": 1.5})");
  r = vision::ParseRequest(ocr_body);
  assert(r.ocr.size() == 1 && !r.ocr_list && r.ocr[0].box_threshold == 0.5 &&
         r.ocr[0].heat_threshold == 0.4 && r.ocr[0].unclip_ratio == 1.5 && r.ocr[0].scale == 1);
  ocr_body["ocr"] = json::parse(R"([{"box_threshold": 0.5}, {"box_threshold": 0.5, "scale": 3}])");
  r = vision::ParseRequest(ocr_body);
  assert(r.ocr.size() == 2 && r.ocr_list && r.ocr[0].scale == 1 && r.ocr[1].scale == 3 &&
         r.ocr[1].box_threshold == 0.5 && r.ocr[1].heat_threshold == 0.4);
  for (const char* bad_options :
       {R"({"scale": 0})", R"({"scale": 2.5})", R"({"scale": 5})", R"({"box_threshold": 0})",
        R"({"heat_threshold": "x"})", R"({"unclip_ratio": -1})", R"({"box": 0.5})", R"([])",
        R"([{}, {}, {}, {}, {}])", R"([1])", R"("fast")"}) {
    assert(Throws([&] {
      json::Value bad = Body("AAAA", {"ocr"});
      bad["ocr"] = json::parse(bad_options);
      (void)vision::ParseRequest(bad);
    }));
  }
  assert(Throws([&] {
    json::Value bad = Body("AAAA", {"face"});
    bad["ocr"] = json::Value::object();
    (void)vision::ParseRequest(bad);
  }));
  assert(Throws([&] {
    json::Value bad = Body("AAAA", {"ocr"});
    json::Value t = json::Value::object();
    json::Value o = json::Value::object();
    o["iou"] = 0.5;
    t["ocr"] = std::move(o);
    bad["thresholds"] = std::move(t);
    (void)vision::ParseRequest(bad);
  }));
  assert(Throws([&] {
    json::Value bad = Body("AAAA", {"face"});
    json::Value t = json::Value::object();
    t["ocr"] = json::Value::object();
    bad["thresholds"] = std::move(t);
    (void)vision::ParseRequest(bad);
  }));
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
  // Unknown task: 400 naming every task; no --vision-dir: 503 for each task.
  auto response = service.Handle(Post(Body("AAAA", {"cat"})));
  assert(response->status == 400);
  assert(response->body.find("person, head, face, hand, ocr, wd14_tags") != std::string::npos);
  for (const char* task : {"person", "ocr", "wd14_tags"}) {
    response = service.Handle(Post(Body("AAAA", {task})));
    assert(response->status == 503);
    assert(response->body.find("vision_unavailable") != std::string::npos);
    assert(response->body.find("--vision-dir") != std::string::npos);
  }
  vision::Options missing;
  missing.model_dir = "Z:/does-not-exist/vision";
  const vision::VisionService service_missing(missing);
  response = service_missing.Handle(Post(Body("AAAA", {"face"})));
  assert(response->status == 503);
  assert(response->body.find("face_detect_v1.4_s.safetensors") != std::string::npos);
  response = service_missing.Handle(Post(Body("AAAA", {"wd14_tags"})));
  assert(response->status == 503);
  assert(response->body.find("wd-swinv2-tagger-v3.safetensors") != std::string::npos);
  response = service_missing.Handle(Post(Body("AAAA", {"ocr"})));
  assert(response->status == 503);
  assert(response->body.find("ch_PP-OCRv4_det.safetensors") != std::string::npos);
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
  // Further wd14_tags pictures from the Ablage travel inline as well.
  json::Value with_refs = Body("AAAA", {"wd14_tags"});
  json::Value refs_opt = json::Value::object();
  json::Value ref_list = json::Value::array();
  ref_list.push_back("/v1/jobs/j1/files/a.png");
  refs_opt["images"] = std::move(ref_list);
  refs_opt["general_threshold"] = 0.4;
  with_refs["wd14_tags"] = std::move(refs_opt);
  response = service.Handle(Post(with_refs));
  assert(response->status == 200);
  const json::Value sent_refs = json::parse(partner->last_body);
  const json::Value* sent_opt = sent_refs.find("wd14_tags");
  assert(sent_opt != nullptr && sent_opt->member_double("general_threshold") == 0.4);
  assert(sent_opt->find("images")->items()[0].str().starts_with("data:image/png;base64,"));
  // ocr (models missing here too): the passes and thresholds.ocr travel as
  // sent.
  json::Value with_ocr = Body("/v1/jobs/j1/files/a.png", {"ocr", "face"});
  with_ocr["ocr"] = json::parse(R"([{"box_threshold": 0.5}, {"box_threshold": 0.5, "scale": 3}])");
  with_ocr["thresholds"] = json::parse(R"({"ocr": {"heat": 0.35}, "face": {"conf": 0.4}})");
  response = service.Handle(Post(with_ocr));
  assert(response->status == 200);
  const json::Value sent_ocr = json::parse(partner->last_body);
  assert(sent_ocr.member_str("image").starts_with("data:image/png;base64,"));
  assert(sent_ocr.find("ocr")->dump() == with_ocr.find("ocr")->dump());
  assert(sent_ocr.find("thresholds")->dump() == with_ocr.find("thresholds")->dump());
  assert(sent_ocr.find("tasks")->dump() == with_ocr.find("tasks")->dump());
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
  assert(answer.find("detections")->find("ocr") == nullptr);
  // ocr when the converted PP-OCRv4 files are in the same directory.
  if (std::filesystem::exists(std::filesystem::path(dir) / "ch_PP-OCRv4_rec.dict.txt")) {
    response = service.Handle(Post(Body(data_url, {"ocr"})));
    std::cout << response->body.substr(0, 600) << '\n';
    assert(response->status == 200);
    const json::Value texts = json::parse(response->body);
    const json::Value* list = texts.find("detections")->find("ocr");
    assert(list != nullptr && list->is_array());
    for (const json::Value& t : list->items()) {
      assert(t.find("text")->is_string());
      assert(t.find("points")->size() == 4 && t.find("box")->size() == 4);
    }
    assert(texts.find("models")->find("ocr")->member_str("rec_model") == "ch_PP-OCRv4_rec");
    // OCR reads the original even when the detectors see the brightened
    // copy (below 256: always brightened).
    json::Value dark = Body(data_url, {"ocr"});
    dark["brighten_dark"] = json::parse(R"({"below": 256})");
    response = service.Handle(Post(dark));
    assert(response->status == 200);
    const json::Value dark_texts = json::parse(response->body);
    assert(dark_texts.find("brightened")->as_bool());
    assert(dark_texts.find("detections")->find("ocr")->dump() == list->dump());
    // erkennung_schrift.py in one request: box_threshold 0.5 on the frame
    // and on a 3x LANCZOS enlargement, its boxes divided back by 3.
    json::Value passes = Body(data_url, {"ocr"});
    passes["ocr"] = json::parse(R"([{"box_threshold": 0.5}, {"box_threshold": 0.5, "scale": 3}])");
    response = service.Handle(Post(passes));
    assert(response->status == 200);
    const json::Value jp = json::parse(response->body);
    const auto& per_pass = jp.find("detections")->find("ocr")->items();
    const auto& infos = jp.find("models")->find("ocr")->items();
    assert(per_pass.size() == 2 && infos.size() == 2);
    assert(infos[1].find("scale")->as_double() == 3 && infos[1].member_double("box") == 0.5);
    const double width = jp.find("width")->as_double();
    const double height = jp.find("height")->as_double();
    for (const json::Value& t : per_pass[1].items()) {
      const auto& b = t.find("box")->items();
      assert(b[0].as_double() >= 0 && b[2].as_double() <= width);
      assert(b[1].as_double() >= 0 && b[3].as_double() <= height);
    }
    std::cout << "ocr passes: " << per_pass[0].size() << " texts plain, " << per_pass[1].size()
              << " texts 3x, " << jp.member_double("seconds") << " s\n";
  }
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
  // wd14_tags: whole frame, the person cut-outs (8 % margin), an explicit
  // region and a further picture (the frame again, so the same tags).
  {
    json::Value whole = Body(data_url, {"wd14_tags"});
    response = service.Handle(Post(whole));
    assert(response->status == 200);
    const json::Value jw = json::parse(response->body);
    const json::Value& w0 = jw.find("detections")->find("wd14_tags")->items().at(0);
    assert(w0.member_str("source") == "image" && w0.find("rating")->size() == 4);
    assert(!w0.find("general")->empty());
    assert(jw.find("models")->find("wd14_tags")->member_str("model") ==
           "SmilingWolf/wd-swinv2-tagger-v3");
    std::cout << "wd14 whole frame: " << w0.find("general")->size() << " general tags, "
              << jw.member_double("seconds") << " s\n";
    json::Value cut = Body(data_url, {"wd14_tags"});
    json::Value opt = json::Value::object();
    opt["regions"] = "person";
    json::Value more = json::Value::array();
    more.push_back(data_url);
    opt["images"] = std::move(more);
    cut["wd14_tags"] = std::move(opt);
    response = service.Handle(Post(cut));
    assert(response->status == 200);
    const json::Value jc = json::parse(response->body);
    const auto& entries = jc.find("detections")->find("wd14_tags")->items();
    const auto& people = jc.find("detections")->find("person")->items();
    assert(people.size() == persons && entries.size() == persons + 1);
    for (std::size_t i = 0; i < persons; ++i) {
      assert(entries[i].member_str("source") == "person");
      const auto& pb = people[i].find("box")->items();
      const auto want = gufo::models::ccip::MarginCropBox(
          {pb[0].as_double(), pb[1].as_double(), pb[2].as_double(), pb[3].as_double()}, 0.08,
          static_cast<int>(jc.find("width")->as_double()), static_cast<int>(jc.find("height")->as_double()));
      for (std::size_t c = 0; c < 4; ++c)
        assert(static_cast<int>(entries[i].find("box")->items()[c].as_double()) == want[c]);
    }
    const json::Value& extra = entries.back();
    assert(extra.member_str("source") == "images" && extra.find("index")->as_double() == 0);
    assert(extra.find("general")->dump() == w0.find("general")->dump());
    std::cout << "wd14 " << persons << " person cut-outs + 1 picture: "
              << jc.member_double("seconds") << " s (with person detection)\n";
    json::Value region = Body(data_url, {"wd14_tags"});
    json::Value ro = json::Value::object();
    ro["regions"] = json::parse("[[100, 100, 400, 500]]");
    ro["margin"] = 0;
    region["wd14_tags"] = std::move(ro);
    response = service.Handle(Post(region));
    assert(response->status == 200);
    const json::Value jr = json::parse(response->body);
    const json::Value& r0 = jr.find("detections")->find("wd14_tags")->items().at(0);
    assert(r0.member_str("source") == "region" && r0.find("width")->as_double() == 300 &&
           r0.find("height")->as_double() == 400);
    json::Value outside = Body(data_url, {"wd14_tags"});
    json::Value oo = json::Value::object();
    oo["regions"] = json::parse("[[5000, 5000, 6000, 6000]]");
    outside["wd14_tags"] = std::move(oo);
    assert(service.Handle(Post(outside))->status == 400);
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
