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
// Request limits: encoded bytes and decoded pixels over all pictures,
// decoding only after the CPU turn. The model checks hand_fingers and
// person_count with a scripted chat (no model).
// Written with AI assistance (Claude Code).
#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <png.h>

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
  // At most kWd14MaxRegions explicit regions (each is one tagger run).
  {
    const auto regions = [](std::size_t n) {
      json::Value list = json::Value::array();
      for (std::size_t i = 0; i < n; ++i)
        list.push_back(json::parse("[0, 0, 1, 1]"));
      json::Value body = Body("AAAA", {"wd14_tags"});
      json::Value o = json::Value::object();
      o["regions"] = std::move(list);
      body["wd14_tags"] = std::move(o);
      return body;
    };
    assert(vision::ParseRequest(regions(vision::kWd14MaxRegions)).wd14.boxes.size() ==
           vision::kWd14MaxRegions);
    assert(Throws([&] { (void)vision::ParseRequest(regions(vision::kWd14MaxRegions + 1)); }));
  }
  for (const char* bad_options :
       {R"({"regions": [[10, 20, 5, 30]]})", R"({"regions": "face"})",
        R"({"regions": [[0, 0, 1e12, 10]]})", R"({"regions": [[-2e6, 0, 10, 10]]})",
        R"({"general_threshold": 2})", R"({"colour": 1})", R"({"images": [1]})",
        R"({"regions": "person", "use_detections": "person"})"}) {
    assert(Throws([&] {
      json::Value bad = Body("AAAA", {"wd14_tags"});
      bad["wd14_tags"] = json::parse(bad_options);
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
  // pose_boxes: given boxes instead of RT-DETR, at most kMaxPoseBoxes.
  {
    json::Value given = Body("AAAA", {"pose_wholebody"});
    given["pose_boxes"] = json::parse("[[10, 20, 110, 220.5], [0, 0, 5, 5]]");
    r = vision::ParseRequest(given);
    assert(r.pose_boxes.size() == 2 && r.pose_boxes[0][3] == 220.5f);
    json::Value many = json::Value::array();
    for (std::size_t i = 0; i <= vision::kMaxPoseBoxes; ++i)
      many.push_back(json::parse("[0, 0, 1, 1]"));
    for (const char* bad_boxes : {"[]", "[[0, 0, 1]]", "[[5, 0, 1, 1]]", "[[0, 0, 1e12, 1]]",
                                  "\"person\""}) {
      assert(Throws([&] {
        json::Value bad = Body("AAAA", {"pose"});
        bad["pose_boxes"] = json::parse(bad_boxes);
        (void)vision::ParseRequest(bad);
      }));
    }
    assert(Throws([&] {
      json::Value bad = Body("AAAA", {"pose"});
      bad["pose_boxes"] = many;
      (void)vision::ParseRequest(bad);
    }));
    assert(Throws([&] {  // without a pose task
      json::Value bad = Body("AAAA", {"face"});
      bad["pose_boxes"] = json::parse("[[0, 0, 1, 1]]");
      (void)vision::ParseRequest(bad);
    }));
  }
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
  // "rec": the recognizer per pass, ch_v4 (imgutils) unless named.
  assert(r.ocr[0].rec == "ch_v4" && r.ocr[1].rec == "ch_v4");
  ocr_body["ocr"] = json::parse(R"([{"rec": "latin_v5"}, {"scale": 3}])");
  r = vision::ParseRequest(ocr_body);
  assert(r.ocr.size() == 2 && r.ocr[0].rec == "latin_v5" && r.ocr[1].rec == "ch_v4");
  ocr_body["ocr"] = json::parse(R"({"rec": "ch_v4"})");
  assert(vision::ParseRequest(ocr_body).ocr[0].rec == "ch_v4");
  for (const char* bad_options :
       {R"({"scale": 0})", R"({"scale": 2.5})", R"({"scale": 5})", R"({"box_threshold": 0})",
        R"({"heat_threshold": "x"})", R"({"unclip_ratio": -1})", R"({"box": 0.5})", R"([])",
        R"([{}, {}, {}, {}, {}])", R"([1])", R"("fast")", R"({"rec": "german"})",
        R"({"rec": 5})", R"({"rec": "LATIN_V5"})"}) {
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
  assert(response->body.find("person, head, face, hand, hand_fingers, person_count, pose, "
                             "pose_wholebody, ocr, wd14_tags") != std::string::npos);
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
  response = service_missing.Handle(Post(Body("AAAA", {"pose"})));
  assert(response->status == 503);
  assert(response->body.find("rtdetr_r50vd_coco_o365.safetensors") != std::string::npos);
  // hand_fingers needs the hand detector.
  response = service_missing.Handle(Post(Body("AAAA", {"hand_fingers"})));
  assert(response->status == 503);
  assert(response->body.find("hand_detect_v1.0_s.safetensors") != std::string::npos);
  // person_count needs the person detector.
  response = service_missing.Handle(Post(Body("AAAA", {"person_count"})));
  assert(response->status == 503);
  assert(response->body.find("person_detect_v1.1_m.safetensors") != std::string::npos);
}

HttpResponse JsonAnswer(int status, std::string body) {
  HttpResponse response;
  response.status = status;
  response.body = std::move(body);
  response.headers = {{"Content-Type", "application/json"}};
  return response;
}

/// A chat answer as the router's /v1/chat/completions gives it.
std::string ChatAnswer(const std::string& content) {
  json::Value message = json::Value::object();
  message["role"] = "assistant";
  message["content"] = content;
  json::Value choice = json::Value::object();
  choice["index"] = 0;
  choice["message"] = std::move(message);
  choice["finish_reason"] = "stop";
  json::Value choices = json::Value::array();
  choices.push_back(std::move(choice));
  json::Value body = json::Value::object();
  body["object"] = "chat.completion";
  body["choices"] = std::move(choices);
  return body.dump();
}

/// The same answer as a stream: the content in two delta chunks, a ping
/// comment in between, [DONE] at the end.
std::string StreamAnswer(const std::string& content) {
  const auto chunk = [](const std::string& piece, const char* finish) {
    json::Value delta = json::Value::object();
    delta["content"] = piece;
    json::Value choice = json::Value::object();
    choice["index"] = 0;
    choice["delta"] = std::move(delta);
    if (finish != nullptr)
      choice["finish_reason"] = finish;
    json::Value choices = json::Value::array();
    choices.push_back(std::move(choice));
    json::Value body = json::Value::object();
    body["object"] = "chat.completion.chunk";
    body["choices"] = std::move(choices);
    return "data: " + body.dump() + "\n\n";
  };
  const std::size_t half = content.size() / 2;
  return chunk(content.substr(0, half), nullptr) + ": ping\n\n" +
         chunk(content.substr(half), "stop") + "data: [DONE]\n\n";
}

/// A verdict as the model writes it.
std::string Verdict(bool plausibel, bool sicher, int fingers, const std::string& verdeckt,
                    const std::string& befunde, const std::string& befund) {
  return std::string("{\"fingers_sichtbar\": ") + std::to_string(fingers) +
         ", \"verdeckt\": [" + verdeckt + "], \"befunde\": [" + befunde +
         "], \"plausibel\": " + (plausibel ? "true" : "false") +
         ", \"sicher\": " + (sicher ? "true" : "false") + ", \"befund\": \"" + befund + "\"}";
}

bool ParseFails(const std::string& body) {
  try {
    (void)vision::ParseFingerAnswer(body);
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

bool BadFingerOption(const char* key, json::Value value, std::vector<std::string> tasks) {
  json::Value bad = Body("AAAA", std::move(tasks));
  json::Value o = json::Value::object();
  o[key] = std::move(value);
  bad["hand_fingers_options"] = std::move(o);
  return Throws([&] { (void)vision::ParseRequest(bad); });
}

/// "hand_fingers": options, the crop box, the chat body, the answer parsing
/// and CountFingers with a scripted chat (no model).
void TestHandFingers() {
  // The task runs the hand detector too; defaults.
  vision::Request r = vision::ParseRequest(Body("AAAA", {"hand_fingers"}));
  assert(r.tasks.size() == 1 && r.tasks[0].task == "hand");
  assert(r.hand_fingers.enabled && r.hand_fingers.model == "qwen3.8-27b");
  assert(r.hand_fingers.max_hands == 6 && r.hand_fingers.thinking);
  assert(r.hand_fingers.timeout_seconds == 600 && r.hand_fingers.margin == 0.25);
  assert(r.hand_fingers.wait_seconds == 600 && r.hand_fingers.max_tokens == 8192);
  assert(!vision::ParseRequest(Body("AAAA", {"hand"})).hand_fingers.enabled);
  // With hand listed too, and options.
  json::Value body = Body("AAAA", {"face", "hand", "hand_fingers"});
  json::Value options = json::Value::object();
  options["model"] = "qwen3.8-flash";
  options["max_hands"] = 2;
  options["thinking"] = false;
  options["timeout_seconds"] = 30;
  options["wait_seconds"] = 0;
  options["margin"] = 0.5;
  options["max_tokens"] = 64;
  body["hand_fingers_options"] = std::move(options);
  r = vision::ParseRequest(body);
  assert(r.tasks.size() == 2 && r.tasks[1].task == "hand");
  assert(r.hand_fingers.model == "qwen3.8-flash" && r.hand_fingers.max_hands == 2);
  assert(!r.hand_fingers.thinking && r.hand_fingers.timeout_seconds == 30);
  assert(r.hand_fingers.wait_seconds == 0);
  assert(r.hand_fingers.margin == 0.5 && r.hand_fingers.max_tokens == 64);
  assert(BadFingerOption("max_hands", 0, {"hand_fingers"}));
  assert(BadFingerOption("max_hands", 17, {"hand_fingers"}));
  assert(BadFingerOption("max_hands", 2.5, {"hand_fingers"}));
  assert(BadFingerOption("model", "", {"hand_fingers"}));
  assert(BadFingerOption("thinking", "nein", {"hand_fingers"}));
  assert(BadFingerOption("timeout_seconds", 1, {"hand_fingers"}));
  assert(BadFingerOption("wait_seconds", -1, {"hand_fingers"}));
  assert(BadFingerOption("wait_seconds", 3601, {"hand_fingers"}));
  assert(BadFingerOption("margin", 1.5, {"hand_fingers"}));
  assert(BadFingerOption("farbe", 1, {"hand_fingers"}));
  assert(BadFingerOption("model", "x", {"hand"}));  // options without the task

  // Crop: 25 % of the side on every side, clipped.
  auto crop = vision::FingerCropBox({100, 200, 140, 280}, 0.25, 1000, 1000);
  assert((crop == std::array<int, 4>{90, 180, 150, 300}));
  crop = vision::FingerCropBox({0, 0, 10, 10}, 0.25, 12, 11);
  assert((crop == std::array<int, 4>{0, 0, 12, 11}));

  // The chat body: model, image, the anatomy question, schema, streamed,
  // thinking on.
  const vision::HandFingersRequest defaults;
  const json::Value chat = vision::FingerChatBody(defaults, "data:image/png;base64,AAAA");
  assert(chat.member_str("model") == "qwen3.8-27b");
  const json::Value& part = chat.find("messages")->items()[0].find("content")->items()[0];
  assert(part.member_str("type") == "image_url");
  const std::string prompt =
      chat.find("messages")->items()[0].find("content")->items()[1].member_str("text");
  assert(prompt.find("Ist diese Hand so möglich?") != std::string::npos);
  assert(prompt.find("NOT an error") != std::string::npos);
  assert(chat.find("stream")->as_bool());
  assert(chat.find("response_format")->member_str("type") == "json_schema");
  assert(chat.find("chat_template_kwargs")->find("enable_thinking")->as_bool());
  assert(chat.member_double("max_tokens") == 8192);
  const json::Value& schema = *chat.find("response_format")->find("json_schema")->find("schema");
  const json::Value& props = *schema.find("properties");
  assert(props.find("fingers_sichtbar")->member_double("maximum") == 10);
  assert(props.find("fingers") == nullptr);
  assert(props.find("plausibel")->member_str("type") == "boolean");
  assert(props.find("verdeckt")->find("items")->find("enum")->size() == 5);
  const json::Value& finding = *props.find("befunde")->find("items");
  assert(finding.find("properties")->find("art")->find("enum")->size() == 12);
  assert(finding.find("properties")->find("art")->find("enum")->items()[11].str() == "sonstiges");
  assert(finding.find("required")->size() == 2 && !finding.find("additionalProperties")->as_bool());
  assert(schema.find("required")->size() == 6);

  // Parsing the answer: plain, streamed, fenced, after a think block;
  // refusals.
  auto verdict = vision::ParseFingerAnswer(
      ChatAnswer(Verdict(true, true, 5, "", "", "Hand normal")));
  assert(verdict.plausibel && verdict.sicher && verdict.fingers_sichtbar == 5);
  assert(verdict.verdeckt.empty() && verdict.befunde.empty() && verdict.befund == "Hand normal");
  // Four fingers, the thumb hidden behind the others: plausible, unsure.
  verdict = vision::ParseFingerAnswer(StreamAnswer(
      Verdict(true, false, 4, "\"daumen\", \"daumen\"", "", "Hand normal, Daumen verdeckt")));
  assert(verdict.plausibel && !verdict.sicher && verdict.fingers_sichtbar == 4);
  assert(verdict.verdeckt.size() == 1 && verdict.verdeckt[0] == "daumen");
  assert(verdict.befund == "Hand normal, Daumen verdeckt");
  verdict = vision::ParseFingerAnswer(ChatAnswer(
      "```json\n" +
      Verdict(false, true, 6, "",
              "{\"art\": \"zu_viele_finger\", \"stelle\": \"linke Hand\"}, "
              "{\"art\": \"zweite_hand_am_arm\", \"stelle\": \"rechts am Tisch\"}",
              "sechs Finger") +
      "\n```"));
  assert(!verdict.plausibel && verdict.fingers_sichtbar == 6 && verdict.befunde.size() == 2);
  assert(verdict.befunde[0].art == "zu_viele_finger" && verdict.befunde[0].stelle == "linke Hand");
  assert(verdict.befunde[1].art == "zweite_hand_am_arm");
  verdict = vision::ParseFingerAnswer(ChatAnswer(
      "<think>{\"fingers_sichtbar\": 1}</think>\n" +
      Verdict(false, true, 0, "", "{\"art\": \"arm_ohne_hand\", \"stelle\": \"Mitte\"}",
              "Arm ohne Hand")));
  assert(verdict.fingers_sichtbar == 0 && verdict.befund == "Arm ohne Hand");
  assert(ParseFails("nicht json"));
  assert(ParseFails("{\"choices\": []}"));
  assert(ParseFails(ChatAnswer("")));
  assert(ParseFails(StreamAnswer("")));
  assert(ParseFails(ChatAnswer("fünf")));
  assert(ParseFails("data: {\"error\": {\"message\": \"worker gone\"}}\n\n"));
  assert(ParseFails(ChatAnswer(Verdict(true, true, 11, "", "", ""))));
  assert(ParseFails(ChatAnswer("{\"fingers_sichtbar\": 4.5, \"verdeckt\": [], \"befunde\": [], "
                               "\"plausibel\": true, \"sicher\": true, \"befund\": \"\"}")));
  assert(ParseFails(ChatAnswer("{\"fingers_sichtbar\": 5, \"verdeckt\": [], \"befunde\": [], "
                               "\"plausibel\": true, \"sicher\": \"ja\", \"befund\": \"\"}")));
  assert(ParseFails(ChatAnswer("{\"fingers_sichtbar\": 5, \"verdeckt\": [], \"befunde\": [], "
                               "\"sicher\": true, \"befund\": \"\"}")));  // no plausibel
  assert(ParseFails(ChatAnswer("{\"fingers\": 5, \"sicher\": true, \"befund\": \"x\"}")));  // old
  assert(ParseFails(ChatAnswer(Verdict(true, true, 4, "\"zeh\"", "", ""))));
  assert(ParseFails(ChatAnswer(
      Verdict(false, true, 5, "", "{\"art\": \"hässlich\", \"stelle\": \"\"}", ""))));
  assert(ParseFails(ChatAnswer(Verdict(false, true, 5, "", "{\"art\": \"sonstiges\"}", ""))));
  assert(ParseFails(ChatAnswer(Verdict(true, true, 5, "", "", "x").substr(0, 40))));

  // CountFingers with a scripted chat: an answer, a refusal, garbage, and
  // a hand beyond max_hands. Every hand keeps its box; no exception.
  gufo::models::ccip::RgbImage image;
  image.width = 64;
  image.height = 48;
  image.rgb.assign(static_cast<std::size_t>(64) * 48 * 3, 200);
  const std::vector<vision::HandBox> hands = {{{10, 10, 30, 30}, 0.9},
                                              {{40, 5, 60, 25}, 0.8},
                                              {{0, 30, 20, 48}, 0.7},
                                              {{5, 5, 15, 15}, 0.6}};
  vision::HandFingersRequest options3;
  options3.max_hands = 3;
  std::vector<HttpRequest> asked;
  const std::function<HttpResponse(const HttpRequest&)> serve = [&asked](const HttpRequest& inner) {
    asked.push_back(inner);
    if (asked.size() == 1) {
      // Streamed, as the router answers a stream: true chat.
      HttpResponse streamed;
      streamed.headers = {{"Content-Type", "text/event-stream"}};
      streamed.streaming_body = [](const HttpResponse::BodyWriter& write) {
        (void)write(": ping\n\n");
        (void)write(StreamAnswer(Verdict(true, false, 4, "\"daumen\"", "", "Daumen verdeckt")));
      };
      return streamed;
    }
    if (asked.size() == 2)
      return JsonAnswer(503, "{\"error\": {\"message\": \"model is loading\", \"code\": \"busy\"}}");
    return JsonAnswer(200, ChatAnswer("kein JSON"));
  };
  HttpRequest outer = Post(Body("AAAA", {"hand_fingers"}));
  outer.headers.emplace_back("Authorization", "Bearer user-token");
  outer.headers.emplace_back("X-Other", "nicht weiter");
  const json::Value list = vision::CountFingers(options3, image, hands, serve, outer);
  assert(asked.size() == 3);
  assert(asked[0].method == "POST" && asked[0].path == "/v1/chat/completions");
  bool token = false;
  bool other = false;
  for (const auto& [name, value] : asked[0].headers) {
    token = token || (name == "Authorization" && value == "Bearer user-token");
    other = other || name == "X-Other";
  }
  assert(token && !other);
  assert(asked[0].is_cancelled && !asked[0].is_cancelled());  // the generation clock
  const json::Value sent = json::parse(asked[0].body);
  const std::string url = sent.find("messages")->items()[0].find("content")->items()[0]
                              .find("image_url")->member_str("url");
  assert(url.starts_with("data:image/png;base64,"));
  assert(list.size() == 4);
  const json::Value& first = list.items()[0];
  assert(first.member_double("fingers_sichtbar") == 4 && !first.find("sicher")->as_bool());
  assert(first.find("plausibel")->as_bool() && first.find("befunde")->empty());
  assert(first.find("verdeckt")->items()[0].str() == "daumen");
  assert(first.member_str("befund") == "Daumen verdeckt" && first.find("fehler") == nullptr);
  assert(first.find("waited_seconds") != nullptr && first.find("fingers") == nullptr);
  assert(first.find("box")->items()[2].as_double() == 30);
  assert(first.find("crop")->items()[0].as_double() == 5 && first.find("seconds") != nullptr);
  assert(first.member_double("score") == 0.9);
  const json::Value& second = list.items()[1];
  assert(second.find("fingers_sichtbar") == nullptr);
  assert(second.member_str("fehler").find("503") != std::string::npos);
  assert(second.member_str("fehler").find("model is loading") != std::string::npos);
  assert(list.items()[2].member_str("fehler").find("no JSON object") != std::string::npos);
  assert(list.items()[3].member_str("fehler").find("max_hands") != std::string::npos);
  // No chat available, and a cancelled caller: fehler per hand, no call.
  const json::Value none = vision::CountFingers(options3, image, {hands[0]}, nullptr, outer);
  assert(none.items()[0].member_str("fehler") == "no chat on this router");
  HttpRequest gone = outer;
  gone.is_cancelled = [] { return true; };
  asked.clear();
  const json::Value cancelled = vision::CountFingers(options3, image, {hands[0]}, serve, gone);
  assert(asked.empty() && cancelled.items()[0].member_str("fehler") == "cancelled");

  // 503 memory_busy (another model holds the GPU): asked again after
  // Retry-After while wait_seconds allows; the generation clock starts only
  // with the first chunk.
  std::vector<HttpRequest> busy_asked;
  const std::string busy_body =
      "{\"error\": {\"message\": \"memory is in use by minimax-h3\", \"code\": "
      "\"memory_busy\", \"retry_after_s\": 1}}";
  const std::function<HttpResponse(const HttpRequest&)> busy_serve =
      [&busy_asked, &busy_body](const HttpRequest& inner) {
        busy_asked.push_back(inner);
        if (busy_asked.size() == 1) {
          HttpResponse refused = JsonAnswer(503, busy_body);
          refused.headers.emplace_back("Retry-After", "1");
          return refused;
        }
        return JsonAnswer(200, ChatAnswer(Verdict(true, true, 5, "", "", "Hand normal")));
      };
  vision::HandFingersRequest patient;
  patient.wait_seconds = 5;
  const json::Value retried = vision::CountFingers(patient, image, {hands[0]}, busy_serve, outer);
  assert(busy_asked.size() == 2);
  assert(retried.items()[0].member_str("befund") == "Hand normal");
  assert(retried.items()[0].member_double("seconds") >= 1.0);
  // Without wait time: that hand fails, the later hands are not asked.
  busy_asked.clear();
  vision::HandFingersRequest hasty;
  hasty.wait_seconds = 0;
  const json::Value refused =
      vision::CountFingers(hasty, image, {hands[0], hands[1]}, busy_serve, outer);
  assert(busy_asked.size() == 1);
  assert(refused.items()[0].member_str("fehler").find("not free within wait_seconds") !=
         std::string::npos);
  assert(refused.items()[1].member_str("fehler").starts_with("not checked: model"));
  // A stream that outlives timeout_seconds once it has started: fehler.
  vision::HandFingersRequest quick;
  quick.timeout_seconds = 0;  // below the option range, for the test
  const std::function<HttpResponse(const HttpRequest&)> slow =
      [](const HttpRequest&) {
        HttpResponse streamed;
        streamed.streaming_body = [](const HttpResponse::BodyWriter& write) {
          if (!write("data: {\"choices\": [{\"index\": 0, \"delta\": {\"content\": \"{\"}}]}\n\n"))
            return;
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
          (void)write("data: [DONE]\n\n");
        };
        return streamed;
      };
  const json::Value late = vision::CountFingers(quick, image, {hands[0]}, slow, outer);
  assert(late.items()[0].member_str("fehler").find("s of generation") != std::string::npos);

  // GenerationStarted: the role chunk, prompt_progress, pings and usage do
  // not start the generation clock; content, reasoning, a finish or an
  // error do; a line counts only when it is whole.
  const std::string role_chunk =
      "data: {\"choices\": [{\"index\": 0, \"delta\": {\"role\": \"assistant\"}}]}\n\n";
  const std::string reasoning_chunk =
      "data: {\"choices\": [{\"index\": 0, \"delta\": {\"reasoning_content\": \"Die\"}}]}\n\n";
  {
    std::size_t at = 0;
    std::string stream = ": ping\n\n" + role_chunk;
    assert(!vision::GenerationStarted(stream, &at) && at == stream.size());
    stream += "data: {\"choices\": [{\"index\": 0, \"delta\": {}, \"prompt_progress\": "
              "{\"processed\": 10}}]}\n\ndata: {\"choices\": [], \"usage\": {}}\n\n";
    stream += "data: {\"choices\": [{\"index\": 0, \"delta\": {\"content\": \"\"}}]}\n\n";
    assert(!vision::GenerationStarted(stream, &at));
    stream += reasoning_chunk.substr(0, 30);  // half a line
    assert(!vision::GenerationStarted(stream, &at));
    stream += reasoning_chunk.substr(30);
    assert(vision::GenerationStarted(stream, &at));
    std::size_t zero = 0;
    assert(vision::GenerationStarted(StreamAnswer("{"), &zero));
    zero = 0;
    assert(vision::GenerationStarted("data: {\"error\": {\"message\": \"x\"}}\n\n", &zero));
    zero = 0;
    assert(vision::GenerationStarted(
        "data: {\"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": \"stop\"}]}\n",
        &zero));
  }

  // The 2.2.10 timeouts: the 27B busy with another chat. The worker sends
  // the role chunk after 5 s in its queue (stream_start_delay), then only
  // pings until the model takes the hand. That wait counts against
  // wait_seconds, not timeout_seconds; the generation then has its own
  // clock. Scaled down: queue `queued_ms`, generation `generating_ms`.
  const auto queued_worker = [&](int queued_ms, int generating_ms) {
    return std::function<HttpResponse(const HttpRequest&)>(
        [=](const HttpRequest& inner) {
          HttpResponse streamed;
          streamed.headers = {{"Content-Type", "text/event-stream"}};
          streamed.streaming_body = [=](const HttpResponse::BodyWriter& write) {
            const auto begin = std::chrono::steady_clock::now();
            if (!write(role_chunk))
              return;
            // Queued: pings, the proxy checks is_cancelled between chunks.
            while (std::chrono::steady_clock::now() - begin <
                   std::chrono::milliseconds(queued_ms)) {
              if (inner.is_cancelled() || !write(": ping\n\n"))
                return;
              std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            // Thinking, then the verdict.
            const auto thinking = std::chrono::steady_clock::now();
            do {
              if (inner.is_cancelled() || !write(reasoning_chunk))
                return;
              std::this_thread::sleep_for(std::chrono::milliseconds(50));
            } while (std::chrono::steady_clock::now() - thinking <
                     std::chrono::milliseconds(generating_ms));
            (void)write(StreamAnswer(Verdict(true, true, 5, "", "", "Hand normal")));
          };
          return streamed;
        });
  };
  // Queued 2.5 s, thinking 0.3 s, timeout_seconds 1, wait_seconds 10: the
  // hand gets its verdict (before the fix: "timeout after 1 s of
  // generation", as the direction's three hands after 300 s).
  vision::HandFingersRequest busy_27b;
  busy_27b.timeout_seconds = 1;  // below the option range, for the test
  busy_27b.wait_seconds = 10;
  const json::Value waited =
      vision::CountFingers(busy_27b, image, {hands[0]}, queued_worker(2500, 300), outer);
  assert(waited.items()[0].find("fehler") == nullptr);
  assert(waited.items()[0].member_str("befund") == "Hand normal");
  assert(waited.items()[0].member_double("waited_seconds") >= 2.4);
  assert(waited.items()[0].member_double("seconds") >= 2.7);
  // Still queued when wait_seconds ends: the wait error, the later hands
  // are not asked.
  vision::HandFingersRequest short_wait;
  short_wait.timeout_seconds = 10;
  short_wait.wait_seconds = 1;
  const json::Value not_started = vision::CountFingers(
      short_wait, image, {hands[0], hands[1]}, queued_worker(3000, 0), outer);
  assert(not_started.items()[0].member_str("fehler").find("did not start within wait_seconds") !=
         std::string::npos);
  assert(not_started.items()[0].member_double("seconds") < 2.5);
  assert(not_started.items()[1].member_str("fehler").starts_with("not checked: model"));
  // Started at once but thinking longer than timeout_seconds: the
  // generation error.
  const json::Value too_long =
      vision::CountFingers(busy_27b, image, {hands[0]}, queued_worker(0, 3000), outer);
  assert(too_long.items()[0].member_str("fehler").find("timeout after 1 s of generation") !=
         std::string::npos);
  assert(too_long.items()[0].member_double("seconds") < 2.5);
  // models.hand_fingers names the model.
  assert(vision::HandFingersModelInfo(defaults).member_str("model") == "qwen3.8-27b");
  assert(vision::HandFingersModelInfo(defaults).member_double("wait_seconds") == 600);
}

/// A person verdict as the model writes it.
std::string PersonVerdictText(const std::string& personen, const std::string& befunde,
                              bool sicher, const std::string& befund) {
  return "{\"personen\": [" + personen + "], \"befunde\": [" + befunde +
         "], \"sicher\": " + (sicher ? "true" : "false") + ", \"befund\": \"" + befund + "\"}";
}

bool PersonParseFails(const std::string& body) {
  try {
    (void)vision::ParsePersonAnswer(body);
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

bool BadPersonOption(const char* key, json::Value value, std::vector<std::string> tasks) {
  json::Value bad = Body("AAAA", std::move(tasks));
  json::Value o = json::Value::object();
  o[key] = std::move(value);
  bad["person_count_options"] = std::move(o);
  return Throws([&] { (void)vision::ParseRequest(bad); });
}

/// "person_count": options, the question, the chat body, the answer
/// parsing, gufo's count check and verdict, and CheckPersons with a
/// scripted chat (no model). GUFO_PERSON_BODY_OUT=<file>: the chat bodies
/// of two example requests are written there (for comparing the prompt
/// test script's bodies).
void TestPersonCount() {
  // The task runs the person detector too; defaults.
  vision::Request r = vision::ParseRequest(Body("AAAA", {"person_count"}));
  assert(r.tasks.size() == 1 && r.tasks[0].task == "person");
  assert(r.person_count.enabled && r.person_count.model == "qwen3.8-27b");
  assert(r.person_count.thinking && r.person_count.expected == -1);
  assert(r.person_count.timeout_seconds == 600 && r.person_count.wait_seconds == 600);
  assert(r.person_count.max_tokens == 8192 && r.person_count.references.empty());
  assert(!vision::ParseRequest(Body("AAAA", {"person"})).person_count.enabled);
  // Both model checks in one request: hand and person detectors.
  r = vision::ParseRequest(Body("AAAA", {"hand_fingers", "person_count"}));
  assert(r.tasks.size() == 2 && r.tasks[0].task == "person" && r.tasks[1].task == "hand");
  // Options: expected, references as objects and plain strings.
  json::Value body = Body("AAAA", {"person", "person_count"});
  json::Value options = json::Value::object();
  options["model"] = "qwen3.8-flash";
  options["thinking"] = false;
  options["timeout_seconds"] = 30;
  options["wait_seconds"] = 0;
  options["max_tokens"] = 64;
  options["expected"] = 2;
  json::Value references = json::Value::array();
  json::Value bernd = json::Value::object();
  bernd["name"] = "Bernd";
  bernd["image"] = "/v1/jobs/j1/files/bernd.png";
  references.push_back(std::move(bernd));
  references.push_back("BBBB");
  options["references"] = std::move(references);
  body["person_count_options"] = std::move(options);
  r = vision::ParseRequest(body);
  assert(r.tasks.size() == 1 && r.tasks[0].task == "person");
  assert(r.person_count.model == "qwen3.8-flash" && !r.person_count.thinking);
  assert(r.person_count.timeout_seconds == 30 && r.person_count.wait_seconds == 0);
  assert(r.person_count.max_tokens == 64 && r.person_count.expected == 2);
  assert(r.person_count.references.size() == 2);
  assert(r.person_count.references[0].name == "Bernd");
  assert(r.person_count.references[0].image == "/v1/jobs/j1/files/bernd.png");
  assert(r.person_count.references[1].name == "Figur 2");
  assert(r.person_count.references[1].image == "BBBB");
  // German aliases.
  body = Body("AAAA", {"person_count"});
  options = json::Value::object();
  options["erwartet"] = 0;
  json::Value referenzen = json::Value::array();
  referenzen.push_back("CCCC");
  options["referenzen"] = std::move(referenzen);
  body["person_count_options"] = std::move(options);
  r = vision::ParseRequest(body);
  assert(r.person_count.expected == 0 && r.person_count.references.size() == 1);
  assert(r.person_count.references[0].name == "Figur 1");
  // Refused options.
  assert(BadPersonOption("expected", 21, {"person_count"}));
  assert(BadPersonOption("expected", -1, {"person_count"}));
  assert(BadPersonOption("expected", 1.5, {"person_count"}));
  assert(BadPersonOption("model", "", {"person_count"}));
  assert(BadPersonOption("thinking", "ja", {"person_count"}));
  assert(BadPersonOption("timeout_seconds", 1, {"person_count"}));
  assert(BadPersonOption("wait_seconds", 3601, {"person_count"}));
  assert(BadPersonOption("max_tokens", 9000, {"person_count"}));
  assert(BadPersonOption("max_hands", 2, {"person_count"}));  // a hand_fingers option
  assert(BadPersonOption("expected", 1, {"person"}));          // options without the task
  assert(BadPersonOption("references", "AAAA", {"person_count"}));
  json::Value five = json::Value::array();
  for (int i = 0; i < 5; ++i)
    five.push_back("AAAA");
  assert(BadPersonOption("references", five, {"person_count"}));
  const auto one_reference = [](const char* name, const char* image) {
    json::Value list = json::Value::array();
    json::Value item = json::Value::object();
    if (name != nullptr)
      item["name"] = name;
    if (image != nullptr)
      item["image"] = image;
    list.push_back(std::move(item));
    return list;
  };
  assert(BadPersonOption("references", one_reference("Bernd", nullptr), {"person_count"}));
  assert(BadPersonOption("references", one_reference("Be\"rnd", "AAAA"), {"person_count"}));
  assert(BadPersonOption("references", one_reference("", "AAAA"), {"person_count"}));
  json::Value twice = one_reference("Bernd", "AAAA");
  twice.push_back(one_reference("Bernd", "BBBB").items()[0]);
  assert(BadPersonOption("references", twice, {"person_count"}));
  json::Value number = json::Value::array();
  number.push_back(5);
  assert(BadPersonOption("references", number, {"person_count"}));
  json::Value both = Body("AAAA", {"person_count"});
  json::Value both_options = json::Value::object();
  both_options["expected"] = 1;
  both_options["erwartet"] = 2;
  both["person_count_options"] = std::move(both_options);
  assert(Throws([&] { (void)vision::ParseRequest(both); }));

  // The question: hidden body parts are no finding; the expected count and
  // the reference names are worked in.
  vision::PersonCountRequest plain;
  plain.enabled = true;
  std::string prompt = vision::PersonPrompt(plain);
  assert(prompt.find("Hidden body parts are NOT an error") != std::string::npos);
  assert(prompt.find("extra_limb") != std::string::npos);
  assert(prompt.find("duplicate_person") != std::string::npos);
  assert(prompt.find("here none is, so do not use it") != std::string::npos);
  assert(prompt.find("reference") == std::string::npos);
  assert(prompt.find("Mann im grauen Mantel") != std::string::npos);
  vision::PersonCountRequest with = plain;
  with.expected = 1;
  with.references = {{"Bernd", "AAAA"}, {"Mayk", "BBBB"}};
  prompt = vision::PersonPrompt(with);
  assert(prompt.find("exactly 1 person;") != std::string::npos);
  assert(prompt.find("in this order: Bernd, Mayk.") != std::string::npos);
  assert(prompt.find("\"unbekannt\"") != std::string::npos);
  with.expected = 3;
  assert(vision::PersonPrompt(with).find("exactly 3 persons;") != std::string::npos);
  with.expected = 1;

  // The chat body: "Szene:", the picture, each reference after its name,
  // the question last; schema; streamed, thinking on.
  const json::Value chat =
      vision::PersonChatBody(with, "data:image/png;base64,SCENE", {"data:image/png;base64,REF1",
                                                                    "data:image/png;base64,REF2"});
  assert(chat.member_str("model") == "qwen3.8-27b");
  const json::Value& parts = *chat.find("messages")->items()[0].find("content");
  assert(parts.size() == 7);
  assert(parts.items()[0].member_str("text") == "Szene:");
  assert(parts.items()[1].find("image_url")->member_str("url") == "data:image/png;base64,SCENE");
  assert(parts.items()[2].member_str("text") == "Referenz Bernd:");
  assert(parts.items()[3].find("image_url")->member_str("url") == "data:image/png;base64,REF1");
  assert(parts.items()[4].member_str("text") == "Referenz Mayk:");
  assert(parts.items()[6].member_str("text") == vision::PersonPrompt(with));
  assert(chat.find("stream")->as_bool());
  assert(chat.find("chat_template_kwargs")->find("enable_thinking")->as_bool());
  assert(chat.member_double("max_tokens") == 8192 && chat.member_double("temperature") == 0);
  const json::Value& format = *chat.find("response_format");
  assert(format.member_str("type") == "json_schema");
  assert(format.find("json_schema")->member_str("name") == "person_count");
  assert(format.find("json_schema")->find("strict")->as_bool());
  const json::Value& schema = *format.find("json_schema")->find("schema");
  assert(schema.find("required")->size() == 4 && !schema.find("additionalProperties")->as_bool());
  const json::Value& props = *schema.find("properties");
  assert(props.find("personen")->find("items")->find("required")->size() == 2);
  const json::Value& finding = *props.find("befunde")->find("items")->find("properties");
  assert(finding.find("art")->find("enum")->size() == 4);
  assert(finding.find("art")->find("enum")->items()[0].str() == "extra_limb");
  assert(finding.find("sicherheit")->find("enum")->size() == 3);
  assert(props.find("sicher")->member_str("type") == "boolean");
  assert(props.find("plausibel") == nullptr);
  const json::Value lone = vision::PersonChatBody(plain, "data:image/png;base64,SCENE", {});
  assert(lone.find("messages")->items()[0].find("content")->size() == 3);
  if (const char* out = std::getenv("GUFO_PERSON_BODY_OUT"); out != nullptr && *out != '\0') {
    json::Value bodies = json::Value::object();
    bodies["ohne"] = lone;
    bodies["mit"] = chat;
    std::ofstream(out, std::ios::binary) << bodies.dump();
  }

  // Parsing the answer: plain, streamed, fenced, after a think block.
  auto verdict = vision::ParsePersonAnswer(ChatAnswer(PersonVerdictText(
      "{\"stelle\": \"links am Tisch\", \"wer\": \"Bernd\"}", "", true, "eine Person, normal")));
  assert(verdict.personen.size() == 1 && verdict.personen[0].wer == "Bernd");
  assert(verdict.personen[0].stelle == "links am Tisch");
  assert(verdict.befunde.empty() && verdict.sicher && verdict.befund == "eine Person, normal");
  verdict = vision::ParsePersonAnswer(StreamAnswer(PersonVerdictText(
      "{\"stelle\": \"Mitte\", \"wer\": \"Bernd\"}",
      "{\"art\": \"extra_limb\", \"stelle\": \"rechts hinter Bernd\", \"sicherheit\": \"hoch\"}",
      true, "durchscheinender Arm")));
  assert(verdict.befunde.size() == 1 && verdict.befunde[0].art == "extra_limb");
  assert(verdict.befunde[0].sicherheit == "hoch" && verdict.befunde[0].quelle == "modell");
  verdict = vision::ParsePersonAnswer(ChatAnswer(
      "<think>{\"personen\": []}</think>\n```json\n" +
      PersonVerdictText("", "", false, "dunkel") + "\n```"));
  assert(verdict.personen.empty() && !verdict.sicher);
  assert(PersonParseFails(ChatAnswer("zwei Personen")));
  assert(PersonParseFails(ChatAnswer(
      PersonVerdictText("", "{\"art\": \"ghost\", \"stelle\": \"\", \"sicherheit\": \"hoch\"}",
                        true, ""))));
  assert(PersonParseFails(ChatAnswer(PersonVerdictText(
      "", "{\"art\": \"extra_limb\", \"stelle\": \"\", \"sicherheit\": \"sehr\"}", true, ""))));
  assert(PersonParseFails(ChatAnswer(
      PersonVerdictText("", "{\"art\": \"extra_limb\", \"stelle\": \"x\"}", true, ""))));
  assert(PersonParseFails(ChatAnswer(PersonVerdictText("{\"stelle\": \"links\"}", "", true, ""))));
  assert(PersonParseFails(ChatAnswer(PersonVerdictText("\"Bernd\"", "", true, ""))));
  assert(PersonParseFails(ChatAnswer(
      "{\"personen\": 2, \"befunde\": [], \"sicher\": true, \"befund\": \"\"}")));
  assert(PersonParseFails(ChatAnswer(
      "{\"personen\": [], \"befunde\": [], \"sicher\": \"ja\", \"befund\": \"\"}")));
  assert(PersonParseFails(ChatAnswer("{\"personen\": [], \"befunde\": [], \"sicher\": true}")));
  std::string many;
  for (int i = 0; i < 33; ++i)
    many += std::string(i == 0 ? "" : ", ") + "{\"stelle\": \"x\", \"wer\": \"y\"}";
  assert(PersonParseFails(ChatAnswer(PersonVerdictText(many, "", true, ""))));

  // gufo's count check and the verdict.
  const auto seen = [](int n) {
    std::vector<vision::PersonSeen> list(static_cast<std::size_t>(n));
    return list;
  };
  vision::PersonVerdict two;
  two.personen = seen(2);
  two.sicher = true;
  vision::PersonVerdict v = two;
  vision::CheckPersonCount(with, v);  // expected 1, two listed
  assert(v.befunde.size() == 1 && v.befunde[0].art == "count_mismatch");
  assert(v.befunde[0].quelle == "gufo" && v.befunde[0].sicherheit == "hoch");
  assert(v.befunde[0].stelle == "2 gezählt, 1 erwartet");
  assert(vision::PersonUrteil(v) == "fehler");
  v = two;
  v.sicher = false;
  vision::CheckPersonCount(with, v);
  assert(v.befunde.size() == 1 && v.befunde[0].sicherheit == "mittel");
  assert(vision::PersonUrteil(v) == "pruefen");
  // The model's own count_mismatch is kept, none added.
  v = two;
  v.befunde.push_back({"count_mismatch", "eine Person zu viel", "niedrig", "modell"});
  vision::CheckPersonCount(with, v);
  assert(v.befunde.size() == 1 && v.befunde[0].quelle == "modell");
  // Without expected a model's count_mismatch is dropped, other findings
  // stay.
  v = two;
  v.befunde.push_back({"count_mismatch", "x", "hoch", "modell"});
  v.befunde.push_back({"duplicate_person", "Bernd zweimal", "mittel", "modell"});
  vision::CheckPersonCount(plain, v);
  assert(v.befunde.size() == 1 && v.befunde[0].art == "duplicate_person");
  assert(vision::PersonUrteil(v) == "pruefen");
  // Matching count: nothing added.
  v = two;
  v.personen = seen(1);
  vision::CheckPersonCount(with, v);
  assert(v.befunde.empty() && vision::PersonUrteil(v) == "ok");
  v.sicher = false;
  assert(vision::PersonUrteil(v) == "pruefen");
  v.sicher = true;
  v.befunde.push_back({"extra_limb", "rechts", "niedrig", "modell"});
  assert(vision::PersonUrteil(v) == "pruefen");
  v.befunde.push_back({"extra_limb", "links", "hoch", "modell"});
  assert(vision::PersonUrteil(v) == "fehler");

  // CheckPersons with a scripted chat.
  gufo::models::ccip::RgbImage image;
  image.width = 64;
  image.height = 48;
  image.rgb.assign(static_cast<std::size_t>(64) * 48 * 3, 120);
  gufo::models::ccip::RgbImage reference = image;
  reference.width = 16;
  reference.height = 32;
  reference.rgb.assign(static_cast<std::size_t>(16) * 32 * 3, 60);
  std::vector<HttpRequest> asked;
  std::string answer = PersonVerdictText(
      "{\"stelle\": \"links\", \"wer\": \"Bernd\"}, {\"stelle\": \"rechts\", \"wer\": \"Bernd\"}",
      "{\"art\": \"duplicate_person\", \"stelle\": \"Bernd links und rechts\", "
      "\"sicherheit\": \"hoch\"}",
      true, "Bernd zweimal");
  const std::function<HttpResponse(const HttpRequest&)> serve =
      [&asked, &answer](const HttpRequest& inner) {
        asked.push_back(inner);
        HttpResponse streamed;
        streamed.headers = {{"Content-Type", "text/event-stream"}};
        const std::string text = answer;
        streamed.streaming_body = [text](const HttpResponse::BodyWriter& write) {
          (void)write(StreamAnswer(text));
        };
        return streamed;
      };
  HttpRequest outer = Post(Body("AAAA", {"person_count"}));
  outer.headers.emplace_back("Cookie", "gufo_session=abc");
  outer.headers.emplace_back("X-Other", "nicht weiter");
  vision::PersonCountRequest one_bernd = plain;
  one_bernd.expected = 1;
  one_bernd.references = {{"Bernd", "AAAA"}};
  json::Value entry =
      vision::CheckPersons(one_bernd, image, {reference}, 2, serve, outer);
  assert(asked.size() == 1 && asked[0].path == "/v1/chat/completions");
  bool cookie = false;
  bool other = false;
  for (const auto& [name, value] : asked[0].headers) {
    cookie = cookie || (name == "Cookie" && value == "gufo_session=abc");
    other = other || name == "X-Other";
  }
  assert(cookie && !other);
  const json::Value sent = json::parse(asked[0].body);
  const json::Value& sent_parts = *sent.find("messages")->items()[0].find("content");
  assert(sent_parts.size() == 5);
  assert(sent_parts.items()[1].find("image_url")->member_str("url").starts_with(
      "data:image/png;base64,"));
  assert(sent_parts.items()[2].member_str("text") == "Referenz Bernd:");
  assert(entry.member_double("personen") == 2 && entry.member_double("erwartet") == 1);
  assert(entry.member_double("detektor_personen") == 2);
  assert(entry.find("liste")->size() == 2);
  assert(entry.find("liste")->items()[1].member_str("wer") == "Bernd");
  // The model's duplicate plus gufo's count_mismatch.
  assert(entry.find("befunde")->size() == 2);
  assert(entry.find("befunde")->items()[0].member_str("art") == "duplicate_person");
  assert(entry.find("befunde")->items()[1].member_str("art") == "count_mismatch");
  assert(entry.find("befunde")->items()[1].member_str("quelle") == "gufo");
  assert(entry.member_str("urteil") == "fehler" && entry.find("sicher")->as_bool());
  assert(entry.member_str("befund") == "Bernd zweimal");
  assert(entry.find("seconds") != nullptr && entry.find("waited_seconds") != nullptr);
  assert(entry.find("fehler") == nullptr);
  // A good picture without expected: ok, no erwartet.
  answer = PersonVerdictText("{\"stelle\": \"Mitte\", \"wer\": \"Mann im Cardigan\"}", "", true,
                             "eine Person, alles normal");
  entry = vision::CheckPersons(plain, image, {}, 1, serve, outer);
  assert(entry.member_str("urteil") == "ok" && entry.find("erwartet") == nullptr);
  assert(entry.find("befunde")->empty() && entry.member_double("personen") == 1);
  // Garbage, a refusal, no chat, a cancelled caller: fehler, no exception.
  answer = "kein JSON";
  entry = vision::CheckPersons(plain, image, {}, 1, serve, outer);
  assert(entry.member_str("fehler").find("no JSON object") != std::string::npos);
  assert(entry.find("urteil") == nullptr && entry.member_double("detektor_personen") == 1);
  const std::function<HttpResponse(const HttpRequest&)> refuse = [](const HttpRequest&) {
    return JsonAnswer(404, "{\"error\": {\"message\": \"model not found\"}}");
  };
  entry = vision::CheckPersons(one_bernd, image, {reference}, 0, refuse, outer);
  assert(entry.member_str("fehler") == "chat 404: model not found");
  assert(entry.member_double("erwartet") == 1);
  entry = vision::CheckPersons(plain, image, {}, 0, nullptr, outer);
  assert(entry.member_str("fehler") == "no chat on this router");
  HttpRequest gone = outer;
  gone.is_cancelled = [] { return true; };
  asked.clear();
  entry = vision::CheckPersons(plain, image, {}, 0, serve, gone);
  assert(asked.empty() && entry.member_str("fehler") == "cancelled");
  // models.person_count.
  const json::Value info = vision::PersonCountModelInfo(one_bernd);
  assert(info.member_str("model") == "qwen3.8-27b" && info.member_str("detector") == "person");
  assert(info.member_double("references") == 1 && info.find("thinking")->as_bool());
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
  // A wrong pool key (401/403) or a partner error (500): back here as well.
  for (int status : {401, 403, 500}) {
    partner->status = status;
    response = service.Handle(Post(Body("AAAA", {"face"})));
    assert(response->status == 503 && response->body.find("vision_unavailable") != std::string::npos);
  }
  // A 400 of the partner (the request is wrong) is the answer.
  partner->status = 400;
  response = service.Handle(Post(Body("AAAA", {"face"})));
  assert(response->status == 400);
  partner->status = 0;
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

/// A grey RGB PNG of width x height (compresses to a few hundred KB).
std::string UniformPng(int width, int height) {
  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  image.width = static_cast<png_uint_32>(width);
  image.height = static_cast<png_uint_32>(height);
  image.format = PNG_FORMAT_RGB;
  const std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * 3, 128);
  png_alloc_size_t size = 0;
  assert(png_image_write_to_memory(&image, nullptr, &size, 0, pixels.data(), 0, nullptr));
  std::string out(size, '\0');
  assert(png_image_write_to_memory(&image, out.data(), &size, 0, pixels.data(), 0, nullptr));
  out.resize(size);
  return out;
}

/// All pictures of one request share kMaxRequestBytes (encoded): four
/// Ablage pictures of 20 MB each are refused before anything is decoded.
/// A page user (session cookie) reads an Ablage path with its own session.
void TestAblageCookie() {
  vision::Options options;
  std::string cookie;
  options.serve = [&cookie](const HttpRequest& inner) {
    cookie = inner.header("Cookie");
    HttpResponse response;
    response.status = 404;
    response.reason = "Not Found";
    response.body = R"({"error":{"code":"not_found","message":"no such job"}})";
    return response;
  };
  const vision::VisionService service(options);
  HttpRequest request = Post(Body("/v1/jobs/abc/files/a.png", {"face"}));
  request.headers.emplace_back("Cookie", "gufo_sitzung=s-1");
  assert(service.Handle(request)->status == 404);
  assert(cookie == "gufo_sitzung=s-1");
}

void TestRequestBytes() {
  vision::Options options;
  options.serve = [](const HttpRequest&) {
    HttpResponse response;
    response.body.assign(static_cast<std::size_t>(20'000'000), 'x');
    response.headers = {{"Content-Type", "image/png"}};
    return response;
  };
  const vision::VisionService service(options);
  json::Value body = Body("/v1/jobs/abc/files/a.png", {"wd14_tags"});
  json::Value opt = json::Value::object();
  json::Value pictures = json::Value::array();
  for (const char* name : {"/v1/jobs/abc/files/b.png", "/v1/jobs/abc/files/c.png",
                           "/v1/jobs/abc/files/d.png"})
    pictures.push_back(std::string(name));
  opt["images"] = std::move(pictures);
  body["wd14_tags"] = std::move(opt);
  const auto response = service.Handle(Post(body));
  assert(response->status == 400);
  assert(response->body.find("together exceed 64 MB") != std::string::npos);
  assert(response->body.find("wd14_tags.images[2]") != std::string::npos);
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
    // kMaxOcrPixels holds for a pass without enlargement too: 400.
    {
      const std::string large = "data:image/png;base64," +
                                gufo::core::EncodeBase64(std::string_view(UniformPng(6000, 4100)));
      const auto refused = service.Handle(Post(Body(large, {"ocr"})));
      assert(refused->status == 400);
      assert(refused->body.find("ocr scale 1: 6000x4100 is more than 24 megapixels") !=
             std::string::npos);
    }
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
    // latin_v5 (German umlauts): its own recognizer beside ch_v4 in one
    // request, 503 naming the file when it is not converted.
    json::Value both = Body(data_url, {"ocr"});
    both["ocr"] = json::parse(R"([{"rec": "latin_v5"}, {"rec": "ch_v4"}])");
    response = service.Handle(Post(both));
    if (std::filesystem::exists(std::filesystem::path(dir) / "latin_PP-OCRv5_mobile_rec.dict.txt")) {
      assert(response->status == 200);
      const json::Value jl = json::parse(response->body);
      const auto& latin_infos = jl.find("models")->find("ocr")->items();
      assert(latin_infos[0].member_str("rec") == "latin_v5" &&
             latin_infos[0].member_str("rec_model") == "latin_PP-OCRv5_mobile_rec" &&
             latin_infos[0].member_str("rec_repo") == "monkt/paddleocr-onnx");
      assert(latin_infos[1].member_str("rec_model") == "ch_PP-OCRv4_rec");
      // Same detector: the same boxes in both passes.
      const auto& lists = jl.find("detections")->find("ocr")->items();
      assert(lists[0].size() == lists[1].size());
      for (const json::Value& t : lists[0].items())
        std::cout << "latin_v5: " << t.member_str("text") << '\n';
    } else {
      assert(response->status == 503);
      assert(response->body.find("latin_PP-OCRv5_mobile_rec.safetensors") != std::string::npos);
    }
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
    // Decoding waits for the CPU turn: a broken picture is only refused
    // once the request's turn has come (waiting requests hold no pixels).
    assert(queue->Enter({}) == cpu::CpuQueue::Result::kOk);
    auto waiting = std::async(std::launch::async, [&] {
      return busy_service.Handle(Post(Body("AAAA", {"face"})));
    });
    assert(waiting.wait_for(std::chrono::milliseconds(400)) == std::future_status::timeout);
    queue->Leave();
    assert(waiting.get()->status == 400);
  }
  // All pictures of one request together stay within kMaxRequestPixels:
  // 32 MP each, the image and two wd14_tags pictures (96 MP) are refused.
  {
    const std::string big = "data:image/png;base64," +
                            gufo::core::EncodeBase64(std::string_view(UniformPng(8000, 4000)));
    json::Value many = Body(big, {"wd14_tags"});
    json::Value opt = json::Value::object();
    json::Value pictures = json::Value::array();
    pictures.push_back(big);
    pictures.push_back(big);
    opt["images"] = std::move(pictures);
    many["wd14_tags"] = std::move(opt);
    response = service.Handle(Post(many));
    assert(response->status == 400);
    assert(response->body.find("megapixels") != std::string::npos);
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
  // pose_boxes: the boxes RT-DETR found, given back, give the same points
  // (ViTPose / DWPose in exactly these boxes); no "score", no RT-DETR info.
  {
    const auto found = service.Handle(Post(Body(data_url, {"pose", "pose_wholebody"})));
    assert(found->status == 200);
    const json::Value jf = json::parse(found->body);
    const auto& poses = jf.find("poses")->items();
    std::cout << "pose persons found: " << poses.size() << '\n';
    if (!poses.empty()) {
      json::Value boxes = json::Value::array();
      for (const json::Value& p : poses)
        boxes.push_back(*p.find("box"));
      json::Value again = Body(data_url, {"pose", "pose_wholebody"});
      again["pose_boxes"] = std::move(boxes);
      const auto given = service.Handle(Post(again));
      assert(given->status == 200);
      const json::Value jg = json::parse(given->body);
      const auto& gposes = jg.find("poses")->items();
      assert(gposes.size() == poses.size());
      double worst = 0;
      for (std::size_t i = 0; i < poses.size(); ++i) {
        assert(gposes[i].find("score") == nullptr);
        for (const char* key : {"keypoints", "wholebody"}) {
          const auto& a = poses[i].find(key)->items();
          const auto& b = gposes[i].find(key)->items();
          assert(a.size() == b.size());
          for (std::size_t k = 0; k < a.size(); ++k)
            for (std::size_t c = 0; c < 3; ++c)
              worst = std::max(worst, std::fabs(a[k].items()[c].as_double() -
                                                b[k].items()[c].as_double()));
        }
      }
      std::cout << "pose_boxes vs RT-DETR boxes: max diff " << worst << '\n';
      assert(worst <= 1e-4);  // the boxes round-trip through JSON doubles
      assert(jg.find("models")->find("pose_detector")->member_str("model") == "pose_boxes");
    }
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
  TestRequestBytes();
  TestAblageCookie();
  TestHandFingers();
  TestPersonCount();
  const char* dir = std::getenv("GUFO_VISION_DIR");
  const char* frame = std::getenv("GUFO_VISION_FRAME");
  if (dir != nullptr && *dir != '\0' && frame != nullptr && *frame != '\0')
    TestReal(dir, frame);
  else
    std::cout << "real detections skipped (set GUFO_VISION_DIR and GUFO_VISION_FRAME)\n";
  std::cout << "router_vision_test: ok\n";
  return 0;
}
