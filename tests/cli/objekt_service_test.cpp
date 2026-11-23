// gufo serve objekt (ObjektService) with a fake runner, no model, no GPU:
// numbers are range-checked before the int conversion (1e300 steps, a
// negative seed) and a DELETE while the job runs leaves no job directory;
// TRELLIS.2 options: `images` (views), `resolution` 1024 / 1536 per runner, `multi_mode`;
// print_ready (druckfertig): validation, defaults, print.stl / print.glb, restart.
// AI-assisted (Claude Code), 2026-11-12; views and resolutions 2026-11-22.
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "src/cli/serve/media/objekt_service.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen_image_21/model.hpp"

namespace {

int failures = 0;

#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::cerr << __FILE__ << ":" << __LINE__ << " CHECK " #cond "\n"; \
      ++failures;                                                       \
    }                                                                   \
  } while (0)

namespace json = gufo::json;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;

// Waits for the cancel, then still returns a result: a runner that notices
// the cancel only after its last step.
struct FakeRunner : gufo::triposg::ObjektRunner {
  std::atomic<bool> started{false};
  bool Run(const gufo::triposg::ObjektRequest&,
           const gufo::triposg::ObjektProgress&,
           const std::atomic<bool>* cancel, gufo::triposg::ObjektResult* result,
           std::string*) override {
    started = true;
    for (int i = 0; i < 2000 && !cancel->load(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    result->glb = {'g', 'l', 'T', 'F'};
    result->preview_png = {1, 2, 3};
    return true;
  }
  std::size_t weight_bytes() const override { return 0; }
};

std::string Base64(const std::vector<std::uint8_t>& data) {
  static constexpr char kTable[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  std::size_t i = 0;
  for (; i + 2 < data.size(); i += 3) {
    const unsigned v = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
    out += kTable[(v >> 18) & 63];
    out += kTable[(v >> 12) & 63];
    out += kTable[(v >> 6) & 63];
    out += kTable[v & 63];
  }
  if (i + 1 == data.size()) {
    const unsigned v = data[i] << 16;
    out += kTable[(v >> 18) & 63];
    out += kTable[(v >> 12) & 63];
    out += "==";
  } else if (i + 2 == data.size()) {
    const unsigned v = (data[i] << 16) | (data[i + 1] << 8);
    out += kTable[(v >> 18) & 63];
    out += kTable[(v >> 12) & 63];
    out += kTable[(v >> 6) & 63];
    out += '=';
  }
  return out;
}

std::string Png() {
  gufo::models::qwen_image_21::Image image;
  image.width = 4;
  image.height = 4;
  image.rgba.assign(4 * 4 * 4, 200);
  return Base64(gufo::models::qwen_image_21::EncodePng(image));
}

HttpResponse Post(gufo::server::ObjektService& service, const std::string& extra) {
  HttpRequest request;
  request.method = "POST";
  request.path = "/v1/3d/generations";
  request.body = R"({"model":"triposg","image":")" + Png() + "\"" + extra + "}";
  return service.Handle(request);
}

std::string Code(const HttpResponse& response) {
  try {
    return json::parse(response.body).find("error")->member_str("code");
  } catch (const std::exception&) {
    return "";
  }
}

void TestNumbersRangeChecked(gufo::server::ObjektService& service) {
  CHECK(Code(Post(service, R"(,"steps":1e300)")) == "invalid_steps");
  CHECK(Code(Post(service, R"(,"steps":-1e300)")) == "invalid_steps");
  CHECK(Code(Post(service, R"(,"steps":0)")) == "invalid_steps");
  CHECK(Code(Post(service, R"(,"seed":-1)")) == "invalid_seed");
  CHECK(Code(Post(service, R"(,"seed":1e30)")) == "invalid_seed");
  CHECK(Code(Post(service, R"(,"guidance":1e300)")) == "invalid_guidance");
  CHECK(Code(Post(service, R"(,"resolution":1e20)")) == "invalid_resolution");
  CHECK(Code(Post(service, R"(,"resolution":384)")) == "invalid_resolution");
  CHECK(Code(Post(service, R"(,"max_faces":1e20)")) == "invalid_max_faces");
  CHECK(Code(Post(service, R"(,"max_faces":-1)")) == "invalid_max_faces");
}

void TestDeleteWhileRunning(gufo::server::ObjektService& service,
                            FakeRunner& runner,
                            const std::filesystem::path& root) {
  const HttpResponse created = Post(service, R"(,"steps":10,"seed":7)");
  CHECK(created.status == 202);
  const std::string id = json::parse(created.body).member_str("id");
  for (int i = 0; i < 400 && !runner.started.load(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  CHECK(runner.started.load());
  CHECK(std::filesystem::exists(root / id));  // job.json while it runs
  HttpRequest remove;
  remove.method = "DELETE";
  remove.path = "/v1/3d/generations/" + id;
  CHECK(service.Handle(remove).status == 200);
  // The worker confirms the cancel by finishing Run(); then the directory
  // is gone and no result file was written.
  bool gone = false;
  for (int i = 0; i < 400 && !gone; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    gone = !std::filesystem::exists(root / id);
  }
  CHECK(gone);
  HttpRequest status;
  status.method = "GET";
  status.path = "/v1/3d/generations/" + id;
  CHECK(service.Handle(status).status == 404);
}

// TRELLIS.2-like runner: 1024 / 1536, up to 4 views; records the last request.
struct ViewsRunner : gufo::triposg::ObjektRunner {
  std::atomic<int> runs{0};
  int views = 0, resolution = 0, width2 = 0;
  std::string mode;
  bool Run(const gufo::triposg::ObjektRequest& r, const gufo::triposg::ObjektProgress& progress,
           const std::atomic<bool>*, gufo::triposg::ObjektResult* result, std::string*) override {
    views = 1 + static_cast<int>(r.more_views.size());
    resolution = r.resolution;
    mode = r.multi_mode;
    width2 = r.more_views.empty() ? 0 : r.more_views[0].width;
    progress("shape", 1, 2);
    progress("structure", 1, 2);   // an earlier phase reported late: progress must not go back
    result->glb = {'g', 'l', 'T', 'F'};
    result->resolution = r.resolution == 1536 ? 1408 : r.resolution;
    result->tokens = 1234;
    ++runs;
    return true;
  }
  std::size_t weight_bytes() const override { return 0; }
  std::vector<int> resolutions() const override { return {256, 512, 1024, 1536}; }
  int max_views() const override { return 4; }
};

HttpResponse PostBody(gufo::server::ObjektService& service, const std::string& body) {
  HttpRequest request;
  request.method = "POST";
  request.path = "/v1/3d/generations";
  request.body = body;
  return service.Handle(request);
}

json::Value WaitDone(gufo::server::ObjektService& service, const std::string& id) {
  HttpRequest status;
  status.method = "GET";
  status.path = "/v1/3d/generations/" + id;
  for (int i = 0; i < 400; ++i) {
    const json::Value v = json::parse(service.Handle(status).body);
    if (v.member_str("status") == "completed" || v.member_str("status") == "failed") return v;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return json::Value::object();
}

void TestViewsAndResolutions(gufo::server::ObjektService& service, ViewsRunner& runner) {
  const std::string png = Png();
  const std::string four = "[\"" + png + "\",\"data:image/png;base64," + png + "\",\"" + png + "\",\"" + png + "\"]";
  // TripoSG (the default runner) keeps 256 / 512 and one image
  CHECK(Code(Post(service, R"(,"resolution":1024)")) == "invalid_resolution");
  CHECK(Code(PostBody(service, R"({"model":"triposg","images":)" + four + "}")) == "too_many_images");
  // TRELLIS.2: validation
  CHECK(Code(PostBody(service, R"({"model":"trellis2","image":")" + png + R"(","images":)" + four + "}")) == "invalid_images");
  CHECK(Code(PostBody(service, R"({"model":"trellis2","images":[]})")) == "invalid_images");
  CHECK(Code(PostBody(service, R"({"model":"trellis2","images":"x"})")) == "invalid_images");
  CHECK(Code(PostBody(service, R"({"model":"trellis2","images":[")" + png + R"(",7]})")) == "invalid_image");
  CHECK(Code(PostBody(service, R"({"model":"trellis2","images":[")" + png + R"(","!!"]})")) == "invalid_image");
  CHECK(Code(PostBody(service, R"({"model":"trellis2","images":[")" + png + "\",\"" + png + "\",\"" + png + "\",\"" + png + "\",\"" + png + "\"]}")) == "too_many_images");
  CHECK(Code(PostBody(service, R"({"model":"trellis2","images":)" + four + R"(,"multi_mode":"concat"})")) == "invalid_multi_mode");
  CHECK(Code(PostBody(service, R"({"model":"trellis2","images":)" + four + R"(,"multi_mode":3})")) == "invalid_multi_mode");
  CHECK(Code(PostBody(service, R"({"model":"trellis2","image":")" + png + R"(","resolution":768})")) == "invalid_resolution");
  CHECK(Code(PostBody(service, R"({"model":"trellis2","image":")" + png + R"(","resolution":1024.5})")) == "invalid_resolution");
  CHECK(Code(PostBody(service, R"({"model":"trellis2","image":")" + png + R"(","resolution":2048})")) == "invalid_resolution");
  // four views, 1536, multidiffusion
  HttpResponse created = PostBody(service, R"({"model":"trellis2","images":)" + four + R"(,"resolution":1536,"multi_mode":"multidiffusion"})");
  CHECK(created.status == 202);
  json::Value snap = json::parse(created.body);
  CHECK(snap.member_double("views") == 4 && snap.member_str("multi_mode") == "multidiffusion" && snap.member_double("resolution") == 1536);
  json::Value done = WaitDone(service, snap.member_str("id"));
  CHECK(done.member_str("status") == "completed");
  CHECK(runner.views == 4 && runner.resolution == 1536 && runner.mode == "multidiffusion" && runner.width2 == 4);
  CHECK(done.member_double("progress") == 100);
  const json::Value* data = done.find("data");
  CHECK(data && data->is_array() && !data->empty() && data->items()[0].member_double("resolution") == 1408 &&
        data->items()[0].member_double("tokens") == 1234);
  // one image, 1024, default mode; old clients (no new fields) unchanged
  created = PostBody(service, R"({"model":"trellis2","image":")" + png + R"(","resolution":1024})");
  CHECK(created.status == 202);
  snap = json::parse(created.body);
  CHECK(snap.member_double("views") == 1 && snap.find("multi_mode") == nullptr);
  WaitDone(service, snap.member_str("id"));
  CHECK(runner.views == 1 && runner.resolution == 1024 && runner.mode == "stochastic");
  created = PostBody(service, R"({"model":"trellis2","images":[")" + png + R"("]})");
  CHECK(created.status == 202);
  WaitDone(service, json::parse(created.body).member_str("id"));
  CHECK(runner.views == 1 && runner.resolution == 512);
}

// Runner that hands back a closed, thin bar (Y up, 1 x 0.1 x 0.1) with vertex colours when the
// request is print_ready (as the TRELLIS.2 / TripoSG runners do).
struct PrintRunner : gufo::triposg::ObjektRunner {
  std::atomic<bool> last_print{false};
  bool Run(const gufo::triposg::ObjektRequest& r, const gufo::triposg::ObjektProgress&, const std::atomic<bool>*,
           gufo::triposg::ObjektResult* result, std::string*) override {
    last_print = r.print_ready;
    result->glb = {'g', 'l', 'T', 'F'};
    if (r.print_ready) {
      gufo::mesh::PrintMesh& m = result->print_source;
      for (int k = 0; k < 8; ++k) {
        m.positions.insert(m.positions.end(), {(k & 1) ? 0.05f : -0.05f, (k & 2) ? 1.0f : 0.0f, (k & 4) ? 0.05f : -0.05f});
        m.colors.insert(m.colors.end(), {255, 0, 0, 255});
      }
      const int quads[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}};
      for (const auto& q : quads) {
        m.indices.insert(m.indices.end(), {std::uint32_t(q[0]), std::uint32_t(q[1]), std::uint32_t(q[2])});
        m.indices.insert(m.indices.end(), {std::uint32_t(q[0]), std::uint32_t(q[2]), std::uint32_t(q[3])});
      }
    }
    return true;
  }
  std::size_t weight_bytes() const override { return 0; }
};

HttpResponse Get(gufo::server::ObjektService& service, const std::string& path) {
  HttpRequest request;
  request.method = "GET";
  request.path = path;
  return service.Handle(request);
}

std::string Header(const HttpResponse& response, const std::string& name) {
  for (const auto& [k, v] : response.headers)
    if (k == name) return v;
  return "";
}

json::Value WaitFinished(gufo::server::ObjektService& service, const std::string& id) {
  // The print step takes longer than the fake runs above.
  for (int i = 0; i < 100; ++i) {
    const json::Value v = WaitDone(service, id);
    if (v.member_str("status") == "completed" || v.member_str("status") == "failed") return v;
  }
  return json::Value::object();
}

const json::Value* PrintOf(const json::Value& done) {
  const json::Value* data = done.find("data");
  if (!data || !data->is_array() || data->empty()) return nullptr;
  return data->items()[0].find("print");
}

void TestPrintReady(gufo::server::ObjektService& service, PrintRunner& runner, const std::filesystem::path& root) {
  // validation
  CHECK(Code(Post(service, R"(,"print_ready":"yes")")) == "invalid_print_ready");
  CHECK(Code(Post(service, R"(,"print_ready":true,"print_height_mm":0.5)")) == "invalid_print_height");
  CHECK(Code(Post(service, R"(,"print_ready":true,"print_height_mm":"28")")) == "invalid_print_height");
  CHECK(Code(Post(service, R"(,"print_ready":true,"print_eye_height_mm":1e300)")) == "invalid_print_height");
  CHECK(Code(Post(service, R"(,"print_ready":true,"print_height_mm":30,"print_eye_height_mm":28)")) == "invalid_print_height");
  CHECK(Code(Post(service, R"(,"print_ready":true,"print_eye_ratio":2)")) == "invalid_print_eye_ratio");
  CHECK(Code(Post(service, R"(,"print_ready":true,"print_format":"3mf")")) == "invalid_print_format");
  // default off: old requests unchanged, no print files
  HttpResponse created = Post(service, "");
  CHECK(created.status == 202);
  json::Value snap = json::parse(created.body);
  CHECK(snap.find("print_ready") && !snap.find("print_ready")->as_bool() && snap.find("print_height_mm") == nullptr);
  json::Value done = WaitFinished(service, snap.member_str("id"));
  CHECK(done.member_str("status") == "completed" && !runner.last_print.load());
  CHECK(done.find("data") && PrintOf(done) == nullptr);
  CHECK(Code(Get(service, "/v1/3d/generations/" + snap.member_str("id") + "/print.stl")) == "objekt_print_missing");
  // eye height default 28 mm when no height is given, STL only
  created = Post(service, R"(,"print_ready":true,"print_format":"stl")");
  CHECK(created.status == 202);
  snap = json::parse(created.body);
  CHECK(snap.member_double("print_eye_height_mm") == 28 && snap.member_double("print_eye_ratio") == 0.93);
  done = WaitFinished(service, snap.member_str("id"));
  const json::Value* print = PrintOf(done);
  CHECK(print && print->member_str("status") == "completed" && print->find("stl_url") && !print->find("glb_url"));
  CHECK(print && std::fabs(print->member_double("height_mm") - 28 / 0.93) < 1e-6);
  CHECK(Code(Get(service, "/v1/3d/generations/" + snap.member_str("id") + "/print.glb")) == "objekt_print_missing");
  // total height 40 mm, both files
  created = Post(service, R"(,"print_ready":true,"print_height_mm":40)");
  CHECK(created.status == 202);
  const std::string id = json::parse(created.body).member_str("id");
  done = WaitFinished(service, id);
  CHECK(runner.last_print.load());
  print = PrintOf(done);
  CHECK(print && print->member_str("status") == "completed");
  CHECK(print && print->find("watertight") && print->find("watertight")->as_bool() && print->member_double("bodies") == 1);
  const json::Value* check = print ? print->find("check") : nullptr;
  CHECK(check && check->find("watertight")->as_bool() && check->find("winding_consistent")->as_bool() &&
        check->find("volume_positive")->as_bool() && check->member_double("bodies") == 1 && check->member_double("boundary_edges") == 0);
  CHECK(check && check->member_str("line") ==
                     "watertight yes, boundary edges 0, non-manifold edges 0, winding consistent yes, volume > 0 yes, bodies 1");
  CHECK(print && print->find("error") == nullptr);
  CHECK(print && std::fabs(print->member_double("height_mm") - 40) < 1e-6);
  CHECK(print && print->member_str("stl_url") == "/v1/3d/generations/" + id + "/print.stl" &&
        print->member_str("glb_url") == "/v1/3d/generations/" + id + "/print.glb");
  const HttpResponse stl = Get(service, "/v1/3d/generations/" + id + "/print.stl");
  CHECK(stl.status == 200 && Header(stl, "Content-Type") == "model/stl");
  CHECK(print && stl.body.size() == 84 + 50 * static_cast<std::size_t>(print->member_double("faces")));
  CHECK(Header(stl, "Content-Disposition").find(id + "-druck.stl") != std::string::npos);
  const HttpResponse glb = Get(service, "/v1/3d/generations/" + id + "/print.glb");
  CHECK(glb.status == 200 && glb.body.size() > 12 && glb.body.compare(0, 4, "glTF") == 0);
  CHECK(std::filesystem::exists(root / id / "print.stl") && std::filesystem::exists(root / id / "print.glb"));
}

}  // namespace

int main() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "gufo-objekt-service-test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  auto runner = std::make_shared<FakeRunner>();
  {
    gufo::server::ObjektServiceOptions options;
    options.runner = runner;
    options.storage_root = root;
    gufo::server::ObjektService service(options);
    CHECK(service.ready());
    TestNumbersRangeChecked(service);
    TestDeleteWhileRunning(service, *runner, root);
  }
  {
    auto views = std::make_shared<ViewsRunner>();
    gufo::server::ObjektServiceOptions options;
    options.runner = std::make_shared<FakeRunner>();
    options.runners["trellis2"] = views;
    options.storage_root = root / "views";
    gufo::server::ObjektService service(options);
    CHECK(service.ready());
    TestViewsAndResolutions(service, *views);
  }
  {
    auto printer = std::make_shared<PrintRunner>();
    gufo::server::ObjektServiceOptions options;
    options.runner = printer;
    options.storage_root = root / "print";
    std::string id;
    {
      gufo::server::ObjektService service(options);
      CHECK(service.ready());
      TestPrintReady(service, *printer, options.storage_root);
      HttpResponse created = Post(service, R"(,"print_ready":true,"print_height_mm":35)");
      id = json::parse(created.body).member_str("id");
      WaitFinished(service, id);
    }
    // A restarted worker still serves the print files of a finished job.
    gufo::server::ObjektService again(options);
    const json::Value v = json::parse(Get(again, "/v1/3d/generations/" + id).body);
    const json::Value* print = PrintOf(v);
    CHECK(print && print->member_str("status") == "completed" && std::fabs(print->member_double("height_mm") - 35) < 1e-6);
    CHECK(Get(again, "/v1/3d/generations/" + id + "/print.stl").status == 200);
  }
  std::filesystem::remove_all(root);
  if (failures != 0) {
    std::cerr << failures << " objekt service check(s) failed\n";
    return 1;
  }
  std::cout << "objekt_service_test: all passed\n";
  return 0;
}
