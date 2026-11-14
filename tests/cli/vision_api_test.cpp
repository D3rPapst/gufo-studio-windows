// `gufo serve vision` request handling with fake models: fields, errors,
// hf.py's depth normalisation and segment-by-labels rules. No GPU.
// Written with AI assistance (Claude Code).
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <png.h>

#include "src/cli/serve/media/vision_api.hpp"
#include "src/core/base64.hpp"
#include "src/core/json.hpp"

using gufo::server::HttpRequest;
using gufo::server::HttpResponse;
using gufo::server::VisionBackends;
using gufo::server::VisionBox;
using gufo::server::VisionMask;
using gufo::server::VisionModelService;
namespace json = gufo::json;

namespace {

// 4 x 2 RGB PNG as base64.
std::string TinyPng() {
  std::vector<unsigned char> rgb(4 * 2 * 3, 128);
  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  image.width = 4;
  image.height = 2;
  image.format = PNG_FORMAT_RGB;
  png_alloc_size_t size = 0;
  assert(png_image_write_to_memory(&image, nullptr, &size, 0, rgb.data(), 0, nullptr));
  std::vector<unsigned char> png(size);
  assert(png_image_write_to_memory(&image, png.data(), &size, 0, rgb.data(), 0, nullptr));
  png.resize(size);
  return gufo::core::EncodeBase64(std::span<const std::uint8_t>(png.data(), png.size()));
}

HttpResponse Post(VisionModelService& service, const std::string& path, const std::string& body) {
  HttpRequest request;
  request.method = "POST";
  request.path = path;
  request.body = body;
  return service.Handle(request);
}

json::Value Body(const HttpResponse& response) { return json::parse(response.body); }

}  // namespace

int main() {
  const std::string image = TinyPng();
  std::size_t segment_boxes = 0;
  VisionBackends backends;
  backends.depth_sizes = {"large", "small"};
  backends.depth = [](const std::string& size, const gufo::core::Image& img,
                      std::vector<float>* depth, std::string*) {
    depth->resize(static_cast<std::size_t>(img.width) * img.height);
    for (std::size_t i = 0; i < depth->size(); ++i)
      (*depth)[i] = static_cast<float>(i) * (size == "small" ? 2.0F : 1.0F);
    return true;
  };
  backends.find = [](const gufo::core::Image&, const gufo::server::VisionFindRequest& request,
                     std::vector<VisionBox>* boxes, std::string*) {
    assert(request.labels.size() == 2 && request.labels[0] == "person");
    *boxes = {{{0, 0, 1, 1}, 0.4F, "person"},
              {{0, 0, 4, 2}, 0.9F, "person"},  // the whole picture: dropped by segment
              {{1, 0, 2, 1}, 0.8F, "sword"},
              {{2, 0, 3, 1}, 0.5F, "person"}};
    return true;
  };
  backends.segment = [&](const gufo::core::Image& img, const gufo::server::VisionSegmentPrompt& p,
                         std::vector<VisionMask>* masks, std::string*) {
    segment_boxes = p.boxes.size();
    for (const auto& b : p.boxes) {
      VisionMask m;
      m.width = static_cast<int>(img.width);
      m.height = static_cast<int>(img.height);
      m.mask.assign(img.width * img.height, 0);
      m.mask[static_cast<int>(b[0])] = 1;
      m.score = 0.75F;
      masks->push_back(m);
    }
    return true;
  };
  VisionModelService service("vision", backends);

  // Depth: percentile normalisation over the map, PNG answer, sizes.
  auto r = Post(service, "/v1/vision/depth", "{\"image\":\"" + image + "\",\"size\":\"small\"}");
  assert(r.status == 200);
  auto body = Body(r);
  assert(body.member_str("size") == "small");
  assert(body.find("data")->items().size() == 1);
  const json::Value& item = body.find("data")->items()[0];
  assert(item.member_size("width") == 4 && item.member_str("format") == "png16");
  assert(std::abs(item.member_double("max") - 14.0) < 1e-6);
  r = Post(service, "/v1/vision/depth", "{\"image\":\"" + image + "\",\"size\":\"huge\"}");
  assert(r.status == 400);
  r = Post(service, "/v1/vision/depth", "{\"images\":[\"" + image + "\",\"" + image + "\"],\"format\":\"f32\"}");
  assert(r.status == 200 && Body(r).find("data")->items().size() == 2);
  {
    std::vector<float> a = {0, 1, 2, 3, 4};
    double lo = 0, hi = 0;
    gufo::server::DepthPercentiles({&a}, &lo, &hi);
    assert(lo == 0 && hi == 0);  // every 97th value: only a[0]
  }

  // Find: sorted by score.
  r = Post(service, "/v1/vision/find", "{\"image\":\"" + image + "\",\"labels\":[\"Person\",\"sword.\"]}");
  assert(r.status == 200);
  body = Body(r);
  const auto& dets = body.find("detections")->items();
  assert(dets.size() == 4 && dets[0].member_double("score") > 0.89);
  r = Post(service, "/v1/vision/find", "{\"image\":\"" + image + "\"}");
  assert(r.status == 400);

  // Segment by labels: whole-picture box dropped, top 1 per label, union mask.
  r = Post(service, "/v1/vision/segment",
           "{\"image\":\"" + image + "\",\"labels\":[\"person\",\"sword\"],\"top\":1}");
  assert(r.status == 200);
  body = Body(r);
  assert(segment_boxes == 2);
  assert(body.find("found")->find("person")->as_double() == 1);
  assert(body.find("found")->find("sword")->as_double() == 1);
  assert(body.find("mask")->is_string());
  // Segment by boxes.
  r = Post(service, "/v1/vision/segment",
           "{\"image\":\"" + image + "\",\"boxes\":[[1,0,2,1],[3,0,4,1]]}");
  assert(r.status == 200 && Body(r).find("masks")->items().size() == 2);
  r = Post(service, "/v1/vision/segment", "{\"image\":\"" + image + "\"}");
  assert(r.status == 400);
  r = Post(service, "/v1/vision/segment",
           "{\"image\":\"" + image + "\",\"boxes\":[[1,0,2,1]],\"points\":[[1,1]]}");
  assert(r.status == 400);

  // Missing models are 501.
  VisionModelService bare("vision", VisionBackends{});
  assert(Post(bare, "/v1/vision/depth", "{\"image\":\"" + image + "\"}").status == 501);
  assert(Post(bare, "/v1/vision/find", "{\"image\":\"" + image + "\",\"text\":\"a.\"}").status == 501);
  HttpRequest get;
  get.method = "GET";
  get.path = "/v1/vision/depth";
  assert(service.Handle(get).status == 405);
  std::puts("vision_api_test: ok");
  return 0;
}
