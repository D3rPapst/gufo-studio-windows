// POST /v1/vision/detect "wd14_tags" (see vision_wd14.hpp).
// Written with AI assistance (Claude Code).
#include "src/cli/serve/router/vision_wd14.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace gufo::router::vision {
namespace {

double Probability(const json::Value& value, const std::string& what) {
  if (!value.is_number())
    throw std::invalid_argument(what + " must be a number");
  const double v = value.as_double();
  if (!(v >= 0.0 && v <= 1.0))
    throw std::invalid_argument(what + " must be in [0, 1]");
  return v;
}

json::Value TagObject(const models::ccip::TagList& tags) {
  json::Value out = json::Value::object();
  for (const auto& [name, p] : tags)
    out.append_member(name, json::Value(static_cast<double>(p)));
  return out;
}

void ParseRegions(const json::Value& value, Wd14Request* request, const std::string& what) {
  if (value.is_string()) {
    if (value.str() != "person")
      throw std::invalid_argument(what + ": \"person\" or a list of [x0, y0, x1, y1] boxes");
    request->persons = true;
    return;
  }
  if (!value.is_array())
    throw std::invalid_argument(what + ": \"person\" or a list of [x0, y0, x1, y1] boxes");
  for (const json::Value& box : value.items()) {
    if (!box.is_array() || box.size() != 4)
      throw std::invalid_argument(what + ": every box is [x0, y0, x1, y1]");
    std::array<double, 4> b{};
    for (std::size_t i = 0; i < 4; ++i) {
      if (!box.items()[i].is_number())
        throw std::invalid_argument(what + ": box values must be numbers");
      b[i] = box.items()[i].as_double();
    }
    if (!(b[2] > b[0] && b[3] > b[1]))
      throw std::invalid_argument(what + ": every box needs x1 > x0 and y1 > y0");
    request->boxes.push_back(b);
  }
  if (request->boxes.empty())
    throw std::invalid_argument(what + ": no boxes");
}

}  // namespace

Wd14Request ParseWd14(const json::Value& body, bool requested) {
  Wd14Request request;
  request.enabled = requested;
  const json::Value* options = body.find(kWd14Task);
  if (options == nullptr || options->is_null())
    return request;
  if (!requested)
    throw std::invalid_argument("wd14_tags options given, but the task is not requested");
  if (!options->is_object())
    throw std::invalid_argument("wd14_tags must be an object of options");
  bool regions_seen = false;
  for (const auto& [key, value] : options->members()) {
    const std::string what = "wd14_tags." + key;
    if (key == "general_threshold") {
      request.general_threshold = Probability(value, what);
    } else if (key == "character_threshold") {
      request.character_threshold = Probability(value, what);
    } else if (key == "regions" || key == "use_detections") {
      if (regions_seen)
        throw std::invalid_argument("wd14_tags: give regions or use_detections, not both");
      regions_seen = true;
      if (key == "use_detections" && !(value.is_string() && value.str() == "person"))
        throw std::invalid_argument(what + ": only \"person\"");
      ParseRegions(value, &request, what);
    } else if (key == "margin") {
      if (!value.is_number() || !(value.as_double() >= 0.0 && value.as_double() <= 1.0))
        throw std::invalid_argument(what + " must be a number in [0, 1]");
      request.margin = value.as_double();
    } else if (key == "images") {
      if (!value.is_array())
        throw std::invalid_argument(what + " must be a list of pictures");
      for (const json::Value& item : value.items()) {
        if (!item.is_string() || item.str().empty())
          throw std::invalid_argument(what + ": every picture is a data URL, base64, an https "
                                             "URL or an Ablage path");
        request.images.push_back(item.str());
      }
      if (request.images.size() > 16)
        throw std::invalid_argument(what + ": at most 16 pictures");
    } else {
      throw std::invalid_argument(what + ": use general_threshold, character_threshold, "
                                         "regions, use_detections, margin or images");
    }
  }
  return request;
}

json::Value Wd14Entry(const models::ccip::Wd14Tags& tags, const std::string& source, int index,
                      const std::array<int, 4>* box, int width, int height) {
  json::Value entry = json::Value::object();
  entry.append_member("source", json::Value(source));
  if (source != "image")
    entry.append_member("index", json::Value(index));
  if (box != nullptr) {
    json::Value b = json::Value::array();
    for (int v : *box)
      b.push_back(v);
    entry.append_member("box", std::move(b));
  }
  entry.append_member("width", json::Value(width));
  entry.append_member("height", json::Value(height));
  entry.append_member("rating", TagObject(tags.rating));
  entry.append_member("general", TagObject(tags.general));
  entry.append_member("character", TagObject(tags.character));
  return entry;
}

json::Value Wd14ModelInfo(const Wd14Request& request) {
  const auto round6 = [](double v) { return std::round(v * 1e6) / 1e6; };
  json::Value model = json::Value::object();
  model["repo"] = kWd14Repo;
  model["model"] = kWd14Model;
  model["general_threshold"] = round6(request.general_threshold);
  model["character_threshold"] = round6(request.character_threshold);
  if (request.persons || !request.boxes.empty())
    model["margin"] = round6(request.margin);
  return model;
}

json::Value RunWd14(const models::ccip::Wd14Tagger& net, const models::ccip::RgbImage& image,
                    const Wd14Request& request,
                    const std::vector<std::array<int, 4>>& persons,
                    const std::vector<models::ccip::RgbImage>& extra) {
  json::Value list = json::Value::array();
  const auto tag_region = [&](const std::array<double, 4>& region, const char* source, int index) {
    const std::array<int, 4> cut =
        models::ccip::MarginCropBox(region, request.margin, image.width, image.height);
    if (cut[2] <= cut[0] || cut[3] <= cut[1])
      throw std::invalid_argument(std::string("wd14_tags: ") + source + " " +
                                  std::to_string(index) + " lies outside the image");
    const models::ccip::RgbImage piece =
        models::ccip::Crop(image, cut[0], cut[1], cut[2], cut[3]);
    list.push_back(Wd14Entry(net.Tag(piece, request.general_threshold,
                                     request.character_threshold),
                             source, index, &cut, piece.width, piece.height));
  };
  if (request.persons) {
    for (std::size_t i = 0; i < persons.size(); ++i) {
      const auto& p = persons[i];
      tag_region({static_cast<double>(p[0]), static_cast<double>(p[1]),
                  static_cast<double>(p[2]), static_cast<double>(p[3])},
                 "person", static_cast<int>(i));
    }
  } else if (!request.boxes.empty()) {
    for (std::size_t i = 0; i < request.boxes.size(); ++i)
      tag_region(request.boxes[i], "region", static_cast<int>(i));
  } else {
    list.push_back(Wd14Entry(net.Tag(image, request.general_threshold,
                                     request.character_threshold),
                             "image", 0, nullptr, image.width, image.height));
  }
  for (std::size_t i = 0; i < extra.size(); ++i) {
    list.push_back(Wd14Entry(net.Tag(extra[i], request.general_threshold,
                                     request.character_threshold),
                             "images", static_cast<int>(i), nullptr, extra[i].width,
                             extra[i].height));
  }
  return list;
}

}  // namespace gufo::router::vision
