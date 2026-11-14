// The GPU models of `gufo serve vision` behind VisionBackends (see
// vision_api.hpp): each loads on its first request and stays loaded.
// Written with AI assistance (Claude Code).
#include <map>
#include <memory>
#include <mutex>

#include "src/cli/serve/media/vision_api.hpp"

#if defined(GUFO_HAVE_VISION_DSG)
#include "src/models/vision_dsg/depth_anything.hpp"
#include "src/models/vision_dsg/grounding_dino.hpp"
#include "src/models/vision_dsg/sam.hpp"
#endif

namespace gufo::server {
namespace {

/// Hugging Face repo and license of each model (answers' `source`).
json::Value Source(const std::string& task, const std::string& size) {
  json::Value out = json::Value::object();
  if (task == "depth" && size.rfind("distill", 0) == 0) {
    out["repo"] = "xingyang1/Distill-Any-Depth-Large-hf";
    out["license"] = "MIT";
  } else if (task == "depth") {
    const std::string name = size == "small" ? "Small" : size == "base" ? "Base" : "Large";
    out["repo"] = "depth-anything/Depth-Anything-V2-" + name + "-hf";
    out["license"] = size == "small" ? "Apache-2.0" : "CC-BY-NC-4.0";
    if (size != "small")
      out["note"] = "non-commercial use only";
  } else if (task == "find" && size.find("llmdet") != std::string::npos) {
    out["repo"] = "iSEE-Laboratory/llmdet_base";
    out["license"] = "Apache-2.0";
  } else if (task == "find") {
    out["repo"] = "IDEA-Research/grounding-dino-base";
    out["license"] = "Apache-2.0";
  } else {
    out["repo"] = "facebook/sam-vit-huge";
    out["license"] = "Apache-2.0";
  }
  return out;
}

}  // namespace

bool BuildVisionBackends(const VisionServeOptions& options,
                         VisionBackends* backends, std::string* error) {
#if defined(GUFO_HAVE_VISION_DSG)
  using models::vision_dsg::DepthAnything;
  using models::vision_dsg::DepthResult;
  *backends = {};
  backends->describe = Source;
  if (!options.depth.empty()) {
    struct DepthModels {
      std::mutex mutex;
      std::map<std::string, std::filesystem::path> dirs;
      std::map<std::string, std::unique_ptr<DepthAnything>> loaded;
    };
    auto models = std::make_shared<DepthModels>();
    for (const auto& [name, dir] : options.depth) {
      if (models->dirs.count(name) != 0) {
        *error = "--depth-model name '" + name + "' is repeated";
        return false;
      }
      models->dirs[name] = dir;
      backends->depth_sizes.push_back(name);
    }
    backends->depth = [models](const std::string& size, const core::Image& image,
                               std::vector<float>* depth, std::string* failure) {
      std::lock_guard lock(models->mutex);
      auto& model = models->loaded[size];
      if (model == nullptr) {
        model = DepthAnything::Load(models->dirs.at(size), failure);
        if (model == nullptr)
          return false;
      }
      DepthResult result;
      if (!model->Run(image.pixels.data(), static_cast<int>(image.width),
                      static_cast<int>(image.height), &result, failure))
        return false;
      *depth = std::move(result.depth);
      return true;
    };
  }
  if (!options.find.empty()) {
    struct Find {
      std::mutex mutex;
      std::map<std::string, std::filesystem::path> dirs;
      std::map<std::string, std::unique_ptr<models::vision_dsg::GroundingDino>> loaded;
      std::string first;
    };
    auto find = std::make_shared<Find>();
    for (const auto& [name, dir] : options.find) {
      if (find->dirs.count(name) != 0) {
        *error = "--find-model name '" + name + "' is repeated";
        return false;
      }
      find->dirs[name] = dir;
      backends->find_names.push_back(name);
    }
    find->first = options.find.front().first;
    backends->find = [find](const core::Image& image, const VisionFindRequest& request,
                            std::vector<VisionBox>* boxes, std::string* failure) {
      std::lock_guard lock(find->mutex);
      const std::string name = request.detector.empty() ? find->first : request.detector;
      const auto dir = find->dirs.find(name);
      if (dir == find->dirs.end()) {
        *failure = "unknown detector " + name;
        return false;
      }
      auto& model = find->loaded[name];
      if (model == nullptr) {
        model = models::vision_dsg::GroundingDino::Load(dir->second, failure);
        if (model == nullptr)
          return false;
      }
      // hf.py: ". ".join(labels) + "."
      std::string text;
      for (const std::string& label : request.labels)
        text += (text.empty() ? "" : ". ") + label;
      text += ".";
      std::vector<models::vision_dsg::GroundingBox> found;
      if (!model->Run(image.pixels.data(), static_cast<int>(image.width),
                      static_cast<int>(image.height), text, request.box_threshold,
                      request.text_threshold, &found, nullptr, failure))
        return false;
      boxes->clear();
      for (auto& f : found)
        boxes->push_back({f.box, f.score, std::move(f.label)});
      return true;
    };
  }
  if (!options.segment.empty()) {
    struct Segment {
      std::mutex mutex;
      std::filesystem::path dir;
      std::unique_ptr<models::vision_dsg::Sam> model;
    };
    auto segment = std::make_shared<Segment>();
    segment->dir = options.segment;
    backends->segment = [segment](const core::Image& image, const VisionSegmentPrompt& prompt,
                                  std::vector<VisionMask>* masks, std::string* failure) {
      std::lock_guard lock(segment->mutex);
      if (segment->model == nullptr) {
        segment->model = models::vision_dsg::Sam::Load(segment->dir, failure);
        if (segment->model == nullptr)
          return false;
      }
      models::vision_dsg::SamPrompt p;
      p.boxes = prompt.boxes;
      p.points = prompt.points;
      p.labels = prompt.point_labels;
      p.multimask = prompt.multimask;
      models::vision_dsg::SamOutput out;
      if (!segment->model->Run(image.pixels.data(), static_cast<int>(image.width),
                               static_cast<int>(image.height), p, &out, failure))
        return false;
      masks->clear();
      for (std::size_t i = 0; i < out.masks.size(); ++i)
        masks->push_back({out.width, out.height, std::move(out.masks[i]), out.scores[i]});
      return true;
    };
  }
  return true;
#else
  (void)options;
  (void)backends;
  *error = "this gufo build has no vision models (needs HIP)";
  return false;
#endif
}

}  // namespace gufo::server
