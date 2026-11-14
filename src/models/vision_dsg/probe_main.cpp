// vision-dsg-probe: one model, one image, raw outputs as float32 files for
// the oracle comparison (src/models/vision_dsg/reference/compare.py).
//   vision-dsg-probe depth <model_dir> <image> <out_prefix> [repeats]
//   vision-dsg-probe find <model_dir> <image> <text> <out_prefix> [repeats]
//   vision-dsg-probe segment <model_dir> <image> <x0,y0,x1,y1;...> <out_prefix> [repeats]
// Written with AI assistance (Claude Code).
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/vision_dsg/depth_anything.hpp"
#include "src/models/vision_dsg/grounding_dino.hpp"
#include "src/models/vision_dsg/sam.hpp"

namespace {

using gufo::models::vision_dsg::DepthAnything;
using gufo::models::vision_dsg::DepthResult;

double Since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t)
      .count();
}

bool WriteFloats(const std::string& path, const std::vector<float>& values) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
  return static_cast<bool>(out);
}

int Depth(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: vision-dsg-probe depth <model_dir> <image> <out_prefix> [repeats]\n");
    return 2;
  }
  const int repeats = argc > 5 ? std::atoi(argv[5]) : 1;
  gufo::core::Image image;
  try {
    image = gufo::core::DecodeImage(gufo::core::ReadImageFile(argv[3]));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "image: %s\n", e.what());
    return 1;
  }
  const int w = static_cast<int>(image.width), h = static_cast<int>(image.height);
  std::string error;
  auto t = std::chrono::steady_clock::now();
  auto model = DepthAnything::Load(argv[2], &error);
  if (model == nullptr) {
    std::fprintf(stderr, "load: %s\n", error.c_str());
    return 1;
  }
  std::printf("load %.2f s, %s, %.1f MB weights\n", Since(t), model->size_name().c_str(),
              model->weight_bytes() / 1e6);
  int nw = 0, nh = 0;
  const std::vector<float> pixels =
      gufo::models::vision_dsg::DepthPixels(image.pixels.data(), w, h, &nw, &nh);
  DepthResult result;
  for (int r = 0; r < std::max(1, repeats); ++r) {
    t = std::chrono::steady_clock::now();
    if (!model->Run(image.pixels.data(), w, h, &result, &error)) {
      std::fprintf(stderr, "run: %s\n", error.c_str());
      return 1;
    }
    std::printf("run %d: %.3f s (net %dx%d)\n", r, Since(t), nw, nh);
  }
  const std::string prefix = argv[4];
  if (!WriteFloats(prefix + ".pixel.f32", pixels) ||
      !WriteFloats(prefix + ".pred.f32", result.raw) ||
      !WriteFloats(prefix + ".full.f32", result.depth)) {
    std::fprintf(stderr, "cannot write %s.*\n", prefix.c_str());
    return 1;
  }
  return 0;
}

bool LoadImage(const char* path, gufo::core::Image* image) {
  try {
    *image = gufo::core::DecodeImage(gufo::core::ReadImageFile(path));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "image: %s\n", e.what());
    return false;
  }
}

int Find(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr, "usage: vision-dsg-probe find <model_dir> <image> <text> <out_prefix> [repeats]\n");
    return 2;
  }
  const int repeats = argc > 6 ? std::atoi(argv[6]) : 1;
  gufo::core::Image image;
  if (!LoadImage(argv[3], &image))
    return 1;
  std::string error;
  auto t = std::chrono::steady_clock::now();
  auto model = gufo::models::vision_dsg::GroundingDino::Load(argv[2], &error);
  if (model == nullptr) {
    std::fprintf(stderr, "load: %s\n", error.c_str());
    return 1;
  }
  std::printf("load %.2f s, %.1f MB weights\n", Since(t), model->weight_bytes() / 1e6);
  std::vector<gufo::models::vision_dsg::GroundingBox> boxes;
  gufo::models::vision_dsg::GroundingRaw raw;
  for (int r = 0; r < std::max(1, repeats); ++r) {
    t = std::chrono::steady_clock::now();
    if (!model->Run(image.pixels.data(), static_cast<int>(image.width),
                    static_cast<int>(image.height), argv[4], 0.3F, 0.25F, &boxes, &raw,
                    &error)) {
      std::fprintf(stderr, "run: %s\n", error.c_str());
      return 1;
    }
    std::printf("run %d: %.3f s, %zu boxes\n", r, Since(t), boxes.size());
  }
  const std::string prefix = argv[5];
  if (!WriteFloats(prefix + ".logits.f32", raw.logits) ||
      !WriteFloats(prefix + ".boxes.f32", raw.boxes))
    return 1;
  std::ofstream out(prefix + ".dets.txt", std::ios::binary);
  for (const auto& b : boxes)
    out << b.score << '\t' << b.box[0] << '\t' << b.box[1] << '\t' << b.box[2] << '\t'
        << b.box[3] << '\t' << b.label << '\n';
  return 0;
}

int Segment(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr, "usage: vision-dsg-probe segment <model_dir> <image> <x0,y0,x1,y1;...> <out_prefix> [repeats]\n");
    return 2;
  }
  const int repeats = argc > 6 ? std::atoi(argv[6]) : 1;
  gufo::core::Image image;
  if (!LoadImage(argv[3], &image))
    return 1;
  gufo::models::vision_dsg::SamPrompt prompt;
  std::string spec = argv[4];
  std::size_t at = 0;
  while (at < spec.size()) {
    const std::size_t end = std::min(spec.find(';', at), spec.size());
    std::array<float, 4> box{};
    if (std::sscanf(spec.substr(at, end - at).c_str(), "%f,%f,%f,%f", &box[0], &box[1], &box[2],
                    &box[3]) == 4)
      prompt.boxes.push_back(box);
    at = end + 1;
  }
  std::string error;
  auto t = std::chrono::steady_clock::now();
  auto model = gufo::models::vision_dsg::Sam::Load(argv[2], &error);
  if (model == nullptr) {
    std::fprintf(stderr, "load: %s\n", error.c_str());
    return 1;
  }
  std::printf("load %.2f s, %.1f MB weights\n", Since(t), model->weight_bytes() / 1e6);
  gufo::models::vision_dsg::SamOutput out;
  for (int r = 0; r < std::max(1, repeats); ++r) {
    t = std::chrono::steady_clock::now();
    if (!model->Run(image.pixels.data(), static_cast<int>(image.width),
                    static_cast<int>(image.height), prompt, &out, &error)) {
      std::fprintf(stderr, "run: %s\n", error.c_str());
      return 1;
    }
    std::printf("run %d: %.3f s, %zu masks\n", r, Since(t), out.masks.size());
  }
  const std::string prefix = argv[5];
  std::vector<float> low, scores(out.scores.begin(), out.scores.end());
  for (const auto& l : out.low_res)
    low.insert(low.end(), l.begin(), l.end());
  std::ofstream masks(prefix + ".masks.u8", std::ios::binary);
  for (const auto& m : out.masks)
    masks.write(reinterpret_cast<const char*>(m.data()), static_cast<std::streamsize>(m.size()));
  return WriteFloats(prefix + ".lowres.f32", low) && WriteFloats(prefix + ".scores.f32", scores) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "depth")
    return Depth(argc, argv);
  if (mode == "find")
    return Find(argc, argv);
  if (mode == "segment")
    return Segment(argc, argv);
  std::fprintf(stderr, "usage: vision-dsg-probe depth|find|segment ...\n");
  return 2;
}
