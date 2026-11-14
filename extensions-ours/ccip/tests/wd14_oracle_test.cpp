// /v1/vision/detect "wd14_tags" against dghs-imgutils (reference/make_wd14_oracle.py).
// Env: GUFO_VISION_DIR (dir with wd-swinv2-tagger-v3.safetensors/.json from
// reference/convert_wd14.py), GUFO_WD14_ORACLE (dir with manifest.json),
// optional GUFO_CCIP_THREADS. Exit 77 (skip) without them.
// Pass, per keyframe and per person cut-out (erkennung.py: 8 % margin):
// the cut-out box equal to Pillow's, the 448x448 BGR input bit-exact, every
// one of the 10861 probabilities within 1e-3 of imgutils, and the same
// rating / general (> 0.35) / character (> 0.85) tags. A tag whose imgutils
// probability lies within 1e-3 of its threshold may differ (reported).
// Written with AI assistance (Claude Code).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gufo/ext/json.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ccip/tensors.hpp"
#include "ccip/wd14.hpp"

namespace ccip = gufo::models::ccip;

namespace {

const char* Env(const char* name) {
  const char* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? v : nullptr;
}

struct Stats {
  double max_prob = 0;
  double max_emb = 0;
  double ms = 0;
  int cases = 0;
  int border = 0;
};

// Compares one picture; prints a line and returns pass/fail.
bool Check(const ccip::Wd14Tagger& net, const ccip::RgbImage& img, const ccip::TensorFile& f,
           const std::string& prefix, const gufo::json::Value& want, double gthr, double cthr,
           const std::string& label, Stats* stats) {
  const std::vector<float> x = ccip::Wd14Input(img, net.InputSize());
  const auto& ref_x = f.F32(prefix + "input448");
  std::size_t differ = x.size() == ref_x.size() ? 0 : x.size();
  for (std::size_t i = 0; i < x.size() && i < ref_x.size(); ++i) {
    if (x[i] != ref_x[i]) ++differ;
  }
  const auto a = std::chrono::steady_clock::now();
  std::vector<float> probs, emb;
  net.Forward(x.data(), &probs, &emb);
  const auto b = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(b - a).count();
  const auto& ref_p = f.F32(prefix + "prediction");
  const auto& ref_e = f.F32(prefix + "embedding");
  double max_p = 0, max_e = 0;
  for (std::size_t i = 0; i < probs.size() && i < ref_p.size(); ++i)
    max_p = std::max(max_p, std::fabs(static_cast<double>(probs[i]) - ref_p[i]));
  for (std::size_t i = 0; i < emb.size() && i < ref_e.size(); ++i)
    max_e = std::max(max_e, std::fabs(static_cast<double>(emb[i]) - ref_e[i]));
  const ccip::Wd14Tags got = net.Select(probs, gthr, cthr);
  bool same_tags = true;
  int border = 0;
  const auto compare = [&](const ccip::TagList& mine, const char* key, double thr) {
    std::set<std::string> g, w;
    for (const auto& [name, p] : mine) g.insert(name);
    for (const auto& [name, p] : want.find(key)->members()) w.insert(name);
    for (const auto& name : g) {
      if (w.count(name) != 0) continue;
      // Only in ours: allowed when imgutils' value is right below the threshold.
      double ref = -1;
      for (const auto& [n2, p] : mine)
        if (n2 == name) ref = p;
      if (std::fabs(ref - thr) <= 1e-3) {
        ++border;
        std::printf("    border: +%s %.6f\n", name.c_str(), ref);
      } else {
        same_tags = false;
        std::printf("    extra %s: %s %.6f\n", key, name.c_str(), ref);
      }
    }
    for (const auto& name : w) {
      if (g.count(name) != 0) continue;
      const double ref = want.find(key)->member_double(name);
      if (std::fabs(ref - thr) <= 1e-3) {
        ++border;
        std::printf("    border: -%s %.6f\n", name.c_str(), ref);
      } else {
        same_tags = false;
        std::printf("    missing %s: %s %.6f\n", key, name.c_str(), ref);
      }
    }
    return g.size();
  };
  const std::size_t nr = compare(got.rating, "rating", -1.0);
  const std::size_t ng = compare(got.general, "general", gthr);
  const std::size_t nc = compare(got.character, "character", cthr);
  const bool pass = differ == 0 && max_p <= 1e-3 && same_tags;
  std::printf("  %-24s input %zu differ  tags %zu/%zu/%zu  max prob diff %.2e  emb %.2e  %.0f ms  %s\n",
              label.c_str(), differ, nr, ng, nc, max_p, max_e, ms, pass ? "ok" : "FAIL");
  stats->max_prob = std::max(stats->max_prob, max_p);
  stats->max_emb = std::max(stats->max_emb, max_e);
  stats->ms += ms;
  stats->cases += 1;
  stats->border += border;
  return pass;
}

}  // namespace

int main() {
  const char* model_dir = Env("GUFO_VISION_DIR");
  const char* oracle_dir = Env("GUFO_WD14_ORACLE");
  if (!model_dir || !oracle_dir) {
    std::printf("skip: set GUFO_VISION_DIR and GUFO_WD14_ORACLE\n");
    return 77;
  }
#ifdef _OPENMP
  if (const char* t = Env("GUFO_CCIP_THREADS")) omp_set_num_threads(std::atoi(t));
#endif
  const std::filesystem::path odir(oracle_dir);
  std::ifstream mf(odir / "manifest.json", std::ios::binary);
  if (!mf) {
    std::printf("missing manifest.json\n");
    return 1;
  }
  std::stringstream ss;
  ss << mf.rdbuf();
  const gufo::json::Value manifest = gufo::json::parse(ss.str());
  std::printf("imgutils %s\n", manifest.member_str("imgutils").c_str());
  const double gthr = manifest.member_double("general_threshold");
  const double cthr = manifest.member_double("character_threshold");
  const double margin = manifest.member_double("margin");
  std::string error;
  const auto load_a = std::chrono::steady_clock::now();
  auto net = ccip::Wd14Tagger::Load(model_dir, ccip::kWd14Stem, &error);
  if (!net) {
    std::printf("load: %s\n", error.c_str());
    return 1;
  }
  std::printf("loaded in %.1f s\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - load_a).count());
  bool ok = true;
  Stats frames, crops;
  for (const auto& item : manifest.find("images")->items()) {
    ccip::TensorFile f;
    if (!f.Load(odir / item.member_str("file"), &error)) {
      std::printf("oracle: %s\n", error.c_str());
      return 1;
    }
    const ccip::Tensor* t = f.Find("rgb");
    if (t == nullptr || t->shape.size() != 3 || t->shape[2] != 3) {
      std::printf("rgb tensor missing\n");
      return 1;
    }
    ccip::RgbImage img;
    img.height = static_cast<int>(t->shape[0]);
    img.width = static_cast<int>(t->shape[1]);
    img.rgb = t->u8;
    std::printf("\n%s (%dx%d)\n", item.member_str("source").c_str(), img.width, img.height);
    ok = Check(*net, img, f, "", item, gthr, cthr, "whole picture", &frames) && ok;
    int k = 0;
    for (const auto& crop : item.find("crops")->items()) {
      const auto& p = crop.find("person")->items();
      const std::array<double, 4> box = {p[0].as_double(), p[1].as_double(), p[2].as_double(),
                                         p[3].as_double()};
      const std::array<int, 4> cb = ccip::MarginCropBox(box, margin, img.width, img.height);
      const auto& want_cb = crop.find("crop_pixels")->items();
      bool same_box = true;
      for (int c = 0; c < 4; ++c)
        same_box = same_box && cb[static_cast<std::size_t>(c)] == static_cast<int>(want_cb[static_cast<std::size_t>(c)].as_double());
      if (!same_box) {
        std::printf("  crop %d box %d,%d,%d,%d differs from Pillow's\n", k, cb[0], cb[1], cb[2], cb[3]);
        ok = false;
      }
      const ccip::RgbImage cut = ccip::Crop(img, cb[0], cb[1], cb[2], cb[3]);
      ok = Check(*net, cut, f, "crop" + std::to_string(k) + ".", crop, gthr, cthr,
                 "person " + std::to_string(k) + " " + std::to_string(cut.width) + "x" +
                     std::to_string(cut.height),
                 &crops) &&
           ok;
      ++k;
    }
  }
  std::printf("\nwhole pictures: %d, max prob diff %.2e, max embedding diff %.2e, %.0f ms each\n",
              frames.cases, frames.max_prob, frames.max_emb,
              frames.cases ? frames.ms / frames.cases : 0.0);
  std::printf("person cut-outs: %d, max prob diff %.2e, max embedding diff %.2e, %.0f ms each\n",
              crops.cases, crops.max_prob, crops.max_emb, crops.cases ? crops.ms / crops.cases : 0.0);
  std::printf("tags at the threshold (within 1e-3): %d\n", frames.border + crops.border);
  std::printf("\n%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
