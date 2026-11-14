// Small CPU fp32 building blocks for the pose networks (RT-DETR, ViTPose,
// DWPose): CHW feature maps, row-major token matrices, weights from a
// TensorFile. Batch size 1 everywhere. Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_CCIP_CPU_NN_HPP_
#define GUFO_MODELS_CCIP_CPU_NN_HPP_

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "ccip/tensors.hpp"

namespace gufo::models::ccip::nn {

struct Map {
  int c{0}, h{0}, w{0};
  std::vector<float> d;  // CHW
};

enum class Act { kNone, kRelu, kSilu, kGelu };

void Activate(float* x, std::size_t n, Act act);

// Weights of one network (F32 safetensors). Linear weights stay [out, in]
// and run through a dot-product GEMM (no transposed copy: RAM is scarce).
class Weights {
public:
  bool Load(const std::filesystem::path& file, std::string* error);
  [[nodiscard]] bool Has(std::string_view name) const { return file_.Find(name) != nullptr; }
  [[nodiscard]] const std::vector<float>& Raw(std::string_view name) const { return file_.F32(name); }
  [[nodiscard]] const Tensor& Get(std::string_view name) const;

private:
  TensorFile file_;
};

// y = act(conv(x) + b); weight [out, in/groups, k, k]; square kernel.
Map Conv(const Map& x, const float* weight, const float* bias, int out, int k, int stride, int pad,
         int groups, Act act);
// "name.weight" + "name.bias" (bias optional).
Map Conv(const Weights& w, const std::string& name, const Map& x, int stride, int pad, Act act,
         int groups = 1);
Map MaxPool(const Map& x, int k, int stride, int pad);
// AvgPool2d(k, stride, pad 0, ceil_mode=True); the divisor counts the pixels inside the input.
Map AvgPoolCeil(const Map& x, int k, int stride);
Map UpsampleNearest2(const Map& x);
// F.interpolate(scale_factor=s, mode="bilinear", align_corners=False).
Map UpsampleBilinear(const Map& x, int s);
Map Concat(const Map& a, const Map& b);
void AddInPlace(Map* a, const Map& b);

// Tokens [rows, dim] row-major. Linear "name": weight [out, in], bias [out].
std::vector<float> Linear(const Weights& w, const std::string& name, const float* x, int rows,
                          Act act = Act::kNone);
void LayerNorm(float* x, int rows, int dim, const float* g, const float* b, float eps);
void LayerNorm(const Weights& w, const std::string& name, float* x, int rows, int dim, float eps);
void SoftmaxRows(float* x, int rows, int cols);
// Multi-head attention: q [lq, dim], k/v [lk, dim] (already projected),
// head_dim = dim / heads, scaling head_dim^-0.5; returns [lq, dim].
std::vector<float> Attention(const float* q, const float* k, const float* v, int lq, int lk, int dim,
                             int heads);
// Map <-> tokens (CHW <-> [h*w, c]).
std::vector<float> ToTokens(const Map& x);
Map FromTokens(const std::vector<float>& t, int c, int h, int w);

}  // namespace gufo::models::ccip::nn

#endif  // GUFO_MODELS_CCIP_CPU_NN_HPP_
