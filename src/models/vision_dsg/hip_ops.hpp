// Shared FP32 building blocks of the vision worker models (Depth Anything V2,
// Grounding DINO, SAM): a safetensors weight store on the GPU, rocBLAS
// linear layers and convolutions (im2col + SGEMM), layer norm, softmax,
// multi-head attention and resampling. Row-major token matrices [T, C] and
// single-image NCHW maps [C, H, W]. Everything is FP32 so the results follow
// the transformers reference (oracle tolerances in docs/models/vision-dsg.md).
// Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_VISION_DSG_HIP_OPS_HPP_
#define GUFO_MODELS_VISION_DSG_HIP_OPS_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gufo::models::vision_dsg {

/// Owning device buffer of floats.
class DevBuf {
public:
  DevBuf() = default;
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& other) noexcept;
  DevBuf& operator=(DevBuf&& other) noexcept;
  ~DevBuf();
  /// Allocates `count` floats (contents undefined); keeps a big enough
  /// allocation.
  bool Reserve(std::size_t count, std::string* error);
  bool Upload(const float* host, std::size_t count, std::string* error);
  bool Download(float* host, std::size_t count, std::string* error) const;
  [[nodiscard]] float* f() const { return data_; }
  [[nodiscard]] std::size_t capacity() const { return capacity_; }

private:
  float* data_{nullptr};
  std::size_t capacity_{0};
};

/// All tensors of one or more safetensors files as FP32 on the GPU (F32,
/// F16 and BF16 sources; I64 is skipped).
class WeightStore {
public:
  bool Load(const std::filesystem::path& file, std::string* error);
  [[nodiscard]] const float* Get(const std::string& name) const;
  [[nodiscard]] bool Has(const std::string& name) const;
  [[nodiscard]] std::vector<std::size_t> Shape(const std::string& name) const;
  /// Host copy of one tensor (empty when missing).
  [[nodiscard]] std::vector<float> Host(const std::string& name) const;
  [[nodiscard]] std::size_t bytes() const { return bytes_; }

private:
  std::map<std::string, DevBuf, std::less<>> tensors_;
  std::map<std::string, std::vector<std::size_t>, std::less<>> shapes_;
  std::size_t bytes_{0};
};

enum class Act { kNone = 0, kGelu = 1, kRelu = 2, kSigmoid = 3 };

/// rocBLAS handle and the ops that need it.
class Gpu {
public:
  Gpu();
  ~Gpu();
  Gpu(const Gpu&) = delete;
  Gpu& operator=(const Gpu&) = delete;
  bool Init(std::string* error);

  /// out[T, R] = in[T, C] . w[R, C]^T + b (b may be null), then act.
  /// `ldin`/`ldout` default to C/R.
  bool Linear(const float* in, int tokens, const float* w, const float* b,
              int rows, int columns, float* out, Act act, std::string* error,
              int ldin = 0, int ldout = 0);
  /// Plain row-major GEMM: c[M, N] = alpha * op(a) . op(b) + beta * c.
  bool Gemm(bool trans_a, bool trans_b, int m, int n, int k, float alpha,
            const float* a, int lda, const float* b, int ldb, float beta,
            float* c, int ldc, std::string* error);
  /// NCHW convolution, zero padding, groups 1; *oh/*ow get the output size.
  bool Conv2d(const float* in, int cin, int h, int w, const float* weight,
              const float* bias, int cout, int k, int stride, int pad,
              float* out, int* oh, int* ow, std::string* error);
  /// Transposed convolution with kernel == stride, no padding.
  bool ConvTransposeKS(const float* in, int cin, int h, int w,
                       const float* weight, const float* bias, int cout, int k,
                       float* out, std::string* error);
  /// Multi-head attention: q/k/v rows of `heads * head_dim` values at the
  /// given leading dimensions; out [tq, heads * head_dim] (ld `ldo`).
  /// `key_mask` [tk] (0 = masked) and `bias` [tq, tk] (added to the scaled
  /// scores of every head) are optional.
  bool Attention(const float* q, int ldq, const float* k, int ldk,
                 const float* v, int ldv, int tq, int tk, int heads,
                 int head_dim, float scale, float* out, int ldo,
                 std::string* error, const float* key_mask = nullptr,
                 const float* bias = nullptr);

  void* blas() const { return blas_; }

private:
  void* blas_{nullptr};
  DevBuf cols_, scores_;
};

// Element-wise and normalisation kernels (default stream).
void LayerNorm(const float* x, const float* w, const float* b, float* out,
               int rows, int dim, float eps);
void Softmax(float* x, int rows, int columns, float scale);
void Add(float* x, const float* y, std::size_t n);
void AddTo(float* out, const float* a, const float* b, std::size_t n);
void Scale(float* x, float s, std::size_t n);
void Activate(float* x, std::size_t n, Act act);
/// x[r, c] += gamma[c] * y[r, c].
void ScaledAdd(float* x, const float* y, const float* gamma, std::size_t rows,
               int dim);
/// x[r, c] += b[c].
void AddRowBias(float* x, const float* b, std::size_t rows, int columns);
/// NCHW x[c, p] += b[c].
void AddChannelBias(float* x, const float* b, int channels, std::size_t pixels);
/// Tokens [rows, dim] (row stride `ld`) -> NCHW [dim, rows].
void TokensToMap(const float* tokens, int rows, int dim, int ld, float* map);
/// NCHW [dim, rows] -> tokens [rows, dim].
void MapToTokens(const float* map, int rows, int dim, float* tokens);
/// Bilinear resize of NCHW maps (torch interpolate, no antialias).
void ResizeBilinear(const float* in, int channels, int h, int w, float* out,
                    int oh, int ow, bool align_corners);
/// Bicubic resize (A = -0.75, align_corners false, scale = in / out) of NCHW
/// maps, like torch interpolate(mode="bicubic", size=...).
void ResizeBicubic(const float* in, int channels, int h, int w, float* out,
                   int oh, int ow);
/// Window tokens [windows * ws * ws, C] (row-major windows over the map
/// zero-padded to multiples of ws, cyclically shifted by -shift) of tokens
/// x [h * w, C]; WindowScatterAdd adds them back (reverse shift, crop).
void WindowGather(const float* x, int h, int w, int c, int ws, int shift,
                  float* out);
void WindowScatterAdd(const float* windows, int h, int w, int c, int ws,
                      int shift, float* hidden);
/// GroupNorm of an NCHW map [channels, pixels] with `groups` groups.
void GroupNorm(const float* x, const float* w, const float* b, float* out,
               int channels, std::size_t pixels, int groups, float eps);
bool Sync(const char* what, std::string* error);

/// Host bicubic (same rule as ResizeBicubic) for small grids.
std::vector<float> HostResizeBicubic(const std::vector<float>& in,
                                     int channels, int h, int w, int oh,
                                     int ow);

}  // namespace gufo::models::vision_dsg

#endif  // GUFO_MODELS_VISION_DSG_HIP_OPS_HPP_
