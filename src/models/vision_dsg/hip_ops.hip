// Shared FP32 building blocks of the vision worker models (hip_ops.hpp).
// Written with AI assistance (Claude Code).
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <utility>

#include "src/core/json.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/models/vision_dsg/hip_ops.hpp"

namespace gufo::models::vision_dsg {
namespace {

constexpr int kThreads = 256;

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

bool Check(hipError_t status, const char* what, std::string* error) {
  if (status == hipSuccess)
    return true;
  SetError(error, std::string(what) + ": " + hipGetErrorString(status));
  return false;
}

unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

rocblas_handle Handle(void* blas) {
  return static_cast<rocblas_handle>(blas);
}

__device__ __forceinline__ float BlockSum(float value, float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  const float result = shared[0];
  __syncthreads();
  return result;
}

__device__ __forceinline__ float BlockMax(float value, float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] =
          fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    __syncthreads();
  }
  const float result = shared[0];
  __syncthreads();
  return result;
}

__device__ __forceinline__ float ApplyAct(float v, int act) {
  if (act == 1)
    return 0.5F * v * (1.0F + erff(v * 0.7071067811865476F));
  if (act == 2)
    return fmaxf(v, 0.0F);
  if (act == 3)
    return 1.0F / (1.0F + expf(-v));
  return v;
}

// Two-pass layer norm of one row per block (mean, then centred squares).
__global__ void LayerNormKernel(const float* x, const float* w, const float* b,
                                float* out, int dim, float eps) {
  __shared__ float shared[kThreads];
  const float* row = x + static_cast<std::size_t>(blockIdx.x) * dim;
  float sum = 0;
  for (int c = threadIdx.x; c < dim; c += kThreads)
    sum += row[c];
  const float mean = BlockSum(sum, shared) / dim;
  float squares = 0;
  for (int c = threadIdx.x; c < dim; c += kThreads) {
    const float d = row[c] - mean;
    squares += d * d;
  }
  const float rstd = rsqrtf(BlockSum(squares, shared) / dim + eps);
  float* target = out + static_cast<std::size_t>(blockIdx.x) * dim;
  for (int c = threadIdx.x; c < dim; c += kThreads) {
    float v = (row[c] - mean) * rstd;
    if (w != nullptr)
      v *= w[c];
    if (b != nullptr)
      v += b[c];
    target[c] = v;
  }
}

__global__ void SoftmaxKernel(float* x, int columns, float scale) {
  __shared__ float shared[kThreads];
  float* row = x + static_cast<std::size_t>(blockIdx.x) * columns;
  float maximum = -INFINITY;
  for (int c = threadIdx.x; c < columns; c += kThreads)
    maximum = fmaxf(maximum, row[c] * scale);
  maximum = BlockMax(maximum, shared);
  float sum = 0;
  for (int c = threadIdx.x; c < columns; c += kThreads) {
    const float e =
        maximum == -INFINITY ? 0.0F : expf(row[c] * scale - maximum);
    row[c] = e;
    sum += e;
  }
  sum = BlockSum(sum, shared);
  const float inverse = sum > 0 ? 1.0F / sum : 0.0F;
  for (int c = threadIdx.x; c < columns; c += kThreads)
    row[c] *= inverse;
}

// scores[.., j] = -inf where key_mask[j] == 0.
__global__ void MaskKeysKernel(float* scores, const float* key_mask, int tk,
                               std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count && key_mask[i % tk] == 0.0F)
    scores[i] = -INFINITY;
}

// scores[c, i, j] = scores[c, i, j] * scale + bias[i, j].
__global__ void ScaleBiasKernel(float* scores, const float* bias, float scale,
                                std::size_t per_head, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    scores[i] = scores[i] * scale + bias[i % per_head];
}

// One block per group: mean and variance over channels_per_group * pixels.
__global__ void GroupNormKernel(const float* x, const float* w, const float* b,
                                float* out, int per_group, std::size_t pixels,
                                float eps) {
  __shared__ float shared[kThreads];
  const std::size_t n = static_cast<std::size_t>(per_group) * pixels;
  const float* g = x + static_cast<std::size_t>(blockIdx.x) * n;
  float sum = 0;
  for (std::size_t i = threadIdx.x; i < n; i += kThreads)
    sum += g[i];
  const float mean = BlockSum(sum, shared) / static_cast<float>(n);
  float squares = 0;
  for (std::size_t i = threadIdx.x; i < n; i += kThreads) {
    const float d = g[i] - mean;
    squares += d * d;
  }
  const float rstd =
      rsqrtf(BlockSum(squares, shared) / static_cast<float>(n) + eps);
  float* o = out + static_cast<std::size_t>(blockIdx.x) * n;
  for (std::size_t i = threadIdx.x; i < n; i += kThreads) {
    const int c = blockIdx.x * per_group + static_cast<int>(i / pixels);
    o[i] = (g[i] - mean) * rstd * w[c] + b[c];
  }
}

__global__ void AddKernel(float* x, const float* y, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += y[i];
}

__global__ void AddToKernel(float* out, const float* a, const float* b,
                            std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] + b[i];
}

__global__ void ScaleKernel(float* x, float s, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] *= s;
}

__global__ void ActKernel(float* x, std::size_t n, int act) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] = ApplyAct(x[i], act);
}

__global__ void ScaledAddKernel(float* x, const float* y, const float* gamma,
                                int dim, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += gamma[i % dim] * y[i];
}

__global__ void RowBiasActKernel(float* x, const float* b, int columns, int ld,
                                 std::size_t n, int act) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const std::size_t r = i / columns;
  const int c = static_cast<int>(i % columns);
  float* p = x + r * ld + c;
  float v = *p;
  if (b != nullptr)
    v += b[c];
  *p = ApplyAct(v, act);
}

__global__ void ChannelBiasKernel(float* x, const float* b, std::size_t pixels,
                                  std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += b[i / pixels];
}

__global__ void TokensToMapKernel(const float* tokens, int rows, int dim,
                                  int ld, float* map) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(rows) * dim)
    return;
  const int c = static_cast<int>(i / rows);
  const int p = static_cast<int>(i % rows);
  map[i] = tokens[static_cast<std::size_t>(p) * ld + c];
}

__global__ void MapToTokensKernel(const float* map, int rows, int dim,
                                  float* tokens) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(rows) * dim)
    return;
  const int p = static_cast<int>(i / dim);
  const int c = static_cast<int>(i % dim);
  tokens[i] = map[static_cast<std::size_t>(c) * rows + p];
}

// Window tokens [windows * ws * ws, C] of the zero-padded, cyclically shifted
// (by -shift) map x [H * W, C].
__global__ void WindowGatherKernel(const float* x, int h, int w, int c, int ws,
                                   int shift, int hp, int wp, float* out) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(hp) * wp * c;
  if (i >= total)
    return;
  const int ch = static_cast<int>(i % c);
  const std::size_t token = i / c;
  const int n = ws * ws;
  const int win = static_cast<int>(token / n), local = static_cast<int>(token % n);
  const int nww = wp / ws;
  const int ys = (win / nww) * ws + local / ws, xs = (win % nww) * ws + local % ws;
  const int y = (ys + shift) % hp, xx = (xs + shift) % wp;
  out[i] = (y < h && xx < w) ? x[(static_cast<std::size_t>(y) * w + xx) * c + ch] : 0.0F;
}

// hidden[y, x] += windows[token of (y, x)] (reverse shift, crop).
__global__ void WindowScatterAddKernel(const float* windows, int h, int w, int c,
                                       int ws, int shift, int hp, int wp,
                                       float* hidden) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(h) * w * c;
  if (i >= total)
    return;
  const int ch = static_cast<int>(i % c);
  const std::size_t p = i / c;
  const int y = static_cast<int>(p / w), xx = static_cast<int>(p % w);
  const int ys = (y - shift + hp) % hp, xs = (xx - shift + wp) % wp;
  const int nww = wp / ws;
  const int win = (ys / ws) * nww + xs / ws;
  const int local = (ys % ws) * ws + xs % ws;
  hidden[i] += windows[(static_cast<std::size_t>(win) * ws * ws + local) * c + ch];
}

// cols[(ci * k + ky) * k + kx, p] for output pixels p0 .. p0 + count.
__global__ void Im2ColKernel(const float* in, int cin, int h, int w, int k,
                             int stride, int pad, int ow, int p0, int count,
                             float* cols) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(cin) * k * k * count;
  if (i >= total)
    return;
  const int p = static_cast<int>(i % count);
  const int row = static_cast<int>(i / count);
  const int kx = row % k, ky = (row / k) % k, ci = row / (k * k);
  const int op = p0 + p;
  const int oy = op / ow, ox = op % ow;
  const int y = oy * stride - pad + ky, x = ox * stride - pad + kx;
  cols[i] = (y >= 0 && y < h && x >= 0 && x < w)
                ? in[(static_cast<std::size_t>(ci) * h + y) * w + x]
                : 0.0F;
}

// Transposed convolution, kernel == stride: out[co, y, x] = b[co] +
// sum_ci in[ci, y / k, x / k] * w[ci, co, y % k, x % k].
__global__ void ConvTransposeKSKernel(const float* in, int cin, int h, int w,
                                      const float* weight, const float* bias,
                                      int cout, int k, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int oh = h * k, ow = w * k;
  const std::size_t plane = static_cast<std::size_t>(oh) * ow;
  if (i >= plane * cout)
    return;
  const int co = static_cast<int>(i / plane);
  const int p = static_cast<int>(i % plane);
  const int y = p / ow, x = p % ow;
  const int sy = y / k, sx = x / k, ky = y % k, kx = x % k;
  float sum = bias != nullptr ? bias[co] : 0.0F;
  const std::size_t in_plane = static_cast<std::size_t>(h) * w;
  for (int ci = 0; ci < cin; ++ci)
    sum += in[ci * in_plane + sy * w + sx] *
           weight[((static_cast<std::size_t>(ci) * cout + co) * k + ky) * k +
                  kx];
  out[i] = sum;
}

__global__ void BilinearKernel(const float* in, int channels, int h, int w,
                               float* out, int oh, int ow, int align) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t plane = static_cast<std::size_t>(oh) * ow;
  if (i >= plane * channels)
    return;
  const int c = static_cast<int>(i / plane);
  const int p = static_cast<int>(i % plane);
  const int y = p / ow, x = p % ow;
  float ry, rx;
  if (align) {
    const float sy = oh > 1 ? static_cast<float>(h - 1) / (oh - 1) : 0.0F;
    const float sx = ow > 1 ? static_cast<float>(w - 1) / (ow - 1) : 0.0F;
    ry = sy * y;
    rx = sx * x;
  } else {
    const float sy = static_cast<float>(h) / oh;
    const float sx = static_cast<float>(w) / ow;
    ry = fmaxf(sy * (y + 0.5F) - 0.5F, 0.0F);
    rx = fmaxf(sx * (x + 0.5F) - 0.5F, 0.0F);
  }
  const int y0 = min(static_cast<int>(ry), h - 1);
  const int x0 = min(static_cast<int>(rx), w - 1);
  const int y1 = y0 < h - 1 ? y0 + 1 : y0, x1 = x0 < w - 1 ? x0 + 1 : x0;
  const float ly = ry - y0, lx = rx - x0;
  const float* s = in + static_cast<std::size_t>(c) * h * w;
  out[i] = (1 - ly) * ((1 - lx) * s[y0 * w + x0] + lx * s[y0 * w + x1]) +
           ly * ((1 - lx) * s[y1 * w + x0] + lx * s[y1 * w + x1]);
}

__device__ __host__ inline float CubicNear(float x, float a) {
  return ((a + 2) * x - (a + 3)) * x * x + 1;
}
__device__ __host__ inline float CubicFar(float x, float a) {
  return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
}

__global__ void BicubicKernel(const float* in, int channels, int h, int w,
                              float* out, int oh, int ow) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t plane = static_cast<std::size_t>(oh) * ow;
  if (i >= plane * channels)
    return;
  const int c = static_cast<int>(i / plane);
  const int p = static_cast<int>(i % plane);
  const int y = p / ow, x = p % ow;
  const float sy = static_cast<float>(h) / oh;
  const float sx = static_cast<float>(w) / ow;
  const float ry = sy * (y + 0.5F) - 0.5F, rx = sx * (x + 0.5F) - 0.5F;
  const int iy = static_cast<int>(floorf(ry));
  const int ix = static_cast<int>(floorf(rx));
  const float ty = ry - iy, tx = rx - ix;
  constexpr float a = -0.75F;
  const float cy[4] = {CubicFar(ty + 1, a), CubicNear(ty, a),
                       CubicNear(1 - ty, a), CubicFar(2 - ty, a)};
  const float cx[4] = {CubicFar(tx + 1, a), CubicNear(tx, a),
                       CubicNear(1 - tx, a), CubicFar(2 - tx, a)};
  const float* s = in + static_cast<std::size_t>(c) * h * w;
  float sum = 0;
  for (int j = 0; j < 4; ++j) {
    const int yy = min(max(iy - 1 + j, 0), h - 1);
    float row = 0;
    for (int q = 0; q < 4; ++q) {
      const int xx = min(max(ix - 1 + q, 0), w - 1);
      row += cx[q] * s[yy * w + xx];
    }
    sum += cy[j] * row;
  }
  out[i] = sum;
}

bool ReadAll(platform::PlatformFile file, void* output, std::size_t bytes,
             std::uint64_t offset) {
  auto* target = static_cast<unsigned char*>(output);
  while (bytes > 0) {
    const std::size_t piece = std::min<std::size_t>(bytes, 64U << 20U);
    const std::size_t done = platform::ReadFileAt(file, target, piece, offset);
    if (done == 0)
      return false;
    target += done;
    offset += done;
    bytes -= done;
  }
  return true;
}

float HalfToFloat(std::uint16_t h) {
  const int sign = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
  float value;
  if (e == 0)
    value = std::ldexp(static_cast<float>(m), -24);
  else if (e == 31)
    value = m != 0 ? NAN : INFINITY;
  else
    value = std::ldexp(static_cast<float>(1024 + m), e - 25);
  return sign != 0 ? -value : value;
}

}  // namespace

// ---- DevBuf ---------------------------------------------------------------

DevBuf::DevBuf(DevBuf&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      capacity_(std::exchange(other.capacity_, 0)) {}

DevBuf& DevBuf::operator=(DevBuf&& other) noexcept {
  if (this != &other) {
    if (data_ != nullptr)
      (void)hipFree(data_);
    data_ = std::exchange(other.data_, nullptr);
    capacity_ = std::exchange(other.capacity_, 0);
  }
  return *this;
}

DevBuf::~DevBuf() {
  if (data_ != nullptr)
    (void)hipFree(data_);
}

bool DevBuf::Reserve(std::size_t count, std::string* error) {
  if (count <= capacity_ && data_ != nullptr)
    return true;
  if (data_ != nullptr)
    (void)hipFree(data_);
  data_ = nullptr;
  capacity_ = 0;
  void* p = nullptr;
  if (!Check(hipMalloc(&p, std::max<std::size_t>(count, 64) * 4),
             "device allocation", error))
    return false;
  data_ = static_cast<float*>(p);
  capacity_ = count;
  return true;
}

bool DevBuf::Upload(const float* host, std::size_t count, std::string* error) {
  return Reserve(count, error) &&
         Check(hipMemcpy(data_, host, count * 4, hipMemcpyHostToDevice),
               "upload", error);
}

bool DevBuf::Download(float* host, std::size_t count,
                      std::string* error) const {
  return Check(hipMemcpy(host, data_, count * 4, hipMemcpyDeviceToHost),
               "download", error);
}

// ---- WeightStore ----------------------------------------------------------

bool WeightStore::Load(const std::filesystem::path& path, std::string* error) {
  std::string open_error;
  auto file = platform::OpenReadFile(path, &open_error);
  if (!platform::FileValid(file)) {
    SetError(error, "cannot open " + path.string() + ": " + open_error);
    return false;
  }
  struct Closer {
    platform::PlatformFile file;
    ~Closer() { platform::CloseFile(file); }
  } closer{file};
  const auto file_size = platform::FileSizeBytes(file);
  std::uint64_t header_size = 0;
  if (!file_size || platform::ReadFileAt(file, &header_size, 8, 0) != 8 ||
      header_size == 0 || header_size > (64U << 20U) ||
      header_size > *file_size - 8) {
    SetError(error, "invalid safetensors header in " + path.string());
    return false;
  }
  std::string text(static_cast<std::size_t>(header_size), '\0');
  if (!ReadAll(file, text.data(), text.size(), 8)) {
    SetError(error, "short read of " + path.string());
    return false;
  }
  try {
    const json::Value header = json::parse(text);
    std::vector<unsigned char> raw;
    std::vector<float> host;
    for (const auto& [name, entry] : header.members()) {
      if (name == "__metadata__")
        continue;
      const std::string dtype = entry.member_str("dtype");
      const json::Value* shape = entry.find("shape");
      const json::Value* offsets = entry.find("data_offsets");
      if (shape == nullptr || offsets == nullptr || offsets->size() != 2) {
        SetError(error, "invalid tensor entry " + name + " in " +
                            path.string());
        return false;
      }
      if (dtype == "I64" || dtype == "I32" || dtype == "BOOL")
        continue;  // buffers such as position ids
      const std::size_t element =
          dtype == "F32" ? 4 : (dtype == "F16" || dtype == "BF16") ? 2 : 0;
      if (element == 0) {
        SetError(error, "unsupported dtype " + dtype + " of " + name);
        return false;
      }
      std::vector<std::size_t> dims;
      std::size_t count = 1;
      for (const auto& dim : shape->items()) {
        dims.push_back(dim.as_size());
        count *= dims.back();
      }
      const std::uint64_t begin = offsets->items()[0].as_size();
      const std::uint64_t end = offsets->items()[1].as_size();
      if (end - begin != count * element ||
          8 + header_size + end > *file_size) {
        SetError(error, "invalid tensor extent " + name);
        return false;
      }
      host.resize(count);
      if (element == 4) {
        if (!ReadAll(file, host.data(), count * 4, 8 + header_size + begin)) {
          SetError(error, "short read of " + name);
          return false;
        }
      } else {
        raw.resize(count * 2);
        if (!ReadAll(file, raw.data(), count * 2, 8 + header_size + begin)) {
          SetError(error, "short read of " + name);
          return false;
        }
        for (std::size_t i = 0; i < count; ++i) {
          std::uint16_t bits;
          std::memcpy(&bits, raw.data() + 2 * i, 2);
          if (dtype == "F16") {
            host[i] = HalfToFloat(bits);
          } else {
            const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
            std::memcpy(&host[i], &wide, 4);
          }
        }
      }
      DevBuf buffer;
      if (!buffer.Upload(host.data(), count, error))
        return false;
      bytes_ += count * 4;
      shapes_[name] = std::move(dims);
      tensors_[name] = std::move(buffer);
    }
  } catch (const std::exception& e) {
    SetError(error, path.string() + ": " + e.what());
    return false;
  }
  return true;
}

const float* WeightStore::Get(const std::string& name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : it->second.f();
}

bool WeightStore::Has(const std::string& name) const {
  return tensors_.count(name) != 0;
}

std::vector<std::size_t> WeightStore::Shape(const std::string& name) const {
  const auto it = shapes_.find(name);
  return it == shapes_.end() ? std::vector<std::size_t>{} : it->second;
}

std::vector<float> WeightStore::Host(const std::string& name) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end())
    return {};
  std::size_t count = 1;
  for (std::size_t d : shapes_.at(name))
    count *= d;
  std::vector<float> out(count);
  if (!it->second.Download(out.data(), count, nullptr))
    return {};
  return out;
}

// ---- Gpu ------------------------------------------------------------------

Gpu::Gpu() = default;

Gpu::~Gpu() {
  if (blas_ != nullptr)
    (void)rocblas_destroy_handle(Handle(blas_));
}

bool Gpu::Init(std::string* error) {
  rocblas_handle handle = nullptr;
  if (rocblas_create_handle(&handle) != rocblas_status_success) {
    SetError(error, "rocBLAS handle could not be created");
    return false;
  }
  blas_ = handle;
  return true;
}

bool Gpu::Gemm(bool trans_a, bool trans_b, int m, int n, int k, float alpha,
               const float* a, int lda, const float* b, int ldb, float beta,
               float* c, int ldc, std::string* error) {
  // Row-major C = op(A) op(B) is column-major C^T = op(B)^T op(A)^T.
  const rocblas_status status = rocblas_sgemm(
      Handle(blas_),
      trans_b ? rocblas_operation_transpose : rocblas_operation_none,
      trans_a ? rocblas_operation_transpose : rocblas_operation_none, n, m, k,
      &alpha, b, ldb, a, lda, &beta, c, ldc);
  if (status != rocblas_status_success) {
    SetError(error, "rocBLAS SGEMM failed");
    return false;
  }
  return true;
}

bool Gpu::Linear(const float* in, int tokens, const float* w, const float* b,
                 int rows, int columns, float* out, Act act,
                 std::string* error, int ldin, int ldout) {
  if (w == nullptr) {
    SetError(error, "missing linear weight");
    return false;
  }
  if (ldin == 0)
    ldin = columns;
  if (ldout == 0)
    ldout = rows;
  if (!Gemm(false, true, tokens, rows, columns, 1.0F, in, ldin, w, columns,
            0.0F, out, ldout, error))
    return false;
  if (b != nullptr || act != Act::kNone) {
    const std::size_t n = static_cast<std::size_t>(tokens) * rows;
    RowBiasActKernel<<<Blocks(n), kThreads>>>(out, b, rows, ldout, n,
                                              static_cast<int>(act));
  }
  return Check(hipGetLastError(), "linear", error);
}

bool Gpu::Conv2d(const float* in, int cin, int h, int w, const float* weight,
                 const float* bias, int cout, int k, int stride, int pad,
                 float* out, int* oh_out, int* ow_out, std::string* error) {
  if (weight == nullptr) {
    SetError(error, "missing convolution weight");
    return false;
  }
  const int oh = (h + 2 * pad - k) / stride + 1;
  const int ow = (w + 2 * pad - k) / stride + 1;
  if (oh_out != nullptr)
    *oh_out = oh;
  if (ow_out != nullptr)
    *ow_out = ow;
  const int pixels = oh * ow;
  const std::size_t K = static_cast<std::size_t>(cin) * k * k;
  if (k == 1 && stride == 1 && pad == 0) {
    // out[cout, P] = W[cout, cin] . in[cin, P]
    if (!Gemm(false, false, cout, pixels, cin, 1.0F, weight, cin, in, pixels,
              0.0F, out, pixels, error))
      return false;
  } else {
    // im2col in chunks of at most 64 Mi floats.
    const std::size_t budget = 64U << 20U;
    const int chunk = static_cast<int>(std::max<std::size_t>(
        1, std::min<std::size_t>(pixels, budget / std::max<std::size_t>(K, 1))));
    if (!cols_.Reserve(K * chunk, error))
      return false;
    for (int p0 = 0; p0 < pixels; p0 += chunk) {
      const int count = std::min(chunk, pixels - p0);
      Im2ColKernel<<<Blocks(K * count), kThreads>>>(in, cin, h, w, k, stride,
                                                    pad, ow, p0, count,
                                                    cols_.f());
      if (!Gemm(false, false, cout, count, static_cast<int>(K), 1.0F, weight,
                static_cast<int>(K), cols_.f(), count, 0.0F, out + p0, pixels,
                error))
        return false;
    }
  }
  if (bias != nullptr) {
    const std::size_t n = static_cast<std::size_t>(cout) * pixels;
    ChannelBiasKernel<<<Blocks(n), kThreads>>>(out, bias, pixels, n);
  }
  return Check(hipGetLastError(), "convolution", error);
}

bool Gpu::ConvTransposeKS(const float* in, int cin, int h, int w,
                          const float* weight, const float* bias, int cout,
                          int k, float* out, std::string* error) {
  const std::size_t n = static_cast<std::size_t>(cout) * h * k * w * k;
  ConvTransposeKSKernel<<<Blocks(n), kThreads>>>(in, cin, h, w, weight, bias,
                                                 cout, k, out);
  return Check(hipGetLastError(), "transposed convolution", error);
}

bool Gpu::Attention(const float* q, int ldq, const float* k, int ldk,
                    const float* v, int ldv, int tq, int tk, int heads,
                    int head_dim, float scale, float* out, int ldo,
                    std::string* error, const float* key_mask,
                    const float* bias) {
  // Heads in chunks so the score buffer [chunk, tq, tk] stays below 64 Mi
  // floats.
  const std::size_t per_head = static_cast<std::size_t>(tq) * tk;
  const int chunk = static_cast<int>(std::max<std::size_t>(
      1, std::min<std::size_t>(
             heads, (64U << 20U) / std::max<std::size_t>(per_head, 1))));
  if (!scores_.Reserve(per_head * chunk, error))
    return false;
  const float one = 1.0F, zero = 0.0F;
  for (int h0 = 0; h0 < heads; h0 += chunk) {
    const int count = std::min(chunk, heads - h0);
    // Column-major S^T[tk, tq] = K_h Q_h^T per head.
    if (rocblas_sgemm_strided_batched(
            Handle(blas_), rocblas_operation_transpose, rocblas_operation_none,
            tk, tq, head_dim, &one, k + static_cast<std::size_t>(h0) * head_dim,
            ldk, head_dim, q + static_cast<std::size_t>(h0) * head_dim, ldq,
            head_dim, &zero, scores_.f(), tk,
            static_cast<rocblas_stride>(per_head),
            count) != rocblas_status_success) {
      SetError(error, "attention scores failed");
      return false;
    }
    if (key_mask != nullptr) {
      const std::size_t n = per_head * count;
      MaskKeysKernel<<<Blocks(n), kThreads>>>(scores_.f(), key_mask, tk, n);
    }
    float softmax_scale = scale;
    if (bias != nullptr) {
      const std::size_t n = per_head * count;
      ScaleBiasKernel<<<Blocks(n), kThreads>>>(scores_.f(), bias, scale,
                                               per_head, n);
      softmax_scale = 1.0F;
    }
    SoftmaxKernel<<<static_cast<unsigned>(static_cast<std::size_t>(count) *
                                          tq),
                    kThreads>>>(scores_.f(), tk, softmax_scale);
    // out_h[i] = sum_j p[i, j] v_h[j]
    if (rocblas_sgemm_strided_batched(
            Handle(blas_), rocblas_operation_none, rocblas_operation_none,
            head_dim, tq, tk, &one, v + static_cast<std::size_t>(h0) * head_dim,
            ldv, head_dim, scores_.f(), tk,
            static_cast<rocblas_stride>(per_head), &zero,
            out + static_cast<std::size_t>(h0) * head_dim, ldo, head_dim,
            count) != rocblas_status_success) {
      SetError(error, "attention values failed");
      return false;
    }
  }
  return Check(hipGetLastError(), "attention", error);
}

// ---- free kernels ---------------------------------------------------------

void LayerNorm(const float* x, const float* w, const float* b, float* out,
               int rows, int dim, float eps) {
  LayerNormKernel<<<rows, kThreads>>>(x, w, b, out, dim, eps);
}

void Softmax(float* x, int rows, int columns, float scale) {
  SoftmaxKernel<<<rows, kThreads>>>(x, columns, scale);
}

void Add(float* x, const float* y, std::size_t n) {
  AddKernel<<<Blocks(n), kThreads>>>(x, y, n);
}

void AddTo(float* out, const float* a, const float* b, std::size_t n) {
  AddToKernel<<<Blocks(n), kThreads>>>(out, a, b, n);
}

void Scale(float* x, float s, std::size_t n) {
  ScaleKernel<<<Blocks(n), kThreads>>>(x, s, n);
}

void Activate(float* x, std::size_t n, Act act) {
  ActKernel<<<Blocks(n), kThreads>>>(x, n, static_cast<int>(act));
}

void ScaledAdd(float* x, const float* y, const float* gamma, std::size_t rows,
               int dim) {
  const std::size_t n = rows * dim;
  ScaledAddKernel<<<Blocks(n), kThreads>>>(x, y, gamma, dim, n);
}

void AddRowBias(float* x, const float* b, std::size_t rows, int columns) {
  const std::size_t n = rows * columns;
  RowBiasActKernel<<<Blocks(n), kThreads>>>(x, b, columns, columns, n, 0);
}

void AddChannelBias(float* x, const float* b, int channels,
                    std::size_t pixels) {
  const std::size_t n = static_cast<std::size_t>(channels) * pixels;
  ChannelBiasKernel<<<Blocks(n), kThreads>>>(x, b, pixels, n);
}

void TokensToMap(const float* tokens, int rows, int dim, int ld, float* map) {
  TokensToMapKernel<<<Blocks(static_cast<std::size_t>(rows) * dim),
                      kThreads>>>(tokens, rows, dim, ld, map);
}

void MapToTokens(const float* map, int rows, int dim, float* tokens) {
  MapToTokensKernel<<<Blocks(static_cast<std::size_t>(rows) * dim),
                      kThreads>>>(map, rows, dim, tokens);
}

void ResizeBilinear(const float* in, int channels, int h, int w, float* out,
                    int oh, int ow, bool align_corners) {
  const std::size_t n = static_cast<std::size_t>(channels) * oh * ow;
  BilinearKernel<<<Blocks(n), kThreads>>>(in, channels, h, w, out, oh, ow,
                                          align_corners ? 1 : 0);
}

void ResizeBicubic(const float* in, int channels, int h, int w, float* out,
                   int oh, int ow) {
  const std::size_t n = static_cast<std::size_t>(channels) * oh * ow;
  BicubicKernel<<<Blocks(n), kThreads>>>(in, channels, h, w, out, oh, ow);
}

void WindowGather(const float* x, int h, int w, int c, int ws, int shift,
                  float* out) {
  const int hp = (h + ws - 1) / ws * ws, wp = (w + ws - 1) / ws * ws;
  const std::size_t n = static_cast<std::size_t>(hp) * wp * c;
  WindowGatherKernel<<<Blocks(n), kThreads>>>(x, h, w, c, ws, shift, hp, wp, out);
}

void WindowScatterAdd(const float* windows, int h, int w, int c, int ws,
                      int shift, float* hidden) {
  const int hp = (h + ws - 1) / ws * ws, wp = (w + ws - 1) / ws * ws;
  const std::size_t n = static_cast<std::size_t>(h) * w * c;
  WindowScatterAddKernel<<<Blocks(n), kThreads>>>(windows, h, w, c, ws, shift,
                                                  hp, wp, hidden);
}

void GroupNorm(const float* x, const float* w, const float* b, float* out,
               int channels, std::size_t pixels, int groups, float eps) {
  GroupNormKernel<<<groups, kThreads>>>(x, w, b, out, channels / groups,
                                        pixels, eps);
}

bool Sync(const char* what, std::string* error) {
  return Check(hipGetLastError(), what, error) &&
         Check(hipDeviceSynchronize(), what, error);
}

std::vector<float> HostResizeBicubic(const std::vector<float>& in,
                                     int channels, int h, int w, int oh,
                                     int ow) {
  std::vector<float> out(static_cast<std::size_t>(channels) * oh * ow);
  const float sy = static_cast<float>(h) / oh;
  const float sx = static_cast<float>(w) / ow;
  constexpr float a = -0.75F;
  for (int y = 0; y < oh; ++y) {
    const float ry = sy * (y + 0.5F) - 0.5F;
    const int iy = static_cast<int>(std::floor(ry));
    const float ty = ry - iy;
    const float cy[4] = {CubicFar(ty + 1, a), CubicNear(ty, a),
                         CubicNear(1 - ty, a), CubicFar(2 - ty, a)};
    for (int x = 0; x < ow; ++x) {
      const float rx = sx * (x + 0.5F) - 0.5F;
      const int ix = static_cast<int>(std::floor(rx));
      const float tx = rx - ix;
      const float cx[4] = {CubicFar(tx + 1, a), CubicNear(tx, a),
                           CubicNear(1 - tx, a), CubicFar(2 - tx, a)};
      for (int c = 0; c < channels; ++c) {
        const float* s = in.data() + static_cast<std::size_t>(c) * h * w;
        float sum = 0;
        for (int j = 0; j < 4; ++j) {
          const int yy = std::clamp(iy - 1 + j, 0, h - 1);
          float row = 0;
          for (int q = 0; q < 4; ++q)
            row += cx[q] * s[yy * w + std::clamp(ix - 1 + q, 0, w - 1)];
          sum += cy[j] * row;
        }
        out[(static_cast<std::size_t>(c) * oh + y) * ow + x] = sum;
      }
    }
  }
  return out;
}

}  // namespace gufo::models::vision_dsg
