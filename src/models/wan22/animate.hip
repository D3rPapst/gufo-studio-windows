// Written with AI assistance (Claude Code).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <gufo/ext/json.hpp>
#include <gufo/ext/platform/mapped_file.hpp>
#include "src/models/wan22/animate.hpp"
#include "src/models/wan22/animate_host.hpp"
#include "src/models/wan22/gemm.hpp"

// Wan 2.2 Animate: pose patch embedding, LIA motion encoder, face encoder
// and the face adapter blocks (comfy/ldm/wan/model_animate.py), run on the
// transformer's stream through Transformer21Hook with its workspace.

namespace gufo::models::wan22 {
namespace {

constexpr int kWidth = Transformer21::kModelWidth;  // 5120
constexpr int kHeads = 40;
constexpr int kHeadDim = 128;
constexpr int kDitLayers = 40;
constexpr int kAdapterStride = 5;
constexpr int kPosePatch = Animate::kPoseChannels * 4;  // 64
constexpr int kMotionWidth = 512;
constexpr int kMotionDim = 20;
constexpr int kFaceHeads = 4;
constexpr int kFaceChannels = 1024;
constexpr int kTokens = Animate::kMotionTokens;  // 5
constexpr int kFace = Animate::kFaceSize;        // 512
constexpr float kEpsilon = 1e-6F;
constexpr int kRowThreads = 256;
constexpr int kRowValues = kWidth / kRowThreads;  // 20

static_assert(kWidth == kHeads * kHeadDim);
static_assert(kTokens == kFaceHeads + 1);

__device__ __forceinline__ float FromBf16(std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t ToBf16(float value) {
  const unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7FFFFFFFU) > 0x7F800000U)
    return static_cast<std::uint16_t>((bits >> 16U) | 0x40U);
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >>
                                    16U);
}

// OCP FP8 e4m3fn (bias 7, no infinities).
__device__ __forceinline__ float Fp8(unsigned int byte) {
  const unsigned int e = (byte >> 3U) & 15U, m = byte & 7U;
  const float v = e == 0 ? static_cast<float>(m) * 0.001953125F
                         : ldexpf(1.0F + static_cast<float>(m) * 0.125F,
                                  static_cast<int>(e) - 7);
  return (byte & 0x80U) != 0 ? -v : v;
}

template <int kThreads>
__device__ __forceinline__ float BlockSum(float value, float* shared) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
#pragma unroll
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  const float result = shared[0];
  __syncthreads();
  return result;
}

// out[i] = bf16(fp8(w[i]) * scale), four values a thread.
__global__ void UnpackKernel(const std::uint32_t* weight, std::size_t quads,
                             float scale, std::uint16_t* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= quads)
    return;
  const std::uint32_t w = weight[i];
  ushort4 v;
  v.x = ToBf16(Fp8(w & 0xFFU) * scale);
  v.y = ToBf16(Fp8((w >> 8U) & 0xFFU) * scale);
  v.z = ToBf16(Fp8((w >> 16U) & 0xFFU) * scale);
  v.w = ToBf16(Fp8(w >> 24U) * scale);
  reinterpret_cast<ushort4*>(out)[i] = v;
}

// ---- Pose ----------------------------------------------------------------

// patches[t, c * 4 + kh * 2 + kw] = pose[c, f, 2h + kh, 2w + kw], BF16.
__global__ void PosePatchKernel(const float* pose, int frames, int height,
                                int width, std::uint16_t* patches) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int ph = height / 2, pw = width / 2;
  const std::size_t tokens = static_cast<std::size_t>(frames) * ph * pw;
  if (index >= tokens * kPosePatch)
    return;
  const std::size_t token = index / kPosePatch;
  const int column = static_cast<int>(index % kPosePatch);
  const int c = column / 4, kh = (column / 2) % 2, kw = column % 2;
  const int w = static_cast<int>(token % pw);
  const int h = static_cast<int>((token / pw) % ph);
  const int f = static_cast<int>(token / (static_cast<std::size_t>(pw) * ph));
  patches[index] =
      ToBf16(pose[((static_cast<std::size_t>(c) * frames + f) * height +
                   2 * h + kh) *
                      width +
                  2 * w + kw]);
}

// hidden[i] += y[i] (y BF16).
__global__ void AddKernel(const std::uint16_t* y, std::size_t count,
                          float* hidden) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    hidden[i] += FromBf16(y[i]);
}

// ---- Motion encoder (HWC FP32 activations) -------------------------------

// ConvLayer(3, 32, 1): out[p, o] = lrelu(sum_c w[o, c] x[c, p] + b[o]) *
// sqrt(2); x channel-major [3, H * W] (a face crop), out HWC.
__global__ void FirstConvKernel(const float* x, std::size_t pixels,
                                const float* weight, const float* bias,
                                int out_channels, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pixels * out_channels)
    return;
  const std::size_t p = i / out_channels;
  const int o = static_cast<int>(i % out_channels);
  float v = bias[o];
  for (int c = 0; c < 3; ++c)
    v += weight[o * 3 + c] * x[c * pixels + p];
  v = v >= 0.0F ? v : 0.2F * v;
  out[i] = v * 1.41421356237F;
}

// cols[(y * ow + x), c * k * k + ky * k + kx] = x[y * s + ky - pad,
// x * s + kx - pad, c] (zero outside), BF16.
__global__ void Im2ColKernel(const float* in, int height, int width,
                             int channels, int kernel, int stride, int pad,
                             int out_h, int out_w, std::uint16_t* cols) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t columns =
      static_cast<std::size_t>(channels) * kernel * kernel;
  if (index >= static_cast<std::size_t>(out_h) * out_w * columns)
    return;
  const std::size_t p = index / columns;
  const int column = static_cast<int>(index % columns);
  const int c = column / (kernel * kernel);
  const int ky = (column / kernel) % kernel, kx = column % kernel;
  const int oy = static_cast<int>(p / out_w), ox = static_cast<int>(p % out_w);
  const int y = oy * stride + ky - pad, x = ox * stride + kx - pad;
  float v = 0.0F;
  if (y >= 0 && y < height && x >= 0 && x < width)
    v = in[(static_cast<std::size_t>(y) * width + x) * channels + c];
  cols[index] = ToBf16(v);
}

// upfirdn2d (up = down = 1): zero pad by `pad`, then the flipped 4 x 4
// kernel; out [(H + 2 pad - 3), (W + 2 pad - 3), C] HWC.
__global__ void BlurKernel(const float* in, int height, int width,
                           int channels, int pad, const float* kernel,
                           float* out) {
  const int out_h = height + 2 * pad - 3, out_w = width + 2 * pad - 3;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(out_h) * out_w * channels)
    return;
  const int c = static_cast<int>(index % channels);
  const std::size_t p = index / channels;
  const int oy = static_cast<int>(p / out_w), ox = static_cast<int>(p % out_w);
  float v = 0.0F;
  for (int i = 0; i < 4; ++i) {
    const int y = oy + i - pad;
    if (y < 0 || y >= height)
      continue;
    for (int j = 0; j < 4; ++j) {
      const int x = ox + j - pad;
      if (x < 0 || x >= width)
        continue;
      v += kernel[(3 - i) * 4 + (3 - j)] *
           in[(static_cast<std::size_t>(y) * width + x) * channels + c];
    }
  }
  out[index] = v;
}

// FusedLeakyReLU: out = lrelu(y + b, 0.2) * sqrt(2), y BF16 [P, C].
__global__ void LeakyKernel(const std::uint16_t* y, const float* bias,
                            std::size_t count, int channels, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  float v = FromBf16(y[i]) + bias[i % channels];
  v = v >= 0.0F ? v : 0.2F * v;
  out[i] = v * 1.41421356237F;
}

// ResBlock: out = (main + skip) / sqrt(2), skip BF16.
__global__ void ResKernel(const float* main, const std::uint16_t* skip,
                          std::size_t count, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = (main[i] + FromBf16(skip[i])) * 0.70710678118F;
}

// ---- Face adapter --------------------------------------------------------

// Layer norm without affine of [rows, 5120] FP32 -> BF16.
__global__ void PlainNormKernel(const float* hidden, std::uint16_t* out) {
  __shared__ float shared[kRowThreads];
  const std::size_t row = blockIdx.x;
  const float* x = hidden + row * kWidth;
  float values[kRowValues];
  float sum = 0;
#pragma unroll
  for (int i = 0; i < kRowValues; ++i) {
    values[i] = x[threadIdx.x + i * kRowThreads];
    sum += values[i];
  }
  const float mean = BlockSum<kRowThreads>(sum, shared) / kWidth;
  float squares = 0;
#pragma unroll
  for (int i = 0; i < kRowValues; ++i) {
    values[i] -= mean;
    squares += values[i] * values[i];
  }
  const float rstd =
      rsqrtf(BlockSum<kRowThreads>(squares, shared) / kWidth + kEpsilon);
  std::uint16_t* target = out + row * kWidth;
#pragma unroll
  for (int i = 0; i < kRowValues; ++i)
    target[threadIdx.x + i * kRowThreads] = ToBf16(values[i] * rstd);
}

// Keys: per-head RMS norm (k_norm, eps 1e-6) of the first half of the
// linear1_kv rows [rows, 2 * 5120]; values: the second half, both BF16
// [rows, 5120]. One block (128 threads) per (row, head).
__global__ void MotionKeysKernel(const std::uint16_t* kv, int rows,
                                 const float* k_norm, std::uint16_t* keys,
                                 std::uint16_t* values) {
  __shared__ float shared[kHeadDim];
  const int row = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
  const std::uint16_t* in = kv + static_cast<std::size_t>(row) * 2 * kWidth;
  const int c = head * kHeadDim + d;
  const float k = FromBf16(in[c]);
  const float rstd =
      rsqrtf(BlockSum<kHeadDim>(k * k, shared) / kHeadDim + kEpsilon);
  keys[static_cast<std::size_t>(row) * kWidth + c] = ToBf16(k * rstd * k_norm[d]);
  values[static_cast<std::size_t>(row) * kWidth + c] = in[kWidth + c];
  (void)rows;
}

// The FaceBlock attention of token t (latent frame t / frame_tokens) and
// head h over that frame's 5 motion tokens: q = rms(q) * q_norm,
// softmax(q k^T / sqrt(128)) v. One block (128 threads) per (token, head).
__global__ void FaceAttentionKernel(const std::uint16_t* query,
                                    const float* q_norm,
                                    const std::uint16_t* keys,
                                    const std::uint16_t* values,
                                    int frame_tokens, int motion_frames,
                                    std::uint16_t* out) {
  __shared__ float shared[kHeadDim];
  const std::size_t token = blockIdx.x;
  const int head = blockIdx.y, d = threadIdx.x;
  const int c = head * kHeadDim + d;
  const int frame = min(static_cast<int>(token / frame_tokens),
                        motion_frames - 1);
  float q = FromBf16(query[token * kWidth + c]);
  const float rstd =
      rsqrtf(BlockSum<kHeadDim>(q * q, shared) / kHeadDim + kEpsilon);
  q = FromBf16(ToBf16(q * rstd * q_norm[d]));
  float scores[kTokens];
  float best = -INFINITY;
#pragma unroll
  for (int j = 0; j < kTokens; ++j) {
    const std::size_t row = static_cast<std::size_t>(frame) * kTokens + j;
    scores[j] = BlockSum<kHeadDim>(q * FromBf16(keys[row * kWidth + c]),
                                   shared) *
                0.08838834764831845F;
    best = fmaxf(best, scores[j]);
  }
  float total = 0;
#pragma unroll
  for (int j = 0; j < kTokens; ++j) {
    scores[j] = expf(scores[j] - best);
    total += scores[j];
  }
  float v = 0;
#pragma unroll
  for (int j = 0; j < kTokens; ++j) {
    const std::size_t row = static_cast<std::size_t>(frame) * kTokens + j;
    v += scores[j] * FromBf16(values[row * kWidth + c]);
  }
  out[token * kWidth + c] = ToBf16(v / total);
}

unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + 255) / 256);
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

bool Check(hipError_t status, const char* operation, std::string* error) {
  if (status == hipSuccess)
    return true;
  SetError(error, std::string("Wan 2.2 Animate ") + operation + ": " +
                      hipGetErrorString(status));
  return false;
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

float Bf16ToFloat(std::uint16_t h) {
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16U;
  float value;
  std::memcpy(&value, &bits, 4);
  return value;
}

float HostFp8(std::uint8_t byte) {
  const int e = (byte >> 3) & 15, m = byte & 7;
  const float v = e == 0 ? static_cast<float>(m) * 0.001953125F
                         : std::ldexp(1.0F + static_cast<float>(m) * 0.125F,
                                      e - 7);
  return (byte & 0x80U) != 0 ? -v : v;
}

std::uint16_t HostBf16(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, 4);
  if ((bits & 0x7FFFFFFFU) > 0x7F800000U)
    return static_cast<std::uint16_t>((bits >> 16U) | 0x40U);
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >>
                                    16U);
}

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)) {}
  ~DeviceBuffer() {
    if (data_ != nullptr)
      (void)hipFree(data_);
  }
  bool Allocate(std::size_t bytes, std::string* error) {
    if (data_ != nullptr)
      (void)hipFree(data_);
    data_ = nullptr;
    return Check(hipMalloc(&data_, std::max<std::size_t>(bytes, 256)),
                 "device allocation", error);
  }
  void Release() {
    if (data_ != nullptr)
      (void)hipFree(data_);
    data_ = nullptr;
  }
  [[nodiscard]] bool empty() const { return data_ == nullptr; }
  std::uint16_t* get() const { return static_cast<std::uint16_t*>(data_); }
  float* f32() const { return static_cast<float*>(data_); }
  void* raw() const { return data_; }

private:
  void* data_{nullptr};
};

enum class DType { kF8, kF16, kBF16, kF32 };

struct TensorLocation {
  std::uint64_t offset{0};
  std::uint64_t bytes{0};
  std::uint64_t count{0};
  DType dtype{DType::kF32};
  std::vector<std::size_t> shape;
};

// A linear layer: Kijai's scaled FP8 or BF16 [rows, columns], BF16 bias.
struct Linear {
  DeviceBuffer weight;
  bool fp8{false};
  float scale{1};
  int rows{0}, columns{0};
  DeviceBuffer bias;
};

// One conv of the motion encoder as a GEMM: BF16 [out, in * k * k] with
// EqualConv2d's 1 / sqrt(in * k * k) folded in.
struct Conv {
  DeviceBuffer weight;
  int in{0}, out{0}, kernel{1};
};

struct ResBlock {
  Conv conv1, conv2, skip;
  DeviceBuffer bias1, bias2;      // FP32 [in], [out] (FusedLeakyReLU)
  DeviceBuffer blur2, blur_skip;  // FP32 [4 * 4]
};

struct FaceBlock {
  Linear kv, q, out;
  DeviceBuffer q_norm, k_norm;  // FP32 [128]
};

// The motion tokens of one context slot, per face block.
struct MotionSlot {
  int frames{0};               // DiT latent frames (reference included)
  DeviceBuffer keys, values;   // BF16 [blocks][frames * 5][5120]
  std::vector<float> motion;   // [count, 512]
  std::vector<float> tokens;   // [frames, 5, 5120]
  void Clear() {
    frames = 0;
    keys.Release();
    values.Release();
    motion.clear();
    tokens.clear();
  }
};

}  // namespace

struct Animate::Impl {
  ~Impl() {
    if (platform::FileValid(file))
      platform::CloseFile(file);
  }
  bool Index(const std::filesystem::path& path, std::string* error);
  const TensorLocation* Find(const std::string& name) const {
    const auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : &it->second;
  }
  // Any dtype as FP32 (FP8 times its `<base>.scale_weight`).
  bool ReadFloats(const std::string& name, std::size_t count,
                  std::vector<float>* out, std::string* error);
  bool UploadBf16(DeviceBuffer& buffer, const std::vector<float>& host,
                  std::string* error);
  bool UploadF32(DeviceBuffer& buffer, const std::vector<float>& host,
                 std::string* error);
  bool LoadVector(DeviceBuffer& buffer, const std::string& name,
                  std::size_t count, bool bf16, std::string* error);
  bool LoadLinear(Linear& linear, const std::string& base, int rows,
                  int columns, std::string* error);
  bool LoadConv(Conv& conv, const std::string& name, int in, int out,
                int kernel, std::string* error);
  // The BF16 weights of `linear`: FP8 unpacked into `scratch`.
  const std::uint16_t* Weights(const Linear& linear, std::uint16_t* scratch);
  bool Gemm(const Linear& linear, const std::uint16_t* input,
            std::uint16_t* out, int tokens, std::uint16_t* scratch,
            std::string* error) {
    return gemm.Run(Weights(linear, scratch), linear.bias.get(), input, out,
                    tokens, linear.rows, linear.columns, error);
  }
  // One face crop [3, 512, 512] -> EncoderApp feature [512].
  bool EncodeApp(const float* crop, std::vector<float>* feature,
                 std::string* error);
  bool ConvGemm(const Conv& conv, const float* in, int height, int width,
                int stride, int pad, int out_h, int out_w, std::string* error);
  // Motion tokens -> per-block keys / values of `slot`.
  bool UploadMotion(int slot, std::string* error);
  bool SetFaceFrames(int slot, const std::vector<float>& motion, int count,
                     int dit_frames, std::string* error);

  platform::PlatformFile file{platform::kInvalidFile};
  std::map<std::string, TensorLocation, std::less<>> tensors;
  std::size_t weight_bytes{0};
  hipStream_t stream{nullptr};
  LinearGemm gemm;

  // Pose.
  DeviceBuffer pose_weight, pose_bias;  // BF16 [W, 64], [W]
  DeviceBuffer pose_embed;              // BF16 [pose tokens, W]
  int pose_frames{0}, pose_height{0}, pose_width{0};

  // Motion encoder.
  DeviceBuffer first_weight, first_bias;  // FP32 [32, 3], [32]
  std::vector<ResBlock> res_blocks;
  Conv last;  // 512 -> 512, 4 x 4
  MotionHead head;
  DeviceBuffer act_a, act_b, act_c, act_d;  // FP32 HWC work buffers
  DeviceBuffer cols, gemm_out;              // BF16
  std::size_t act_capacity{0}, cols_capacity{0}, out_capacity{0};

  // Face encoder (host) and adapter.
  FaceEncoderWeights face_encoder;
  std::vector<FaceBlock> face_blocks;
  DeviceBuffer scratch;  // BF16 unpack target outside Forward (FP8 blocks)
  MotionSlot slots[Transformer21::kContextSlots];

  Transformer21BlockState state;
  bool active{false};
};

bool Animate::Impl::Index(const std::filesystem::path& path,
                          std::string* error) {
  std::string open_error;
  file = platform::OpenReadFile(path, &open_error);
  if (!platform::FileValid(file)) {
    SetError(error, "cannot open " + path.string() + ": " + open_error);
    return false;
  }
  const auto file_size = platform::FileSizeBytes(file);
  std::uint64_t header_size = 0;
  if (!file_size || platform::ReadFileAt(file, &header_size, 8, 0) != 8 ||
      header_size == 0 || header_size > (64U << 20U) ||
      header_size > *file_size - 8) {
    SetError(error, "invalid safetensors header in " + path.string());
    return false;
  }
  std::string text(static_cast<std::size_t>(header_size), '\0');
  if (platform::ReadFileAt(file, text.data(), text.size(), 8) != text.size()) {
    SetError(error, "short read of " + path.string());
    return false;
  }
  try {
    const json::Value header = json::parse(text);
    for (const auto& [name, entry] : header.members()) {
      std::string key = name;
      for (const char* prefix : {"model.diffusion_model.", "diffusion_model."})
        if (key.rfind(prefix, 0) == 0)
          key = key.substr(std::strlen(prefix));
      if (key.rfind("pose_patch_embedding.", 0) != 0 &&
          key.rfind("motion_encoder.", 0) != 0 &&
          key.rfind("face_encoder.", 0) != 0 &&
          key.rfind("face_adapter.", 0) != 0)
        continue;
      const std::string dtype = entry.member_str("dtype");
      TensorLocation location;
      std::uint64_t element;
      if (dtype == "F8_E4M3") {
        location.dtype = DType::kF8;
        element = 1;
      } else if (dtype == "F16") {
        location.dtype = DType::kF16;
        element = 2;
      } else if (dtype == "BF16") {
        location.dtype = DType::kBF16;
        element = 2;
      } else if (dtype == "F32") {
        location.dtype = DType::kF32;
        element = 4;
      } else {
        SetError(error, "unsupported Animate tensor type " + dtype + " of " +
                            name);
        return false;
      }
      const json::Value* shape = entry.find("shape");
      const json::Value* offsets = entry.find("data_offsets");
      if (shape == nullptr || offsets == nullptr || offsets->size() != 2) {
        SetError(error, "invalid tensor entry " + name);
        return false;
      }
      location.count = 1;
      for (const auto& dim : shape->items()) {
        location.shape.push_back(dim.as_size());
        location.count *= location.shape.back();
      }
      const std::uint64_t begin = offsets->items()[0].as_size();
      const std::uint64_t end = offsets->items()[1].as_size();
      if (end < begin || end - begin != location.count * element ||
          8 + header_size + end > *file_size) {
        SetError(error, "invalid tensor extent " + name);
        return false;
      }
      location.offset = 8 + header_size + begin;
      location.bytes = end - begin;
      tensors[key] = std::move(location);
    }
  } catch (const std::exception& e) {
    SetError(error, path.string() + ": " + e.what());
    return false;
  }
  return true;
}

bool Animate::Impl::ReadFloats(const std::string& name, std::size_t count,
                               std::vector<float>* out, std::string* error) {
  const TensorLocation* location = Find(name);
  if (location == nullptr || location->count != count) {
    SetError(error, "missing or misshaped Animate tensor " + name);
    return false;
  }
  float scale = 1.0F;
  if (location->dtype == DType::kF8) {
    const std::string base = name.substr(0, name.rfind('.'));
    std::vector<float> s;
    if (!ReadFloats(base + ".scale_weight", 1, &s, error))
      return false;
    scale = s[0];
  }
  std::vector<std::byte> bytes(location->bytes);
  if (platform::ReadFileAt(file, bytes.data(), bytes.size(),
                           location->offset) != bytes.size()) {
    SetError(error, "short read of " + name);
    return false;
  }
  out->resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    std::uint16_t half;
    switch (location->dtype) {
    case DType::kF32:
      std::memcpy(&(*out)[i], bytes.data() + 4 * i, 4);
      break;
    case DType::kF16:
      std::memcpy(&half, bytes.data() + 2 * i, 2);
      (*out)[i] = HalfToFloat(half);
      break;
    case DType::kBF16:
      std::memcpy(&half, bytes.data() + 2 * i, 2);
      (*out)[i] = Bf16ToFloat(half);
      break;
    case DType::kF8:
      (*out)[i] = HostFp8(static_cast<std::uint8_t>(bytes[i])) * scale;
      break;
    }
  }
  return true;
}

bool Animate::Impl::UploadBf16(DeviceBuffer& buffer,
                               const std::vector<float>& host,
                               std::string* error) {
  std::vector<std::uint16_t> half(host.size());
  for (std::size_t i = 0; i < host.size(); ++i)
    half[i] = HostBf16(host[i]);
  weight_bytes += half.size() * 2;
  return buffer.Allocate(half.size() * 2, error) &&
         Check(hipMemcpy(buffer.raw(), half.data(), half.size() * 2,
                         hipMemcpyHostToDevice),
               "weight upload", error);
}

bool Animate::Impl::UploadF32(DeviceBuffer& buffer,
                              const std::vector<float>& host,
                              std::string* error) {
  weight_bytes += host.size() * 4;
  return buffer.Allocate(host.size() * 4, error) &&
         Check(hipMemcpy(buffer.raw(), host.data(), host.size() * 4,
                         hipMemcpyHostToDevice),
               "weight upload", error);
}

bool Animate::Impl::LoadVector(DeviceBuffer& buffer, const std::string& name,
                               std::size_t count, bool bf16,
                               std::string* error) {
  std::vector<float> host;
  return ReadFloats(name, count, &host, error) &&
         (bf16 ? UploadBf16(buffer, host, error)
               : UploadF32(buffer, host, error));
}

bool Animate::Impl::LoadLinear(Linear& linear, const std::string& base,
                               int rows, int columns, std::string* error) {
  const std::string name = base + ".weight";
  const TensorLocation* w = Find(name);
  const std::size_t count = static_cast<std::size_t>(rows) * columns;
  if (w == nullptr || w->count != count) {
    SetError(error, "missing or misshaped Animate tensor " + name);
    return false;
  }
  linear.rows = rows;
  linear.columns = columns;
  if (!LoadVector(linear.bias, base + ".bias", rows, true, error))
    return false;
  if (w->dtype != DType::kF8) {
    std::vector<float> host;
    return ReadFloats(name, count, &host, error) &&
           UploadBf16(linear.weight, host, error);
  }
  std::vector<float> scale;
  if (!ReadFloats(base + ".scale_weight", 1, &scale, error))
    return false;
  linear.fp8 = true;
  linear.scale = scale[0];
  std::vector<std::byte> bytes(w->bytes);
  if (platform::ReadFileAt(file, bytes.data(), bytes.size(), w->offset) !=
      bytes.size()) {
    SetError(error, "short read of " + name);
    return false;
  }
  weight_bytes += bytes.size();
  return linear.weight.Allocate(bytes.size(), error) &&
         Check(hipMemcpy(linear.weight.raw(), bytes.data(), bytes.size(),
                         hipMemcpyHostToDevice),
               "weight upload", error);
}

bool Animate::Impl::LoadConv(Conv& conv, const std::string& name, int in,
                             int out, int kernel, std::string* error) {
  const std::size_t columns = static_cast<std::size_t>(in) * kernel * kernel;
  std::vector<float> host;
  if (!ReadFloats(name, out * columns, &host, error))
    return false;
  const float scale = 1.0F / std::sqrt(static_cast<float>(columns));
  for (float& v : host)
    v *= scale;
  conv.in = in;
  conv.out = out;
  conv.kernel = kernel;
  return UploadBf16(conv.weight, host, error);
}

const std::uint16_t* Animate::Impl::Weights(const Linear& linear,
                                            std::uint16_t* target) {
  if (!linear.fp8)
    return linear.weight.get();
  const std::size_t count =
      static_cast<std::size_t>(linear.rows) * linear.columns;
  UnpackKernel<<<Blocks(count / 4), 256, 0, stream>>>(
      static_cast<const std::uint32_t*>(linear.weight.raw()), count / 4,
      linear.scale, target);
  return target;
}

bool Animate::Impl::ConvGemm(const Conv& conv, const float* in, int height,
                             int width, int stride, int pad, int out_h,
                             int out_w, std::string* error) {
  const std::size_t pixels = static_cast<std::size_t>(out_h) * out_w;
  const std::size_t columns =
      static_cast<std::size_t>(conv.in) * conv.kernel * conv.kernel;
  if (pixels * columns > cols_capacity || pixels * conv.out > out_capacity) {
    SetError(error, "Wan 2.2 Animate: motion encoder buffers too small");
    return false;
  }
  Im2ColKernel<<<Blocks(pixels * columns), 256, 0, stream>>>(
      in, height, width, conv.in, conv.kernel, stride, pad, out_h, out_w,
      cols.get());
  return gemm.Run(conv.weight.get(), nullptr, cols.get(), gemm_out.get(),
                  static_cast<int>(pixels), conv.out,
                  static_cast<int>(columns), error);
}

bool Animate::Impl::EncodeApp(const float* crop, std::vector<float>* feature,
                              std::string* error) {
  // crop: host [3, 512, 512] -> act_d (channel-major).
  const std::size_t pixels = static_cast<std::size_t>(kFace) * kFace;
  if (!Check(hipMemcpyAsync(act_d.raw(), crop, pixels * 3 * 4,
                            hipMemcpyHostToDevice, stream),
             "face upload", error))
    return false;
  FirstConvKernel<<<Blocks(pixels * 32), 256, 0, stream>>>(
      act_d.f32(), pixels, first_weight.f32(), first_bias.f32(), 32,
      act_a.f32());
  float* x = act_a.f32();
  float* spare = act_b.f32();
  int size = kFace;
  for (const ResBlock& b : res_blocks) {
    const int in = b.conv1.in, out = b.conv2.out, half = size / 2;
    const std::size_t full = static_cast<std::size_t>(size) * size;
    const std::size_t small = static_cast<std::size_t>(half) * half;
    // conv1 (3 x 3, pad 1) + leaky -> act_c.
    if (!ConvGemm(b.conv1, x, size, size, 1, 1, size, size, error))
      return false;
    LeakyKernel<<<Blocks(full * in), 256, 0, stream>>>(
        gemm_out.get(), b.bias1.f32(), full * in, in, act_c.f32());
    // conv2: blur (pad 2) -> 3 x 3 stride 2 + leaky -> spare.
    BlurKernel<<<Blocks((size + 1) * static_cast<std::size_t>(size + 1) * in),
                 256, 0, stream>>>(act_c.f32(), size, size, in, 2,
                                   b.blur2.f32(), act_d.f32());
    if (!ConvGemm(b.conv2, act_d.f32(), size + 1, size + 1, 2, 0, half, half,
                  error))
      return false;
    LeakyKernel<<<Blocks(small * out), 256, 0, stream>>>(
        gemm_out.get(), b.bias2.f32(), small * out, out, spare);
    // skip: blur (pad 1) -> 1 x 1 stride 2 (no bias, no activation).
    BlurKernel<<<Blocks((size - 1) * static_cast<std::size_t>(size - 1) * in),
                 256, 0, stream>>>(x, size, size, in, 1, b.blur_skip.f32(),
                                   act_d.f32());
    if (!ConvGemm(b.skip, act_d.f32(), size - 1, size - 1, 2, 0, half, half,
                  error))
      return false;
    ResKernel<<<Blocks(small * out), 256, 0, stream>>>(spare, gemm_out.get(),
                                                       small * out, x);
    size = half;
  }
  // 4 x 4 valid conv to [512].
  if (!ConvGemm(last, x, size, size, 1, 0, 1, 1, error))
    return false;
  std::vector<std::uint16_t> half(kMotionWidth);
  if (!Check(hipMemcpyAsync(half.data(), gemm_out.raw(), half.size() * 2,
                            hipMemcpyDeviceToHost, stream),
             "motion download", error) ||
      !Check(hipStreamSynchronize(stream), "motion encoder", error))
    return false;
  feature->resize(kMotionWidth);
  for (int i = 0; i < kMotionWidth; ++i)
    (*feature)[i] = Bf16ToFloat(half[i]);
  return true;
}

bool Animate::Impl::SetFaceFrames(int slot, const std::vector<float>& motion,
                                  int count, int dit_frames,
                                  std::string* error) {
  MotionSlot& m = slots[slot];
  m.motion = motion;
  m.tokens = RunFaceEncoder(face_encoder, motion, count, dit_frames);
  if (m.tokens.empty()) {
    SetError(error, "Wan 2.2 Animate: face encoder input does not fit");
    return false;
  }
  m.frames = dit_frames;
  return UploadMotion(slot, error);
}

bool Animate::Impl::UploadMotion(int slot, std::string* error) {
  MotionSlot& m = slots[slot];
  const std::size_t rows = static_cast<std::size_t>(m.frames) * kTokens;
  const std::size_t W = kWidth, n = face_blocks.size();
  DeviceBuffer tokens, normed, kv;
  if (!tokens.Allocate(rows * W * 4, error) ||
      !normed.Allocate(rows * W * 2, error) ||
      !kv.Allocate(rows * 2 * W * 2, error) ||
      !m.keys.Allocate(n * rows * W * 2, error) ||
      !m.values.Allocate(n * rows * W * 2, error) ||
      !Check(hipMemcpyAsync(tokens.raw(), m.tokens.data(), rows * W * 4,
                            hipMemcpyHostToDevice, stream),
             "motion upload", error))
    return false;
  // pre_norm_motion, linear1_kv, k_norm.
  PlainNormKernel<<<static_cast<unsigned>(rows), kRowThreads, 0, stream>>>(
      tokens.f32(), normed.get());
  for (std::size_t i = 0; i < n; ++i) {
    const FaceBlock& b = face_blocks[i];
    if (!Gemm(b.kv, normed.get(), kv.get(), static_cast<int>(rows),
              scratch.get(), error))
      return false;
    MotionKeysKernel<<<dim3(static_cast<unsigned>(rows), kHeads), kHeadDim, 0,
                       stream>>>(kv.get(), static_cast<int>(rows),
                                 b.k_norm.f32(), m.keys.get() + i * rows * W,
                                 m.values.get() + i * rows * W);
  }
  return Check(hipGetLastError(), "motion tokens", error) &&
         Check(hipStreamSynchronize(stream), "motion tokens", error);
}

Animate::Animate(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Animate::~Animate() = default;

const std::vector<float>& Animate::motion(int slot) const {
  return impl_->slots[slot].motion;
}
const std::vector<float>& Animate::motion_tokens(int slot) const {
  return impl_->slots[slot].tokens;
}
int Animate::face_blocks() const {
  return static_cast<int>(impl_->face_blocks.size());
}
std::size_t Animate::weight_bytes() const {
  return impl_->weight_bytes;
}

std::unique_ptr<Animate> Animate::Load(const std::filesystem::path& path,
                                       hipStream_t stream,
                                       std::string* error) {
  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  m.stream = stream;
  if (!m.Index(path, error))
    return nullptr;
  int blocks = 0;
  while (m.Find("face_adapter.fuser_blocks." + std::to_string(blocks) +
                ".linear2.weight") != nullptr)
    ++blocks;
  const TensorLocation* pose = m.Find("pose_patch_embedding.weight");
  if (blocks == 0 || blocks > kDitLayers / kAdapterStride || pose == nullptr ||
      pose->count != static_cast<std::uint64_t>(kWidth) * kPosePatch ||
      m.Find("motion_encoder.dec.direction.weight") == nullptr ||
      m.Find("face_encoder.padding_tokens") == nullptr) {
    SetError(error, "not a Wan 2.2 Animate 14B checkpoint "
                    "(pose_patch_embedding [5120, 16, 1, 2, 2], "
                    "motion_encoder, face_encoder, face_adapter): " +
                        path.string());
    return nullptr;
  }
  if (!m.gemm.Init(stream, error))
    return nullptr;
  constexpr std::size_t W = kWidth;
  // Pose patch embedding (Conv3d [W, 16, 1, 2, 2]: c * 4 + kh * 2 + kw).
  {
    std::vector<float> host;
    if (!m.ReadFloats("pose_patch_embedding.weight", W * kPosePatch, &host,
                      error) ||
        !m.UploadBf16(m.pose_weight, host, error) ||
        !m.LoadVector(m.pose_bias, "pose_patch_embedding.bias", W, true,
                      error))
      return nullptr;
  }
  // Motion encoder: ConvLayer(3, 32, 1), 7 ResBlocks, EqualConv2d 4 x 4.
  {
    const std::string p = "motion_encoder.enc.net_app.convs.";
    std::vector<float> host;
    if (!m.ReadFloats(p + "0.0.weight", 32 * 3, &host, error))
      return nullptr;
    for (float& v : host)
      v /= std::sqrt(3.0F);
    if (!m.UploadF32(m.first_weight, host, error) ||
        !m.LoadVector(m.first_bias, p + "0.1.bias", 32, false, error))
      return nullptr;
    const int channels[8] = {32, 64, 128, 256, 512, 512, 512, 512};
    m.res_blocks = std::vector<ResBlock>(7);
    for (int i = 0; i < 7; ++i) {
      ResBlock& b = m.res_blocks[i];
      const int in = channels[i], out = channels[i + 1];
      const std::string r = p + std::to_string(i + 1) + ".";
      if (!m.LoadConv(b.conv1, r + "conv1.0.weight", in, in, 3, error) ||
          !m.LoadVector(b.bias1, r + "conv1.1.bias", in, false, error) ||
          !m.LoadVector(b.blur2, r + "conv2.0.kernel", 16, false, error) ||
          !m.LoadConv(b.conv2, r + "conv2.1.weight", in, out, 3, error) ||
          !m.LoadVector(b.bias2, r + "conv2.2.bias", out, false, error) ||
          !m.LoadVector(b.blur_skip, r + "skip.0.kernel", 16, false, error) ||
          !m.LoadConv(b.skip, r + "skip.1.weight", in, out, 1, error))
        return nullptr;
    }
    if (!m.LoadConv(m.last, p + "8.weight", 512, kMotionWidth, 4, error))
      return nullptr;
    // Encoder.fc (EqualLinear: weight / sqrt(in), lr_mul 1) and Direction.
    m.head.width = kMotionWidth;
    m.head.motion = kMotionDim;
    for (int i = 0; i < 5; ++i) {
      const int out = i < 4 ? kMotionWidth : kMotionDim;
      const std::string f = "motion_encoder.enc.fc." + std::to_string(i) + ".";
      std::vector<float> weight, bias;
      if (!m.ReadFloats(f + "weight", static_cast<std::size_t>(out) *
                                          kMotionWidth,
                        &weight, error) ||
          !m.ReadFloats(f + "bias", out, &bias, error))
        return nullptr;
      for (float& v : weight)
        v /= std::sqrt(static_cast<float>(kMotionWidth));
      m.head.weights.push_back(std::move(weight));
      m.head.biases.push_back(std::move(bias));
    }
    if (!m.ReadFloats("motion_encoder.dec.direction.weight",
                      static_cast<std::size_t>(kMotionWidth) * kMotionDim,
                      &host, error))
      return nullptr;
    SetMotionDirections(host, &m.head);
  }
  // Face encoder (host FP32).
  {
    FaceEncoderWeights& f = m.face_encoder;
    const std::string p = "face_encoder.";
    const std::size_t wide = static_cast<std::size_t>(kFaceHeads) *
                             kFaceChannels;
    if (!m.ReadFloats(p + "conv1_local.conv.weight", wide * kMotionWidth * 3,
                      &f.conv1_weight, error) ||
        !m.ReadFloats(p + "conv1_local.conv.bias", wide, &f.conv1_bias,
                      error) ||
        !m.ReadFloats(p + "conv2.conv.weight",
                      static_cast<std::size_t>(kFaceChannels) * kFaceChannels *
                          3,
                      &f.conv2_weight, error) ||
        !m.ReadFloats(p + "conv2.conv.bias", kFaceChannels, &f.conv2_bias,
                      error) ||
        !m.ReadFloats(p + "conv3.conv.weight", f.conv2_weight.size(),
                      &f.conv3_weight, error) ||
        !m.ReadFloats(p + "conv3.conv.bias", kFaceChannels, &f.conv3_bias,
                      error) ||
        !m.ReadFloats(p + "out_proj.weight", W * kFaceChannels, &f.out_weight,
                      error) ||
        !m.ReadFloats(p + "out_proj.bias", W, &f.out_bias, error) ||
        !m.ReadFloats(p + "padding_tokens", W, &f.padding, error))
      return nullptr;
  }
  // Face adapter blocks.
  m.face_blocks = std::vector<FaceBlock>(blocks);
  bool fp8 = false;
  for (int i = 0; i < blocks; ++i) {
    FaceBlock& b = m.face_blocks[i];
    const std::string p = "face_adapter.fuser_blocks." + std::to_string(i) + ".";
    if (!m.LoadLinear(b.kv, p + "linear1_kv", 2 * kWidth, kWidth, error) ||
        !m.LoadLinear(b.q, p + "linear1_q", kWidth, kWidth, error) ||
        !m.LoadLinear(b.out, p + "linear2", kWidth, kWidth, error) ||
        !m.LoadVector(b.q_norm, p + "q_norm.weight", kHeadDim, false, error) ||
        !m.LoadVector(b.k_norm, p + "k_norm.weight", kHeadDim, false, error))
      return nullptr;
    fp8 = fp8 || b.kv.fp8;
  }
  if (fp8 && !m.scratch.Allocate(static_cast<std::size_t>(2) * W * W * 2,
                                 error))
    return nullptr;
  platform::CloseFile(m.file);
  m.file = platform::kInvalidFile;
  return std::unique_ptr<Animate>(new Animate(std::move(impl)));
}

bool Animate::SetPose(std::span<const float> latents, int frames, int height,
                      int width, std::string* error) {
  Impl& m = *impl_;
  if (latents.empty()) {
    m.pose_embed.Release();
    m.pose_frames = 0;
    return true;
  }
  if (frames <= 0 || height <= 0 || width <= 0 || height % 2 != 0 ||
      width % 2 != 0 ||
      latents.size() !=
          static_cast<std::size_t>(kPoseChannels) * frames * height * width) {
    SetError(error, "Wan 2.2 Animate: pose latents do not match their shape");
    return false;
  }
  const std::size_t tokens =
      static_cast<std::size_t>(frames) * (height / 2) * (width / 2);
  DeviceBuffer input, patches;
  if (!input.Allocate(latents.size() * 4, error) ||
      !patches.Allocate(tokens * kPosePatch * 2, error) ||
      !m.pose_embed.Allocate(tokens * kWidth * 2, error) ||
      !Check(hipMemcpyAsync(input.raw(), latents.data(), latents.size() * 4,
                            hipMemcpyHostToDevice, m.stream),
             "pose upload", error))
    return false;
  PosePatchKernel<<<Blocks(tokens * kPosePatch), 256, 0, m.stream>>>(
      input.f32(), frames, height, width, patches.get());
  if (!m.gemm.Run(m.pose_weight.get(), m.pose_bias.get(), patches.get(),
                  m.pose_embed.get(), static_cast<int>(tokens), kWidth,
                  kPosePatch, error) ||
      !Check(hipStreamSynchronize(m.stream), "pose embedding", error))
    return false;
  m.pose_frames = frames;
  m.pose_height = height;
  m.pose_width = width;
  return true;
}

namespace {

// Lazily sized work buffers of the motion encoder (the largest stage:
// 512 x 512 x 32 activations, 3 x 3 im2col of 32 channels at 512 x 512).
bool ReserveEncoder(DeviceBuffer& a, DeviceBuffer& b, DeviceBuffer& c,
                    DeviceBuffer& d, DeviceBuffer& cols, DeviceBuffer& out,
                    std::size_t* act_capacity, std::size_t* cols_capacity,
                    std::size_t* out_capacity, std::string* error) {
  if (*act_capacity != 0)
    return true;
  const std::size_t act = static_cast<std::size_t>(kFace + 1) * (kFace + 1) * 32;
  const std::size_t columns = static_cast<std::size_t>(kFace) * kFace * 32 * 9;
  const std::size_t outputs = static_cast<std::size_t>(kFace) * kFace * 32;
  if (!a.Allocate(act * 4, error) || !b.Allocate(act * 4, error) ||
      !c.Allocate(act * 4, error) || !d.Allocate(act * 4, error) ||
      !cols.Allocate(columns * 2, error) || !out.Allocate(outputs * 2, error))
    return false;
  *act_capacity = act;
  *cols_capacity = columns;
  *out_capacity = outputs;
  return true;
}

}  // namespace

bool Animate::SetFace(int slot, std::span<const float> face, int count,
                      int dit_frames, std::string* error) {
  Impl& m = *impl_;
  if (slot < 0 || slot >= Transformer21::kContextSlots) {
    SetError(error, "Wan 2.2 Animate: invalid context slot");
    return false;
  }
  MotionSlot& s = m.slots[slot];
  if (count == 0) {
    s.Clear();
    return true;
  }
  const std::size_t crop = static_cast<std::size_t>(3) * kFace * kFace;
  if (count < 0 || dit_frames < 1 || face.size() != crop * count) {
    SetError(error, "Wan 2.2 Animate: face crops do not match their count");
    return false;
  }
  if (!ReserveEncoder(m.act_a, m.act_b, m.act_c, m.act_d, m.cols, m.gemm_out,
                      &m.act_capacity, &m.cols_capacity, &m.out_capacity,
                      error))
    return false;
  std::vector<float> motion(static_cast<std::size_t>(count) * kMotionWidth);
  std::vector<float> feature;
  for (int f = 0; f < count; ++f) {
    if (!m.EncodeApp(face.data() + f * crop, &feature, error))
      return false;
    const std::vector<float> vector = MotionVector(m.head, feature);
    std::copy(vector.begin(), vector.end(),
              motion.begin() + static_cast<std::size_t>(f) * kMotionWidth);
  }
  return m.SetFaceFrames(slot, motion, count, dit_frames, error);
}

bool Animate::SetUniformFace(int slot, float value, int count, int dit_frames,
                             std::string* error) {
  Impl& m = *impl_;
  if (slot < 0 || slot >= Transformer21::kContextSlots || count < 0 ||
      dit_frames < 1) {
    SetError(error, "Wan 2.2 Animate: invalid uniform face");
    return false;
  }
  if (count == 0) {
    m.slots[slot].Clear();
    return true;
  }
  if (!ReserveEncoder(m.act_a, m.act_b, m.act_c, m.act_d, m.cols, m.gemm_out,
                      &m.act_capacity, &m.cols_capacity, &m.out_capacity,
                      error))
    return false;
  const std::vector<float> crop(static_cast<std::size_t>(3) * kFace * kFace,
                                value);
  std::vector<float> feature;
  if (!m.EncodeApp(crop.data(), &feature, error))
    return false;
  const std::vector<float> vector = MotionVector(m.head, feature);
  std::vector<float> motion(static_cast<std::size_t>(count) * kMotionWidth);
  for (int f = 0; f < count; ++f)
    std::copy(vector.begin(), vector.end(),
              motion.begin() + static_cast<std::size_t>(f) * kMotionWidth);
  return m.SetFaceFrames(slot, motion, count, dit_frames, error);
}

bool Animate::Begin(const float* input, int frames, int height, int width,
                    const float* temb, hipStream_t stream,
                    std::string* error) {
  (void)input, (void)temb, (void)stream;
  Impl& m = *impl_;
  m.active = false;
  if (m.pose_frames != 0 &&
      (frames < 2 || height != m.pose_height || width != m.pose_width)) {
    SetError(error, "Wan 2.2 Animate: pose latents and video latent differ "
                    "in shape");
    return false;
  }
  return true;
}

bool Animate::BeginBlocks(const Transformer21BlockState& s, hipStream_t stream,
                          std::string* error) {
  Impl& m = *impl_;
  m.active = false;
  if (s.context_slot < 0 || s.context_slot >= Transformer21::kContextSlots ||
      s.hidden == nullptr) {
    SetError(error, "Wan 2.2 Animate: invalid transformer state");
    return false;
  }
  // x[:, :, 1 : pose frames + 1] += pose_patch_embedding(pose)[:, :, :T - 1].
  if (m.pose_frames > 0) {
    const std::size_t frame_tokens =
        static_cast<std::size_t>(s.height / 2) * (s.width / 2);
    const std::size_t frames =
        std::min<std::size_t>(m.pose_frames, s.frames - 1);
    const std::size_t count = frames * frame_tokens * kWidth;
    AddKernel<<<Blocks(count), 256, 0, stream>>>(
        m.pose_embed.get(), count, s.hidden + frame_tokens * kWidth);
  }
  const MotionSlot& slot = m.slots[s.context_slot];
  if (slot.frames > 0 && slot.frames != s.frames) {
    SetError(error, "Wan 2.2 Animate: face motion and video latent differ in "
                    "frames");
    return false;
  }
  m.state = s;
  m.state.hidden = nullptr;  // only valid here
  m.active = slot.frames > 0;
  return Check(hipGetLastError(), "pose", error);
}

bool Animate::AfterBlock(int layer, float* hidden, hipStream_t stream,
                         std::string* error) {
  Impl& m = *impl_;
  if (!m.active || layer % kAdapterStride != 0)
    return true;
  const int n = layer / kAdapterStride;
  if (n >= static_cast<int>(m.face_blocks.size()))
    return true;
  const Transformer21BlockState& s = m.state;
  const MotionSlot& slot = m.slots[s.context_slot];
  const FaceBlock& b = m.face_blocks[n];
  const std::size_t W = kWidth, T = s.tokens, count = T * W;
  const std::size_t rows = static_cast<std::size_t>(slot.frames) * kTokens;
  const int frame_tokens = (s.height / 2) * (s.width / 2);
  // q = linear1_q(pre_norm_feat(x)); attention to the frame's motion
  // tokens; x += linear2(attention).
  PlainNormKernel<<<static_cast<unsigned>(T), kRowThreads, 0, stream>>>(
      hidden, s.normed);
  if (!m.Gemm(b.q, s.normed, s.linear, s.tokens, s.weights, error))
    return false;
  FaceAttentionKernel<<<dim3(static_cast<unsigned>(T), kHeads), kHeadDim, 0,
                        stream>>>(s.linear, b.q_norm.f32(),
                                  slot.keys.get() + n * rows * W,
                                  slot.values.get() + n * rows * W,
                                  frame_tokens, slot.frames, s.attention);
  if (!m.Gemm(b.out, s.attention, s.linear, s.tokens, s.weights, error))
    return false;
  AddKernel<<<Blocks(count), 256, 0, stream>>>(s.linear, count, hidden);
  return Check(hipGetLastError(), "face adapter", error);
}

}  // namespace gufo::models::wan22
