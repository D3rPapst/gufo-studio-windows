#if defined(ENGINE_ENABLE_HIP)

// MiniMax-H3-Image 2D ViT VAE decoder on gfx1151, F32 throughout. A 256-pixel
// tile is 16 x 16 latent tokens + 5 suffix rows = 261 rows, so the F32 path is
// cheap and stays the reference arithmetic (no 16-bit mode).

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include <gufo/ext/platform/mapped_file.hpp>
#include "src/models/minimax_h3/image_vae.hpp"
#include "src/models/minimax_h3/video_vae_ops.hip.hpp"

namespace gufo::minimax_h3 {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kWeightStagingBytes = 16ULL << 20U;
constexpr std::size_t kSector = 4096;
constexpr std::size_t kSectorSlack = 2U * kSector;
constexpr std::uint32_t kChannels = kH3ImageVaeLatentChannels;
constexpr std::uint32_t kHidden = kH3ImageVaeHidden;
constexpr std::uint32_t kHeads = kH3ImageVaeHeads;
constexpr std::uint32_t kHeadDimension = kH3ImageVaeHeadDimension;
constexpr std::uint32_t kAttentionWidth = kHeads * kHeadDimension;
constexpr std::uint32_t kFeedForward = kH3ImageVaeFeedForward;
constexpr std::uint32_t kRegisters = kH3ImageVaeRegisters;
constexpr std::uint32_t kRopeHalf = kH3ImageVaeRopeHalf;
constexpr std::uint32_t kOutputPatch = kH3ImageVaeOutputPatch;
constexpr float kNormEpsilon = kH3ImageVaeNormEpsilon;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool IsCancelled(const CancellationToken* cancellation) {
  return cancellation != nullptr && cancellation->IsCancelled();
}

double Milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::string HipError(std::string_view operation, hipError_t status) {
  return std::string(operation) + ": " + hipGetErrorString(status);
}

bool CheckLaunch(std::string_view operation, std::string* error) {
  const hipError_t status = hipPeekAtLastError();
  if (status == hipSuccess) {
    return true;
  }
  SetError(error, HipError(operation, status));
  return false;
}

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { Reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        bytes_(std::exchange(other.bytes_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Reset();
      data_ = std::exchange(other.data_, nullptr);
      bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
  }

  bool Allocate(std::size_t bytes, std::string* error) {
    Reset();
    if (bytes == 0) {
      return true;
    }
    const hipError_t status = hipMalloc(&data_, bytes);
    if (status != hipSuccess) {
      SetError(error, HipError("hipMalloc MiniMax-H3-Image VAE", status));
      return false;
    }
    bytes_ = bytes;
    return true;
  }

  void Reset() noexcept {
    if (data_ != nullptr) {
      (void)hipFree(data_);
      data_ = nullptr;
    }
    bytes_ = 0;
  }

  template<typename T>
  [[nodiscard]] T* As() noexcept {
    return static_cast<T*>(data_);
  }
  template<typename T>
  [[nodiscard]] const T* As() const noexcept {
    return static_cast<const T*>(data_);
  }
  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] const void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

const TensorDescriptor* RequireTensor(
    const ModelInventory& inventory, std::string_view name,
    std::initializer_list<std::uint64_t> shape, std::string* error) {
  const TensorDescriptor* descriptor = inventory.FindTensor(name);
  if (descriptor == nullptr) {
    SetError(error,
             "MiniMax-H3-Image VAE tensor is missing: " + std::string(name));
    return nullptr;
  }
  if (descriptor->dtype != DType::kFloat32 ||
      descriptor->shape !=
          std::vector<std::uint64_t>(shape.begin(), shape.end())) {
    SetError(error, "MiniMax-H3-Image VAE tensor contract differs: " +
                        std::string(name));
    return nullptr;
  }
  return descriptor;
}

bool LoadTensor(const ModelInventory& inventory,
                const TensorDescriptor& descriptor, hipStream_t stream,
                void* staging, const CancellationToken* cancellation,
                DeviceBuffer* output, std::string* error) {
  if (descriptor.payload_bytes > std::numeric_limits<std::size_t>::max() ||
      !output->Allocate(static_cast<std::size_t>(descriptor.payload_bytes),
                        error)) {
    return false;
  }
  const auto source_bytes = static_cast<std::size_t>(descriptor.payload_bytes);
  const std::filesystem::path path = inventory.model_root() / descriptor.shard;
  std::string open_error;
  const platform::PlatformFile file = platform::OpenReadFile(path, &open_error);
  if (!platform::FileValid(file)) {
    SetError(error, "cannot open MiniMax-H3-Image VAE " + path.string() +
                        ": " + open_error);
    output->Reset();
    return false;
  }
  std::size_t copied = 0;
  bool success = true;
  while (copied < source_bytes) {
    if (IsCancelled(cancellation)) {
      SetError(error, "MiniMax-H3-Image VAE weight load cancelled");
      success = false;
      break;
    }
    const std::size_t chunk =
        std::min(kWeightStagingBytes, source_bytes - copied);
    const std::uint64_t position = descriptor.file_offset + copied;
    const std::uint64_t aligned = position / kSector * kSector;
    const auto lead = static_cast<std::size_t>(position - aligned);
    const std::size_t span = (lead + chunk + kSector - 1U) / kSector * kSector;
    if (platform::ReadFileParallel(file, staging, span, aligned) <
        lead + chunk) {
      SetError(error, "short read loading MiniMax-H3-Image VAE tensor " +
                          descriptor.name);
      success = false;
      break;
    }
    hipError_t status =
        hipMemcpyAsync(static_cast<std::byte*>(output->data()) + copied,
                       static_cast<const std::byte*>(staging) + lead, chunk,
                       hipMemcpyHostToDevice, stream);
    if (status == hipSuccess) {
      status = hipStreamSynchronize(stream);
    }
    if (status != hipSuccess) {
      SetError(error, HipError("copy MiniMax-H3-Image VAE tensor", status));
      success = false;
      break;
    }
    copied += chunk;
  }
  platform::CloseFile(file);
  if (!success) {
    output->Reset();
  }
  return success;
}

class Blas {
public:
  Blas() {
    if (rocblas_create_handle(&handle_) != rocblas_status_success) {
      handle_ = nullptr;
      return;
    }
    (void)rocblas_set_atomics_mode(handle_, rocblas_atomics_not_allowed);
  }
  ~Blas() {
    if (handle_ != nullptr) {
      (void)rocblas_destroy_handle(handle_);
    }
  }
  Blas(const Blas&) = delete;
  Blas& operator=(const Blas&) = delete;

  [[nodiscard]] bool valid() const noexcept { return handle_ != nullptr; }

  // output[batch][rows] = input[batch][columns] * weight[rows][columns]^T
  bool Linear(const float* weight, const float* input, float* output,
              std::size_t batch, std::size_t rows, std::size_t columns,
              hipStream_t stream, std::string* error) {
    if (!SetStream(stream, error)) {
      return false;
    }
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    return Check(
        rocblas_gemm_ex(handle_, rocblas_operation_transpose,
                        rocblas_operation_none, static_cast<rocblas_int>(rows),
                        static_cast<rocblas_int>(batch),
                        static_cast<rocblas_int>(columns), &alpha, weight,
                        rocblas_datatype_f32_r,
                        static_cast<rocblas_int>(columns), input,
                        rocblas_datatype_f32_r,
                        static_cast<rocblas_int>(columns), &beta, output,
                        rocblas_datatype_f32_r, static_cast<rocblas_int>(rows),
                        output, rocblas_datatype_f32_r,
                        static_cast<rocblas_int>(rows), rocblas_datatype_f32_r,
                        rocblas_gemm_algo_standard, 0, 0),
        "linear", error);
  }

  // scores[head][query][key] = scale * q . k
  bool Scores(const float* query, const float* key, float* scores,
              std::size_t sequence, float scale, hipStream_t stream,
              std::string* error) {
    if (!SetStream(stream, error)) {
      return false;
    }
    constexpr float beta = 0.0F;
    const auto rows = static_cast<rocblas_int>(sequence);
    const auto token_stride = static_cast<rocblas_int>(kAttentionWidth);
    const auto head_stride = static_cast<rocblas_stride>(kHeadDimension);
    const auto score_stride = static_cast<rocblas_stride>(sequence * sequence);
    return Check(
        rocblas_gemm_strided_batched_ex(
            handle_, rocblas_operation_transpose, rocblas_operation_none, rows,
            rows, kHeadDimension, &scale, key, rocblas_datatype_f32_r,
            token_stride, head_stride, query, rocblas_datatype_f32_r,
            token_stride, head_stride, &beta, scores, rocblas_datatype_f32_r,
            rows, score_stride, scores, rocblas_datatype_f32_r, rows,
            score_stride, kHeads, rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard, 0, 0),
        "attention score", error);
  }

  bool Values(const float* value, const float* probabilities, float* output,
              std::size_t sequence, hipStream_t stream, std::string* error) {
    if (!SetStream(stream, error)) {
      return false;
    }
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const auto rows = static_cast<rocblas_int>(sequence);
    const auto token_stride = static_cast<rocblas_int>(kAttentionWidth);
    const auto head_stride = static_cast<rocblas_stride>(kHeadDimension);
    const auto probability_stride =
        static_cast<rocblas_stride>(sequence * sequence);
    return Check(
        rocblas_gemm_strided_batched_ex(
            handle_, rocblas_operation_none, rocblas_operation_none,
            kHeadDimension, rows, rows, &alpha, value, rocblas_datatype_f32_r,
            token_stride, head_stride, probabilities, rocblas_datatype_f32_r,
            rows, probability_stride, &beta, output, rocblas_datatype_f32_r,
            token_stride, head_stride, output, rocblas_datatype_f32_r,
            token_stride, head_stride, kHeads, rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard, 0, 0),
        "attention value", error);
  }

private:
  bool SetStream(hipStream_t stream, std::string* error) {
    if (handle_ == nullptr ||
        rocblas_set_stream(handle_, stream) != rocblas_status_success) {
      SetError(error, "invalid MiniMax-H3-Image VAE rocBLAS handle");
      return false;
    }
    return true;
  }

  static bool Check(rocblas_status status, std::string_view phase,
                    std::string* error) {
    if (status == rocblas_status_success) {
      return true;
    }
    SetError(error, "MiniMax-H3-Image VAE " + std::string(phase) +
                        " GEMM failed: " +
                        std::to_string(static_cast<int>(status)));
    return false;
  }

  rocblas_handle handle_{nullptr};
};

struct BlockWeights {
  DeviceBuffer norm1;
  DeviceBuffer qkv_weight;
  DeviceBuffer qkv_bias;
  DeviceBuffer output_weight;
  DeviceBuffer output_bias;
  DeviceBuffer scale1;
  DeviceBuffer norm2;
  DeviceBuffer fc1_weight;
  DeviceBuffer fc1_bias;
  DeviceBuffer fc2_weight;
  DeviceBuffer fc2_bias;
  DeviceBuffer scale2;
};

}  // namespace

struct ImageVaeDecoder::Impl {
  const ModelInventory* inventory{nullptr};
  ImageVaePlan plan;
  hipStream_t stream{nullptr};
  void* staging{nullptr};
  Blas blas;

  DeviceBuffer latent_mean;
  DeviceBuffer latent_std;
  DeviceBuffer post_weight;
  DeviceBuffer post_bias;
  DeviceBuffer embed_weight;
  DeviceBuffer embed_bias;
  DeviceBuffer registers;
  std::vector<BlockWeights> blocks;
  DeviceBuffer final_norm_weight;
  DeviceBuffer final_norm_bias;
  DeviceBuffer output_weight;
  DeviceBuffer output_bias;

  DeviceBuffer normalized_latent;
  DeviceBuffer latent_rows;
  DeviceBuffer post;
  DeviceBuffer hidden;
  DeviceBuffer norm;
  DeviceBuffer qkv;
  DeviceBuffer query;
  DeviceBuffer key;
  DeviceBuffer value;
  DeviceBuffer attention_scores;
  DeviceBuffer attention_heads;
  DeviceBuffer branch;
  DeviceBuffer fc1;
  DeviceBuffer activated;
  DeviceBuffer rope_cos;
  DeviceBuffer rope_sin;
  DeviceBuffer projected;

  std::vector<float> projected_host;
  ImageVaeTelemetry base_telemetry;

  ~Impl() {
    if (staging != nullptr) {
      (void)hipHostFree(staging);
    }
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
  }

  bool Copy(void* target, const void* source, std::size_t bytes,
            hipMemcpyKind kind, std::string_view label, std::string* error) {
    const hipError_t status =
        hipMemcpyAsync(target, source, bytes, kind, stream);
    if (status != hipSuccess) {
      SetError(error, HipError(label, status));
      return false;
    }
    return true;
  }

  bool LoadNamed(const std::string& name,
                 std::initializer_list<std::uint64_t> shape,
                 DeviceBuffer* output, const CancellationToken* cancellation,
                 std::string* error) {
    const TensorDescriptor* descriptor =
        RequireTensor(*inventory, name, shape, error);
    if (descriptor == nullptr ||
        !LoadTensor(*inventory, *descriptor, stream, staging, cancellation,
                    output, error)) {
      return false;
    }
    base_telemetry.weight_bytes += output->bytes();
    base_telemetry.cumulative_allocation_bytes += output->bytes();
    return true;
  }

  bool LoadWeights(const CancellationToken* cancellation, std::string* error) {
    const auto begin = Clock::now();
    if (!LoadNamed("post_quant_conv.weight", {kChannels, kChannels, 1, 1},
                   &post_weight, cancellation, error) ||
        !LoadNamed("post_quant_conv.bias", {kChannels}, &post_bias,
                   cancellation, error) ||
        !LoadNamed("decoder.x_embedder.weight", {kHidden, kChannels},
                   &embed_weight, cancellation, error) ||
        !LoadNamed("decoder.x_embedder.bias", {kHidden}, &embed_bias,
                   cancellation, error) ||
        !LoadNamed("decoder.register_tokens", {1, kRegisters, kHidden},
                   &registers, cancellation, error)) {
      return false;
    }
    blocks.resize(kH3ImageVaeBlocks);
    for (std::size_t block = 0; block < blocks.size(); ++block) {
      const std::string prefix =
          "decoder.transformer_blocks." + std::to_string(block) + ".";
      BlockWeights& weight = blocks[block];
      if (!LoadNamed(prefix + "norm1.weight", {kHidden}, &weight.norm1,
                     cancellation, error) ||
          !LoadNamed(prefix + "attn.to_qkv.weight",
                     {kAttentionWidth * 3U, kHidden}, &weight.qkv_weight,
                     cancellation, error) ||
          !LoadNamed(prefix + "attn.to_qkv.bias", {kAttentionWidth * 3U},
                     &weight.qkv_bias, cancellation, error) ||
          !LoadNamed(prefix + "attn.to_out.weight", {kHidden, kAttentionWidth},
                     &weight.output_weight, cancellation, error) ||
          !LoadNamed(prefix + "attn.to_out.bias", {kHidden},
                     &weight.output_bias, cancellation, error) ||
          !LoadNamed(prefix + "scale1", {kHidden}, &weight.scale1, cancellation,
                     error) ||
          !LoadNamed(prefix + "norm2.weight", {kHidden}, &weight.norm2,
                     cancellation, error) ||
          !LoadNamed(prefix + "ff.w1.weight", {kFeedForward * 2U, kHidden},
                     &weight.fc1_weight, cancellation, error) ||
          !LoadNamed(prefix + "ff.w1.bias", {kFeedForward * 2U},
                     &weight.fc1_bias, cancellation, error) ||
          !LoadNamed(prefix + "ff.w2.weight", {kHidden, kFeedForward},
                     &weight.fc2_weight, cancellation, error) ||
          !LoadNamed(prefix + "ff.w2.bias", {kHidden}, &weight.fc2_bias,
                     cancellation, error) ||
          !LoadNamed(prefix + "scale2", {kHidden}, &weight.scale2, cancellation,
                     error)) {
        return false;
      }
    }
    if (!LoadNamed("decoder.norm_out.weight", {kHidden}, &final_norm_weight,
                   cancellation, error) ||
        !LoadNamed("decoder.norm_out.bias", {kHidden}, &final_norm_bias,
                   cancellation, error) ||
        !LoadNamed("decoder.proj_out.weight", {kOutputPatch, kHidden},
                   &output_weight, cancellation, error) ||
        !LoadNamed("decoder.proj_out.bias", {kOutputPatch}, &output_bias,
                   cancellation, error)) {
      return false;
    }
    base_telemetry.load_ms = Milliseconds(begin, Clock::now());
    return true;
  }

  bool AllocateRuntime(const ImageVaeLatentNormalization& normalization,
                       std::string* error) {
    const auto f32 = [](std::size_t elements) {
      return elements * sizeof(float);
    };
    const std::size_t sequence = plan.sequence;
    const std::size_t latent_elements =
        static_cast<std::size_t>(kChannels) * plan.latent_height *
        plan.latent_width;
    if (!latent_mean.Allocate(f32(kChannels), error) ||
        !latent_std.Allocate(f32(kChannels), error) ||
        !normalized_latent.Allocate(f32(latent_elements), error) ||
        !latent_rows.Allocate(f32(plan.patches * kChannels), error) ||
        !post.Allocate(latent_rows.bytes(), error) ||
        !hidden.Allocate(f32(sequence * kHidden), error) ||
        !norm.Allocate(hidden.bytes(), error) ||
        !qkv.Allocate(f32(sequence * kAttentionWidth * 3U), error) ||
        !query.Allocate(f32(sequence * kAttentionWidth), error) ||
        !key.Allocate(query.bytes(), error) ||
        !value.Allocate(query.bytes(), error) ||
        !attention_scores.Allocate(f32(kHeads * sequence * sequence), error) ||
        !attention_heads.Allocate(query.bytes(), error) ||
        !branch.Allocate(f32(sequence * kHidden), error) ||
        !fc1.Allocate(f32(sequence * kFeedForward * 2U), error) ||
        !activated.Allocate(f32(sequence * kFeedForward), error) ||
        !rope_cos.Allocate(f32(sequence * kRopeHalf), error) ||
        !rope_sin.Allocate(rope_cos.bytes(), error) ||
        !projected.Allocate(f32(plan.patches * kOutputPatch), error)) {
      return false;
    }
    std::vector<float> cosine;
    std::vector<float> sine;
    BuildImageVaeRope(plan.tile_latent_height, plan.tile_latent_width, &cosine,
                      &sine);
    if (!Copy(latent_mean.data(), normalization.mean.data(), f32(kChannels),
              hipMemcpyHostToDevice, "copy H3-Image latent mean", error) ||
        !Copy(latent_std.data(), normalization.deviation.data(), f32(kChannels),
              hipMemcpyHostToDevice, "copy H3-Image latent std", error) ||
        !Copy(rope_cos.data(), cosine.data(), rope_cos.bytes(),
              hipMemcpyHostToDevice, "copy H3-Image VAE RoPE", error) ||
        !Copy(rope_sin.data(), sine.data(), rope_sin.bytes(),
              hipMemcpyHostToDevice, "copy H3-Image VAE RoPE", error)) {
      return false;
    }
    const hipError_t status = hipStreamSynchronize(stream);
    if (status != hipSuccess) {
      SetError(error, HipError("synchronize H3-Image VAE setup", status));
      return false;
    }
    projected_host.resize(static_cast<std::size_t>(plan.patches) *
                          kOutputPatch);
    const std::uint64_t scratch =
        latent_rows.bytes() + post.bytes() + hidden.bytes() + norm.bytes() +
        qkv.bytes() + query.bytes() * 4U + attention_scores.bytes() +
        branch.bytes() + fc1.bytes() + activated.bytes() + rope_cos.bytes() +
        rope_sin.bytes() + projected.bytes();
    base_telemetry.persistent_bytes = base_telemetry.weight_bytes +
                                      latent_mean.bytes() + latent_std.bytes() +
                                      normalized_latent.bytes();
    base_telemetry.scratch_bytes = scratch;
    base_telemetry.peak_live_bytes =
        base_telemetry.persistent_bytes + base_telemetry.scratch_bytes;
    base_telemetry.cumulative_allocation_bytes += scratch;
    base_telemetry.scratch_address =
        reinterpret_cast<std::uintptr_t>(qkv.data());
    return true;
  }

  bool Linear(const DeviceBuffer& input, const DeviceBuffer& weight,
              const DeviceBuffer& bias, DeviceBuffer* output, std::size_t rows,
              std::size_t input_width, std::size_t output_width,
              std::string* error) {
    if (!blas.Linear(weight.As<float>(), input.As<float>(),
                     output->As<float>(), rows, output_width, input_width,
                     stream, error)) {
      return false;
    }
    video_vae_ops::LaunchAddBias(output->As<float>(), bias.As<float>(),
                                 static_cast<std::uint32_t>(rows),
                                 static_cast<std::uint32_t>(output_width),
                                 stream);
    return CheckLaunch("H3-Image VAE linear bias", error);
  }

  bool RunBlock(BlockWeights& weight, std::string* error) {
    const std::uint32_t sequence = plan.sequence;
    video_vae_ops::LaunchRmsNorm(hidden.As<float>(), weight.norm1.As<float>(),
                                 norm.As<float>(), sequence, kHidden,
                                 kNormEpsilon, stream);
    if (!CheckLaunch("H3-Image VAE norm1", error) ||
        !blas.Linear(weight.qkv_weight.As<float>(), norm.As<float>(),
                     qkv.As<float>(), sequence, kAttentionWidth * 3U, kHidden,
                     stream, error)) {
      return false;
    }
    // Adds the QKV bias, RMS-normalizes q/k per head and rotates the first
    // 2 * kRopeHalf channels (generic in heads, head width and rope width).
    video_vae_ops::LaunchVideoQkvRope(
        qkv.As<float>(), weight.qkv_bias.As<float>(), rope_cos.As<float>(),
        rope_sin.As<float>(), query.As<float>(), key.As<float>(),
        value.As<float>(), sequence, kHeads, kHeadDimension, kRopeHalf,
        kNormEpsilon, stream);
    if (!CheckLaunch("H3-Image VAE QKV RoPE", error) ||
        !blas.Scores(query.As<float>(), key.As<float>(),
                     attention_scores.As<float>(), sequence,
                     1.0F / std::sqrt(static_cast<float>(kHeadDimension)),
                     stream, error)) {
      return false;
    }
    video_vae_ops::LaunchSoftmaxScoresInPlace(attention_scores.As<float>(),
                                              sequence, kHeads, stream);
    if (!CheckLaunch("H3-Image VAE softmax", error) ||
        !blas.Values(value.As<float>(), attention_scores.As<float>(),
                     attention_heads.As<float>(), sequence, stream, error) ||
        !blas.Linear(weight.output_weight.As<float>(),
                     attention_heads.As<float>(), branch.As<float>(), sequence,
                     kHidden, kAttentionWidth, stream, error)) {
      return false;
    }
    video_vae_ops::LaunchScaleAdd(hidden.As<float>(), branch.As<float>(),
                                  weight.output_bias.As<float>(),
                                  weight.scale1.As<float>(), sequence, kHidden,
                                  stream);
    video_vae_ops::LaunchRmsNorm(hidden.As<float>(), weight.norm2.As<float>(),
                                 norm.As<float>(), sequence, kHidden,
                                 kNormEpsilon, stream);
    if (!CheckLaunch("H3-Image VAE attention residual", error) ||
        !blas.Linear(weight.fc1_weight.As<float>(), norm.As<float>(),
                     fc1.As<float>(), sequence, kFeedForward * 2U, kHidden,
                     stream, error)) {
      return false;
    }
    video_vae_ops::LaunchSwiGlu(fc1.As<float>(), weight.fc1_bias.As<float>(),
                                activated.As<float>(), sequence, kFeedForward,
                                stream);
    if (!CheckLaunch("H3-Image VAE SwiGLU", error) ||
        !blas.Linear(weight.fc2_weight.As<float>(), activated.As<float>(),
                     branch.As<float>(), sequence, kHidden, kFeedForward,
                     stream, error)) {
      return false;
    }
    video_vae_ops::LaunchScaleAdd(hidden.As<float>(), branch.As<float>(),
                                  weight.fc2_bias.As<float>(),
                                  weight.scale2.As<float>(), sequence, kHidden,
                                  stream);
    return CheckLaunch("H3-Image VAE MLP residual", error);
  }

  bool RunTile(int tile_y, int tile_x, const CancellationToken* cancellation,
               std::vector<float>* output, std::string* error) {
    if (IsCancelled(cancellation)) {
      SetError(error, "MiniMax-H3-Image VAE cancelled before tile");
      return false;
    }
    const auto begin = Clock::now();
    // The 2D latent is a single-frame video latent for the shared kernel.
    video_vae_ops::LaunchPrepareTile(
        normalized_latent.As<float>(), latent_mean.As<float>(),
        latent_std.As<float>(), latent_rows.As<float>(), 1,
        static_cast<std::uint32_t>(plan.latent_height),
        static_cast<std::uint32_t>(plan.latent_width), 0,
        static_cast<std::uint32_t>(
            plan.y_axis.starts[static_cast<std::size_t>(tile_y)] /
            kH3ImageVaeSpatialRatio),
        static_cast<std::uint32_t>(
            plan.x_axis.starts[static_cast<std::size_t>(tile_x)] /
            kH3ImageVaeSpatialRatio),
        1, plan.tile_latent_height, plan.tile_latent_width, kChannels, stream);
    if (!CheckLaunch("prepare H3-Image VAE tile", error) ||
        !Linear(latent_rows, post_weight, post_bias, &post, plan.patches,
                kChannels, kChannels, error) ||
        !Linear(post, embed_weight, embed_bias, &hidden, plan.patches,
                kChannels, kHidden, error) ||
        !Copy(hidden.As<float>() + static_cast<std::size_t>(plan.patches) *
                                       kHidden,
              registers.data(), registers.bytes(), hipMemcpyDeviceToDevice,
              "pack H3-Image VAE registers", error)) {
      return false;
    }
    hipError_t status = hipMemsetAsync(
        hidden.As<float>() +
            static_cast<std::size_t>(plan.patches + kRegisters) * kHidden,
        0, static_cast<std::size_t>(kHidden) * sizeof(float), stream);
    if (status != hipSuccess) {
      SetError(error, HipError("pack H3-Image VAE zero token", status));
      return false;
    }
    for (std::size_t block = 0; block < blocks.size(); ++block) {
      if (IsCancelled(cancellation)) {
        SetError(error, "MiniMax-H3-Image VAE cancelled before block " +
                            std::to_string(block));
        return false;
      }
      if (!RunBlock(blocks[block], error)) {
        return false;
      }
      base_telemetry.dispatches += 9;
    }
    // Only the patch rows reach the image.
    video_vae_ops::LaunchLayerNorm(
        hidden.As<float>(), final_norm_weight.As<float>(),
        final_norm_bias.As<float>(), norm.As<float>(), plan.patches, kHidden,
        kNormEpsilon, stream);
    if (!CheckLaunch("H3-Image VAE output norm", error) ||
        !Linear(norm, output_weight, output_bias, &projected, plan.patches,
                kHidden, kOutputPatch, error) ||
        !Copy(projected_host.data(), projected.data(), projected.bytes(),
              hipMemcpyDeviceToHost, "read H3-Image VAE tile", error)) {
      return false;
    }
    status = hipStreamSynchronize(stream);
    if (status != hipSuccess) {
      SetError(error, HipError("synchronize H3-Image VAE tile", status));
      return false;
    }
    output->assign(static_cast<std::size_t>(plan.y_axis.length) *
                       plan.x_axis.length * 3U,
                   0.0F);
    if (!UnpackImageVaePatches(projected_host, plan.tile_latent_height,
                               plan.tile_latent_width, *output, error)) {
      return false;
    }
    base_telemetry.dispatches += 6;
    base_telemetry.tile_ms.push_back(Milliseconds(begin, Clock::now()));
    return true;
  }
};

ImageVaeDecoder::ImageVaeDecoder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ImageVaeDecoder::~ImageVaeDecoder() = default;
ImageVaeDecoder::ImageVaeDecoder(ImageVaeDecoder&&) noexcept = default;
ImageVaeDecoder& ImageVaeDecoder::operator=(ImageVaeDecoder&&) noexcept =
    default;

std::unique_ptr<ImageVaeDecoder> ImageVaeDecoder::Create(
    const ModelInventory& inventory, const ImageVaeOptions& options,
    const CancellationToken* cancellation, ImageVaeTelemetry* telemetry,
    std::string* error) {
  if (IsCancelled(cancellation)) {
    SetError(error, "MiniMax-H3-Image VAE creation cancelled");
    return nullptr;
  }
  auto plan =
      ResolveImageVaePlan(options.latent_height, options.latent_width, error);
  if (!plan) {
    return nullptr;
  }
  auto normalization = ReadImageVaeConfig(
      inventory.model_root() / "vae" / "config.json", error);
  if (!normalization) {
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->inventory = &inventory;
  impl->plan = std::move(*plan);
  if (!impl->blas.valid()) {
    SetError(error, "cannot create the MiniMax-H3-Image VAE rocBLAS handle");
    return nullptr;
  }
  hipError_t status =
      hipStreamCreateWithFlags(&impl->stream, hipStreamNonBlocking);
  if (status == hipSuccess) {
    status = hipHostMalloc(&impl->staging, kWeightStagingBytes + kSectorSlack,
                           hipHostMallocDefault);
  }
  if (status != hipSuccess) {
    SetError(error, HipError("create MiniMax-H3-Image VAE resources", status));
    return nullptr;
  }
  if (!impl->LoadWeights(cancellation, error) ||
      !impl->AllocateRuntime(*normalization, error)) {
    return nullptr;
  }
  (void)hipHostFree(impl->staging);
  impl->staging = nullptr;
  if (telemetry != nullptr) {
    *telemetry = impl->base_telemetry;
  }
  return std::unique_ptr<ImageVaeDecoder>(new ImageVaeDecoder(std::move(impl)));
}

bool ImageVaeDecoder::Decode(std::span<const float> normalized_latent,
                             const CancellationToken* cancellation,
                             ImageVaeProgress progress, void* progress_opaque,
                             DecodedImage* output, ImageVaeTelemetry* telemetry,
                             std::string* error) {
  if (output != nullptr) {
    *output = {};
  }
  if (impl_ == nullptr || output == nullptr ||
      normalized_latent.size_bytes() != impl_->normalized_latent.bytes()) {
    SetError(error, "invalid MiniMax-H3-Image VAE decode arguments");
    return false;
  }
  const auto begin = Clock::now();
  if (!impl_->Copy(impl_->normalized_latent.data(), normalized_latent.data(),
                   normalized_latent.size_bytes(), hipMemcpyHostToDevice,
                   "copy H3-Image normalized latent", error)) {
    return false;
  }
  const ImageVaePlan& plan = impl_->plan;
  const int x_count = static_cast<int>(plan.x_axis.starts.size());
  const int total = static_cast<int>(plan.y_axis.starts.size()) * x_count;
  std::vector<std::vector<float>> tiles(static_cast<std::size_t>(total));
  int completed = 0;
  for (int tile_y = 0; tile_y < static_cast<int>(plan.y_axis.starts.size());
       ++tile_y) {
    for (int tile_x = 0; tile_x < x_count; ++tile_x) {
      if (!impl_->RunTile(
              tile_y, tile_x, cancellation,
              &tiles[static_cast<std::size_t>(tile_y * x_count + tile_x)],
              error)) {
        return false;
      }
      ++completed;
      if (progress != nullptr) {
        progress(completed, total, progress_opaque);
      }
    }
  }
  if (!StitchVideoVaeTiles(plan.y_axis, plan.x_axis, 1, tiles,
                           &output->normalized_rgb, error)) {
    return false;
  }
  for (std::size_t index = 0; index < output->normalized_rgb.size(); ++index) {
    if (!std::isfinite(output->normalized_rgb[index])) {
      impl_->base_telemetry.first_non_finite =
          "image_rgb[" + std::to_string(index) + "]";
      SetError(error, "MiniMax-H3-Image VAE produced non-finite RGB");
      return false;
    }
  }
  output->height = plan.height;
  output->width = plan.width;
  impl_->base_telemetry.decode_ms = Milliseconds(begin, Clock::now());
  if (telemetry != nullptr) {
    *telemetry = impl_->base_telemetry;
  }
  return true;
}

const ImageVaePlan& ImageVaeDecoder::plan() const noexcept {
  static const ImageVaePlan empty;
  return impl_ == nullptr ? empty : impl_->plan;
}

}  // namespace gufo::minimax_h3

#endif  // ENGINE_ENABLE_HIP
