#ifndef GUFO_MODELS_MINIMAX_H3_RUNTIME_HPP_
#define GUFO_MODELS_MINIMAX_H3_RUNTIME_HPP_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <gufo/ext/platform/mapped_file.hpp>

namespace gufo::minimax_h3 {

enum class DType : std::uint8_t {
  kBFloat16,
  kFloat32,
  kInt32,
};

[[nodiscard]] std::string_view ToString(DType dtype) noexcept;
[[nodiscard]] std::size_t DTypeSize(DType dtype) noexcept;

enum class Phase : std::uint8_t {
  kPromptEncoder,
  kAdaLnPrecompute,
  kDitCore,
  kVisualVae,
  kAudioVae,
};

[[nodiscard]] std::string_view ToString(Phase phase) noexcept;

enum class ResidencyMode : std::uint8_t {
  kDeviceCopy,
  kMappedReadOnly,
};

[[nodiscard]] std::string_view ToString(ResidencyMode mode) noexcept;

struct TensorDescriptor {
  std::string component;
  std::string name;
  DType dtype{DType::kBFloat16};
  std::vector<std::uint64_t> shape;
  std::filesystem::path shard;
  std::uint64_t file_offset{0};
  std::uint64_t payload_bytes{0};
  Phase phase{Phase::kDitCore};
};

struct ShardDescriptor {
  std::filesystem::path relative_path;
  std::uint64_t file_bytes{0};
  std::uint64_t payload_offset{0};
  std::size_t tensor_count{0};
};

struct InspectionTelemetry {
  std::uint64_t json_bytes_read{0};
  std::uint64_t safetensors_header_bytes_read{0};
  std::uint64_t payload_bytes_read{0};
  std::uint64_t mapped_bytes{0};
  std::uint64_t device_bytes{0};
};

struct InspectionOptions {
  bool require_pinned_manifest{true};
};

class ModelInventory {
public:
  ModelInventory() = default;

  [[nodiscard]] static std::optional<ModelInventory> Inspect(
      const std::filesystem::path& model_root,
      const std::filesystem::path& source_manifest,
      std::string* error = nullptr);
  [[nodiscard]] static std::optional<ModelInventory> Inspect(
      const std::filesystem::path& model_root,
      const std::filesystem::path& source_manifest,
      const InspectionOptions& options, std::string* error = nullptr);

  /// A MiniMax H3 Fun-ControlNet-Union checkpoint (alibaba-pai, VideoX-Fun
  /// layout: control_proj_in plus control_blocks.<i>), read from its
  /// safetensors header alone. Its tensors are "transformer" tensors named
  /// as in the file; the control blocks' adaln_proj land in the AdaLN
  /// precompute phase like the base blocks'.
  [[nodiscard]] static std::optional<ModelInventory> InspectControlPatch(
      const std::filesystem::path& file, std::string* error = nullptr);

  /// This (pinned FL2VA) inventory with the transformer read from another
  /// partition of the same model root, e.g. "Ref2VA": its
  /// <partition>/transformer index and safetensors headers must name
  /// exactly the FL2VA transformer's tensors with the same dtypes and
  /// shapes (the released partitions share one architecture). The text
  /// encoder and the VAEs stay FL2VA's.
  [[nodiscard]] std::optional<ModelInventory> WithTransformerPartition(
      std::string_view partition, std::string* error = nullptr) const;

  /// MiniMax-H3-Image release (`MiniMaxAI/MiniMax-H3-Image`, partition
  /// "image"): `transformer/` (indexed BF16/F32 shards, the H3 DiT without
  /// its audio projections), `pdd_fused_heads.safetensors` (component "pdd",
  /// tensors renamed to `pdd.video_out.weight` / `pdd.video_out.bias`) and
  /// `vae/diffusion_pytorch_model.safetensors` (component "image_vae").
  /// The Qwen3-VL text encoder is bit-identical to the H3 FL2VA one and is
  /// read from `text_encoder_dir` (empty: no prompt-encoder tensors). The
  /// image release has no pinned gufo manifest: every safetensors header is
  /// checked against its index and against the tensor counts of the
  /// published release instead.
  [[nodiscard]] static std::optional<ModelInventory> InspectImage(
      const std::filesystem::path& image_root,
      const std::filesystem::path& text_encoder_dir,
      std::string* error = nullptr);

  [[nodiscard]] const std::filesystem::path& model_root() const noexcept {
    return model_root_;
  }
  [[nodiscard]] std::span<const TensorDescriptor> tensors() const noexcept {
    return tensors_;
  }
  [[nodiscard]] std::span<const ShardDescriptor> shards() const noexcept {
    return shards_;
  }
  [[nodiscard]] const InspectionTelemetry& telemetry() const noexcept {
    return telemetry_;
  }
  [[nodiscard]] const TensorDescriptor* FindTensor(
      std::string_view name) const noexcept;
  [[nodiscard]] std::vector<const TensorDescriptor*> TensorsForPhase(
      Phase phase) const;
  [[nodiscard]] std::uint64_t TensorBytesForPhase(Phase phase) const noexcept;

private:
  std::filesystem::path model_root_;
  std::vector<TensorDescriptor> tensors_;
  std::vector<ShardDescriptor> shards_;
  std::unordered_map<std::string, std::size_t> tensor_index_;
  InspectionTelemetry telemetry_;
};

/// Reads one tensor into host memory as F32 (BF16 widened). For small CPU
/// paths (reference decoders, config tensors), not for the large phases.
[[nodiscard]] bool ReadTensorF32(const ModelInventory& inventory,
                                 const TensorDescriptor& descriptor,
                                 std::vector<float>* output,
                                 std::string* error = nullptr);

// Tensor counts of the MiniMax-H3-Image release (revision 8ced6006).
inline constexpr std::size_t kH3ImageTransformerTensors = 531;
inline constexpr std::size_t kH3ImagePddTensors = 2;

class CancellationToken {
public:
  void Cancel() noexcept { cancelled_.store(true, std::memory_order_release); }
  [[nodiscard]] bool IsCancelled() const noexcept {
    return cancelled_.load(std::memory_order_acquire);
  }

private:
  std::atomic<bool> cancelled_{false};
};

class FailureInjector {
public:
  explicit FailureInjector(std::optional<std::size_t> fail_after = std::nullopt)
      : fail_after_(fail_after) {}

  [[nodiscard]] bool ShouldFail(std::string_view operation);
  [[nodiscard]] std::size_t operations() const noexcept { return operations_; }
  [[nodiscard]] std::string_view last_operation() const noexcept {
    return last_operation_;
  }

private:
  std::optional<std::size_t> fail_after_;
  std::size_t operations_{0};
  std::string last_operation_;
};

class ResidencyBackend {
public:
  virtual ~ResidencyBackend() = default;

  ResidencyBackend(const ResidencyBackend&) = delete;
  ResidencyBackend& operator=(const ResidencyBackend&) = delete;
  ResidencyBackend(ResidencyBackend&&) = delete;
  ResidencyBackend& operator=(ResidencyBackend&&) = delete;

  [[nodiscard]] virtual bool IsSupportedTarget(std::string* error) = 0;
  [[nodiscard]] virtual bool CreateStream(void** stream,
                                          std::string* error) = 0;
  virtual void DestroyStream(void* stream) noexcept = 0;
  [[nodiscard]] virtual bool Allocate(std::size_t bytes, void** device,
                                      std::string* error) = 0;
  virtual void Free(void* device) noexcept = 0;
  [[nodiscard]] virtual bool RegisterReadOnlyMapped(void* host,
                                                    std::size_t bytes,
                                                    void** device_alias,
                                                    std::string* error) = 0;
  virtual void Unregister(void* host) noexcept = 0;
  [[nodiscard]] virtual bool CopyToDevice(void* device, const void* host,
                                          std::size_t bytes, void* stream,
                                          std::string* error) = 0;
  [[nodiscard]] virtual bool Synchronize(void* stream, std::string* error) = 0;

protected:
  ResidencyBackend() = default;

private:
  friend class PhaseSession;
  [[nodiscard]] bool TryAcquirePhase() noexcept {
    bool expected = false;
    return phase_active_.compare_exchange_strong(expected, true,
                                                 std::memory_order_acq_rel);
  }
  void ReleasePhase() noexcept {
    phase_active_.store(false, std::memory_order_release);
  }

  std::atomic<bool> phase_active_{false};
};

[[nodiscard]] std::unique_ptr<ResidencyBackend> CreateHipResidencyBackend(
    std::string* error = nullptr);

// GUFO_H3_TRACE=1: prints the stage and the device's free memory to stderr.
void TraceStage(const char* stage);
[[nodiscard]] constexpr ResidencyMode DefaultResidencyMode() noexcept {
  return ResidencyMode::kDeviceCopy;
}

struct LoadOptions {
  ResidencyMode mode{ResidencyMode::kDeviceCopy};
  std::size_t persistent_bytes{0};
  std::size_t scratch_bytes{0};
  bool prefault{true};
};

struct MemoryTelemetry {
  std::uint64_t file_backed_bytes{0};
  std::uint64_t registered_host_bytes{0};
  std::uint64_t device_weight_bytes{0};
  std::uint64_t persistent_bytes{0};
  std::uint64_t scratch_bytes{0};
  std::uint64_t current_live_bytes{0};
  std::uint64_t peak_live_bytes{0};
  std::uint64_t prefaulted_bytes{0};
};

struct ResidentTensor {
  const TensorDescriptor* descriptor{nullptr};
  const void* device_data{nullptr};
  std::size_t bytes{0};
};

class PhaseSession {
public:
  ~PhaseSession();

  PhaseSession(const PhaseSession&) = delete;
  PhaseSession& operator=(const PhaseSession&) = delete;
  PhaseSession(PhaseSession&& other) noexcept;
  PhaseSession& operator=(PhaseSession&& other) noexcept;

  [[nodiscard]] static std::optional<PhaseSession> Load(
      const ModelInventory& inventory, Phase phase, const LoadOptions& options,
      ResidencyBackend& backend, const CancellationToken* cancellation,
      FailureInjector* failures, std::string* error = nullptr);

  [[nodiscard]] Phase phase() const noexcept { return phase_; }
  [[nodiscard]] ResidencyMode mode() const noexcept { return mode_; }
  [[nodiscard]] std::span<const ResidentTensor> tensors() const noexcept {
    return tensors_;
  }
  [[nodiscard]] const ResidentTensor* FindTensor(
      std::string_view name) const noexcept;
  [[nodiscard]] const MemoryTelemetry& telemetry() const noexcept {
    return telemetry_;
  }
  [[nodiscard]] void* stream() const noexcept { return stream_; }
  [[nodiscard]] void* persistent_data() const noexcept { return persistent_; }
  [[nodiscard]] void* scratch_data() const noexcept { return scratch_; }

  void BeginTimedExecution() noexcept { timed_execution_ = true; }
  [[nodiscard]] bool timed_execution() const noexcept {
    return timed_execution_;
  }

private:
  struct Mapping {
    std::filesystem::path shard;
    platform::RawMappedFile mapping_file;
    void* host{nullptr};
    void* device_alias{nullptr};
    std::size_t bytes{0};
    std::uint64_t file_offset{0};
    bool registered{false};
  };

  explicit PhaseSession(ResidencyBackend& backend) : backend_(&backend) {}
  void Release() noexcept;

  ResidencyBackend* backend_{nullptr};
  Phase phase_{Phase::kPromptEncoder};
  ResidencyMode mode_{ResidencyMode::kDeviceCopy};
  void* stream_{nullptr};
  void* device_weights_{nullptr};
  void* persistent_{nullptr};
  void* scratch_{nullptr};
  std::vector<Mapping> mappings_;
  std::vector<ResidentTensor> tensors_;
  MemoryTelemetry telemetry_;
  bool timed_execution_{false};
  bool phase_lease_{false};
};

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_RUNTIME_HPP_
