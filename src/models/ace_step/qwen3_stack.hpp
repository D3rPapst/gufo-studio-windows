// Qwen3 transformer stack shared by the ACE-Step models (model.hip) and
// the YuE2 port (src/models/yue2): pre-norm layers with QK-RMSNorm, RoPE
// theta 1e6 and SwiGLU, BF16 weights and GEMM operands, FP32 residual stream.
// Moved unchanged from model.hip. Written with AI assistance (Claude Code).
#ifndef GUFO_MODELS_ACE_STEP_QWEN3_STACK_HPP_
#define GUFO_MODELS_ACE_STEP_QWEN3_STACK_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "src/models/ace_step/attention.hip.hpp"
#include "src/models/ace_step/gemm.hpp"
#include "src/models/ace_step/kernels.hpp"
#include "src/models/ace_step/weights.hpp"

namespace gufo::models::ace_step::qwen3 {

inline constexpr int kHeadDim = 128;
inline constexpr int kWindow = 128;

struct Shape {
  int width{0};
  int heads{0};
  int kv_heads{0};
  int ffn{0};
  [[nodiscard]] int q_dim() const { return heads * kHeadDim; }
  [[nodiscard]] int kv_dim() const { return kv_heads * kHeadDim; }
};

struct Attention {
  DeviceBuffer qkv;  // self: q | k | v rows [q_dim + 2 kv_dim, width]
  DeviceBuffer q;    // cross: [q_dim, width]
  DeviceBuffer kv;   // cross: k | v rows [2 kv_dim, width]
  DeviceBuffer q_norm, k_norm, out;
};

struct Mlp {
  DeviceBuffer gate_up;  // gate | up rows [2 ffn, width]
  DeviceBuffer down;     // [width, ffn]
};

struct EncoderLayer {
  DeviceBuffer input_norm, post_norm;
  Attention attn;
  Mlp mlp;
};

// Stacks the rows of `parts` (each [rows_i, columns]) into one BF16 matrix.
inline bool LoadStacked(SafetensorsFile& file, DeviceBuffer& buffer,
                 std::initializer_list<std::pair<std::string, int>> parts,
                 int columns, std::string* error) {
  std::size_t rows = 0;
  for (const auto& part : parts)
    rows += part.second;
  if (!buffer.Allocate(rows * columns * 2, error))
    return false;
  std::size_t row = 0;
  for (const auto& [name, part_rows] : parts) {
    if (!file.Upload(name, static_cast<std::size_t>(part_rows) * columns,
                     buffer.bf16() + row * columns, true, error))
      return false;
    row += part_rows;
  }
  return true;
}

inline bool LoadAttention(SafetensorsFile& file, const std::string& p,
                          const Shape& s, bool cross, Attention& a,
                          std::string* error) {
  const int w = s.width;
  if (cross) {
    if (!file.Load(p + "q_proj.weight", std::size_t(s.q_dim()) * w, a.q, true,
                   error) ||
        !LoadStacked(file, a.kv,
                     {{p + "k_proj.weight", s.kv_dim()},
                      {p + "v_proj.weight", s.kv_dim()}},
                     w, error))
      return false;
  } else if (!LoadStacked(file, a.qkv,
                          {{p + "q_proj.weight", s.q_dim()},
                           {p + "k_proj.weight", s.kv_dim()},
                           {p + "v_proj.weight", s.kv_dim()}},
                          w, error)) {
    return false;
  }
  return file.Load(p + "q_norm.weight", kHeadDim, a.q_norm, true, error) &&
         file.Load(p + "k_norm.weight", kHeadDim, a.k_norm, true, error) &&
         file.Load(p + "o_proj.weight", std::size_t(w) * s.q_dim(), a.out, true,
                   error);
}

inline bool LoadMlp(SafetensorsFile& file, const std::string& p, const Shape& s,
             Mlp& m, std::string* error) {
  return LoadStacked(
             file, m.gate_up,
             {{p + "gate_proj.weight", s.ffn}, {p + "up_proj.weight", s.ffn}},
             s.width, error) &&
         file.Load(p + "down_proj.weight", std::size_t(s.width) * s.ffn, m.down,
                   true, error);
}

inline bool LoadEncoderLayers(SafetensorsFile& file, const std::string& prefix,
                       int count, const Shape& s,
                       std::vector<EncoderLayer>& layers, std::string* error) {
  layers.resize(count);
  for (int i = 0; i < count; ++i) {
    const std::string p = prefix + "layers." + std::to_string(i) + ".";
    EncoderLayer& l = layers[i];
    if (!file.Load(p + "input_layernorm.weight", s.width, l.input_norm, true,
                   error) ||
        !file.Load(p + "post_attention_layernorm.weight", s.width, l.post_norm,
                   true, error) ||
        !LoadAttention(file, p + "self_attn.", s, false, l.attn, error) ||
        !LoadMlp(file, p + "mlp.", s, l.mlp, error))
      return false;
  }
  return true;
}

// Scratch buffers and the primitive sequence shared by every stack.
struct Engine {
  hipStream_t stream{nullptr};
  LinearGemm gemm;
  DeviceBuffer normed, qkv, query, key, value, packed, attn, proj, ffn, act;
  DeviceBuffer ids;

  ~Engine() {
    if (stream != nullptr) {
      (void)hipStreamSynchronize(stream);
      (void)hipStreamDestroy(stream);
    }
  }

  bool Init(std::string* error) {
    return Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
                 "stream", error) &&
           gemm.Init(stream, error);
  }

  bool Reserve(int rows, const Shape& s, std::string* error) {
    const std::size_t r = rows, w = s.width, q = s.q_dim(), kv = s.kv_dim();
    const std::size_t f = s.ffn;
    return normed.Reserve(r * w * 2, error) &&
           qkv.Reserve(r * (q + 2 * kv) * 2, error) &&
           query.Reserve(r * q * 2, error) && key.Reserve(r * kv * 2, error) &&
           value.Reserve(r * kv * 2, error) &&
           packed.Reserve(std::size_t(PaddedRows(rows)) * kv * 2, error) &&
           attn.Reserve(r * q * 2, error) && proj.Reserve(r * w * 2, error) &&
           ffn.Reserve(r * 2 * f * 2, error) && act.Reserve(r * f * 2, error);
  }

  bool Gemm(const DeviceBuffer& weight, const DeviceBuffer* bias,
            const std::uint16_t* in, std::uint16_t* out, int tokens, int rows,
            int columns, std::string* error) {
    return gemm.Run(weight.bf16(), bias != nullptr ? bias->bf16() : nullptr, in,
                    out, tokens, rows, columns, error);
  }

  // proj = o_proj(attention(normed)) for self-attention. With mask.group,
  // rows form independent sequences of that length (RoPE restarts).
  bool SelfAttention(const Attention& a, const Shape& s, int rows,
                     AttentionMask mask, std::string* error) {
    const int stride = s.q_dim() + 2 * s.kv_dim();
    if (!Gemm(a.qkv, nullptr, normed.bf16(), qkv.bf16(), rows, stride, s.width,
              error))
      return false;
    const int period = static_cast<int>(mask.group);
    LaunchHeads(qkv.bf16(), stride, 0, rows, s.heads, a.q_norm.bf16(), true,
                query.bf16(), stream, nullptr, period);
    LaunchHeads(qkv.bf16(), stride, s.q_dim(), rows, s.kv_heads,
                a.k_norm.bf16(), true, key.bf16(), stream, nullptr, period);
    LaunchHeads(qkv.bf16(), stride, s.q_dim() + s.kv_dim(), rows, s.kv_heads,
                nullptr, false, value.bf16(), stream);
    LaunchPackValues(value.bf16(), packed.bf16(), rows, s.kv_heads, stream);
    LaunchAttention(query.bf16(), key.bf16(), packed.bf16(), attn.bf16(), rows,
                    rows, s.heads, s.kv_heads, mask, stream);
    return Gemm(a.out, nullptr, attn.bf16(), proj.bf16(), rows, s.width,
                s.q_dim(), error);
  }

  // proj = down(silu(gate(normed)) * up(normed)).
  bool FeedForward(const Mlp& m, const Shape& s, int rows, std::string* error) {
    if (!Gemm(m.gate_up, nullptr, normed.bf16(), ffn.bf16(), rows, 2 * s.ffn,
              s.width, error))
      return false;
    LaunchSwiGlu(ffn.bf16(), rows, s.ffn, act.bf16(), stream);
    return Gemm(m.down, nullptr, act.bf16(), proj.bf16(), rows, s.width, s.ffn,
                error);
  }

  // Pre-norm layers on the FP32 residual `hidden` [rows, width]; even layers
  // use the +-128 window when `windowed`.
  bool EncoderLayers(const std::vector<EncoderLayer>& layers, const Shape& s,
                     float* hidden, int rows, bool windowed, bool causal,
                     std::string* error, int group = 0) {
    if (!Reserve(rows, s, error))
      return false;
    for (std::size_t i = 0; i < layers.size(); ++i) {
      const EncoderLayer& l = layers[i];
      AttentionMask mask;
      mask.causal = causal;
      mask.group = static_cast<std::uint32_t>(group);
      mask.window = windowed && i % 2 == 0 ? kWindow : 0;
      LaunchRmsNorm(hidden, l.input_norm.bf16(), nullptr, nullptr, rows,
                    s.width, normed.bf16(), nullptr, stream);
      if (!SelfAttention(l.attn, s, rows, mask, error))
        return false;
      LaunchResidual(hidden, proj.bf16(), nullptr, rows, s.width, stream);
      LaunchRmsNorm(hidden, l.post_norm.bf16(), nullptr, nullptr, rows, s.width,
                    normed.bf16(), nullptr, stream);
      if (!FeedForward(l.mlp, s, rows, error))
        return false;
      LaunchResidual(hidden, proj.bf16(), nullptr, rows, s.width, stream);
    }
    return true;
  }

  // Synchronous: callers pass short-lived host vectors.
  bool Upload(DeviceBuffer& buffer, const void* data, std::size_t bytes,
              std::string* error) {
    return buffer.Reserve(bytes, error) &&
           Check(hipMemcpyAsync(buffer.get(), data, bytes,
                                hipMemcpyHostToDevice, stream),
                 "upload", error) &&
           Check(hipStreamSynchronize(stream), "upload", error);
  }

  bool Download(const void* device, std::size_t bytes, void* host,
                std::string* error) {
    return Check(hipMemcpyAsync(host, device, bytes, hipMemcpyDeviceToHost,
                                stream),
                 "download", error) &&
           Check(hipStreamSynchronize(stream), "download", error);
  }

  // Uploads FP32 rows and converts them to BF16 in `target`.
  bool UploadBf16(std::span<const float> data, DeviceBuffer& staging,
                  DeviceBuffer& target, std::string* error) {
    return Upload(staging, data.data(), data.size() * 4, error) &&
           target.Reserve(data.size() * 2, error) &&
           (LaunchToBf16(staging.f32(), data.size(), target.bf16(), stream),
            true);
  }
};

}  // namespace gufo::models::ace_step::qwen3

#endif  // GUFO_MODELS_ACE_STEP_QWEN3_STACK_HPP_
