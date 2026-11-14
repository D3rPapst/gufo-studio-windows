// YuE2-3B on HIP (gfx1151): the AR and NAR weight sets of every layer run
// through the ACE-Step Qwen3 stack (ace_step/qwen3_stack.hpp: BF16 weights
// and GEMMs, FP32 residual stream, the same attention kernel and KV cache
// layout as the ACE-Step 5 Hz LM). YuE2-specific parts:
//   - logits of a phase slice as an FP32 matrix-vector product (the
//     reference samples from FP32 logits),
//   - the NAR input vae2llm(x) + time_embedder(t) + latent_pos_embed and the
//     llm2vae head in FP32,
//   - NAR attention: the frames see the AR keys / values of the chunk's
//     prefix and each other (no mask), at RoPE positions after the prefix.
// Written with AI assistance (Claude Code).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <string>
#include <utility>

#include "src/models/ace_step/qwen3_stack.hpp"
#include "src/models/yue2/model.hpp"

namespace gufo::models::yue2 {
namespace {

using namespace ace_step;         // NOLINT: DeviceBuffer, Launch*, Fail
using namespace ace_step::qwen3;  // NOLINT: Engine, EncoderLayer, Shape

constexpr int kThreads = 256;
constexpr int kFrequencies = 256;

__device__ __forceinline__ float Bf16(std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ float WaveSum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    value += __shfl_down(value, offset);
  return value;
}

// out[r] = W[r, :] . x for `count` rows (one wave per row), FP32.
__global__ void LogitsKernel(const std::uint16_t* weight, const float* x,
                             int width, int count, float* out) {
  extern __shared__ float xs[];
  for (int c = threadIdx.x; c < width; c += blockDim.x)
    xs[c] = x[c];
  __syncthreads();
  const int waves = blockDim.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  const int row = blockIdx.x * waves + threadIdx.x / warpSize;
  if (row >= count)
    return;
  const std::uint16_t* w = weight + static_cast<std::size_t>(row) * width;
  float sum = 0;
  for (int c = lane * 8; c < width; c += warpSize * 8) {
    const uint4 packed = *reinterpret_cast<const uint4*>(w + c);
    const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      sum += __uint_as_float(words[k] << 16U) * xs[c + 2 * k];
      sum += __uint_as_float(words[k] & 0xFFFF0000U) * xs[c + 2 * k + 1];
    }
  }
  sum = WaveSum(sum);
  if (lane == 0)
    out[row] = sum;
}

// hidden[r, c] = (vae2llm_b[c] + sum_k x[r, k] W[c, k]) + time[c] + pe[r, c]
// for the NAR rows (LATENT_START, frames, LATENT_END), FP32.
__global__ void NarInputKernel(const float* x, const std::uint16_t* weight,
                               const std::uint16_t* bias, const float* time,
                               const std::uint16_t* pe, int width,
                               std::size_t count, float* hidden) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const int c = static_cast<int>(i % width);
  const std::size_t r = i / width;
  const float* row = x + r * kLatentChannels;
  const std::uint16_t* w = weight + static_cast<std::size_t>(c) *
                                        kLatentChannels;
  float sum = 0;
#pragma unroll 8
  for (int k = 0; k < kLatentChannels; ++k)
    sum += row[k] * Bf16(w[k]);
  hidden[i] = (sum + Bf16(bias[c])) + time[c] + Bf16(pe[i]);
}

// out[r, o] = llm2vae_b[o] + normed[r, :] . W[o, :] (64 outputs per row,
// one block per row), FP32.
__global__ void NarOutputKernel(const float* normed,
                                const std::uint16_t* weight,
                                const std::uint16_t* bias, int width,
                                float* out) {
  extern __shared__ float xs[];
  const std::size_t row = blockIdx.x;
  for (int c = threadIdx.x; c < width; c += blockDim.x)
    xs[c] = normed[row * width + c];
  __syncthreads();
  const int waves = blockDim.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  for (int o = threadIdx.x / warpSize; o < kLatentChannels; o += waves) {
    const std::uint16_t* w = weight + static_cast<std::size_t>(o) * width;
    float sum = 0;
    for (int c = lane; c < width; c += warpSize)
      sum += Bf16(w[c]) * xs[c];
    sum = WaveSum(sum);
    if (lane == 0)
      out[row * kLatentChannels + o] = sum + Bf16(bias[o]);
  }
}

unsigned GridFor(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

struct KvCache {
  DeviceBuffer key, value;  // all layers, concatenated
  int capacity{0};
  int length{0};
};

double Since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       start)
      .count();
}

}  // namespace

struct Model::Impl {
  Engine engine;
  Shape shape{2048, 16, 8, 6144};
  int layer_count{0};
  DeviceBuffer embed, head, norm;
  std::vector<EncoderLayer> ar, nar;
  DeviceBuffer vae2llm_w, vae2llm_b, llm2vae_w, llm2vae_b;
  DeviceBuffer time1_w, time1_b, time2_w, time2_b, pe;
  int pe_rows{0};
  DeviceBuffer hidden, positions, normed32, logits;
  DeviceBuffer state, time_in, time_mid, time_emb, velocity;
  KvCache cache[2];
  std::size_t weight_bytes{0};
  int acoustic_frames{0};

  [[nodiscard]] std::size_t LayerKeys(const KvCache& c) const {
    return std::size_t(shape.kv_heads) * c.capacity * kHeadDim;
  }
  [[nodiscard]] std::size_t LayerValues(const KvCache& c) const {
    return std::size_t(shape.kv_heads) * PaddedRows(c.capacity) * kHeadDim;
  }

  bool Allocate(KvCache& c, int capacity, std::string* error) {
    c.capacity = capacity;
    c.length = 0;
    const std::size_t keys = LayerKeys(c) * layer_count * 2;
    const std::size_t values = LayerValues(c) * layer_count * 2;
    return c.key.Reserve(keys, error) && c.value.Reserve(values, error) &&
           Check(hipMemsetAsync(c.key.get(), 0, keys, engine.stream),
                 "memset", error) &&
           Check(hipMemsetAsync(c.value.get(), 0, values, engine.stream),
                 "memset", error);
  }

  bool Reserve(int rows, std::string* error) {
    const std::size_t r = std::max(rows, 2);
    return hidden.Reserve(r * shape.width * 4, error) &&
           positions.Reserve(r * 4, error) &&
           normed32.Reserve(r * shape.width * 4, error) &&
           engine.Reserve(static_cast<int>(r), shape, error);
  }

  // Runs hidden rows [first, first + rows) (RoPE positions in `positions`)
  // of cache `seq` through layer `i` of the weight set `l`. Their keys and
  // values go to cache rows [length, length + rows); they attend to the
  // `length` cached rows and to each other (causally with `causal`).
  bool Layer(const EncoderLayer& l, int i, int seq, int first, int rows,
             bool causal, std::string* error) {
    Engine& e = engine;
    const Shape& s = shape;
    KvCache& c = cache[seq];
    float* x = hidden.f32() + std::size_t(first) * s.width;
    const int* pos = static_cast<const int*>(positions.get()) + first;
    std::uint16_t* key_cache = c.key.bf16() + LayerKeys(c) * i;
    std::uint16_t* value_cache = c.value.bf16() + LayerValues(c) * i;
    const int stride = s.q_dim() + 2 * s.kv_dim();
    LaunchRmsNorm(x, l.input_norm.bf16(), nullptr, nullptr, rows, s.width,
                  e.normed.bf16(), nullptr, e.stream);
    if (!e.Gemm(l.attn.qkv, nullptr, e.normed.bf16(), e.qkv.bf16(), rows,
                stride, s.width, error))
      return false;
    LaunchHeads(e.qkv.bf16(), stride, 0, rows, s.heads, l.attn.q_norm.bf16(),
                true, e.query.bf16(), e.stream, pos);
    LaunchHeads(e.qkv.bf16(), stride, s.q_dim(), rows, s.kv_heads,
                l.attn.k_norm.bf16(), true, e.key.bf16(), e.stream, pos);
    LaunchHeads(e.qkv.bf16(), stride, s.q_dim() + s.kv_dim(), rows, s.kv_heads,
                nullptr, false, e.value.bf16(), e.stream);
    LaunchStoreCache(e.key.bf16(), e.value.bf16(), rows, s.kv_heads, c.length,
                     c.capacity, key_cache, value_cache, e.stream);
    AttentionMask mask;
    mask.causal = causal;
    mask.capacity = static_cast<std::uint32_t>(c.capacity);
    LaunchAttention(e.query.bf16(), key_cache, value_cache, e.attn.bf16(), rows,
                    c.length + rows, s.heads, s.kv_heads, mask, e.stream);
    if (!e.Gemm(l.attn.out, nullptr, e.attn.bf16(), e.proj.bf16(), rows,
                s.width, s.q_dim(), error))
      return false;
    LaunchResidual(x, e.proj.bf16(), nullptr, rows, s.width, e.stream);
    LaunchRmsNorm(x, l.post_norm.bf16(), nullptr, nullptr, rows, s.width,
                  e.normed.bf16(), nullptr, e.stream);
    if (!e.FeedForward(l.mlp, s, rows, error))
      return false;
    LaunchResidual(x, e.proj.bf16(), nullptr, rows, s.width, e.stream);
    return true;
  }

  // Embeds `ids` into hidden rows from `first` with RoPE positions from
  // `start`.
  bool Embed(std::span<const std::uint32_t> ids, int first, int start,
             std::string* error) {
    std::vector<int> host(ids.size()), pos(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] >= kVocabulary)
        return Fail(error, "YuE2 token id out of range");
      host[i] = static_cast<int>(ids[i]);
      pos[i] = start + static_cast<int>(i);
    }
    if (!UploadPositions(pos, first, error) ||
        !engine.Upload(engine.ids, host.data(), host.size() * 4, error))
      return false;
    LaunchGather(embed.bf16(), static_cast<const int*>(engine.ids.get()),
                 static_cast<int>(ids.size()), shape.width,
                 hidden.f32() + std::size_t(first) * shape.width,
                 engine.stream);
    return true;
  }

  bool UploadPositions(const std::vector<int>& pos, int first,
                       std::string* error) {
    return Check(hipMemcpyAsync(static_cast<int*>(positions.get()) + first,
                                pos.data(), pos.size() * 4,
                                hipMemcpyHostToDevice, engine.stream),
                 "upload", error) &&
           Check(hipStreamSynchronize(engine.stream), "upload", error);
  }

  // Prefills `ids` into cache `seq` (from row 0, causal) and leaves the
  // FP32 final-normed last row in normed32 row `seq`.
  bool Prefill(std::span<const std::uint32_t> ids, int seq,
               std::string* error) {
    const int rows = static_cast<int>(ids.size());
    if (!Embed(ids, 0, 0, error))
      return false;
    for (int i = 0; i < layer_count; ++i)
      if (!Layer(ar[i], i, seq, 0, rows, true, error))
        return false;
    cache[seq].length = rows;
    LaunchRmsNorm(hidden.f32() + std::size_t(rows - 1) * shape.width,
                  norm.bf16(), nullptr, nullptr, 1, shape.width, nullptr,
                  normed32.f32() + std::size_t(seq) * shape.width,
                  engine.stream);
    return true;
  }

  // Phase-slice logits of normed32 rows [0, sequences) to host.
  bool Logits(Phase phase, int sequences, std::vector<float>* out,
              std::string* error) {
    const VocabularySlice slice = PhaseSlice(phase);
    if (!logits.Reserve(std::size_t(sequences) * slice.count * 4, error))
      return false;
    const int waves = kThreads / 32;
    for (int q = 0; q < sequences; ++q)
      LogitsKernel<<<(slice.count + waves - 1) / waves, kThreads,
                     shape.width * 4, engine.stream>>>(
          head.bf16() + std::size_t(slice.first) * shape.width,
          normed32.f32() + std::size_t(q) * shape.width, shape.width,
          static_cast<int>(slice.count),
          logits.f32() + std::size_t(q) * slice.count);
    out->resize(std::size_t(sequences) * slice.count);
    return engine.Download(logits.get(), out->size() * 4, out->data(), error);
  }
};

Model::Model(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Model::~Model() = default;

std::size_t Model::weight_bytes() const {
  return impl_->weight_bytes;
}

std::size_t Model::device_bytes() const {
  const Impl& m = *impl_;
  std::size_t bytes = m.weight_bytes;
  for (const KvCache& c : m.cache)
    bytes += c.key.bytes() + c.value.bytes();
  return bytes;
}

std::unique_ptr<Model> Model::Load(const std::filesystem::path& safetensors,
                                   std::string* error) {
  auto impl = std::make_unique<Impl>();
  Impl& m = *impl;
  if (!m.engine.Init(error))
    return nullptr;
  SafetensorsFile file;
  if (!file.Open(safetensors, m.engine.stream, error))
    return nullptr;
  const auto* embed = file.Find("model.embed_tokens.weight");
  const auto* q = file.Find("model.layers.0.self_attn.q_proj.weight");
  const auto* k = file.Find("model.layers.0.self_attn.k_proj.weight");
  const auto* gate = file.Find("model.layers.0.mlp.gate_proj.weight");
  const auto* pe = file.Find("latent_pos_embed.pe");
  if (embed == nullptr || embed->shape.size() != 2 ||
      embed->shape[0] != kVocabulary || q == nullptr || k == nullptr ||
      gate == nullptr || pe == nullptr || pe->shape.size() != 2 ||
      !file.Has("model.layers.0.nar_self_attn.q_proj.weight") ||
      !file.Has("vae2llm.weight") || !file.Has("lm_head.weight")) {
    Fail(error, "not a YuE2-3B model.safetensors: " + safetensors.string());
    return nullptr;
  }
  m.shape.width = static_cast<int>(embed->shape[1]);
  m.shape.heads = static_cast<int>(q->shape[0] / kHeadDim);
  m.shape.kv_heads = static_cast<int>(k->shape[0] / kHeadDim);
  m.shape.ffn = static_cast<int>(gate->shape[0]);
  while (file.Has("model.layers." + std::to_string(m.layer_count) +
                  ".input_layernorm.weight"))
    ++m.layer_count;
  if (m.layer_count == 0 || pe->shape[1] != std::size_t(m.shape.width) ||
      m.shape.width % 256 != 0) {
    Fail(error, "unexpected YuE2 geometry: " + safetensors.string());
    return nullptr;
  }
  const std::size_t w = m.shape.width;
  m.pe_rows = static_cast<int>(pe->shape[0]);
  bool ok =
      file.Load("model.embed_tokens.weight", embed->count(), m.embed, true,
                error) &&
      file.Load("lm_head.weight", embed->count(), m.head, true, error) &&
      file.Load("model.norm.weight", w, m.norm, true, error) &&
      LoadEncoderLayers(file, "model.", m.layer_count, m.shape, m.ar, error);
  m.nar.resize(m.layer_count);
  for (int i = 0; ok && i < m.layer_count; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    EncoderLayer& l = m.nar[i];
    ok = file.Load(p + "nar_input_layernorm.weight", w, l.input_norm, true,
                   error) &&
         file.Load(p + "nar_pre_mlp_layernorm.weight", w, l.post_norm, true,
                   error) &&
         LoadAttention(file, p + "nar_self_attn.", m.shape, false, l.attn,
                       error) &&
         LoadMlp(file, p + "nar_mlp.", m.shape, l.mlp, error);
  }
  ok = ok &&
       file.Load("vae2llm.weight", w * kLatentChannels, m.vae2llm_w, true,
                 error) &&
       file.Load("vae2llm.bias", w, m.vae2llm_b, true, error) &&
       file.Load("llm2vae.weight", w * kLatentChannels, m.llm2vae_w, true,
                 error) &&
       file.Load("llm2vae.bias", kLatentChannels, m.llm2vae_b, true, error) &&
       file.Load("time_embedder.mlp.0.weight", w * kFrequencies, m.time1_w,
                 true, error) &&
       file.Load("time_embedder.mlp.0.bias", w, m.time1_b, true, error) &&
       file.Load("time_embedder.mlp.2.weight", w * w, m.time2_w, true,
                 error) &&
       file.Load("time_embedder.mlp.2.bias", w, m.time2_b, true, error) &&
       file.Load("latent_pos_embed.pe", pe->count(), m.pe, true, error) &&
       file.Finish(error);
  if (!ok)
    return nullptr;
  m.weight_bytes = file.uploaded_bytes();
  return std::unique_ptr<Model>(new Model(std::move(impl)));
}

bool Model::PrefixLogits(std::span<const std::uint32_t> prefix, Phase phase,
                         std::vector<float>* logits, std::string* error) {
  Impl& m = *impl_;
  const int rows = static_cast<int>(prefix.size());
  if (rows == 0)
    return Fail(error, "empty YuE2 prefix");
  return m.Reserve(rows, error) && m.Allocate(m.cache[0], rows, error) &&
         m.Prefill(prefix, 0, error) && m.Logits(phase, 1, logits, error);
}

bool Model::Generate(std::span<const std::uint32_t> prefix,
                     std::span<const std::uint32_t> negative, float guidance,
                     Phase phase, const PhaseSampling& sampling,
                     std::uint64_t seed, bool legacy_off, PhaseResult* result,
                     const std::function<bool(int)>& progress,
                     std::string* error) {
  Impl& m = *impl_;
  const auto start = std::chrono::steady_clock::now();
  const bool guided = guidance != 1.0F;
  if (guided && negative.empty())
    return Fail(error, "YuE2 guidance needs a negative prefix");
  const std::span<const std::uint32_t> prompts[2] = {prefix, negative};
  const int sequences = guided ? 2 : 1;
  int longest = 1;
  for (int q = 0; q < sequences; ++q) {
    if (prompts[q].empty())
      return Fail(error, "empty YuE2 prefix");
    longest = std::max(longest, static_cast<int>(prompts[q].size()));
  }
  if (sampling.max_tokens < 1 ||
      longest + sampling.max_tokens > kContext)
    return Fail(error, "YuE2 prefix + max_tokens exceed the 24576 context");
  if (!m.Reserve(longest, error))
    return false;
  for (int q = 0; q < sequences; ++q)
    if (!m.Allocate(m.cache[q], longest + sampling.max_tokens + 1, error) ||
        !m.Prefill(prompts[q], q, error))
      return false;
  if (!Check(hipStreamSynchronize(m.engine.stream), "prefill", error))
    return false;
  result->tokens.clear();
  result->truncated = true;
  result->prefill_seconds = Since(start);
  const VocabularySlice slice = PhaseSlice(phase);
  const std::uint32_t end = phase == Phase::kAbc ? kAbcEnd : kMusicEnd;
  std::mt19937_64 rng(seed);
  std::vector<float> logits, scores(slice.count);
  for (int step = 0; step < sampling.max_tokens; ++step) {
    if (!m.Logits(phase, sequences, &logits, error))
      return false;
    for (std::uint32_t i = 0; i < slice.count; ++i)
      scores[i] = guided ? logits[slice.count + i] +
                               guidance * (logits[i] - logits[slice.count + i])
                         : logits[i];
    const std::uint32_t token = SampleToken(scores, phase, sampling,
                                            result->tokens, step, legacy_off,
                                            &rng);
    if (token == end) {
      result->truncated = false;
      break;
    }
    result->tokens.push_back(token);
    if (progress && !progress(step + 1))
      return Fail(error, "cancelled");
    if (step + 1 == sampling.max_tokens)
      break;
    for (int q = 0; q < sequences; ++q)
      if (!m.Embed(std::span<const std::uint32_t>(&token, 1), q,
                   m.cache[q].length, error))
        return false;
    for (int i = 0; i < m.layer_count; ++i)
      for (int q = 0; q < sequences; ++q)
        if (!m.Layer(m.ar[i], i, q, q, 1, false, error))
          return false;
    for (int q = 0; q < sequences; ++q) {
      ++m.cache[q].length;
      LaunchRmsNorm(m.hidden.f32() + std::size_t(q) * m.shape.width,
                    m.norm.bf16(), nullptr, nullptr, 1, m.shape.width, nullptr,
                    m.normed32.f32() + std::size_t(q) * m.shape.width,
                    m.engine.stream);
    }
  }
  result->seconds = Since(start);
  return true;
}

bool Model::BeginAcoustic(std::span<const std::uint32_t> ar_tokens, int frames,
                          std::string* error) {
  Impl& m = *impl_;
  const int prefix = static_cast<int>(ar_tokens.size());
  const int rows = frames + 2;
  if (prefix < 1 || frames < 1 || prefix + rows > kContext ||
      rows > m.pe_rows)
    return Fail(error, "YuE2 acoustic chunk exceeds the context");
  if (!m.Reserve(std::max(prefix, rows), error) ||
      !m.Allocate(m.cache[0], prefix + rows, error) ||
      !m.Prefill(ar_tokens, 0, error))
    return false;
  std::vector<int> pos(rows);
  for (int j = 0; j < rows; ++j)
    pos[j] = prefix + j;
  if (!m.UploadPositions(pos, 0, error))
    return false;
  m.acoustic_frames = frames;
  return m.state.Reserve(std::size_t(rows) * kLatentChannels * 4, error) &&
         m.velocity.Reserve(std::size_t(frames) * kLatentChannels * 4,
                            error) &&
         m.time_in.Reserve(kFrequencies * 4, error) &&
         m.time_mid.Reserve(std::size_t(m.shape.width) * 4, error) &&
         m.time_emb.Reserve(std::size_t(m.shape.width) * 4, error);
}

bool Model::Velocity(std::span<const float> state, double raw_t,
                     std::vector<float>* velocity, std::string* error) {
  Impl& m = *impl_;
  const int frames = m.acoustic_frames;
  const int rows = frames + 2;
  const int w = m.shape.width;
  if (frames < 1 || state.size() != std::size_t(frames) * kLatentChannels)
    return Fail(error, "YuE2 velocity state has the wrong size");
  // x at all NAR rows: zero LATENT_START / LATENT_END, the state between.
  std::vector<float> x(std::size_t(rows) * kLatentChannels, 0.0F);
  std::copy(state.begin(), state.end(), x.begin() + kLatentChannels);
  // t = sigmoid(raw) (timestep_shift 1), sinusoid [cos | sin] of 128
  // frequencies exp(-ln(10000) i / 128) as TimestepEmbedder.
  const float t = 1.0F / (1.0F + std::exp(-static_cast<float>(raw_t)));
  std::vector<float> sinusoid(kFrequencies);
  const int half = kFrequencies / 2;
  for (int i = 0; i < half; ++i) {
    const float frequency = std::exp(-static_cast<float>(std::log(10000.0)) *
                                     static_cast<float>(i) / half);
    sinusoid[i] = std::cos(t * frequency);
    sinusoid[half + i] = std::sin(t * frequency);
  }
  Engine& e = m.engine;
  if (!Check(hipMemcpyAsync(m.state.get(), x.data(), x.size() * 4,
                            hipMemcpyHostToDevice, e.stream),
             "upload", error) ||
      !Check(hipMemcpyAsync(m.time_in.get(), sinusoid.data(), kFrequencies * 4,
                            hipMemcpyHostToDevice, e.stream),
             "upload", error))
    return false;
  LaunchMatVec(m.time1_w.bf16(), m.time1_b.bf16(), m.time_in.f32(),
               kFrequencies, w, false, m.time_mid.f32(), e.stream);
  LaunchMatVec(m.time2_w.bf16(), m.time2_b.bf16(), m.time_mid.f32(), w, w, true,
               m.time_emb.f32(), e.stream);
  const std::size_t count = std::size_t(rows) * w;
  NarInputKernel<<<GridFor(count), kThreads, 0, e.stream>>>(
      m.state.f32(), m.vae2llm_w.bf16(), m.vae2llm_b.bf16(), m.time_emb.f32(),
      m.pe.bf16(), w, count, m.hidden.f32());
  for (int i = 0; i < m.layer_count; ++i)
    if (!m.Layer(m.nar[i], i, 0, 0, rows, false, error))
      return false;
  LaunchRmsNorm(m.hidden.f32() + w, m.norm.bf16(), nullptr, nullptr, frames,
                w, nullptr, m.normed32.f32(), e.stream);
  NarOutputKernel<<<frames, kThreads, w * 4, e.stream>>>(
      m.normed32.f32(), m.llm2vae_w.bf16(), m.llm2vae_b.bf16(), w,
      m.velocity.f32());
  velocity->resize(state.size());
  return e.Download(m.velocity.get(), velocity->size() * 4, velocity->data(),
                    error);
}

bool Model::Solve(std::span<const float> noise, int steps,
                  std::vector<float>* latents,
                  const std::function<bool(int)>& progress,
                  std::string* error) {
  if (steps < 1)
    return Fail(error, "YuE2 needs at least one ODE step");
  // nar.CachedNAR.solve: midpoint from t = 1 (noise) to t = 0, raw times
  // logit(t) clamped to +-20.
  const auto raw = [](double t) {
    if (t >= 1.0)
      return 20.0;
    if (t <= 0.0)
      return -20.0;
    return std::clamp(std::log(t / (1.0 - t)), -20.0, 20.0);
  };
  std::vector<float> s(noise.begin(), noise.end()), mid(s.size()), v;
  const double dt = 1.0 / steps;
  for (int step = 0; step < steps; ++step) {
    const double t = 1.0 - step * dt;
    if (!Velocity(s, raw(t), &v, error))
      return false;
    const float half_dt = static_cast<float>(dt / 2);
    for (std::size_t i = 0; i < s.size(); ++i)
      mid[i] = s[i] - v[i] * half_dt;
    if (!Velocity(mid, raw(t - dt / 2), &v, error))
      return false;
    const float full_dt = static_cast<float>(dt);
    for (std::size_t i = 0; i < s.size(); ++i)
      s[i] = s[i] - v[i] * full_dt;
    if (progress && !progress(step + 1))
      return Fail(error, "cancelled");
  }
  for (const float value : s)
    if (!std::isfinite(value))
      return Fail(error, "YuE2 flow matching produced non-finite latents");
  *latents = std::move(s);
  return true;
}

}  // namespace gufo::models::yue2
