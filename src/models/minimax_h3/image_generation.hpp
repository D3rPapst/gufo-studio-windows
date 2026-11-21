#ifndef GUFO_MODELS_MINIMAX_H3_IMAGE_GENERATION_HPP_
#define GUFO_MODELS_MINIMAX_H3_IMAGE_GENERATION_HPP_

// MiniMax-H3-Image text-to-image (MiniMaxAI/MiniMax-H3-Image, "t2i"): the
// prompt verbatim through the shared Qwen3-VL text encoder (hidden_states[50]
// of the H3 FL2VA encoder, bit-identical to the image release's), the H3 DiT
// without audio over [text | image] for the six fixed PDD steps (one output
// head each, Euler on the velocity, no CFG), the 2D ViT VAE decoder and the
// pipeline's 8-bit post-processing. A canvas scaled up to
// min_generation_area is resized back with Lanczos.
//
// Edits ("image_conditioned", up to 16 reference images): every reference
// enters twice, as a "Picture i: " label + Qwen3-VL vision block in front of
// the prompt (vision tower from the mmproj GGUF) and as a VAE latent held at
// t = 0.999 in the packed sequence [text | references | image].

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/minimax_h3/denoiser.hpp"
#include "src/models/minimax_h3/image_edit.hpp"
#include "src/models/minimax_h3/generation.hpp"
#include "src/models/minimax_h3/image_vae.hpp"
#include "src/models/minimax_h3/prompt_encoder.hpp"
#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/sampling.hpp"
#include "src/models/minimax_h3/video_vae_encoder.hpp"

namespace gufo::minimax_h3 {

/// The name the license asks to show wherever the model is offered.
inline constexpr std::string_view kH3ImageModelName = "MiniMax-H3-Image";
/// max_sequence_length of the pipeline: the prompt is truncated to it.
inline constexpr std::size_t kH3ImageMaximumPromptTokens = 4096;

struct ImageGenerationRequest {
  // The MiniMax-H3-Image release (transformer/, vae/, pdd_fused_heads, ...).
  std::filesystem::path model_root;
  // The MiniMax H3 FL2VA model root and its pinned source manifest: their
  // FL2VA/text_encoder is the shared conditioner.
  std::filesystem::path text_encoder_root;
  std::filesystem::path text_encoder_manifest;
  std::string prompt;
  std::uint64_t seed{0};
  int width{2048};
  int height{2048};
  std::int64_t min_generation_area{kH3ImageDefaultMinimumArea};
  // Edit: reference images in "<Picture i>" order (empty: text to image)
  // and the Qwen3-VL vision tower GGUF they need.
  std::vector<ImageEditInput> references;
  std::filesystem::path vision_projector;
};

struct ImageGenerationResult {
  int width{0};
  int height{0};
  int canvas_width{0};
  int canvas_height{0};
  // Interleaved RGB8, row-major, width * height * 3.
  std::vector<std::uint8_t> rgb;
};

struct ImageGenerationTelemetry {
  double inventory_ms{0.0};
  double tokenizer_ms{0.0};
  double prompt_ms{0.0};
  double denoiser_ms{0.0};
  double vae_ms{0.0};
  double resize_ms{0.0};
  double total_ms{0.0};
  // Edits: reference sizing, vision tower (inside prompt_ms when isolated)
  // and the VAE encoding of the references.
  double reference_plan_ms{0.0};
  double vision_ms{0.0};
  double reference_encode_ms{0.0};
  std::size_t references{0};
  std::size_t vision_tokens{0};
  std::size_t reference_rows{0};
  std::size_t prompt_tokens{0};
  bool prompt_truncated{false};
  // Edits: the text-encoder token limit of this request (4096, more with
  // enough free device memory; ImageEditTokenLimit) and the free bytes it
  // was computed from.
  std::size_t prompt_token_limit{0};
  std::uint64_t prompt_limit_free_bytes{0};
  PromptEncoderTelemetry prompt_encoder;
  DenoiserTelemetry denoiser;
  ImageVaeTelemetry vae;
  VideoVaeEncoderTelemetry reference_encoder;
};

/// Lanczos (a = 3, support widened by the scale when shrinking) resize of
/// interleaved RGB8, horizontal pass first, as PIL's Image.resize(LANCZOS).
[[nodiscard]] bool ResizeRgb8Lanczos(const std::vector<std::uint8_t>& source,
                                     int source_width, int source_height,
                                     int width, int height,
                                     std::vector<std::uint8_t>* output,
                                     std::string* error = nullptr);

[[nodiscard]] bool GenerateTextImage(const ImageGenerationRequest& request,
                                     const CancellationToken* cancellation,
                                     GenerationProgress progress,
                                     void* progress_opaque,
                                     ImageGenerationResult* result,
                                     ImageGenerationTelemetry* telemetry,
                                     std::string* error = nullptr);

[[nodiscard]] std::string ImageGenerationTelemetryJson(
    const ImageGenerationTelemetry& telemetry);

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_IMAGE_GENERATION_HPP_
