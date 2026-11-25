#include "src/models/minimax_h3/image_generation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <utility>

#include "src/models/minimax_h3/dit.hpp"
#include "src/models/minimax_h3/prompt_vision.hpp"
#include "src/models/minimax_h3/tokenizer.hpp"

namespace gufo::minimax_h3 {
namespace {

using Clock = std::chrono::steady_clock;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

double Milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

void Report(GenerationProgress progress, void* opaque, std::string_view phase,
            int completed, int total) {
  if (progress != nullptr) {
    progress(phase, completed, total, opaque);
  }
}

struct ProgressBridge {
  GenerationProgress callback{nullptr};
  void* opaque{nullptr};
  std::string_view phase;
};

void ForwardProgress(int completed, int total, void* opaque) {
  auto* bridge = static_cast<ProgressBridge*>(opaque);
  if (bridge != nullptr && bridge->callback != nullptr) {
    bridge->callback(bridge->phase, completed, total, bridge->opaque);
  }
}

double Sinc(double x) {
  if (x == 0.0) {
    return 1.0;
  }
  x *= 3.14159265358979323846;
  return std::sin(x) / x;
}

double Lanczos(double x) {
  return (x > -3.0 && x < 3.0) ? Sinc(x) * Sinc(x / 3.0) : 0.0;
}

struct Taps {
  int first{0};
  std::vector<double> weights;
};

// PIL precompute_coeffs for one output axis.
std::vector<Taps> Coefficients(int input, int output) {
  const double scale = static_cast<double>(input) / output;
  const double filter_scale = std::max(scale, 1.0);
  const double support = 3.0 * filter_scale;
  std::vector<Taps> taps(static_cast<std::size_t>(output));
  for (int index = 0; index < output; ++index) {
    const double center = (index + 0.5) * scale;
    int first = static_cast<int>(center - support + 0.5);
    first = std::max(first, 0);
    int last = static_cast<int>(center + support + 0.5);
    last = std::min(last, input);
    Taps& tap = taps[static_cast<std::size_t>(index)];
    tap.first = first;
    double sum = 0.0;
    for (int x = first; x < last; ++x) {
      const double weight = Lanczos((x - center + 0.5) / filter_scale);
      tap.weights.push_back(weight);
      sum += weight;
    }
    if (sum != 0.0) {
      for (double& weight : tap.weights) {
        weight /= sum;
      }
    }
  }
  return taps;
}

std::uint8_t Clip8(double value) {
  return static_cast<std::uint8_t>(
      std::clamp(std::floor(value + 0.5), 0.0, 255.0));
}

}  // namespace

bool ResizeRgb8Lanczos(const std::vector<std::uint8_t>& source,
                       int source_width, int source_height, int width,
                       int height, std::vector<std::uint8_t>* output,
                       std::string* error) {
  if (output == nullptr || source_width < 1 || source_height < 1 ||
      width < 1 || height < 1 ||
      source.size() != static_cast<std::size_t>(source_width) *
                           static_cast<std::size_t>(source_height) * 3U) {
    SetError(error, "invalid Lanczos resize arguments");
    return false;
  }
  if (width == source_width && height == source_height) {
    *output = source;
    return true;
  }
  // Horizontal pass into an 8-bit intermediate, then the vertical pass.
  std::vector<std::uint8_t> horizontal;
  const std::vector<std::uint8_t>* rows = &source;
  if (width != source_width) {
    const auto taps = Coefficients(source_width, width);
    horizontal.resize(static_cast<std::size_t>(width) * source_height * 3U);
    for (int y = 0; y < source_height; ++y) {
      const std::uint8_t* line =
          source.data() + static_cast<std::size_t>(y) * source_width * 3U;
      for (int x = 0; x < width; ++x) {
        const Taps& tap = taps[static_cast<std::size_t>(x)];
        for (int channel = 0; channel < 3; ++channel) {
          double sum = 0.0;
          for (std::size_t k = 0; k < tap.weights.size(); ++k) {
            sum += tap.weights[k] *
                   line[(static_cast<std::size_t>(tap.first) + k) * 3U +
                        static_cast<std::size_t>(channel)];
          }
          horizontal[(static_cast<std::size_t>(y) * width + x) * 3U +
                     static_cast<std::size_t>(channel)] = Clip8(sum);
        }
      }
    }
    rows = &horizontal;
  }
  if (height == source_height) {
    *output = std::move(horizontal);
    return true;
  }
  const auto taps = Coefficients(source_height, height);
  output->assign(static_cast<std::size_t>(width) * height * 3U, 0);
  const std::size_t stride = static_cast<std::size_t>(width) * 3U;
  for (int y = 0; y < height; ++y) {
    const Taps& tap = taps[static_cast<std::size_t>(y)];
    for (std::size_t column = 0; column < stride; ++column) {
      double sum = 0.0;
      for (std::size_t k = 0; k < tap.weights.size(); ++k) {
        sum += tap.weights[k] *
               (*rows)[(static_cast<std::size_t>(tap.first) + k) * stride +
                       column];
      }
      (*output)[static_cast<std::size_t>(y) * stride + column] = Clip8(sum);
    }
  }
  return true;
}

bool GenerateTextImage(const ImageGenerationRequest& request,
                       const CancellationToken* cancellation,
                       GenerationProgress progress, void* progress_opaque,
                       ImageGenerationResult* result,
                       ImageGenerationTelemetry* telemetry,
                       std::string* error) {
  const auto total_begin = Clock::now();
  ImageGenerationTelemetry local;
  if (result == nullptr) {
    SetError(error, "invalid MiniMax-H3-Image result");
    return false;
  }
  *result = {};
  auto canvas = ResolveImageCanvas(request.width, request.height,
                                   request.min_generation_area, error);
  if (!canvas) {
    return false;
  }
  auto geometry = ImageGenerationGeometry(*canvas, error);
  if (!geometry) {
    return false;
  }
  auto begin = Clock::now();
  auto inventory = ModelInventory::InspectImage(request.model_root, {}, error);
  if (!inventory) {
    return false;
  }
  local.inventory_ms = Milliseconds(begin, Clock::now());

  // Edit: size the references for the VAE and for the vision tower.
  const bool edit = !request.references.empty();
  std::vector<ImageEditReference> references;
  if (edit) {
    if (request.vision_projector.empty()) {
      SetError(error,
               "MiniMax-H3-Image edits need the Qwen3-VL vision projector "
               "(--h3-vision-projector)");
      return false;
    }
    begin = Clock::now();
    if (!PlanImageEditReferences(request.references, &references, error)) {
      return false;
    }
    local.references = references.size();
    local.reference_plan_ms = Milliseconds(begin, Clock::now());
  }

  // The prompt verbatim, without a chat template, truncated to 4096 tokens;
  // every row is a text row (tag 1). An edit puts a "Picture i: " label and
  // a vision block (tag 0) per reference in front of it.
  begin = Clock::now();
  Tokenizer tokenizer;
  std::vector<std::uint32_t> tokens;
  std::vector<std::uint8_t> tags;
  std::vector<PromptVisionBlock> vision_blocks;
  std::vector<KeyframeImage> vision_images;
  std::size_t prompt_token_limit = kPromptEncoderMaximumTokens;
  if (!Tokenizer::Load(request.model_root / "tokenizer" / "tokenizer.json",
                       &tokenizer, error) ||
      !tokenizer.EncodePrompt(request.prompt, &tokens, error)) {
    return false;
  }
  if (tokens.empty()) {
    SetError(error, "MiniMax-H3-Image prompt has no tokens");
    return false;
  }
  if (!edit) {
    if (tokens.size() > kH3ImageMaximumPromptTokens) {
      tokens.resize(kH3ImageMaximumPromptTokens);
      local.prompt_truncated = true;
    }
    tags.assign(tokens.size(), 1);
  } else {
    std::vector<std::vector<std::uint32_t>> labels(references.size());
    std::vector<std::pair<int, int>> sizes;
    for (std::size_t index = 0; index < references.size(); ++index) {
      if (!tokenizer.Encode(ImageEditLabel(index), false, &labels[index],
                            error)) {
        return false;
      }
      sizes.emplace_back(references[index].vision_width,
                         references[index].vision_height);
      vision_images.push_back({.width = references[index].vision_width,
                               .height = references[index].vision_height,
                               .rgb = references[index].vision_rgb});
    }
    // The whole presentation must fit the text encoder's limit for this
    // request, checked before any GPU work (the provider already rejects
    // such requests with HTTP 400): 4096 tokens, more when the device has
    // memory for the longer prompt phase and the DiT's extra rows. Measured
    // here, right before the prompt phase, and handed to the prompt child,
    // so the check and the encoder use the same number.
    {
      std::vector<std::pair<int, int>> uploads;
      std::vector<std::size_t> label_counts;
      std::size_t condition_rows = 0;
      for (std::size_t index = 0; index < references.size(); ++index) {
        uploads.emplace_back(request.references[index].width,
                             request.references[index].height);
        label_counts.push_back(labels[index].size());
        condition_rows +=
            static_cast<std::size_t>(references[index].latent_height / 2) *
            static_cast<std::size_t>(references[index].latent_width / 2);
      }
      const ImageEditMemory memory = CurrentImageEditMemory();
      const std::size_t image_rows =
          static_cast<std::size_t>(geometry->latent_height / 2) *
          static_cast<std::size_t>(geometry->latent_width / 2);
      prompt_token_limit =
          ImageEditTokenLimit(memory, condition_rows, image_rows);
      local.prompt_token_limit = prompt_token_limit;
      local.prompt_limit_free_bytes = memory.free_bytes;
      if (!CheckImageEditTokenBudget(
              CountImageEditTokens(uploads, label_counts, tokens.size()),
              prompt_token_limit, error)) {
        return false;
      }
    }
    ImageEditPresentation presentation;
    if (!BuildImageEditPresentation(labels, sizes, tokens, &presentation,
                                    error)) {
      return false;
    }
    tokens = std::move(presentation.tokens);
    tags = std::move(presentation.tags);
    local.prompt_truncated = presentation.prompt_truncated;
    local.vision_tokens = presentation.vision_tokens;
    for (const ImageEditVisionSlot& slot : presentation.slots) {
      PromptVisionBlock block;
      block.begin = slot.begin;
      block.grid_height = slot.grid_height;
      block.grid_width = slot.grid_width;
      vision_blocks.push_back(std::move(block));
    }
  }
  local.prompt_tokens = tokens.size();
  local.tokenizer_ms = Milliseconds(begin, Clock::now());

  Report(progress, progress_opaque, "prompt", 0, 1);
  begin = Clock::now();
  PromptEmbedding embedding;
  TraceStage("h3-image prompt phase start");
  if (PromptIsolationEnabled()) {
    // The vision tower and the text encoder run in a short-lived child, so
    // their device memory is gone before the DiT loads.
    const PromptIsolationInput isolated{
        .model_root = request.text_encoder_root,
        .manifest = request.text_encoder_manifest,
        .projector = edit ? request.vision_projector : std::filesystem::path(),
        .images = vision_images,
        .tokens = tokens,
        .blocks = vision_blocks,
        .maximum_tokens = prompt_token_limit};
    if (!EncodePromptIsolated(isolated, cancellation, &embedding,
                              &local.prompt_encoder, &local.vision_ms,
                              error)) {
      return false;
    }
  } else {
    if (edit) {
      const auto vision_begin = Clock::now();
      std::vector<PromptVisionBlock> encoded;
      if (!EncodeKeyframeVision(request.vision_projector, vision_images,
                                cancellation, &encoded, error)) {
        return false;
      }
      if (encoded.size() != vision_blocks.size()) {
        SetError(error, "MiniMax-H3-Image vision block count differs");
        return false;
      }
      for (std::size_t index = 0; index < encoded.size(); ++index) {
        if (encoded[index].grid_height != vision_blocks[index].grid_height ||
            encoded[index].grid_width != vision_blocks[index].grid_width) {
          SetError(error, "MiniMax-H3-Image vision grid differs");
          return false;
        }
        encoded[index].begin = vision_blocks[index].begin;
      }
      vision_blocks = std::move(encoded);
      local.vision_ms = Milliseconds(vision_begin, Clock::now());
    }
    auto text_inventory = ModelInventory::Inspect(
        request.text_encoder_root, request.text_encoder_manifest, error);
    PromptEncoderOptions prompt_options;
    prompt_options.vision_blocks = vision_blocks;
    prompt_options.maximum_tokens = prompt_token_limit;
    if (!text_inventory ||
        !EncodePromptLayer50(*text_inventory, tokens, prompt_options,
                             cancellation, &embedding, &local.prompt_encoder,
                             error)) {
      return false;
    }
  }
  vision_blocks.clear();
  vision_images.clear();
  if (embedding.tokens != tokens.size() ||
      embedding.width != kH3TextConditionWidth) {
    SetError(error, "MiniMax-H3-Image prompt embedding has an unexpected shape");
    return false;
  }
  TraceStage("h3-image prompt encoded");
  local.prompt_ms = Milliseconds(begin, Clock::now());
  Report(progress, progress_opaque, "prompt", 1, 1);

  // Edit: the references through the image VAE encoder (same CNN as the H3
  // VisualVAE encoder, one frame, tiled), each posterior sampled, rounded to
  // float16, cropped to whole patches, normalized and held at t = 0.999.
  std::vector<ReferenceLatent> reference_latents;
  if (edit) {
    Report(progress, progress_opaque, "references", 0,
           static_cast<int>(references.size()));
    begin = Clock::now();
    ImageVaeStatistics statistics;
    if (!LoadImageVaeStatistics(request.model_root, &statistics, error)) {
      return false;
    }
    for (std::size_t index = 0; index < references.size(); ++index) {
      const ImageEditReference& reference = references[index];
      const KeyframeImage image{.width = reference.vae_width,
                                .height = reference.vae_height,
                                .rgb = reference.vae_rgb};
      std::vector<KeyframeMoments> moments;
      VideoVaeEncoderTelemetry encoder_telemetry;
      ReferenceLatent latent;
      latent.placement = ImageEditPlacement(reference);
      if (!EncodeKeyframeMoments(*inventory, std::span(&image, 1),
                                 cancellation, &moments, &encoder_telemetry,
                                 error) ||
          moments.size() != 1U ||
          !BuildImageEditConditioning(
              moments.front().moments, moments.front().latent_height,
              moments.front().latent_width, reference.latent_height,
              reference.latent_width, statistics,
              ImageEditPosteriorSeed(request.seed, index),
              ImageEditAnchorSeed(request.seed, index), &latent.video,
              error)) {
        if (error != nullptr && error->empty()) {
          *error = "MiniMax-H3-Image reference encoding failed";
        }
        return false;
      }
      local.reference_encoder.load_ms += encoder_telemetry.load_ms;
      local.reference_encoder.encode_ms += encoder_telemetry.encode_ms;
      local.reference_encoder.tiles += encoder_telemetry.tiles;
      local.reference_encoder.weight_bytes = encoder_telemetry.weight_bytes;
      local.reference_encoder.scratch_bytes =
          std::max(local.reference_encoder.scratch_bytes,
                   encoder_telemetry.scratch_bytes);
      local.reference_rows +=
          static_cast<std::size_t>(reference.latent_height / 2) *
          static_cast<std::size_t>(reference.latent_width / 2);
      reference_latents.push_back(std::move(latent));
      Report(progress, progress_opaque, "references",
             static_cast<int>(index + 1), static_cast<int>(references.size()));
    }
    references.clear();
    local.reference_encode_ms = Milliseconds(begin, Clock::now());
    TraceStage("h3-image references encoded");
  }

  // Noise of the single latent frame, [24][h][w] (the plain request seed,
  // as text to image; the references use their own derived streams).
  std::vector<float> latent(kH3VideoLatentChannels *
                            static_cast<std::size_t>(geometry->latent_height) *
                            static_cast<std::size_t>(geometry->latent_width));
  NormalRng rng(request.seed);
  rng.Fill(latent);

  begin = Clock::now();
  auto denoiser = DenoiserSession::Create(
      *inventory,
      {.geometry = *geometry,
       .evaluations = kH3ImagePddSteps,
       .text_rows = embedding.tokens,
       .active_blocks = static_cast<int>(kDitBlocks),
       .reuse_interval = 1,
       .image = true,
       .sigmas = kH3ImagePddSigmas},
      {.layer50 = embedding.values,
       .tags = tags,
       .references = reference_latents},
      cancellation, &local.denoiser, error);
  if (denoiser == nullptr) {
    return false;
  }
  reference_latents.clear();
  ProgressBridge denoiser_progress{progress, progress_opaque, "denoiser"};
  if (!denoiser->DenoiseWithProgress(latent, {}, cancellation, ForwardProgress,
                                     &denoiser_progress, &local.denoiser,
                                     error)) {
    return false;
  }
  denoiser.reset();
  local.denoiser_ms = Milliseconds(begin, Clock::now());
  TraceStage("h3-image denoiser released");

  begin = Clock::now();
  auto decoder = ImageVaeDecoder::Create(
      *inventory,
      {.latent_height = geometry->latent_height,
       .latent_width = geometry->latent_width},
      cancellation, &local.vae, error);
  if (decoder == nullptr) {
    return false;
  }
  ProgressBridge vae_progress{progress, progress_opaque, "vae"};
  DecodedImage decoded;
  if (!decoder->Decode(latent, cancellation, ForwardProgress, &vae_progress,
                       &decoded, &local.vae, error)) {
    return false;
  }
  decoder.reset();
  local.vae_ms = Milliseconds(begin, Clock::now());
  TraceStage("h3-image decoder released");

  std::vector<std::uint8_t> canvas_rgb;
  FinalizeImageVaeRgb(decoded.normalized_rgb, &canvas_rgb);
  begin = Clock::now();
  if (!ResizeRgb8Lanczos(canvas_rgb, decoded.width, decoded.height,
                         canvas->output_width, canvas->output_height,
                         &result->rgb, error)) {
    return false;
  }
  local.resize_ms = Milliseconds(begin, Clock::now());
  result->width = canvas->output_width;
  result->height = canvas->output_height;
  result->canvas_width = canvas->width;
  result->canvas_height = canvas->height;
  local.total_ms = Milliseconds(total_begin, Clock::now());
  if (telemetry != nullptr) {
    *telemetry = std::move(local);
  }
  return true;
}

std::string ImageGenerationTelemetryJson(
    const ImageGenerationTelemetry& telemetry) {
  std::ostringstream json;
  json << "{\"model\":\"" << kH3ImageModelName << "\""
       << ",\"inventory_ms\":" << telemetry.inventory_ms
       << ",\"tokenizer_ms\":" << telemetry.tokenizer_ms
       << ",\"prompt_ms\":" << telemetry.prompt_ms
       << ",\"denoiser_ms\":" << telemetry.denoiser_ms
       << ",\"vae_ms\":" << telemetry.vae_ms
       << ",\"resize_ms\":" << telemetry.resize_ms
       << ",\"total_ms\":" << telemetry.total_ms
       << ",\"references\":" << telemetry.references
       << ",\"reference_plan_ms\":" << telemetry.reference_plan_ms
       << ",\"vision_ms\":" << telemetry.vision_ms
       << ",\"vision_tokens\":" << telemetry.vision_tokens
       << ",\"reference_encode_ms\":" << telemetry.reference_encode_ms
       << ",\"reference_tiles\":" << telemetry.reference_encoder.tiles
       << ",\"reference_rows\":" << telemetry.reference_rows
       << ",\"prompt_tokens\":" << telemetry.prompt_tokens
       << ",\"prompt_truncated\":"
       << (telemetry.prompt_truncated ? "true" : "false")
       << ",\"prompt_token_limit\":" << telemetry.prompt_token_limit
       << ",\"prompt_limit_free_bytes\":" << telemetry.prompt_limit_free_bytes
       << ",\"denoiser_setup_ms\":" << telemetry.denoiser.setup_ms
       << ",\"denoiser_peak_bytes\":" << telemetry.denoiser.peak_live_bytes
       << ",\"vae_tiles\":" << telemetry.vae.tile_ms.size()
       << ",\"vae_peak_bytes\":" << telemetry.vae.peak_live_bytes << '}';
  return json.str();
}

}  // namespace gufo::minimax_h3
