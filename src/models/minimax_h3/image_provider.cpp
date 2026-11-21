// `gufo serve image` provider "minimax-h3-image": MiniMax-H3-Image text to
// image and image edits ("<Picture i>" in upload order) on the H3 DiT, the
// shared H3 text encoder with its Qwen3-VL vision tower and the image VAE.
// The release takes up to 16 references, but every one costs at least 1024
// text-encoder tokens and gufo's encoder takes 4096 in all: at most 3 images
// (squares; wide ones cost more), checked with the prompt before queueing.
// AI-assisted (Claude Code).

#include <gufo/ext/generated/minimax_h3_constants.hpp>
#include <gufo/ext/image.hpp>
#include <gufo/ext/registry.hpp>

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <gufo/ext/host.hpp>
#include "src/models/minimax_h3/image_generation.hpp"
#include "src/models/minimax_h3/json.hpp"
#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/tokenizer.hpp"

namespace gufo::models::minimax_h3 {
namespace {

namespace h3 = gufo::minimax_h3;

bool ProbeMinimaxH3Image(std::string_view model_root) {
  try {
    const std::filesystem::path root(model_root);
    if (!std::filesystem::is_regular_file(root / "model_index.json") ||
        !std::filesystem::is_regular_file(root / "pdd_fused_heads.safetensors") ||
        !std::filesystem::is_regular_file(
            root / "transformer" / "model.safetensors.index.json")) {
      return false;
    }
    const auto index =
        h3::json::ParseFile(root / "model_index.json", 1U << 20U);
    const auto* name = index.Find("_class_name");
    return name != nullptr && name->IsString() &&
           name->AsString() == "MiniMaxH3ImagePipeline";
  } catch (...) {
    return false;
  }
}

bool ResolveMinimaxH3ImageArtifacts(std::string_view model_root,
                                    std::string* error) {
  return h3::ModelInventory::InspectImage(std::filesystem::path(model_root), {},
                                          error)
      .has_value();
}

#if defined(ENGINE_ENABLE_HIP)

void Log(const std::string& line) {
  server::Logger::Log(server::LogLevel::kInfo, "image", line);
}

std::filesystem::path EnvironmentPath(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::filesystem::path() : std::filesystem::path(value);
}

// The pinned H3 FL2VA source manifest, as the video provider resolves it.
std::filesystem::path SourceManifest() {
  std::filesystem::path manifest = EnvironmentPath("GUFO_H3_SOURCE_MANIFEST");
#if defined(GUFO_H3_SOURCE_MANIFEST)
  if (manifest.empty()) {
    manifest = GUFO_H3_SOURCE_MANIFEST;
  }
#endif
  return manifest;
}

// "gufo": {"min_generation_area": N} of the request extension.
std::int64_t MinimumArea(const std::string& extension_json) {
  if (extension_json.empty()) {
    return h3::kH3ImageDefaultMinimumArea;
  }
  const auto root = h3::json::Parse(extension_json);
  const auto* gufo = root.Find("gufo");
  if (gufo == nullptr || !gufo->IsObject()) {
    return h3::kH3ImageDefaultMinimumArea;
  }
  const auto* area = gufo->Find("min_generation_area");
  if (area == nullptr) {
    return h3::kH3ImageDefaultMinimumArea;
  }
  if (!area->IsInteger()) {
    throw std::invalid_argument(
        "gufo.min_generation_area must be an integer pixel count");
  }
  return area->AsInt64();
}

class MinimaxH3ImageRunner : public ext::ImageRunner {
public:
  MinimaxH3ImageRunner(std::filesystem::path model_root,
                       std::filesystem::path text_encoder_root,
                       std::filesystem::path manifest,
                       std::filesystem::path vision_projector)
      : model_root_(std::move(model_root)),
        text_encoder_root_(std::move(text_encoder_root)),
        manifest_(std::move(manifest)),
        vision_projector_(std::move(vision_projector)) {
    Log("event=provider_ready model=" + std::string(h3::kH3ImageModelName) +
        " text_encoder_root=" + text_encoder_root_.string() + " edits=" +
        (vision_projector_.empty() ? "off" : "on"));
  }

  std::optional<ext::ImageCapabilities> Capabilities() const override {
    // Edits need the Qwen3-VL vision tower for the "Picture i" blocks.
    return ext::ImageCapabilities{
        .edits = !vision_projector_.empty(),
        .transparent_background = false,
        .max_side = h3::kH3ImageMaximumSide,
        .max_pixels = static_cast<std::uint64_t>(h3::kH3ImageMaximumPixels),
        .max_images = h3::kH3ImageEditMaximumImages};
  }

  void ValidateRequest(const ext::ImageRequest& request) const override {
    if (!request.named_images.empty()) {
      throw std::invalid_argument(
          "MiniMax-H3-Image takes reference images (\"image\") only, no "
          "init image, mask or control image");
    }
    if (!request.images.empty() && vision_projector_.empty()) {
      throw std::invalid_argument(
          "MiniMax-H3-Image edits need --h3-vision-projector");
    }
    if (request.images.size() > h3::kH3ImageEditMaximumImages) {
      throw std::invalid_argument(
          "MiniMax-H3-Image takes at most " +
          std::to_string(h3::kH3ImageEditMaximumImages) +
          " reference images in gufo (each costs at least 1024 of the 4096 "
          "text-encoder tokens); got " +
          std::to_string(request.images.size()));
    }
    for (const auto& image : request.images) {
      if (image.width < 32 || image.height < 32) {
        throw std::invalid_argument(
            "MiniMax-H3-Image reference images need at least 32 pixels on "
            "each side");
      }
    }
    if (!request.images.empty()) {
      CheckEditTokens(request);
    }
    if (request.steps != 0 && request.steps != h3::kH3ImagePddSteps) {
      throw std::invalid_argument(
          "MiniMax-H3-Image runs its fixed 6 distilled steps (steps 0 or 6)");
    }
    std::string error;
    if (!h3::ResolveImageCanvas(request.width, request.height,
                                MinimumArea(request.extension_json), &error)) {
      throw std::invalid_argument(error);
    }
  }

  ext::ImageResult Generate(const ext::ImageRequest& request,
                            const ext::CancellationCheck& is_cancelled) override {
    ValidateRequest(request);
    h3::ImageGenerationRequest internal{
        .model_root = model_root_,
        .text_encoder_root = text_encoder_root_,
        .text_encoder_manifest = manifest_,
        .prompt = request.prompt,
        .seed = request.seed,
        .width = request.width,
        .height = request.height,
        .min_generation_area = MinimumArea(request.extension_json),
        .references = {},
        .vision_projector = vision_projector_,
    };
    // Reference images in upload order; an alpha channel is dropped, as
    // PIL's convert("RGB") does.
    for (const auto& image : request.images) {
      h3::ImageEditInput input{.width = image.width, .height = image.height};
      const std::size_t pixels =
          static_cast<std::size_t>(image.width) * image.height;
      input.rgb.resize(pixels * 3U);
      for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        input.rgb[pixel * 3U + 0U] = image.rgba[pixel * 4U + 0U];
        input.rgb[pixel * 3U + 1U] = image.rgba[pixel * 4U + 1U];
        input.rgb[pixel * 3U + 2U] = image.rgba[pixel * 4U + 2U];
      }
      internal.references.push_back(std::move(input));
    }
    h3::CancellationToken token;
    std::jthread watcher;
    if (is_cancelled) {
      watcher =
          std::jthread([&token, &is_cancelled](const std::stop_token& stop) {
            while (!stop.stop_requested()) {
              if (is_cancelled()) {
                token.Cancel();
                break;
              }
              std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
          });
    }
    h3::ImageGenerationResult result;
    h3::ImageGenerationTelemetry telemetry;
    std::string error;
    const bool ok = h3::GenerateTextImage(internal, &token, nullptr, nullptr,
                                          &result, &telemetry, &error);
    if (watcher.joinable()) {
      watcher.request_stop();
      watcher.join();
    }
    if (!ok) {
      throw std::runtime_error(std::string(h3::kH3ImageModelName) + ": " +
                               error);
    }
    std::ostringstream line;
    line << "event=generated model=" << h3::kH3ImageModelName
         << " size=" << result.width << 'x' << result.height
         << " canvas=" << result.canvas_width << 'x' << result.canvas_height
         << " references=" << request.images.size()
         << " steps=" << h3::kH3ImagePddSteps
         << " telemetry=" << h3::ImageGenerationTelemetryJson(telemetry);
    Log(line.str());
    ext::ImageBuffer image{.width = result.width, .height = result.height};
    image.rgba.resize(static_cast<std::size_t>(result.width) * result.height *
                      4U);
    for (std::size_t pixel = 0;
         pixel < static_cast<std::size_t>(result.width) * result.height;
         ++pixel) {
      image.rgba[pixel * 4U + 0U] = result.rgb[pixel * 3U + 0U];
      image.rgba[pixel * 4U + 1U] = result.rgb[pixel * 3U + 1U];
      image.rgba[pixel * 4U + 2U] = result.rgb[pixel * 3U + 2U];
      image.rgba[pixel * 4U + 3U] = 255;
    }
    return ext::ImageResult{
        .image = std::move(image),
        .metrics =
            ext::ImageMetrics{
                .prompt_ms = telemetry.prompt_ms,
                .denoise_ms = telemetry.denoiser_ms,
                .vae_ms = telemetry.vae_ms,
                .total_ms = telemetry.total_ms,
                .resident_weight_bytes = static_cast<std::size_t>(
                    telemetry.denoiser.core_weight_bytes),
                .steps = h3::kH3ImagePddSteps,
            },
    };
  }

  // The edit presentation (labels, vision blocks, prompt) against the text
  // encoder's 4096 tokens, with the release tokenizer, before any GPU work.
  void CheckEditTokens(const ext::ImageRequest& request) const {
    std::string error;
    std::vector<std::uint32_t> prompt;
    std::vector<std::size_t> labels;
    {
      std::lock_guard lock(tokenizer_mutex_);
      if (!tokenizer_) {
        auto tokenizer = std::make_unique<h3::Tokenizer>();
        if (!h3::Tokenizer::Load(model_root_ / "tokenizer" / "tokenizer.json",
                                 tokenizer.get(), &error)) {
          throw std::runtime_error("MiniMax-H3-Image tokenizer: " + error);
        }
        tokenizer_ = std::move(tokenizer);
      }
      if (!tokenizer_->EncodePrompt(request.prompt, &prompt, &error)) {
        throw std::invalid_argument("MiniMax-H3-Image prompt: " + error);
      }
      for (std::size_t index = 0; index < request.images.size(); ++index) {
        std::vector<std::uint32_t> ids;
        if (!tokenizer_->Encode(h3::ImageEditLabel(index), false, &ids,
                                &error)) {
          throw std::runtime_error("MiniMax-H3-Image label: " + error);
        }
        labels.push_back(ids.size());
      }
    }
    std::vector<std::pair<int, int>> sizes;
    for (const auto& image : request.images) {
      sizes.emplace_back(image.width, image.height);
    }
    if (!h3::CheckImageEditTokenBudget(
            h3::CountImageEditTokens(sizes, labels, prompt.size()),
            h3::kPromptEncoderMaximumTokens, &error)) {
      throw std::invalid_argument(error);
    }
  }

  std::string Query(std::string_view topic) const override {
    if (topic == "model") {
      return "{\"name\":\"" + std::string(h3::kH3ImageModelName) + "\"}";
    }
    return {};
  }

private:
  std::filesystem::path model_root_;
  std::filesystem::path text_encoder_root_;
  std::filesystem::path manifest_;
  std::filesystem::path vision_projector_;
  // Loaded on the first edit request (about 7 MB of JSON).
  mutable std::mutex tokenizer_mutex_;
  mutable std::unique_ptr<h3::Tokenizer> tokenizer_;
};

std::unique_ptr<ext::ImageRunner> CreateWithOptions(
    const ext::ImageRunnerOptions& options, std::string* error) {
  try {
    std::filesystem::path text_root;
    if (const auto* value = ext::FindOption<std::string>(options.options,
                                                         "--h3-root")) {
      text_root = *value;
    }
    if (text_root.empty()) {
      text_root = EnvironmentPath("GUFO_H3_ROOT");
    }
    if (text_root.empty()) {
      throw std::invalid_argument(
          "MiniMax-H3-Image needs the MiniMax H3 (FL2VA) model root for its "
          "shared text encoder: --h3-root PATH or GUFO_H3_ROOT");
    }
    if (!std::filesystem::is_directory(text_root / "FL2VA" / "text_encoder")) {
      throw std::invalid_argument("no FL2VA/text_encoder under --h3-root " +
                                  text_root.string());
    }
    std::filesystem::path manifest;
    if (const auto* value = ext::FindOption<std::string>(options.options,
                                                         "--h3-manifest")) {
      manifest = *value;
    }
    if (manifest.empty()) {
      manifest = SourceManifest();
    }
    if (manifest.empty() || !std::filesystem::is_regular_file(manifest)) {
      throw std::invalid_argument(
          "MiniMax-H3-Image needs the pinned H3 source manifest "
          "(GUFO_H3_SOURCE_MANIFEST)");
    }
    // Optional: the Qwen3-VL vision tower (mmproj GGUF converted from the
    // H3 text encoder) that edits need; without it edits stay off.
    std::filesystem::path projector;
    if (const auto* value = ext::FindOption<std::string>(
            options.options, "--h3-vision-projector")) {
      projector = *value;
    }
    if (projector.empty()) {
      projector = EnvironmentPath("GUFO_H3_VISION_PROJECTOR");
    }
    if (!projector.empty() && !std::filesystem::is_regular_file(projector)) {
      throw std::invalid_argument("--h3-vision-projector file not found: " +
                                  projector.string());
    }
    std::string inspect_error;
    if (!h3::ModelInventory::InspectImage(options.model_root, {},
                                          &inspect_error)) {
      throw std::invalid_argument(inspect_error);
    }
    return std::make_unique<MinimaxH3ImageRunner>(
        options.model_root, text_root, manifest, projector);
  } catch (const std::exception& exception) {
    if (error != nullptr) {
      *error = exception.what();
    }
    return nullptr;
  }
}

std::unique_ptr<ext::ImageRunner> Create(const std::string& model_root,
                                         std::string* error) {
  return CreateWithOptions(
      ext::ImageRunnerOptions{.model_root = model_root, .options = {}}, error);
}

#endif

}  // namespace
}  // namespace gufo::models::minimax_h3

void minimax_h3_image_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_minimax_h3::kProvider_minimax_h3_image);
#if defined(ENGINE_ENABLE_HIP)
  using namespace gufo::models::minimax_h3;
  registry.AddImageProvider({
      .id = "minimax-h3-image",
      .probe = &ProbeMinimaxH3Image,
      .resolve = &ResolveMinimaxH3ImageArtifacts,
      .create = &Create,
      .create_with_options = &CreateWithOptions,
      .preview_latent = nullptr,
  });
#endif
}
