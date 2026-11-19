#ifndef GUFO_CLI_SERVE_ROUTER_COMFY_IMAGE_MODELS_HPP_
#define GUFO_CLI_SERVE_ROUTER_COMFY_IMAGE_MODELS_HPP_

// The gufo image models a ComfyUI image graph (KSampler on an empty latent,
// TextEncodeQwenImage21 or VAEEncode) can run on, chosen by the
// UNETLoader `unet_name` / CheckpointLoaderSimple `ckpt_name`. One table in
// comfy_image_models.cpp lists each model: the name pattern that selects
// it, the ModelIds field with its router id and the graph features its
// worker can run. A new image model is one more row there.
//
// Name matching ignores folders ("qwen/..."), the extension (.safetensors,
// .gguf, ...), case and the separators "_", "-", ".", " ": the family token
// ("qwenimage", "mossimage") must appear in what is left. "turbo",
// "lightning" or "distill" pick the row's fast (step-distilled) variant. The
// router id itself (e.g. "qwen-image-2.1-turbo", as /object_info lists it)
// selects that row directly. A precision tag (int8, fp8, gguf, Q4_K_M,
// ...) only warns: gufo always runs its own registered weights.

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/router/comfy_graph.hpp"

namespace gufo::router::comfy {

/// The graph features a gufo image worker runs (gufo fields of
/// /v1/images/*). A graph that needs a missing one fails validation.
struct ImageModelCaps {
  bool references{true};       // TextEncodeQwenImage21 images / ReferenceLatent
  bool image_to_image{true};   // denoise < 1 (gufo.strength)
  bool mask{true};             // SetLatentNoiseMask / VAEEncodeForInpaint
  bool control{true};          // Qwen-Image Fun ControlNet
  bool regions{true};          // ConditioningSetMask / SetArea / Combine
  bool loras{true};            // LoraLoader(ModelOnly)
  bool negative_prompt{true};  // gufo.negative_prompt
  bool guidance{true};         // KSampler cfg -> gufo.guidance
  bool guidance_steps{true};   // gufo.guidance_steps
  int fixed_steps{0};          // > 0: the only step count (Turbo: 8)
  int min_steps{1};
  int max_steps{100};
  int min_side{0};             // 0 = not checked here
  int max_side{0};
  double fixed_guidance{0};    // > 0: the worker's own CFG (MOSS: 3.0)
};

/// One gufo image model the ComfyUI adapter maps loader names to.
struct ImageModelSpec {
  /// The ModelIds field holding the router id (so presets and tests can
  /// rename it; empty = the model is not offered).
  std::string ModelIds::*id;
  std::string_view display;  // "Qwen-Image 2.1 Turbo"
  /// Lower-case alphanumeric token the normalized loader name must contain.
  std::string_view family;
  /// True for the step-distilled variant chosen by turbo/lightning/distill.
  bool fast{false};
  /// Lower-case substrings naming the release gufo serves ("2.1", ...). A
  /// family name without any of them runs with a warning. Empty = any.
  std::vector<std::string_view> releases;
  ImageModelCaps caps;
};

/// Every row, in /object_info order.
const std::vector<ImageModelSpec>& ImageModelTable();

/// The loader name without folders and extension, lower case, letters and
/// digits only: "Qwen/qwen_image_2.1_int8.safetensors" -> "qwenimage21int8".
std::string CompactLoaderName(std::string_view name);

/// The precision / quantization a checkpoint name states when it is not
/// BF16 ("int8", "fp8", "gguf", "q4", ...), "" otherwise.
std::string NonBf16Tag(std::string_view name);

/// The router ids of the table's models that `models` serves (all with an
/// id when `models.available` is empty), in table order.
std::vector<std::string> ServedImageModels(const ModelIds& models);

struct ImageModelChoice {
  const ImageModelSpec* spec{nullptr};  // null: `error` says why
  std::string id;
  std::string error;  // a value_not_in_list message
  std::vector<std::string> warnings;
};

/// The model for a loader name. `key` is the input ("unet_name" /
/// "ckpt_name") and `node` the loader's node id, both for the messages.
ImageModelChoice ChooseImageModel(std::string_view name, std::string_view key,
                                  std::string_view node,
                                  const ModelIds& models);

/// A graph feature `spec` cannot run: `at` is the sampler or loader node.
struct ImageGraphProblem {
  std::string type;  // "unsupported_value" / "unsupported_link"
  std::string message;
};

/// Checks `task` (built from the graph) against `spec`'s features. Fields
/// the worker ignores (negative prompt, cfg on MOSS-Image) are cleared with a
/// warning; features it cannot run return a problem.
std::optional<ImageGraphProblem> FitImageTask(const ImageModelSpec& spec,
                                              const std::string& id,
                                              const std::string& sampler,
                                              ImageTask* task,
                                              std::vector<std::string>* warnings);

}  // namespace gufo::router::comfy

#endif  // GUFO_CLI_SERVE_ROUTER_COMFY_IMAGE_MODELS_HPP_
