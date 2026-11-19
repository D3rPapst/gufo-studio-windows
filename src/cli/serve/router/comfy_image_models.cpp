#include "src/cli/serve/router/comfy_image_models.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::router::comfy {
namespace {

std::string LowerText(std::string_view text) {
  std::string out(text);
  for (char& c : out)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

std::string Number(double value) {
  char text[32];
  std::snprintf(text, sizeof text, "%g", value);
  return text;
}

bool IsAlnum(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) != 0;
}

/// Lower-case file name without folders; `extension` receives a known
/// model-file extension ("gguf", ...) that was cut off.
std::string BareName(std::string_view name, std::string* extension) {
  if (const auto slash = name.find_last_of("/\\");
      slash != std::string_view::npos)
    name.remove_prefix(slash + 1);
  std::string lower = LowerText(name);
  for (const std::string_view ext :
       {".safetensors", ".sft", ".gguf", ".ckpt", ".pth", ".pt", ".bin"}) {
    if (lower.size() > ext.size() && lower.ends_with(ext)) {
      if (extension != nullptr)
        *extension = std::string(ext.substr(1));
      lower.resize(lower.size() - ext.size());
      break;
    }
  }
  return lower;
}

bool NamesFastVariant(std::string_view compact) {
  for (const std::string_view token : {"turbo", "lightning", "distill"})
    if (compact.find(token) != std::string_view::npos)
      return true;
  return false;
}

std::string ListText(const std::vector<std::string>& ids) {
  std::string out = "[";
  for (std::size_t i = 0; i < ids.size(); ++i)
    out += (i ? ", '" : "'") + ids[i] + "'";
  return out + "]";
}

std::string NotInList(std::string_view key, std::string_view name,
                      const ModelIds& models, std::string_view reason) {
  return "Value not in list: " + std::string(key) + ": '" + std::string(name) +
         "' not in " + ListText(ServedImageModels(models)) +
         " (the gufo image models of this router; " + std::string(reason) +
         ")";
}

bool Served(const ModelIds& models, const std::string& id) {
  return !id.empty() &&
         (models.available.empty() ||
          std::find(models.available.begin(), models.available.end(), id) !=
              models.available.end());
}

}  // namespace

// The image models of the ComfyUI adapter. A new model: one row here (its
// ModelIds field, name token and the graph features its worker runs).
const std::vector<ImageModelSpec>& ImageModelTable() {
  static const std::vector<std::string_view> kQwen21 = {
      "2.1", "2_1", "2-1", "image21", "image_21", "image-21"};
  static const std::vector<ImageModelSpec> table = {
      {.id = &ModelIds::image,
       .display = "Qwen-Image 2.1",
       .family = "qwenimage",
       .fast = false,
       .releases = kQwen21,
       .caps = {}},
      // Step-distilled: a fixed 8-step schedule, no partial-strength
      // image to image (docs/models/qwen-image-2.1.md).
      {.id = &ModelIds::image_turbo,
       .display = "Qwen-Image 2.1 Turbo",
       .family = "qwenimage",
       .fast = true,
       .releases = kQwen21,
       .caps = {.image_to_image = false, .fixed_steps = 8}},
      // MOSS-Image SFT (moss-image extension): text to image only, its own
      // CFG 3.0 against an empty prompt, sides 256..2048.
      {.id = &ModelIds::moss_image,
       .display = "MOSS-Image",
       .family = "mossimage",
       .fast = false,
       .releases = {},
       .caps = {.references = false,
                .image_to_image = false,
                .mask = false,
                .control = false,
                .regions = false,
                .loras = false,
                .negative_prompt = false,
                .guidance = false,
                .guidance_steps = false,
                .min_side = 256,
                .max_side = 2048,
                .fixed_guidance = 3.0}},
  };
  return table;
}

std::string CompactLoaderName(std::string_view name) {
  const std::string bare = BareName(name, nullptr);
  std::string out;
  for (const char c : bare)
    if (IsAlnum(c))
      out.push_back(c);
  return out;
}

std::string NonBf16Tag(std::string_view name) {
  std::string extension;
  const std::string bare = BareName(name, &extension);
  if (extension == "gguf")
    return "gguf";
  std::vector<std::string> tokens(1);
  for (const char c : bare) {
    if (IsAlnum(c))
      tokens.back().push_back(c);
    else if (!tokens.back().empty())
      tokens.emplace_back();
  }
  for (const std::string& token : tokens) {
    if (token.empty())
      continue;
    for (const std::string_view tag :
         {"int8", "int4", "fp8", "fp4", "nf4", "fp16", "fp32", "e4m3", "e5m2",
          "gguf", "svdq", "convrot"})
      if (token.find(tag) != std::string::npos)
        return std::string(tag);
    // llama.cpp quant names: Q4_K_M, Q8_0, IQ3_XXS ("q4", "q8", "iq3").
    const std::size_t digit = token.starts_with("iq") ? 2
                              : token.starts_with("q") ? 1
                                                       : 0;
    if (digit > 0 && token.size() > digit &&
        std::isdigit(static_cast<unsigned char>(token[digit])) &&
        std::all_of(token.begin() + digit, token.end(), [](char c) {
          return std::isdigit(static_cast<unsigned char>(c)) || c == 'k' ||
                 c == 's' || c == 'm' || c == 'l' || c == 'x';
        }))
      return token;
  }
  return {};
}

std::vector<std::string> ServedImageModels(const ModelIds& models) {
  std::vector<std::string> ids;
  for (const ImageModelSpec& spec : ImageModelTable()) {
    const std::string& id = models.*spec.id;
    if (Served(models, id) &&
        std::find(ids.begin(), ids.end(), id) == ids.end())
      ids.push_back(id);
  }
  return ids;
}

ImageModelChoice ChooseImageModel(std::string_view name, std::string_view key,
                                  std::string_view node,
                                  const ModelIds& models) {
  ImageModelChoice choice;
  const std::string compact = CompactLoaderName(name);
  const auto& table = ImageModelTable();
  const ImageModelSpec* spec = nullptr;
  bool direct = false;
  // The router id itself (what /object_info lists).
  for (const ImageModelSpec& row : table) {
    const std::string& id = models.*row.id;
    if (!id.empty() && compact == CompactLoaderName(id)) {
      spec = &row;
      direct = true;
      break;
    }
  }
  if (spec == nullptr) {
    const bool fast = NamesFastVariant(compact);
    const ImageModelSpec* family = nullptr;
    for (const ImageModelSpec& row : table) {
      if (compact.find(row.family) == std::string::npos)
        continue;
      family = family != nullptr ? family : &row;
      if (row.fast == fast) {
        spec = &row;
        break;
      }
    }
    if (spec == nullptr) {
      choice.error =
          family == nullptr
              ? NotInList(key, name, models,
                          "the name names none of them; gufo does not load "
                          "other checkpoint files")
              : NotInList(key, name, models,
                          std::string("gufo has no ") +
                              (fast ? "turbo / lightning / distilled "
                                    : "non-distilled ") +
                              std::string(family->display));
      return choice;
    }
  }
  const std::string& id = models.*spec->id;
  if (!Served(models, id)) {
    const std::string want = id.empty() ? std::string(spec->display) : id;
    choice.error = NotInList(
        key, name, models,
        std::string(name) + " is a " + std::string(spec->display) +
            " checkpoint, but the router serves no model '" + want +
            "' (add an [external/" + want + "] section to the router preset)");
    return choice;
  }
  choice.spec = spec;
  choice.id = id;
  const std::string prefix = "node " + std::string(node) + ": ";
  if (!direct && !spec->releases.empty()) {
    const std::string bare = BareName(name, nullptr);
    const bool named = std::any_of(
        spec->releases.begin(), spec->releases.end(),
        [&](std::string_view r) { return bare.find(r) != std::string::npos; });
    if (!named)
      choice.warnings.push_back(prefix + std::string(name) +
                                " does not name " +
                                std::string(spec->display) + "; gufo runs " +
                                id + " instead");
  }
  if (const std::string tag = NonBf16Tag(name); !tag.empty())
    choice.warnings.push_back(
        prefix + "gufo runs its own BF16 " + std::string(spec->display) +
        " weights (" + id + ") for " + std::string(name) + "; the file's " +
        tag +
        " weights are not loaded (the name only selects the model, so the "
        "image is the same as with the BF16 file)");
  return choice;
}

std::optional<ImageGraphProblem> FitImageTask(
    const ImageModelSpec& spec, const std::string& id,
    const std::string& sampler, ImageTask* task,
    std::vector<std::string>* warnings) {
  const ImageModelCaps& caps = spec.caps;
  const std::string who = std::string(spec.display) + " (" + id + ")";
  const auto refuse = [&](std::string_view type, const std::string& what) {
    return ImageGraphProblem{std::string(type), who + " " + what};
  };
  if (caps.fixed_steps > 0 && task->steps != caps.fixed_steps)
    return refuse("unsupported_value",
                  "runs exactly " + std::to_string(caps.fixed_steps) +
                      " steps, not " + std::to_string(task->steps) +
                      "; set steps to " + std::to_string(caps.fixed_steps));
  if (caps.fixed_steps == 0 &&
      (task->steps < caps.min_steps || task->steps > caps.max_steps))
    return refuse("unsupported_value",
                  "runs " + std::to_string(caps.min_steps) + ".." +
                      std::to_string(caps.max_steps) + " steps, not " +
                      std::to_string(task->steps));
  if (!caps.image_to_image && task->strength)
    return refuse("unsupported_value",
                  "has no partial-strength image to image; set denoise to 1");
  if (!caps.references && !task->references.empty())
    return refuse("unsupported_link",
                  "runs text to image only: reference images "
                  "(TextEncodeQwenImage21 images, ReferenceLatent) have no "
                  "counterpart; use a Qwen-Image 2.1 loader for edits");
  if (!caps.mask && !task->mask.name.empty())
    return refuse("unsupported_link",
                  "has no masked editing (SetLatentNoiseMask / "
                  "VAEEncodeForInpaint)");
  if (!caps.control && task->control)
    return refuse("unsupported_link", "has no ControlNet");
  if (!caps.regions && !task->regions.empty())
    return refuse("unsupported_link",
                  "has no regional conditioning (ConditioningSetMask / "
                  "SetArea / Combine)");
  if (!caps.loras && !task->loras.empty())
    return refuse("unsupported_link",
                  "takes no LoRAs; remove the LoraLoader nodes");
  if (!caps.guidance_steps && task->guidance_steps)
    return refuse("unsupported_value",
                  "has no guidance_steps (its guidance is fixed); remove the "
                  "input");
  if (caps.max_side > 0) {
    int width = 0, height = 0;
    if (task->size_source == SizeSource::kExplicit) {
      // As the adapter sends them: multiples of 32.
      width = std::max(32, static_cast<int>(std::lround(task->width / 32.0)) * 32);
      height =
          std::max(32, static_cast<int>(std::lround(task->height / 32.0)) * 32);
    } else if (task->size_source == SizeSource::kSquare) {
      width = height = task->resolution;
    }
    if (width > 0 && (std::min(width, height) < caps.min_side ||
                      std::max(width, height) > caps.max_side))
      return refuse("unsupported_value",
                    "renders sides of " + std::to_string(caps.min_side) +
                        ".." + std::to_string(caps.max_side) + " pixels, not " +
                        std::to_string(width) + "x" + std::to_string(height));
  }
  const std::string prefix = "node " + sampler + ": ";
  if (!caps.negative_prompt && !task->negative_prompt.empty()) {
    warnings->push_back(prefix + "the negative prompt is ignored (" + who +
                        " guides against an empty prompt)");
    task->negative_prompt.clear();
  }
  if (!caps.guidance && task->guidance) {
    if (caps.fixed_guidance > 0 &&
        std::abs(*task->guidance - caps.fixed_guidance) > 1e-6)
      warnings->push_back(prefix + "cfg " + Number(*task->guidance) +
                          " is ignored (" + who + " runs its own CFG " +
                          Number(caps.fixed_guidance) +
                          ")");
    task->guidance.reset();
  }
  return std::nullopt;
}

}  // namespace gufo::router::comfy
