// Unit tests for the MiniMax H3 nodes of the ComfyUI adapter beyond the
// FilmStudio graphs in router_comfy_graph_test: /object_info signatures,
// EmptyMiniMaxH3LatentAV, MiniMaxH3SigmaShift and the audio noise mask.
// Graphs are built from the node signatures of ComfyUI's
// comfy_extras/nodes_minimax_h3.py.

#include <cassert>
#include <iostream>
#include <string>
#include <variant>

#include "src/cli/serve/router/comfy_graph.hpp"
#include "src/cli/serve/router/comfy_node_info.hpp"
#include "src/core/json.hpp"

namespace comfy = gufo::router::comfy;
namespace json = gufo::json;

namespace {

// ComfyUI text to video+audio: EmptyMiniMaxH3LatentAV + CLIPTextEncode.
constexpr const char* kH3Empty = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "strength_model": 1.0}},
 "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "type": "minimax", "device": "default"}},
 "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
 "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
 "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "a quiet office at night"}},
 "7": {"class_type": "EmptyMiniMaxH3LatentAV", "inputs": {"width": 896, "height": 512, "length": 124}},
 "8": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["6", 0], "latent_image": ["7", 0], "seed": 5, "steps": 8, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
 "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
 "10": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["8", 0], "vae": ["5", 0]}},
 "11": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "h3/f"}},
 "12": {"class_type": "SaveAudio", "inputs": {"audio": ["10", 0], "filename_prefix": "h3/ton"}}
})";

comfy::Plan PlanOf(const std::string& text) {
  comfy::Plan plan;
  comfy::PlanError error;
  const bool ok = comfy::BuildPlan(json::parse(text), {}, &plan, &error);
  if (!ok)
    std::cerr << "unexpected plan error: " << error.message << "\n";
  assert(ok);
  return plan;
}

comfy::PlanError ErrorOf(const std::string& text) {
  comfy::Plan plan;
  comfy::PlanError error;
  assert(!comfy::BuildPlan(json::parse(text), {}, &plan, &error));
  return error;
}

std::string Replace(std::string text, const std::string& from,
                    const std::string& to) {
  const auto at = text.find(from);
  assert(at != std::string::npos);
  return text.replace(at, from.size(), to);
}

/// `text` with `nodes` (JSON members, each ending in ",\n") put in front of
/// node "8".
std::string WithNodes(const std::string& text, const std::string& nodes) {
  return Replace(text, " \"8\": {", nodes + " \"8\": {");
}

void TestEmptyLatent() {
  const comfy::Plan plan = PlanOf(kH3Empty);
  assert(plan.tasks.size() == 1);
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kH3);
  assert(video.prompt == "a quiet office at night");
  assert(video.width == 896 && video.height == 512 && video.frames == 124);
  assert(video.steps == 8 && video.seed == 5 && video.lora == "turbo");
  assert(!video.first_frame && !video.last_frame);
  // Guides chain onto the CLIPTextEncode conditioning as in ComfyUI.
  const std::string guided = Replace(
      WithNodes(kH3Empty,
                R"( "20": {"class_type": "LoadImage", "inputs": {"image": "start.png"}},
 "21": {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": ["6", 0], "latent": ["7", 0], "frame_idx": 0, "vae": ["4", 0], "image": ["20", 0]}},
)"),
      R"("positive": ["6", 0], "latent_image")",
      R"("positive": ["21", 0], "latent_image")");
  const comfy::VideoTask h3 = std::get<comfy::VideoTask>(PlanOf(guided).tasks[0].spec);
  assert((h3.first_frame == comfy::FileRef{"start.png", "input"}));
  assert(h3.prompt == "a quiet office at night");
  // The prompt must come from a text encoder, not a zeroed one.
  const std::string zeroed = Replace(
      WithNodes(kH3Empty,
                R"( "22": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
)"),
      R"("positive": ["6", 0])", R"("positive": ["22", 0])");
  assert(ErrorOf(zeroed).message.find("ConditioningZeroOut") !=
         std::string::npos);
}

void TestSigmaShift() {
  // MiniMaxH3SigmaShift on the model chain reaches the worker.
  const std::string shifted = Replace(
      WithNodes(kH3Empty,
                R"( "30": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["2", 0], "shift_video": 12.0, "shift_audio": 6.5}},
)"),
      R"("model": ["2", 0], "positive")", R"("model": ["30", 0], "positive")");
  const comfy::Plan plan = PlanOf(shifted);
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.shift_video == 12.0 && video.shift_audio == 6.5);
  assert(plan.warnings.empty());
  const json::Value body = comfy::VideoRequestBody(video);
  const json::Value* shift = body.find("gufo")->find("sigma_shift");
  assert(shift != nullptr);
  assert(shift->find("video")->as_double() == 12.0);
  assert(shift->find("audio")->as_double() == 6.5);
  // The trained 12 / 3 send nothing (older workers keep working).
  std::string trained = shifted;
  trained = Replace(trained, "\"shift_audio\": 6.5", "\"shift_audio\": 3.0");
  const comfy::VideoTask plain =
      std::get<comfy::VideoTask>(PlanOf(trained).tasks[0].spec);
  assert(!plain.shift_video && !plain.shift_audio);
  assert(comfy::VideoRequestBody(plain).find("gufo")->find("sigma_shift") ==
         nullptr);
  // Out of ComfyUI's range.
  std::string zero = Replace(shifted, "\"shift_audio\": 6.5",
                             "\"shift_audio\": 0");
  assert(ErrorOf(zero).node_errors.find("30") != nullptr);
  // The nearest shift node wins, as in ComfyUI (applied last).
  std::string twice = Replace(
      WithNodes(shifted,
                R"( "31": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["30", 0], "shift_video": 10.0, "shift_audio": 3.0}},
)"),
      R"("model": ["30", 0], "positive")", R"("model": ["31", 0], "positive")");
  const comfy::VideoTask nearest = std::get<comfy::VideoTask>(PlanOf(twice).tasks[0].spec);
  assert(nearest.shift_video == 10.0 && nearest.shift_audio == 3.0);
  // cfg is not silently dropped any more: a warning says so.
  const comfy::Plan cfg = PlanOf(Replace(kH3Empty, "\"cfg\": 1.0", "\"cfg\": 4.0"));
  assert(cfg.warnings.size() == 1 &&
         cfg.warnings[0].find("CFG-distilled") != std::string::npos);
}

void TestAudioMask() {
  // ComfyUI's audio noise mask: VAEEncodeAudio of a silent track,
  // SetLatentNoiseMask with SolidMask 0 (keep everything) and
  // LTXVConcatAVLatent in front of the sampler.
  const std::string masked = Replace(
      WithNodes(kH3Empty,
                R"( "40": {"class_type": "LoadAudio", "inputs": {"audio": "stille.wav"}},
 "41": {"class_type": "VAEEncodeAudio", "inputs": {"audio": ["40", 0], "vae": ["5", 0]}},
 "42": {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": 512, "height": 512}},
 "43": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["41", 0], "mask": ["42", 0]}},
 "44": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["7", 0], "audio_latent": ["43", 0]}},
)"),
      R"("latent_image": ["7", 0])", R"("latent_image": ["44", 0])");
  const comfy::Plan plan = PlanOf(masked);
  assert(plan.warnings.empty());
  const comfy::VideoTask video =
      std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kH3);
  assert(video.prompt == "a quiet office at night" && video.frames == 124);
  assert((video.audio_source == comfy::FileRef{"stille.wav", "input"}));
  assert(!video.audio_source_from_video);
  assert(video.audio_mask.size() == 1024);
  for (const float value : video.audio_mask)
    assert(value == 0.0F);
  const json::Value body = comfy::VideoRequestBody(video);
  const json::Value* mask = body.find("gufo")->find("audio_mask");
  assert(mask != nullptr && mask->find("mask")->size() == 1024);
  // The second half of the time axis generates: MaskComposite of a 0 and a
  // 1 SolidMask.
  const std::string half = Replace(
      WithNodes(masked,
                R"( "45": {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": 256, "height": 512}},
 "46": {"class_type": "MaskComposite", "inputs": {"destination": ["42", 0], "source": ["45", 0], "x": 256, "y": 0, "operation": "add"}},
)"),
      R"("mask": ["42", 0])", R"("mask": ["46", 0])");
  const comfy::VideoTask split =
      std::get<comfy::VideoTask>(PlanOf(half).tasks[0].spec);
  assert(split.audio_mask.size() == 1024);
  assert(split.audio_mask[0] == 0.0F && split.audio_mask[255] == 0.0F);
  assert(split.audio_mask[256] == 1.0F && split.audio_mask[512] == 0.0F &&
         split.audio_mask[1023] == 1.0F);
  // The soundtrack of a video works as the source too.
  const std::string from_video = Replace(
      Replace(masked,
              R"("40": {"class_type": "LoadAudio", "inputs": {"audio": "stille.wav"}})",
              R"("40": {"class_type": "LoadVideo", "inputs": {"file": "raum.mp4"}},
 "47": {"class_type": "GetVideoComponents", "inputs": {"video": ["40", 0]}})"),
      R"("audio": ["40", 0], "vae")", R"("audio": ["47", 1], "vae")");
  const comfy::VideoTask sound =
      std::get<comfy::VideoTask>(PlanOf(from_video).tasks[0].spec);
  assert(sound.audio_source_from_video);
  // Without a mask ComfyUI regenerates the audio: a warning, no mask.
  const std::string unmasked = Replace(
      masked, R"("audio_latent": ["43", 0])", R"("audio_latent": ["41", 0])");
  const comfy::Plan plain = PlanOf(unmasked);
  assert(plain.warnings.size() == 1);
  assert(!std::get<comfy::VideoTask>(plain.tasks[0].spec).audio_source);
  // A soft mask is rounded, with a warning.
  const comfy::Plan soft = PlanOf(Replace(masked, R"("value": 0.0)",
                                          R"("value": 0.3)"));
  assert(soft.warnings.size() == 1 &&
         soft.warnings[0].find("below 0.5") != std::string::npos);
}

void TestNodeInfo() {
  // Every class with a signature is a class the adapter accepts.
  for (const std::string& cls : comfy::SupportedClassTypes()) {
    const auto info = comfy::NodeInfo(cls);
    if (!info)
      continue;
    assert(info->member_str("name") == cls);
    assert(info->find("input")->find("required") != nullptr);
    assert(info->find("output")->is_array());
    assert(info->find("output")->size() ==
           info->find("output_name")->size());
  }
  assert(comfy::NodeInfo("EmptyMiniMaxH3LatentAV"));
  const auto shift = comfy::NodeInfo("MiniMaxH3SigmaShift");
  assert(shift);
  const json::Value* required = shift->find("input")->find("required");
  const json::Value& audio = *required->find("shift_audio");
  assert(audio.items()[0].str() == "FLOAT");
  assert(audio.items()[1].find("default")->as_double() == 3.0);
  assert(required->find("shift_video")->items()[1].find("default")->as_double() ==
         12.0);
  assert(shift->member_str("display_name") == "ModelSamplingMiniMaxH3");
  const auto guide = comfy::NodeInfo("MiniMaxH3AddGuide");
  assert(guide);
  const json::Value* optional = guide->find("input")->find("optional");
  assert(optional->find("audio")->items()[0].str() == "AUDIO");
  assert(guide->find("input")->find("required")->find("frame_idx")
             ->items()[1].find("min")->as_double() == -9999);
  const auto video = comfy::NodeInfo("MiniMaxH3ImageToVideo");
  assert(video);
  assert(video->find("output")->items()[1].str() == "LATENT");
  assert(video->find("input")->find("required")->find("length")
             ->items()[1].find("max")->as_double() == 345);
  // Classes without a signature keep the bare entry.
  assert(!comfy::NodeInfo("KSampler"));
}

}  // namespace

int main() {
  TestNodeInfo();
  TestEmptyLatent();
  TestSigmaShift();
  TestAudioMask();
  std::cout << "router_comfy_h3_test: all passed\n";
  return 0;
}
