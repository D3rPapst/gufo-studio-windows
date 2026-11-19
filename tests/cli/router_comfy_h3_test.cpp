// Unit tests for the MiniMax H3 nodes of the ComfyUI adapter beyond the
// FilmStudio graphs in router_comfy_graph_test: /object_info signatures,
// EmptyMiniMaxH3LatentAV, MiniMaxH3SigmaShift, the audio noise mask,
// MiniMaxH3FunControlNetApply (gufo.control) and MiniMaxH3ReferenceToVideo
// (gufo.references).
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

/// kH3Empty with a MiniMaxH3FunControlNetApply ("50") between the LoRA and
/// the sampler; the control video is LoadVideo -> GetVideoComponents.
std::string WithControl(const std::string& control_inputs) {
  return Replace(
      WithNodes(kH3Empty,
                std::string(R"( "51": {"class_type": "ModelPatchLoader", "inputs": {"name": "MiniMax-H3-Fun-Controlnet-Union-2.0.safetensors"}},
 "52": {"class_type": "LoadVideo", "inputs": {"file": "pose.mp4"}},
 "53": {"class_type": "GetVideoComponents", "inputs": {"video": ["52", 0]}},
 "50": {"class_type": "MiniMaxH3FunControlNetApply", "inputs": {"model": ["2", 0], "model_patch": ["51", 0], "vae": ["4", 0], )") +
                    control_inputs + "}},\n"),
      R"("model": ["2", 0], "positive")", R"("model": ["50", 0], "positive")");
}

void TestFunControlNet() {
  const std::string pose = WithControl(
      R"("strength": 0.8, "start_percent": 0.0, "end_percent": 0.9, "control_video": ["53", 0])");
  const comfy::Plan plan = PlanOf(pose);
  assert(plan.tasks.size() == 1);
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kH3);
  assert(video.h3_control && !video.h3_references);
  assert(video.h3_control_strength == 0.8);
  assert(video.h3_control_start == 0.0 && video.h3_control_end == 0.9);
  assert((video.control_video == comfy::FileRef{"pose.mp4", "input"}));
  assert(!video.h3_control_mask && !video.h3_control_source);
  // The turbo LoRA stays on, with a warning (the ControlNet was trained
  // without it).
  assert(video.lora == "turbo");
  assert(plan.warnings.size() == 1 &&
         plan.warnings[0].find("turbo") != std::string::npos);
  const json::Value body = comfy::VideoRequestBody(video);
  const json::Value* control = body.find("gufo")->find("control");
  assert(control != nullptr);
  assert(control->find("strength")->as_double() == 0.8);
  assert(control->find("start_percent")->as_double() == 0.0);
  assert(control->find("end_percent")->as_double() == 0.9);
  // The executor attaches the videos; the body has none yet.
  assert(control->find("video") == nullptr);
  // The other H3 options stay out.
  assert(body.find("gufo")->find("references") == nullptr);
  assert(body.find("gufo")->find("sigma_shift") == nullptr);
  assert(body.find("gufo")->find("audio_mask") == nullptr);
  // Without the LoRA: no warning.
  const comfy::Plan plain_model = PlanOf(Replace(
      pose, R"("model": ["2", 0], "model_patch")",
      R"("model": ["1", 0], "model_patch")"));
  assert(plain_model.warnings.empty());
  assert(std::get<comfy::VideoTask>(plain_model.tasks[0].spec).lora.empty());
  // Strength 0, or neither control video nor mask: ComfyUI returns the
  // model unchanged, so does gufo.
  const comfy::VideoTask off = std::get<comfy::VideoTask>(
      PlanOf(Replace(pose, R"("strength": 0.8)", R"("strength": 0.0)"))
          .tasks[0].spec);
  assert(!off.h3_control && !off.control_video);
  assert(comfy::VideoRequestBody(off).find("gufo")->find("control") == nullptr);
  const comfy::VideoTask empty = std::get<comfy::VideoTask>(
      PlanOf(WithControl(
                 R"("strength": 1.0, "start_percent": 0.0, "end_percent": 1.0)"))
          .tasks[0].spec);
  assert(!empty.h3_control);
  // Inpaint: the mask through ImageToMask (red) on a loaded video, and the
  // source video behind it.
  const std::string inpaint = WithNodes(
      WithControl(
          R"("strength": 1.0, "start_percent": 0.0, "end_percent": 1.0, "mask": ["56", 0], "source_video": ["58", 0])"),
      R"( "54": {"class_type": "LoadVideo", "inputs": {"file": "maske.mp4"}},
 "55": {"class_type": "GetVideoComponents", "inputs": {"video": ["54", 0]}},
 "56": {"class_type": "ImageToMask", "inputs": {"image": ["55", 0], "channel": "red"}},
 "57": {"class_type": "LoadVideo", "inputs": {"file": "quelle.mp4"}},
 "58": {"class_type": "GetVideoComponents", "inputs": {"video": ["57", 0]}},
)");
  const comfy::VideoTask masked =
      std::get<comfy::VideoTask>(PlanOf(inpaint).tasks[0].spec);
  assert(masked.h3_control && !masked.control_video);
  assert((masked.h3_control_mask == comfy::FileRef{"maske.mp4", "input"}));
  assert((masked.h3_control_source == comfy::FileRef{"quelle.mp4", "input"}));
  // The mask is read from the red channel only.
  assert(ErrorOf(Replace(inpaint, R"("channel": "red")",
                         R"("channel": "green")"))
             .message.find("red channel") != std::string::npos);
  // A mask that is not an ImageToMask of a video is refused.
  const std::string solid = Replace(
      inpaint,
      R"("56": {"class_type": "ImageToMask", "inputs": {"image": ["55", 0], "channel": "red"}})",
      R"("56": {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": 512, "height": 512}})");
  assert(ErrorOf(solid).message.find("ImageToMask") != std::string::npos);
  // Out of ComfyUI's range.
  assert(ErrorOf(Replace(pose, R"("strength": 0.8)", R"("strength": 11.0)"))
             .message.find("strength") != std::string::npos);
  assert(ErrorOf(Replace(pose, R"("end_percent": 0.9)",
                         R"("end_percent": 1.5)"))
             .message.find("percent") != std::string::npos);
  // Two active control nodes on one model chain: gufo applies one.
  const std::string twice = Replace(
      WithNodes(pose,
                R"( "59": {"class_type": "MiniMaxH3FunControlNetApply", "inputs": {"model": ["50", 0], "model_patch": ["51", 0], "vae": ["4", 0], "strength": 1.0, "start_percent": 0.0, "end_percent": 1.0, "control_video": ["53", 0]}},
)"),
      R"("model": ["50", 0], "positive")", R"("model": ["59", 0], "positive")");
  assert(ErrorOf(twice).message.find("two") != std::string::npos);
}

// ComfyUI Ref2VA: MiniMaxH3ReferenceToVideo makes the conditioning and the
// latent; autogrow inputs in ComfyUI's API naming ("group.prefix_N").
constexpr const char* kH3Reference = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va_bf16.safetensors", "weight_dtype": "default"}},
 "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "type": "minimax", "device": "default"}},
 "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
 "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
 "20": {"class_type": "LoadImage", "inputs": {"image": "person.png"}},
 "21": {"class_type": "LoadImage", "inputs": {"image": "raum.png"}},
 "22": {"class_type": "LoadVideo", "inputs": {"file": "gang.mp4"}},
 "23": {"class_type": "GetVideoComponents", "inputs": {"video": ["22", 0]}},
 "24": {"class_type": "LoadAudio", "inputs": {"audio": "stimme.wav"}},
 "7": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0], "prompt": "<Picture 1> walks through <Picture 2> like in <Video 1>, speaking with <Audio 1>", "width": 896, "height": 512, "length": 124, "ref_image_size": "max", "ref_images.ref_image_1": ["20", 0], "ref_images.ref_image_2": ["21", 0], "ref_videos.ref_video_1": ["23", 0], "ref_video_audios.ref_video_audio_1": ["23", 1], "ref_audios.ref_audio_1": ["24", 0]}},
 "8": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["7", 0], "latent_image": ["7", 1], "seed": 9, "steps": 50, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
 "10": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["8", 0], "vae": ["5", 0]}},
 "11": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "h3/ref"}},
 "12": {"class_type": "SaveAudio", "inputs": {"audio": ["10", 0], "filename_prefix": "h3/refton"}}
})";

void TestReferenceToVideo() {
  const comfy::Plan plan = PlanOf(kH3Reference);
  assert(plan.tasks.size() == 1);
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kH3);
  assert(video.prompt.starts_with("<Picture 1> walks"));
  assert(video.width == 896 && video.height == 512 && video.frames == 124);
  assert(video.steps == 50 && video.seed == 9 && video.lora.empty());
  assert(video.h3_references && !video.h3_control);
  assert(video.h3_ref_image_max);
  assert(video.h3_ref_images.size() == 2);
  assert((video.h3_ref_images[0] == comfy::FileRef{"person.png", "input"}));
  assert((video.h3_ref_images[1] == comfy::FileRef{"raum.png", "input"}));
  assert(video.h3_ref_videos.size() == 1);
  assert((video.h3_ref_videos[0].file == comfy::FileRef{"gang.mp4", "input"}));
  assert(video.h3_ref_videos[0].audio &&
         video.h3_ref_videos[0].audio_from_video);
  assert(video.h3_ref_audios.size() == 1);
  assert((video.h3_ref_audios[0] == comfy::FileRef{"stimme.wav", "input"}));
  assert(!video.first_frame && !video.last_frame);
  const json::Value body = comfy::VideoRequestBody(video);
  const json::Value* references = body.find("gufo")->find("references");
  assert(references != nullptr);
  assert(references->member_str("image_size") == "max");
  // The executor attaches the files; the body has none yet.
  assert(references->find("images") == nullptr);
  assert(body.find("gufo")->find("control") == nullptr);
  // Plain input names (no group prefix) work too; "match" is the default.
  std::string plain = kH3Reference;
  for (const auto& [from, to] :
       {std::pair{"\"ref_images.ref_image_1\"", "\"ref_image_1\""},
        std::pair{"\"ref_images.ref_image_2\"", "\"ref_image_2\""},
        std::pair{"\"ref_videos.ref_video_1\"", "\"ref_video_1\""},
        std::pair{"\"ref_video_audios.ref_video_audio_1\"",
                  "\"ref_video_audio_1\""},
        std::pair{"\"ref_audios.ref_audio_1\"", "\"ref_audio_1\""},
        std::pair{"\"ref_image_size\": \"max\", ", ""}})
    plain = Replace(plain, from, to);
  const comfy::VideoTask named =
      std::get<comfy::VideoTask>(PlanOf(plain).tasks[0].spec);
  assert(!named.h3_ref_image_max);
  assert(named.h3_ref_images.size() == 2 && named.h3_ref_videos.size() == 1 &&
         named.h3_ref_audios.size() == 1);
  assert(comfy::VideoRequestBody(named)
             .find("gufo")
             ->find("references")
             ->member_str("image_size") == "match");
  // A soundtrack from LoadAudio is a file, not the video's own track.
  const comfy::VideoTask loaded = std::get<comfy::VideoTask>(
      PlanOf(Replace(kH3Reference,
                     R"("ref_video_audios.ref_video_audio_1": ["23", 1])",
                     R"("ref_video_audios.ref_video_audio_1": ["24", 0])"))
          .tasks[0].spec);
  assert(loaded.h3_ref_videos[0].audio &&
         !loaded.h3_ref_videos[0].audio_from_video);
  assert((*loaded.h3_ref_videos[0].audio ==
          comfy::FileRef{"stimme.wav", "input"}));
  // Audio references alone are refused (as in ComfyUI's Ref2VA).
  std::string audio_only = kH3Reference;
  for (const char* input :
       {R"("ref_images.ref_image_1": ["20", 0], )",
        R"("ref_images.ref_image_2": ["21", 0], )",
        R"("ref_videos.ref_video_1": ["23", 0], )",
        R"("ref_video_audios.ref_video_audio_1": ["23", 1], )"})
    audio_only = Replace(audio_only, input, "");
  assert(ErrorOf(audio_only).message.find("audio references") !=
         std::string::npos);
  // gufo's size limits apply as for every H3 job.
  assert(ErrorOf(Replace(kH3Reference, "\"width\": 896", "\"width\": 1600"))
             .message.find("multiples of 32") != std::string::npos);
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
  const auto reference = comfy::NodeInfo("MiniMaxH3ReferenceToVideo");
  assert(reference);
  assert(reference->find("output")->items()[0].str() == "CONDITIONING" &&
         reference->find("output")->items()[1].str() == "LATENT");
  // Autogrow inputs: prefix / min / max sit in the template (ComfyUI V3).
  const json::Value* ref_optional = reference->find("input")->find("optional");
  assert(ref_optional->find("ref_images")->items()[1].find("template")
             ->find("max")->as_double() ==
         9);
  assert(ref_optional->find("ref_videos")->items()[1].find("template")
             ->find("max")->as_double() ==
         3);
  assert(ref_optional->find("ref_audios")->items()[1].find("template")
             ->find("max")->as_double() ==
         3);
  const auto control = comfy::NodeInfo("MiniMaxH3FunControlNetApply");
  assert(control);
  assert(control->find("output")->items()[0].str() == "MODEL");
  const json::Value* control_required =
      control->find("input")->find("required");
  assert(control_required->find("strength")->items()[1].find("max")
             ->as_double() == 10.0);
  assert(control_required->find("end_percent")->items()[1].find("default")
             ->as_double() == 1.0);
  assert(control->find("input")->find("optional")->find("mask") != nullptr);
  // Classes without a signature keep the bare entry.
  assert(!comfy::NodeInfo("KSampler"));
}

}  // namespace

int main() {
  TestNodeInfo();
  TestEmptyLatent();
  TestSigmaShift();
  TestAudioMask();
  TestFunControlNet();
  TestReferenceToVideo();
  std::cout << "router_comfy_h3_test: all passed\n";
  return 0;
}
