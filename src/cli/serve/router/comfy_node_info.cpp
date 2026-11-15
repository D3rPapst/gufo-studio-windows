#include "src/cli/serve/router/comfy_node_info.hpp"

#include <string>

namespace gufo::router::comfy {
namespace {

struct Entry {
  std::string_view name;
  std::string_view python_module;  // as ComfyUI reports it
  std::string_view json;    // the /object_info object without name/module
};

// As ComfyUI's comfy_extras/nodes_minimax_h3.py defines them (V1 form of
// the io.Schema), with gufo's ranges and notes on what gufo does.
constexpr Entry kEntries[] = {
    {"EmptyMiniMaxH3LatentAV", "comfy_extras.nodes_minimax_h3", R"({
 "input": {"required": {
  "width": ["INT", {"default": 1344, "min": 256, "max": 1344, "step": 32, "tooltip": "gufo: multiples of 32, 256..1344, at most 768x1344 pixels."}],
  "height": ["INT", {"default": 768, "min": 256, "max": 1344, "step": 32}],
  "length": ["INT", {"default": 124, "min": 5, "max": 345, "step": 17, "tooltip": "Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s). gufo renders 22..345 frames."}]}},
 "input_order": {"required": ["width", "height", "length"]},
 "output": ["LATENT"], "output_is_list": [false], "output_name": ["LATENT"],
 "display_name": "Empty MiniMax H3 AV Latent",
 "description": "Joint video+audio latent for MiniMax H3. Duration snaps to the model's 17k+5 frame grid at 24 fps. gufo: the positive conditioning comes from CLIPTextEncode (text to video+audio).",
 "category": "model/latent/minimax"})"},
    {"MiniMaxH3ImageToVideo", "comfy_extras.nodes_minimax_h3", R"({
 "input": {"required": {
  "clip": ["CLIP", {}],
  "vae": ["VAE", {}],
  "prompt": ["STRING", {"multiline": true, "dynamic_prompts": true}],
  "width": ["INT", {"default": 1344, "min": 256, "max": 1344, "step": 32, "tooltip": "gufo: multiples of 32, 256..1344, at most 768x1344 pixels."}],
  "height": ["INT", {"default": 768, "min": 256, "max": 1344, "step": 32}],
  "length": ["INT", {"default": 124, "min": 5, "max": 345, "step": 17, "tooltip": "Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s). gufo renders 22..345 frames."}]},
  "optional": {
  "first_frame": ["IMAGE", {}],
  "last_frame": ["IMAGE", {}]}},
 "input_order": {"required": ["clip", "vae", "prompt", "width", "height", "length"], "optional": ["first_frame", "last_frame"]},
 "output": ["CONDITIONING", "LATENT"], "output_is_list": [false, false], "output_name": ["positive", "LATENT"],
 "display_name": "MiniMax H3 Image to Video",
 "description": "t2va and fl2va: prompt (+ optional first/last keyframes) -> conditioning + AV latent.",
 "category": "model/conditioning/minimax"})"},
    {"MiniMaxH3AddGuide", "comfy_extras.nodes_minimax_h3", R"({
 "input": {"required": {
  "positive": ["CONDITIONING", {}],
  "latent": ["LATENT", {}],
  "frame_idx": ["INT", {"default": 0, "min": -9999, "max": 9999, "tooltip": "Frame index to anchor the image or the clip's first frame at. Negative values are counted from the end of the video."}]},
  "optional": {
  "vae": ["VAE", {"tooltip": "Video VAE, needed when an image is connected."}],
  "audio_vae": ["VAE", {"tooltip": "Audio VAE, needed when an audio is connected."}],
  "image": ["IMAGE", {"tooltip": "Image or video frames to anchor. Multi-frame batches are anchored as a clip and cropped down to the model's valid clip lengths: 5, 22, 39... (17k + 5) frames. Batches shorter than 5 frames use only the first image. gufo: LoadImage, or LoadVideo -> GetVideoComponents for a clip; at most 8 images and clips per video."}],
  "audio": ["AUDIO", {"tooltip": "Soundtrack to anchor starting at the same frame index, cropped to the video's remaining duration. gufo: LoadAudio or the soundtrack of LoadVideo -> GetVideoComponents; one audio anchor per video."}]}},
 "input_order": {"required": ["positive", "latent", "frame_idx"], "optional": ["vae", "audio_vae", "image", "audio"]},
 "output": ["CONDITIONING"], "output_is_list": [false], "output_name": ["positive"],
 "display_name": "Add Guide for MiniMax H3",
 "description": "Anchor an image, a short clip, audio, or a clip with its soundtrack at any frame of a MiniMax H3 video. Chain several nodes to anchor several frames.",
 "category": "model/conditioning/minimax"})"},
    {"MiniMaxH3SigmaShift", "comfy_extras.nodes_minimax_h3", R"({
 "input": {"required": {
  "model": ["MODEL", {}],
  "shift_video": ["FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}],
  "shift_audio": ["FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}]}},
 "input_order": {"required": ["model", "shift_video", "shift_audio"]},
 "output": ["MODEL"], "output_is_list": [false], "output_name": ["MODEL"],
 "display_name": "ModelSamplingMiniMaxH3",
 "description": "Set the video/audio flow shifts.",
 "category": "model/patch/minimax",
 "search_aliases": ["sigma shift", "minimax shift"]})"},
    // The audio noise mask of an H3 video: VAEEncodeAudio of the source
    // audio, SetLatentNoiseMask (0 = keep, 1 = generate), then this node
    // with the H3 latent as video_latent.
    {"LTXVConcatAVLatent", "comfy_extras.nodes_lt", R"({
 "input": {"required": {
  "video_latent": ["LATENT", {}],
  "audio_latent": ["LATENT", {}]}},
 "input_order": {"required": ["video_latent", "audio_latent"]},
 "output": ["LATENT"], "output_is_list": [false], "output_name": ["latent"],
 "display_name": "Concat AV Latent",
 "description": "Merge a video latent and an audio latent into a joint AV latent (any AV model, e.g. LTXV or MiniMax H3). gufo: MiniMax H3 only; audio_latent from VAEEncodeAudio through SetLatentNoiseMask (SolidMask / MaskComposite / InvertMask) holds the source audio where the mask is below 0.5.",
 "category": "model/latent/ltxv"})"},
    {"VAEEncodeAudio", "comfy_extras.nodes_audio", R"({
 "input": {"required": {
  "audio": ["AUDIO", {}],
  "vae": ["VAE", {}]}},
 "input_order": {"required": ["audio", "vae"]},
 "output": ["LATENT"], "output_is_list": [false], "output_name": ["LATENT"],
 "display_name": "VAE Encode Audio",
 "description": "gufo: the source audio of an H3 audio noise mask (LTXVConcatAVLatent).",
 "category": "model/latent",
 "search_aliases": ["audio to latent"]})"},
};

}  // namespace

std::optional<json::Value> NodeInfo(std::string_view class_type) {
  for (const Entry& entry : kEntries) {
    if (entry.name != class_type)
      continue;
    json::Value info = json::parse(entry.json);
    info["name"] = std::string(entry.name);
    info["python_module"] = std::string(entry.python_module);
    info["output_node"] = false;
    // The remaining fields ComfyUI's /object_info carries for every node.
    info["is_input_list"] = false;
    json::Value tooltips = json::Value::array();
    for (std::size_t i = 0; i < info.find("output")->size(); ++i)
      tooltips.push_back(nullptr);
    info["output_tooltips"] = std::move(tooltips);
    info["deprecated"] = false;
    info["experimental"] = false;
    info["api_node"] = false;
    return info;
  }
  return std::nullopt;
}

}  // namespace gufo::router::comfy
