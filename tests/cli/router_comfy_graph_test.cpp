// Unit tests for the ComfyUI workflow mapping (comfy_graph). The Qwen-Image
// and Qwen3-TTS fixtures are FilmStudio v100 workflows as ComfyUI recorded
// them in /history (long prompt texts shortened); the H3 and Wan graphs are
// built from the node signatures ComfyUI's /object_info reports.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "src/cli/serve/router/comfy_graph.hpp"
#include "src/cli/serve/router/comfy_image_models.hpp"
#include "src/cli/serve/router/comfy_node_info.hpp"
#include "src/cli/serve/router/comfy_preview.hpp"
#include "src/core/json.hpp"

namespace comfy = gufo::router::comfy;
namespace json = gufo::json;

namespace {

constexpr const char* kQwenText = R"({
 "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
 "2": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
 "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
 "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["1", 0], "prompt": "A single mature adult male office worker", "negative_prompt": "chibi, childlike", "vae": ["2", 0], "resolution": 1024}},
 "6": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["5", 2], "seed": 1769224114, "steps": 25, "cfg": 2.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
 "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "life3_zukunftsszenarien/v100_lauf_kurz/stil/demo_personen_r1"}}
})";

constexpr const char* kQwenCanvas = R"({
 "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
 "2": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
 "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
 "r1": {"class_type": "LoadImage", "inputs": {"image": "referenz1_seed1141445518_00001_.png"}},
 "r2": {"class_type": "LoadImage", "inputs": {"image": "referenz1_seed1059564922_00001_.png"}},
 "r10": {"class_type": "LoadImage", "inputs": {"image": "door_reader.png [output]"}},
 "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["1", 0], "prompt": "medium shot of an office", "negative_prompt": "extra people", "vae": ["2", 0], "resolution": 1024, "images.image_10": ["r10", 0], "images.image_1": ["r1", 0], "images.image_2": ["r2", 0]}},
 "c1": {"class_type": "LoadImage", "inputs": {"image": "leinwand.png"}},
 "c2": {"class_type": "VAEEncode", "inputs": {"pixels": ["c1", 0], "vae": ["2", 0]}},
 "6": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["c2", 0], "seed": 1865683568, "steps": 25, "cfg": 2.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
 "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "life3/keyframes/a1_s01_shot01_kf0"}}
})";

constexpr const char* kVoiceDesign = R"({
 "2": {"class_type": "FB_Qwen3TTSVoiceDesign", "inputs": {"text": "Ich bin ALIA, das KI-System dieses Bueros.", "instruct": "Frau, 30 Jahre, kalte Stimme", "seed": 236224718, "model_choice": "1.7B", "device": "auto", "precision": "bf16", "language": "German", "unload_model_after_generate": true}},
 "3": {"class_type": "SaveAudio", "inputs": {"audio": ["2", 0], "filename_prefix": "life3/stimmen/alia_probe_seed236224718"}}
})";

constexpr const char* kVoiceClone = R"({
 "1": {"class_type": "LoadAudio", "inputs": {"audio": "fabian_referenz.flac"}},
 "2": {"class_type": "FB_Qwen3TTSVoiceClone", "inputs": {"target_text": "Wir bleiben an unseren Plaetzen.", "seed": 1616559399, "ref_audio": ["1", 0], "ref_text": "Hallo und herzlich willkommen!", "model_choice": "1.7B", "device": "auto", "precision": "bf16", "language": "German", "unload_model_after_generate": true}},
 "3": {"class_type": "SaveAudio", "inputs": {"audio": ["2", 0], "filename_prefix": "life3/stimmen/fabian_s03"}}
})";

constexpr const char* kH3 = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "strength_model": 1.0}},
 "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "type": "minimax_h3", "device": "default"}},
 "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
 "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
 "10": {"class_type": "LoadImage", "inputs": {"image": "kf0.png"}},
 "11": {"class_type": "LoadImage", "inputs": {"image": "kf1.png"}},
 "20": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["3", 0], "vae": ["4", 0], "prompt": "five office workers at CRT screens", "width": 1344, "height": 768, "length": 120, "first_frame": ["10", 0]}},
 "21": {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": ["20", 0], "latent": ["20", 1], "frame_idx": -1, "vae": ["4", 0], "image": ["11", 0]}},
 "22": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["20", 0]}},
 "30": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["21", 0], "negative": ["22", 0], "latent_image": ["20", 1], "seed": 7, "steps": 8, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
 "31": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["30", 0]}},
 "32": {"class_type": "VAEDecode", "inputs": {"samples": ["31", 0], "vae": ["4", 0]}},
 "33": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["31", 1], "audio_vae": ["5", 0]}},
 "40": {"class_type": "SaveImage", "inputs": {"images": ["32", 0], "filename_prefix": "film/clips/a1_s01_shot01"}},
 "41": {"class_type": "SaveAudio", "inputs": {"audio": ["33", 0], "filename_prefix": "film/clips/a1_s01_shot01"}}
})";

// FilmStudio V100 h3_fortsetzung_anker.json (continuation from the last
// frames and the sound of the previous part), with BasicGuider +
// SamplerCustomAdvanced already rewritten to the KSampler "12" the adapter
// turns them into.
constexpr const char* kH3Anchor = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "strength_model": 1.0, "lora_name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"}},
 "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "type": "minimax", "device": "default"}},
 "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
 "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
 "6": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["3", 0], "vae": ["4", 0], "prompt": "STYLE\nSHOT 1: The scene continues seamlessly from the anchored opening clip ...", "width": 896, "height": 512, "length": 124}},
 "12": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["7", 0], "latent_image": ["6", 1], "seed": 43, "steps": 8, "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
 "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
 "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
 "15": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "film/video/roh/a1_s02_x05_h3/t1/f"}},
 "17": {"class_type": "SaveAudio", "inputs": {"audio": ["14", 0], "filename_prefix": "film/video/roh/a1_s02_x05_h3/t1/ton"}},
 "22": {"class_type": "LoadVideo", "inputs": {"file": "life3_film_a1_s02_x05_h3_anker1.mp4"}},
 "23": {"class_type": "GetVideoComponents", "inputs": {"video": ["22", 0]}},
 "7": {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": ["6", 0], "latent": ["6", 1], "frame_idx": 0, "vae": ["4", 0], "audio_vae": ["5", 0], "image": ["23", 0], "audio": ["23", 1]}}
})";

constexpr const char* kWan = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_ti2v_5B_fp16.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
 "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
 "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8}},
 "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "a fox in the snow"}},
 "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "blurry"}},
 "9": {"class_type": "LoadImage", "inputs": {"image": "fuchs.png"}},
 "55": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"vae": ["3", 0], "width": 1280, "height": 704, "length": 49, "batch_size": 1, "start_image": ["9", 0]}},
 "3s": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["55", 0], "seed": 42, "steps": 20, "cfg": 5, "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1}},
 "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3s", 0], "vae": ["3", 0]}},
 "57": {"class_type": "CreateVideo", "inputs": {"images": ["8", 0], "fps": 24}},
 "58": {"class_type": "SaveVideo", "inputs": {"video": ["57", 0], "filename_prefix": "video/wan", "format": "auto", "codec": "auto"}}
})";

// ACE-Step 1.5 as ComfyUI's audio_ace_step_1_5 template wires it (node
// signatures from /object_info on .56).
constexpr const char* kAceStep = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "acestep_v1.5_xl_turbo_bf16.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "qwen_0.6b_ace15.safetensors", "clip_name2": "qwen_4b_ace15.safetensors", "type": "ace", "device": "default"}},
 "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ace_1.5_vae.safetensors"}},
 "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3}},
 "5": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {"clip": ["2", 0], "tags": "90s anime opening, synth pop, female vocals", "lyrics": "[verse]\nNeon im Regen", "seed": 11, "bpm": 128, "duration": 45, "timesignature": "4", "language": "de", "keyscale": "A minor", "generate_audio_codes": true, "cfg_scale": 2.0, "temperature": 0.85, "top_p": 0.9, "top_k": 0, "min_p": 0.0}},
 "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
 "7": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {"seconds": 45, "batch_size": 1}},
 "8": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0], "seed": 11, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "9": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
 "10": {"class_type": "SaveAudio", "inputs": {"audio": ["9", 0], "filename_prefix": "film/musik/titel"}}
})";

// Wan 2.1 I2V 14B + Uni3C the ComfyUI-core way (WanUni3CControlnetApply on
// the model, WanImageToVideo for the start image); the camera comes from
// gufo's GufoCameraMove node.
constexpr const char* kUni3C = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "strength_model": 1.0}},
 "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp16.safetensors", "type": "wan", "device": "default"}},
 "4": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
 "5": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
 "6": {"class_type": "ModelPatchLoader", "inputs": {"name": "Wan21_Uni3C_controlnet_fp16.safetensors"}},
 "10": {"class_type": "LoadImage", "inputs": {"image": "kf_shot07.png"}},
 "11": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["5", 0], "image": ["10", 0], "crop": "none"}},
 "12": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "slow orbit around the office desk"}},
 "13": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": ""}},
 "14": {"class_type": "GufoCameraMove", "inputs": {"camera": "orbit_left", "speed": 0.5, "ease": "ease_in_out", "amount": -1}},
 "15": {"class_type": "WanUni3CControlnetApply", "inputs": {"model": ["2", 0], "model_patch": ["6", 0], "vae": ["4", 0], "render_video": ["14", 0], "strength": 1.0, "start_percent": 0.0, "end_percent": 1.0}},
 "16": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["15", 0], "shift": 5}},
 "20": {"class_type": "WanImageToVideo", "inputs": {"positive": ["12", 0], "negative": ["13", 0], "vae": ["4", 0], "width": 832, "height": 480, "length": 81, "batch_size": 1, "clip_vision_output": ["11", 0], "start_image": ["10", 0]}},
 "30": {"class_type": "KSampler", "inputs": {"model": ["16", 0], "positive": ["20", 0], "negative": ["20", 1], "latent_image": ["20", 2], "seed": 99, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "31": {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}},
 "32": {"class_type": "CreateVideo", "inputs": {"images": ["31", 0], "fps": 16}},
 "33": {"class_type": "SaveVideo", "inputs": {"video": ["32", 0], "filename_prefix": "film/kamera/shot07", "format": "mp4", "codec": "h264"}}
})";

// FilmStudio v100 H3 speaking shot as its pipeline sends it: ComfyUI's
// split sampler (SamplerCustomAdvanced) instead of a KSampler.
constexpr const char* kH3Custom = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "strength_model": 1.0, "lora_name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"}},
 "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "type": "minimax", "device": "default"}},
 "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
 "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
 "6": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["3", 0], "vae": ["4", 0], "prompt": "SHOT 1: The scene opens exactly on <Picture 1>. Lars speaks.", "width": 896, "height": 512, "length": 124, "first_frame": ["20", 0]}},
 "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
 "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
 "10": {"class_type": "BasicScheduler", "inputs": {"model": ["2", 0], "scheduler": "simple", "steps": 8, "denoise": 1.0}},
 "11": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["7", 0]}},
 "12": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["8", 0], "guider": ["11", 0], "sampler": ["9", 0], "sigmas": ["10", 0], "latent_image": ["6", 1]}},
 "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
 "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
 "15": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "film/video/roh/a1_s01_x01_h3/t0/f"}},
 "17": {"class_type": "SaveAudio", "inputs": {"audio": ["14", 0], "filename_prefix": "film/video/roh/a1_s01_x01_h3/t0/ton"}},
 "20": {"class_type": "LoadImage", "inputs": {"image": "a1_s01_shot01_kf0_896x512.png"}},
 "21": {"class_type": "LoadAudio", "inputs": {"audio": "life3_film_a1_s01_x01_h3.wav"}},
 "7": {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": ["6", 0], "latent": ["6", 1], "frame_idx": 0, "audio_vae": ["5", 0], "audio": ["21", 0]}}
})";

// FilmStudio v100: RIFE x4 of a finished shot (no generator in the graph).
constexpr const char* kRifeOnly = R"({
 "1": {"class_type": "LoadVideo", "inputs": {"file": "life3_film_a1_s01_x01_roh.mp4"}},
 "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
 "3": {"class_type": "RIFE VFI", "inputs": {"frames": ["2", 0], "ckpt_name": "rife49.pth", "clear_cache_after_n_frames": 10, "multiplier": 4, "fast_mode": false, "ensemble": true, "scale_factor": 1.0, "dtype": "float16", "torch_compile": false, "batch_size": 1}},
 "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "film/video/orig/a1_s01_x01/f"}},
 "5": {"class_type": "CreateVideo", "inputs": {"images": ["3", 0], "fps": 96.0, "codec": "h264"}},
 "6": {"class_type": "SaveVideo", "inputs": {"video": ["5", 0], "filename_prefix": "film/video/seg_a1_s01_x01", "format": "mp4"}}
})";

// Wan 2.1 VACE 14B (ComfyUI's WanVaceToVideo template with the lightx2v T2V
// LoRA): control video and an inverted mask video, a reference image,
// TrimVideoLatent between the sampler and the decode.
constexpr const char* kVace = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.1_vace_14B_fp16.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors", "strength_model": 1.0}},
 "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
 "4": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
 "5": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": 8.0}},
 "12": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "the woman walks through the office, cinematic"}},
 "13": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "blurry, static"}},
 "20": {"class_type": "LoadVideo", "inputs": {"file": "pose_shot03.mp4"}},
 "21": {"class_type": "GetVideoComponents", "inputs": {"video": ["20", 0]}},
 "26": {"class_type": "LoadVideo", "inputs": {"file": "mask_shot03.mp4"}},
 "22": {"class_type": "GetVideoComponents", "inputs": {"video": ["26", 0]}},
 "27": {"class_type": "ImageToMask", "inputs": {"channel": "red", "image": ["22", 0]}},
 "23": {"class_type": "InvertMask", "inputs": {"mask": ["27", 0]}},
 "24": {"class_type": "LoadImage", "inputs": {"image": "figur_anna.png"}},
 "25": {"class_type": "LoadImageMask", "inputs": {"image": "maske_tisch.png", "channel": "red"}},
 "30": {"class_type": "WanVaceToVideo", "inputs": {"positive": ["12", 0], "negative": ["13", 0], "vae": ["4", 0], "width": 832, "height": 480, "length": 81, "batch_size": 1, "strength": 0.8, "control_video": ["21", 0], "control_masks": ["23", 0], "reference_image": ["24", 0]}},
 "31": {"class_type": "KSampler", "inputs": {"model": ["5", 0], "positive": ["30", 0], "negative": ["30", 1], "latent_image": ["30", 2], "seed": 7, "steps": 4, "cfg": 1.0, "sampler_name": "lcm", "scheduler": "simple", "denoise": 1.0}},
 "32": {"class_type": "TrimVideoLatent", "inputs": {"samples": ["31", 0], "trim_amount": ["30", 3]}},
 "33": {"class_type": "VAEDecode", "inputs": {"samples": ["32", 0], "vae": ["4", 0]}},
 "34": {"class_type": "CreateVideo", "inputs": {"images": ["33", 0], "fps": 16}},
 "35": {"class_type": "SaveVideo", "inputs": {"video": ["34", 0], "filename_prefix": "film/vace/shot03", "format": "mp4", "codec": "h264"}}
})";

// The film workflow that stopped at "EmptyImage" (VACE keyframes, ComfyUI
// API format, 31 nodes) 1:1: control video = ImageBatch chain of
// LoadImage -> ImageScale keyframes and grey EmptyImage frames, mask =
// ImageBatch chain of black / white EmptyImage frames -> ImageToMask, two
// KSamplerAdvanced stages on the Wan 2.2 Fun VACE high / low noise models.
constexpr const char* kVaceKeyframes = R"json({
 "clip": {
  "class_type": "CLIPLoader",
  "inputs": {
   "clip_name": "umt5_xxl_fp16.safetensors",
   "type": "wan",
   "device": "default"
  }
 },
 "tp": {
  "class_type": "CLIPTextEncode",
  "inputs": {
   "clip": [
    "clip",
    0
   ],
   "text": "the camera stays static. A woman walks forward."
  }
 },
 "tn": {
  "class_type": "CLIPTextEncode",
  "inputs": {
   "clip": [
    "clip",
    0
   ],
   "text": "blurry"
  }
 },
 "vae": {
  "class_type": "VAELoader",
  "inputs": {
   "vae_name": "wan_2.1_vae.safetensors"
  }
 },
 "kf0": {
  "class_type": "LoadImage",
  "inputs": {
   "image": "kf0_beispiel.png"
  }
 },
 "kfs0": {
  "class_type": "ImageScale",
  "inputs": {
   "image": [
    "kf0",
    0
   ],
   "upscale_method": "lanczos",
   "width": 896,
   "height": 512,
   "crop": "center"
  }
 },
 "grau0": {
  "class_type": "EmptyImage",
  "inputs": {
   "width": 896,
   "height": 512,
   "batch_size": 23,
   "color": 8355711
  }
 },
 "kf1": {
  "class_type": "LoadImage",
  "inputs": {
   "image": "kf1_beispiel.png"
  }
 },
 "kfs1": {
  "class_type": "ImageScale",
  "inputs": {
   "image": [
    "kf1",
    0
   ],
   "upscale_method": "lanczos",
   "width": 896,
   "height": 512,
   "crop": "center"
  }
 },
 "grau1": {
  "class_type": "EmptyImage",
  "inputs": {
   "width": 896,
   "height": 512,
   "batch_size": 24,
   "color": 8355711
  }
 },
 "cb10": {
  "class_type": "ImageBatch",
  "inputs": {
   "image1": [
    "kfs0",
    0
   ],
   "image2": [
    "grau0",
    0
   ]
  }
 },
 "cb11": {
  "class_type": "ImageBatch",
  "inputs": {
   "image1": [
    "cb10",
    0
   ],
   "image2": [
    "kfs1",
    0
   ]
  }
 },
 "cb12": {
  "class_type": "ImageBatch",
  "inputs": {
   "image1": [
    "cb11",
    0
   ],
   "image2": [
    "grau1",
    0
   ]
  }
 },
 "m0": {
  "class_type": "EmptyImage",
  "inputs": {
   "width": 896,
   "height": 512,
   "batch_size": 1,
   "color": 0
  }
 },
 "m1": {
  "class_type": "EmptyImage",
  "inputs": {
   "width": 896,
   "height": 512,
   "batch_size": 23,
   "color": 16777215
  }
 },
 "m2": {
  "class_type": "EmptyImage",
  "inputs": {
   "width": 896,
   "height": 512,
   "batch_size": 1,
   "color": 0
  }
 },
 "m3": {
  "class_type": "EmptyImage",
  "inputs": {
   "width": 896,
   "height": 512,
   "batch_size": 24,
   "color": 16777215
  }
 },
 "mb17": {
  "class_type": "ImageBatch",
  "inputs": {
   "image1": [
    "m0",
    0
   ],
   "image2": [
    "m1",
    0
   ]
  }
 },
 "mb18": {
  "class_type": "ImageBatch",
  "inputs": {
   "image1": [
    "mb17",
    0
   ],
   "image2": [
    "m2",
    0
   ]
  }
 },
 "mb19": {
  "class_type": "ImageBatch",
  "inputs": {
   "image1": [
    "mb18",
    0
   ],
   "image2": [
    "m3",
    0
   ]
  }
 },
 "mask": {
  "class_type": "ImageToMask",
  "inputs": {
   "image": [
    "mb19",
    0
   ],
   "channel": "red"
  }
 },
 "vace": {
  "class_type": "WanVaceToVideo",
  "inputs": {
   "positive": [
    "tp",
    0
   ],
   "negative": [
    "tn",
    0
   ],
   "vae": [
    "vae",
    0
   ],
   "width": 896,
   "height": 512,
   "length": 49,
   "batch_size": 1,
   "strength": 1.0,
   "control_video": [
    "cb12",
    0
   ],
   "control_masks": [
    "mask",
    0
   ]
  }
 },
 "mh": {
  "class_type": "UNETLoader",
  "inputs": {
   "unet_name": "wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors",
   "weight_dtype": "default"
  }
 },
 "ml": {
  "class_type": "UNETLoader",
  "inputs": {
   "unet_name": "wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors",
   "weight_dtype": "default"
  }
 },
 "sh": {
  "class_type": "ModelSamplingSD3",
  "inputs": {
   "model": [
    "mh",
    0
   ],
   "shift": 8.0
  }
 },
 "sl": {
  "class_type": "ModelSamplingSD3",
  "inputs": {
   "model": [
    "ml",
    0
   ],
   "shift": 8.0
  }
 },
 "k1": {
  "class_type": "KSamplerAdvanced",
  "inputs": {
   "steps": 20,
   "cfg": 3.5,
   "sampler_name": "euler",
   "scheduler": "simple",
   "positive": [
    "vace",
    0
   ],
   "negative": [
    "vace",
    1
   ],
   "model": [
    "sh",
    0
   ],
   "add_noise": "enable",
   "noise_seed": 533302944,
   "latent_image": [
    "vace",
    2
   ],
   "start_at_step": 0,
   "end_at_step": 10,
   "return_with_leftover_noise": "enable"
  }
 },
 "k2": {
  "class_type": "KSamplerAdvanced",
  "inputs": {
   "steps": 20,
   "cfg": 3.5,
   "sampler_name": "euler",
   "scheduler": "simple",
   "positive": [
    "vace",
    0
   ],
   "negative": [
    "vace",
    1
   ],
   "model": [
    "sl",
    0
   ],
   "add_noise": "disable",
   "noise_seed": 0,
   "latent_image": [
    "k1",
    0
   ],
   "start_at_step": 10,
   "end_at_step": 10000,
   "return_with_leftover_noise": "disable"
  }
 },
 "trim": {
  "class_type": "TrimVideoLatent",
  "inputs": {
   "samples": [
    "k2",
    0
   ],
   "trim_amount": [
    "vace",
    3
   ]
  }
 },
 "dec": {
  "class_type": "VAEDecode",
  "inputs": {
   "samples": [
    "trim",
    0
   ],
   "vae": [
    "vae",
    0
   ]
  }
 },
 "save": {
  "class_type": "SaveImage",
  "inputs": {
   "images": [
    "dec",
    0
   ],
   "filename_prefix": "test/vace_schluessel"
  }
 }
})json";

// InfiniteTalk as ComfyUI's template wires it (WanInfiniteTalkToVideo with
// a DynamicCombo "mode"; its sub-inputs are named "mode.<input>").
constexpr const char* kTalk = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "strength_model": 1.0}},
 "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp16.safetensors", "type": "wan", "device": "default"}},
 "4": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
 "5": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
 "6": {"class_type": "ModelPatchLoader", "inputs": {"name": "wan2.1_infiniteTalk_single_fp16.safetensors"}},
 "7": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": "wav2vec2-chinese-base_fp16.safetensors"}},
 "8": {"class_type": "LoadAudio", "inputs": {"audio": "anna_satz1.wav"}},
 "9": {"class_type": "AudioEncoderEncode", "inputs": {"audio_encoder": ["7", 0], "audio": ["8", 0]}},
 "10": {"class_type": "LoadImage", "inputs": {"image": "anna_portrait.png"}},
 "11": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["5", 0], "image": ["10", 0], "crop": "none"}},
 "12": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "a woman talks calmly to the camera"}},
 "13": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": ""}},
 "16": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["20", 0], "shift": 5}},
 "20": {"class_type": "WanInfiniteTalkToVideo", "inputs": {"mode": "single_speaker", "model": ["2", 0], "model_patch": ["6", 0], "positive": ["12", 0], "negative": ["13", 0], "vae": ["4", 0], "width": 832, "height": 480, "length": 81, "clip_vision_output": ["11", 0], "start_image": ["10", 0], "audio_encoder_output_1": ["9", 0], "motion_frame_count": 9, "audio_scale": 1.0}},
 "30": {"class_type": "KSampler", "inputs": {"model": ["16", 0], "positive": ["20", 1], "negative": ["20", 2], "latent_image": ["20", 3], "seed": 7, "steps": 4, "cfg": 1.0, "sampler_name": "lcm", "scheduler": "simple", "denoise": 1.0}},
 "31": {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}},
 "32": {"class_type": "CreateVideo", "inputs": {"images": ["31", 0], "audio": ["8", 0], "fps": 25}},
 "33": {"class_type": "SaveVideo", "inputs": {"video": ["32", 0], "filename_prefix": "film/dialog/anna_s01", "format": "mp4", "codec": "h264"}}
})";

comfy::Plan PlanOf(const char* text) {
  comfy::Plan plan;
  comfy::PlanError error;
  const bool ok = comfy::BuildPlan(json::parse(text), {}, &plan, &error);
  if (!ok)
    std::cerr << "unexpected plan error: " << error.message << "\n";
  assert(ok);
  return plan;
}

/// The plan's warnings without the note that gufo runs its BF16 Qwen-Image
/// transformer for the fixtures' int8 UNETLoader file.
std::vector<std::string> OtherWarnings(const comfy::Plan& plan) {
  std::vector<std::string> out;
  for (const std::string& warning : plan.warnings)
    if (warning.find("weights are not loaded") == std::string::npos)
      out.push_back(warning);
  return out;
}

comfy::PlanError ErrorOf(const std::string& text) {
  comfy::Plan plan;
  comfy::PlanError error;
  assert(!comfy::BuildPlan(json::parse(text), {}, &plan, &error));
  return error;
}

void TestQwenTextToImage() {
  const comfy::Plan plan = PlanOf(kQwenText);
  assert(plan.tasks.size() == 1);
  const auto& task = plan.tasks[0];
  assert(task.node_id == "6");
  const auto& image = std::get<comfy::ImageTask>(task.spec);
  assert(image.model == "qwen-image-2.1");
  assert(image.prompt == "A single mature adult male office worker");
  assert(image.negative_prompt == "chibi, childlike");
  assert(image.seed == 1769224114ULL);
  assert(image.steps == 25);
  assert(image.size_source == comfy::SizeSource::kSquare);
  assert(image.resolution == 1024);
  assert(image.references.empty());
  assert(task.outputs.size() == 1);
  assert(task.outputs[0].node_id == "8");
  assert(task.outputs[0].kind == comfy::OutputKind::kImage);
  assert(task.outputs[0].prefix ==
         "life3_zukunftsszenarien/v100_lauf_kurz/stil/demo_personen_r1");
  // Negative prompt and cfg pass through as gufo fields.
  assert(image.guidance && *image.guidance == 2.5);
  assert(!image.strength);
  const json::Value extension = comfy::ImageExtension(image);
  assert(extension.member_str("negative_prompt") == "chibi, childlike");
  assert(extension.find("guidance")->as_double() == 2.5);
  assert(extension.find("strength") == nullptr);
}

void TestQwenReferencesAndCanvas() {
  const comfy::Plan plan = PlanOf(kQwenCanvas);
  const auto& image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert(image.size_source == comfy::SizeSource::kCanvas);
  assert((image.canvas == comfy::FileRef{"leinwand.png", "input"}));
  // images.image_N in N order, the "[output]" tag picks the folder.
  assert(image.references.size() == 3);
  assert(image.references[0].name == "referenz1_seed1141445518_00001_.png");
  assert(image.references[1].name == "referenz1_seed1059564922_00001_.png");
  assert((image.references[2] == comfy::FileRef{"door_reader.png", "output"}));
}

void TestVoiceDesign() {
  const comfy::Plan plan = PlanOf(kVoiceDesign);
  const auto& speech = std::get<comfy::SpeechTask>(plan.tasks[0].spec);
  assert(speech.model == "qwen3-tts-voicedesign");
  assert(speech.voice == "voice-design");
  assert(speech.language == "german");
  assert(speech.instructions == "Frau, 30 Jahre, kalte Stimme");
  assert(speech.seed && *speech.seed == 236224718ULL);
  assert(!speech.reference_audio);
  assert(plan.tasks[0].outputs[0].kind == comfy::OutputKind::kAudio);
}

void TestVoiceClone() {
  const comfy::Plan plan = PlanOf(kVoiceClone);
  const auto& speech = std::get<comfy::SpeechTask>(plan.tasks[0].spec);
  assert(speech.model == "qwen3-tts-clone");
  assert(speech.voice == "voice-clone");
  assert((speech.reference_audio ==
          comfy::FileRef{"fabian_referenz.flac", "input"}));
  assert(speech.reference_text == "Hallo und herzlich willkommen!");
  assert(!speech.embedding_only);
}

void TestH3() {
  const comfy::Plan plan = PlanOf(kH3);
  assert(plan.tasks.size() == 1);
  const auto& task = plan.tasks[0];
  assert(task.outputs.size() == 2);
  const auto& video = std::get<comfy::VideoTask>(task.spec);
  assert(video.family == comfy::VideoFamily::kH3);
  assert(video.model == "minimax-h3");
  assert(video.width == 1344 && video.height == 768);
  assert(video.frames == 124);  // 120 snapped up to 17k+5
  assert(video.steps == 8 && video.seed == 7);
  assert(video.lora == "turbo" && !video.lora_strength);
  assert((video.first_frame == comfy::FileRef{"kf0.png", "input"}));
  assert((video.last_frame == comfy::FileRef{"kf1.png", "input"}));
  const json::Value body = comfy::VideoRequestBody(video);
  assert(body.member_str("size") == "1344x768");
  assert(body.find("seconds")->as_double() == 4.833);
  // seconds and gufo.frames must agree after gufo's 24-fps alignment, short
  // FilmStudio shots (39, 73, 107 frames) included.
  for (const int frames : {39, 56, 73, 107, 124, 141, 243, 345}) {
    comfy::VideoTask longer = video;
    longer.frames = frames;
    const double seconds =
        comfy::VideoRequestBody(longer).find("seconds")->as_double();
    const int aligned =
        comfy::SnapH3Frames(static_cast<int>(std::ceil(seconds * 24)));
    assert(seconds > 1 && seconds <= 14.375 && aligned == frames);
  }
  std::string medium = kH3;
  medium.replace(medium.find("\"length\": 120"), 13, "\"length\": 73");
  assert(std::get<comfy::VideoTask>(PlanOf(medium.c_str()).tasks[0].spec)
             .frames == 73);
  std::string longer = kH3;
  longer.replace(longer.find("\"length\": 120"), 13, "\"length\": 400");
  assert(ErrorOf(longer).message.find("22..345") != std::string::npos);
  assert(body.find("gufo")->find("frames")->as_double() == 124);
  assert(body.find("gufo")->member_str("lora") == "turbo");
  assert(body.find("gufo")->find("steps")->as_double() == 8);
}

void TestH3Guides() {
  std::string text = kH3;
  // 832x512 (FilmStudio v100) is a gufo H3 size now.
  std::string resized = text;
  resized.replace(resized.find("\"width\": 1344"), 13, "\"width\": 832");
  resized.replace(resized.find("\"height\": 768"), 13, "\"height\": 512");
  const comfy::Plan plan = PlanOf(resized.c_str());
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.width == 832 && video.height == 512);
  assert(comfy::VideoRequestBody(video).member_str("size") == "832x512");
  // Sizes outside multiples of 32 / 256..1344 / 768x1344 pixels are a 400.
  std::string odd = resized;
  odd.replace(odd.find("\"width\": 832"), 12, "\"width\": 840");
  comfy::PlanError error = ErrorOf(odd);
  assert(error.message.find("840x512") != std::string::npos);
  assert(error.node_errors.find("20") != nullptr);
  std::string big = text;
  big.replace(big.find("\"height\": 768"), 13, "\"height\": 1344");
  assert(ErrorOf(big).message.find("1344x1344") != std::string::npos);
  // Intermediate guide images become gufo.keyframes; an audio guide becomes
  // the audio reference.
  std::string guides = text;
  guides.replace(
      guides.find(" \"30\": {"), 8,
      " \"12\": {\"class_type\": \"LoadImage\", \"inputs\": "
      "{\"image\": \"mid.png\"}},\n"
      " \"13\": {\"class_type\": \"LoadAudio\", \"inputs\": "
      "{\"audio\": \"zeile.flac\"}},\n"
      " \"23\": {\"class_type\": \"MiniMaxH3AddGuide\", \"inputs\": "
      "{\"positive\": [\"21\", 0], \"latent\": [\"20\", 1], "
      "\"frame_idx\": 60, \"vae\": [\"4\", 0], \"image\": [\"12\", 0]}},\n"
      " \"24\": {\"class_type\": \"MiniMaxH3AddGuide\", \"inputs\": "
      "{\"positive\": [\"23\", 0], \"latent\": [\"20\", 1], "
      "\"frame_idx\": 0, \"audio_vae\": [\"5\", 0], \"audio\": [\"13\", 0]}},\n"
      " \"30\": {");
  guides.replace(guides.find("\"positive\": [\"21\", 0], \"negative\""), 21,
                 "\"positive\": [\"24\", 0]");
  const comfy::Plan guided = PlanOf(guides.c_str());
  const auto& h3 = std::get<comfy::VideoTask>(guided.tasks[0].spec);
  assert(h3.keyframes.size() == 1 && h3.keyframes[0].first == 60);
  assert((h3.keyframes[0].second == comfy::FileRef{"mid.png", "input"}));
  assert((h3.first_frame == comfy::FileRef{"kf0.png", "input"}));
  assert((h3.last_frame == comfy::FileRef{"kf1.png", "input"}));
  assert((h3.audio_reference == comfy::FileRef{"zeile.flac", "input"}));
  assert(h3.audio_reference_frame == 0);
  assert(comfy::VideoRequestBody(h3)
             .find("gufo")
             ->find("audio_reference_frame")
             ->as_double() == 0);
}

void TestH3CustomSampler() {
  // SamplerCustomAdvanced <- RandomNoise / KSamplerSelect / BasicScheduler /
  // BasicGuider folds into the KSampler path: same task as a KSampler graph.
  const comfy::Plan plan = PlanOf(kH3Custom);
  assert(plan.tasks.size() == 1);
  const auto& task = plan.tasks[0];
  assert(task.node_id == "12" && task.class_type == "KSampler");
  assert(task.outputs.size() == 2);
  const auto& video = std::get<comfy::VideoTask>(task.spec);
  assert(video.family == comfy::VideoFamily::kH3);
  assert(video.width == 896 && video.height == 512 && video.frames == 124);
  assert(video.steps == 8 && video.seed == 42);
  assert(video.lora == "turbo" && !video.lora_strength);
  assert((video.first_frame ==
          comfy::FileRef{"a1_s01_shot01_kf0_896x512.png", "input"}));
  assert((video.audio_reference ==
          comfy::FileRef{"life3_film_a1_s01_x01_h3.wav", "input"}));
  assert(video.audio_reference_frame == 0);
  assert(plan.warnings.empty());
  // The second output (denoised_output) is the same latent for gufo.
  std::string denoised = kH3Custom;
  denoised.replace(denoised.find("\"samples\": [\"12\", 0], \"vae\": [\"4\", 0]"),
                   37, "\"samples\": [\"12\", 1], \"vae\": [\"4\", 0]");
  assert(PlanOf(denoised.c_str()).tasks.size() == 1);
  // CFGGuider carries positive, negative and cfg.
  std::string cfg = kH3Custom;
  cfg.replace(cfg.find("\"BasicGuider\", \"inputs\": {\"model\": [\"2\", 0], "
                       "\"conditioning\": [\"7\", 0]}"),
              70,
              "\"CFGGuider\", \"inputs\": {\"model\": [\"2\", 0], "
              "\"positive\": [\"7\", 0], \"negative\": [\"6\", 0], "
              "\"cfg\": 1.0}");
  assert(std::get<comfy::VideoTask>(PlanOf(cfg.c_str()).tasks[0].spec).seed ==
         42);
  // Other noise sources are not gufo's sampler.
  std::string disabled = kH3Custom;
  disabled.replace(disabled.find("\"RandomNoise\", \"inputs\": {\"noise_seed\": 42}"),
                   43, "\"DisableNoise\", \"inputs\": {}");
  const comfy::PlanError error = ErrorOf(disabled);
  assert(error.message.find("RandomNoise") != std::string::npos);
  assert(error.node_errors.find("12") != nullptr);
  // A scheduler on another model than the guider is a warning.
  std::string other = kH3Custom;
  other.replace(other.find("\"BasicScheduler\", \"inputs\": {\"model\": [\"2\", 0]"),
                46, "\"BasicScheduler\", \"inputs\": {\"model\": [\"1\", 0]");
  assert(PlanOf(other.c_str()).warnings.size() == 1);
}

void TestH3AnchorClip() {
  const comfy::Plan plan = PlanOf(kH3Anchor);
  assert(plan.tasks.size() == 1);
  const auto& h3 = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(h3.family == comfy::VideoFamily::kH3);
  assert(h3.width == 896 && h3.height == 512 && h3.frames == 124);
  assert(h3.seed == 43 && h3.steps == 8 && h3.lora == "turbo");
  // The frame batch is a guide clip, its soundtrack the audio guide; no
  // first frame, no single guide images.
  const comfy::FileRef clip{"life3_film_a1_s02_x05_h3_anker1.mp4", "input"};
  assert((h3.guide_video == clip));
  assert(h3.guide_video_frame == 0);
  assert((h3.audio_reference == clip));
  assert(h3.audio_reference_from_video);
  assert(h3.audio_reference_frame == 0);
  assert(!h3.first_frame && !h3.last_frame && h3.keyframes.empty());
  const json::Value body = comfy::VideoRequestBody(h3);
  assert(body.member_str("size") == "896x512");
  assert(body.find("gufo")->find("frames")->as_double() == 124);
  assert(body.find("gufo")->find("audio_reference_frame")->as_double() == 0);
  // The executor adds the clip and the extracted sound itself.
  assert(body.find("gufo")->find("guide_video") == nullptr);
  assert(body.find("audio_reference") == nullptr);

  const auto replaced = [](std::string text, const std::string& from,
                           const std::string& to) {
    const std::size_t at = text.find(from);
    assert(at != std::string::npos);
    text.replace(at, from.size(), to);
    return text;
  };
  // A guide clip at a later frame keeps its frame_idx.
  const std::string later =
      replaced(kH3Anchor, "\"frame_idx\": 0", "\"frame_idx\": 17");
  const comfy::Plan shifted = PlanOf(later.c_str());
  const auto& moved = std::get<comfy::VideoTask>(shifted.tasks[0].spec);
  assert(moved.guide_video_frame == 17 && moved.audio_reference_frame == 17);

  // Without the audio link only the picture is anchored.
  const std::string silent =
      replaced(kH3Anchor, ", \"audio\": [\"23\", 1]", "");
  const comfy::Plan quiet = PlanOf(silent.c_str());
  const auto& mute = std::get<comfy::VideoTask>(quiet.tasks[0].spec);
  assert(mute.guide_video && !mute.audio_reference);

  // The images come from output 0, the audio from output 1.
  const std::string swapped = replaced(kH3Anchor, "\"image\": [\"23\", 0]",
                                       "\"image\": [\"23\", 1]");
  assert(ErrorOf(swapped).message.find("output 0 of GetVideoComponents") !=
         std::string::npos);
  // GetVideoComponents must read a LoadVideo.
  const std::string loose =
      replaced(kH3Anchor, "\"LoadVideo\", \"inputs\": {\"file\"",
               "\"LoadImage\", \"inputs\": {\"image\"");
  assert(ErrorOf(loose).message.find("LoadVideo") != std::string::npos);
}

void TestWan() {
  const comfy::Plan plan = PlanOf(kWan);
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kWan);
  assert(video.model == "wan2.2-ti2v-5b");
  assert(video.prompt == "a fox in the snow");
  assert(video.negative_prompt == "blurry");
  assert(video.frames == 49 && video.guidance && *video.guidance == 5);
  assert((video.first_frame == comfy::FileRef{"fuchs.png", "input"}));
  assert(plan.tasks[0].outputs[0].kind == comfy::OutputKind::kVideo);
  const json::Value body = comfy::VideoRequestBody(video);
  assert(body.find("seconds")->as_double() == 2);
  assert(body.member_str("size") == "1280x704");
  assert(body.find("gufo")->member_str("negative_prompt") == "blurry");
}

void TestUnsupportedNodes() {
  const comfy::PlanError error = ErrorOf(R"({
    "1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
    "2": {"class_type": "EmptyAceStepLatentAudio", "inputs": {"seconds": 30, "batch_size": 1}},
    "3": {"class_type": "FancyUpscaler", "inputs": {"image": ["2", 0]}},
    "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "x"}}
  })");
  assert(error.type == "invalid_prompt");
  assert(error.message.find("FancyUpscaler (node 3)") != std::string::npos);
  assert(error.message.find("EmptyAceStepLatentAudio (node 2)") !=
         std::string::npos);
  assert(error.message.find("port pending") != std::string::npos);
  assert(error.node_errors.find("2") != nullptr);
  assert(error.node_errors.find("3") != nullptr);
  assert(
      ErrorOf(
          R"({"1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}}})")
          .type == "prompt_no_outputs");
  // A save without a generator upstream.
  assert(ErrorOf(R"({
    "1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
    "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}}
  })")
             .type == "prompt_outputs_failed_validation");
}

std::string Replace(std::string text, const std::string& from,
                    const std::string& to) {
  const auto at = text.find(from);
  assert(at != std::string::npos);
  return text.replace(at, from.size(), to);
}

void TestImageToImage() {
  const std::string text =
      Replace(kQwenCanvas, "\"denoise\": 1.0", "\"denoise\": 0.6");
  const comfy::Plan plan = PlanOf(text.c_str());
  const auto& image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert(image.strength && *image.strength == 0.6);
  assert((image.init_image == comfy::FileRef{"leinwand.png", "input"}));
  assert(comfy::ImageExtension(image).find("strength")->as_double() == 0.6);
  // Without an encoded image there is nothing to denoise from.
  const comfy::PlanError error =
      ErrorOf(Replace(kQwenText, "\"denoise\": 1.0", "\"denoise\": 0.5"));
  assert(error.message.find("VAEEncode") != std::string::npos);
}

void TestMaskedEdit() {
  // SetLatentNoiseMask over VAEEncode with LoadImage's MASK output (1 - alpha).
  std::string text = Replace(
      kQwenCanvas, R"("latent_image": ["c2", 0])", R"("latent_image": ["m2", 0])");
  text = Replace(text, R"( "6": {"class_type": "KSampler")",
                 R"( "m1": {"class_type": "LoadImage", "inputs": {"image": "maske.png"}},
 "m2": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["c2", 0], "mask": ["m1", 1]}},
 "6": {"class_type": "KSampler")");
  comfy::Plan plan = PlanOf(text.c_str());
  auto image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert((image.init_image == comfy::FileRef{"leinwand.png", "input"}));
  assert((image.mask == comfy::FileRef{"maske.png", "input"}));
  assert(image.mask_channel == "alpha" && !image.strength);
  assert(image.size_source == comfy::SizeSource::kCanvas);
  assert(image.references.size() == 3);
  assert(comfy::ImageExtension(image).member_str("mask_channel") == "alpha");
  // With denoise below 1 both apply.
  plan = PlanOf(Replace(text, "\"denoise\": 1.0", "\"denoise\": 0.7").c_str());
  image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert(image.strength && *image.strength == 0.7 && !image.mask.name.empty());
  // LoadImageMask keeps its channel.
  const std::string channel = Replace(
      text, R"("m1": {"class_type": "LoadImage", "inputs": {"image": "maske.png"}})",
      R"("m1": {"class_type": "LoadImageMask", "inputs": {"image": "maske.png [output]", "channel": "red"}})");
  plan = PlanOf(Replace(channel, R"("mask": ["m1", 1])", R"("mask": ["m1", 0])")
                    .c_str());
  image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert((image.mask == comfy::FileRef{"maske.png", "output"}));
  assert(image.mask_channel == "red");
  assert(ErrorOf(Replace(channel, "\"red\"", "\"hue\"")).message.find(
             "channel") != std::string::npos);
  // The IMAGE output of LoadImage is not a mask.
  assert(ErrorOf(Replace(text, R"("mask": ["m1", 1])", R"("mask": ["m1", 0])"))
             .message.find("MASK output") != std::string::npos);
  // SetLatentNoiseMask needs an encoded image.
  assert(ErrorOf(Replace(text, R"("samples": ["c2", 0])",
                         R"("samples": ["5", 2])"))
             .message.find("VAEEncode") != std::string::npos);
  // VAEEncodeForInpaint maps to the same masked edit, with a warning.
  std::string inpaint = Replace(
      kQwenCanvas,
      R"("c2": {"class_type": "VAEEncode", "inputs": {"pixels": ["c1", 0], "vae": ["2", 0]}})",
      R"("c2": {"class_type": "VAEEncodeForInpaint", "inputs": {"pixels": ["c1", 0], "vae": ["2", 0], "mask": ["c1", 1], "grow_mask_by": 6}})");
  plan = PlanOf(inpaint.c_str());
  image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert((image.init_image == comfy::FileRef{"leinwand.png", "input"}));
  assert((image.mask == comfy::FileRef{"leinwand.png", "input"}));
  assert(image.mask_channel == "alpha");
  bool warned = false;
  for (const auto& warning : plan.warnings)
    warned |= warning.find("VAEEncodeForInpaint") != std::string::npos;
  assert(warned);
  // Without a mask nothing changes: no mask field is sent.
  plan = PlanOf(kQwenCanvas);
  image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert(image.mask.name.empty() && image.init_image.name.empty());
  assert(comfy::ImageExtension(image).find("mask_channel") == nullptr);
}

// Regional prompts (M12): the five-column desk scene at 1344x768. The global
// prompt carries place and light, each person a column mask (268 px wide,
// y 250..768) with an own reference, the door a background mask in column
// 5; ConditioningCombine chains them in order.
constexpr const char* kQwenRegional = R"({
 "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
 "2": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
 "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
 "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["1", 0], "prompt": "office at night, five desks", "negative_prompt": "extra person", "vae": ["2", 0], "resolution": 1024}},
 "i1": {"class_type": "LoadImage", "inputs": {"image": "ref_a.png"}},
 "i2": {"class_type": "LoadImage", "inputs": {"image": "ref_b.png"}},
 "i3": {"class_type": "LoadImage", "inputs": {"image": "ref_c.png"}},
 "i4": {"class_type": "LoadImage", "inputs": {"image": "ref_d.png"}},
 "i5": {"class_type": "LoadImage", "inputs": {"image": "ref_e.png [output]"}},
 "t1": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["1", 0], "prompt": "A (<image1>) sits at the desk", "negative_prompt": "", "vae": ["2", 0], "resolution": 1024, "images.image_1": ["i1", 0]}},
 "t2": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["1", 0], "prompt": "B (<image1>) sits at the desk", "negative_prompt": "", "vae": ["2", 0], "resolution": 1024, "images.image_1": ["i2", 0]}},
 "t3": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["1", 0], "prompt": "C (<image1>) sits at the desk", "negative_prompt": "", "vae": ["2", 0], "resolution": 1024, "images.image_1": ["i3", 0]}},
 "t4": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["1", 0], "prompt": "D (<image1>) sits at the desk", "negative_prompt": "", "vae": ["2", 0], "resolution": 1024, "images.image_1": ["i4", 0]}},
 "t5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["1", 0], "prompt": "E (<image1>) sits at the desk", "negative_prompt": "", "vae": ["2", 0], "resolution": 1024, "images.image_1": ["i5", 0]}},
 "td": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": "a closed office door with an access panel"}},
 "m0": {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": 1344, "height": 768}},
 "mc": {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": 268, "height": 518}},
 "md": {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": 268, "height": 250}},
 "k1": {"class_type": "MaskComposite", "inputs": {"destination": ["m0", 0], "source": ["mc", 0], "x": 0, "y": 250, "operation": "add"}},
 "k2": {"class_type": "MaskComposite", "inputs": {"destination": ["m0", 0], "source": ["mc", 0], "x": 268, "y": 250, "operation": "add"}},
 "k3": {"class_type": "MaskComposite", "inputs": {"destination": ["m0", 0], "source": ["mc", 0], "x": 536, "y": 250, "operation": "add"}},
 "k4": {"class_type": "MaskComposite", "inputs": {"destination": ["m0", 0], "source": ["mc", 0], "x": 804, "y": 250, "operation": "add"}},
 "k5": {"class_type": "MaskComposite", "inputs": {"destination": ["m0", 0], "source": ["mc", 0], "x": 1072, "y": 250, "operation": "add"}},
 "kd": {"class_type": "MaskComposite", "inputs": {"destination": ["m0", 0], "source": ["md", 0], "x": 1072, "y": 0, "operation": "add"}},
 "s1": {"class_type": "ConditioningSetMask", "inputs": {"conditioning": ["t1", 0], "mask": ["k1", 0], "strength": 1.0, "set_cond_area": "mask bounds"}},
 "s2": {"class_type": "ConditioningSetMask", "inputs": {"conditioning": ["t2", 0], "mask": ["k2", 0], "strength": 1.0, "set_cond_area": "mask bounds"}},
 "s3": {"class_type": "ConditioningSetMask", "inputs": {"conditioning": ["t3", 0], "mask": ["k3", 0], "strength": 1.0, "set_cond_area": "mask bounds"}},
 "s4": {"class_type": "ConditioningSetMask", "inputs": {"conditioning": ["t4", 0], "mask": ["k4", 0], "strength": 1.0, "set_cond_area": "mask bounds"}},
 "s5": {"class_type": "ConditioningSetMask", "inputs": {"conditioning": ["t5", 0], "mask": ["k5", 0], "strength": 1.0, "set_cond_area": "mask bounds"}},
 "sd": {"class_type": "ConditioningSetMask", "inputs": {"conditioning": ["td", 0], "mask": ["kd", 0], "strength": 0.8, "set_cond_area": "default"}},
 "c1": {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["5", 0], "conditioning_2": ["s1", 0]}},
 "c2": {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["c1", 0], "conditioning_2": ["s2", 0]}},
 "c3": {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["c2", 0], "conditioning_2": ["s3", 0]}},
 "c4": {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["s4", 0], "conditioning_2": ["s5", 0]}},
 "c5": {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["c3", 0], "conditioning_2": ["c4", 0]}},
 "c6": {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["c5", 0], "conditioning_2": ["sd", 0]}},
 "e": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1344, "height": 768, "batch_size": 1}},
 "6": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["c6", 0], "negative": ["5", 1], "latent_image": ["e", 0], "seed": 22101, "steps": 25, "cfg": 2.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
 "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "regional/desk"}}
})";

float MaskAt(const comfy::MaskGrid& mask, int x, int y) {
  return mask.values[static_cast<std::size_t>(y) * mask.width + x];
}

void TestRegionalMasks() {
  const comfy::Plan plan = PlanOf(kQwenRegional);
  assert(plan.tasks.size() == 1);
  const auto& image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  // The conditioning without area or mask is the base prompt.
  assert(image.prompt == "office at night, five desks");
  assert(image.references.empty());
  assert(image.base_strength == 1.0);
  assert(image.negative_prompt == "extra person");
  assert(image.size_source == comfy::SizeSource::kExplicit);
  assert(image.width == 1344 && image.height == 768);
  // Combine order: conditioning_1 before conditioning_2, depth first.
  assert(image.regions.size() == 6);
  const char* prompts[] = {"A", "B", "C", "D", "E"};
  for (int i = 0; i < 5; ++i) {
    const auto& region = image.regions[i];
    assert(region.prompt.starts_with(prompts[i]));
    assert(region.node == "t" + std::to_string(i + 1));
    assert(region.references.size() == 1);
    assert(region.mask && !region.mask_file && region.mask_bounds);
    assert(region.strength == 1.0 && !region.area);
    const auto& mask = *region.mask;
    assert(mask.width == 1344 && mask.height == 768);
    // Column i: x in [268 i, 268 i + 268), y in [250, 768).
    const int left = 268 * i;
    assert(MaskAt(mask, left, 250) == 1.0F);
    assert(MaskAt(mask, left + 267, 767) == 1.0F);
    assert(MaskAt(mask, left, 249) == 0.0F);
    if (left + 268 < 1344)
      assert(MaskAt(mask, left + 268, 400) == 0.0F);
    if (left > 0)
      assert(MaskAt(mask, left - 1, 400) == 0.0F);
  }
  assert((image.regions[4].references[0] ==
          comfy::FileRef{"ref_e.png", "output"}));
  const auto& door = image.regions[5];
  assert(door.prompt == "a closed office door with an access panel");
  assert(door.references.empty() && !door.mask_bounds);
  assert(door.strength == 0.8);
  assert(MaskAt(*door.mask, 1100, 100) == 1.0F &&
         MaskAt(*door.mask, 1100, 250) == 0.0F &&
         MaskAt(*door.mask, 1071, 100) == 0.0F);

  // gufo.regions: one mask byte per 16-pixel cell (84 x 48 here).
  const json::Value field = comfy::ImageRegionsField(image, 1344, 768);
  assert(field.size() == 6);
  const json::Value& first = field.items()[0];
  assert(first.member_str("prompt") == "A (<image1>) sits at the desk");
  assert(first.find("strength") == nullptr);
  assert(first.find("mask_bounds")->as_bool());
  assert(first.find("images")->as_double() == 1);
  const json::Value* mask = first.find("mask");
  assert(mask->find("width")->as_double() == 84);
  assert(mask->find("height")->as_double() == 48);
  const auto bytes = comfy::Base64Decode(mask->member_str("data"));
  assert(bytes && bytes->size() == 84 * 48);
  const auto cell = [&](int row, int col) {
    return static_cast<unsigned char>((*bytes)[row * 84 + col]);
  };
  // Cell rows 0..14 end at y 240 (outside); row 15 (240..256) is 6/16
  // inside; column 16 (256..272) is 12/16 inside.
  assert(cell(10, 0) == 0 && cell(20, 0) == 255 && cell(47, 15) == 255);
  assert(cell(15, 0) == 96);   // round(6 / 16 * 255)
  assert(cell(20, 16) == 191);  // round(12 / 16 * 255)
  assert(cell(20, 17) == 0);
  const json::Value& door_field = field.items()[5];
  assert(door_field.find("strength")->as_double() == 0.8);
  assert(door_field.find("images") == nullptr);
  assert(door_field.find("mask_bounds") == nullptr);
  // The base prompt keeps its gufo fields.
  assert(comfy::ImageExtension(image).member_str("negative_prompt") ==
         "extra person");
}

void TestRegionalAreas() {
  // SetArea (pixels), SetAreaPercentage (fractions), SetAreaStrength and a
  // SetMask on top of an area.
  std::string text = Replace(
      kQwenRegional,
      R"("s1": {"class_type": "ConditioningSetMask", "inputs": {"conditioning": ["t1", 0], "mask": ["k1", 0], "strength": 1.0, "set_cond_area": "mask bounds"}})",
      R"("s1": {"class_type": "ConditioningSetArea", "inputs": {"conditioning": ["t1", 0], "width": 268, "height": 518, "x": 0, "y": 250, "strength": 1.5}})");
  text = Replace(
      text,
      R"("s2": {"class_type": "ConditioningSetMask", "inputs": {"conditioning": ["t2", 0], "mask": ["k2", 0], "strength": 1.0, "set_cond_area": "mask bounds"}})",
      R"("s2": {"class_type": "ConditioningSetAreaPercentage", "inputs": {"conditioning": ["t2", 0], "width": 0.2, "height": 0.675, "x": 0.2, "y": 0.325, "strength": 1.0}})");
  // s3: an area, then a mask with "mask bounds" (bounds win), strengths
  // multiply; s4: mask bounds first, then an area (ComfyUI resets bounds).
  text = Replace(
      text,
      R"("s3": {"class_type": "ConditioningSetMask", "inputs": {"conditioning": ["t3", 0], "mask": ["k3", 0], "strength": 1.0, "set_cond_area": "mask bounds"}})",
      R"("a3": {"class_type": "ConditioningSetArea", "inputs": {"conditioning": ["t3", 0], "width": 400, "height": 768, "x": 500, "y": 0, "strength": 2.0}},
 "s3": {"class_type": "ConditioningSetMask", "inputs": {"conditioning": ["a3", 0], "mask": ["k3", 0], "strength": 0.5, "set_cond_area": "mask bounds"}})");
  text = Replace(
      text,
      R"("s4": {"class_type": "ConditioningSetMask", "inputs": {"conditioning": ["t4", 0], "mask": ["k4", 0], "strength": 1.0, "set_cond_area": "mask bounds"}})",
      R"("b4": {"class_type": "ConditioningSetMask", "inputs": {"conditioning": ["t4", 0], "mask": ["k4", 0], "strength": 1.0, "set_cond_area": "mask bounds"}},
 "s4": {"class_type": "ConditioningSetArea", "inputs": {"conditioning": ["b4", 0], "width": 268, "height": 518, "x": 804, "y": 250, "strength": 1.0}})");
  // The base prompt at strength 0.5 through ConditioningSetAreaStrength.
  text = Replace(text,
                 R"("conditioning_1": ["5", 0], "conditioning_2": ["s1", 0])",
                 R"("conditioning_1": ["g", 0], "conditioning_2": ["s1", 0])");
  text = Replace(
      text, R"( "e": {"class_type")",
      R"( "g": {"class_type": "ConditioningSetAreaStrength", "inputs": {"conditioning": ["5", 0], "strength": 0.5}},
 "e": {"class_type")");
  const comfy::Plan plan = PlanOf(text.c_str());
  const auto& image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert(image.base_strength == 0.5);
  assert(image.regions.size() == 6);
  const auto& a = image.regions[0];
  assert(a.area && !a.area_fraction && !a.mask && a.strength == 1.5);
  assert(((*a.area) == std::array<double, 4>{0, 250, 268, 518}));
  const auto& b = image.regions[1];
  assert(b.area && b.area_fraction && !b.mask);
  assert(((*b.area) == std::array<double, 4>{0.2, 0.325, 0.2, 0.675}));
  const auto& c = image.regions[2];
  assert(c.area && c.mask && c.mask_bounds && c.strength == 1.0);
  const auto& d = image.regions[3];
  assert(d.area && d.mask && !d.mask_bounds);
  const json::Value field = comfy::ImageRegionsField(image, 1344, 768);
  const json::Value* area = field.items()[1].find("area");
  assert(area->member_str("unit") == "fraction" &&
         area->find("x")->as_double() == 0.2);
  assert(field.items()[0].find("area")->member_str("unit") == "px");
  assert(field.items()[0].find("strength")->as_double() == 1.5);
  // Values outside the node ranges fail validation.
  assert(ErrorOf(Replace(text, R"("width": 0.2, "height": 0.675)",
                         R"("width": 1.2, "height": 0.675)"))
             .message.find("0..1") != std::string::npos);
  assert(ErrorOf(Replace(text, R"("strength": 1.5)", R"("strength": 11)"))
             .message.find("strength") != std::string::npos);
}

void TestRegionalCombineAndErrors() {
  // Without a conditioning free of area and mask the base prompt is empty
  // and weighs nothing.
  std::string text = Replace(
      kQwenRegional,
      R"("conditioning_1": ["5", 0], "conditioning_2": ["s1", 0])",
      R"("conditioning_1": ["s1", 0], "conditioning_2": ["s1", 0])");
  comfy::Plan plan = PlanOf(text.c_str());
  auto image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert(image.prompt.empty() && image.base_strength == 0);
  assert(image.regions.size() == 7);
  assert(image.regions[0].prompt == image.regions[1].prompt);
  // A second plain conditioning is a full-image region.
  text = Replace(kQwenRegional,
                 R"("conditioning_1": ["c5", 0], "conditioning_2": ["sd", 0])",
                 R"("conditioning_1": ["c5", 0], "conditioning_2": ["td", 0])");
  plan = PlanOf(text.c_str());
  image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert(image.regions.size() == 6 && !image.regions[5].mask &&
         !image.regions[5].area);
  assert(image.prompt == "office at night, five desks");
  // A combined negative is not supported.
  assert(ErrorOf(Replace(kQwenRegional, R"("negative": ["5", 1])",
                         R"("negative": ["c1", 0])"))
             .message.find("positive input only") != std::string::npos);
  // ZeroOut inside the regional positive.
  assert(
      ErrorOf(Replace(
                  Replace(kQwenRegional,
                          R"("conditioning_1": ["5", 0], "conditioning_2": ["s1", 0])",
                          R"("conditioning_1": ["z", 0], "conditioning_2": ["s1", 0])"),
                  R"( "e": {"class_type")",
                  R"( "z": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
 "e": {"class_type")"))
          .message.find("ConditioningZeroOut") != std::string::npos);
  // Bad set_cond_area and composite operation.
  assert(ErrorOf(Replace(kQwenRegional, R"("set_cond_area": "default")",
                         R"("set_cond_area": "bounds")"))
             .message.find("set_cond_area") != std::string::npos);
  assert(ErrorOf(Replace(kQwenRegional,
                         R"("x": 0, "y": 250, "operation": "add")",
                         R"("x": 0, "y": 250, "operation": "divide")"))
             .message.find("operation") != std::string::npos);
  // Mask files: LoadImageMask keeps its channel, InvertMask flips it; a
  // file cannot be composited.
  text = Replace(
      kQwenRegional,
      R"("conditioning": ["td", 0], "mask": ["kd", 0])",
      R"("conditioning": ["td", 0], "mask": ["fi", 0])");
  text = Replace(
      text, R"( "e": {"class_type")",
      R"( "f": {"class_type": "LoadImageMask", "inputs": {"image": "door_mask.png", "channel": "red"}},
 "fi": {"class_type": "InvertMask", "inputs": {"mask": ["f", 0]}},
 "e": {"class_type")");
  plan = PlanOf(text.c_str());
  image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  const auto& door = image.regions[5];
  assert(!door.mask && door.mask_file && door.mask_invert);
  assert((*door.mask_file == comfy::FileRef{"door_mask.png", "input"}));
  assert(door.mask_channel == "red");
  const json::Value field = comfy::ImageRegionsField(image, 1344, 768);
  assert(field.items()[5].member_str("mask_channel") == "red");
  assert(field.items()[5].find("mask_invert")->as_bool());
  assert(field.items()[5].find("mask") == nullptr);
  std::string composite = Replace(
      kQwenRegional, R"("destination": ["m0", 0], "source": ["md", 0])",
      R"("destination": ["m0", 0], "source": ["f", 0])");
  composite = Replace(
      composite, R"( "e": {"class_type")",
      R"( "f": {"class_type": "LoadImageMask", "inputs": {"image": "door_mask.png", "channel": "red"}},
 "e": {"class_type")");
  assert(ErrorOf(composite).message.find("mask files") != std::string::npos);
  // InvertMask of a computed mask is computed.
  text = Replace(kQwenRegional,
                 R"("conditioning": ["td", 0], "mask": ["kd", 0])",
                 R"("conditioning": ["td", 0], "mask": ["iv", 0])");
  text = Replace(text, R"( "e": {"class_type")",
                 R"( "iv": {"class_type": "InvertMask", "inputs": {"mask": ["kd", 0]}},
 "e": {"class_type")");
  plan = PlanOf(text.c_str());
  image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert(MaskAt(*image.regions[5].mask, 1100, 100) == 0.0F &&
         MaskAt(*image.regions[5].mask, 10, 700) == 1.0F);
}

void TestMaskComposite() {
  comfy::MaskGrid destination{4, 3, std::vector<float>(12, 0.5F)};
  comfy::MaskGrid source{2, 2, {1.0F, 0.25F, 0.0F, 0.75F}};
  // Placed at (3, 2): only the top-left source pixel is visible.
  auto out = comfy::CompositeMask(destination, source, 3, 2, "add");
  assert(MaskAt(out, 3, 2) == 1.0F);  // 0.5 + 1 clamped
  assert(MaskAt(out, 2, 2) == 0.5F && MaskAt(out, 3, 1) == 0.5F);
  out = comfy::CompositeMask(destination, source, 1, 0, "multiply");
  assert(MaskAt(out, 1, 0) == 0.5F && MaskAt(out, 2, 0) == 0.125F &&
         MaskAt(out, 1, 1) == 0.0F && MaskAt(out, 2, 1) == 0.375F &&
         MaskAt(out, 0, 0) == 0.5F && MaskAt(out, 3, 0) == 0.5F);
  out = comfy::CompositeMask(destination, source, 0, 0, "subtract");
  assert(MaskAt(out, 0, 0) == 0.0F && MaskAt(out, 1, 0) == 0.25F &&
         MaskAt(out, 0, 1) == 0.5F);
  // Logic operations round both sides (0.5 rounds to 0, half to even).
  comfy::MaskGrid ones{2, 1, {1.0F, 0.0F}};
  comfy::MaskGrid other{2, 1, {1.0F, 1.0F}};
  out = comfy::CompositeMask(ones, other, 0, 0, "and");
  assert(out.values == (std::vector<float>{1.0F, 0.0F}));
  out = comfy::CompositeMask(ones, other, 0, 0, "or");
  assert(out.values == (std::vector<float>{1.0F, 1.0F}));
  out = comfy::CompositeMask(ones, other, 0, 0, "xor");
  assert(out.values == (std::vector<float>{0.0F, 1.0F}));
  out = comfy::CompositeMask(destination, ones, 0, 0, "and");
  assert(MaskAt(out, 0, 0) == 0.0F && MaskAt(out, 3, 2) == 0.5F);
  // Outside the destination nothing changes.
  out = comfy::CompositeMask(destination, source, 9, 9, "add");
  assert(out == destination);
  bool rejected = false;
  try {
    (void)comfy::CompositeMask(destination, source, 0, 0, "max");
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  // Area-average resample, stretched.
  comfy::MaskGrid half{4, 2, {1, 1, 0, 0, 1, 1, 0, 0}};
  auto small = comfy::ResampleMask(half, 2, 1);
  assert(small.values == (std::vector<float>{1.0F, 0.0F}));
  small = comfy::ResampleMask(half, 1, 1);
  assert(small.values == (std::vector<float>{0.5F}));
  const auto big = comfy::ResampleMask(half, 8, 4);
  assert(MaskAt(big, 3, 3) == 1.0F && MaskAt(big, 4, 0) == 0.0F);
}

void TestQwenLoras() {
  // LoraLoader -> LoraLoaderModelOnly (strength 0) -> LoraLoaderModelOnly ->
  // KSampler: "loras" in the order ComfyUI applies them, loader side first.
  std::string text = Replace(
      kQwenText, R"("model": ["4", 0], "positive")",
      R"("model": ["L3", 0], "positive")");
  text = Replace(text, R"("clip": ["1", 0], "prompt")",
                 R"("clip": ["L1", 1], "prompt")");
  text = Replace(
      text, R"( "5": {)",
      R"( "L1": {"class_type": "LoraLoader", "inputs": {"model": ["4", 0], "clip": ["1", 0], "lora_name": "figuren\\Anime_Consistency.safetensors", "strength_model": 0.8, "strength_clip": 1.0}},
 "L2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["L1", 0], "lora_name": "off.safetensors", "strength_model": 0.0}},
 "L3": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["L2", 0], "lora_name": "styles/detail.sft", "strength_model": -0.5}},
 "5": {)");
  const comfy::Plan plan = PlanOf(text.c_str());
  const auto& image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert(image.loras.size() == 2);
  assert(image.loras[0].first == "Anime_Consistency" &&
         image.loras[0].second == 0.8);
  assert(image.loras[1].first == "detail" && image.loras[1].second == -0.5);
  const json::Value loras = comfy::ImageLoras(image);
  assert(loras.is_array() && loras.size() == 2);
  assert(loras.items()[0].member_str("name") == "Anime_Consistency");
  assert(loras.items()[1].find("scale")->as_double() == -0.5);
  // The CLIP half of LoraLoader has no gufo counterpart: one warning.
  int clip_warnings = 0;
  for (const auto& warning : plan.warnings)
    clip_warnings += warning.find("strength_clip") != std::string::npos;
  assert(clip_warnings == 1);
  // The extension object stays free of LoRA fields.
  assert(comfy::ImageExtension(image).find("loras") == nullptr);

  // Without LoRA nodes nothing is sent, so the worker's defaults apply.
  const comfy::Plan plain = PlanOf(kQwenText);
  assert(comfy::ImageLoras(std::get<comfy::ImageTask>(plain.tasks[0].spec))
             .is_null());

  // The same file twice adds up; strength_clip 0 does not warn.
  std::string twice = Replace(
      kQwenText, R"("model": ["4", 0], "positive")",
      R"("model": ["L2", 0], "positive")");
  twice = Replace(
      twice, R"( "5": {)",
      R"( "L1": {"class_type": "LoraLoader", "inputs": {"model": ["4", 0], "clip": ["1", 0], "lora_name": "a.safetensors", "strength_model": 0.25, "strength_clip": 0.0}},
 "L2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["L1", 0], "lora_name": "sub/a.safetensors", "strength_model": 0.5}},
 "5": {)");
  const comfy::Plan summed = PlanOf(twice.c_str());
  const auto& merged = std::get<comfy::ImageTask>(summed.tasks[0].spec);
  assert(merged.loras.size() == 1 && merged.loras[0].first == "a" &&
         merged.loras[0].second == 0.75);
  assert(OtherWarnings(summed).empty());

  // Out-of-range strength fails at the node.
  const comfy::PlanError error = ErrorOf(Replace(
      text, R"("strength_model": 0.8)", R"("strength_model": 12)"));
  assert(error.node_errors.find("L1") != nullptr);

  assert(comfy::LoraName("anime.safetensors") == "anime");
  assert(comfy::LoraName("A/B\\c.SafeTensors") == "c");
  assert(comfy::LoraName("plain") == "plain");
  assert(comfy::LoraName("x.ckpt") == "x.ckpt");
}

void TestQwenTurbo() {
  // A UNETLoader naming a Turbo checkpoint goes to the Turbo model, not to
  // qwen-image-2.1; everything else in the graph maps as before.
  std::string turbo = Replace(kQwenText, "qwen_image_2.1_int8_convrot",
                              "qwen_image_2.1_turbo_bf16");
  turbo = Replace(turbo, "\"steps\": 25, \"cfg\": 2.5",
                  "\"steps\": 8, \"cfg\": 1.0");
  const comfy::Plan plan = PlanOf(turbo.c_str());
  const auto& image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert(image.model == "qwen-image-2.1-turbo");
  assert(image.steps == 8);
  assert(image.prompt == "A single mature adult male office worker");
  assert(plan.warnings.empty());
  assert(comfy::IsQwenImageTurboFile("Qwen_Image_2.1_TURBO.safetensors"));
  assert(!comfy::IsQwenImageTurboFile("qwen_image_2.1_int8_convrot.safetensors"));

  const auto plan_with = [](const std::string& text,
                            const comfy::ModelIds& models, comfy::Plan* plan,
                            comfy::PlanError* error) {
    return comfy::BuildPlan(json::parse(text), models, plan, error);
  };
  // The router serves the Turbo model: same mapping.
  comfy::ModelIds served;
  served.available = {"qwen-image-2.1", "qwen-image-2.1-turbo"};
  {
    comfy::Plan p;
    comfy::PlanError e;
    assert(plan_with(turbo, served, &p, &e));
    assert(std::get<comfy::ImageTask>(p.tasks[0].spec).model ==
           "qwen-image-2.1-turbo");
  }
  // Without it: a validation error at the loader, never a silent fallback.
  comfy::ModelIds base_only;
  base_only.available = {"qwen-image-2.1"};
  {
    comfy::Plan p;
    comfy::PlanError e;
    assert(!plan_with(turbo, base_only, &p, &e));
    assert(p.tasks.empty());
    assert(e.type == "prompt_outputs_failed_validation");
    const json::Value* node = e.node_errors.find("4");
    assert(node != nullptr);
    assert(node->member_str("class_type") == "UNETLoader");
    assert(node->find("errors")->items()[0].member_str("type") ==
           "value_not_in_list");
    assert(e.message.find("qwen-image-2.1-turbo") != std::string::npos);
    // The plain graph still runs on qwen-image-2.1 with that model list.
    comfy::Plan plain;
    assert(plan_with(kQwenText, base_only, &plain, &e));
    assert(std::get<comfy::ImageTask>(plain.tasks[0].spec).model ==
           "qwen-image-2.1");
  }
  // No Turbo model id configured: also an error at the loader.
  comfy::ModelIds no_turbo;
  no_turbo.image_turbo.clear();
  {
    comfy::Plan p;
    comfy::PlanError e;
    assert(!plan_with(turbo, no_turbo, &p, &e));
    assert(e.node_errors.find("4") != nullptr);
  }
  // Turbo runs exactly 8 steps: other counts fail at the sampler.
  const comfy::PlanError steps = ErrorOf(
      Replace(turbo, "\"steps\": 8", "\"steps\": 20"));
  assert(steps.node_errors.find("6") != nullptr);
  assert(steps.message.find("exactly 8 steps, not 20") != std::string::npos);
  // No partial-strength image to image on Turbo.
  std::string canvas = Replace(kQwenCanvas, "qwen_image_2.1_int8_convrot",
                               "qwen_image_2.1_turbo_bf16");
  canvas = Replace(canvas, "\"steps\": 25", "\"steps\": 8");
  assert(std::get<comfy::ImageTask>(PlanOf(canvas.c_str()).tasks[0].spec)
             .model == "qwen-image-2.1-turbo");
  const comfy::PlanError denoise =
      ErrorOf(Replace(canvas, "\"denoise\": 1.0", "\"denoise\": 0.6"));
  assert(denoise.node_errors.find("6") != nullptr);
  assert(denoise.message.find("denoise") != std::string::npos);
}

void TestQwenGuidanceSteps() {
  // gufo's optional KSampler input guidance_steps -> gufo.guidance_steps.
  const std::string text =
      Replace(kQwenText, "\"cfg\": 2.5,", "\"cfg\": 2.5, \"guidance_steps\": 12,");
  const comfy::Plan plan = PlanOf(text.c_str());
  const auto& image = std::get<comfy::ImageTask>(plan.tasks[0].spec);
  assert(image.guidance_steps && *image.guidance_steps == 12);
  const json::Value extension = comfy::ImageExtension(image);
  assert(extension.find("guidance")->as_double() == 2.5);
  assert(extension.find("guidance_steps")->as_double() == 12);
  // Without the input nothing is sent (the worker guides every step).
  const auto& plain =
      std::get<comfy::ImageTask>(PlanOf(kQwenText).tasks[0].spec);
  assert(!plain.guidance_steps);
  assert(comfy::ImageExtension(plain).find("guidance_steps") == nullptr);
  // 0 is sent as 0 (every step).
  const auto& zero = std::get<comfy::ImageTask>(
      PlanOf(Replace(text, "\"guidance_steps\": 12", "\"guidance_steps\": 0")
                 .c_str())
          .tasks[0]
          .spec);
  assert(zero.guidance_steps && *zero.guidance_steps == 0);
  // KSamplerAdvanced takes it too.
  std::string advanced = Replace(text, "\"KSampler\"", "\"KSamplerAdvanced\"");
  advanced = Replace(advanced, "\"seed\": 1769224114", "\"noise_seed\": 1769224114");
  const auto& adv =
      std::get<comfy::ImageTask>(PlanOf(advanced.c_str()).tasks[0].spec);
  assert(adv.guidance_steps && *adv.guidance_steps == 12);
  // Out of range, fractional or text: a validation error at the sampler.
  for (const char* bad : {"\"guidance_steps\": 101", "\"guidance_steps\": -1",
                          "\"guidance_steps\": 2.5",
                          "\"guidance_steps\": \"12\""}) {
    const comfy::PlanError error =
        ErrorOf(Replace(text, "\"guidance_steps\": 12", bad));
    assert(error.node_errors.find("6") != nullptr);
    assert(error.message.find("guidance_steps") != std::string::npos);
  }
  // Other families have no gufo.guidance_steps: a warning, not a failure.
  const std::string wan =
      Replace(kWan, "\"cfg\": 5", "\"cfg\": 5, \"guidance_steps\": 4");
  assert(wan != kWan);
  bool warned = false;
  for (const auto& warning : PlanOf(wan.c_str()).warnings)
    warned = warned || warning.find("guidance_steps is ignored") !=
                           std::string::npos;
  assert(warned);
}

void TestQwenLoaderName() {
  // gufo runs its BF16 Qwen-Image 2.1 transformer whatever unet_name says;
  // the name only picks Turbo. A non-BF16 file name warns ...
  const comfy::Plan int8 = PlanOf(kQwenText);
  assert(std::get<comfy::ImageTask>(int8.tasks[0].spec).model ==
         "qwen-image-2.1");
  assert(int8.warnings.size() == 1);
  assert(int8.warnings[0].find("node 4") != std::string::npos);
  assert(int8.warnings[0].find("BF16") != std::string::npos);
  assert(int8.warnings[0].find("int8") != std::string::npos);
  // ... the BF16 file is what runs: no warning, same model.
  const comfy::Plan bf16 = PlanOf(
      Replace(kQwenText, "qwen_image_2.1_int8_convrot", "qwen_image_2.1_bf16")
          .c_str());
  assert(bf16.warnings.empty());
  assert(std::get<comfy::ImageTask>(bf16.tasks[0].spec).model ==
         "qwen-image-2.1");
  // Another Qwen-Image release runs on 2.1 with a warning.
  const comfy::Plan other = PlanOf(
      Replace(kQwenText, "qwen_image_2.1_int8_convrot", "qwen_image_2512_bf16")
          .c_str());
  assert(other.warnings.size() == 1 &&
         other.warnings[0].find("does not name Qwen-Image 2.1") !=
             std::string::npos);
  // Not a Qwen-Image file at all: a validation error at the loader.
  const comfy::PlanError error = ErrorOf(
      Replace(kQwenText, "qwen_image_2.1_int8_convrot", "flux1-dev"));
  const json::Value* node = error.node_errors.find("4");
  assert(node != nullptr);
  assert(node->member_str("class_type") == "UNETLoader");
  assert(node->find("errors")->items()[0].member_str("type") ==
         "value_not_in_list");
  assert(error.message.find("flux1-dev") != std::string::npos);
}

constexpr const char* kMossText = R"({
 "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3.5_9b_bf16.safetensors", "type": "stable_diffusion", "device": "default"}},
 "2": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2_small_vae.safetensors"}},
 "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "moss/MOSS_Image_SFT_bf16.safetensors", "weight_dtype": "default"}},
 "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": "A red ceramic teapot on a wooden table"}},
 "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": "blurry"}},
 "3": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 768, "batch_size": 1}},
 "6": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["9", 0], "latent_image": ["3", 0], "seed": 42, "steps": 40, "cfg": 4.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
 "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "moss"}}
})";

void TestImageModelChoice() {
  // comfy_image_models: the loader name picks the gufo image model.
  const comfy::ModelIds all;  // available empty: every row with an id
  const auto pick = [&](const char* name) {
    return comfy::ChooseImageModel(name, "unet_name", "4", all);
  };
  // Qwen-Image 2.1, whatever folder, case, separators or extension.
  for (const char* name :
       {"qwen_image_2.1_bf16.safetensors", "Qwen/Qwen-Image-2.1.safetensors",
        "image\\QWEN_IMAGE_2_1_BF16.SAFETENSORS", "qwen-image-2.1",
        "qwen_image_2.1_int8_convrot.safetensors",
        "qwen-image-2.1-Q4_K_M.gguf"}) {
    const auto choice = pick(name);
    assert(choice.spec != nullptr && choice.id == "qwen-image-2.1");
  }
  assert(pick("qwen_image_2.1_bf16.safetensors").warnings.empty());
  assert(pick("qwen-image-2.1").warnings.empty());
  // Turbo / Lightning / distilled names -> the Turbo model.
  for (const char* name :
       {"qwen_image_2.1_turbo_bf16.safetensors",
        "sub/Qwen-Image-2.1-TURBO-fp8_e4m3fn.safetensors",
        "qwen-image-2.1-lightning-8steps.safetensors",
        "Qwen_Image_2.1_Distill.safetensors", "qwen-image-2.1-turbo"}) {
    const auto choice = pick(name);
    assert(choice.spec != nullptr && choice.id == "qwen-image-2.1-turbo");
  }
  // MOSS-Image.
  for (const char* name : {"moss_image_sft_bf16.safetensors",
                           "OpenMOSS/MOSS-Image-SFT.safetensors", "moss-image"}) {
    const auto choice = pick(name);
    assert(choice.spec != nullptr && choice.id == "moss-image");
  }
  // MOSS-Music is not an image model; MOSS-Image has no Turbo.
  assert(pick("moss_music_8b.safetensors").spec == nullptr);
  const auto moss_turbo = pick("moss_image_turbo.safetensors");
  assert(moss_turbo.spec == nullptr &&
         moss_turbo.error.find("no turbo / lightning / distilled MOSS-Image") !=
             std::string::npos);
  // Unknown: ComfyUI's "value not in list" with the router's image models.
  const auto flux = pick("flux1-dev.safetensors");
  assert(flux.spec == nullptr);
  assert(flux.error.starts_with(
      "Value not in list: unet_name: 'flux1-dev.safetensors' not in "
      "['qwen-image-2.1', 'qwen-image-2.1-turbo', 'moss-image']"));
  // The list is what the router serves.
  comfy::ModelIds two;
  two.available = {"qwen3.8-27b", "qwen-image-2.1", "moss-image"};
  assert((comfy::ServedImageModels(two) ==
          std::vector<std::string>{"qwen-image-2.1", "moss-image"}));
  assert((comfy::ServedImageModels(all) ==
          std::vector<std::string>{"qwen-image-2.1", "qwen-image-2.1-turbo",
                                   "moss-image"}));
  const auto missing =
      comfy::ChooseImageModel("qwen_image_2.1_turbo_bf16.safetensors",
                              "ckpt_name", "4", two);
  assert(missing.spec == nullptr);
  assert(missing.error.find("ckpt_name: 'qwen_image_2.1_turbo_bf16."
                            "safetensors' not in ['qwen-image-2.1', "
                            "'moss-image']") != std::string::npos);
  assert(missing.error.find("[external/qwen-image-2.1-turbo]") !=
         std::string::npos);
  // Precision tags warn; BF16 and untagged names do not.
  assert(comfy::NonBf16Tag("qwen_image_2.1_int8_convrot.safetensors") == "int8");
  assert(comfy::NonBf16Tag("x/Qwen-Image-2.1-Q4_K_M.gguf") == "gguf");
  assert(comfy::NonBf16Tag("qwen-image-2.1-Q4_K_M.safetensors") == "q4");
  assert(comfy::NonBf16Tag("qwen_image_2.1_fp8_e4m3fn.safetensors") == "fp8");
  assert(comfy::NonBf16Tag("qwen_image_2.1_bf16.safetensors").empty());
  assert(comfy::NonBf16Tag("qwen_image_2.1.safetensors").empty());
  const auto quant = pick("qwen-image-2.1-Q8_0.gguf");
  assert(quant.warnings.size() == 1 &&
         quant.warnings[0].find("gguf weights are not loaded") !=
             std::string::npos &&
         quant.warnings[0].find("node 4") != std::string::npos);

  // Whole graphs. MOSS text to image: negative prompt and cfg are not run
  // (warnings), the size and steps go through.
  const comfy::Plan moss = PlanOf(kMossText);
  const auto& image = std::get<comfy::ImageTask>(moss.tasks[0].spec);
  assert(image.model == "moss-image");
  assert(image.prompt == "A red ceramic teapot on a wooden table");
  assert(image.negative_prompt.empty() && !image.guidance);
  assert(image.width == 1024 && image.height == 768 && image.steps == 40);
  assert(comfy::ImageExtension(image).empty());
  assert(moss.warnings.size() == 2);
  assert(moss.warnings[0].find("negative prompt is ignored") != std::string::npos);
  assert(moss.warnings[1].find("cfg 4 is ignored") != std::string::npos);
  // cfg 3 and an empty negative: nothing to warn about.
  std::string quiet = Replace(kMossText, "\"cfg\": 4.0", "\"cfg\": 3.0");
  quiet = Replace(quiet, "\"text\": \"blurry\"", "\"text\": \"\"");
  assert(PlanOf(quiet.c_str()).warnings.empty());
  // TextEncodeQwenImage21 without references is a prompt for MOSS too.
  const std::string moss_qwen_text =
      Replace(kQwenText, "qwen_image_2.1_int8_convrot.safetensors",
              "moss_image_sft.safetensors");
  assert(std::get<comfy::ImageTask>(PlanOf(moss_qwen_text.c_str()).tasks[0].spec)
             .model == "moss-image");
  // What MOSS-Image cannot run fails at the sampler with the reason ...
  const comfy::PlanError edit = ErrorOf(
      Replace(kQwenCanvas, "qwen_image_2.1_int8_convrot.safetensors",
              "moss_image_sft.safetensors"));
  assert(edit.node_errors.find("6") != nullptr);
  assert(edit.message.find("MOSS-Image (moss-image) runs text to image only") !=
         std::string::npos);
  const std::string lora = Replace(
      Replace(kMossText, "\"model\": [\"4\", 0]", "\"model\": [\"l\", 0]"),
      " \"5\": {",
      " \"l\": {\"class_type\": \"LoraLoaderModelOnly\", \"inputs\": "
      "{\"model\": [\"4\", 0], \"lora_name\": \"winkel.safetensors\", "
      "\"strength_model\": 1.0}},\n \"5\": {");
  assert(ErrorOf(lora).message.find("takes no LoRAs") != std::string::npos);
  const comfy::PlanError small =
      ErrorOf(Replace(kMossText, "\"width\": 1024", "\"width\": 128"));
  assert(small.message.find("256..2048") != std::string::npos);
  assert(ErrorOf(Replace(kMossText, "\"cfg\": 4.0,",
                         "\"cfg\": 4.0, \"guidance_steps\": 5,"))
             .message.find("guidance_steps") != std::string::npos);
  // ... never a silent fallback to Qwen-Image. Not served: at the loader.
  comfy::ModelIds qwen_only;
  qwen_only.available = {"qwen-image-2.1", "qwen-image-2.1-turbo"};
  comfy::Plan p;
  comfy::PlanError e;
  assert(!comfy::BuildPlan(json::parse(kMossText), qwen_only, &p, &e));
  assert(e.node_errors.find("4") != nullptr);
  assert(e.node_errors.find("4")->find("errors")->items()[0].member_str(
             "type") == "value_not_in_list");
  assert(e.message.find("[external/moss-image]") != std::string::npos);
  // CheckpointLoaderSimple names the ckpt_name input.
  const std::string checkpoint = Replace(
      Replace(kQwenText, "\"UNETLoader\"", "\"CheckpointLoaderSimple\""),
      "\"unet_name\": \"qwen_image_2.1_int8_convrot.safetensors\"",
      "\"ckpt_name\": \"sdxl/juggernautXL_v9.safetensors\"");
  const comfy::PlanError sdxl = ErrorOf(checkpoint);
  assert(sdxl.node_errors.find("4") != nullptr);
  assert(sdxl.message.find("ckpt_name: 'sdxl/juggernautXL_v9.safetensors' not "
                           "in ['qwen-image-2.1'") != std::string::npos);
}

void TestRife() {
  // VAEDecode -> RIFE VFI x4 -> SaveImage; the audio output stays plain.
  std::string text =
      Replace(kH3,
              "\"40\": {\"class_type\": \"SaveImage\", \"inputs\": "
              "{\"images\": [\"32\", 0]",
              "\"39\": {\"class_type\": \"RIFE VFI\", \"inputs\": "
              "{\"ckpt_name\": \"rife49.pth\", "
              "\"frames\": [\"32\", 0], \"clear_cache_after_n_frames\": 10, "
              "\"multiplier\": 4, "
              "\"fast_mode\": true, \"ensemble\": true, \"scale_factor\": 1.0, "
              "\"dtype\": "
              "\"float32\", \"torch_compile\": false, \"batch_size\": 1}},\n"
              " \"40\": {\"class_type\": \"SaveImage\", \"inputs\": "
              "{\"images\": [\"39\", 0]");
  const comfy::Plan plan = PlanOf(text.c_str());
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.interpolate == 4 && video.fps == 96);
  assert(comfy::VideoRequestBody(video)
             .find("gufo")
             ->find("interpolate")
             ->as_double() == 4);
  // rife49 (FilmStudio) and rife47 are the weights gufo serves: no warning.
  for (const auto& warning : plan.warnings)
    assert(warning.find("rife") == std::string::npos);
  const comfy::Plan other =
      PlanOf(Replace(text, "rife49.pth", "rife426.pth").c_str());
  bool warned = false;
  for (const auto& warning : other.warnings)
    warned = warned || warning.find("rife426") != std::string::npos;
  assert(warned);
  // A second, plain image output of the same video conflicts.
  text = Replace(text, " \"41\": {",
                 " \"42\": {\"class_type\": \"PreviewImage\", \"inputs\": "
                 "{\"images\": [\"32\", 0]}},\n \"41\": {");
  assert(ErrorOf(text).message.find("same RIFE") != std::string::npos);
  // RIFE on a still image has no gufo endpoint.
  const comfy::PlanError still = ErrorOf(
      Replace(kQwenText, "\"images\": [\"7\", 0]",
              "\"images\": [\"9\", 0]}}, \"9\": {\"class_type\": \"RIFE VFI\", "
              "\"inputs\": {\"frames\": [\"7\", 0], \"multiplier\": 2"));
  assert(still.message.find("H3 or Wan") != std::string::npos);
}

void TestRifeOnly() {
  // LoadVideo -> RIFE VFI -> SaveImage + SaveVideo: one post-processing job
  // (gufo source_video + gufo.interpolate), both outputs from it.
  const comfy::Plan plan = PlanOf(kRifeOnly);
  assert(plan.tasks.size() == 1);
  const auto& task = plan.tasks[0];
  assert(task.node_id == "1" && task.class_type == "LoadVideo");
  assert(task.outputs.size() == 2);
  const auto& video = std::get<comfy::VideoTask>(task.spec);
  assert(video.family == comfy::VideoFamily::kProcess);
  assert(video.model == "gufo-rife");
  assert((video.source_video ==
          comfy::FileRef{"life3_film_a1_s01_x01_roh.mp4", "input"}));
  assert(video.interpolate == 4 && video.fps_num == 0 && video.fps == 96);
  assert(plan.warnings.empty());
  const json::Value body = comfy::VideoRequestBody(video);
  assert(body.member_str("model") == "gufo-rife");
  assert(!body.contains("prompt") && !body.contains("size"));
  assert(body.find("gufo")->find("interpolate")->as_double() == 4);
  assert(!body.find("gufo")->contains("seed"));
  // Without a gufo-rife model on the router the H3 server (--rife-model)
  // takes it.
  comfy::ModelIds ids;
  ids.available = {"minimax-h3", "qwen-image-2.1"};
  comfy::Plan fallback;
  comfy::PlanError error;
  assert(comfy::BuildPlan(json::parse(kRifeOnly), ids, &fallback, &error));
  assert(std::get<comfy::VideoTask>(fallback.tasks[0].spec).model ==
         "minimax-h3");
  // x8 for stills: 192 fps.
  const comfy::Plan stills = PlanOf(
      Replace(Replace(kRifeOnly, "\"multiplier\": 4", "\"multiplier\": 8"),
              "\"fps\": 96.0", "\"fps\": 192.0")
          .c_str());
  assert(std::get<comfy::VideoTask>(stills.tasks[0].spec).interpolate == 8);
  // A LoadVideo straight into SaveVideo has nothing for gufo to do.
  assert(ErrorOf(Replace(kRifeOnly, "\"images\": [\"3\", 0], \"fps\"",
                         "\"images\": [\"2\", 0], \"fps\""))
             .message.find("no generating node") != std::string::npos);
}

void TestRifeSourceFps() {
  // Regie #56-719 (router 1.4.20/21): a 12 fps shot, RIFE x5, SaveImage and
  // CreateVideo 60 fps -> SaveVideo. The adapter probes the file's rate, so
  // 12 x 5 = 60 is the RIFE rate and both outputs agree. Before, 24 fps was
  // assumed (x5 = 120 -> "resample to 60") and the SaveImage output of the
  // same job differed -> "same RIFE interpolation".
  const std::string text =
      Replace(Replace(kRifeOnly, "\"multiplier\": 4", "\"multiplier\": 5"),
              "\"fps\": 96.0", "\"fps\": 60.0");
  const auto plan_with = [](const std::string& prompt, double source_fps) {
    comfy::Plan plan;
    comfy::PlanError error;
    const bool ok = comfy::BuildPlan(
        json::parse(prompt), {}, &plan, &error,
        [source_fps](const comfy::FileRef& file) {
          assert(file.name == "life3_film_a1_s01_x01_roh.mp4" &&
                 file.folder == "input");
          return source_fps;
        });
    if (!ok)
      std::cerr << "unexpected plan error: " << error.message << "\n";
    assert(ok);
    return plan;
  };
  const comfy::Plan twelve = plan_with(text, 12);
  assert(twelve.tasks.size() == 1 && twelve.tasks[0].outputs.size() == 2);
  const auto& video = std::get<comfy::VideoTask>(twelve.tasks[0].spec);
  assert(video.source_fps == 12 && video.interpolate == 5);
  assert(video.fps_num == 0 && video.fps == 60);
  assert(twelve.warnings.empty());
  assert(comfy::VideoRequestBody(video)
             .find("gufo")
             ->find("interpolate")
             ->as_double() == 5);
  // 24 fps x5 -> CreateVideo 120: the same.
  const comfy::Plan full = plan_with(
      Replace(text, "\"fps\": 60.0", "\"fps\": 120.0"), 24);
  const auto& at_24 = std::get<comfy::VideoTask>(full.tasks[0].spec);
  assert(at_24.source_fps == 24 && at_24.interpolate == 5 &&
         at_24.fps_num == 0 && at_24.fps == 120);
  // Without a probe (no ffprobe) 24 fps is assumed: 60 is then a resample
  // of the x5 frames; the SaveImage output of the job takes those frames,
  // no conflict.
  const comfy::Plan assumed = PlanOf(text.c_str());
  const auto& as_24 = std::get<comfy::VideoTask>(assumed.tasks[0].spec);
  assert(as_24.source_fps == 0 && as_24.interpolate == 5 &&
         as_24.fps_num == 60 && as_24.fps == 60);
  assert(assumed.tasks[0].outputs.size() == 2);
  // fps rounding: 59.94 for 60, and a probed 11.988 (12000/1001) source.
  const comfy::Plan ntsc =
      plan_with(Replace(text, "\"fps\": 60.0", "\"fps\": 59.94"), 12);
  assert(std::get<comfy::VideoTask>(ntsc.tasks[0].spec).fps_num == 0);
  assert(std::get<comfy::VideoTask>(ntsc.tasks[0].spec).fps == 60);
  const comfy::Plan probed = plan_with(text, 11.988);
  assert(std::get<comfy::VideoTask>(probed.tasks[0].spec).fps_num == 0);
  assert(std::get<comfy::VideoTask>(probed.tasks[0].spec).fps == 60);
  // Still rejected: a second RIFE chain with another multiplier, and two
  // video outputs asking for different rates.
  const comfy::PlanError multiplier = ErrorOf(Replace(
      text, " \"4\": {",
      " \"7\": {\"class_type\": \"RIFE VFI\", \"inputs\": {\"frames\": "
      "[\"2\", 0], \"multiplier\": 2}},\n \"8\": {\"class_type\": "
      "\"PreviewImage\", \"inputs\": {\"images\": [\"7\", 0]}},\n \"4\": {"));
  assert(multiplier.message.find("same RIFE") != std::string::npos);
  assert(multiplier.message.find("multiplier x") != std::string::npos);
  const comfy::PlanError rates = ErrorOf(Replace(
      text, " \"4\": {",
      " \"7\": {\"class_type\": \"CreateVideo\", \"inputs\": {\"images\": "
      "[\"3\", 0], \"fps\": 48.0, \"codec\": \"h264\"}},\n \"8\": "
      "{\"class_type\": \"SaveVideo\", \"inputs\": {\"video\": [\"7\", 0], "
      "\"filename_prefix\": \"film/b\", \"format\": \"mp4\"}},\n \"4\": {"));
  assert(rates.message.find("same RIFE") != std::string::npos);
  assert(rates.message.find("frame rate") != std::string::npos);
}

constexpr const char* kDialogue = R"({
 "1": {"class_type": "LoadAudio", "inputs": {"audio": "anna.wav"}},
 "2": {"class_type": "LoadAudio", "inputs": {"audio": "ben.flac"}},
 "3": {"class_type": "FB_Qwen3TTSVoiceClonePrompt", "inputs": {"ref_audio": ["1", 0], "ref_text": "Hallo, ich bin Anna.", "model_choice": "1.7B", "device": "auto", "precision": "bf16", "attention": "auto"}},
 "4": {"class_type": "FB_Qwen3TTSVoiceClonePrompt", "inputs": {"ref_audio": ["2", 0], "ref_text": "", "model_choice": "1.7B", "device": "auto", "precision": "bf16", "attention": "auto", "x_vector_only": true}},
 "5": {"class_type": "FB_Qwen3TTSRoleBank", "inputs": {"role_name_1": "Anna", "prompt_1": ["3", 0], "role_name_2": "Ben", "prompt_2": ["4", 0]}},
 "6": {"class_type": "FB_Qwen3TTSDialogueInference", "inputs": {"script": "Anna: Guten Morgen.\n\nben: Morgen!\r\nAnna： Wie geht es dir?", "role_bank": ["5", 0], "model_choice": "1.7B", "device": "auto", "precision": "bf16", "language": "Auto", "pause_linebreak": 0.25, "period_pause": 0.4, "comma_pause": 0.2, "question_pause": 0.6, "hyphen_pause": 0.3, "merge_outputs": true, "batch_size": 4, "seed": 9}},
 "7": {"class_type": "SaveAudio", "inputs": {"audio": ["6", 0], "filename_prefix": "dialog/szene1"}},
 "8": {"class_type": "FB_Qwen3TTSVoiceClone", "inputs": {"target_text": "Allein.", "voice_clone_prompt": ["3", 0], "model_choice": "1.7B", "device": "auto", "precision": "bf16", "language": "German"}},
 "9": {"class_type": "SaveAudio", "inputs": {"audio": ["8", 0], "filename_prefix": "dialog/anna"}}
})";

void TestDialogue() {
  const comfy::Plan plan = PlanOf(kDialogue);
  assert(plan.tasks.size() == 2);
  const comfy::Task* dialogue_task = nullptr;
  const comfy::Task* clone_task = nullptr;
  for (const auto& task : plan.tasks)
    (task.node_id == "6" ? dialogue_task : clone_task) = &task;
  const auto& dialogue = std::get<comfy::DialogueTask>(dialogue_task->spec);
  assert(dialogue.lines.size() == 3 && dialogue.merge);
  assert(dialogue.pause_seconds == 0.25);
  assert(dialogue.lines[0].text == "Guten Morgen.");
  assert((dialogue.lines[0].reference_audio ==
          comfy::FileRef{"anna.wav", "input"}));
  assert(dialogue.lines[0].reference_text == "Hallo, ich bin Anna.");
  assert(!dialogue.lines[0].embedding_only);
  assert(dialogue.lines[1].text == "Morgen!");
  assert((dialogue.lines[1].reference_audio ==
          comfy::FileRef{"ben.flac", "input"}));
  assert(dialogue.lines[1].embedding_only);
  assert(dialogue.lines[2].text == "Wie geht es dir?");
  assert(dialogue.lines[2].model == "qwen3-tts-clone");
  assert(dialogue.lines[2].seed && *dialogue.lines[2].seed == 9);
  // ComfyUI "Auto" stays automatic; dropping it would make gufo speak English.
  assert(dialogue.lines[0].language == "auto");
  const auto& clone = std::get<comfy::SpeechTask>(clone_task->spec);
  assert((clone.reference_audio == comfy::FileRef{"anna.wav", "input"}));
  assert(clone.reference_text == "Hallo, ich bin Anna.");
  assert(clone.language == "german");
  // An unknown role is a validation error.
  const comfy::PlanError error =
      ErrorOf(Replace(kDialogue, "ben: Morgen!", "Carl: Morgen!"));
  assert(error.message.find("Carl") != std::string::npos);
}

void TestAceStep() {
  const comfy::Plan plan = PlanOf(kAceStep);
  assert(plan.tasks.size() == 1);
  const auto& task = plan.tasks[0];
  assert(task.node_id == "8");
  const auto& music = std::get<comfy::MusicTask>(task.spec);
  assert(music.model == "ace-step-1.5-xl-turbo");
  assert(music.tags == "90s anime opening, synth pop, female vocals");
  assert(music.duration == 45 && music.seed == 11 && music.steps == 8);
  assert(task.outputs.size() == 1 &&
         task.outputs[0].kind == comfy::OutputKind::kAudio);
  assert(task.outputs[0].prefix == "film/musik/titel");
  const json::Value body = comfy::MusicRequestBody(music);
  assert(body.member_str("model") == "ace-step-1.5-xl-turbo");
  assert(body.member_str("prompt") == music.tags);
  assert(body.member_str("lyrics") == "[verse]\nNeon im Regen");
  assert(body.find("duration")->as_double() == 45);
  assert(body.find("seed")->as_double() == 11);
  assert(body.find("steps")->as_double() == 8);
  assert(body.find("bpm")->as_double() == 128);
  assert(body.member_str("timesignature") == "4");
  assert(body.member_str("language") == "de");
  assert(body.member_str("keyscale") == "A minor");
  assert(body.find("audio_codes")->as_bool());
  assert(body.find("lm_cfg")->as_double() == 2.0);
  assert(body.find("lm_temperature")->as_double() == 0.85);
  assert(body.find("lm_top_p")->as_double() == 0.9);
  assert(body.member_str("response_format") == "wav");
  assert(plan.warnings.empty());
  // A different encoder duration, seed or a top_k is reported, not fatal.
  std::string text = Replace(kAceStep, "\"duration\": 45", "\"duration\": 60");
  text = Replace(text, "\"top_k\": 0", "\"top_k\": 40");
  assert(PlanOf(text.c_str()).warnings.size() == 2);
  // PreviewAudio works too; a SaveImage of music does not.
  const comfy::Plan preview = PlanOf(
      Replace(kAceStep,
              "\"SaveAudio\", \"inputs\": {\"audio\": [\"9\", 0], "
              "\"filename_prefix\": \"film/musik/titel\"}",
              "\"PreviewAudio\", \"inputs\": {\"audio\": [\"9\", 0]}")
          .c_str());
  assert(preview.tasks[0].outputs[0].temp);
  // SaveAudioMP3 / SaveAudioOpus (v100_lauf_169's music used SaveAudioMP3).
  for (const char* node : {"SaveAudioMP3", "SaveAudioOpus"}) {
    const comfy::Plan coded = PlanOf(
        Replace(kAceStep, "\"SaveAudio\", \"inputs\": {\"audio\": [\"9\", 0], "
                          "\"filename_prefix\": \"film/musik/titel\"}",
                std::string("\"") + node +
                    "\", \"inputs\": {\"audio\": [\"9\", 0], "
                    "\"filename_prefix\": \"film/musik/titel\", "
                    "\"quality\": \"V0\"}")
            .c_str());
    assert(coded.tasks.size() == 1);
    assert(coded.tasks[0].outputs[0].kind == comfy::OutputKind::kAudio);
    assert(coded.tasks[0].outputs[0].class_type == node);
    assert(coded.tasks[0].outputs[0].prefix == "film/musik/titel");
  }
  assert(ErrorOf(Replace(kAceStep, "\"SaveAudio\", \"inputs\": {\"audio\"",
                         "\"SaveImage\", \"inputs\": {\"images\""))
             .message.find("cannot store") != std::string::npos);
  assert(ErrorOf(Replace(kAceStep, "\"seconds\": 45", "\"seconds\": 700"))
             .message.find("1..600") != std::string::npos);
  assert(ErrorOf(Replace(kAceStep, "\"batch_size\": 1", "\"batch_size\": 2"))
             .node_errors.find("7") != nullptr);
}

void TestUni3C() {
  const comfy::Plan plan = PlanOf(kUni3C);
  assert(plan.tasks.size() == 1);
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kCamera);
  assert(video.model == "wan2.1-uni3c");
  assert(video.prompt == "slow orbit around the office desk");
  assert(video.width == 832 && video.height == 480 && video.frames == 81);
  assert(video.fps == 16 && video.steps == 4 && video.seed == 99);
  assert((video.first_frame == comfy::FileRef{"kf_shot07.png", "input"}));
  assert(video.camera && video.camera->type == "orbit_left");
  assert(video.camera->speed == 0.5 && video.camera->ease == "ease_in_out");
  assert(!video.camera->amount && !video.render_video);
  assert(plan.warnings.empty());
  const json::Value body = comfy::VideoRequestBody(video);
  assert(body.member_str("model") == "wan2.1-uni3c");
  assert(body.member_str("size") == "832x480");
  // A single video stays H.264 crf 18 (no gufo.lossless).
  assert(!video.lossless && !body.find("gufo")->contains("lossless"));
  assert(body.find("seconds")->as_double() == 5);
  const json::Value* gufo = body.find("gufo");
  assert(gufo->find("frames")->as_double() == 81);
  assert(gufo->find("steps")->as_double() == 4);
  assert(gufo->find("seed")->as_double() == 99);
  assert(gufo->find("negative_prompt") == nullptr);
  assert(gufo->find("guidance") == nullptr);
  const json::Value* camera = gufo->find("camera");
  assert(camera->member_str("type") == "orbit_left");
  assert(camera->find("speed")->as_double() == 0.5);
  assert(camera->member_str("ease") == "ease_in_out");
  assert(camera->find("amount") == nullptr);
  // 33 frames = 2 s; an explicit amount is sent.
  std::string shorter = Replace(kUni3C, "\"length\": 81", "\"length\": 33");
  shorter = Replace(shorter, "\"amount\": -1", "\"amount\": 30");
  const comfy::Plan two_plan = PlanOf(shorter.c_str());
  const auto& two = std::get<comfy::VideoTask>(two_plan.tasks[0].spec);
  const json::Value two_body = comfy::VideoRequestBody(two);
  assert(two_body.find("seconds")->as_double() == 2);
  assert(two_body.find("gufo")->find("camera")->find("amount")->as_double() ==
         30);
  // A rendered guide video (LoadVideo -> GetVideoComponents) instead.
  std::string rendered = Replace(
      kUni3C,
      "\"14\": {\"class_type\": \"GufoCameraMove\", \"inputs\": {\"camera\": "
      "\"orbit_left\", \"speed\": 0.5, \"ease\": \"ease_in_out\", "
      "\"amount\": -1}}",
      "\"14\": {\"class_type\": \"GetVideoComponents\", \"inputs\": "
      "{\"video\": [\"17\", 0]}},\n \"17\": {\"class_type\": \"LoadVideo\", "
      "\"inputs\": {\"file\": \"render_shot07.mp4\"}}");
  const comfy::Plan guided_plan = PlanOf(rendered.c_str());
  const auto& guided =
      std::get<comfy::VideoTask>(guided_plan.tasks[0].spec);
  assert(!guided.camera);
  assert((guided.render_video ==
          comfy::FileRef{"render_shot07.mp4", "input"}));
  // Errors: unknown move, bad ease, too large, wrong length, no start image,
  // a foreign LoRA.
  assert(ErrorOf(Replace(kUni3C, "\"orbit_left\"", "\"spiral\""))
             .message.find("spiral") != std::string::npos);
  assert(ErrorOf(Replace(kUni3C, "\"ease_in_out\"", "\"bounce\""))
             .message.find("ease") != std::string::npos);
  assert(ErrorOf(Replace(kUni3C, "\"height\": 480", "\"height\": 832"))
             .message.find("832x832") != std::string::npos);
  assert(ErrorOf(Replace(kUni3C, "\"length\": 81", "\"length\": 80"))
             .message.find("4k+1") != std::string::npos);
  assert(ErrorOf(Replace(kUni3C, ", \"start_image\": [\"10\", 0]", ""))
             .message.find("start_image") != std::string::npos);
  assert(ErrorOf(Replace(kUni3C, "lightx2v_I2V_14B_480p_cfg_step_distill",
                         "anime_style"))
             .message.find("LoRA") != std::string::npos);
  // RIFE after the camera video doubles 16 fps.
  const std::string rife = Replace(
      kUni3C, "\"32\": {\"class_type\": \"CreateVideo\", \"inputs\": "
              "{\"images\": [\"31\", 0]",
      "\"34\": {\"class_type\": \"RIFE VFI\", \"inputs\": {\"frames\": "
      "[\"31\", 0], \"multiplier\": 2}},\n \"32\": {\"class_type\": "
      "\"CreateVideo\", \"inputs\": {\"images\": [\"34\", 0]");
  const comfy::Plan smooth_plan = PlanOf(rife.c_str());
  const auto& smooth =
      std::get<comfy::VideoTask>(smooth_plan.tasks[0].spec);
  assert(smooth.interpolate == 2 && smooth.fps == 32);
}

void TestVace() {
  // Wan 2.1 VACE 14B the ComfyUI-core way: control video, mask video
  // (ImageToMask, inverted) and a reference image into WanVaceToVideo,
  // TrimVideoLatent between the sampler and the decode.
  const comfy::Plan plan = PlanOf(kVace);
  assert(plan.tasks.size() == 1);
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kVace);
  assert(video.model == "wan2.1-vace-14b");
  assert(video.prompt == "the woman walks through the office, cinematic");
  assert(video.negative_prompt == "blurry, static");
  assert(video.width == 832 && video.height == 480 && video.frames == 81);
  assert(video.fps == 16 && video.steps == 4 && video.seed == 7);
  assert(!video.guidance);
  assert((video.control_video ==
          comfy::FileRef{"pose_shot03.mp4", "input"}));
  assert((video.vace_mask == comfy::FileRef{"mask_shot03.mp4", "input"}));
  assert(video.vace_mask_channel == "red" && video.vace_invert_mask);
  assert(video.reference_images.size() == 1);
  assert((video.reference_images[0] ==
          comfy::FileRef{"figur_anna.png", "input"}));
  assert(video.vace_strength && *video.vace_strength == 0.8);
  assert(video.vace_sampler == "lcm");
  assert(video.vace_shift && *video.vace_shift == 8.0);
  assert(video.vace_lora);
  assert(plan.warnings.empty());
  assert(plan.tasks[0].outputs.size() == 1);
  const json::Value body = comfy::VideoRequestBody(video);
  assert(body.member_str("model") == "wan2.1-vace-14b");
  assert(body.member_str("size") == "832x480");
  assert(body.find("seconds")->as_double() == 5);
  const json::Value* gufo = body.find("gufo");
  assert(gufo->find("frames")->as_double() == 81);
  assert(gufo->find("steps")->as_double() == 4);
  assert(gufo->find("seed")->as_double() == 7);
  assert(gufo->member_str("negative_prompt") == "blurry, static");
  assert(gufo->find("guidance") == nullptr);
  const json::Value* vace = gufo->find("vace");
  assert(vace != nullptr);
  assert(vace->find("strength")->as_double() == 0.8);
  assert(vace->member_str("mask_channel") == "red");
  assert(vace->find("invert_mask")->as_bool());
  assert(vace->member_str("sampler") == "lcm");
  assert(vace->find("shift")->as_double() == 8);
  assert(vace->find("lora") == nullptr);
  // The media are added by the executor, not the body.
  assert(vace->find("control_video") == nullptr);

  // Without LoRA, cfg 6 and uni_pc: guidance sent, no LoRA, euler warned.
  std::string full = Replace(kVace, "\"model\": [\"2\", 0], \"shift\"",
                             "\"model\": [\"1\", 0], \"shift\"");
  full = Replace(full, "\"cfg\": 1.0, \"sampler_name\": \"lcm\"",
                 "\"cfg\": 6.0, \"sampler_name\": \"uni_pc\"");
  full = Replace(full, "\"steps\": 4", "\"steps\": 20");
  const comfy::Plan full_plan = PlanOf(full.c_str());
  const auto& slow = std::get<comfy::VideoTask>(full_plan.tasks[0].spec);
  assert(slow.guidance && *slow.guidance == 6.0 && slow.steps == 20);
  assert(!slow.vace_lora && slow.vace_sampler == "euler");
  assert(full_plan.warnings.size() == 1 &&
         full_plan.warnings[0].find("uni_pc") != std::string::npos);
  const json::Value slow_body = comfy::VideoRequestBody(slow);
  assert(slow_body.find("gufo")->find("guidance")->as_double() == 6);
  assert(!slow_body.find("gufo")->find("vace")->find("lora")->as_bool());

  // A still mask (LoadImageMask) is used for every frame (warning), no
  // reference, no control video: plain inpainting of a region.
  std::string still = Replace(
      kVace, "\"control_masks\": [\"23\", 0]", "\"control_masks\": [\"25\", 0]");
  still = Replace(still, ", \"reference_image\": [\"24\", 0]", "");
  still = Replace(still, "\"control_video\": [\"21\", 0], ", "");
  const comfy::Plan still_plan = PlanOf(still.c_str());
  const auto& region = std::get<comfy::VideoTask>(still_plan.tasks[0].spec);
  assert(!region.control_video && region.reference_images.empty());
  assert((region.vace_mask == comfy::FileRef{"maske_tisch.png", "input"}));
  assert(region.vace_mask_channel == "red" && !region.vace_invert_mask);
  assert(still_plan.warnings.size() == 1 &&
         still_plan.warnings[0].find("every frame") != std::string::npos);

  // A loaded still as the control video is a one-frame batch (ComfyUI pads
  // it with grey).
  const comfy::Plan one_plan = PlanOf(Replace(kVace, "\"control_video\": [\"21\", 0]",
                                              "\"control_video\": [\"24\", 0]")
                                          .c_str());
  const auto& one = std::get<comfy::VideoTask>(one_plan.tasks[0].spec);
  assert(!one.control_video && one.control_runs.size() == 1);
  assert((one.control_runs[0].image ==
          comfy::FileRef{"figur_anna.png", "input"}));

  // Errors: a node other than a video or frames in front of the control
  // video, an alpha mask, a foreign LoRA, a wrong length, too large,
  // denoise < 1.
  assert(ErrorOf(Replace(kVace, "\"control_video\": [\"21\", 0]",
                         "\"control_video\": [\"12\", 0]"))
             .message.find("LoadVideo") != std::string::npos);
  assert(ErrorOf(Replace(kVace, "\"channel\": \"red\", \"image\": [\"22\", 0]",
                         "\"channel\": \"alpha\", \"image\": [\"22\", 0]"))
             .message.find("alpha") != std::string::npos);
  assert(ErrorOf(Replace(kVace, "lightx2v_T2V_14B_cfg_step_distill",
                         "anime_style"))
             .message.find("LoRA") != std::string::npos);
  assert(ErrorOf(Replace(kVace, "\"length\": 81", "\"length\": 80"))
             .message.find("4k+1") != std::string::npos);
  assert(ErrorOf(Replace(kVace, "\"width\": 832", "\"width\": 1600"))
             .message.find("1280x720") != std::string::npos);
  assert(ErrorOf(Replace(kVace, "\"denoise\": 1.0", "\"denoise\": 0.7"))
             .message.find("denoise") != std::string::npos);
  // VACE classes are known to the adapter.
  const auto& classes = comfy::SupportedClassTypes();
  for (const char* name : {"WanVaceToVideo", "TrimVideoLatent", "ImageToMask",
                           "InvertMask"})
    assert(std::find(classes.begin(), classes.end(), name) != classes.end());
  // The split sampler is lowered before planning; /object_info still lists it.
  for (const char* name : {"SamplerCustomAdvanced", "RandomNoise", "BasicGuider",
                           "CFGGuider", "KSamplerSelect", "BasicScheduler"})
    assert(std::find(classes.begin(), classes.end(), name) != classes.end());
  assert(std::is_sorted(classes.begin(), classes.end()));
  assert(std::adjacent_find(classes.begin(), classes.end()) == classes.end());
}

bool SolidRunIs(const comfy::FrameRun& run, std::uint32_t color, int width,
                int height, int count) {
  return !run.image && run.color == color && run.width == width &&
         run.height == height && run.count == count && run.fits.empty();
}

void TestVaceKeyframes() {
  // The film workflow 1:1: frames built in the graph become control_runs /
  // mask_runs, the two KSamplerAdvanced stages one 20-step sampling.
  const comfy::Plan plan = PlanOf(kVaceKeyframes);
  assert(plan.tasks.size() == 1);
  assert(plan.tasks[0].node_id == "k1");
  assert(plan.tasks[0].outputs.size() == 1);
  assert(plan.tasks[0].outputs[0].kind == comfy::OutputKind::kImage);
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kVace);
  assert(video.model == "wan2.1-vace-14b");
  assert(video.prompt == "the camera stays static. A woman walks forward.");
  assert(video.negative_prompt == "blurry");
  assert(video.width == 896 && video.height == 512 && video.frames == 49);
  assert(video.steps == 20 && video.seed == 533302944);
  assert(video.guidance && *video.guidance == 3.5);
  assert(video.vace_sampler == "euler" && !video.vace_lora);
  assert(video.vace_shift && *video.vace_shift == 8.0);
  assert(!video.vace_strength);
  assert(!video.control_video && !video.vace_mask);
  // Control: keyframe, 23 grey, keyframe, 24 grey.
  assert(video.control_runs.size() == 4);
  comfy::FrameFit lanczos;
  lanczos.width = 896;
  lanczos.height = 512;
  lanczos.crop = true;
  lanczos.method = "lanczos";
  for (int i : {0, 2}) {
    const comfy::FrameRun& key = video.control_runs[i];
    assert(key.count == 1 && key.fits.size() == 1 && key.fits[0] == lanczos);
  }
  assert((video.control_runs[0].image ==
          comfy::FileRef{"kf0_beispiel.png", "input"}));
  assert((video.control_runs[2].image ==
          comfy::FileRef{"kf1_beispiel.png", "input"}));
  assert(SolidRunIs(video.control_runs[1], 8355711, 896, 512, 23));
  assert(SolidRunIs(video.control_runs[3], 8355711, 896, 512, 24));
  assert(comfy::FrameCount(video.control_runs) == 49);
  // Mask: black (keep) on the keyframes, white (generate) between.
  assert(video.mask_runs.size() == 4);
  assert(SolidRunIs(video.mask_runs[0], 0, 896, 512, 1));
  assert(SolidRunIs(video.mask_runs[1], 0xFFFFFF, 896, 512, 23));
  assert(SolidRunIs(video.mask_runs[2], 0, 896, 512, 1));
  assert(SolidRunIs(video.mask_runs[3], 0xFFFFFF, 896, 512, 24));
  assert(video.vace_mask_channel == "red" && !video.vace_invert_mask);
  // Warnings: the split sampling and the two Wan 2.2 checkpoints.
  assert(plan.warnings.size() == 3);
  const auto warned = [&](const std::string& text) {
    return std::any_of(plan.warnings.begin(), plan.warnings.end(),
                       [&](const std::string& w) {
                         return w.find(text) != std::string::npos;
                       });
  };
  assert(warned("k1, k2") && warned("node mh:") && warned("node ml:"));
  const json::Value body = comfy::VideoRequestBody(video);
  assert(body.member_str("size") == "896x512");
  const json::Value* gufo = body.find("gufo");
  assert(gufo->find("frames")->as_double() == 49);
  assert(gufo->find("steps")->as_double() == 20);
  assert(gufo->find("guidance")->as_double() == 3.5);
  const json::Value* vace = gufo->find("vace");
  assert(vace->member_str("mask_channel") == "red");
  assert(vace->member_str("sampler") == "euler");
  assert(!vace->find("lora")->as_bool());
  assert(vace->find("control_video") == nullptr);  // the executor's
  // The executor's ffmpeg filter for a 1024x1024 keyframe: ImageScale's
  // centre crop + lanczos, already at the video size.
  const comfy::ImageSizeProbe square = [](const comfy::FileRef&) {
    return std::optional<std::pair<int, int>>({1024, 1024});
  };
  assert(comfy::FrameRunFilter(video.control_runs, 0, 896, 512, square) ==
         "crop=1024:586:0:219,scale=896:512:flags=lanczos,format=rgb24");
  assert(!comfy::FrameRunFilter(video.control_runs, 1, 896, 512, square));
  assert(!comfy::FrameRunFilter(video.control_runs, 0, 896, 512, {}));

  // RepeatImageBatch of smaller grey frames: ImageBatch fits them to the
  // first frame's size (896x512).
  const std::string repeated = Replace(
      kVaceKeyframes,
      "\"grau1\": {\n  \"class_type\": \"EmptyImage\",\n  \"inputs\": {\n"
      "   \"width\": 896,\n   \"height\": 512,\n   \"batch_size\": 24,",
      "\"grau1\": {\n  \"class_type\": \"RepeatImageBatch\",\n  \"inputs\": "
      "{\n   \"image\": [\"grau1e\", 0],\n   \"amount\": 4\n  }\n },\n "
      "\"grau1e\": {\n  \"class_type\": \"EmptyImage\",\n  \"inputs\": {\n"
      "   \"width\": 448,\n   \"height\": 256,\n   \"batch_size\": 6,");
  const comfy::Plan repeat_plan = PlanOf(repeated.c_str());
  const auto& repeat = std::get<comfy::VideoTask>(repeat_plan.tasks[0].spec);
  assert(repeat.control_runs.size() == 7);
  assert(comfy::FrameCount(repeat.control_runs) == 49);
  for (std::size_t i = 3; i < 7; ++i) {
    const comfy::FrameRun& run = repeat.control_runs[i];
    assert(run.count == 6 && run.width == 448 && run.fits.size() == 1);
    assert(run.fits[0].like_run == 0 && run.fits[0].like_fits == 1);
    assert((comfy::FrameRunSize(repeat.control_runs, i, 1, {}) ==
            std::pair<int, int>{896, 512}));
  }

  // With wan2.2-fun-vace-a14b in the preset the two stages run 1:1 as its
  // two experts: high noise steps 0..9 (k1, mh), low noise 10..19 (k2, ml).
  comfy::ModelIds served;
  served.available = {"wan2.1-vace-14b", "wan2.2-fun-vace-a14b"};
  const auto plan_with = [&](const std::string& text,
                             const comfy::ModelIds& models) {
    comfy::Plan out;
    comfy::PlanError error;
    const bool ok = comfy::BuildPlan(json::parse(text), models, &out, &error);
    if (!ok)
      std::cerr << "unexpected plan error: " << error.message << "\n";
    assert(ok);
    return out;
  };
  const comfy::Plan fun = plan_with(kVaceKeyframes, served);
  assert(fun.tasks.size() == 1 && fun.tasks[0].node_id == "k1");
  const auto& fun_video = std::get<comfy::VideoTask>(fun.tasks[0].spec);
  assert(fun_video.family == comfy::VideoFamily::kVace);
  assert(fun_video.model == "wan2.2-fun-vace-a14b");
  assert(fun_video.steps == 20 && fun_video.seed == 533302944);
  assert(fun_video.vace_switch_step && *fun_video.vace_switch_step == 10);
  assert(fun_video.guidance && *fun_video.guidance == 3.5);
  assert(!fun_video.vace_high_guidance);
  assert(fun_video.vace_shift && *fun_video.vace_shift == 8.0);
  assert(comfy::FrameCount(fun_video.control_runs) == 49);
  // No "one model" / "runs as Wan 2.1" warnings: the graph runs as built.
  if (!fun.warnings.empty())
    std::cerr << "unexpected warning: " << fun.warnings[0] << "\n";
  assert(fun.warnings.empty());
  const json::Value fun_body = comfy::VideoRequestBody(fun_video);
  assert(fun_body.member_str("model") == "wan2.2-fun-vace-a14b");
  const json::Value* fun_vace = fun_body.find("gufo")->find("vace");
  assert(fun_vace->find("switch_step")->as_double() == 10);
  assert(fun_vace->find("high_guidance") == nullptr);
  assert(fun_vace->find("lora") && !fun_vace->find("lora")->as_bool());
  assert(fun_vace->find("lora_high") == nullptr);
  // LoRA per expert: Lightning on the high-noise stage only ->
  // gufo.vace.lora_high true, lora_low false; on both -> the plain flag.
  const std::string high_lora = Replace(
      Replace(kVaceKeyframes,
              "   \"model\": [\n    \"mh\",\n    0\n   ],\n   \"shift\"",
              "   \"model\": [\n    \"lh\",\n    0\n   ],\n   \"shift\""),
      " \"sl\": {",
      " \"lh\": {\"class_type\": \"LoraLoaderModelOnly\", \"inputs\": "
      "{\"model\": [\"mh\", 0], \"lora_name\": "
      "\"Wan2.2-Fun-A14B-Lightning-high_noise.safetensors\", "
      "\"strength_model\": 1.0}},\n \"sl\": {");
  const comfy::Plan high_only = plan_with(high_lora, served);
  const auto& high_only_video =
      std::get<comfy::VideoTask>(high_only.tasks[0].spec);
  assert(high_only_video.vace_lora && high_only_video.vace_lora_low &&
         !*high_only_video.vace_lora_low);
  for (const std::string& warning : high_only.warnings)
    std::cerr << "unexpected warning: " << warning << "\n";
  assert(high_only.warnings.empty());
  const json::Value high_only_body = comfy::VideoRequestBody(high_only_video);
  const json::Value* high_only_vace =
      high_only_body.find("gufo")->find("vace");
  assert(high_only_vace->find("lora") == nullptr);
  assert(high_only_vace->find("lora_high")->as_bool() &&
         !high_only_vace->find("lora_low")->as_bool());
  // Per-expert cfg (high 3.5, low 1) -> guidance 1 + high_guidance 3.5; a
  // different low-noise shift warns.
  const std::string low_cfg = Replace(
      Replace(kVaceKeyframes,
              "   \"cfg\": 3.5,\n   \"sampler_name\": \"euler\",\n   "
              "\"scheduler\": \"simple\",\n   \"positive\": [\n    \"vace\",\n"
              "    0\n   ],\n   \"negative\": [\n    \"vace\",\n    1\n   ],\n"
              "   \"model\": [\n    \"sl\",",
              "   \"cfg\": 1.0,\n   \"sampler_name\": \"euler\",\n   "
              "\"scheduler\": \"simple\",\n   \"positive\": [\n    \"vace\",\n"
              "    0\n   ],\n   \"negative\": [\n    \"vace\",\n    1\n   ],\n"
              "   \"model\": [\n    \"sl\","),
      "    \"ml\",\n    0\n   ],\n   \"shift\": 8.0",
      "    \"ml\",\n    0\n   ],\n   \"shift\": 5.0");
  const comfy::Plan split_cfg = plan_with(low_cfg, served);
  const auto& split_video =
      std::get<comfy::VideoTask>(split_cfg.tasks[0].spec);
  assert(!split_video.guidance);
  assert(split_video.vace_high_guidance &&
         *split_video.vace_high_guidance == 3.5);
  assert(split_cfg.warnings.size() == 1 &&
         split_cfg.warnings[0].find("shift differs") != std::string::npos);
  // One checkpoint on both stages stays on Wan 2.1 VACE.
  const comfy::Plan same = plan_with(
      Replace(kVaceKeyframes, "wan2.2_fun_vace_low_noise_14B",
              "wan2.2_fun_vace_high_noise_14B"),
      served);
  const auto& same_video = std::get<comfy::VideoTask>(same.tasks[0].spec);
  assert(same_video.model == "wan2.1-vace-14b" &&
         !same_video.vace_switch_step);

  // Errors: a gap between the stages, a colour out of range.
  assert(ErrorOf(Replace(kVaceKeyframes, "\"start_at_step\": 10,",
                         "\"start_at_step\": 12,"))
             .message.find("continue") != std::string::npos);
  assert(ErrorOf(Replace(kVaceKeyframes, "\"color\": 16777215",
                         "\"color\": 16777216"))
             .message.find("color") != std::string::npos);
}

// Wan 2.2 I2V A14B as ComfyUI's video_wan2_2_14B_i2v template wires it with
// the lightx2v 4-step LoRAs: WanImageToVideo (no CLIP vision) ->
// KSamplerAdvanced on the high-noise model (steps 0..2, leftover noise) ->
// KSamplerAdvanced on the low-noise model (2..end).
constexpr const char* kWan22I2V = R"({
 "37": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
 "56": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
 "91": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["37", 0], "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "strength_model": 1.0}},
 "92": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["56", 0], "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "strength_model": 1.0}},
 "54": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["91", 0], "shift": 5.0}},
 "55": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["92", 0], "shift": 5.0}},
 "38": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
 "39": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
 "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["38", 0], "text": "the woman turns her head and smiles"}},
 "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["38", 0], "text": "blurry, static"}},
 "52": {"class_type": "LoadImage", "inputs": {"image": "shot03_start.png [output]"}},
 "50": {"class_type": "WanImageToVideo", "inputs": {"positive": ["6", 0], "negative": ["7", 0], "vae": ["39", 0], "width": 832, "height": 480, "length": 81, "batch_size": 1, "start_image": ["52", 0]}},
 "57": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["54", 0], "add_noise": "enable", "noise_seed": 778899, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "positive": ["50", 0], "negative": ["50", 1], "latent_image": ["50", 2], "start_at_step": 0, "end_at_step": 2, "return_with_leftover_noise": "enable"}},
 "58": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["55", 0], "add_noise": "disable", "noise_seed": 0, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "positive": ["50", 0], "negative": ["50", 1], "latent_image": ["57", 0], "start_at_step": 2, "end_at_step": 10000, "return_with_leftover_noise": "disable"}},
 "8": {"class_type": "VAEDecode", "inputs": {"samples": ["58", 0], "vae": ["39", 0]}},
 "60": {"class_type": "CreateVideo", "inputs": {"images": ["8", 0], "fps": 16}},
 "61": {"class_type": "SaveVideo", "inputs": {"video": ["60", 0], "filename_prefix": "film/i2v/shot03", "format": "auto", "codec": "auto"}}
})";

void TestWan22I2V() {
  // Without wan2.2-i2v-a14b in the preset the chain is refused with a
  // message naming the model.
  const comfy::PlanError missing = ErrorOf(kWan22I2V);
  assert(missing.message.find("wan2.2-i2v-a14b") != std::string::npos);
  assert(missing.message.find("does not serve") != std::string::npos);

  comfy::ModelIds served;
  served.available = {"wan2.1-uni3c", "wan2.2-i2v-a14b"};
  const auto plan_with = [&](const std::string& text) {
    comfy::Plan out;
    comfy::PlanError error;
    const bool ok = comfy::BuildPlan(json::parse(text), served, &out, &error);
    if (!ok)
      std::cerr << "unexpected plan error: " << error.message << "\n";
    assert(ok);
    return out;
  };
  // Served: one two-expert job, high noise steps 0..1 (57), low 2..3 (58).
  const comfy::Plan plan = plan_with(kWan22I2V);
  assert(plan.tasks.size() == 1 && plan.tasks[0].node_id == "57");
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kI2V22);
  assert(video.model == "wan2.2-i2v-a14b" && video.fps == 16);
  assert(video.width == 832 && video.height == 480 && video.frames == 81);
  assert(video.steps == 4 && video.seed == 778899);
  assert(video.first_frame &&
         video.first_frame->name == "shot03_start.png" &&
         video.first_frame->folder == "output");
  assert(video.prompt == "the woman turns her head and smiles");
  assert(video.negative_prompt == "blurry, static");
  assert(!video.guidance && !video.i2v_high_guidance);
  assert(video.i2v_switch_step && *video.i2v_switch_step == 2);
  assert(video.i2v_sampler == "euler");
  assert(video.i2v_shift && *video.i2v_shift == 5.0);
  assert(video.i2v_lora_high && video.i2v_lora_low);
  for (const std::string& warning : plan.warnings)
    std::cerr << "unexpected warning: " << warning << "\n";
  assert(plan.warnings.empty());
  const json::Value body = comfy::VideoRequestBody(video);
  assert(body.member_str("model") == "wan2.2-i2v-a14b");
  assert(body.member_str("size") == "832x480");
  assert(body.find("seconds")->as_double() == 5.0);
  const json::Value* gufo = body.find("gufo");
  assert(gufo->find("frames")->as_double() == 81);
  assert(gufo->find("guidance") == nullptr);
  const json::Value* i2v = gufo->find("i2v");
  assert(i2v != nullptr && i2v->find("switch_step")->as_double() == 2);
  assert(i2v->member_str("sampler") == "euler");
  assert(i2v->find("shift")->as_double() == 5.0);
  assert(i2v->find("lora") == nullptr && i2v->find("high_guidance") == nullptr);
  assert(i2v->find("lora_high") == nullptr && i2v->find("lora_low") == nullptr);

  // Per-expert cfg (high 3.5, low 1): guidance stays 1, high_guidance 3.5.
  const comfy::Plan cfg = plan_with(Replace(
      kWan22I2V, "\"steps\": 4, \"cfg\": 1.0, \"sampler_name\": \"euler\", "
                 "\"scheduler\": \"simple\", \"positive\": [\"50\", 0], "
                 "\"negative\": [\"50\", 1], \"latent_image\": [\"50\", 2]",
      "\"steps\": 4, \"cfg\": 3.5, \"sampler_name\": \"euler\", "
      "\"scheduler\": \"simple\", \"positive\": [\"50\", 0], "
      "\"negative\": [\"50\", 1], \"latent_image\": [\"50\", 2]"));
  const auto& cfg_video = std::get<comfy::VideoTask>(cfg.tasks[0].spec);
  assert(!cfg_video.guidance && cfg_video.i2v_high_guidance &&
         *cfg_video.i2v_high_guidance == 3.5);
  assert(cfg.warnings.empty());

  // One KSampler on the high-noise file: both experts, gufo's boundary.
  std::string single = Replace(kWan22I2V, "\"latent_image\": [\"57\", 0]",
                               "\"latent_image\": [\"50\", 2]");
  single = Replace(single, "\"samples\": [\"58\", 0]", "\"samples\": [\"57\", 0]");
  single = Replace(single, "\"end_at_step\": 2,", "\"end_at_step\": 10000,");
  const std::string one_sampler = Replace(
      single,
      " \"58\": {\"class_type\": \"KSamplerAdvanced\"",
      " \"x58\": {\"class_type\": \"KSamplerAdvanced\"");
  // "x58" is not upstream of any output any more: only 57 generates.
  const comfy::Plan one = plan_with(one_sampler);
  assert(one.tasks.size() == 1);
  const auto& one_video = std::get<comfy::VideoTask>(one.tasks[0].spec);
  assert(one_video.family == comfy::VideoFamily::kI2V22 &&
         !one_video.i2v_switch_step);
  assert(one.warnings.size() == 1 &&
         one.warnings[0].find("boundary") != std::string::npos);

  // Without the LoRAs: gufo.i2v.lora false.
  std::string plain = Replace(kWan22I2V, "\"model\": [\"91\", 0], \"shift\"",
                              "\"model\": [\"37\", 0], \"shift\"");
  plain = Replace(plain, "\"model\": [\"92\", 0], \"shift\"",
                  "\"model\": [\"56\", 0], \"shift\"");
  const comfy::Plan no_lora = plan_with(plain);
  const auto& no_lora_video =
      std::get<comfy::VideoTask>(no_lora.tasks[0].spec);
  assert(!no_lora_video.i2v_lora_high && !no_lora_video.i2v_lora_low);
  assert(comfy::VideoRequestBody(no_lora_video)
             .find("gufo")
             ->find("i2v")
             ->find("lora")
             ->as_bool() == false);

  // Errors: another LoRA, no start image, a gap between the stages.
  comfy::Plan out;
  comfy::PlanError error;
  assert(!comfy::BuildPlan(
      json::parse(Replace(kWan22I2V,
                          "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise",
                          "my_style_lora")),
      served, &out, &error));
  assert(error.message.find("takes no other LoRA") != std::string::npos);
  assert(!comfy::BuildPlan(
      json::parse(Replace(kWan22I2V, ", \"start_image\": [\"52\", 0]", "")),
      served, &out, &error));
  assert(error.message.find("start_image") != std::string::npos);
  assert(!comfy::BuildPlan(
      json::parse(Replace(kWan22I2V, "\"start_at_step\": 2,",
                          "\"start_at_step\": 3,")),
      served, &out, &error));
  assert(error.message.find("continue") != std::string::npos);

  // A single sampler on a Wan 2.1 I2V file stays on the camera family.
  const std::string wan21 = Replace(
      one_sampler, "wan2.2_i2v_high_noise_14B_fp8_scaled",
      "wan2.1_i2v_480p_14B_fp8_scaled");
  const comfy::Plan camera = plan_with(wan21);
  assert(std::get<comfy::VideoTask>(camera.tasks[0].spec).family ==
         comfy::VideoFamily::kCamera);

  // LoRA per expert: the low-noise LoRA alone (the reverse of anti_talk).
  const comfy::Plan low_only = plan_with(
      Replace(kWan22I2V, "\"model\": [\"91\", 0], \"shift\"",
              "\"model\": [\"37\", 0], \"shift\""));
  const auto& low_only_video =
      std::get<comfy::VideoTask>(low_only.tasks[0].spec);
  assert(!low_only_video.i2v_lora_high && low_only_video.i2v_lora_low);
  assert(low_only.warnings.empty());
  const json::Value low_only_body = comfy::VideoRequestBody(low_only_video);
  const json::Value* low_only_i2v = low_only_body.find("gufo")->find("i2v");
  assert(low_only_i2v->find("lora") == nullptr);
  assert(!low_only_i2v->find("lora_high")->as_bool() &&
         low_only_i2v->find("lora_low")->as_bool());
  // The high-noise LoRA file on the low-noise stage warns (it runs as the
  // low-noise expert's LoRA).
  const comfy::Plan swapped = plan_with(
      Replace(kWan22I2V, "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise",
              "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise"));
  assert(swapped.warnings.size() == 1 &&
         swapped.warnings[0].find("node 58") != std::string::npos &&
         swapped.warnings[0].find("low-noise step-distill LoRA") !=
             std::string::npos);
  assert(std::get<comfy::VideoTask>(swapped.tasks[0].spec).i2v_lora_low);
}

// The Regie's "anti_talk" Wan 2.2 I2V graph (19.11., seed 151490467):
// lightx2v on the high-noise expert at cfg 1 (steps 0..3 of 6), the
// low-noise expert WITHOUT LoRA at cfg 3.5 with the anti-talk negative
// prompt. Before 2.2.9 gufo ran the LoRA on both experts (one
// gufo.i2v.lora flag). The graph is tests/fixtures/comfy/
// i2v_anti_talk_regie.json.
void TestI2VAntiTalk() {
  const std::filesystem::path file =
      std::filesystem::path(__FILE__).parent_path().parent_path() /
      "fixtures" / "comfy" / "i2v_anti_talk_regie.json";
  std::ifstream in(file, std::ios::binary);
  assert(in && "tests/fixtures/comfy/i2v_anti_talk_regie.json");
  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  comfy::ModelIds served;
  served.available = {"wan2.2-i2v-a14b"};
  comfy::Plan plan;
  comfy::PlanError error;
  const bool ok = comfy::BuildPlan(json::parse(text), served, &plan, &error);
  if (!ok)
    std::cerr << "unexpected plan error: " << error.message << "\n";
  assert(ok && plan.tasks.size() == 1 && plan.tasks[0].node_id == "11");
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kI2V22);
  assert(video.width == 896 && video.height == 512 && video.frames == 73);
  assert(video.steps == 6 && video.seed == 151490467);
  assert(video.i2v_switch_step && *video.i2v_switch_step == 3);
  assert(video.guidance && *video.guidance == 3.5);
  assert(video.i2v_high_guidance && *video.i2v_high_guidance == 1.0);
  assert(video.i2v_shift && *video.i2v_shift == 5.0);
  assert(video.negative_prompt.find("talking, speaking") == 0);
  assert(video.i2v_lora_high && !video.i2v_lora_low);
  for (const std::string& warning : plan.warnings)
    std::cerr << "unexpected warning: " << warning << "\n";
  assert(plan.warnings.empty());
  const json::Value body = comfy::VideoRequestBody(video);
  const json::Value* gufo = body.find("gufo");
  assert(gufo->find("guidance")->as_double() == 3.5);
  const json::Value* i2v = gufo->find("i2v");
  assert(i2v->find("switch_step")->as_double() == 3);
  assert(i2v->find("high_guidance")->as_double() == 1.0);
  assert(i2v->find("lora") == nullptr);
  assert(i2v->find("lora_high")->as_bool() &&
         !i2v->find("lora_low")->as_bool());
}

// Wan 2.1 FLF2V as ComfyUI's template wires it (CLIP vision of the start
// image); gufo runs it on VACE.
constexpr const char* kFirstLast = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.1_flf2v_720p_14B_fp16.safetensors", "weight_dtype": "default"}},
 "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
 "4": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
 "5": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8.0}},
 "6": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
 "10": {"class_type": "LoadImage", "inputs": {"image": "start.png"}},
 "11": {"class_type": "LoadImage", "inputs": {"image": "ende.png [output]"}},
 "7": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["6", 0], "image": ["10", 0], "crop": "none"}},
 "12": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "a door opens slowly"}},
 "13": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "blurry"}},
 "20": {"class_type": "WanFirstLastFrameToVideo", "inputs": {"positive": ["12", 0], "negative": ["13", 0], "vae": ["4", 0], "width": 832, "height": 480, "length": 33, "batch_size": 1, "clip_vision_start_image": ["7", 0], "start_image": ["10", 0], "end_image": ["11", 0]}},
 "31": {"class_type": "KSampler", "inputs": {"model": ["5", 0], "positive": ["20", 0], "negative": ["20", 1], "latent_image": ["20", 2], "seed": 11, "steps": 20, "cfg": 5.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "33": {"class_type": "VAEDecode", "inputs": {"samples": ["31", 0], "vae": ["4", 0]}},
 "34": {"class_type": "CreateVideo", "inputs": {"images": ["33", 0], "fps": 16}},
 "35": {"class_type": "SaveVideo", "inputs": {"video": ["34", 0], "filename_prefix": "film/flf/shot01", "format": "mp4", "codec": "h264"}}
})";

void TestFirstLastFrame() {
  const comfy::Plan plan = PlanOf(kFirstLast);
  assert(plan.tasks.size() == 1);
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kVace);
  assert(video.model == "wan2.1-vace-14b");
  assert(video.prompt == "a door opens slowly");
  assert(video.width == 832 && video.height == 480 && video.frames == 33);
  assert(video.steps == 20 && video.seed == 11);
  assert(video.guidance && *video.guidance == 5.0 && !video.vace_lora);
  // Start frame, 31 grey frames, end frame; mask keeps the two.
  assert(video.control_runs.size() == 3);
  assert((video.control_runs[0].image == comfy::FileRef{"start.png", "input"}));
  assert(video.control_runs[0].fits.empty() && video.control_runs[0].count == 1);
  assert(SolidRunIs(video.control_runs[1], 0x808080, 832, 480, 31));
  assert((video.control_runs[2].image == comfy::FileRef{"ende.png", "output"}));
  assert(video.mask_runs.size() == 3);
  assert(SolidRunIs(video.mask_runs[0], 0, 832, 480, 1));
  assert(SolidRunIs(video.mask_runs[1], 0xFFFFFF, 832, 480, 31));
  assert(SolidRunIs(video.mask_runs[2], 0, 832, 480, 1));
  assert(video.vace_mask_channel == "red");
  // Warnings: FLF on VACE, the ignored CLIP vision, the FLF2V checkpoint.
  assert(plan.warnings.size() == 3);
  assert(plan.warnings[0].find("FLF2V") != std::string::npos);
  assert(plan.warnings[1].find("clip_vision_start_image") != std::string::npos);
  assert(plan.warnings[2].find("flf2v") != std::string::npos);
  assert(comfy::VideoRequestBody(video).find("gufo")->find("vace")
             ->member_str("mask_channel") == "red");
  // The executor fits a 1280x720 start image like the node (bilinear,
  // centre crop).
  const comfy::ImageSizeProbe hd = [](const comfy::FileRef&) {
    return std::optional<std::pair<int, int>>({1280, 720});
  };
  assert(comfy::FrameRunFilter(video.control_runs, 0, 832, 480, hd) ==
         "crop=1248:720:16:0,scale=832:480:flags=bilinear,format=rgb24");

  // Only a start image: start frame, then 32 frames to generate.
  const comfy::Plan start_plan =
      PlanOf(Replace(kFirstLast, ", \"end_image\": [\"11\", 0]", "").c_str());
  const auto& start = std::get<comfy::VideoTask>(start_plan.tasks[0].spec);
  assert(start.control_runs.size() == 2 && start.mask_runs.size() == 2);
  assert(SolidRunIs(start.control_runs[1], 0x808080, 832, 480, 32));
  assert(SolidRunIs(start.mask_runs[1], 0xFFFFFF, 832, 480, 32));

  // Errors: more start and end frames than the length, a wrong length.
  std::string crowded = Replace(
      kFirstLast, "\"start_image\": [\"10\", 0], \"end_image\": [\"11\", 0]",
      "\"start_image\": [\"40\", 0], \"end_image\": [\"41\", 0]");
  crowded = Replace(
      crowded, " \"12\":",
      " \"40\": {\"class_type\": \"RepeatImageBatch\", \"inputs\": {\"image\": "
      "[\"10\", 0], \"amount\": 20}},\n \"41\": {\"class_type\": "
      "\"RepeatImageBatch\", \"inputs\": {\"image\": [\"11\", 0], \"amount\": "
      "20}},\n \"12\":");
  assert(ErrorOf(crowded).message.find("more than length 33") !=
         std::string::npos);
  assert(ErrorOf(Replace(kFirstLast, "\"length\": 33", "\"length\": 32"))
             .message.find("4k+1") != std::string::npos);

  // Known classes with ComfyUI's signatures.
  const auto& classes = comfy::SupportedClassTypes();
  for (const char* name :
       {"EmptyImage", "RepeatImageBatch", "WanFirstLastFrameToVideo"})
    assert(std::find(classes.begin(), classes.end(), name) != classes.end());
  const auto flf = comfy::NodeInfo("WanFirstLastFrameToVideo");
  assert(flf && flf->find("output")->size() == 3);
  const json::Value* optional = flf->find("input")->find("optional");
  for (const char* key : {"clip_vision_start_image", "clip_vision_end_image",
                          "start_image", "end_image"})
    assert(optional->find(key) != nullptr);
  const auto empty = comfy::NodeInfo("EmptyImage");
  assert(empty && empty->find("input")->find("required")->find("color")
                      ->items()[1].find("max")->as_double() == 16777215);
}

// Wan 2.2 Animate as ComfyUI's template wires it (pose and face video from
// LoadVideo, reference image with CLIP vision, the lightx2v and relight
// LoRAs, TrimVideoLatent and ImageFromBatch behind the decode); gufo runs
// it on VACE.
constexpr const char* kAnimate = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "strength_model": 1.0}},
 "8": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": "WanAnimate_relight_lora_fp16.safetensors", "strength_model": 1.0}},
 "5": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["8", 0], "shift": 8.0}},
 "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
 "4": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
 "6": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
 "7": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["6", 0], "image": ["24", 0], "crop": "none"}},
 "12": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "a woman dances in the office"}},
 "13": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "blurry"}},
 "20": {"class_type": "LoadVideo", "inputs": {"file": "pose_tanz.mp4"}},
 "21": {"class_type": "GetVideoComponents", "inputs": {"video": ["20", 0]}},
 "22": {"class_type": "LoadVideo", "inputs": {"file": "gesicht_tanz.mp4"}},
 "23": {"class_type": "GetVideoComponents", "inputs": {"video": ["22", 0]}},
 "24": {"class_type": "LoadImage", "inputs": {"image": "figur_anna.png"}},
 "30": {"class_type": "WanAnimateToVideo", "inputs": {"positive": ["12", 0], "negative": ["13", 0], "vae": ["4", 0], "width": 832, "height": 480, "length": 77, "batch_size": 1, "continue_motion_max_frames": 5, "video_frame_offset": 0, "clip_vision_output": ["7", 0], "reference_image": ["24", 0], "face_video": ["23", 0], "pose_video": ["21", 0]}},
 "31": {"class_type": "KSampler", "inputs": {"model": ["5", 0], "positive": ["30", 0], "negative": ["30", 1], "latent_image": ["30", 2], "seed": 7, "steps": 6, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
 "32": {"class_type": "TrimVideoLatent", "inputs": {"samples": ["31", 0], "trim_amount": ["30", 3]}},
 "33": {"class_type": "VAEDecode", "inputs": {"samples": ["32", 0], "vae": ["4", 0]}},
 "36": {"class_type": "ImageFromBatch", "inputs": {"image": ["33", 0], "batch_index": ["30", 4], "length": 4096}},
 "34": {"class_type": "CreateVideo", "inputs": {"images": ["36", 0], "fps": 16}},
 "35": {"class_type": "SaveVideo", "inputs": {"video": ["34", 0], "filename_prefix": "film/animate/tanz01", "format": "mp4", "codec": "h264"}}
})";

void TestAnimate() {
  const comfy::Plan plan = PlanOf(kAnimate);
  assert(plan.tasks.size() == 1 && plan.tasks[0].node_id == "31");
  assert(plan.tasks[0].outputs.size() == 1);
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kVace);
  assert(video.model == "wan2.1-vace-14b");
  assert(video.prompt == "a woman dances in the office");
  assert(video.negative_prompt == "blurry");
  assert(video.width == 832 && video.height == 480 && video.frames == 77);
  assert(video.fps == 16 && video.steps == 6 && video.seed == 7);
  assert((video.control_video == comfy::FileRef{"pose_tanz.mp4", "input"}));
  assert(video.control_runs.empty() && video.mask_runs.empty());
  assert(!video.vace_mask);  // the whole picture is generated
  assert(video.reference_images.size() == 1);
  assert((video.reference_images[0] ==
          comfy::FileRef{"figur_anna.png", "input"}));
  assert(video.vace_lora && video.vace_sampler == "euler");
  assert(video.vace_shift && *video.vace_shift == 8.0);
  // Warnings: Animate on VACE (face), face_video, clip_vision_output, the
  // relight LoRA, the Animate checkpoint.
  const auto warned = [&](const comfy::Plan& p, const std::string& text) {
    return std::any_of(p.warnings.begin(), p.warnings.end(),
                       [&](const std::string& w) {
                         return w.find(text) != std::string::npos;
                       });
  };
  assert(plan.warnings.size() == 5);
  assert(warned(plan, "facial expression"));
  assert(warned(plan, "face_video is ignored"));
  assert(warned(plan, "clip_vision_output is ignored"));
  assert(warned(plan, "WanAnimate_relight_lora"));
  assert(warned(plan, "Wan2_2-Animate-14B"));

  // Character replacement: background video + still mask, no pose.
  std::string replace = Replace(kAnimate, "\"pose_video\": [\"21\", 0]",
                                "\"background_video\": [\"21\", 0], "
                                "\"character_mask\": [\"40\", 0]");
  replace = Replace(replace, " \"30\":",
                    " \"40\": {\"class_type\": \"LoadImageMask\", \"inputs\": "
                    "{\"image\": \"figur_maske.png\", \"channel\": \"red\"}},"
                    "\n \"30\":");
  const comfy::Plan swap_plan = PlanOf(replace.c_str());
  const auto& swap = std::get<comfy::VideoTask>(swap_plan.tasks[0].spec);
  assert((swap.control_video == comfy::FileRef{"pose_tanz.mp4", "input"}));
  assert((swap.vace_mask == comfy::FileRef{"figur_maske.png", "input"}));
  assert(swap.vace_mask_channel == "red" && !swap.vace_invert_mask);
  assert(!warned(swap_plan, "still mask"));  // Animate repeats it too
  assert(!swap.animate_background && swap.animate_background_runs.empty());

  // Replacement with a pose (18.11.): pose + background + mask. The control
  // video is the pose inside the mask and the background outside (merged
  // by the executor), the mask goes to VACE.
  std::string both = Replace(kAnimate, "\"pose_video\": [\"21\", 0]",
                             "\"pose_video\": [\"21\", 0], "
                             "\"background_video\": [\"26\", 0], "
                             "\"character_mask\": [\"40\", 0]");
  both = Replace(both, " \"30\":",
                 " \"25\": {\"class_type\": \"LoadVideo\", \"inputs\": "
                 "{\"file\": \"buero_leer.mp4\"}},\n"
                 " \"26\": {\"class_type\": \"GetVideoComponents\", "
                 "\"inputs\": {\"video\": [\"25\", 0]}},\n"
                 " \"40\": {\"class_type\": \"LoadImageMask\", \"inputs\": "
                 "{\"image\": \"figur_maske.png\", \"channel\": \"green\"}},"
                 "\n \"30\":");
  const comfy::Plan both_plan = PlanOf(both.c_str());
  const auto& merged = std::get<comfy::VideoTask>(both_plan.tasks[0].spec);
  assert((merged.control_video == comfy::FileRef{"pose_tanz.mp4", "input"}));
  assert((merged.animate_background ==
          comfy::FileRef{"buero_leer.mp4", "input"}));
  assert(merged.animate_background_runs.empty());
  assert((merged.vace_mask == comfy::FileRef{"figur_maske.png", "input"}));
  assert(merged.vace_mask_channel == "green" && !merged.vace_invert_mask);
  assert(both_plan.warnings.size() == 5);  // the same five as above
  assert(!warned(both_plan, "ignored next to"));
  assert(!warned(both_plan, "is ignored without"));
  const json::Value merged_body = comfy::VideoRequestBody(merged);
  assert(merged_body.find("gufo")->find("vace")->member_str("mask_channel") ==
         "green");
  // The merge: background, pose and mask fitted alike, the green channel
  // picks the pose; InvertMask negates it.
  const std::string filter = comfy::AnimateMergeFilter(832, 480, "green", false);
  assert(filter.find("[0:v]scale=832:480:force_original_aspect_ratio="
                     "increase") == 0);
  assert(filter.find("crop=832:480") != std::string::npos);
  assert(filter.find("extractplanes=g,format=gbrp[mask]") !=
         std::string::npos);
  assert(filter.find("negate") == std::string::npos);
  assert(filter.ends_with("[bg][pose][mask]maskedmerge[out]"));
  assert(comfy::AnimateMergeFilter(832, 480, "red", true)
             .find("extractplanes=r,negate,format=gbrp[mask]") !=
         std::string::npos);
  // Background next to the pose without a mask (or the other way round):
  // ignored with a warning, the whole picture follows the pose.
  const comfy::Plan half_plan = PlanOf(
      Replace(both, ", \"character_mask\": [\"40\", 0]", "").c_str());
  const auto& half = std::get<comfy::VideoTask>(half_plan.tasks[0].spec);
  assert(!half.animate_background && !half.vace_mask);
  assert(warned(half_plan, "background_video is ignored without "
                           "character_mask"));

  // A pose built from frames with an offset: frames 10.. kept, the last
  // one repeated up to length (as ComfyUI pads the pose video).
  std::string frames = Replace(kAnimate, "\"pose_video\": [\"21\", 0]",
                               "\"pose_video\": [\"41\", 0]");
  frames = Replace(frames, "\"video_frame_offset\": 0",
                   "\"video_frame_offset\": 10");
  frames = Replace(frames, ", \"face_video\": [\"23\", 0]", "");
  frames = Replace(frames, " \"30\":",
                   " \"41\": {\"class_type\": \"RepeatImageBatch\", "
                   "\"inputs\": {\"image\": [\"42\", 0], \"amount\": 40}},\n"
                   " \"42\": {\"class_type\": \"LoadImage\", \"inputs\": "
                   "{\"image\": \"pose_stand.png\"}},\n \"30\":");
  const comfy::Plan frame_plan = PlanOf(frames.c_str());
  const auto& posed = std::get<comfy::VideoTask>(frame_plan.tasks[0].spec);
  // RepeatImageBatch: 40 runs of one frame; 0..9 skipped (count 0), 10..39
  // kept, the last one stretched by the 47 missing frames.
  assert(!posed.control_video && posed.control_runs.size() == 40);
  assert(comfy::FrameCount(posed.control_runs) == 77);
  assert(posed.control_runs[9].count == 0 && posed.control_runs[10].count == 1);
  assert(posed.control_runs[39].count == 48);
  assert((posed.control_runs[39].image ==
          comfy::FileRef{"pose_stand.png", "input"}));

  // Errors: an offset into a loaded pose video, a wrong length, too large.
  assert(ErrorOf(Replace(kAnimate, "\"video_frame_offset\": 0",
                         "\"video_frame_offset\": 10"))
             .message.find("video_frame_offset") != std::string::npos);
  assert(ErrorOf(Replace(kAnimate, "\"length\": 77", "\"length\": 76"))
             .message.find("4k+1") != std::string::npos);
  assert(ErrorOf(Replace(kAnimate, "\"width\": 832", "\"width\": 1600"))
             .message.find("1280x720") != std::string::npos);

  // Known class with ComfyUI's outputs in order.
  const auto& classes = comfy::SupportedClassTypes();
  assert(std::find(classes.begin(), classes.end(), "WanAnimateToVideo") !=
         classes.end());
  const auto info = comfy::NodeInfo("WanAnimateToVideo");
  assert(info && info->find("output")->size() == 6);
  const json::Value* names = info->find("output_name");
  assert(names->items()[3].str() == "trim_latent");
  assert(names->items()[4].str() == "trim_image");
  assert(names->items()[5].str() == "video_frame_offset");
  assert(info->find("input")->find("optional")->find("character_mask")
             ->items()[0].str() == "MASK");
}

// The film's pose animation (18.11., 1:1 from
// gufo-arbeit/comfy-knoten/pose_animation_api.json): ImageScale behind
// GetVideoComponents on the pose video and in front of reference_image /
// CLIPVisionEncode; the executor scales both before sending.
constexpr const char* kPoseAnimation = R"json({
 "clip": {
  "class_type": "CLIPLoader",
  "inputs": {
   "clip_name": "umt5_xxl_fp16.safetensors",
   "type": "wan",
   "device": "default"
  }
 },
 "vae": {
  "class_type": "VAELoader",
  "inputs": {
   "vae_name": "wan_2.1_vae.safetensors"
  }
 },
 "m": {
  "class_type": "UNETLoader",
  "inputs": {
   "unet_name": "wan2.2_animate_14B_int8_convrot.safetensors",
   "weight_dtype": "default"
  }
 },
 "lora": {
  "class_type": "LoraLoaderModelOnly",
  "inputs": {
   "model": [
    "m",
    0
   ],
   "strength_model": 1.0,
   "lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
  }
 },
 "shift": {
  "class_type": "ModelSamplingSD3",
  "inputs": {
   "model": [
    "lora",
    0
   ],
   "shift": 8.0
  }
 },
 "cv": {
  "class_type": "CLIPVisionLoader",
  "inputs": {
   "clip_name": "clip_vision_h.safetensors"
  }
 },
 "img": {
  "class_type": "LoadImage",
  "inputs": {
   "image": "start_beispiel.png"
  }
 },
 "lv": {
  "class_type": "LoadVideo",
  "inputs": {
   "file": "pose_beispiel.mp4"
  }
 },
 "tp": {
  "class_type": "CLIPTextEncode",
  "inputs": {
   "clip": [
    "clip",
    0
   ],
   "text": "A woman raises her right hand and waves."
  }
 },
 "tn": {
  "class_type": "CLIPTextEncode",
  "inputs": {
   "clip": [
    "clip",
    0
   ],
   "text": "blurry"
  }
 },
 "gv": {
  "class_type": "GetVideoComponents",
  "inputs": {
   "video": [
    "lv",
    0
   ]
  }
 },
 "pose": {
  "class_type": "ImageScale",
  "inputs": {
   "image": [
    "gv",
    0
   ],
   "upscale_method": "lanczos",
   "width": 896,
   "height": 512,
   "crop": "disabled"
  }
 },
 "start": {
  "class_type": "ImageScale",
  "inputs": {
   "image": [
    "img",
    0
   ],
   "upscale_method": "lanczos",
   "width": 896,
   "height": 512,
   "crop": "disabled"
  }
 },
 "cve": {
  "class_type": "CLIPVisionEncode",
  "inputs": {
   "clip_vision": [
    "cv",
    0
   ],
   "image": [
    "start",
    0
   ],
   "crop": "none"
  }
 },
 "a0_anim": {
  "class_type": "WanAnimateToVideo",
  "inputs": {
   "positive": [
    "tp",
    0
   ],
   "negative": [
    "tn",
    0
   ],
   "vae": [
    "vae",
    0
   ],
   "width": 896,
   "height": 512,
   "length": 49,
   "batch_size": 1,
   "continue_motion_max_frames": 5,
   "video_frame_offset": 0,
   "clip_vision_output": [
    "cve",
    0
   ],
   "reference_image": [
    "start",
    0
   ],
   "pose_video": [
    "pose",
    0
   ]
  }
 },
 "a0_k": {
  "class_type": "KSampler",
  "inputs": {
   "model": [
    "shift",
    0
   ],
   "seed": 7,
   "steps": 6,
   "cfg": 1.0,
   "sampler_name": "euler",
   "scheduler": "simple",
   "positive": [
    "a0_anim",
    0
   ],
   "negative": [
    "a0_anim",
    1
   ],
   "latent_image": [
    "a0_anim",
    2
   ],
   "denoise": 1.0
  }
 },
 "a0_trim": {
  "class_type": "TrimVideoLatent",
  "inputs": {
   "samples": [
    "a0_k",
    0
   ],
   "trim_amount": [
    "a0_anim",
    3
   ]
  }
 },
 "a0_dec": {
  "class_type": "VAEDecode",
  "inputs": {
   "samples": [
    "a0_trim",
    0
   ],
   "vae": [
    "vae",
    0
   ]
  }
 },
 "a0_cut": {
  "class_type": "ImageFromBatch",
  "inputs": {
   "image": [
    "a0_dec",
    0
   ],
   "batch_index": [
    "a0_anim",
    4
   ],
   "length": 4096
  }
 },
 "genau": {
  "class_type": "ImageFromBatch",
  "inputs": {
   "image": [
    "a0_cut",
    0
   ],
   "batch_index": 0,
   "length": 49
  }
 },
 "save": {
  "class_type": "SaveImage",
  "inputs": {
   "images": [
    "genau",
    0
   ],
   "filename_prefix": "test/pose_animation"
  }
 }
})json";

void TestPoseAnimationScaled() {
  comfy::FrameFit fit;
  fit.width = 896;
  fit.height = 512;
  fit.crop = false;
  fit.method = "lanczos";
  const std::vector<comfy::FrameFit> fits{fit};
  // wan2.2-animate-14b served: pose video and reference with their fits.
  comfy::ModelIds models;
  models.available = {"wan2.1-vace-14b", "wan2.2-animate-14b"};
  comfy::Plan plan;
  comfy::PlanError error;
  const bool ok =
      comfy::BuildPlan(json::parse(kPoseAnimation), models, &plan, &error);
  if (!ok)
    std::cerr << "unexpected plan error: " << error.message << "\n";
  assert(ok);
  assert(plan.tasks.size() == 1 && plan.tasks[0].node_id == "a0_k");
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kAnimate);
  assert(video.width == 896 && video.height == 512 && video.frames == 49);
  assert((video.pose_video == comfy::FileRef{"pose_beispiel.mp4", "input"}));
  assert(video.pose_runs.empty() && video.pose_video_fits == fits);
  assert(video.reference_images.size() == 1 &&
         (video.reference_images[0] ==
          comfy::FileRef{"start_beispiel.png", "input"}));
  assert(video.reference_fits == fits);
  assert(comfy::FitFilter(fits) ==
         "format=rgb24,scale=w=896:h=512:flags=lanczos,format=rgb24");
  assert(comfy::FitFilter({}) == "format=rgb24");

  // Without the Animate model: the VACE fallback, the same fits.
  const comfy::Plan vace_plan = PlanOf(kPoseAnimation);
  const auto& vace = std::get<comfy::VideoTask>(vace_plan.tasks[0].spec);
  assert(vace.family == comfy::VideoFamily::kVace);
  assert((vace.control_video == comfy::FileRef{"pose_beispiel.mp4", "input"}));
  assert(vace.control_runs.empty() && vace.control_video_fits == fits);
  assert((vace.reference_images[0] ==
          comfy::FileRef{"start_beispiel.png", "input"}));
  assert(vace.reference_fits == fits);

  // ImageScaleBy and a centre crop, chained: applied source first.
  std::string chain = Replace(kPoseAnimation, "\"crop\": \"disabled\"",
                              "\"crop\": \"center\"");
  chain = Replace(chain, "[\n    \"gv\",", "[\n    \"half\",");
  chain = Replace(chain, " \"pose\": {",
                  " \"half\": {\"class_type\": \"ImageScaleBy\", \"inputs\": "
                  "{\"image\": [\"gv\", 0], \"upscale_method\": \"area\", "
                  "\"scale_by\": 0.5}},\n \"pose\": {");
  const comfy::Plan chain_plan = PlanOf(chain.c_str());
  const auto& chained = std::get<comfy::VideoTask>(chain_plan.tasks[0].spec);
  assert(chained.control_video_fits.size() == 2);
  assert(chained.control_video_fits[0].scale_by == 0.5);
  assert(chained.control_video_fits[1].crop);
  assert(comfy::FitFilter(chained.control_video_fits) ==
         "format=rgb24,scale=w=round(iw*0.5):h=round(ih*0.5):flags=area,"
         "crop=w=iw-2*round((iw-ih*896/512+abs(iw-ih*896/512))/4):"
         "h=ih-2*round((ih-iw*512/896+abs(ih-iw*512/896))/4):"
         "x=round((iw-ih*896/512+abs(iw-ih*896/512))/4):"
         "y=round((ih-iw*512/896+abs(ih-iw*512/896))/4):exact=1,"
         "scale=w=896:h=512:flags=lanczos,format=rgb24");
  // An ImageScaleBy still: its size in a frame run.
  std::vector<comfy::FrameRun> runs(1);
  runs[0].width = 100;
  runs[0].height = 50;
  comfy::FrameFit by;
  by.scale_by = 1.5;
  by.crop = false;
  runs[0].fits.push_back(by);
  const auto scaled = comfy::FrameRunSize(runs, 0, 1, {});
  assert(scaled && scaled->first == 150 && scaled->second == 75);
  // A reference that is not a LoadImage behind the scale: rejected.
  assert(ErrorOf(Replace(kPoseAnimation, "[\n    \"img\",", "[\n    \"gv\","))
             .message.find("reference_image") != std::string::npos);
}

// The same templates when the router serves wan2.2-animate-14b (it is in
// the preset): every WanAnimateToVideo input goes to gufo.animate.
void TestAnimateNative() {
  comfy::ModelIds models;
  models.available = {"wan2.1-vace-14b", "wan2.2-animate-14b"};
  const auto plan_of = [&](const std::string& text) {
    comfy::Plan plan;
    comfy::PlanError error;
    const bool ok = comfy::BuildPlan(json::parse(text), models, &plan, &error);
    if (!ok)
      std::cerr << "unexpected plan error: " << error.message << "\n";
    assert(ok);
    return plan;
  };
  const auto error_of = [&](const std::string& text) {
    comfy::Plan plan;
    comfy::PlanError error;
    assert(!comfy::BuildPlan(json::parse(text), models, &plan, &error));
    return error;
  };
  const auto warned = [](const comfy::Plan& p, const std::string& text) {
    return std::any_of(p.warnings.begin(), p.warnings.end(),
                       [&](const std::string& w) {
                         return w.find(text) != std::string::npos;
                       });
  };
  const comfy::Plan plan = plan_of(kAnimate);
  assert(plan.tasks.size() == 1 && plan.tasks[0].node_id == "31");
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kAnimate);
  assert(video.model == "wan2.2-animate-14b");
  assert(video.prompt == "a woman dances in the office");
  assert(video.negative_prompt == "blurry");
  assert(video.width == 832 && video.height == 480 && video.frames == 77);
  assert(video.fps == 16 && video.steps == 6 && video.seed == 7);
  assert((video.pose_video == comfy::FileRef{"pose_tanz.mp4", "input"}));
  assert((video.face_video == comfy::FileRef{"gesicht_tanz.mp4", "input"}));
  assert(!video.background_video && !video.vace_mask && !video.control_video);
  assert(video.reference_images.size() == 1 &&
         (video.reference_images[0] ==
          comfy::FileRef{"figur_anna.png", "input"}));
  assert(video.vace_lora && video.vace_sampler == "euler");
  assert(video.vace_shift && *video.vace_shift == 8.0);
  assert(!video.animate_trim_continue && video.animate_offset == 0);
  // Only the relight LoRA is left out; no VACE warnings.
  assert(plan.warnings.size() == 1);
  assert(warned(plan, "WanAnimate_relight_lora"));
  assert(!warned(plan, "facial expression"));
  const json::Value body = comfy::VideoRequestBody(video);
  assert(body.member_str("model") == "wan2.2-animate-14b");
  assert(body.member_str("size") == "832x480");
  assert(body.find("seconds")->as_double() == 4.75);
  const json::Value* animate = body.find("gufo")->find("animate");
  assert(animate != nullptr && animate->find("length")->as_double() == 77);
  assert(animate->member_str("sampler") == "euler" &&
         animate->find("shift")->as_double() == 8.0);
  assert(!animate->contains("lora") && !animate->contains("video_frame_offset"));
  assert(body.find("gufo")->find("frames")->as_double() == 77);

  // Without clip_vision_output: a note that gufo uses it anyway.
  const comfy::Plan no_clip =
      plan_of(Replace(kAnimate, "\"clip_vision_output\": [\"7\", 0], ", ""));
  assert(warned(no_clip, "CLIP vision"));

  // Character replacement: background video + still mask (no pose).
  std::string replace = Replace(kAnimate, "\"pose_video\": [\"21\", 0]",
                                "\"background_video\": [\"21\", 0], "
                                "\"character_mask\": [\"40\", 0]");
  replace = Replace(replace, " \"30\":",
                    " \"40\": {\"class_type\": \"LoadImageMask\", \"inputs\": "
                    "{\"image\": \"figur_maske.png\", \"channel\": \"red\"}},"
                    "\n \"30\":");
  const comfy::Plan swap_plan = plan_of(replace);
  const auto& swap = std::get<comfy::VideoTask>(swap_plan.tasks[0].spec);
  assert((swap.background_video == comfy::FileRef{"pose_tanz.mp4", "input"}));
  assert(!swap.pose_video);
  assert((swap.vace_mask == comfy::FileRef{"figur_maske.png", "input"}));
  assert(swap.vace_mask_channel == "red");
  assert(comfy::VideoRequestBody(swap).find("gufo")->find("animate")
             ->member_str("mask_channel") == "red");

  // A seek into loaded videos is gufo's own (gufo.animate.video_frame_offset).
  const comfy::Plan seek = plan_of(Replace(
      kAnimate, "\"video_frame_offset\": 0", "\"video_frame_offset\": 72"));
  const auto& seeked = std::get<comfy::VideoTask>(seek.tasks[0].spec);
  assert(seeked.animate_offset == 72);
  assert(comfy::VideoRequestBody(seeked).find("gufo")->find("animate")
             ->find("video_frame_offset")->as_double() == 72);

  // continue_motion from a loaded video, ImageFromBatch on trim_image:
  // gufo drops the continue frames itself (trim_continue).
  std::string resume = Replace(kAnimate, "\"pose_video\": [\"21\", 0]",
                               "\"pose_video\": [\"21\", 0], "
                               "\"continue_motion\": [\"51\", 0]");
  resume = Replace(resume, "\"video_frame_offset\": 0",
                   "\"video_frame_offset\": 77");
  resume = Replace(resume, " \"30\":",
                   " \"50\": {\"class_type\": \"LoadVideo\", \"inputs\": "
                   "{\"file\": \"tanz01.mp4\"}},\n"
                   " \"51\": {\"class_type\": \"GetVideoComponents\", "
                   "\"inputs\": {\"video\": [\"50\", 0]}},\n \"30\":");
  const comfy::Plan resume_plan = plan_of(resume);
  const auto& resumed = std::get<comfy::VideoTask>(resume_plan.tasks[0].spec);
  assert((resumed.continue_motion == comfy::FileRef{"tanz01.mp4", "input"}));
  assert(resumed.animate_trim_continue && resumed.animate_offset == 77);
  const json::Value resumed_body = comfy::VideoRequestBody(resumed);
  const json::Value* resumed_animate =
      resumed_body.find("gufo")->find("animate");
  assert(resumed_animate->find("trim_continue")->as_bool());
  assert(resumed_animate->find("continue_motion_max_frames")->as_double() ==
         5);

  // ImageFromBatch on a loaded video is a cut of that video (FileRef::cuts,
  // cut by the executor): frames 72..76 of the pose video.
  std::string picked = Replace(kAnimate, "\"pose_video\": [\"21\", 0]",
                               "\"pose_video\": [\"21\", 0], "
                               "\"continue_motion\": [\"52\", 0]");
  picked = Replace(picked, " \"30\":",
                   " \"52\": {\"class_type\": \"ImageFromBatch\", \"inputs\": "
                   "{\"image\": [\"21\", 0], \"batch_index\": 72, "
                   "\"length\": 5}},\n \"30\":");
  const comfy::Plan picked_plan = plan_of(picked);
  assert((std::get<comfy::VideoTask>(picked_plan.tasks[0].spec)
              .continue_motion ==
          comfy::FileRef{"pose_tanz.mp4", "input",
                         std::vector<comfy::FrameCut>{{72, 5}}}));

  // Errors: continue_motion from frames picked in the graph that are no
  // loaded video (as a chained node's ImageFromBatch would), no pose and
  // no background, a background without mask.
  const std::string chained =
      Replace(picked, "{\"image\": [\"21\", 0], \"batch_index\": 72",
              "{\"image\": [\"24\", 0], \"batch_index\": 72");
  assert(error_of(chained).message.find("chained") != std::string::npos);
  assert(error_of(Replace(kAnimate, ", \"pose_video\": [\"21\", 0]", ""))
             .message.find("pose_video") != std::string::npos);
  assert(error_of(Replace(kAnimate, "\"pose_video\": [\"21\", 0]",
                          "\"background_video\": [\"21\", 0]"))
             .message.find("character_mask") != std::string::npos);
}

// Chained Wan 2.2 Animate chunks as the Regie's pipeline builds them
// (pipeline/mehrsprecher.py, 19.11.): 81 frames in chunks of 77, chunk 2
// with continue_motion = chunk 1's decode and video_frame_offset = chunk 1's
// output 5, joined by ImageFromBatch / ImageBatch, cut to 81 before
// SaveImage. Before: "output depends on more than one generating node".
// The graph is tests/fixtures/comfy/animate_kette_regie.json.
std::string AnimateChainGraph() {
  const std::filesystem::path file =
      std::filesystem::path(__FILE__).parent_path().parent_path() /
      "fixtures" / "comfy" / "animate_kette_regie.json";
  std::ifstream in(file, std::ios::binary);
  assert(in && "tests/fixtures/comfy/animate_kette_regie.json");
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

void TestAnimateChain() {
  comfy::ModelIds models;
  models.available = {"wan2.1-vace-14b", "wan2.2-animate-14b"};
  const auto plan_of = [&](const std::string& text) {
    comfy::Plan plan;
    comfy::PlanError error;
    const bool ok = comfy::BuildPlan(json::parse(text), models, &plan, &error);
    if (!ok)
      std::cerr << "unexpected plan error: " << error.message << "\n";
    assert(ok);
    return plan;
  };
  const auto error_of = [&](const std::string& text) {
    comfy::Plan plan;
    comfy::PlanError error;
    assert(!comfy::BuildPlan(json::parse(text), models, &plan, &error));
    return error;
  };
  const auto warned = [](const comfy::Plan& p, const std::string& text) {
    return std::any_of(p.warnings.begin(), p.warnings.end(),
                       [&](const std::string& w) {
                         return w.find(text) != std::string::npos;
                       });
  };
  const std::string regie = AnimateChainGraph();

  // One job: chunk length 77, continue 5, 77 + 72 = 149 frames cut to 81.
  const comfy::Plan plan = plan_of(regie);
  assert(plan.tasks.size() == 1 && plan.tasks[0].node_id == "a0_k");
  assert(plan.tasks[0].outputs.size() == 1);
  assert(plan.tasks[0].outputs[0].node_id == "save");
  assert(plan.tasks[0].outputs[0].kind == comfy::OutputKind::kImage);
  assert(plan.tasks[0].outputs[0].max_frames == 0);  // 81 is 4k+1
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kAnimate);
  assert(video.model == "wan2.2-animate-14b");
  assert(video.frames == 81 && video.animate_length == 77 &&
         video.animate_continue_frames == 5);
  assert(video.animate_offset == 0 && !video.continue_motion &&
         video.continue_runs.empty() && !video.animate_trim_continue);
  assert((video.pose_video ==
          comfy::FileRef{"mehrsprecher_pose.mp4", "input"}));
  assert(video.seed == 7 && video.steps == 6);
  assert(!warned(plan, "seed") && !warned(plan, "drops"));
  const json::Value body = comfy::VideoRequestBody(video);
  assert(body.find("seconds")->as_double() == 5.0);
  assert(body.find("gufo")->find("frames")->as_double() == 81);
  const json::Value* animate = body.find("gufo")->find("animate");
  assert(animate->find("length")->as_double() == 77);
  assert(animate->find("continue_frames")->as_double() == 5);
  assert(!animate->contains("video_frame_offset"));

  // A cut that is no 4k+1: 85 frames rendered, the SaveImage keeps 80.
  const comfy::Plan cut80 = plan_of(Replace(regie, "\"length\": 81", "\"length\": 82"));
  const auto& v82 = std::get<comfy::VideoTask>(cut80.tasks[0].spec);
  assert(v82.frames == 85 && cut80.tasks[0].outputs[0].max_frames == 82);

  // No cut: every frame of both chunks (77 + 72).
  const comfy::Plan whole = plan_of(
      Replace(regie, "\"images\": [\"n\", 0]", "\"images\": [\"join\", 0]"));
  const auto& all = std::get<comfy::VideoTask>(whole.tasks[0].spec);
  assert(all.frames == 149 && whole.tasks[0].outputs[0].max_frames == 0);
  assert(comfy::VideoRequestBody(all).find("seconds")->as_double() == 9.25);

  // Three chunks: 77 + 72 + 72.
  std::string three = Replace(regie, " \"join\":",
      " \"a2_anim\": {\"class_type\": \"WanAnimateToVideo\", \"inputs\": "
      "{\"positive\": [\"pos\", 0], \"negative\": [\"neg\", 0], \"vae\": "
      "[\"vae\", 0], \"width\": 832, \"height\": 480, \"length\": 77, "
      "\"batch_size\": 1, \"continue_motion_max_frames\": 5, "
      "\"video_frame_offset\": [\"a1_anim\", 5], \"continue_motion\": "
      "[\"a1_dec\", 0], \"clip_vision_output\": [\"cve\", 0], "
      "\"reference_image\": [\"ref\", 0], \"face_video\": [\"face\", 0], "
      "\"pose_video\": [\"pose\", 0]}},\n"
      " \"a2_k\": {\"class_type\": \"KSampler\", \"inputs\": {\"model\": "
      "[\"ms\", 0], \"positive\": [\"a2_anim\", 0], \"negative\": "
      "[\"a2_anim\", 1], \"latent_image\": [\"a2_anim\", 2], \"seed\": 9, "
      "\"steps\": 6, \"cfg\": 1.0, \"sampler_name\": \"euler\", "
      "\"scheduler\": \"simple\", \"denoise\": 1.0}},\n"
      " \"a2_trim\": {\"class_type\": \"TrimVideoLatent\", \"inputs\": "
      "{\"samples\": [\"a2_k\", 0], \"trim_amount\": [\"a2_anim\", 3]}},\n"
      " \"a2_dec\": {\"class_type\": \"VAEDecode\", \"inputs\": {\"samples\": "
      "[\"a2_trim\", 0], \"vae\": [\"vae\", 0]}},\n"
      " \"join2\": {\"class_type\": \"ImageBatch\", \"inputs\": {\"image1\": "
      "[\"join\", 0], \"image2\": [\"a2_dec\", 0]}},\n \"join\":");
  three = Replace(three, "\"images\": [\"n\", 0]", "\"images\": [\"join2\", 0]");
  const comfy::Plan three_plan = plan_of(three);
  assert(three_plan.tasks.size() == 1 && three_plan.tasks[0].node_id == "a0_k");
  assert(std::get<comfy::VideoTask>(three_plan.tasks[0].spec).frames == 221);
  // Chunk 3 has another seed and no ImageFromBatch on its trim_image.
  assert(warned(three_plan, "seed of node a0_k"));
  assert(warned(three_plan, "node a2_anim: gufo drops"));

  // Errors: a chunk with another prompt, a literal video_frame_offset, an
  // output of one chunk alone, a cut that does not start at frame 0.
  std::string other_prompt = Replace(
      regie, "\"positive\": [\"pos\", 0], \"negative\": [\"neg\", 0], "
             "\"vae\": [\"vae\", 0], \"width\": 832, \"height\": 480, "
             "\"length\": 77, \"batch_size\": 1, "
             "\"continue_motion_max_frames\": 5, \"video_frame_offset\": "
             "[\"a0_anim\", 5]",
      "\"positive\": [\"pos2\", 0], \"negative\": [\"neg\", 0], "
      "\"vae\": [\"vae\", 0], \"width\": 832, \"height\": 480, "
      "\"length\": 77, \"batch_size\": 1, "
      "\"continue_motion_max_frames\": 5, \"video_frame_offset\": "
      "[\"a0_anim\", 5]");
  other_prompt = Replace(other_prompt, " \"neg\":",
                         " \"pos2\": {\"class_type\": \"CLIPTextEncode\", "
                         "\"inputs\": {\"clip\": [\"clip\", 0], \"text\": "
                         "\"someone else\"}},\n \"neg\":");
  assert(error_of(other_prompt).message.find("prompts") != std::string::npos);
  assert(error_of(Replace(regie, "\"video_frame_offset\": [\"a0_anim\", 5]",
                          "\"video_frame_offset\": 72"))
             .message.find("video_frame_offset") != std::string::npos);
  const std::string one_chunk = Replace(
      regie, " \"join\":",
      " \"save0\": {\"class_type\": \"SaveImage\", \"inputs\": {\"images\": "
      "[\"a0_dec\", 0], \"filename_prefix\": \"teil0\"}},\n \"join\":");
  assert(error_of(one_chunk).message.find("one chunk") != std::string::npos);
  assert(error_of(Replace(regie, "\"batch_index\": 0, \"length\": 81",
                          "\"batch_index\": 4, \"length\": 81"))
             .message.find("batch_index 0") != std::string::npos);
  // Chunks of different length (the first one 81, the second 77).
  assert(error_of(Replace(regie, "\"length\": 77", "\"length\": 81"))
             .message.find("length") != std::string::npos);

  // Without wan2.2-animate-14b the chain is not run on VACE.
  comfy::ModelIds vace_only;
  comfy::Plan vace_plan;
  comfy::PlanError vace_error;
  assert(!comfy::BuildPlan(json::parse(regie), vace_only, &vace_plan,
                           &vace_error));
  assert(vace_error.message.find("chained generators") != std::string::npos);
}

// Chained Wan 2.1 I2V chunks as the Regie's FilmStudio builds them
// (mehrsprecher.py bewegung_wf, 19.11.): 81 + 81 + 25 frames, each next
// WanImageToVideo starting from the last frame of the chunk before
// (ImageFromBatch over its VAEDecode; batch_index 80, then 4095 = clamped to
// the last), the decodes joined by ImageBatch with the repeated start frame
// dropped (ImageFromBatch batch_index 1), one SaveImage. Before: "output
// depends on more than one generating node". The graph is
// tests/fixtures/comfy/i2v_kette_regie.json.
std::string I2VChainGraph() {
  const std::filesystem::path file =
      std::filesystem::path(__FILE__).parent_path().parent_path() /
      "fixtures" / "comfy" / "i2v_kette_regie.json";
  std::ifstream in(file, std::ios::binary);
  assert(in && "tests/fixtures/comfy/i2v_kette_regie.json");
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

void TestI2VChain() {
  comfy::ModelIds models;
  models.available = {"wan2.1-uni3c", "wan2.1-vace-14b"};
  const auto plan_of = [&](const std::string& text) {
    comfy::Plan plan;
    comfy::PlanError error;
    const bool ok = comfy::BuildPlan(json::parse(text), models, &plan, &error);
    if (!ok)
      std::cerr << "unexpected plan error: " << error.message << "\n";
    assert(ok);
    return plan;
  };
  const auto error_of = [&](const std::string& text) {
    comfy::Plan plan;
    comfy::PlanError error;
    assert(!comfy::BuildPlan(json::parse(text), models, &plan, &error));
    return error;
  };
  const auto frame = [](int chunk, int index) {
    return comfy::ChainFrame{chunk, index};
  };
  const std::string regie = I2VChainGraph();

  // One task of three chunks on wan2.1-uni3c, each with its own prompt,
  // seed and length; the start frames are the last frames before.
  const comfy::Plan plan = plan_of(regie);
  assert(plan.tasks.size() == 1 && plan.tasks[0].node_id == "c0_k");
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kCamera);
  assert(video.model == "wan2.1-uni3c");
  assert(video.frames == 81 && video.seed == 11 && video.steps == 4);
  assert(video.prompt == "Ulrika turns her head to the window");
  assert((video.first_frame == comfy::FileRef{"mehrsprecher_kf0.png", "input"}));
  assert(video.i2v_chunks.size() == 2 && video.i2v_starts.size() == 2);
  const comfy::VideoTask& c1 = video.i2v_chunks[0];
  const comfy::VideoTask& c2 = video.i2v_chunks[1];
  assert(c1.frames == 81 && c1.seed == 12 && !c1.first_frame);
  assert(c1.prompt == "Ulrika stands up and walks to the window");
  assert(c2.frames == 25 && c2.seed == 13 && !c2.first_frame);
  assert(c2.prompt == "Ulrika looks out of the window");
  assert(c1.model == "wan2.1-uni3c" && c2.width == 832 && c2.height == 480);
  assert(video.i2v_starts[0] == frame(0, 80));
  assert(video.i2v_starts[1] == frame(1, 80));  // 4095 clamps to the last
  const json::Value c2_body = comfy::VideoRequestBody(c2);
  assert(c2_body.find("seconds")->as_double() == 1.5);
  assert(c2_body.find("gufo")->find("frames")->as_double() == 25);
  assert(c2_body.find("gufo")->find("seed")->as_double() == 13);
  assert(c2_body.member_str("prompt") == "Ulrika looks out of the window");
  // Every chunk asks the worker for a lossless MP4 (gufo.lossless): the
  // next start frame and the joined frames are the decoded pixels.
  assert(video.lossless && c1.lossless && c2.lossless);
  assert(c2_body.find("gufo")->find("lossless")->as_bool());
  assert(comfy::VideoRequestBody(video).find("gufo")->find("lossless")->as_bool());
  {
    comfy::VideoTask i2v22 = c1;
    i2v22.family = comfy::VideoFamily::kI2V22;
    assert(comfy::VideoRequestBody(i2v22).find("gufo")->find("lossless")->as_bool());
    i2v22.lossless = false;
    assert(!comfy::VideoRequestBody(i2v22).find("gufo")->contains("lossless"));
  }
  // The SaveImage: 81 + 80 + 24 frames, the repeated start frames dropped
  // as ImageFromBatch(batch_index 1) does in ComfyUI.
  assert(plan.tasks[0].outputs.size() == 1);
  const auto& frames = plan.tasks[0].outputs[0].chain_frames;
  assert(plan.tasks[0].outputs[0].kind == comfy::OutputKind::kImage);
  assert(frames.size() == 185);
  assert(frames[0] == frame(0, 0) && frames[80] == frame(0, 80));
  assert(frames[81] == frame(1, 1) && frames[160] == frame(1, 80));
  assert(frames[161] == frame(2, 1) && frames[184] == frame(2, 24));

  // Without ImageFromBatch in the join: every frame, the start frames twice.
  std::string whole = Replace(regie, "\"image2\": [\"c1_rest\", 0]",
                              "\"image2\": [\"c1_dec\", 0]");
  whole = Replace(whole, "\"image2\": [\"c2_rest\", 0]",
                  "\"image2\": [\"c2_dec\", 0]");
  const comfy::Plan whole_plan = plan_of(whole);
  const auto& all = whole_plan.tasks[0].outputs[0].chain_frames;
  assert(all.size() == 187);
  assert(all[80] == frame(0, 80) && all[81] == frame(1, 0));
  assert(all[162] == frame(2, 0) && all[186] == frame(2, 24));

  // A cut after the join (ImageFromBatch(0, 100)) and a SaveVideo
  // (CreateVideo) of the joined frames.
  const comfy::Plan cut_plan = plan_of(Replace(
      regie, " \"save\":",
      " \"cut\": {\"class_type\": \"ImageFromBatch\", \"inputs\": {\"image\": "
      "[\"join2\", 0], \"batch_index\": 0, \"length\": 100}},\n"
      " \"cv\": {\"class_type\": \"CreateVideo\", \"inputs\": {\"images\": "
      "[\"join2\", 0], \"fps\": 16}},\n"
      " \"sv\": {\"class_type\": \"SaveVideo\", \"inputs\": {\"video\": "
      "[\"cv\", 0], \"filename_prefix\": \"mehrsprecher/bewegung\", "
      "\"format\": \"mp4\", \"codec\": \"h264\"}},\n"
      " \"save_cut\": {\"class_type\": \"SaveImage\", \"inputs\": {\"images\": "
      "[\"cut\", 0], \"filename_prefix\": \"cut\"}},\n \"save\":"));
  assert(cut_plan.tasks.size() == 1);
  for (const comfy::OutputTarget& target : cut_plan.tasks[0].outputs) {
    if (target.node_id == "save_cut")
      assert(target.chain_frames.size() == 100 &&
             target.chain_frames[99] == frame(1, 19));
    else
      assert(target.chain_frames.size() == 185);
    if (target.node_id == "sv")
      assert(target.kind == comfy::OutputKind::kVideo);
  }
  assert(cut_plan.tasks[0].outputs.size() == 3);

  // An output of the first chunk alone shares the chain's task.
  const comfy::Plan first_plan = plan_of(Replace(
      regie, " \"save\":",
      " \"save0\": {\"class_type\": \"SaveImage\", \"inputs\": {\"images\": "
      "[\"c0_dec\", 0], \"filename_prefix\": \"teil0\"}},\n \"save\":"));
  assert(first_plan.tasks.size() == 1 &&
         first_plan.tasks[0].outputs.size() == 2);
  for (const comfy::OutputTarget& target : first_plan.tasks[0].outputs)
    assert(target.chain_frames.size() ==
           (target.node_id == "save0" ? 81u : 185u));

  // The LoRA is checked per chunk: chunk 3 on a foreign LoRA fails.
  std::string lora = Replace(
      regie, "\"model\": [\"ms\", 0], \"positive\": [\"c2_i2v\", 0]",
      "\"model\": [\"lora2\", 0], \"positive\": [\"c2_i2v\", 0]");
  lora = Replace(lora, " \"clip\":",
                 " \"lora2\": {\"class_type\": \"LoraLoaderModelOnly\", "
                 "\"inputs\": {\"model\": [\"unet\", 0], \"lora_name\": "
                 "\"anime_style.safetensors\", \"strength_model\": 1.0}},\n"
                 " \"clip\":");
  assert(error_of(lora).message.find("anime_style") != std::string::npos);

  // A real branch: chunks 2 and 3 both continue chunk 1, one output joins
  // all three.
  const comfy::PlanError branch = error_of(
      Replace(regie, "\"image\": [\"c1_dec\", 0], \"batch_index\": 4095",
              "\"image\": [\"c0_dec\", 0], \"batch_index\": 4095"));
  assert(branch.message.find("no single chain") != std::string::npos);
  // ... and as two outputs (chunk 1 + 2, chunk 1 + 3).
  std::string two_outputs =
      Replace(regie, "\"image\": [\"c1_dec\", 0], \"batch_index\": 4095",
              "\"image\": [\"c0_dec\", 0], \"batch_index\": 4095");
  two_outputs = Replace(two_outputs, "\"image1\": [\"join1\", 0]",
                        "\"image1\": [\"c0_dec\", 0]");
  two_outputs = Replace(
      two_outputs, " \"save\":",
      " \"save1\": {\"class_type\": \"SaveImage\", \"inputs\": {\"images\": "
      "[\"join1\", 0], \"filename_prefix\": \"a\"}},\n \"save\":");
  assert(error_of(two_outputs).message.find("same chunk") !=
         std::string::npos);
  // Two independent videos in one output (chunk 2 from the LoadImage).
  std::string independent =
      Replace(regie, "\"start_image\": [\"c1_start\", 0]",
              "\"start_image\": [\"img\", 0]");
  independent = Replace(independent, "\"image\": [\"c1_start\", 0]",
                        "\"image\": [\"img\", 0]");
  assert(error_of(independent).message.find("no single chain") !=
         std::string::npos);
  // A start_image of two frames, another size, RIFE behind the chain.
  assert(error_of(Replace(regie, "\"batch_index\": 80, \"length\": 1",
                          "\"batch_index\": 79, \"length\": 2"))
             .message.find("one frame") != std::string::npos);
  assert(error_of(Replace(regie,
                          "\"width\": 832, \"height\": 480, \"length\": 25",
                          "\"width\": 480, \"height\": 832, \"length\": 25"))
             .message.find("one size") != std::string::npos);
  const std::string rife = Replace(
      Replace(regie, "\"images\": [\"join2\", 0]", "\"images\": [\"rife\", 0]"),
      " \"save\":",
      " \"rife\": {\"class_type\": \"RIFE VFI\", \"inputs\": {\"frames\": "
      "[\"join2\", 0], \"multiplier\": 2}},\n \"save\":");
  assert(error_of(rife).message.find("RIFE") != std::string::npos);
}

void TestInfiniteTalk() {
  const comfy::Plan plan = PlanOf(kTalk);
  assert(plan.tasks.size() == 1);
  assert(plan.tasks[0].node_id == "30");
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kTalk);
  assert(video.model == "wan2.1-infinitetalk-14b");
  assert(video.prompt == "a woman talks calmly to the camera");
  assert(video.width == 832 && video.height == 480 && video.frames == 81);
  assert(video.fps == 25 && video.steps == 4 && video.seed == 7);
  assert((video.first_frame == comfy::FileRef{"anna_portrait.png", "input"}));
  assert(video.talk_audio.size() == 1);
  assert((video.talk_audio[0] == comfy::FileRef{"anna_satz1.wav", "input"}));
  assert(video.talk_masks.empty());
  assert(plan.warnings.empty());
  const json::Value body = comfy::VideoRequestBody(video);
  assert(body.member_str("model") == "wan2.1-infinitetalk-14b");
  assert(body.member_str("size") == "832x480");
  assert(std::abs(body.find("seconds")->as_double() - 3.24) < 1e-9);
  const json::Value* gufo = body.find("gufo");
  assert(gufo->find("frames") == nullptr);
  assert(gufo->find("steps")->as_double() == 4);
  const json::Value* talk = gufo->find("talk");
  assert(talk->find("audio_scale")->as_double() == 1.0);
  assert(talk->find("motion_frames")->as_double() == 9);
  assert(talk->member_str("audio_mode") == "auto");

  // Two speakers: mask_1 from LoadImageMask (alpha), mask_2 from LoadImage's
  // MASK output.
  std::string two = Replace(
      kTalk, "\"mode\": \"single_speaker\",",
      "\"mode\": \"two_speakers\", \"mode.audio_encoder_output_2\": "
      "[\"41\", 0], \"mode.mask_1\": [\"42\", 0], \"mode.mask_2\": "
      "[\"43\", 1],");
  two = Replace(
      two, " \"10\": {",
      " \"40\": {\"class_type\": \"LoadAudio\", \"inputs\": {\"audio\": "
      "\"ben_satz1.wav\"}},\n \"41\": {\"class_type\": "
      "\"AudioEncoderEncode\", \"inputs\": {\"audio_encoder\": [\"7\", 0], "
      "\"audio\": [\"40\", 0]}},\n \"42\": {\"class_type\": "
      "\"LoadImageMask\", \"inputs\": {\"image\": \"maske_anna.png\", "
      "\"channel\": \"alpha\"}},\n \"43\": {\"class_type\": \"LoadImage\", "
      "\"inputs\": {\"image\": \"maske_ben.png\"}},\n \"10\": {");
  const comfy::Plan two_plan = PlanOf(two.c_str());
  const auto& pair = std::get<comfy::VideoTask>(two_plan.tasks[0].spec);
  assert(pair.talk_audio.size() == 2);
  assert((pair.talk_audio[1] == comfy::FileRef{"ben_satz1.wav", "input"}));
  assert(pair.talk_masks.size() == 2);
  assert((pair.talk_masks[0] == comfy::FileRef{"maske_anna.png", "input"}));
  assert((pair.talk_masks[1] == comfy::FileRef{"maske_ben.png", "input"}));
  assert(pair.talk_mask_channels[0] == "inverted_alpha");
  assert(pair.talk_mask_channels[1] == "inverted_alpha");
  // Two speakers without masks, or from an unsupported node, fail.
  assert(ErrorOf(Replace(two, ", \"mode.mask_2\": [\"43\", 1]", ""))
             .message.find("mask_1 and mask_2") != std::string::npos);
  assert(ErrorOf(Replace(two, "\"mode.mask_2\": [\"43\", 1]",
                         "\"mode.mask_2\": [\"43\", 0]"))
             .message.find("LoadImageMask") != std::string::npos);
  // ImageToMask on a loaded still keeps its colour channel (mehrsprecher
  // graphs build speaker masks this way).
  std::string to_mask = Replace(
      two, "\"mode.mask_1\": [\"42\", 0]", "\"mode.mask_1\": [\"45\", 0]");
  to_mask = Replace(
      to_mask, " \"10\": {",
      " \"44\": {\"class_type\": \"LoadImage\", \"inputs\": {\"image\": "
      "\"maske_rot.png\"}},\n \"45\": {\"class_type\": \"ImageToMask\", "
      "\"inputs\": {\"image\": [\"44\", 0], \"channel\": \"red\"}},\n "
      "\"10\": {");
  const comfy::Plan to_mask_plan = PlanOf(to_mask.c_str());
  const auto& red = std::get<comfy::VideoTask>(to_mask_plan.tasks[0].spec);
  assert((red.talk_masks[0] == comfy::FileRef{"maske_rot.png", "input"}));
  assert(red.talk_mask_channels[0] == "red");
  assert(ErrorOf(Replace(to_mask, "\"channel\": \"red\"}},\n \"10\"",
                         "\"channel\": \"alpha\"}},\n \"10\""))
             .message.find("red, green or blue") != std::string::npos);

  // A second window continuing from the first (previous_frames) runs as one
  // job: 81 + (81 - 9) frames.
  std::string chained = Replace(
      kTalk, " \"33\": {",
      " \"50\": {\"class_type\": \"WanInfiniteTalkToVideo\", \"inputs\": "
      "{\"mode\": \"single_speaker\", \"model\": [\"2\", 0], "
      "\"model_patch\": [\"6\", 0], \"positive\": [\"12\", 0], "
      "\"negative\": [\"13\", 0], \"vae\": [\"4\", 0], \"width\": 832, "
      "\"height\": 480, \"length\": 81, \"clip_vision_output\": [\"11\", 0], "
      "\"start_image\": [\"10\", 0], \"audio_encoder_output_1\": [\"9\", 0], "
      "\"motion_frame_count\": 9, \"audio_scale\": 1.0, \"previous_frames\": "
      "[\"31\", 0]}},\n \"51\": {\"class_type\": \"KSampler\", \"inputs\": "
      "{\"model\": [\"50\", 0], \"positive\": [\"50\", 1], \"negative\": "
      "[\"50\", 2], \"latent_image\": [\"50\", 3], \"seed\": 7, \"steps\": 4, "
      "\"cfg\": 1.0, \"sampler_name\": \"lcm\", \"scheduler\": \"simple\", "
      "\"denoise\": 1.0}},\n \"52\": {\"class_type\": \"VAEDecode\", "
      "\"inputs\": {\"samples\": [\"51\", 0], \"vae\": [\"4\", 0]}},\n "
      "\"53\": {\"class_type\": \"ImageFromBatch\", \"inputs\": {\"image\": "
      "[\"52\", 0], \"batch_index\": [\"50\", 4], \"length\": 4096}},\n "
      "\"54\": {\"class_type\": \"ImageBatch\", \"inputs\": {\"image1\": "
      "[\"31\", 0], \"image2\": [\"53\", 0]}},\n \"33\": {");
  chained = Replace(chained, "\"images\": [\"31\", 0], \"audio\"",
                    "\"images\": [\"54\", 0], \"audio\"");
  const comfy::Plan chain_plan = PlanOf(chained.c_str());
  assert(chain_plan.tasks.size() == 1);
  assert(chain_plan.tasks[0].node_id == "51");
  const auto& longer = std::get<comfy::VideoTask>(chain_plan.tasks[0].spec);
  assert(longer.frames == 153);
  assert(chain_plan.warnings.size() == 1 &&
         chain_plan.warnings[0].find("one gufo job") != std::string::npos);
  // Windows with different audio cannot be one job.
  std::string other = Replace(
      chained, "\"previous_frames\": [\"31\", 0]}}",
      "\"previous_frames\": [\"31\", 0]}},\n \"55\": {\"class_type\": "
      "\"LoadAudio\", \"inputs\": {\"audio\": \"other.wav\"}},\n \"56\": "
      "{\"class_type\": \"AudioEncoderEncode\", \"inputs\": "
      "{\"audio_encoder\": [\"7\", 0], \"audio\": [\"55\", 0]}}");
  other = Replace(other,
                  "\"audio_encoder_output_1\": [\"9\", 0], "
                  "\"motion_frame_count\": 9, \"audio_scale\": 1.0, "
                  "\"previous_frames\"",
                  "\"audio_encoder_output_1\": [\"56\", 0], "
                  "\"motion_frame_count\": 9, \"audio_scale\": 1.0, "
                  "\"previous_frames\"");
  assert(ErrorOf(other).message.find("same audio") != std::string::npos);

  // Errors: no start image, too large, foreign LoRA, audio not from the
  // encoder.
  assert(ErrorOf(Replace(kTalk, ", \"start_image\": [\"10\", 0]", ""))
             .message.find("start_image") != std::string::npos);
  assert(ErrorOf(Replace(kTalk, "\"height\": 480", "\"height\": 832"))
             .message.find("832x832") != std::string::npos);
  assert(ErrorOf(Replace(kTalk, "lightx2v_I2V_14B_480p_cfg_step_distill",
                         "anime_style"))
             .message.find("LoRA") != std::string::npos);
  assert(ErrorOf(Replace(kTalk, "\"audio_encoder_output_1\": [\"9\", 0]",
                         "\"audio_encoder_output_1\": [\"8\", 0]"))
             .message.find("AudioEncoderEncode") != std::string::npos);
  // motion_frame_count snaps to 4k+1 with a warning.
  const comfy::Plan snapped = PlanOf(
      Replace(kTalk, "\"motion_frame_count\": 9", "\"motion_frame_count\": 10")
          .c_str());
  assert(std::get<comfy::VideoTask>(snapped.tasks[0].spec).motion_frames ==
         9);
  assert(snapped.warnings.size() == 1);
}

void TestInfiniteTalkVideoToVideo() {
  // The film studio's lip-sync pass: the silent motion video
  // (LoadVideo -> GetVideoComponents -> VAEEncode) is the sampler's latent,
  // WanInfiniteTalkToVideo gives model / positive / negative, denoise 0.5.
  std::string v2v = Replace(
      kTalk, " \"30\": {",
      " \"60\": {\"class_type\": \"LoadVideo\", \"inputs\": {\"file\": "
      "\"anna_motion.mp4\"}},\n \"61\": {\"class_type\": "
      "\"GetVideoComponents\", \"inputs\": {\"video\": [\"60\", 0]}},\n "
      "\"62\": {\"class_type\": \"VAEEncode\", \"inputs\": {\"pixels\": "
      "[\"61\", 0], \"vae\": [\"4\", 0]}},\n \"30\": {");
  v2v = Replace(v2v, "\"latent_image\": [\"20\", 3]",
                "\"latent_image\": [\"62\", 0]");
  v2v = Replace(v2v, "\"denoise\": 1.0", "\"denoise\": 0.5");
  const comfy::Plan plan = PlanOf(v2v.c_str());
  assert(plan.tasks.size() == 1 && plan.tasks[0].node_id == "30");
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kTalk);
  assert(video.frames == 81 && video.steps == 4 && video.seed == 7);
  assert((video.first_frame == comfy::FileRef{"anna_portrait.png", "input"}));
  assert((video.talk_video == comfy::FileRef{"anna_motion.mp4", "input"}));
  assert(video.talk_denoise && *video.talk_denoise == 0.5);
  assert(plan.warnings.empty());
  const json::Value body = comfy::VideoRequestBody(video);
  const json::Value* talk = body.find("gufo")->find("talk");
  assert(talk->find("denoise")->as_double() == 0.5);
  // The executor adds the video itself.
  assert(talk->find("video") == nullptr);
  // Today's graph sends no denoise and no video.
  const comfy::Plan plain_plan = PlanOf(kTalk);
  const auto& plain = std::get<comfy::VideoTask>(plain_plan.tasks[0].spec);
  assert(!plain.talk_video && !plain.talk_denoise);
  assert(comfy::VideoRequestBody(plain).find("gufo")->find("talk")->find(
             "denoise") == nullptr);

  // VHS_LoadVideo works too.
  std::string vhs = Replace(
      v2v, "\"pixels\": [\"61\", 0]", "\"pixels\": [\"63\", 0]");
  vhs = Replace(vhs, " \"30\": {",
                " \"63\": {\"class_type\": \"VHS_LoadVideo\", \"inputs\": "
                "{\"video\": \"anna_vhs.mp4\", \"force_rate\": 0, "
                "\"skip_first_frames\": 0, \"select_every_nth\": 1}},\n "
                "\"30\": {");
  const comfy::Plan vhs_plan = PlanOf(vhs.c_str());
  const auto& from_vhs = std::get<comfy::VideoTask>(vhs_plan.tasks[0].spec);
  assert((from_vhs.talk_video == comfy::FileRef{"anna_vhs.mp4", "input"}));

  // KSamplerAdvanced from step 2 of 4: 2 steps of the 4-step schedule.
  const std::string ksampler =
      "\"30\": {\"class_type\": \"KSampler\", \"inputs\": {\"model\": "
      "[\"16\", 0], \"positive\": [\"20\", 1], \"negative\": [\"20\", 2], "
      "\"latent_image\": [\"62\", 0], \"seed\": 7, \"steps\": 4, \"cfg\": "
      "1.0, \"sampler_name\": \"lcm\", \"scheduler\": \"simple\", "
      "\"denoise\": 0.5}}";
  const std::string advanced = Replace(
      v2v, ksampler,
      "\"30\": {\"class_type\": \"KSamplerAdvanced\", \"inputs\": "
      "{\"model\": [\"16\", 0], \"positive\": [\"20\", 1], \"negative\": "
      "[\"20\", 2], \"latent_image\": [\"62\", 0], \"add_noise\": "
      "\"enable\", \"noise_seed\": 7, \"steps\": 4, \"cfg\": 1.0, "
      "\"sampler_name\": \"lcm\", \"scheduler\": \"simple\", "
      "\"start_at_step\": 2, \"end_at_step\": 10000, "
      "\"return_with_leftover_noise\": \"disable\"}}");
  const comfy::Plan late_plan = PlanOf(advanced.c_str());
  const auto& late = std::get<comfy::VideoTask>(late_plan.tasks[0].spec);
  assert(late.steps == 2 && late.talk_denoise && *late.talk_denoise == 0.5);
  assert(late.seed == 7 && late.talk_video);
  assert(ErrorOf(Replace(advanced, "\"add_noise\": \"enable\"",
                         "\"add_noise\": \"disable\""))
             .message.find("add_noise") != std::string::npos);
  assert(ErrorOf(Replace(advanced, "\"end_at_step\": 10000",
                         "\"end_at_step\": 3"))
             .message.find("end_at_step") != std::string::npos);

  // SamplerCustomAdvanced with BasicScheduler (4 steps) -> SplitSigmas
  // (step 2) low_sigmas: the same 2 steps.
  const std::string custom = Replace(
      v2v, ksampler,
      "\"70\": {\"class_type\": \"RandomNoise\", \"inputs\": "
      "{\"noise_seed\": 7}},\n \"71\": {\"class_type\": \"KSamplerSelect\", "
      "\"inputs\": {\"sampler_name\": \"lcm\"}},\n \"72\": {\"class_type\": "
      "\"BasicScheduler\", \"inputs\": {\"model\": [\"16\", 0], "
      "\"scheduler\": \"simple\", \"steps\": 4, \"denoise\": 1.0}},\n "
      "\"73\": {\"class_type\": \"SplitSigmas\", \"inputs\": {\"sigmas\": "
      "[\"72\", 0], \"step\": 2}},\n \"74\": {\"class_type\": "
      "\"CFGGuider\", \"inputs\": {\"model\": [\"16\", 0], \"positive\": "
      "[\"20\", 1], \"negative\": [\"20\", 2], \"cfg\": 1.0}},\n \"30\": "
      "{\"class_type\": \"SamplerCustomAdvanced\", \"inputs\": {\"noise\": "
      "[\"70\", 0], \"guider\": [\"74\", 0], \"sampler\": [\"71\", 0], "
      "\"sigmas\": [\"73\", 1], \"latent_image\": [\"62\", 0]}}");
  const comfy::Plan custom_plan = PlanOf(custom.c_str());
  const auto& split = std::get<comfy::VideoTask>(custom_plan.tasks[0].spec);
  assert(split.steps == 2 && split.talk_denoise &&
         *split.talk_denoise == 0.5);
  assert((split.talk_video == comfy::FileRef{"anna_motion.mp4", "input"}));
  // BasicScheduler denoise 0.5 (the last 4 of 8 steps) split at 1: the
  // last 3 steps of 8.
  const comfy::Plan deeper_plan =
      PlanOf(Replace(Replace(custom, "\"steps\": 4, \"denoise\": 1.0",
                             "\"steps\": 4, \"denoise\": 0.5"),
                     "\"step\": 2", "\"step\": 1")
                 .c_str());
  const auto& deeper = std::get<comfy::VideoTask>(deeper_plan.tasks[0].spec);
  assert(deeper.steps == 3 && *deeper.talk_denoise == 3.0 / 8.0);
  // Only low_sigmas, and a step inside the schedule.
  assert(ErrorOf(Replace(custom, "\"sigmas\": [\"73\", 1]",
                         "\"sigmas\": [\"73\", 0]"))
             .message.find("low_sigmas") != std::string::npos);
  assert(ErrorOf(Replace(custom, "\"step\": 2", "\"step\": 4"))
             .message.find("SplitSigmas step") != std::string::npos);
  const auto& classes = comfy::SupportedClassTypes();
  assert(std::find(classes.begin(), classes.end(), "SplitSigmas") !=
         classes.end());

  // denoise 1 ignores the video (warning, nothing sent).
  const comfy::Plan full_plan =
      PlanOf(Replace(v2v, "\"denoise\": 0.5", "\"denoise\": 1.0").c_str());
  const auto& full = std::get<comfy::VideoTask>(full_plan.tasks[0].spec);
  assert(!full.talk_video && !full.talk_denoise);
  assert(full_plan.warnings.size() == 1 &&
         full_plan.warnings[0].find("pure noise") != std::string::npos);

  // Errors: denoise < 1 on the empty InfiniteTalk latent, a still image as
  // the video, a previous_frames chain.
  assert(ErrorOf(Replace(kTalk, "\"denoise\": 1.0", "\"denoise\": 0.5"))
             .message.find("VAEEncode") != std::string::npos);
  assert(ErrorOf(Replace(v2v, "\"pixels\": [\"61\", 0]",
                         "\"pixels\": [\"10\", 0]"))
             .message.find("LoadVideo") != std::string::npos);
  std::string chained = Replace(
      v2v, " \"30\": {",
      " \"80\": {\"class_type\": \"WanInfiniteTalkToVideo\", \"inputs\": "
      "{\"mode\": \"single_speaker\", \"model\": [\"2\", 0], "
      "\"model_patch\": [\"6\", 0], \"positive\": [\"12\", 0], "
      "\"negative\": [\"13\", 0], \"vae\": [\"4\", 0], \"width\": 832, "
      "\"height\": 480, \"length\": 81, \"clip_vision_output\": [\"11\", 0], "
      "\"start_image\": [\"10\", 0], \"audio_encoder_output_1\": [\"9\", 0], "
      "\"motion_frame_count\": 9, \"audio_scale\": 1.0}},\n \"81\": "
      "{\"class_type\": \"VAEDecode\", \"inputs\": {\"samples\": [\"80\", 3], "
      "\"vae\": [\"4\", 0]}},\n \"30\": {");
  chained = Replace(chained, "\"audio_scale\": 1.0}},\n \"60\"",
                    "\"audio_scale\": 1.0, \"previous_frames\": [\"81\", "
                    "0]}},\n \"60\"");
  assert(ErrorOf(chained).message.find("previous_frames") !=
         std::string::npos);

  // The start image as frame 0 of the motion video: ImageFromBatch
  // (length 1) over GetVideoComponents <- LoadVideo; the executor extracts
  // the frame with ffmpeg.
  std::string from_frame = Replace(
      v2v, "\"start_image\": [\"10\", 0]", "\"start_image\": [\"64\", 0]");
  from_frame = Replace(
      from_frame, " \"30\": {",
      " \"64\": {\"class_type\": \"ImageFromBatch\", \"inputs\": "
      "{\"image\": [\"61\", 0], \"batch_index\": 0, \"length\": 1}},\n "
      "\"30\": {");
  const comfy::Plan frame_plan = PlanOf(from_frame.c_str());
  const auto& framed = std::get<comfy::VideoTask>(frame_plan.tasks[0].spec);
  assert((framed.first_frame == comfy::FileRef{"anna_motion.mp4", "input"}));
  assert(framed.first_frame_index && *framed.first_frame_index == 0);
  assert((framed.talk_video == comfy::FileRef{"anna_motion.mp4", "input"}));
  assert(frame_plan.warnings.empty());
  assert(!video.first_frame_index);
  // Frame 12 of a VHS_LoadVideo.
  const comfy::Plan vhs_frame_plan = PlanOf(
      Replace(Replace(from_frame, "\"image\": [\"61\", 0], \"batch_index\": 0",
                      "\"image\": [\"65\", 0], \"batch_index\": 12"),
              " \"30\": {",
              " \"65\": {\"class_type\": \"VHS_LoadVideo\", \"inputs\": "
              "{\"video\": \"anna_vhs.mp4\"}},\n \"30\": {")
          .c_str());
  const auto& vhs_frame =
      std::get<comfy::VideoTask>(vhs_frame_plan.tasks[0].spec);
  assert((vhs_frame.first_frame == comfy::FileRef{"anna_vhs.mp4", "input"}));
  assert(*vhs_frame.first_frame_index == 12);
  // Only one frame, from a video.
  assert(ErrorOf(Replace(from_frame, "\"length\": 1}", "\"length\": 2}"))
             .message.find("length 1") != std::string::npos);
  assert(ErrorOf(Replace(from_frame, "\"image\": [\"61\", 0], \"batch_index\"",
                         "\"image\": [\"10\", 0], \"batch_index\""))
             .message.find("LoadVideo") != std::string::npos);
}

// ImageFromBatch between a loaded video and the node that reads it
// (Regie 19.11., Weg 2 of the film pipeline): FileRef::cuts, cut by the
// executor with ComfyUI's clamping.
void TestVideoCuts() {
  // ComfyUI ImageFromBatch: batch_index = min(n - 1, batch_index),
  // length = min(n - batch_index, length).
  using Cuts = std::vector<comfy::FrameCut>;
  assert((comfy::ResolveFrameCuts(285, Cuts{{0, 285}}) ==
          std::pair<int, int>{0, 285}));
  assert((comfy::ResolveFrameCuts(285, Cuts{}) ==
          std::pair<int, int>{0, 285}));
  assert((comfy::ResolveFrameCuts(200, Cuts{{0, 285}}) ==
          std::pair<int, int>{0, 200}));
  assert((comfy::ResolveFrameCuts(285, Cuts{{100, 81}}) ==
          std::pair<int, int>{100, 81}));
  assert((comfy::ResolveFrameCuts(285, Cuts{{100, 81}, {10, 5}}) ==
          std::pair<int, int>{110, 5}));
  // Past the end: the last frame (index), the rest of the batch (length).
  assert((comfy::ResolveFrameCuts(285, Cuts{{400, 10}}) ==
          std::pair<int, int>{284, 1}));
  assert((comfy::ResolveFrameCuts(285, Cuts{{250, 81}}) ==
          std::pair<int, int>{250, 35}));
  assert((comfy::ResolveFrameCuts(285, Cuts{{100, 81}, {90, 4}}) ==
          std::pair<int, int>{180, 1}));
  assert((comfy::ResolveFrameCuts(0, Cuts{{0, 1}}) ==
          std::pair<int, int>{0, 0}));

  std::string v2v = Replace(
      kTalk, " \"30\": {",
      " \"60\": {\"class_type\": \"LoadVideo\", \"inputs\": {\"file\": "
      "\"anna_motion.mp4\"}},\n \"61\": {\"class_type\": "
      "\"GetVideoComponents\", \"inputs\": {\"video\": [\"60\", 0]}},\n "
      "\"66\": {\"class_type\": \"ImageFromBatch\", \"inputs\": {\"image\": "
      "[\"61\", 0], \"batch_index\": 0, \"length\": 285}},\n "
      "\"62\": {\"class_type\": \"VAEEncode\", \"inputs\": {\"pixels\": "
      "[\"66\", 0], \"vae\": [\"4\", 0]}},\n \"30\": {");
  v2v = Replace(v2v, "\"latent_image\": [\"20\", 3]",
                "\"latent_image\": [\"62\", 0]");
  v2v = Replace(v2v, "\"denoise\": 1.0", "\"denoise\": 0.5");
  // The Regie's graph: ImageFromBatch(0, 285) of a 285-frame video -> the
  // whole video (the executor finds nothing to cut).
  const comfy::Plan plan = PlanOf(v2v.c_str());
  const auto& whole = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert((whole.talk_video ==
          comfy::FileRef{"anna_motion.mp4", "input", Cuts{{0, 285}}}));
  assert(plan.warnings.empty());
  assert(comfy::HasFrameCuts(plan.tasks[0]));
  assert(!comfy::HasFrameCuts(PlanOf(kTalk).tasks[0]));
  // The start frame f0 = ImageFromBatch(0, 1) of the same frames: straight
  // on GetVideoComponents (frame 0 of the file) ...
  std::string f0 = Replace(v2v, "\"start_image\": [\"10\", 0]",
                           "\"start_image\": [\"64\", 0]");
  f0 = Replace(f0, " \"30\": {",
               " \"64\": {\"class_type\": \"ImageFromBatch\", \"inputs\": "
               "{\"image\": [\"61\", 0], \"batch_index\": 0, \"length\": "
               "1}},\n \"30\": {");
  const comfy::Plan f0_plan = PlanOf(f0.c_str());
  const auto& direct = std::get<comfy::VideoTask>(f0_plan.tasks[0].spec);
  assert((direct.first_frame == comfy::FileRef{"anna_motion.mp4", "input"}));
  assert(direct.first_frame_index && *direct.first_frame_index == 0);
  assert(direct.talk_video->cuts == (Cuts{{0, 285}}));
  // ... or on the cut the sampler encodes: the whole chain is the cut.
  const comfy::Plan f0_cut_plan = PlanOf(
      Replace(f0, "{\"image\": [\"61\", 0], \"batch_index\": 0, "
                  "\"length\": 1}",
              "{\"image\": [\"66\", 0], \"batch_index\": 0, \"length\": 1}")
          .c_str());
  const auto& chained =
      std::get<comfy::VideoTask>(f0_cut_plan.tasks[0].spec);
  assert((chained.first_frame ==
          comfy::FileRef{"anna_motion.mp4", "input", Cuts{{0, 285}, {0, 1}}}));
  assert(chained.first_frame_index && *chained.first_frame_index == 0);
  assert(f0_cut_plan.warnings.empty());

  // A real cut in the middle, through two ImageFromBatch nodes.
  std::string middle = Replace(v2v, "\"batch_index\": 0, \"length\": 285",
                               "\"batch_index\": 100, \"length\": 81");
  middle =
      Replace(middle, "\"pixels\": [\"66\", 0]", "\"pixels\": [\"67\", 0]");
  middle = Replace(middle, " \"30\": {",
                   " \"67\": {\"class_type\": \"ImageFromBatch\", \"inputs\": "
                   "{\"image\": [\"66\", 0], \"batch_index\": 4, \"length\": "
                   "200}},\n \"30\": {");
  const comfy::Plan middle_plan = PlanOf(middle.c_str());
  const auto& cut = std::get<comfy::VideoTask>(middle_plan.tasks[0].spec);
  assert(cut.talk_video->cuts == (Cuts{{100, 81}, {4, 200}}));
  assert((comfy::ResolveFrameCuts(285, cut.talk_video->cuts) ==
          std::pair<int, int>{104, 77}));
  // An index past the end is ComfyUI's to clamp (no plan error).
  const comfy::Plan past_plan =
      PlanOf(Replace(v2v, "\"batch_index\": 0, \"length\": 285",
                     "\"batch_index\": 4000, \"length\": 10")
                 .c_str());
  const auto& past = std::get<comfy::VideoTask>(past_plan.tasks[0].spec);
  assert(past.talk_video->cuts == (Cuts{{4000, 10}}));
  assert((comfy::ResolveFrameCuts(285, past.talk_video->cuts) ==
          std::pair<int, int>{284, 1}));
  // Out of ImageFromBatch's own range, or not a whole number.
  assert(ErrorOf(Replace(v2v, "\"length\": 285", "\"length\": 0"))
             .message.find("length") != std::string::npos);
  assert(ErrorOf(Replace(v2v, "\"batch_index\": 0, \"length\": 285",
                         "\"batch_index\": 1.5, \"length\": 285"))
             .message.find("batch_index") != std::string::npos);
  // Still a loaded video behind the cut: not a LoadImage.
  assert(ErrorOf(Replace(v2v, "{\"image\": [\"61\", 0], \"batch_index\": 0",
                         "{\"image\": [\"10\", 0], \"batch_index\": 0"))
             .message.find("LoadVideo") != std::string::npos);
  // VHS_LoadVideo behind the cut works the same.
  std::string vhs =
      Replace(v2v, "{\"image\": [\"61\", 0], \"batch_index\": 0",
              "{\"image\": [\"63\", 0], \"batch_index\": 0");
  vhs = Replace(vhs, " \"30\": {",
                " \"63\": {\"class_type\": \"VHS_LoadVideo\", \"inputs\": "
                "{\"video\": \"anna_vhs.mp4\"}},\n \"30\": {");
  const comfy::Plan vhs_plan = PlanOf(vhs.c_str());
  assert((std::get<comfy::VideoTask>(vhs_plan.tasks[0].spec).talk_video ==
          comfy::FileRef{"anna_vhs.mp4", "input", Cuts{{0, 285}}}));

  // MMAudio on a cut of a LoadVideo.
  const comfy::Plan foley_plan = PlanOf(R"({
 "1": {"class_type": "MMAudioModelLoader", "inputs": {"mmaudio_model": "mmaudio_large_44k_v2_fp16.safetensors", "base_precision": "fp16"}},
 "2": {"class_type": "MMAudioFeatureUtilsLoader", "inputs": {"vae_model": "mmaudio_vae_44k_fp16.safetensors", "synchformer_model": "mmaudio_synchformer_fp16.safetensors", "clip_model": "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors", "mode": "44k", "precision": "fp16"}},
 "3": {"class_type": "LoadVideo", "inputs": {"file": "shot16fps.mp4"}},
 "7": {"class_type": "GetVideoComponents", "inputs": {"video": ["3", 0]}},
 "8": {"class_type": "ImageFromBatch", "inputs": {"image": ["7", 0], "batch_index": 16, "length": 48}},
 "4": {"class_type": "MMAudioSampler", "inputs": {"mmaudio_model": ["1", 0], "feature_utils": ["2", 0], "duration": 3, "steps": 25, "cfg": 4.5, "seed": 1, "prompt": "rain", "negative_prompt": "", "mask_away_clip": false, "force_offload": true, "images": ["8", 0]}},
 "5": {"class_type": "SaveAudio", "inputs": {"audio": ["4", 0], "filename_prefix": "ton/s16"}}
})");
  const auto& foley = std::get<comfy::FoleyTask>(foley_plan.tasks[0].spec);
  assert((foley.video ==
          comfy::FileRef{"shot16fps.mp4", "input", Cuts{{16, 48}}}));
  assert(comfy::HasFrameCuts(foley_plan.tasks[0]));

  // Uni3C render_video through ImageFromBatch.
  std::string rendered = Replace(
      kUni3C,
      "\"14\": {\"class_type\": \"GufoCameraMove\", \"inputs\": {\"camera\": "
      "\"orbit_left\", \"speed\": 0.5, \"ease\": \"ease_in_out\", "
      "\"amount\": -1}}",
      "\"14\": {\"class_type\": \"ImageFromBatch\", \"inputs\": {\"image\": "
      "[\"18\", 0], \"batch_index\": 8, \"length\": 81}},\n \"18\": "
      "{\"class_type\": \"GetVideoComponents\", \"inputs\": "
      "{\"video\": [\"17\", 0]}},\n \"17\": {\"class_type\": \"LoadVideo\", "
      "\"inputs\": {\"file\": \"render_shot07.mp4\"}}");
  const comfy::Plan guided_plan = PlanOf(rendered.c_str());
  assert(
      (std::get<comfy::VideoTask>(guided_plan.tasks[0].spec).render_video ==
       comfy::FileRef{"render_shot07.mp4", "input", Cuts{{8, 81}}}));
}

void TestWanFunCamera() {
  // ComfyUI's Wan Fun Camera form: WanCameraEmbedding -> WanCameraImageToVideo.
  std::string text = Replace(
      kUni3C, "\"model\": [\"15\", 0], \"shift\": 5",
      "\"model\": [\"2\", 0], \"shift\": 5");
  text = Replace(
      text,
      "\"20\": {\"class_type\": \"WanImageToVideo\", \"inputs\": "
      "{\"positive\": [\"12\", 0], \"negative\": [\"13\", 0], \"vae\": "
      "[\"4\", 0], \"width\": 832, \"height\": 480, \"length\": 81, "
      "\"batch_size\": 1,",
      "\"18\": {\"class_type\": \"WanCameraEmbedding\", \"inputs\": "
      "{\"camera_pose\": \"Zoom In\", \"width\": 832, \"height\": 480, "
      "\"length\": 81, \"speed\": 2.0}},\n \"20\": {\"class_type\": "
      "\"WanCameraImageToVideo\", \"inputs\": {\"positive\": [\"12\", 0], "
      "\"negative\": [\"13\", 0], \"vae\": [\"4\", 0], \"width\": 832, "
      "\"height\": 480, \"length\": 81, \"batch_size\": 1, "
      "\"camera_conditions\": [\"18\", 0],");
  // Nodes 6, 14 and 15 stay in the graph unconnected: harmless plumbing.
  const comfy::Plan plan = PlanOf(text.c_str());
  const auto& video = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(video.family == comfy::VideoFamily::kCamera);
  assert(video.camera && video.camera->type == "dolly_in");
  assert(video.camera->speed == 2.0);
  assert(plan.warnings.size() == 1 &&
         plan.warnings[0].find("Fun Camera") != std::string::npos);
  // Roll has no gufo preset.
  assert(ErrorOf(Replace(text, "\"Zoom In\"", "\"ClockWise (CW)\""))
             .message.find("GufoCameraMove") != std::string::npos);
  // Without a camera it is plain image to video on wan2.1-uni3c.
  const comfy::Plan plain = PlanOf(
      Replace(text, ", \"camera_conditions\": [\"18\", 0]", "").c_str());
  const auto& still = std::get<comfy::VideoTask>(plain.tasks[0].spec);
  assert(!still.camera && !still.render_video);
  assert(comfy::VideoRequestBody(still).find("gufo")->find("camera") ==
         nullptr);
}

// FilmStudio delivery after Wan 2.2: RIFE x2 (ComfyUI-Frame-Interpolation),
// Real-ESRGAN x4 + ImageScale to 1920x1080, VideoHelperSuite at 60 fps.
constexpr const char* kWanDelivery = R"({
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_ti2v_5B_fp16.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
 "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
 "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "a fox in the snow"}},
 "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
 "55": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"vae": ["3", 0], "width": 1280, "height": 704, "length": 49, "batch_size": 1}},
 "3s": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["55", 0], "seed": 42, "steps": 20, "cfg": 5, "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1}},
 "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3s", 0], "vae": ["3", 0]}},
 "60": {"class_type": "RIFE VFI", "inputs": {"ckpt_name": "rife49.pth", "frames": ["8", 0], "clear_cache_after_n_frames": 10, "multiplier": 2, "fast_mode": true, "ensemble": true, "scale_factor": 1.0}},
 "61": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "realesr-animevideov3.pth"}},
 "62": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["61", 0], "image": ["60", 0]}},
 "63": {"class_type": "ImageScale", "inputs": {"image": ["62", 0], "upscale_method": "lanczos", "width": 1920, "height": 1080, "crop": "center"}},
 "64": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["63", 0], "frame_rate": 60, "loop_count": 0, "filename_prefix": "film/lieferung/shot01", "format": "video/h264-mp4", "pingpong": false, "save_output": true}}
})";

void TestDelivery() {
  const comfy::Plan plan = PlanOf(kWanDelivery);
  const auto& wan = std::get<comfy::VideoTask>(plan.tasks[0].spec);
  assert(wan.interpolate == 2 && wan.fps_num == 60 && wan.fps_den == 1);
  assert(wan.fps == 60);
  assert(wan.upscale && wan.upscale->width == 1920 &&
         wan.upscale->height == 1080);
  assert(wan.upscale->fit == "crop" &&
         wan.upscale->model == "realesr-animevideov3");
  const auto& out = plan.tasks[0].outputs[0];
  assert(out.class_type == "VHS_VideoCombine" &&
         out.kind == comfy::OutputKind::kVideo && !out.temp);
  assert(out.prefix == "film/lieferung/shot01");
  const json::Value body = comfy::VideoRequestBody(wan);
  const json::Value* gufo = body.find("gufo");
  assert(gufo->find("interpolate")->find("fps")->as_double() == 60);
  const json::Value* upscale = gufo->find("upscale");
  assert(upscale->find("width")->as_double() == 1920);
  assert(upscale->find("height")->as_double() == 1080);
  assert(upscale->member_str("fit") == "crop");
  assert(upscale->member_str("model") == "realesr-animevideov3");
  bool speed_note = false;
  for (const auto& warning : plan.warnings)
    speed_note = speed_note || warning.find("resamples") != std::string::npos;
  assert(speed_note);
  // 59.94 is NTSC 60000/1001.
  const comfy::Plan ntsc = PlanOf(
      Replace(kWanDelivery, "\"frame_rate\": 60", "\"frame_rate\": 59.94")
          .c_str());
  const auto& ntsc_video = std::get<comfy::VideoTask>(ntsc.tasks[0].spec);
  assert(ntsc_video.fps_num == 60000 && ntsc_video.fps_den == 1001);
  assert(comfy::VideoRequestBody(ntsc_video)
             .find("gufo")
             ->find("interpolate")
             ->member_str("fps") == "60000/1001");
  // frame_rate = 24 * multiplier keeps the plain multiplier.
  const comfy::Plan plain = PlanOf(
      Replace(kWanDelivery, "\"frame_rate\": 60", "\"frame_rate\": 48")
          .c_str());
  const auto& doubled = std::get<comfy::VideoTask>(plain.tasks[0].spec);
  assert(doubled.fps_num == 0 && doubled.interpolate == 2);
  assert(comfy::VideoRequestBody(doubled)
             .find("gufo")
             ->find("interpolate")
             ->as_double() == 2);
  // Without ImageScale the x4 size, stretched; ImageScaleBy scales it; a
  // zero side keeps the aspect ratio.
  std::string bare = Replace(
      kWanDelivery,
      "\"63\": {\"class_type\": \"ImageScale\", \"inputs\": {\"image\": "
      "[\"62\", 0], \"upscale_method\": \"lanczos\", \"width\": 1920, "
      "\"height\": 1080, \"crop\": \"center\"}},\n",
      "");
  bare = Replace(bare, "\"images\": [\"63\", 0]", "\"images\": [\"62\", 0]");
  const comfy::Plan x4 = PlanOf(bare.c_str());
  const auto& big = std::get<comfy::VideoTask>(x4.tasks[0].spec);
  assert(big.upscale->width == 5120 && big.upscale->height == 2816 &&
         big.upscale->fit == "stretch");
  const comfy::Plan by = PlanOf(
      Replace(kWanDelivery,
              "\"ImageScale\", \"inputs\": {\"image\": [\"62\", 0], "
              "\"upscale_method\": \"lanczos\", \"width\": 1920, \"height\": "
              "1080, \"crop\": \"center\"}",
              "\"ImageScaleBy\", \"inputs\": {\"image\": [\"62\", 0], "
              "\"upscale_method\": \"lanczos\", \"scale_by\": 0.375}")
          .c_str());
  const auto& scaled = std::get<comfy::VideoTask>(by.tasks[0].spec);
  assert(scaled.upscale->width == 1920 && scaled.upscale->height == 1056);
  const comfy::Plan aspect = PlanOf(
      Replace(kWanDelivery, "\"height\": 1080, \"crop\": \"center\"",
              "\"height\": 0, \"crop\": \"disabled\"")
          .c_str());
  const auto& kept = std::get<comfy::VideoTask>(aspect.tasks[0].spec);
  assert(kept.upscale->width == 1920 && kept.upscale->height == 1056 &&
         kept.upscale->fit == "stretch");
  // general-x4v3 is served; AnimeSharp (non-commercial) is not.
  const comfy::Plan general = PlanOf(
      Replace(kWanDelivery, "realesr-animevideov3.pth",
              "realesr-general-x4v3.pth")
          .c_str());
  assert(std::get<comfy::VideoTask>(general.tasks[0].spec).upscale->model ==
         "realesr-general-x4v3");
  const comfy::PlanError sharp = ErrorOf(
      Replace(kWanDelivery, "realesr-animevideov3.pth",
              "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors"));
  assert(sharp.message.find("AnimeSharp") != std::string::npos);
  assert(sharp.message.find("not served by gufo") != std::string::npos);
  assert(sharp.node_errors.find("61") != nullptr);
  // A frame rate without RIFE is only a warning; ImageScale without an
  // upscaler and an upscale of a still image are errors.
  std::string no_rife =
      Replace(kWanDelivery, "\"image\": [\"60\", 0]", "\"image\": [\"8\", 0]");
  no_rife = Replace(
      no_rife,
      " \"60\": {\"class_type\": \"RIFE VFI\", \"inputs\": {\"ckpt_name\": "
      "\"rife49.pth\", \"frames\": [\"8\", 0], "
      "\"clear_cache_after_n_frames\": 10, \"multiplier\": 2, \"fast_mode\": "
      "true, \"ensemble\": true, \"scale_factor\": 1.0}},\n",
      "");
  const comfy::Plan unsampled = PlanOf(no_rife.c_str());
  const auto& as_is = std::get<comfy::VideoTask>(unsampled.tasks[0].spec);
  assert(as_is.fps_num == 0 && as_is.interpolate == 1 && as_is.upscale);
  assert(!unsampled.warnings.empty());
  assert(ErrorOf(Replace(kWanDelivery, "\"image\": [\"62\", 0]",
                         "\"image\": [\"60\", 0]"))
             .message.find("ImageUpscaleWithModel") != std::string::npos);
  assert(ErrorOf(Replace(kQwenText, "\"images\": [\"7\", 0]",
                         "\"images\": [\"9\", 0]}}, \"9\": {\"class_type\": "
                         "\"ImageUpscaleWithModel\", \"inputs\": "
                         "{\"upscale_model\": [\"10\", 0], \"image\": [\"7\", "
                         "0]}}, \"10\": {\"class_type\": "
                         "\"UpscaleModelLoader\", \"inputs\": {\"model_name\": "
                         "\"realesr-animevideov3.pth\""))
             .message.find("video") != std::string::npos);
  // H3: RIFE x4 + CreateVideo at 96 fps stays the multiplier; the camera
  // family upscales its 832x480 x4 picture with ImageScaleBy 0.5.
  const std::string h3_rife = Replace(
      kH3,
      "\"40\": {\"class_type\": \"SaveImage\", \"inputs\": {\"images\": "
      "[\"32\", 0]",
      "\"39\": {\"class_type\": \"RIFE VFI\", \"inputs\": {\"frames\": "
      "[\"32\", 0], \"multiplier\": 4}},\n \"38\": {\"class_type\": "
      "\"CreateVideo\", \"inputs\": {\"images\": [\"39\", 0], \"fps\": 96}},\n "
      "\"40\": {\"class_type\": \"SaveVideo\", \"inputs\": {\"video\": "
      "[\"38\", 0]");
  const comfy::Plan h3 = PlanOf(h3_rife.c_str());
  const auto& h3_video = std::get<comfy::VideoTask>(h3.tasks[0].spec);
  assert(h3_video.interpolate == 4 && h3_video.fps_num == 0 &&
         !h3_video.upscale);
  const comfy::Plan h3_60 =
      PlanOf(Replace(h3_rife, "\"fps\": 96", "\"fps\": 60").c_str());
  assert(std::get<comfy::VideoTask>(h3_60.tasks[0].spec).fps_num == 60);
  const std::string camera = Replace(
      kUni3C,
      "\"32\": {\"class_type\": \"CreateVideo\", \"inputs\": {\"images\": "
      "[\"31\", 0]",
      "\"34\": {\"class_type\": \"UpscaleModelLoader\", \"inputs\": "
      "{\"model_name\": \"realesr-general-x4v3.pth\"}},\n \"35\": "
      "{\"class_type\": \"ImageUpscaleWithModel\", \"inputs\": "
      "{\"upscale_model\": [\"34\", 0], \"image\": [\"31\", 0]}},\n \"36\": "
      "{\"class_type\": \"ImageScaleBy\", \"inputs\": {\"image\": [\"35\", "
      "0], \"upscale_method\": \"lanczos\", \"scale_by\": 0.5}},\n \"32\": "
      "{\"class_type\": \"CreateVideo\", \"inputs\": {\"images\": [\"36\", "
      "0]");
  const comfy::Plan cam = PlanOf(camera.c_str());
  const auto& cam_video = std::get<comfy::VideoTask>(cam.tasks[0].spec);
  assert(cam_video.upscale->width == 1664 && cam_video.upscale->height == 960);
  assert(cam_video.upscale->model == "realesr-general-x4v3");
  assert(comfy::VideoRequestBody(cam_video)
             .find("gufo")
             ->find("upscale")
             ->find("width")
             ->as_double() == 1664);
}

void TestWavJoin() {
  comfy::Wav a{24000, 1, std::string(4, '\x01')};
  comfy::Wav b{24000, 1, std::string(2, '\x02')};
  const auto joined =
      comfy::JoinWavs({comfy::BuildWav(a), comfy::BuildWav(b)}, 0.001);
  assert(joined);
  const auto parsed = comfy::ParseWav(*joined);
  assert(parsed && parsed->sample_rate == 24000 && parsed->channels == 1);
  // 4 bytes + 24 silent samples (48 bytes) + 2 bytes.
  assert(parsed->pcm.size() == 4 + 48 + 2);
  assert(parsed->pcm.substr(4, 48) == std::string(48, '\0'));
  comfy::Wav other{16000, 1, "xx"};
  assert(!comfy::JoinWavs({comfy::BuildWav(a), comfy::BuildWav(other)}, 0));
  assert(!comfy::ParseWav("RIFF....WAVE"));
}

void TestHelpers() {
  assert(comfy::SafeRelative("a\\b/c.png") ==
         std::optional<std::string>("a/b/c.png"));
  assert(!comfy::SafeRelative("../x"));
  assert(!comfy::SafeRelative("a/../../x"));
  assert(!comfy::SafeRelative("/etc/passwd"));
  assert(!comfy::SafeRelative("C:/Windows"));
  assert(comfy::SafeRelative("") == std::optional<std::string>(""));
  assert((comfy::ScaleToArea(1024, 1024, 1024) ==
          std::pair<int, int>{1024, 1024}));
  assert(
      (comfy::ScaleToArea(1344, 832, 1024) == std::pair<int, int>{1312, 800}));
  assert((comfy::ScaleToArea(1000, 600, 0) == std::pair<int, int>{992, 608}));
  assert(comfy::SnapH3Frames(124) == 124);
  assert(comfy::SnapH3Frames(125) == 141);
  assert(comfy::SnapH3Frames(22) == 22);
  assert(comfy::SnapH3Frames(1) == 5);
  std::string png("\x89PNG\r\n\x1a\n\0\0\0\rIHDR\0\0\x05\x40\0\0\x03\x40", 24);
  assert((comfy::ImageDimensions(png) == std::pair<int, int>{1344, 832}));
  const std::string jpeg(
      "\xFF\xD8\xFF\xE0\x00\x04\x00\x00\xFF\xC0\x00\x11\x08\x02\x00\x03\x40",
      17);
  assert((comfy::ImageDimensions(jpeg) == std::pair<int, int>{832, 512}));
  assert(!comfy::ImageDimensions("nope"));
  const std::string bytes("\x00\x01\xFE\xFFhello", 9);
  assert(comfy::Base64Decode(comfy::Base64Encode(bytes)) == bytes);
  assert(comfy::Base64Encode("ab") == "YWI=");
  assert(!comfy::Base64Decode("**"));
  const auto parts = std::vector<comfy::MultipartPart>{
      {"subfolder", "", "", "x"},
      {"image", "a b.png", "image/png", std::string("\r\n--\0bin", 8)}};
  std::vector<comfy::MultipartPart> parsed;
  std::string error;
  assert(comfy::ParseMultipart("multipart/form-data; boundary=XyZ",
                               comfy::BuildMultipart(parts, "XyZ"), &parsed,
                               &error));
  assert(parsed.size() == 2 && parsed[1].filename == "a b.png");
  assert(parsed[1].data == parts[1].data && parsed[0].data == "x");
  const auto name = comfy::SplitPrefix("life3/stimmen/alia");
  assert(name && name->subfolder == "life3/stimmen" && name->base == "alia");
  assert(!comfy::SplitPrefix("../escape"));
  assert(comfy::NextCounter({"alia_00001.flac", "alia_00007_.png",
                             "alia_x_00009_.png", "other_00020_.png"},
                            "alia") == 8);
  assert(comfy::OutputFilename("kf", 3, comfy::OutputKind::kImage, "png") ==
         "kf_00003_.png");
  assert(comfy::OutputFilename("alia", 1, comfy::OutputKind::kAudio, "flac") ==
         "alia_00001.flac");
  assert((comfy::ParseFileRef("sub\\a.png [temp]", "input") ==
          comfy::FileRef{"sub/a.png", "temp"}));
}

}  // namespace

// ComfyUI /ws preview frames: big-endian u32 type, then format or metadata
// length, then the bytes (server.py BinaryEventTypes 1 and 4).
void TestPreviewFrames() {
  const std::string jpeg("\xFF\xD8\xFF\xE0jpeg", 8);
  const std::string one = comfy::PreviewImageFrame(jpeg);
  assert(one.size() == 8 + jpeg.size());
  assert(one.substr(0, 8) == std::string("\0\0\0\x01\0\0\0\x01", 8));
  assert(one.substr(8) == jpeg);
  const std::string png = comfy::PreviewImageFrame("x", comfy::kPreviewPng);
  assert(png.substr(4, 4) == std::string("\0\0\0\x02", 4));
  const std::string meta = R"({"node_id":"6","prompt_id":"p"})";
  const std::string four = comfy::PreviewMetadataFrame(jpeg, meta);
  assert(four.substr(0, 4) == std::string("\0\0\0\x04", 4));
  const auto length = (static_cast<unsigned char>(four[4]) << 24) |
                      (static_cast<unsigned char>(four[5]) << 16) |
                      (static_cast<unsigned char>(four[6]) << 8) |
                      static_cast<unsigned char>(four[7]);
  assert(length == meta.size());
  assert(four.substr(8, meta.size()) == meta);
  assert(four.substr(8 + meta.size()) == jpeg);
  // Lengths above 255 use the higher bytes.
  const std::string long_meta(300, 'm');
  const std::string big = comfy::PreviewMetadataFrame("", long_meta);
  assert(big.substr(4, 4) == std::string("\0\0\x01\x2C", 4));
  assert(comfy::PreviewId("a1b2-c3", "12:4") == "a1b2-c3_12_4");
  assert(comfy::PreviewId(std::string(200, 'x'), "1").size() == 128);
}

// MMAudio as kijai/ComfyUI-MMAudio wires it (node signatures from its
// nodes.py): VHS_LoadVideo frames into MMAudioSampler, the sound to
// SaveAudio and, with the frames, to VHS_VideoCombine.
constexpr const char* kMmaudio = R"({
 "1": {"class_type": "MMAudioModelLoader", "inputs": {"mmaudio_model": "mmaudio_large_44k_v2_fp16.safetensors", "base_precision": "fp16"}},
 "2": {"class_type": "MMAudioFeatureUtilsLoader", "inputs": {"vae_model": "mmaudio_vae_44k_fp16.safetensors", "synchformer_model": "mmaudio_synchformer_fp16.safetensors", "clip_model": "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors", "mode": "44k", "precision": "fp16"}},
 "3": {"class_type": "VHS_LoadVideo", "inputs": {"video": "shot07.mp4", "force_rate": 25, "custom_width": 0, "custom_height": 0, "frame_load_cap": 0, "skip_first_frames": 0, "select_every_nth": 1}},
 "4": {"class_type": "MMAudioSampler", "inputs": {"mmaudio_model": ["1", 0], "feature_utils": ["2", 0], "duration": 8, "steps": 25, "cfg": 4.5, "seed": 42, "prompt": "footsteps on gravel, a door slams", "negative_prompt": "music", "mask_away_clip": false, "force_offload": true, "images": ["3", 0]}},
 "5": {"class_type": "SaveAudio", "inputs": {"audio": ["4", 0], "filename_prefix": "film/ton/shot07"}},
 "6": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["3", 0], "audio": ["4", 0], "frame_rate": 25, "loop_count": 0, "filename_prefix": "film/ton/shot07_mux", "format": "video/h264-mp4", "pingpong": false, "save_output": true}}
})";

void TestFoley() {
  const comfy::Plan plan = PlanOf(kMmaudio);
  assert(plan.tasks.size() == 1);
  const auto& task = plan.tasks[0];
  assert(task.node_id == "4" && task.class_type == "MMAudioSampler");
  const auto& foley = std::get<comfy::FoleyTask>(task.spec);
  assert(foley.model == "mmaudio-44k");
  assert(foley.prompt == "footsteps on gravel, a door slams");
  assert(foley.negative_prompt == "music");
  assert(foley.duration == 8 && foley.steps == 25 && foley.cfg == 4.5);
  assert(foley.seed == 42 && !foley.mask_away_clip);
  assert((foley.video == comfy::FileRef{"shot07.mp4", "input"}));
  assert(!foley.image && foley.force_rate == 25);
  assert(task.outputs.size() == 2);
  for (const auto& output : task.outputs) {
    if (output.node_id == "5")
      assert(output.kind == comfy::OutputKind::kAudio &&
             output.prefix == "film/ton/shot07");
    else
      assert(output.kind == comfy::OutputKind::kVideo &&
             output.prefix == "film/ton/shot07_mux");
  }
  assert(plan.warnings.empty());
  const json::Value body = comfy::FoleyRequestBody(foley);
  assert(body.member_str("model") == "mmaudio-44k");
  assert(body.member_str("prompt") == foley.prompt);
  assert(body.member_str("negative_prompt") == "music");
  assert(body.find("duration")->as_double() == 8);
  assert(body.find("steps")->as_double() == 25);
  assert(body.find("cfg_strength")->as_double() == 4.5);
  assert(body.find("seed")->as_double() == 42);
  assert(body.member_str("frame_mode") == "consecutive");
  assert(body.find("force_rate")->as_double() == 25);
  assert(body.find("video") == nullptr);  // the executor adds the file
  assert(body.member_str("response_format") == "wav");

  // Text to audio: no images, SaveAudio only.
  std::string text = Replace(kMmaudio, ", \"images\": [\"3\", 0]", "");
  text = Replace(text,
                 ",\n \"6\": {\"class_type\": \"VHS_VideoCombine\", \"inputs\": "
                 "{\"images\": [\"3\", 0], \"audio\": [\"4\", 0], "
                 "\"frame_rate\": 25, \"loop_count\": 0, \"filename_prefix\": "
                 "\"film/ton/shot07_mux\", \"format\": \"video/h264-mp4\", "
                 "\"pingpong\": false, \"save_output\": true}}",
                 "");
  const comfy::Plan text_plan = PlanOf(text.c_str());
  const auto& text_foley = std::get<comfy::FoleyTask>(text_plan.tasks[0].spec);
  assert(!text_foley.video && !text_foley.image);
  assert(comfy::FoleyRequestBody(text_foley).find("frame_mode") == nullptr);
  // A video output without frames to put the sound on fails.
  assert(ErrorOf(Replace(kMmaudio, ", \"images\": [\"3\", 0]", ""))
             .message.find("needs the sampler's images") != std::string::npos);
  // 16k models and the 16k feature mode are not served.
  assert(ErrorOf(Replace(kMmaudio, "mmaudio_large_44k_v2_fp16",
                         "mmaudio_small_16k_fp16"))
             .node_errors.find("1") != nullptr);
  assert(ErrorOf(Replace(kMmaudio, "\"mode\": \"44k\"", "\"mode\": \"16k\""))
             .node_errors.find("2") != nullptr);
  // Another 44k checkpoint runs on mmaudio-44k with a warning.
  assert(PlanOf(Replace(kMmaudio, "mmaudio_large_44k_v2_fp16",
                        "mmaudio_medium_44k_fp16")
                    .c_str())
             .warnings.size() == 1);
  // A SaveImage cannot take the sound.
  assert(ErrorOf(Replace(kMmaudio, "\"SaveAudio\", \"inputs\": {\"audio\"",
                         "\"SaveImage\", \"inputs\": {\"images\""))
             .message.find("cannot store") != std::string::npos);
  // One LoadImage frame: MMAudio's image mode (with a warning).
  const std::string still =
      Replace(Replace(kMmaudio,
                      "\"VHS_LoadVideo\", \"inputs\": {\"video\": "
                      "\"shot07.mp4\", \"force_rate\": 25, \"custom_width\": "
                      "0, \"custom_height\": 0, \"frame_load_cap\": 0, "
                      "\"skip_first_frames\": 0, \"select_every_nth\": 1}",
                      "\"LoadImage\", \"inputs\": {\"image\": \"kf.png\"}"),
              ",\n \"6\": {\"class_type\": \"VHS_VideoCombine\", \"inputs\": "
              "{\"images\": [\"3\", 0], \"audio\": [\"4\", 0], "
              "\"frame_rate\": 25, \"loop_count\": 0, \"filename_prefix\": "
              "\"film/ton/shot07_mux\", \"format\": \"video/h264-mp4\", "
              "\"pingpong\": false, \"save_output\": true}}",
              "");
  const comfy::Plan still_plan = PlanOf(still.c_str());
  const auto& still_foley =
      std::get<comfy::FoleyTask>(still_plan.tasks[0].spec);
  assert((still_foley.image == comfy::FileRef{"kf.png", "input"}));
  assert(still_plan.warnings.size() == 1);

  // force_rate 25 keeps ComfyUI-MMAudio's frames-as-25-fps (= the video's
  // timing); without frame options the worker resamples by time at the
  // file's own rate, so duration is not cut to frames / 25 (Regie 19.11.:
  // 81 frames at 16 fps gave 3.25 s instead of 5.06 s).
  assert(foley.consecutive);
  const comfy::Plan plain_plan =
      PlanOf(Replace(kMmaudio, "\"force_rate\": 25", "\"force_rate\": 0")
                 .c_str());
  const auto& plain = std::get<comfy::FoleyTask>(plain_plan.tasks[0].spec);
  assert(!plain.consecutive && plain_plan.warnings.empty());
  const json::Value plain_body = comfy::FoleyRequestBody(plain);
  assert(plain_body.member_str("frame_mode") == "resample");
  assert(plain_body.find("force_rate") == nullptr);
  // Any frame option keeps the ComfyUI way (only it applies them).
  for (const auto& [from, to] :
       {std::pair{"\"frame_load_cap\": 0", "\"frame_load_cap\": 50"},
        std::pair{"\"skip_first_frames\": 0", "\"skip_first_frames\": 4"},
        std::pair{"\"select_every_nth\": 1", "\"select_every_nth\": 2"}}) {
    const comfy::Plan cut_plan = PlanOf(
        Replace(Replace(kMmaudio, "\"force_rate\": 25", "\"force_rate\": 0"),
                from, to)
            .c_str());
    const auto& cut = std::get<comfy::FoleyTask>(cut_plan.tasks[0].spec);
    assert(cut.consecutive);
    assert(comfy::FoleyRequestBody(cut).member_str("frame_mode") ==
           "consecutive");
  }
  // force_rate other than 25: ComfyUI-faithful, with a warning.
  const comfy::Plan rate_plan =
      PlanOf(Replace(kMmaudio, "\"force_rate\": 25", "\"force_rate\": 16")
                 .c_str());
  assert(std::get<comfy::FoleyTask>(rate_plan.tasks[0].spec).consecutive);
  assert(rate_plan.warnings.size() == 1 &&
         rate_plan.warnings[0].find("force_rate 16") != std::string::npos);
  // LoadVideo -> GetVideoComponents: the video object carries its rate.
  const comfy::Plan native_plan = PlanOf(R"({
 "1": {"class_type": "MMAudioModelLoader", "inputs": {"mmaudio_model": "mmaudio_large_44k_v2_fp16.safetensors", "base_precision": "fp16"}},
 "2": {"class_type": "MMAudioFeatureUtilsLoader", "inputs": {"vae_model": "mmaudio_vae_44k_fp16.safetensors", "synchformer_model": "mmaudio_synchformer_fp16.safetensors", "clip_model": "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors", "mode": "44k", "precision": "fp16"}},
 "3": {"class_type": "LoadVideo", "inputs": {"file": "shot16fps.mp4"}},
 "7": {"class_type": "GetVideoComponents", "inputs": {"video": ["3", 0]}},
 "4": {"class_type": "MMAudioSampler", "inputs": {"mmaudio_model": ["1", 0], "feature_utils": ["2", 0], "duration": 5.06, "steps": 25, "cfg": 4.5, "seed": 1, "prompt": "rain", "negative_prompt": "", "mask_away_clip": false, "force_offload": true, "images": ["7", 0]}},
 "5": {"class_type": "SaveAudio", "inputs": {"audio": ["4", 0], "filename_prefix": "ton/s16"}}
})");
  const auto& native = std::get<comfy::FoleyTask>(native_plan.tasks[0].spec);
  assert((native.video == comfy::FileRef{"shot16fps.mp4", "input"}));
  assert(!native.consecutive && native_plan.warnings.empty());
  assert(comfy::FoleyRequestBody(native).member_str("frame_mode") ==
         "resample");

  // A sound clearly shorter than duration warns (for /history); a cut of a
  // few 25 fps frames, text mode and an unreadable WAV do not.
  const auto wav_of = [](double seconds) {
    comfy::Wav wav;
    wav.sample_rate = 44100;
    wav.channels = 1;
    wav.pcm.assign(static_cast<std::size_t>(seconds * 44100) * 2, '\0');
    return comfy::BuildWav(wav);
  };
  const auto short_native =
      comfy::FoleyShortWarning(native, "4", wav_of(3.25));
  assert(short_native && short_native->find("node 4: the sound is 3.25 s, "
                                            "shorter than duration 5.06 s") ==
                             0);
  assert(short_native->find("the video is shorter") != std::string::npos);
  assert(!comfy::FoleyShortWarning(native, "4", wav_of(5.04)));
  assert(!comfy::FoleyShortWarning(native, "4", wav_of(5.2)));
  assert(!comfy::FoleyShortWarning(native, "4", "not a wav"));
  assert(!comfy::FoleyShortWarning(text_foley, "4", wav_of(1.0)));
  const auto short_vhs = comfy::FoleyShortWarning(foley, "4", wav_of(4.0));
  assert(short_vhs && short_vhs->find("25 fps") != std::string::npos);
}

int main() {
  TestQwenTextToImage();
  TestQwenReferencesAndCanvas();
  TestVoiceDesign();
  TestVoiceClone();
  TestH3();
  TestH3Guides();
  TestH3CustomSampler();
  TestH3AnchorClip();
  TestWan();
  TestUnsupportedNodes();
  TestImageToImage();
  TestMaskedEdit();
  TestRegionalMasks();
  TestRegionalAreas();
  TestRegionalCombineAndErrors();
  TestMaskComposite();
  TestQwenLoras();
  TestQwenTurbo();
  TestQwenGuidanceSteps();
  TestQwenLoaderName();
  TestImageModelChoice();
  TestRife();
  TestRifeOnly();
  TestRifeSourceFps();
  TestDialogue();
  TestAceStep();
  TestUni3C();
  TestWanFunCamera();
  TestVace();
  TestVaceKeyframes();
  TestWan22I2V();
  TestI2VAntiTalk();
  TestFirstLastFrame();
  TestAnimate();
  TestAnimateNative();
  TestAnimateChain();
  TestI2VChain();
  TestPoseAnimationScaled();
  TestInfiniteTalk();
  TestInfiniteTalkVideoToVideo();
  TestVideoCuts();
  TestDelivery();
  TestWavJoin();
  TestHelpers();
  TestPreviewFrames();
  TestFoley();
  std::cout << "router_comfy_graph_test: all passed\n";
  return 0;
}
