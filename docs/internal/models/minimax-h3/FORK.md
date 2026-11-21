# MiniMax H3 in gufo-studio-windows

Options, request fields and frame details of the fork's `gufo serve video`,
moved here from the [public model page](../../../models/minimax-h3.md#gufo-studio-windows-additions).
[Implementation notes](README.md) · [Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md)

## Options

The fork's `gufo serve video` takes these options (`gufo serve video
--help`, 2.0.4) and request fields (`src/models/minimax_h3/request.cpp`):

| Option | Enables |
| --- | --- |
| `--lora <NAME=PATH[,steps=N][,strength=S]>` | A transformer LoRA for requests, `"gufo": {"lora": NAME}` (and `lora_strength`); repeatable. A LoRA registered with `steps=N` sets the step count unless the request sends `gufo.steps`, and then runs every step on every block: a step-distilled LoRA (the 8-step turbo LoRA) cannot afford the `fast`/`aggressive` step reuse and block skipping. |
| `--vision-projector <GGUF>` | Qwen3-VL vision tower for keyframe prompt blocks. |
| `--rife-model <SAFETENSORS>` | RIFE 4.7 weights; `"gufo": {"interpolate": N}` or `{"interpolate": {"fps": N}}` and `source_video` jobs. See [RIFE](../../../models/rife/README.md). |
| `--upscale-model <[NAME=]SAFETENSORS>` | Real-ESRGAN SRVGGNetCompact weights (converted); `"gufo": {"upscale": {"width", "height", "fit", "model"}}`; repeatable, the first is the default. See [Real-ESRGAN](../../../models/realesrgan/README.md). |

## First and last frames

`first_frame` and `last_frame` (or OpenAI's `input_reference` for the first
frame) go to `POST /v1/videos` as multipart file parts or base64 `data:`
URLs; gufo-studio-windows adds `gufo.keyframes` for frames in between and
`gufo.guide_video`. The `gufo video` CLI of this build has no keyframe flags.
The server never fetches remote URLs. PNG, JPEG and WebP work.

The first frame (or a lone last frame) sets the video's aspect ratio, between
1:4 and 4:1; the preset sets its size. A 1664 × 928 frame gives a 1376 × 768
video with `exact-1344x768` and 672 × 384 with `exact`. A last frame next to a
first one is cropped to fit. With `dev`, keyframes far from square are refused
because the video would be too small. `fast` and `aggressive` follow the frames
less closely than `exact`.

## Keyframes

`first_frame` (or OpenAI's `input_reference`, the same anchor; both together
are `400 invalid_input_reference`) and `last_frame` anchor the first and the
last frame of the video, as JSON image values or multipart files of those
names. `"gufo": {"keyframes": [{"frame": N, "image": ...}]}` and multipart
files `keyframe_<N>` add guide frames at frame N (negative N counts from the
end); each frame may carry one image and must lie inside the video. `exact`
at a canvas other than `512x512`/`256x256` runs as `exact-<W>x<H>`; a canvas
has sides of 256 to 1344 pixels in multiples of 32 and at most 768 × 1344
pixels.

## Production preset

The production preset of the two Strix Halo machines registers the
`turbo` LoRA with `steps=8`, the vision projector and the RIFE weights
(`gufo-router-combo7.ini`, `[external/minimax-h3]`).

## Flow shifts and the audio noise mask

Two request fields of `gufo.` mirror ComfyUI nodes (feature branch
`feat/h3-knoten`):

| Field | ComfyUI | Effect |
| --- | --- | --- |
| `"sigma_shift": {"video": 12, "audio": 3}` | `MiniMaxH3SigmaShift` (`shift_video`, `shift_audio`) | Flow shift of the video and the audio sigma schedule, each 0.01 to 100; a missing one keeps its trained value (12, 3). |
| `"audio_mask": {"audio": <audio>, "mask": [2N numbers]}` | `LTXVConcatAVLatent` with `SetLatentNoiseMask` over `VAEEncodeAudio` | The audio (WAV, FLAC, Ogg or MP3 as a base64 `data:` URL) and two mask rows of N values each, one per stereo track, 0 = keep, 1 = generate. Each row is stretched over the audio's length; where it is below 0.5, the video's audio track is that audio, held clean on every step (as an audio guide) while the picture is generated with it. A clip shorter than the video generates the rest. |

Values between 0 and 1 are rounded at 0.5: gufo has no partial strength yet.
A silent or room-tone clip with the mask `[0, 0]` keeps the whole track
quiet, the lever against invented speech in shots without dialogue.

The ComfyUI adapter (`gufo router --comfy`) maps these nodes, also
`EmptyMiniMaxH3LatentAV` with a `CLIPTextEncode` prompt, and lists their
inputs in `/object_info` as ComfyUI does. The sampler's `cfg`, the negative
conditioning and `denoise` have no effect on H3 (the weights are
CFG-distilled); the adapter warns when they differ from 1. `sampler_name`
and `scheduler` are not used: gufo samples with its own Euler steps on the
shifted linear schedule.

## MiniMax-H3-Image edits

`serve image` with the MiniMax-H3-Image release (provider `minimax-h3-image`)
answers `/v1/images/edits` when `--h3-vision-projector` (or
`GUFO_H3_VISION_PROJECTOR`) names the Qwen3-VL mmproj GGUF of the H3 text
encoder, the same file as the video's `--vision-projector`. The recipe follows
the diffusers `MiniMaxH3ImageBlocks` "image_conditioned" path (checked against
commit `c215605a`):

| Step | Reference | gufo |
| --- | --- | --- |
| Sizes | `bound_short_edge`: VAE copy keeps a short edge in [1024, 2048], vision copy 1024; PIL Lanczos, no crop; Qwen's `smart_resize` on the vision copy | `PlanImageEditReferences` (`image_edit.cpp`) with `ResizeRgb8Lanczos` |
| Text encoder | per reference `Picture i: ` (no angle brackets) + `<\|vision_start\|>`, (h/32)(w/32) `<\|image_pad\|>`, `<\|vision_end\|>`, then the prompt; vision rows tagged image | `BuildImageEditPresentation`; vision tower and layer 50 in the isolated prompt child |
| VAE | top-left crop to 16 px, ImageNet normalization, tiled CNN encoder (same weights layout as the H3 VisualVAE encoder), posterior sample, float16, crop to whole 2x2 patches, `latents_mean/std` | `EncodeKeyframeMoments` on the image inventory (now 16-px sides), `BuildImageEditConditioning` |
| Layout | `[text \| refs \| image]`, reference j on rotary time text + j with its own aspect grid, the image on text + n | `BuildImageEditLayout` (video condition segment) |
| Steps | references held at t = 0.999 (AdaLN time row), only the image rows step | the denoiser's condition rows, now allowed in image mode |

Deviation: gufo's `NormalRng` replaces the torch generator (posterior and
anchor noise per reference under seeds derived from the request seed, the
image noise under the seed itself), so seeds are reproducible in gufo only.

Status 20.11.2026: CPU unit tests (`minimax_h3_image_test`) compare the sizes,
the layout, the presentation and the conditioning with the reference. Memory:
each 1024 × 1024 reference adds 1,024 DiT rows plus 1,031 text rows, a
2048 × 2048 one 4,096 + 1,031.

GPU test 21.11.2026 (2.2.15 test build, 2048 × 2048, seed 42): the release
example with one reference 160 s (prompt 58 s for 1,757 tokens, vision tower
3.8 s, DiT 84 s); three 1024² references 433 s (prompt 287 s for 3,202
tokens, vision 5.6 s, DiT 122 s, peak 40.1 GB). Three references with wide
images (1024², 1500×1100, 1000×557: 4,386 tokens) failed in the prompt child
with HTTP 500.

### Token budget (2.2.16)

The release has no limit on the whole presentation; gufo's encoder
(`EncodePromptLayer50`, `kPromptEncoderMaximumTokens`) takes 4,096. The vision
copies are not shrunk (the release always brings their short edge to 1024, so
shrinking them would change the conditioning). Instead
`CountImageEditTokens` / `CheckImageEditTokenBudget` (`image_edit.cpp`) count
labels, markers, vision rows and the cut prompt from the upload sizes; the
provider's `ValidateRequest` runs them with the release tokenizer before the
request is queued (HTTP 400), `GenerateTextImage` repeats the check before the
prompt child. `kH3ImageEditMaximumImages` = 3 is what the capabilities
report: four references need at least 4,124 tokens. Raising the limit itself
would need the encoder's 4,096 cap lifted (activations and RoPE tables scale
linearly, attention quadratically) and a check that the DiT rows still fit.

### Slow prompt phase with long presentations

Cause: `LaunchCausalGqa` (`prompt_encoder_ops.hip.hpp`) keeps a row's scores
for all 8 query heads of a key/value group in 48 KiB of shared memory, which
holds about 1,390 tokens. Longer prompts fell back to `CausalGqaKernel`, one
block per query head that reads every key and value row once per head (8×
the traffic) with uncoalesced key loads. The cost grows with the square of
the length on top of that: about 10 s of weight streaming plus attention, 44 s
at 1,757 tokens and 271 s at 3,202 (326-token text-to-image prompts take
11.5 s in all).

Fix (2.2.16): the grouped kernel can keep its scores in a global scratch of at
most 256 MiB (`kGroupedScoreScratchBytes`, row chunks of up to 256 rows at
4,096 tokens), allocated with the activations when the prompt is too long for
shared memory. Same arithmetic in the same order, so its results equal the
shared-memory kernel bit for bit (`minimax_h3_prompt_encoder_hip_test`
`CheckLongAttentionKernel`); it removes the 8× head traffic. Video prompts
with long reference blocks profit too.

Still open: the key loads stay uncoalesced and every row reads all earlier
keys, so attention is still the largest part at 3,000+ tokens. A tiled
(flash-style) kernel or hipBLASLt batched GEMMs for QK^T and PV would remove
that. Measure: `GUFO_H3_PROMPT_INPROCESS=1` + the layer telemetry
(`gpu_ms` per layer) for prompts of 300, 1,750 and 3,200 tokens, before and
after; compare `prompt_ms - vision_ms` in the `event=generated` telemetry.
