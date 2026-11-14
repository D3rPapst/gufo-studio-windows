# MiniMax H3 in gufo-studio-windows

Options and request fields the fork adds to `gufo serve video`, moved here
from the [public model page](../../../models/minimax-h3.md#gufo-studio-windows-additions).
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
