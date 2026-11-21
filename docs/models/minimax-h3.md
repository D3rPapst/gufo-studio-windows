# MiniMax H3

MiniMax H3 generates short videos with sound from a text prompt. Gufo runs its
text-to-video-and-audio path, FL2VA, from BF16 weights and writes an MP4.

It can also start from a first frame, end on a last frame, or both; see
[gufo-studio-windows additions](#gufo-studio-windows-additions) for the rest.
Since 2.2.12 Ref2VA references (images, videos, audio; `gufo.references`) and the Fun-ControlNet-Union (`gufo.control`) are built in (unit-tested; see [gufo-studio-windows additions](#gufo-studio-windows-additions)). 2K regeneration isn't supported.

## Files

> **Warning:** You obtain the checkpoint and accept the MiniMax H3 Community
> License Agreement yourself. Gufo ships no H3 weights or tokenizer, and never
> downloads the model. See the
> [third-party notices](../../THIRD_PARTY_NOTICES.md).

| Item | Pinned identity |
| --- | --- |
| MiniMaxAI/MiniMax-H3, FL2VA | `42ed227ee7df40d41602854ae760620d6eb651fe` |
| Source manifest SHA-256 | `00a83367b87017f1f3d5547a963b20567a46d7aea44f0aac495c816147720d89` |
| License, August 2, 2026, SHA-256 | `59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44` |

After you have access, download the FL2VA files and verify them:

```sh
hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include LICENSE README.md model_index.json "FL2VA/**" \
  --local-dir /var/llms/huggingface/MiniMax-H3
python3 tools/h3/gufo-h3-manifest.py \
  --model-root /var/llms/huggingface/MiniMax-H3 \
  --verify src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json
```

The check runs from a Gufo source checkout and executes no code from the download.
Gufo also checks the files when it loads them and refuses missing, partial or modified files.

## Presets

| Preset | Output | Speed and quality |
| --- | --- | --- |
| `exact` | 512 × 512 | The released schedule, 49 steps on all 50 DiT blocks. |
| `exact-1344x768` | 1344 × 768 | The released schedule at full resolution, 49 steps. |
| `fast` | 512 × 512 | 19 steps at a smaller internal size, skipping some work; approximate. |
| `aggressive` | 512 × 512 | Like `fast`, skipping more; approximate. |
| `dev` | 256 × 256 frames | Four steps and three frames, no audio, for checking a setup. |

"Steps" is the preset's denoiser evaluation count (`ResolveGenerationPreset`
in `src/models/minimax_h3/generation.cpp`); `gufo.steps` (2 to 1000) replaces it.

## Run

Check a setup with the `dev` preset, which writes three frames:

```sh
gufo video --model /var/llms/huggingface/MiniMax-H3 \
  --preset dev --seed 42 --frames-dir /tmp/h3-preview --profile \
  "A red fox walking through snow"
```

For a full video, use `--preset exact --output /tmp/h3.mp4`.

Start a video server:

```sh
gufo serve --host 127.0.0.1 --port 8080 video \
  --model /var/llms/huggingface/MiniMax-H3 \
  --root /var/llms/huggingface/gufo-h3-jobs --ttl 3600
```

`POST /v1/videos` takes JSON or multipart form data:

```json
{
  "model": "minimax-h3",
  "prompt": "A red fox walking through snow",
  "size": "512x512",
  "seconds": 5,
  "gufo": {"seed": 42}
}
```

## First and last frames

Pass `first_frame` and `last_frame` (or OpenAI's `input_reference` for the
first frame) to `POST /v1/videos` as multipart files or base64 `data:` URLs in
PNG, JPEG or WebP; the server never fetches remote URLs. The first frame sets
the aspect ratio, between 1:4 and 4:1. Sizes, cropping, `gufo.keyframes` and
`gufo.guide_video`: [fork notes](../internal/models/minimax-h3/FORK.md#first-and-last-frames).

## Limits

- `size` is `512x512` or `1344x768`, at 24 frames per second. `seconds` is 1,
  or 5 to 14.375; Gufo rounds up to the frame counts the model supports (5
  seconds gives 124 frames).
- Finished videos are deleted after `--ttl` seconds, 3600 by default.
- The prompt can be up to 4,096 tokens.
- Over HTTP, `dev` needs `size: "256x256"` and `gufo.output_format: "ppm"`,
  and returns one frame.
- The server runs one video at a time and queues up to `--video-queue`
  more (default 8, at most 64; before 2.2.8 one); further requests get HTTP
  429 `video_queue_full` with `Retry-After: 1`. A waiting job holds only its
  request in host memory, no GPU memory.

Video jobs run in the background. The [server guide](../server.md#videos)
explains status, progress, downloads, cancellation and expiry.
`DELETE /v1/videos/{id}` cancels a job at any stage and deletes its files.
Saved job status keeps a hash of the prompt, never the prompt itself.

## gufo-studio-windows additions

The fork's `gufo serve video` adds `--lora` (such as the 8-step turbo LoRA),
`--vision-projector` for keyframe prompts, [RIFE](rife/README.md)
interpolation, [Real-ESRGAN](realesrgan/README.md) upscaling and
`gufo.keyframes` guide frames. Every option, request field and the production
preset are in the [fork notes](../internal/models/minimax-h3/FORK.md).

## MiniMax-H3-Image

The fork's `gufo serve image` also runs
[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
(provider `minimax-h3-image`, own community license): one checkpoint for text
to image and image edits on the H3 DiT without audio, six distilled steps, no
CFG and no negative prompt. It reuses the FL2VA text encoder above, so start it
with the image release as `--model` and the H3 root for the text encoder:

```sh
gufo serve --port 8080 image --model /models/MiniMax-H3-Image \
  --h3-root /models/MiniMax-H3 \
  --h3-vision-projector /models/MiniMax-H3-vision/mmproj-BF16.gguf
```

- `POST /v1/images/generations`: `prompt` (used verbatim, at most 4,096
  tokens; the model expects a long English description), `size`, `seed`.
  `steps` is 0 or 6. Sizes up to 8192 px per side and 20 MP; a canvas below
  `gufo.min_generation_area` (default 2048 × 2048) is generated larger and
  resized back with Lanczos.
- `POST /v1/images/edits` (multipart, needs `--h3-vision-projector` or
  `GUFO_H3_VISION_PROJECTOR`): 1 to 16 `image` parts (3 always fit, more
  only with enough free device memory; see the reference limit below), in
  upload order
  `<Picture 1>`, `<Picture 2>`, … in the prompt, within the host's 20 MiB /
  32 MP upload limit. No `mask` or `init_image`. Form fields `gufo.<name>`
  (for example `gufo.min_generation_area`) work as in the JSON body. Each reference is used twice,
  as in the release: as a `Picture i: ` label with a Qwen3-VL vision block in
  front of the prompt (short edge resized to 1024), and as a VAE latent in the
  packed sequence (short edge kept between 1024 and 2048, otherwise resized to
  the nearer bound; cropped at the top left to whole 32-pixel cells) that stays
  at t = 0.999 through all steps. References are never resized to the output
  size; an alpha channel is dropped.
- Noise comes from gufo's own generator, so a seed gives the same image in
  gufo every time but not the image of diffusers or SGLang for that seed.

**Reference limit.** The release accepts up to 16 references and cuts only the
prompt and each label to 4,096 tokens. In gufo everything together goes
through the text encoder: per reference its label (5 tokens, 6 from
`Picture 10`), two vision markers and (short side scaled to 1024 px / 32) ×
(long side / 32) vision tokens, then the prompt. A square image costs 1,031
tokens whatever its size (small images are scaled up), a 16:9 one 1,831.

4,096 tokens always fit: three squares, leaving 1,003 tokens for the prompt.
More are accepted only when the device memory that is free when the request
arrives holds the longer prompt phase and the image transformer with the
extra rows (about 37 GiB plus about 180 KiB per row of text, references and
image), plus a reserve of 10 GiB. On a 64 GB Strix Halo that is the case only
while no other model is loaded: with 53.6 GiB free, fifteen 1024² or six
2048² references fit; with 49 GiB or less it stays at three. Two environment
variables of the server tune it:

| Variable | Effect |
| --- | --- |
| `GUFO_H3_IMAGE_TOKEN_RESERVE_MIB` | Memory kept free (default 10240). |
| `GUFO_H3_IMAGE_MAX_TOKENS` | Hard cap between 4096 and 32768 (default 32768); 4096 restores the fixed limit of three references. |

`max_images` in the capabilities reports what the cap allows (16 by default),
because the server reads it once at startup; the memory is checked per
request. A request over its limit is rejected before any GPU work with HTTP
400 `invalid_request`, the numbers and the current limit, for example 1024²
+ 1500×1100 + 1000×557 with a 109-token prompt needs 4,386 tokens. Long
presentations are slow (attention grows with the square of the length);
timings for more than three references are not measured yet.

**Time** on a Strix Halo (gfx1151, 128 GB), 2048 × 2048 output, measured
21.11.2026 with 2.2.15 builds: one 1024² reference 160 s (prompt phase 58 s),
three 1024² references 433 s (prompt phase 287 s). The prompt phase of long
presentations was dominated by the text encoder's attention fallback; 2.2.16
keeps them on the grouped kernel (see the
[edit notes](../internal/models/minimax-h3/FORK.md#minimax-h3-image-edits)),
new timings follow.

Every reference adds rows to the DiT sequence (a 1024 × 1024 reference about
2,100 including its vision block, a 2048 × 2048 one about 5,100), which costs
attention time and a little memory; see the
[edit notes](../internal/models/minimax-h3/FORK.md#minimax-h3-image-edits) for the recipe and its status.

## Performance

As of November 2026, complete-video timings are still being measured. See the
[benchmarks and memory notes](../internal/models/minimax-h3/BENCHMARKS.md) and
the [quality report](../internal/models/minimax-h3/QUALITY.md).
