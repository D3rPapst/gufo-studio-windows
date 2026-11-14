# Depth, open-vocabulary boxes and masks

`gufo serve vision` runs image models on the GPU, natively and in FP32, with the same
pre- and post-processing as Hugging Face transformers 5.18:

| Path | Models | What it answers |
|---|---|---|
| `POST /v1/vision/depth` | Depth Anything V2 Large / Small, Distill-Any-Depth Large | relative depth (larger = nearer) |
| `POST /v1/vision/find` | Grounding DINO base, LLMDet base | boxes for English labels |
| `POST /v1/vision/segment` | SAM ViT-H (+ a find model) | masks for boxes, points or labels |

The answers match the Python ones ([Checks](#checks)); pool and memory guard handle the
worker like every model. The CPU detectors of [`/v1/vision/detect`](vision-detect.md)
are a separate service.

## Licences

| Model (Hugging Face) | Size | Licence |
|---|---:|---|
| `depth-anything/Depth-Anything-V2-Large-hf` | 1.34 GB | **CC-BY-NC-4.0: non-commercial use only** |
| `depth-anything/Depth-Anything-V2-Small-hf` | 99 MB | Apache-2.0 |
| `xingyang1/Distill-Any-Depth-Large-hf` | 1.34 GB | MIT |
| `IDEA-Research/grounding-dino-base` | 0.93 GB | Apache-2.0 |
| `iSEE-Laboratory/llmdet_base` | 0.94 GB | Apache-2.0 |
| `facebook/sam-vit-huge` | 2.56 GB | Apache-2.0 |

Answers name model and licence in `source`; for commercial work use Small or Distill.

## Setup

Point the worker at the snapshot folders (`config.json`, `model.safetensors`; the find
models also `vocab.txt`); nothing is converted. A router preset section:

```ini
[external/vision]
command = D:\llm-modelle\gufo\gufo.exe
args = serve vision
depth-model = large=D:\llm-modelle\gufo\modelle\regie-56\werkzeugtest_hf\depth_anything_v2_large
depth-model = small=D:\llm-modelle\gufo\modelle\depth-anything\Depth-Anything-V2-Small-hf
depth-model = distill-large=D:\llm-modelle\gufo\modelle\xingyang1\Distill-Any-Depth-Large-hf
find-model = grounding-dino=D:\llm-modelle\gufo\modelle\regie-56\werkzeugtest_hf\grounding_dino_base
find-model = llmdet=D:\llm-modelle\gufo\modelle\iSEE-Laboratory\llmdet_base
segment-model = D:\llm-modelle\gufo\modelle\regie-56\werkzeugtest_hf\sam_vit_huge
```

`depth-model` and `find-model` repeat (the first is the default). Each model loads on
first use and stays: about 1.4 GB GPU memory per large depth model, 1 GB per find model,
2.6 GB for SAM.

## Requests

All take `model` (the preset section id) and `image`: a `data:` URL, plain base64 or a
public `https://` URL; PNG, JPEG or WebP; an alpha channel is dropped.

**Depth**: `{"model": "vision", "image": "...", "size": "large", "format": "png16"}`.
`size` (or `depth_model`) names a depth model. `images` (up to 32) instead of `image`
normalises all pictures together, as the film pipeline does for a clip. `format`: `png16`
(default), `png8` or `f32` (raw values, little-endian float32 base64, height x width).
`normalize`: `percentile` (default: the 2nd and 98th percentile of every 97th value of all
maps, as `hf.py`) or `minmax`. The PNG holds `(depth - lo) / (hi - lo)` clipped to 0..1;
`png8` truncates like `astype(uint8)`, `png16` rounds.

```json
{"object": "vision.depth", "model": "vision", "size": "large",
 "normalization": {"method": "percentile", "lo": 31.2, "hi": 401.7},
 "data": [{"width": 1344, "height": 768, "format": "png16", "min": -3.4, "max": 477.9,
           "b64_json": "iVBORw0..."}],
 "source": {"repo": "depth-anything/Depth-Anything-V2-Large-hf", "license": "CC-BY-NC-4.0"},
 "seconds": 1.5}
```

**Find**: `{"image": "...", "labels": ["person", "sword"]}` (or `"text": "person. sword."`),
optional `detector` (a find model name), `box_threshold` (0.3), `text_threshold` (0.25).
Labels are lower-cased and joined to `person. sword.`. The answer lists `detections`
`{box: [x0, y0, x1, y1], score, label}` in image pixels, best first; `label` is the
decoded prompt text the box matched.

**Segment**, three ways:
- `labels` (or `text`, optional `detector`): `hf.py segment()`. Find, drop boxes larger
  than `max_area` (0.6) of the picture, keep at most `top` (3) per label, run SAM on all
  kept boxes and answer one union `mask` (grey PNG, 255 = region), `found` (boxes per
  label) and `boxes`. No box: `mask` is `null`.
- `boxes`: `[[x0, y0, x1, y1], ...]` in image pixels; one mask per box in `masks`
  `[{b64_json, score}]` (`score` = SAM's predicted IoU).
- `points` `[[x, y], ...]` with `point_labels` (1 = object, 0 = background; default 1):
  one prompt, one mask; `"multimask": true` answers SAM's three masks.

## Speed

Measured 14.11.2026 on the 64 GB machine, keyframes 1344 x 768, model already loaded:

| Task | GPU (gufo) | CPU (transformers, 16 threads) |
|---|---:|---:|
| depth Small | 0.17 s | 0.6 s |
| depth Large, Distill-Any-Depth | 1.4 s | 4.6 s |
| find (Grounding DINO, LLMDet) | 2.1 s | 5.5 s |
| segment, 3 boxes | 3.1 s | 13 s |

## Checks

Against transformers 5.18.0 / torch 2.13 (CPU) on three film keyframes, with
`src/models/vision_dsg/reference/make_*_oracle.py` and `compare_*.py`: depth input
pixels identical, cosine 1.0000000, largest difference below 2e-5 of the depth range
(all three depth models); find: same boxes and labels, boxes within 0.03 px, scores
within 1e-3; segment: mask IoU 1.000, predicted IoU within 1e-6.

transformers 5.18 loads the box heads of both find models wrongly, without a warning
(details in the server reference); gufo uses the heads in the checkpoints, as the oracle
does with `--file-bbox`. Boxes from transformers 5.18 differ from gufo's (up to about
40 px on our keyframes).

## Limits and errors

- One request computes at a time. `400`: bad fields, unreadable image, unknown `size`
  or `detector`.
- `501 task_not_available`: the worker was started without that kind of model.
- `500 depth_failed | find_failed | segment_failed`: the GPU run failed (see message).
- Find resizes to 800 pixels (short side); its tokenizer lower-cases ASCII only.

Details and tests: [server reference](../internal/server.md#vision-worker-gufo-serve-vision).
