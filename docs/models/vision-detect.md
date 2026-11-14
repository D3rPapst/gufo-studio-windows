# Image recognition (imgutils detectors)

`POST /v1/vision/detect` finds people, heads, faces and hands in an image with the
anime detectors of dghs-imgutils, on the router's CPU, with the same models,
thresholds and box rounding as `imgutils.detect`.

Use it when a pipeline today calls `detect_person`, `detect_heads`, `detect_faces` or
`detect_hands` in Python: the answers match, so earlier measurements stay comparable,
and the router queues the requests instead of letting several processes compete for
the cores.

## Request

```json
{
  "image": "data:image/png;base64,iVBORw0KGgo...",
  "tasks": ["person", "head", "face", "hand"],
  "thresholds": {"person": {"conf": 0.3, "iou": 0.5}}
}
```

- `image`: a `data:` URL, plain base64, a public `https://` URL or an Ablage path
  `/v1/jobs/{id}/files/{name}` of one of your own jobs. PNG, JPEG or WebP, at most
  20 MB. An alpha channel is dropped.
- `tasks`: any of `person`, `head`, `face`, `hand`; all four when left out. `ocr` and
  `wd14_tags` are reserved names and answer `501 task_not_available`.
- `thresholds`: optional `conf` (minimum score) and `iou` (overlap above which the
  weaker box is dropped) per task, each in (0, 1].

## Answer

An answer for two tasks (example values):

```json
{
  "object": "vision.detection",
  "width": 1344,
  "height": 768,
  "detections": {
    "person": [{"box": [412, 96, 705, 768], "label": "person", "score": 0.912345}],
    "face": [{"box": [520, 140, 598, 222], "label": "face", "score": 0.874512}]
  },
  "models": {
    "person": {"repo": "deepghs/anime_person_detection", "model": "person_detect_v1.1_m", "conf": 0.3, "iou": 0.5},
    "face": {"repo": "deepghs/anime_face_detection", "model": "face_detect_v1.4_s", "conf": 0.25, "iou": 0.7}
  },
  "seconds": 1.2,
  "queue_seconds": 0.0
}
```

`box` is `[x0, y0, x1, y1]` in whole pixels of your image, each list is sorted by
score. `seconds` is the compute time, `queue_seconds` the wait for the CPU.

## Models

| Task | Hugging Face model | Network | Default conf / iou | ONNX size | Licence |
|---|---|---|---|---:|---|
| `person` | `deepghs/anime_person_detection` `person_detect_v1.1_m` | YOLOv8m | 0.3 / 0.5 | 103 MB | MIT |
| `head` | `deepghs/anime_head_detection` `head_detect_v2.0_s` | YOLOv8s | 0.4 / 0.7 | 45 MB | MIT |
| `face` | `deepghs/anime_face_detection` `face_detect_v1.4_s` | YOLOv8s | 0.25 / 0.7 | 45 MB | MIT |
| `hand` | `deepghs/anime_hand_detection` `hand_detect_v1.0_s` | YOLOv8s | 0.35 / 0.7 | 45 MB | OpenRAIL |

These are the defaults of dghs-imgutils 0.19. Convert each `model.onnx` once (Python
with `onnx`, `numpy` and `safetensors`):

```powershell
python -I extensions-ours\ccip\reference\convert_weights.py yolo `
  <hf-cache>\person_detect_v1.1_m\model.onnx D:\llm-modelle\gufo\modelle\deepghs\vision\gufo person_detect_v1.1_m
```

The router reads `<model>.safetensors` and `<model>.json` from `--vision-dir` (Windows
default `D:\llm-modelle\gufo\modelle\deepghs\vision\gufo`) and loads each model on its first use.

## Limits and errors

- One request computes at a time; `--vision-threads` (default 8) sets its CPU threads.
  Up to `--vision-queue` (default 4) more wait; the next one gets `503 busy` with
  `Retry-After: 5`.
- `400 invalid_request`: unknown task, bad threshold, unreadable image.
- `404`: an Ablage path of a job you can't read.
- `503 vision_unavailable`: a model file is missing; the message names it.
- The request needs the `bild` area and doesn't count against the daily limit.
- It runs on the router that receives it; it is not passed to the pool partner.

Details, processing steps and tests: [server reference](../internal/server.md#image-recognition-post-v1visiondetect).
