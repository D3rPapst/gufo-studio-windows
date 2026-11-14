# Image recognition (imgutils detectors and OCR)

`POST /v1/vision/detect` finds people, heads, faces and hands in an image with the
anime detectors of dghs-imgutils and reads text in it with imgutils' OCR, on the
router's CPU, with the same models, thresholds and box rounding as `imgutils.detect`
and `imgutils.ocr`.

Use it when a pipeline today calls `detect_person`, `detect_heads`, `detect_faces`,
`detect_hands` or `ocr` in Python: the answers match, so earlier measurements stay comparable,
and the router queues the requests instead of letting several processes compete for
the cores.

## Request

```json
{
  "image": "data:image/png;base64,iVBORw0KGgo...",
  "tasks": ["person", "head", "face", "hand", "ocr"],
  "thresholds": {"person": {"conf": 0.3, "iou": 0.5}, "ocr": {"heat": 0.3, "box": 0.7}}
}
```

- `image`: a `data:` URL, plain base64, a public `https://` URL or an Ablage path
  `/v1/jobs/{id}/files/{name}` of one of your own jobs. PNG, JPEG or WebP, at most
  20 MB. An alpha channel is dropped.
- `tasks`: any of `person`, `head`, `face`, `hand`, `ocr`; the four detectors when
  left out (`ocr` only when asked for). `wd14_tags` is a reserved name and answers
  `501 task_not_available`.
- `thresholds`: optional `conf` (minimum score) and `iou` (overlap above which the
  weaker box is dropped) per detector; for `ocr` `heat` (text pixel threshold) and
  `box` (minimum box score). Each in (0, 1].
- `brighten_dark`: optional. `true` runs the detectors on a brightened copy
  (each channel `round(255 * (v / 255) ** 0.5)`) when the image's mean brightness
  (Pillow's `L` channel) is below 70; `{"below": 50, "gamma": 0.7}` changes both
  values. The boxes still refer to your image. Off by default.

## Answer

An answer for two tasks (example values):

```json
{
  "object": "vision.detection",
  "width": 1344,
  "height": 768,
  "mean_luma": 44.083,
  "brightened": true,
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
score, and `score` is the unrounded value imgutils returns. `seconds` is the compute time, `queue_seconds` the wait for the CPU.

An `ocr` entry also has the four corners of the (possibly rotated) text box and the
text; `score` is the box score times the text score, as in `imgutils.ocr.ocr`:

```json
{"box": [399, 253, 865, 298], "points": [[399, 253], [865, 255], [865, 298], [399, 296]],
 "text": "Dr. med.  Anna  Muller", "score": 0.844714, "det_score": 0.868567, "text_score": 0.972541}
```

## Models

| Task | Hugging Face model | Network | Default conf / iou | ONNX size | Licence |
|---|---|---|---|---:|---|
| `person` | `deepghs/anime_person_detection` `person_detect_v1.1_m` | YOLOv8m | 0.3 / 0.5 | 103 MB | MIT |
| `head` | `deepghs/anime_head_detection` `head_detect_v2.0_s` | YOLOv8s | 0.4 / 0.7 | 45 MB | MIT |
| `face` | `deepghs/anime_face_detection` `face_detect_v1.4_s` | YOLOv8s | 0.25 / 0.7 | 45 MB | MIT |
| `hand` | `deepghs/anime_hand_detection` `hand_detect_v1.0_s` | YOLOv8s | 0.35 / 0.7 | 45 MB | OpenRAIL |
| `ocr` | `deepghs/paddleocr` `ch_PP-OCRv4_det` + `ch_PP-OCRv4_rec` | PP-OCRv4 mobile (DBNet + SVTR) | heat 0.3 / box 0.7 | 5 + 11 MB | Apache-2.0 |

These are the defaults of dghs-imgutils 0.19. Convert each `model.onnx` once (Python
with `onnx`, `numpy` and `safetensors`):

```powershell
python -I extensions-ours\ccip\reference\convert_weights.py yolo `
  <hf-cache>\person_detect_v1.1_m\model.onnx D:\llm-modelle\gufo\modelle\deepghs\vision\gufo person_detect_v1.1_m
```

OCR uses its own converter; the recognizer also needs its `dict.txt`:

```powershell
python -I extensions-ours\ccip\reference\convert_ocr.py <paddleocr>\det\ch_PP-OCRv4_det\model.onnx `
  D:\llm-modelle\gufo\modelle\deepghs\vision\gufo ch_PP-OCRv4_det
python -I extensions-ours\ccip\reference\convert_ocr.py <paddleocr>\rec\ch_PP-OCRv4_rec\model.onnx `
  D:\llm-modelle\gufo\modelle\deepghs\vision\gufo ch_PP-OCRv4_rec <paddleocr>\rec\ch_PP-OCRv4_rec\dict.txt
```

The router reads `<model>.safetensors` and `<model>.json` from `--vision-dir` (Windows
default `D:\llm-modelle\gufo\modelle\deepghs\vision\gufo`) and loads each model on its first use.

The OCR model reads Chinese and Latin script. Its alphabet has `ä`, `ü` and `Ü` but not
`ö`, `Ö`, `Ä` or `ß`; those come out as similar letters (`Bahnhofstraße` reads as
`BahnhofstraBe`), and words are often separated by two spaces. imgutils has more
languages as separate models; the service uses its default, like the pipeline.

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
