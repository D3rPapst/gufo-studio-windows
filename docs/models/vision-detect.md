# Image recognition (imgutils detectors)

`POST /v1/vision/detect` finds people, heads, faces and hands in an image with the
anime detectors of dghs-imgutils, and body poses with RT-DETR, ViTPose and DWPose, on
the router's CPU with the same models, thresholds and rounding as the Python originals.

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
- `tasks`: any of `person`, `head`, `face`, `hand`, `pose`, `pose_wholebody`; the four
  detectors when left out. `ocr` and `wd14_tags` are reserved names and answer
  `501 task_not_available`.
- `pose_options`: optional `min_score` (default 0.35) and `max_persons` (default 4) for
  the person search of the pose tasks.
- `thresholds`: optional `conf` (minimum score) and `iou` (overlap above which the
  weaker box is dropped) per task, each in (0, 1].
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

## Poses

`pose` and `pose_wholebody` find the persons with RT-DETR (label person, score above
`min_score`, the `max_persons` best) and add a `poses` list in score order:

```json
{"poses": [{"box": [412.3, 96.1, 705.8, 767.2], "score": 0.93,
            "keypoints": [[530.1, 140.7, 0.91]],
            "wholebody": [[531.0, 141.2, 0.88]]}]}
```

`keypoints` (with `pose`) are the 17 COCO points of ViTPose, `wholebody` (with
`pose_wholebody`) the 133 COCO-WholeBody points of DWPose: 0-16 body, 17-22 feet,
23-90 face (68, mouth 48-67 counted from 23), 91-111 left and 112-132 right hand; each
`[x, y, score]` in image pixels (shortened above). Poses always use the original image,
also with `brighten_dark`.

## Models

| Task | Hugging Face model | Network | Default conf / iou | ONNX size | Licence |
|---|---|---|---|---:|---|
| `person` | `deepghs/anime_person_detection` `person_detect_v1.1_m` | YOLOv8m | 0.3 / 0.5 | 103 MB | MIT |
| `head` | `deepghs/anime_head_detection` `head_detect_v2.0_s` | YOLOv8s | 0.4 / 0.7 | 45 MB | MIT |
| `face` | `deepghs/anime_face_detection` `face_detect_v1.4_s` | YOLOv8s | 0.25 / 0.7 | 45 MB | MIT |
| `hand` | `deepghs/anime_hand_detection` `hand_detect_v1.0_s` | YOLOv8s | 0.35 / 0.7 | 45 MB | OpenRAIL |

| `pose` (persons) | `PekingU/rtdetr_r50vd_coco_o365` | RT-DETR R50-vd | 0.35, 4 persons | 172 MB | Apache-2.0 |
| `pose` | `usyd-community/vitpose-base-simple` | ViTPose-B | - | 344 MB | Apache-2.0 |
| `pose_wholebody` | `yzd-v/DWPose` `dw-ll_ucoco_384.onnx` | RTMPose-l | - | 134 MB | Apache-2.0 |

The detectors are the defaults of dghs-imgutils 0.19; the pose models and their
processing follow transformers 5.18 and onnxruntime as the film pipeline runs them.
Convert the pose models with `convert_pose.py rtdetr|vitpose <hf folder> <dir>` and
`convert_dwpose.py <onnx> <dir>`, each detector `model.onnx` once (Python with `onnx`):

```powershell
python -I extensions-ours\ccip\reference\convert_weights.py yolo `
  <hf-cache>\person_detect_v1.1_m\model.onnx D:\llm-modelle\gufo\modelle\deepghs\vision\gufo person_detect_v1.1_m
```

The router reads `<model>.safetensors` and `<model>.json` from `--vision-dir` (Windows
default `D:\llm-modelle\gufo\modelle\deepghs\vision\gufo`) and loads each model on its first use.

## Limits and errors

- One request computes at a time, in one queue with the CCIP figure check;
  `--vision-threads` (default 8) sets its CPU threads. Up to `--vision-queue`
  (default 4) more wait; the next one gets `503 busy` with `Retry-After: 5`.
- `400 invalid_request`: unknown task, bad threshold, unreadable image.
- `404`: an Ablage path of a job you can't read.
- `503 vision_unavailable`: a model file is missing; the message names it.
- The request needs the `bild` area and doesn't count against the daily limit.
- With a pool partner, a request goes there first when this machine is busy or
  lacks the models; its answer carries `X-Gufo-Ran-On: partner`. `X-Gufo-Pool: local`
  keeps it here, `X-Gufo-Pool: remote` sends it there.

Details, processing steps and tests: [server reference](../internal/server.md#image-recognition-post-v1visiondetect).
