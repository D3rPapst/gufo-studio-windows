# Image recognition (imgutils detectors, OCR, tags and poses)

`POST /v1/vision/detect` finds people, heads, faces and hands in an image with the
anime detectors of dghs-imgutils, reads text in it with imgutils' OCR, tags it
with its WD14 tagger and finds body poses with RT-DETR, ViTPose and DWPose, on the
router's CPU with the same models, thresholds and rounding as the Python originals.

Use it instead of `detect_person`, `detect_heads`, `detect_faces`, `detect_hands`, `ocr`,
`get_wd14_tags` or the pipeline's pose code in Python: the answers match, and the router queues
the requests instead of letting several processes compete for the cores.

## Request

```json
{
  "image": "data:image/png;base64,iVBORw0KGgo...",
  "tasks": ["person", "head", "face", "hand", "ocr", "wd14_tags"],
  "thresholds": {"person": {"conf": 0.3, "iou": 0.5}},
  "ocr": [{"box_threshold": 0.5}, {"box_threshold": 0.5, "scale": 3}],
  "wd14_tags": {"regions": "person"}
}
```

- `image`: a `data:` URL, plain base64, a public `https://` URL or an Ablage path
  `/v1/jobs/{id}/files/{name}` of one of your own jobs. PNG, JPEG or WebP, at most
  20 MB. An alpha channel is dropped.
- `tasks`: any of `person`, `head`, `face`, `hand`, `ocr`, `wd14_tags`, `pose`,
  `pose_wholebody`; the four detectors when left out (the others only when asked for).
  `pose_options`: `min_score` (0.35) and `max_persons` (4) for the pose person search.
- `thresholds`: optional `conf` (minimum score) and `iou` (overlap above which the
  weaker box is dropped) per detector; for `ocr` `heat` and `box`. Each in (0, 1].
- `ocr`: one object or a list of up to 4 passes: `heat_threshold` (0.3), `box_threshold`
  (0.7), `unclip_ratio` (2.0) and `scale` (1 to 4, the image enlarged with Pillow's LANCZOS,
  boxes divided back as decimals). The list above is the pipeline's text search.
- `brighten_dark`: optional. `true` runs the detectors on a brightened copy (each channel
  `round(255 * (v / 255) ** 0.5)`) when the mean brightness (Pillow `L`) is below 70; `{"below": 50, "gamma": 0.7}` changes both
  values. The boxes still refer to your image. OCR and tags always read the original.
- `wd14_tags`: optional settings for the tags:
  `{"general_threshold": 0.35, "character_threshold": 0.85, "regions": "person",
  "margin": 0.08, "images": ["data:image/png;base64,..."]}`. `regions` is `"person"`
  (the person boxes of the same request, the `person` task then runs too) or a list
  of `[x0, y0, x1, y1]` boxes, each grown by `margin` × its longer side. Without
  `regions` the whole image is tagged. `images` adds up to 16 pictures, tagged whole.

## Answer

```json
{
  "object": "vision.detection", "width": 1344, "height": 768,
  "mean_luma": 44.083, "brightened": true,
  "detections": {
    "person": [{"box": [412, 96, 705, 768], "label": "person", "score": 0.912345}],
    "ocr": [{"box": [399, 253, 865, 298], "points": [[399, 253], [865, 255], [865, 298], [399, 296]],
             "text": "Dr. med.  Anna  Muller", "score": 0.844714, "det_score": 0.868567, "text_score": 0.972541}],
    "wd14_tags": [{"source": "person", "index": 0, "box": [389, 72, 728, 768], "width": 339, "height": 696,
                   "rating": {"general": 0.91, "sensitive": 0.08, "questionable": 0.001, "explicit": 0.0002},
                   "general": {"1boy": 0.97, "necktie": 0.88}, "character": {}}]
  },
  "models": {"person": {"repo": "deepghs/anime_person_detection", "model": "person_detect_v1.1_m", "conf": 0.3, "iou": 0.5}},
  "seconds": 1.2, "queue_seconds": 0.0
}
```

- `box` is `[x0, y0, x1, y1]` in whole pixels of your image, lists are sorted by score,
  `score` is imgutils' unrounded value; `seconds`: compute time, `queue_seconds`: CPU wait.
- `ocr`: the box, its four corners and the text; `score` is the box score times the
  text score. When you send `ocr` as a list, `detections.ocr` and `models.ocr` are
  lists with one entry per pass.
- `wd14_tags`: one entry per tagged picture. `source` is `image`, `region` or `person`
  (`index` into your regions or the person list, `box` the cut-out) or `images`.
  Ratings are always listed, the other tags only above their threshold.

## Poses

`pose` and `pose_wholebody` find the persons with RT-DETR (label person, score above
`min_score`, the `max_persons` best) and add `"poses"` in score order, each
`{"box": [x0, y0, x1, y1] (decimals), "score", "keypoints", "wholebody"}`. `keypoints`
(`pose`): the 17 COCO points of ViTPose; `wholebody` (`pose_wholebody`): the 133
COCO-WholeBody points of DWPose (0-16 body, 17-22 feet, 23-90 face (68, mouth 48-67
counted from 23), 91-111 left and 112-132 right hand); each `[x, y, score]` in image
pixels of the original image, also with `brighten_dark`. `"pose_boxes": [[x0, y0, x1, y1], ...]`
(up to 16) skips RT-DETR: the poses are made in exactly these boxes, in your order, without
`score` (the film pipeline keeps the first frame's box for a whole shot).

## Models

| Task | Hugging Face model | Network | Defaults | ONNX size | Licence |
|---|---|---|---|---:|---|
| `person` | `deepghs/anime_person_detection` `person_detect_v1.1_m` | YOLOv8m | 0.3 / 0.5 | 103 MB | MIT |
| `head` | `deepghs/anime_head_detection` `head_detect_v2.0_s` | YOLOv8s | 0.4 / 0.7 | 45 MB | MIT |
| `face` | `deepghs/anime_face_detection` `face_detect_v1.4_s` | YOLOv8s | 0.25 / 0.7 | 45 MB | MIT |
| `hand` | `deepghs/anime_hand_detection` `hand_detect_v1.0_s` | YOLOv8s | 0.35 / 0.7 | 45 MB | OpenRAIL |
| `ocr` | `deepghs/paddleocr` `ch_PP-OCRv4_det` + `ch_PP-OCRv4_rec` | PP-OCRv4 mobile | heat 0.3 / box 0.7 | 5 + 11 MB | Apache-2.0 |
| `wd14_tags` | `SmilingWolf/wd-swinv2-tagger-v3` + `selected_tags.csv` | SwinV2-B, 448 px | 0.35 / 0.85 | 467 MB | Apache-2.0 |
| `pose` (persons) | `PekingU/rtdetr_r50vd_coco_o365` | RT-DETR R50-vd | 0.35, 4 persons | 172 MB | Apache-2.0 |
| `pose` | `usyd-community/vitpose-base-simple` | ViTPose-B | - | 344 MB | Apache-2.0 |
| `pose_wholebody` | `yzd-v/DWPose` `dw-ll_ucoco_384.onnx` | RTMPose-l | - | 134 MB | Apache-2.0 |

The detectors, OCR and tagger are the defaults of dghs-imgutils 0.19, the pose models
follow the film pipeline (transformers 5.18, onnxruntime). The router loads the converted files from `--vision-dir` (Windows default `D:\llm-modelle\gufo\modelle\deepghs\vision\gufo`)
on first use; the converters are in the server reference below.

The OCR model reads Chinese and Latin script. Its alphabet lacks `ö`, `Ö`, `Ä` and `ß`; those come out as similar letters (`Bahnhofstraße` reads as `BahnhofstraBe`).

## Limits and errors

- One request computes at a time, in one queue with the CCIP figure check;
  `--vision-threads` (default 8) sets its CPU threads. Up to `--vision-queue`
  (default 4) more wait; the next one gets `503 busy` with `Retry-After: 5`. Pictures are
  decoded only once the request's turn comes; all of a request's pictures together may
  have at most 64 MB encoded and 64 megapixels decoded (more: `400`).
- Tags take about 0.6 s per picture or cut-out (8 threads); the tagger keeps ~470 MB RAM.
  An enlarged OCR pass may be at most 24 megapixels (3 × a 1920 × 1080 frame fits).
- `400 invalid_request`: unknown task, bad option, unreadable image, region outside it, limit.
- `404`: an Ablage path of a job you can't read; `503 vision_unavailable`: a model is missing.
- The request needs the `bild` area and doesn't count against the daily limit.
- With a pool partner, a request goes there when this machine is busy or lacks the models
  (`X-Gufo-Ran-On: partner`); `X-Gufo-Pool: local` keeps it here, `remote` sends it there.

Details, conversion commands, processing steps and tests: [server reference](../internal/server.md#image-recognition-post-v1visiondetect).
