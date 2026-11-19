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
- `tasks`: any of `person`, `head`, `face`, `hand`, `hand_fingers`, `person_count`, `ocr`,
  `wd14_tags`, `pose`, `pose_wholebody`; the four detectors when left out (the others only when
  asked for). `hand_fingers`: see [Hand check](#hand-check); `person_count`: see
  [Person check](#person-check).
  `pose_options`: `min_score` (0.35) and `max_persons` (4) for the pose person search.
- `thresholds`: optional `conf` (minimum score) and `iou` (overlap above which the
  weaker box is dropped) per detector; for `ocr` `heat` and `box`. Each in (0, 1].
- `ocr`: one object or a list of up to 4 passes: `heat_threshold` (0.3), `box_threshold`
  (0.7), `unclip_ratio` (2.0), `scale` (1 to 4, the image enlarged with Pillow's LANCZOS,
  boxes divided back as decimals) and `rec`, the text recognizer: `"ch_v4"` (default,
  imgutils' PP-OCRv4: Chinese and Latin without umlauts) or `"latin_v5"` (PP-OCRv5 latin:
  German `ä ö ü Ä Ö Ü ß`, French, Polish and 29 more Latin-script languages). The list
  above is the pipeline's text search; `{"rec": "latin_v5"}` reads German text.
- `brighten_dark`: optional. `true` runs the detectors on a brightened copy (each channel
  `round(255 * (v / 255) ** 0.5)`) when the mean brightness (Pillow `L`) is below 70; `{"below": 50, "gamma": 0.7}` changes both
  values. The boxes still refer to your image. OCR and tags always read the original.
- `wd14_tags`: optional settings for the tags:
  `{"general_threshold": 0.35, "character_threshold": 0.85, "regions": "person",
  "margin": 0.08, "images": ["data:image/png;base64,..."]}`. `regions` is `"person"`
  (the person boxes of the same request, the `person` task then runs too) or a list
  of up to 64 `[x0, y0, x1, y1]` boxes, each grown by `margin` × its longer side. Without
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
  lists with one entry per pass. `models.ocr` names the recognizer (`rec`, `rec_repo`,
  `rec_model`).
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

## Hand check

`hand_fingers` checks every hand for the film direction: is this hand possible? It is an
anatomy check, not a finger count. A thumb hidden behind the other fingers, fingers behind an
object, in a fist or outside the frame are no error; only clearly impossible things are
reported. It runs the `hand` detector (also answered), cuts out each hand box with 25 % margin
on every side (at least 224 px on the short side) and asks a vision chat model on this router,
one hand after the other:

```json
{"image": "...", "tasks": ["hand_fingers"],
 "hand_fingers_options": {"model": "qwen3.8-27b", "max_hands": 6, "thinking": true,
                          "timeout_seconds": 300, "wait_seconds": 600, "margin": 0.25,
                          "max_tokens": 4096}}
```

All options are optional (the values above are the defaults; `max_hands` 1 to 16,
`timeout_seconds` 5 to 600, `wait_seconds` 0 to 3600, `margin` 0 to 1, `max_tokens` 32 to
8192, thinking included). The answer has one
entry per hand box, in score order:

```json
"hand_fingers": [
  {"box": [812, 488, 884, 538], "score": 0.81, "crop": [794, 476, 902, 551],
   "plausibel": true, "sicher": false, "fingers_sichtbar": 4, "verdeckt": ["daumen"],
   "befunde": [], "befund": "Hand normal, Daumen verdeckt", "seconds": 6.2,
   "waited_seconds": 1.1},
  {"box": [1210, 400, 1290, 520], "score": 0.62, "crop": [1190, 370, 1310, 550],
   "plausibel": false, "sicher": true, "fingers_sichtbar": 5, "verdeckt": [],
   "befunde": [{"art": "zweite_hand_am_arm", "stelle": "rechts, durchscheinender Arm"}],
   "befund": "durchscheinender zweiter Arm rechts", "seconds": 74.7, "waited_seconds": 0.9},
  {"box": [320, 590, 420, 705], "score": 0.74, "crop": [295, 561, 445, 734],
   "fehler": "chat 507: cannot load 'qwen3.8-27b': ...", "seconds": 0.1}
]
```

- `plausibel` is `false` only when `befunde` lists something impossible. Each finding has an
  `art` (`daumen_falsche_seite`, `daumen_aus_handflaeche`, `finger_verschmolzen`,
  `finger_verknotet`, `zu_viele_finger`, `zu_wenige_finger_unerklaert`,
  `gelenk_falsche_richtung`, `hand_passt_nicht_zum_arm`, `zweite_hand_am_arm`,
  `arm_ohne_hand`, `handflaeche_widerspricht_arm`, `sonstiges`) and a `stelle` (where).
- `fingers_sichtbar` (0 to 10) counts only visible fingers, the thumb included; `verdeckt`
  names the fingers that are hidden (no error). `sicher` is `false` for hidden fingers, a
  fist, overlapping hands, a blurred or very small hand. `befund` is one short German
  sentence.
- `timeout_seconds` counts from the first generated token (thinking included). Before that a
  hand may wait `wait_seconds` for the model (queue, also behind another chat of the same
  model, loading; a `503 memory_busy` while another model holds the GPU is asked again after
  its `Retry-After`). Up to 2.2.10 the clock started already with the empty first stream
  chunk that the model's worker sends after 5 s in its queue, so a busy 27B ended the hands
  with `timeout after 300 s of generation` long before `wait_seconds`. When a hand did not get the model in that
  time, the later hands of the image are not asked (`fehler`: `not checked: ...`).
  `waited_seconds` is that wait, `seconds` the hand's whole time.
- A hand whose question failed (model refused or not loadable, timeout, no valid JSON) gets
  `fehler` instead; the other hands and tasks are answered normally. Hands beyond
  `max_hands` get `fehler` too.
- `models.hand_fingers.model` names the model asked; `hand_fingers_seconds` is the time of
  the questions. The questions run after the CPU part and don't hold the CPU queue.
- The questions are ordinary chats under your own token: you need the `text` area too, they
  count like other chats, and the router loads the model if needed.
- `thinking` is on by default: in the prompt test only the 27B with thinking had no finding
  on the four good hands and found the see-through ghost arm (40 to 170 s per hand). Without
  thinking (5 s per hand) it missed the ghost arm and once reported six fingers on a good
  hand. LFM3-VL-2B (1 s) answers the schema but sees almost nothing (see the prompt test). Treat `sicher: false` and small hands as "look at it", not as a finding.

## Person check

`person_count` checks the persons of a film frame for the film direction: how many are there,
and is each of them possible? A vision chat model on this router looks at the whole picture
once, lists every person it sees and reports:

- `extra_limb`: an arm, hand, leg or head that belongs to no person, a third arm, or a faint,
  see-through body part or ghost figure (the background shows through it),
- `duplicate_person`: the same person (face, hair, clothes) twice in the picture, also as a
  see-through ghost double,
- `count_mismatch`: more or fewer persons than `expected`,
- `sonstiges`: anything else impossible about the persons (two bodies merged into one).

Hidden body parts are no finding: a person behind a desk, another person or an object, cut off
by the frame, hands behind the back, legs under a table. People on posters, screens, in mirrors
or windows are not counted. The task runs the `person` detector too (also answered); its count
is given as `detektor_personen` for comparison, the model does not see it.

```json
{"image": "...", "tasks": ["person_count"],
 "person_count_options": {"expected": 1,
                          "references": [{"name": "Bernd", "image": "/v1/jobs/j1/files/bernd.png"}],
                          "model": "qwen3.8-27b", "thinking": true, "timeout_seconds": 300,
                          "wait_seconds": 600, "max_tokens": 4096}}
```

All options are optional (`model`, `thinking`, the times and `max_tokens` as for
`hand_fingers`). `expected` (0 to 20, also `erwartet`): the persons the scene should show;
without it there is no `count_mismatch`. `references` (also `referenzen`): up to 4 reference
pictures of the film's characters, each `{"name", "image"}` or just the image (named
`Figur 1`, ...); images like `image` (data URL, base64, `https://`, Ablage path). The model
then names each person after them (`wer`) and reports the same character twice as
`duplicate_person`. The picture is sent at most 1536 px on its long side, references at most
768 px. The answer is one object:

```json
"person_count": {
  "personen": 1, "erwartet": 1, "detektor_personen": 2,
  "liste": [{"stelle": "Mitte am Tisch", "wer": "Bernd"}],
  "befunde": [{"art": "extra_limb", "stelle": "rechts hinter Bernd, durchscheinender Arm",
               "sicherheit": "hoch", "quelle": "modell"}],
  "sicher": true, "befund": "durchscheinender Doppelgänger mit zusätzlichem Arm rechts",
  "urteil": "fehler", "seconds": 84.1, "waited_seconds": 1.2}
```

- `personen` is the number of persons in `liste` (`stelle`: where; `wer`: the reference name,
  `unbekannt`, or without references a short description).
- Each finding has `art`, `stelle`, `sicherheit` (`hoch` clearly visible, `mittel` probably,
  `niedrig` possible but unclear) and `quelle`: `modell`, or `gufo` for the `count_mismatch`
  gufo adds itself when `expected` is given, the list has a different number and the model
  did not report one (`sicherheit` `hoch`, `mittel` when the model is unsure). Without
  `expected` a `count_mismatch` of the model is dropped.
- `urteil`: `fehler` when a finding has `sicherheit` `hoch`; `pruefen` when there are only
  `mittel` / `niedrig` findings or `sicher` is `false` (dark, blurred, crowded picture, unsure
  count); else `ok`.
- When the question failed (model refused, timeout, no valid JSON) the object has `fehler`
  instead of the verdict (with `erwartet` and `detektor_personen`); the other tasks are
  answered normally. `models.person_count` names the model; `person_count_seconds` is the time
  of the question. Like `hand_fingers` it is an ordinary chat under your own token, runs after
  the CPU part and waits for the model the same way.

## Models

| Task | Hugging Face model | Network | Defaults | ONNX size | Licence |
|---|---|---|---|---:|---|
| `person` | `deepghs/anime_person_detection` `person_detect_v1.1_m` | YOLOv8m | 0.3 / 0.5 | 103 MB | MIT |
| `head` | `deepghs/anime_head_detection` `head_detect_v2.0_s` | YOLOv8s | 0.4 / 0.7 | 45 MB | MIT |
| `face` | `deepghs/anime_face_detection` `face_detect_v1.4_s` | YOLOv8s | 0.25 / 0.7 | 45 MB | MIT |
| `hand` | `deepghs/anime_hand_detection` `hand_detect_v1.0_s` | YOLOv8s | 0.35 / 0.7 | 45 MB | OpenRAIL |
| `ocr` | `deepghs/paddleocr` `ch_PP-OCRv4_det` + `ch_PP-OCRv4_rec` | PP-OCRv4 mobile | heat 0.3 / box 0.7 | 5 + 11 MB | Apache-2.0 |
| `ocr` `rec: latin_v5` | `monkt/paddleocr-onnx` `languages/latin` (`latin_PP-OCRv5_mobile_rec`, 502 symbols) | PP-OCRv5 latin mobile | as above | 8 MB | Apache-2.0 |
| `wd14_tags` | `SmilingWolf/wd-swinv2-tagger-v3` + `selected_tags.csv` | SwinV2-B, 448 px | 0.35 / 0.85 | 467 MB | Apache-2.0 |
| `pose` (persons) | `PekingU/rtdetr_r50vd_coco_o365` | RT-DETR R50-vd | 0.35, 4 persons | 172 MB | Apache-2.0 |
| `pose` | `usyd-community/vitpose-base-simple` | ViTPose-B | - | 344 MB | Apache-2.0 |
| `pose_wholebody` | `yzd-v/DWPose` `dw-ll_ucoco_384.onnx` | RTMPose-l | - | 134 MB | Apache-2.0 |

The detectors, OCR and tagger are the defaults of dghs-imgutils 0.19, the pose models
follow the film pipeline (transformers 5.18, onnxruntime). The router loads the converted files from `--vision-dir` (Windows default `D:\llm-modelle\gufo\modelle\deepghs\vision\gufo`)
on first use; the converters are in the server reference below.

The default OCR model reads Chinese and Latin script. Its alphabet lacks `ö`, `Ö`, `Ä` and `ß`; those come out as similar letters (`Bahnhofstraße` reads as `BahnhofstraBe`).
`"rec": "latin_v5"` keeps the same detector and reads the lines with PaddleOCR's PP-OCRv5
latin recognizer (the community ONNX export of `monkt/paddleocr-onnx`, converted with
`convert_ocr.py`; repeated symbols are merged as in PaddleOCR). On 160 rendered German
lines (8 fonts, 20 to 48 px, some blurred or light on dark) it read 159 exactly (character
error rate 0.06 %), the default 15 (8.0 %); the one miss is an en dash read as `-`. On
four rendered signs (12 lines) the whole pipeline read 11 lines exactly with `latin_v5`
and none with `ch_v4`.

## Limits and errors

- One request computes at a time, in one queue with the CCIP figure check;
  `--vision-threads` (default 8) sets its CPU threads. Up to `--vision-queue`
  (default 4) more wait; the next one gets `503 busy` with `Retry-After: 5`. Pictures are
  decoded only once the request's turn comes; all of a request's pictures together may
  have at most 64 MB encoded and 64 megapixels decoded (more: `400`).
- Tags take about 0.6 s per picture or cut-out (8 threads); the tagger keeps ~470 MB RAM.
  Every OCR pass, enlarged or not, may be at most 24 megapixels (3 × a 1920 × 1080 frame fits).
- `400 invalid_request`: unknown task, bad option, unreadable image, region outside it, limit.
- `404`: an Ablage path of a job you can't read; `503 vision_unavailable`: a model is missing.
- The request needs the `bild` area and doesn't count against the daily limit.
- With a pool partner, a request goes there when this machine is busy or lacks the models
  (`X-Gufo-Ran-On: partner`); `X-Gufo-Pool: local` keeps it here, `remote` sends it there.

Details, conversion commands, processing steps and tests: [server reference](../internal/server.md#image-recognition-post-v1visiondetect).
