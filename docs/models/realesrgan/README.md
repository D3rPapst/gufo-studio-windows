# Real-ESRGAN upscaling

gufo runs Real-ESRGAN's compact video models (`SRVGGNetCompact` from
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), BSD 3-Clause) natively
on HIP to bring finished videos and images to a delivery size such as
1920x1080, the film format.

Supported checkpoints (Real-ESRGAN release v0.2.5.0, published in the
BSD-3-Clause repository without a separate model license):

| Checkpoint | Body convolutions | Use |
|---|---|---|
| `realesr-animevideov3.pth` | 16 | anime / drawn video (default choice) |
| `realesr-general-x4v3.pth` | 32 | general scenes |
| `realesr-general-wdn-x4v3.pth` | 32 | general, stronger denoising |

All are x4 models with 64 features and PReLU activations. The network follows
`realesrgan/archs/srvgg_arch.py`: 3x3 convolutions, PReLU, a final convolution
to `3 * 4 * 4` channels, `PixelShuffle(4)` plus the nearest-neighbour
upsampled input, clamped to [0, 1] (RGB in [0, 1] as `RealESRGANer` feeds it).
The x4 picture is then resized to the requested size with an antialiased
Lanczos-3 filter (`pad`, `stretch`, and `crop` to a size larger than x4);
`fit crop` cuts like Take One: first in the x4 picture, then `INTER_AREA`
(see below).

## Weights

Convert a checkpoint once; the converter needs only the Python standard
library and reads the pickle with the restricted unpickler of the RIFE
converter (no code from the checkpoint runs):

```sh
python -I tools/upscale/convert_realesrgan_pth.py realesr-animevideov3.pth \
  realesr-animevideov3.safetensors
```

### Converted weights

The three converted files are published as
[D3rPapst/realesrgan-srvgg-gufo](https://huggingface.co/D3rPapst/realesrgan-srvgg-gufo)
(BSD-3-Clause, licence text included; converted from the v0.2.5.0 `.pth` files, SHA-256 of
the sources on the card). The metadata of each file records `format=realesrgan-srvgg`,
`num_conv`, `num_feat=64`, `upscale=4` and `source=<pth name>`, so the header check at
start-up works as with a local conversion.

```powershell
hf download D3rPapst/realesrgan-srvgg-gufo --local-dir D:\llm-modelle\gufo\modelle\upscale
```

The files (`realesr-general-x4v3`, `realesr-general-wdn-x4v3`: 4.9 MB,
`realesr-animevideov3`: 2.5 MB) are what the `[external/gufo-upscale]` section under
[Router section](#router-section) loads (`model =` the general file, two `--model` lines
without `NAME=` the other two; the variant comes from the file name).

## CLI

```sh
gufo video-upscale --model realesr-animevideov3.safetensors \
  --input in.mp4 --output out.mp4 --size 1920x1080
```

`--fit pad` (default) keeps the aspect ratio and the whole picture with black
bars (832x480 -> picture 1872x1080 at x 24 in 1920x1080); `--fit crop` keeps
the aspect ratio and fills the frame (centre crop, Take One's geometry, see
below); `--fit stretch` scales each axis on its own. For `pad` and `stretch`
picture sizes and offsets are even, so 4:2:0 chroma stays aligned. The output
keeps the input frame rate and copies the audio
(`--no-audio` drops it); `--crf N` sets the H.264 quality (default 18). An
output ending in `.png` writes the first frame as an image, so
`--input key.png --output key-1920.png` upscales a still.

Frames run through the network in tiles of at most `--tile` input pixels
(default 512 Ki, so 832x480 is one tile). Each tile carries a halo of one
pixel per convolution, so tiling does not change the result.
`gufo-upscale` is the same driver as a standalone executable.

Diagnostics: `--fp32` runs direct FP32 convolutions instead of FP16 WMMA,
`--bench N` times the first frame, `--reference DIR` compares the network
output against a PyTorch dump (`manifest.json` with `{frame, file}` entries
of FP32 `[4h, 4w, 3]` outputs).

## Server

`gufo serve video --upscale-model [NAME=]realesr-animevideov3.safetensors`
(repeatable; the name defaults to the file stem, the first is the default)
accepts

```json
{"gufo": {"interpolate": {"fps": 60},
          "upscale": {"width": 1920, "height": 1080, "fit": "crop",
                      "model": "realesr-animevideov3"}}}
```

on `POST /v1/videos` (H3 and Wan 2.2). `"upscale": "1920x1080"` is short for
the default model with `pad`. Width and height must be even (at most
7680x4320). The upscale runs after any interpolation (RIFE at the generated
size, then every output frame upscaled) and reports progress phase `upscale`.
Without `--upscale-model` the request is rejected with 400.

## Images and videos on the NPU, GPU or CPU (`gufo serve upscale`)

```
gufo serve upscale --device npu --model realesr-animevideov3=body896q_h288_h3.onnx \
                   [--model realesr-animevideov3=body512q_h280_h3.onnx] \
                   [--npu-cache DIR] --port 8095
gufo serve upscale --device gpu --served-model-name gufo-upscale \
                   --model realesr-general-x4v3.safetensors \
                   --model realesr-general-wdn-x4v3.safetensors \
                   --model realesr-animevideov3.safetensors --port 8095
```

Every weight file carries a **variant** read from its file name: `general`
(realesr-general-x4v3, 32 body convolutions), `wdn` (realesr-general-wdn-x4v3,
32, weaker denoising) and `anime` (realesr-animevideov3, 16). Files given
without `NAME=` share the public model id (`--served-model-name`, else the
file stem) and are chosen per request with `"variant"`; a variant name alone
also works as `model` on the worker itself. On the GPU a wrong safetensors
file fails at start-up with a clear message; `GET /v1/models` lists the
variants with their convolution count.

- `--device npu` runs a Real-ESRGAN **body** ONNX (the convolutions before
  the pixel shuffle, int8 QDQ from AMD Quark, fixed input size) on the AMD
  NPU through ONNX Runtime and the VitisAI execution provider of Windows ML.
  gufo loads `onnxruntime.dll` at run time from the Windows ML framework
  package (`--onnxruntime` overrides) and asks the Windows ML catalog for the
  provider (`--npu-ep` overrides); it never ships either. The first start
  compiles the model into `--npu-cache` (default `<model dir>/npu-cache`,
  cache key = file stem); later starts read the cache.
- `--device cpu` runs the same file on the ONNX Runtime CPU provider
  (`--threads N`). The int8 graph is deterministic: on .198 the CPU result
  of Take One's test frames equals Take One's NPU output pixel for pixel.
- `--device gpu` uses the HIP upscaler above with the converted safetensors
  (`--fp32` for the validation path).

The NPU / CPU path reproduces Take One's `npu_upscale_worker.py`: RGB /255,
two row passes (rows 0..T-1 keep the top half, rows h-T..h-1 the bottom half),
DepthToSpace(4) plus the nearest-neighbour input, clip, `uint8(v*255+0.5)`,
then OpenCV `INTER_AREA` (bit-exact port) and a centre crop: 896x512 ->
rows 16..2031 of the x4 picture -> 1920x1080; 832x512 -> 1950x1200 ->
columns 15..1934 -> 1920x1200 (`fit: "crop"`; this is Take One's NPU path,
the film itself uses 1920x1080). `pad` and `stretch` use the
same area filter.

`fit crop` cuts like Take One on every device: first in the x4 picture, then
`INTER_AREA`. The GPU does the same (after 2.1.3) whenever the output is not
larger than the x4 picture: the network output is rounded to uint8
(`uint8(v*255+0.5)`), cropped in whole rows / columns and resized by a GPU
kernel with the same float operations and OpenCV area tables as the bit-exact
`INTER_AREA` port (896x512 -> rows 16..2031 of 3584x2048 -> 1920x1080, 8.57
output lines off top and bottom; 640x480 -> rows 240..1679 of 2560x1920).
Up to 2.1.3 the GPU resized the whole x4 picture with Lanczos-3 to 1920x1098
and cut at the even row 8 instead of 8.57, one line off Take One (25-29 dB
against Take One's crop of the same x4 picture); now the result equals
`cv2.resize(x4[16:2032], (1920, 1080), interpolation=cv2.INTER_AREA)` (OpenCV
5.0) pixel for pixel. The GPU's x4 picture differs from the int8 NPU body by
its quantisation only (~38-39 dB). `scale: 2` with `fit crop` is therefore a
2x2 area mean on the GPU too. `pad`, `stretch` and `crop` to a size larger
than x4 keep the GPU's Lanczos-3 placement (`PlaceContent`: even sizes and
offsets).

One model name may have several files, one per frame width; the request's
frame picks the file.

### `POST /v1/images/upscale`

```json
{"model": "gufo-upscale", "variant": "general",
 "image": "data:image/png;base64,...",
 "scale": 4, "fit": "crop", "output": "png", "response_format": "b64_json"}
```

- `image`: data URL, raw base64, public https URL or `file:` path.
- `images`: a list of 1..32 such images (or `{"image": ...}` objects) with
  the same settings, answered in order as `data[i]` with `width`, `height`
  and `input`; sync and `b64_json` only.
- `scale`: 4 (native) or 2 (the x4 picture resized to half: OpenCV-exact
  `INTER_AREA` with `fit crop` on every device, Lanczos-3 on the GPU with
  `pad` / `stretch`). `size: "WxH"` or
  `width` + `height` (at most 8192 on a side) replace the scale and use
  `fit` (`crop` default, `pad`, `stretch`).
- `output`: `png` (default) or `jpg` (`quality` 1..100, default 90).
- `response_format`: `b64_json` (default) wraps the encoded image in JSON,
  `png` / `jpg` return the bytes.
- `async: true` answers at once with a job object; read it at
  `GET /v1/images/upscale/{id}` and fetch the result (bytes or the same
  JSON as the sync answer) at `GET /v1/images/upscale/{id}/content`.

The sync answer is `{created, model, variant, device, model_file, width,
height, input, scale, fit, format, data: [{b64_json}], cost: {seconds,
queue_seconds, body_seconds, post_seconds, encode_seconds}}` with a
`Server-Timing` header. The device runs one image at a time (in an `images`
list the next image decodes and the previous one encodes meanwhile); more
than 8 waiting requests give 429.

### `POST /v1/videos/upscale`

```json
{"model": "gufo-upscale", "variant": "anime",
 "video": "data:video/mp4;base64,...", "size": "1920x1080",
 "fit": "crop", "crf": 18, "keep_audio": true}
```

An async job (`"async": false` waits and returns the MP4): FFmpeg decodes
the upload to RGB24 frames, every frame runs through the network (between
any sync image requests), FFmpeg encodes H.264 at the input frame rate with
the first audio stream copied unchanged. `scale` 2 | 4 keeps the aspect
ratio; `size` / `width` + `height` (even, at most 8192) use `fit`. The
answer is a video-job object `{id, object: "video", status: queued |
in_progress | completed | failed | cancelled, progress (0..100), model,
variant, size, input, upscale: {frames, frames_done, fps, audio, crf},
cost, error}` (202 + `Location`); `GET /v1/videos/{id}` reads it,
`GET /v1/videos/{id}/content` returns the result (409 while running) and
`DELETE /v1/videos/{id}` cancels a running job or drops a finished result.
Up to 8 jobs queue; finished results stay for an hour (at most 8). Uploads
are limited to 512 MiB.

`"output_format": "png_zip"` returns the frames losslessly instead of an
MP4: every upscaled frame as an 8-bit RGB PNG (the one-pass encoder below)
in a ZIP without compression (stored entries, CRC-32), named `f_00001.png`,
`f_00002.png`, ... in frame order, `Content-Type: application/zip`. There is
no audio and no `crf`; the job object says `"output_format": "png_zip"`.
Use it when the pipeline encodes the delivery itself, so the frames are not
compressed twice. The archive is held in memory like the MP4 and is limited
to 4 GiB and 65535 frames (no ZIP64): a 1920x1080 frame is about 3-5 MB as
PNG, so send film segments, not whole films.

### Speed

Seconds per 832x512 frame (anime) at the worker, Strix Halo (gfx1151, XDNA 2):

| Path | GPU | NPU |
|---|---:|---:|
| image, x4 PNG (3328x2048) | 0.16 | 0.34 |
| image, `size` 1920x1080 PNG | 0.10 | 0.27 |
| image, `size` 1920x1080 JPEG 95 | 0.07 | 0.21 |
| `images` list of 10, 1920x1080 PNG | 0.06 | 0.22 |
| `/v1/videos/upscale`, 240 frames, 1920x1080 | 0.040 | 0.19 |

The network takes 37 ms on the GPU (145 ms on the NPU). For film segments
send the whole MP4 with the delivery size (`"video": ..., "variant": "anime",
"size": "1920x1080", "fit": "crop", "async": false`); a job adds about 0.4 s.
For lossless frames that the pipeline encodes itself:

```json
{"model": "gufo-upscale", "variant": "anime",
 "video": "data:video/mp4;base64,...", "size": "1920x1080",
 "fit": "crop", "output_format": "png_zip", "async": false}
```

### Router section

The router forwards `/v1/images/upscale` and `/v1/videos/upscale` by the
body's `model` (the section id) and records the job id of a
`/v1/videos/upscale` answer like a `/v1/videos` job, so the status, content
and cancel reads reach the same worker; async image jobs are tracked like
music jobs (`/v1/images/upscale/{id}`). Every `model =` / `--model` file
without `NAME=` takes the served name, so the three variants sit behind one
id and a request picks one with `"variant"` (`general`, `wdn`, `anime`).
Do not write `--model wdn=...` or `--model anime=...`: `NAME=` is a model
name, not a variant, so the worker then serves extra models `wdn` and
`anime`, while `gufo-upscale` knows only `general` and a request for another
variant gets 404. The production section (`gufo-router-combo7.ini`, 2.1):

```ini
[external/gufo-upscale]
command = D:\llm-modelle\gufo\app\gufo-2.1.0-2026-11-14\gufo.exe
args = serve upscale --device gpu --model D:\llm-modelle\gufo\modelle\upscale\realesr-general-wdn-x4v3.safetensors --model D:\llm-modelle\gufo\modelle\upscale\realesr-animevideov3.safetensors
model = D:\llm-modelle\gufo\modelle\upscale\realesr-general-x4v3.safetensors
served-model-name = gufo-upscale
max-request-bytes = 134217728
router-gpu-gb = 1
router-host-gb = 1
```

An NPU section has `args = serve upscale --device npu` and the body ONNX as
`model =` (Take One's `body512q_h280_h3.onnx` for 832x512 frames).

`npu-upscale-probe` (built with `GUFO_ENABLE_ONNXRUNTIME`) runs one image
and compares it with a reference: `npu-upscale-probe --device cpu --model
body896q_h288_h3.onnx --input in.png --size 1920x1080 --compare ref.png`.

## Implementation

`src/models/realesrgan`: every convolution runs through the implicit-GEMM
3x3 kernel of the RIFE port (FP16 WMMA, FP32 accumulation, 64x64 tiles) with
the per-channel PReLU in its epilogue; the last convolution writes FP32. The
pixel shuffle, the nearest-neighbour base, the clamp and the two separable
resize passes are FP32 kernels; the result is downloaded as RGB24. PNG
results are written in one pass (Sub filter, zlib level 1, row strips
deflated on up to 8 threads into one stream).
