# TRELLIS.2 image to textured 3D

`gufo serve objekt` with `trellis2 = DIR` runs TRELLIS.2-4B (Microsoft Research, MIT)
natively: image to GLB with vertex colours or texture over the 512 path (sparse-structure
flow, shape flow, texture flow and the three sparse decoders), and over the original's
`1024_cascade` / `1536_cascade` pipelines when the 1024 flows are installed.
`POST /v1/3d/generations {model: "trellis2", image | images, steps, seed, resolution,
multi_mode, max_faces, texture, preview}`; status, content and preview reads as for
[TripoSG](triposg.md).

## Resolution and several views (branch `feat/trellis2-hq`)

| Field | Values | Meaning |
|---|---|---|
| `resolution` | `512` (default), `1024`, `1536` (`256` is served at 512) | `1024` / `1536` = the original `1024_cascade` / `1536_cascade`: the 512 shape flow, the shape decoder's subdivision levels give the voxels on the 512 grid, they are quantised to a 64³ / 96³ token grid, the shape (and texture) flow 1024 runs there with DINOv3 tokens of the 1024 px image, the decoders build the mesh on the 1024 / 1536 grid. Above 49 152 tokens a 1536 request steps down by 128 (1408, 1280, 1152, 1024) like the original; the grid used is `data[0].resolution`, the token count `data[0].tokens`. Without the 1024 files the worker answers 400 `invalid_resolution` for 1024 / 1536. |
| `images` | array of 1-4 base64 PNGs with alpha or data URLs, instead of `image` | Views of the same object (front, sides, back). Each view is cropped to its own alpha box and encoded by DINOv3; every flow is conditioned on all views. `image` and `images` together are 400 `invalid_images`; more than 4 are 400 `too_many_images` (TripoSG takes one image). |
| `multi_mode` | `stochastic` (default), `multidiffusion` | Only with several views. `stochastic` (TRELLIS v1 `run_multi_image`, PR microsoft/TRELLIS.2#104): step *i* of every flow uses view *i* mod *n*, the same cost as one image. `multidiffusion` (PR #104): the conditional predictions of all views are averaged every step, then CFG; *n* + 1 model calls per guided step. Other values are 400 `invalid_multi_mode`. |

The job snapshot carries `views` (and `multi_mode` with several views). Requests without
the new fields behave as before. Higher resolutions keep far more detail before the
decimation: raise `max_faces` (default 200 000) to keep it in the GLB. Four 1 MB PNGs
fit the worker's default request limit of 8 MB; larger sets need `max-request-bytes`.

The 1024 flows are loaded on the first 1024 / 1536 request (a few seconds) and stay
resident: +4.8 GB GPU next to the 512 weights. Memory and time at 1024 / 1536 are not
measured yet.

**MCP:** `objekt_3d` takes `aufloesung` 256 | 512 | 1024 | 1536, `weitere_ansichten`
(up to 3 more views, sent as `images` with `bild` first) and `multi_modus`.

**Not in release 2.1.0.** The worker and its converter are on the fork branch `feat/trellis2`
(merged with 2.1 as `feat/trellis2-21`); the router side (`/v1/3d/generations`) is in 2.1.0.

**Built with DINOv3.** The image encoder is DINOv3 ViT-L/16 (Meta), gated on Hugging Face
under the DINOv3 License. It is not part of the published weights; convert it yourself (below).

## Files

One safetensors per part, tensor names unchanged except the per-head qk RMS gammas (flattened)
and the 3-D convolutions (one `[Co, 27*Ci]` layout); matrices BF16, the rest F32.
`tools/trellis2/convert_hf_to_gufo.py` (on the feature branch, run with `python -P -E`)
checks each source file (safetensors header, SHA-256 against the Hugging Face LFS id) and
records `gufo.trellis2.source`, `gufo.trellis2.source_file` and `gufo.trellis2.license`
in every output.

### Converted weights

The six Microsoft parts are published as
[D3rPapst/trellis2-gufo](https://huggingface.co/D3rPapst/trellis2-gufo) (9.8 GB, MIT, licence
text included; `trellis2-ss_dec` comes from microsoft/TRELLIS-image-large, also MIT).

```powershell
hf download D3rPapst/trellis2-gufo --local-dir D:\llm-modelle\gufo\modelle\microsoft\gufo-trellis2
```

| File | Size | Content |
|---|---:|---|
| `trellis2-ss_flow.safetensors` | 2.59 GB | sparse-structure flow DiT |
| `trellis2-ss_dec.safetensors` | 148 MB | sparse-structure decoder |
| `trellis2-shape_flow_512.safetensors` | 2.59 GB | shape flow DiT, 512 path |
| `trellis2-shape_dec.safetensors` | 949 MB | sparse shape decoder |
| `trellis2-tex_flow_512.safetensors` | 2.59 GB | texture flow DiT, 512 path |
| `trellis2-tex_dec.safetensors` | 949 MB | sparse texture decoder |
| `trellis2-shape_flow_1024.safetensors` | 2.59 GB | shape flow DiT, 1024 / 1536 cascade (optional) |
| `trellis2-tex_flow_1024.safetensors` | 2.59 GB | texture flow DiT, 1024 / 1536 cascade (optional, with texture) |

The two 1024 files are not in the published set yet; convert them from the Microsoft
snapshot (MIT):

```powershell
hf download microsoft/TRELLIS.2-4B ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json --local-dir <snapshot>
python -P -E tools\trellis2\convert_hf_to_gufo.py --trellis2 <snapshot> --out D:\llm-modelle\gufo\modelle\microsoft\gufo-trellis2 --parts shape_flow_1024,tex_flow_1024
```

**Add the image encoder yourself.** Accept Meta's terms at
[facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m),
download the snapshot (`hf download facebook/dinov3-vitl16-pretrain-lvd1689m --local-dir <snapshot>`)
and run

```powershell
python -P -E tools\trellis2\convert_hf_to_gufo.py --dinov3 <snapshot> --out D:\llm-modelle\gufo\modelle\microsoft\gufo-trellis2 --parts dinov3
```

That writes `trellis2-dinov3.safetensors` (1.21 GB, F32) next to the six files; keep a copy
of the snapshot's DINOv3 licence (`LICENSE.md`) beside it. Without this file the worker
cannot start.

## Router section

From the production preset of the 64 GB machine (`command` points at the build of the
feature branch):

```ini
[external/trellis2]
command = D:\llm-modelle\gufo\app\gufo-trellis2-2026-11-12\gufo.exe
args = serve objekt
trellis2 = D:\llm-modelle\gufo\modelle\microsoft\gufo-trellis2
trellis2-texture = true
root = D:\llm-modelle\gufo\jobs\trellis2
ttl = 86400
served-model-name = trellis2
max-request-bytes = 33554432
router-gpu-gb = 20
router-host-gb = 3
router-idle-seconds = 900
```

## Print-ready (druckfertig)

`print_ready: true` (with `print_eye_height_mm`, default 28, or `print_height_mm`) adds a
watertight, smooth single body in millimetres for miniature printing: `print.stl` (binary,
Z up, feet at z = 0) and `print.glb` (vertex colours, real size in metres), next to the normal
GLB. Fields and algorithm: [server.md](../internal/server.md) ("Print-ready 3D"). The step is
shared with TripoSG and runs on the CPU after the GPU work; on the /objekt/ page it is the
checkbox "druckfertig (geschlossen, glatt)" with "Augenhöhe / Gesamthöhe" in mm.

Measured (23 Nov 2026, CPU of the 64 GB Strix Halo, `gufo-mesh-print` on the 1024 four-view figure,
1 M faces): grid 398×141×729, 19 open slices closed, 1.51 M faces, watertight, one body, winding
consistent, 4.1 s, peak host RAM 0.54 GB (process private bytes). Against the Python reference
(`reparatur.py` v2, same parameters): volume −0.09 % (1536 figure +0.20 %), faces +0.05 %, mean surface
distance 0.3 voxel (p99 0.76 voxel).

## Measured

Strix Halo, 64 GB: GPU peak 19.3 GB with texture, host RAM 1.5-1.8 GB; 38-105 s per object
(chair 38 s, toy 64 s, cup 105 s; 12 Nov 2026).
