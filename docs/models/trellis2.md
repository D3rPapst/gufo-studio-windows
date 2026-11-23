# TRELLIS.2 image to textured 3D

`gufo serve objekt` with `trellis2 = DIR` runs TRELLIS.2-4B (Microsoft Research, MIT)
natively: image to GLB with vertex colours or texture over the 512 path (sparse-structure
flow, shape flow, texture flow and the three sparse decoders).
`POST /v1/3d/generations {model: "trellis2", image, steps, seed, max_faces, texture,
preview}`; status, content and preview reads as for [TripoSG](triposg.md).

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
