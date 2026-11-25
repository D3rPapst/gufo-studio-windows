# TripoSG image to 3D

`gufo serve objekt` turns one image into a GLB mesh with TripoSG (Tripo and VAST AI
Research, MIT): DINOv2-L image encoder, rectified-flow DiT and the shape-VAE decoder,
natively on HIP. gufo uses the image's alpha channel or its own cut-out; the original
background remover (RMBG, non-commercial) is not needed and not included.

**Not in release 2.1.0.** The worker and its converter are on the fork branch `feat/triposg`
(merged with 2.1 as `feat/triposg-21`). The router side is in 2.1.0: `/v1/3d/generations`
wraps a `gufo serve objekt` worker as a job (see [server.md](../internal/server.md)).

`POST /v1/3d/generations {model: "triposg", image, steps, guidance, seed, resolution 256|512,
max_faces}` answers 202 with a job; `GET /v1/3d/generations/{id}` is the status,
`.../content` the GLB, `.../preview` a three-view PNG. The router stores `model.glb`,
`preview.png` and `response.json` in the job and reports the capability `image_to_3d`.
`print_ready: true` (with `print_height_mm` for objects, or `print_eye_height_mm`) adds a
watertight, smooth body in millimetres as `print.stl` and `print.glb`, the same step as for
[TRELLIS.2](trellis2.md#print-ready-druckfertig).

## Files

One safetensors file with the prefixes `dino.*`, `dit.*` and `vae.*` (decoder only);
matrices BF16, norms, biases and tables F32. `tools/triposg/convert_hf_to_gufo.py`
(on the feature branch; standard library + numpy, nothing from the model folder is
executed) builds it from the Hugging Face snapshot and `tools/triposg/verify_gufo_file.py`
checks it.

### Converted weights

Published as [D3rPapst/triposg-gufo](https://huggingface.co/D3rPapst/triposg-gufo) (3.92 GB,
MIT, licence text included; source revision `2c1c516d`, metadata `gufo.triposg.source` /
`gufo.triposg.license` in the file).

```powershell
hf download D3rPapst/triposg-gufo --local-dir D:\llm-modelle\gufo\modelle\VAST-AI\TripoSG\gufo
```

| File | Size | Content |
|---|---:|---|
| `triposg.safetensors` | 3.92 GB | 1128 tensors (423 BF16, 705 F32): `dino.*`, `dit.*`, `vae.*` |

## Router section

From the production preset of the 64 GB machine (`command` points at the build of the
feature branch):

```ini
[external/triposg]
command = D:\llm-modelle\gufo\app\gufo-3d-2026-11-12\gufo.exe
args = serve objekt
model = D:\llm-modelle\gufo\modelle\VAST-AI\TripoSG\gufo\triposg.safetensors
root = D:\llm-modelle\gufo\jobs\objekt
ttl = 86400
served-model-name = triposg
max-request-bytes = 33554432
router-gpu-gb = 8
router-host-gb = 4
router-idle-seconds = 900
```

## Measured

Strix Halo, 64 GB: GPU 1.6 GB after loading, 6.7 GB peak; host RAM 3.9 GB; about 116 s per
object at resolution 512 (12 Nov 2026).
