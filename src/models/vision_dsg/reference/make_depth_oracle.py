"""Depth Anything V2 oracle: transformers 5.18 on the CPU, exactly as hf.py.
Usage: depth_ref.py <model_dir> <out_dir> <image>...
Writes <out_dir>/<stem>.pixel.f32 (pixel_values), .pred.f32 (predicted_depth),
.full.f32 (bicubic to the image size, hf.py), .json (shapes, seconds)."""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

torch.set_num_threads(16)
model_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()
proc = AutoImageProcessor.from_pretrained(model_dir)
model = AutoModelForDepthEstimation.from_pretrained(model_dir).eval()
print("processor", type(proc).__name__, "load", round(time.time() - t0, 2), "s", flush=True)
for path in sys.argv[3:]:
    stem = os.path.splitext(os.path.basename(path))[0]
    img = Image.open(path).convert("RGB")
    with torch.no_grad():
        t = time.time()
        inp = proc(images=img, return_tensors="pt")
        t_pre = time.time() - t
        t = time.time()
        pred = model(**inp).predicted_depth
        t_net = time.time() - t
        p4 = pred.reshape(1, 1, *pred.shape[-2:]).float()
        full = torch.nn.functional.interpolate(p4, size=img.size[::-1], mode="bicubic", align_corners=False)[0, 0]
    inp["pixel_values"].numpy().astype(np.float32).tofile(os.path.join(out_dir, stem + ".pixel.f32"))
    pred.numpy().astype(np.float32).tofile(os.path.join(out_dir, stem + ".pred.f32"))
    full.numpy().astype(np.float32).tofile(os.path.join(out_dir, stem + ".full.f32"))
    meta = {"image": path, "width": img.size[0], "height": img.size[1],
            "pixel_shape": list(inp["pixel_values"].shape), "pred_shape": list(pred.shape),
            "seconds_pre": t_pre, "seconds_net": t_net}
    json.dump(meta, open(os.path.join(out_dir, stem + ".json"), "w"), indent=1)
    print(stem, meta["pixel_shape"], "pre %.2fs net %.2fs" % (t_pre, t_net), flush=True)
