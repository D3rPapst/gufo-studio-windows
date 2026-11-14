"""SAM oracle: transformers 5.18 on the CPU, as hf.py segment() calls it (boxes, multimask_output=False).
Usage: make_segment_oracle.py <model_dir> <out_dir> <boxes "x0,y0,x1,y1;..."> <image>...
Writes <stem>.lowres.f32 (pred_masks [n, 256, 256]), .scores.f32 (iou_scores [n]),
.masks.u8 (post_process_masks, [n, H, W] 0/1) and .json (seconds)."""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

torch.set_num_threads(16)
model_dir, out_dir = sys.argv[1], sys.argv[2]
boxes = [[float(v) for v in b.split(",")] for b in sys.argv[3].split(";") if b]
os.makedirs(out_dir, exist_ok=True)
proc = SamProcessor.from_pretrained(model_dir)
model = SamModel.from_pretrained(model_dir).eval()
for path in sys.argv[4:]:
    stem = os.path.splitext(os.path.basename(path))[0]
    img = Image.open(path).convert("RGB")
    with torch.no_grad():
        sin = proc(img, input_boxes=[boxes], return_tensors="pt")
        t = time.time()
        sout = model(**sin, multimask_output=False)
        seconds = time.time() - t
        masks = proc.image_processor.post_process_masks(sout.pred_masks.cpu(), sin["original_sizes"].cpu(),
                                                        sin["reshaped_input_sizes"].cpu())[0]
    sout.pred_masks[0, :, 0].numpy().astype(np.float32).tofile(os.path.join(out_dir, stem + ".lowres.f32"))
    sout.iou_scores[0, :, 0].numpy().astype(np.float32).tofile(os.path.join(out_dir, stem + ".scores.f32"))
    masks[:, 0].numpy().astype(np.uint8).tofile(os.path.join(out_dir, stem + ".masks.u8"))
    json.dump({"boxes": boxes, "seconds": seconds, "size": list(img.size)}, open(os.path.join(out_dir, stem + ".json"), "w"))
    print(stem, "%.2f s" % seconds, flush=True)
