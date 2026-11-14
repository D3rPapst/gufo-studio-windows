"""Grounding DINO oracle: transformers 5.18 on the CPU, as hf.py segment() calls it.
Usage: make_find_oracle.py [--file-bbox] <model_dir> <out_dir> <text> <image>...
Writes <stem>.logits.f32 ([900, tokens]), .boxes.f32 ([900, 4] cx cy w h) and
.json (input_ids, post_process_grounded_object_detection with 0.3 / 0.25, seconds)."""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection, MMGroundingDinoForObjectDetection

torch.set_num_threads(16)
file_bbox = "--file-bbox" in sys.argv
args = [a for a in sys.argv[1:] if a != "--file-bbox"]
model_dir, out_dir, text = args[0], args[1], args[2]
os.makedirs(out_dir, exist_ok=True)
proc = AutoProcessor.from_pretrained(model_dir)
mm = json.load(open(os.path.join(model_dir, "config.json")))["model_type"] == "mm-grounding-dino"
model = (MMGroundingDinoForObjectDetection if mm else GroundingDinoForObjectDetection).from_pretrained(model_dir).eval()
# transformers 5.18 ties the decoder's box heads wrongly: Grounding DINO's shared
# head (decoder_bbox_embed_share) stays randomly initialised (seeded by the
# process's RNG state, no warning); MM-Grounding-DINO / LLMDet gets head 0 and
# class bias 0 in every layer instead of the per-layer ones in the file. With
# --file-bbox the heads (and biases) are loaded from model.safetensors as the
# checkpoints define them; gufo uses those.
if file_bbox:
    from safetensors import safe_open
    f = safe_open(os.path.join(model_dir, "model.safetensors"), "pt")
    with torch.no_grad():
        for i, head in enumerate(model.bbox_embed):
            if not mm:
                name = "model.decoder.bbox_embed.0"
            else:
                name = "bbox_embed.0" if i == 0 else "model.decoder.bbox_embed.%d" % i
            head = model.bbox_embed[i] = type(head)(256, 256, 4, 3)
            for k, layer in enumerate(head.layers):
                layer.weight.copy_(f.get_tensor(name + ".layers.%d.weight" % k))
                layer.bias.copy_(f.get_tensor(name + ".layers.%d.bias" % k))
        model.model.decoder.bbox_embed = model.bbox_embed
        if mm:
            for i, cls in enumerate(model.class_embed):
                cls.bias = torch.nn.Parameter(f.get_tensor("class_embed.0.bias" if i == 0 else
                                                           "model.decoder.class_embed.%d.bias" % i))
for path in args[3:]:
    stem = os.path.splitext(os.path.basename(path))[0]
    img = Image.open(path).convert("RGB")
    with torch.no_grad():
        inp = proc(images=img, text=text, return_tensors="pt")
        t = time.time()
        out = model(**inp)
        seconds = time.time() - t
        res = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.3, text_threshold=0.25,
                                                          target_sizes=[img.size[::-1]])[0]
    n = inp.input_ids.shape[1]
    out.logits[0, :, :n].numpy().astype(np.float32).tofile(os.path.join(out_dir, stem + ".logits.f32"))
    out.pred_boxes[0].numpy().astype(np.float32).tofile(os.path.join(out_dir, stem + ".boxes.f32"))
    dets = [{"score": float(s), "box": [float(v) for v in b], "label": str(l)}
            for s, b, l in zip(res["scores"], res["boxes"], res["text_labels"])]
    json.dump({"input_ids": inp.input_ids[0].tolist(), "pixel_shape": list(inp.pixel_values.shape),
               "detections": dets, "seconds": seconds}, open(os.path.join(out_dir, stem + ".json"), "w"), indent=1)
    print(stem, list(inp.pixel_values.shape), "%.2f s" % seconds, len(dets), "boxes", flush=True)
