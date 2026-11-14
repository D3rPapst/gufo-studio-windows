# Oracle data for /v1/vision/detect "wd14_tags": runs dghs-imgutils itself (the film pipeline's reference,
# imgutils.tagging.get_wd14_tags with the default SwinV2_v3 model) on fixed keyframes and on the person
# cut-outs the pipeline tags (erkennung.py: detect_person on the gamma-0.5 copy when the mean luma is < 70,
# boxes rounded, 8 % of max(w, h) margin, Pillow crop of the original). Saves per picture the RGB pixels,
# the 448x448 BGR model input of _prepare_image_for_tagging, all 10861 probabilities, the 1024 embedding
# and the rating / general (> 0.35) / character (> 0.85) dicts. The C++ test wd14_oracle_test reads it.
# Written with AI assistance (Claude Code).
#
# Run in the venv that has dghs-imgutils (same version as the film pipeline):
#   python -I make_wd14_oracle.py <out_dir> k1.png k2.png ... [--local-root DIR]
# --local-root DIR reads the ONNX files and selected_tags.csv from DIR/<owner>__<repo>/... (the gufo
# download layout, QUELLE.txt) instead of the Hugging Face cache; nothing is fetched then.
import argparse
import json
import os
import time

import numpy as np
from PIL import Image, ImageStat
from safetensors.numpy import save_file

import imgutils
from imgutils.detect import detect_person
from imgutils.tagging import get_wd14_tags
from imgutils.tagging.wd14 import _prepare_image_for_tagging

DUNKEL = 70  # erkennung.py: mean luma below which the detectors see the brightened copy
MARGIN = 0.08


def use_local_root(root):
    import imgutils.generic.yolo as yolo
    import imgutils.tagging.wd14 as wd14
    try:
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:  # older huggingface_hub
        from huggingface_hub.utils import EntryNotFoundError

    def local_download(repo_id, filename, **_kwargs):
        path = os.path.join(root, repo_id.replace("/", "__"), *filename.split("/"))
        if not os.path.isfile(path):
            raise EntryNotFoundError(f"{repo_id}/{filename} not in {root}")
        return path

    yolo.hf_hub_download = local_download
    yolo.YOLOModel._check_model_name = lambda self, name: None
    wd14.hf_hub_download = local_download


def tag(im):
    a = time.perf_counter()
    rating, general, character, pred, emb = get_wd14_tags(
        im, fmt=("rating", "general", "character", "prediction", "embedding"))
    return {"rating": rating, "general": general, "character": character,
            "seconds": time.perf_counter() - a}, pred, emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--local-root")
    args = ap.parse_args()
    if args.local_root:
        use_local_root(args.local_root)
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {"imgutils": getattr(imgutils, "__version__", None), "general_threshold": 0.35,
                "character_threshold": 0.85, "margin": MARGIN, "images": []}
    try:
        from imgutils.config.meta import __VERSION__
        manifest["imgutils"] = __VERSION__
    except ImportError:
        pass
    get_wd14_tags(Image.new("RGB", (64, 64)))  # load the model before timing
    for i, path in enumerate(args.images):
        im = Image.open(path).convert("RGB")
        rgb = np.ascontiguousarray(np.asarray(im, dtype=np.uint8))
        tensors = {"rgb": rgb}
        entry = {"source": os.path.basename(path), "file": f"wd14_{i}.safetensors", "width": im.width,
                 "height": im.height, "crops": []}
        tensors["input448"] = np.ascontiguousarray(_prepare_image_for_tagging(im, 448)[0])
        res, pred, emb = tag(im)
        entry.update(res)
        tensors["prediction"] = np.asarray(pred, np.float32)
        tensors["embedding"] = np.asarray(emb, np.float32)
        det = im.point(lambda v: round(255 * (v / 255) ** 0.5)) \
            if ImageStat.Stat(im.convert("L")).mean[0] < DUNKEL else im
        persons = [[round(v) for v in box] for box, _, _ in detect_person(det)]
        for k, (x1, y1, x2, y2) in enumerate(persons):
            rand = MARGIN * max(x2 - x1, y2 - y1)
            crop_box = (max(0, x1 - rand), max(0, y1 - rand), min(im.width, x2 + rand), min(im.height, y2 + rand))
            a = im.crop(crop_box)
            c = {"person": [x1, y1, x2, y2], "crop_box": list(crop_box),
                 "crop_pixels": [int(round(v)) for v in crop_box], "width": a.width, "height": a.height}
            tensors[f"crop{k}.input448"] = np.ascontiguousarray(_prepare_image_for_tagging(a, 448)[0])
            r, p, e = tag(a)
            c.update(r)
            tensors[f"crop{k}.prediction"] = np.asarray(p, np.float32)
            tensors[f"crop{k}.embedding"] = np.asarray(e, np.float32)
            entry["crops"].append(c)
        save_file(tensors, os.path.join(args.out_dir, entry["file"]))
        manifest["images"].append(entry)
        print(path, f"{len(entry['general'])} general tags, {len(persons)} persons, {entry['seconds']:.2f} s")
    with open(os.path.join(args.out_dir, "manifest.json"), "wb") as f:
        f.write(json.dumps(manifest, indent=1, ensure_ascii=False).encode("utf-8"))


if __name__ == "__main__":
    main()
