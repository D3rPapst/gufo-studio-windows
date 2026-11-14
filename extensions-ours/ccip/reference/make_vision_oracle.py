# Oracle data for /v1/vision/detect (person/head/face/hand): runs dghs-imgutils itself (the pipeline's
# reference) on fixed keyframes and saves, per image, the decoded RGB pixels, the stretched 640x640 model
# input of imgutils.generic.yolo and the detections of detect_person/detect_heads/detect_faces/detect_hands
# with their default models and thresholds. The C++ test vision_oracle_test reads the result.
# Written with AI assistance (Claude Code).
#
# Run in the venv that has dghs-imgutils (same version as the film pipeline), with the HF cache that holds
# the detector ONNX files (HF_HUB_OFFLINE=1 to make sure nothing is downloaded):
#   python -I make_vision_oracle.py <out_dir> k1.png k2.png ...
# --local-root DIR reads the ONNX files from DIR/<owner>__<repo>/<model>/... (the gufo download layout,
# QUELLE.txt) instead of the Hugging Face cache; nothing is fetched then.
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageStat
from safetensors.numpy import save_file

import imgutils
from imgutils.data import load_image, rgb_encode
from imgutils.detect import detect_faces, detect_hands, detect_heads, detect_person
from imgutils.generic.yolo import _image_preprocess

TASKS = {
    "person": detect_person,
    "head": detect_heads,
    "face": detect_faces,
    "hand": detect_hands,
}


def use_local_root(root):
    import imgutils.generic.yolo as yolo
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--local-root")
    args = ap.parse_args()
    if args.local_root:
        use_local_root(args.local_root)
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {"imgutils": getattr(imgutils, "__version__", None), "images": []}
    try:
        from imgutils.config.meta import __VERSION__
        manifest["imgutils"] = __VERSION__
    except ImportError:
        pass
    for i, path in enumerate(args.images):
        img = load_image(path, mode="RGB")
        rgb = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
        stretched, _, _ = _image_preprocess(img, 640, allow_dynamic=False)
        x = rgb_encode(stretched)
        name = f"vision_{i}.safetensors"
        save_file({"rgb": rgb, "input640": np.ascontiguousarray(x)}, os.path.join(args.out_dir, name))
        # The film pipeline (erkennung.py) runs the detectors on a gamma-0.5 copy when the mean luma is < 70;
        # both variants are recorded, whatever the luma.
        bright = img.point(lambda v: round(255 * (v / 255) ** 0.5))
        entry = {"source": os.path.basename(path), "file": name, "width": img.width, "height": img.height,
                 "mean_luma": ImageStat.Stat(img.convert("L")).mean[0], "detections": {},
                 "detections_gamma05": {}}
        for task, fn in TASKS.items():
            for key, im in (("detections", img), ("detections_gamma05", bright)):
                entry[key][task] = [
                    {"box": [int(v) for v in box], "label": label, "score": float(score)}
                    for box, label, score in fn(im)
                ]
        manifest["images"].append(entry)
        print(path, {t: len(d) for t, d in entry["detections"].items()})
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
