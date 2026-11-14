# Oracle data for the "pose" task of /v1/vision/detect: runs the film pipeline's own reference
# (posen_erkennung.erkenne: RT-DETR r50vd COCO+O365 finds the persons, ViTPose base puts 17 COCO points on each,
# transformers on the CPU) on fixed keyframes and saves, per image, the decoded RGB pixels, the model inputs,
# intermediate tensors and the final boxes and keypoints. The C++ test pose_oracle_test reads the result.
# Written with AI assistance (Claude Code).
#
# Run in a venv with the Regie's versions (transformers 5.18.0, torch 2.13.0 CPU, torchvision 0.29.1, scipy 1.18.1,
# numpy 2.5.3, Pillow 12.3.0):
#   python -I make_pose_oracle.py <model_root> <out_dir> k1.png k2.png ...
# <model_root> holds rtdetr_r50vd_coco_o365/ and vitpose_base_simple/ (Hugging Face folders);
# --wholebody <dw-ll_ucoco_384.onnx> adds DWPose (posen_erkennung.wholebody, copied below) per person.
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from safetensors.numpy import save_file
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

MIN_SCORE = 0.35  # posen_erkennung.erkenne(min_score=0.35)
MAX_PERSONS = 4   # erkenne(max_personen=4)

# posen_erkennung.py (film pipeline), wholebody() and _wb_warp() verbatim apart from the session argument.
WB_GROESSE = (288, 384)
WB_MEAN, WB_STD = np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375])


def _wb_warp(center, scale, out_wh):
    import cv2
    w, h = out_wh
    src = np.float32([center, center + [0, -scale[0] * 0.5], center + [scale[0] * 0.5, -scale[0] * 0.5]])
    dst = np.float32([[w * 0.5, h * 0.5], [w * 0.5, h * 0.5 - w * 0.5], [w, h * 0.5 - w * 0.5]])
    return cv2.getAffineTransform(src, dst)


def wholebody(session, img, box, taps):
    import cv2
    a = np.asarray(img, dtype=np.float32)
    x1, y1, x2, y2 = box
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    scale = np.array([x2 - x1, y2 - y1], dtype=np.float32) * 1.25
    w, h = WB_GROESSE
    scale = np.array([scale[0], scale[0] / (w / h)]) if scale[0] > scale[1] * (w / h) else np.array([scale[1] * (w / h), scale[1]])
    M = _wb_warp(center, scale, WB_GROESSE)
    crop = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR)
    x = ((crop - WB_MEAN) / WB_STD).transpose(2, 0, 1)[None].astype(np.float32)
    sx, sy = session.run(None, {"input": x})
    taps.append((x[0], sx[0], sy[0]))
    kx, ky = sx[0].argmax(1), sy[0].argmax(1)
    vals = np.minimum(sx[0].max(1), sy[0].max(1))
    pts = np.stack([kx, ky], 1).astype(np.float32) / 2.0
    pts = pts / np.array([w, h]) * scale + center - scale / 2
    return np.concatenate([pts, vals[:, None]], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_root")
    ap.add_argument("out_dir")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--wholebody")
    a = ap.parse_args()
    session = None
    if a.wholebody:
        import onnxruntime as ort
        session = ort.InferenceSession(a.wholebody, providers=["CPUExecutionProvider"])
    os.makedirs(a.out_dir, exist_ok=True)
    dp = AutoProcessor.from_pretrained(os.path.join(a.model_root, "rtdetr_r50vd_coco_o365"))
    dm = RTDetrForObjectDetection.from_pretrained(os.path.join(a.model_root, "rtdetr_r50vd_coco_o365")).eval()
    pp = AutoProcessor.from_pretrained(os.path.join(a.model_root, "vitpose_base_simple"))
    pm = VitPoseForPoseEstimation.from_pretrained(os.path.join(a.model_root, "vitpose_base_simple")).eval()
    import transformers, torchvision, scipy, cv2, onnxruntime
    manifest = {"versions": {"transformers": transformers.__version__, "torch": torch.__version__,
                             "torchvision": torchvision.__version__, "scipy": scipy.__version__,
                             "numpy": np.__version__, "pillow": Image.__version__, "opencv": cv2.__version__,
                             "onnxruntime": onnxruntime.__version__,
                             "det_processor": type(dp).__name__, "pose_processor": type(pp).__name__},
                "min_score": MIN_SCORE, "max_persons": MAX_PERSONS, "images": []}
    for i, path in enumerate(a.images):
        img = Image.open(path).convert("RGB")
        tensors = {"rgb": np.ascontiguousarray(np.asarray(img, dtype=np.uint8))}
        with torch.no_grad():
            inp = dp(images=img, return_tensors="pt")
            out = dm(**inp, output_hidden_states=True)
            res = dp.post_process_object_detection(out, target_sizes=torch.tensor([(img.height, img.width)]),
                                                   threshold=MIN_SCORE)[0]
        tensors["det_input"] = inp["pixel_values"][0].numpy()
        tensors["det_logits"] = out.logits[0].numpy()
        tensors["det_boxes"] = out.pred_boxes[0].numpy()
        tensors["det_init_reference"] = out.init_reference_points[0].numpy()
        for lvl, fmap in enumerate(out.encoder_last_hidden_state):
            tensors[f"det_encoder{lvl}"] = fmap[0].numpy()
        detections = [{"score": float(s), "label": int(l), "box": [float(v) for v in b]}
                      for s, l, b in zip(res["scores"], res["labels"], res["boxes"])]
        boxes = [(float(s), [float(v) for v in b]) for s, l, b in zip(res["scores"], res["labels"], res["boxes"])
                 if int(l) == 0]
        boxes = sorted(boxes, key=lambda x: -x[0])[:MAX_PERSONS]
        persons = []
        if boxes:
            xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for _, b in boxes]
            with torch.no_grad():
                pin = pp(img, boxes=[xywh], return_tensors="pt")
                pout = pm(**pin)
                poses = pp.post_process_pose_estimation(pout, boxes=[xywh])[0]
            tensors["pose_input"] = pin["pixel_values"].numpy()
            tensors["pose_heatmaps"] = pout.heatmaps.numpy()
            wb_taps = []
            for (score, box), po in zip(boxes, poses):
                person = {"score": score, "box": box,
                          "keypoints": po["keypoints"].numpy().astype(float).tolist(),
                          "scores": po["scores"].numpy().astype(float).tolist()}
                if session is not None:
                    person["wholebody"] = wholebody(session, img, box, wb_taps).astype(float).tolist()
                persons.append(person)
            if wb_taps:
                tensors["wb_input"] = np.stack([t[0] for t in wb_taps])
                tensors["wb_simcc_x"] = np.stack([t[1] for t in wb_taps])
                tensors["wb_simcc_y"] = np.stack([t[2] for t in wb_taps])
        name = f"pose_{i}.safetensors"
        save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, os.path.join(a.out_dir, name))
        manifest["images"].append({"source": os.path.basename(path), "file": name, "width": img.width,
                                   "height": img.height, "detections": detections, "persons": persons})
        print(os.path.basename(path), len(detections), "detections,", len(persons), "persons", flush=True)
    with open(os.path.join(a.out_dir, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
