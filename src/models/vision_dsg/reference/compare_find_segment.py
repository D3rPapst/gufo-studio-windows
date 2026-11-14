"""Compare vision-dsg-probe find / segment outputs with make_find_oracle.py / make_segment_oracle.py.
Usage: compare_find_segment.py find <oracle_dir> <probe_dir> <stem>...
       compare_find_segment.py segment <oracle_dir> <probe_dir> <stem>...
find: logits / pred_boxes max-abs (per query; only meaningful when the top-k order agrees), then the
post-processed detections paired by nearest box (same count, boxes +-1 px,
scores +-1e-3, same labels). segment: low-res mask logits cosine / max-abs, IoU scores, mask IoU >= 0.999."""
import json
import os
import sys

import numpy as np

mode, oracle, probe = sys.argv[1], sys.argv[2], sys.argv[3]
ok = True
for stem in sys.argv[4:]:
    o = lambda ext: np.fromfile(os.path.join(oracle, stem + ext), dtype="<f4").astype(np.float64)
    p = lambda ext: np.fromfile(os.path.join(probe, stem + ext), dtype="<f4").astype(np.float64)
    row = {"stem": stem}
    if mode == "find":
        lo, lp = o(".logits.f32"), p(".logits.f32")
        bo, bp = o(".boxes.f32"), p(".boxes.f32")
        finite = np.isfinite(lo) & np.isfinite(lp)
        row["logits_maxabs"] = float(np.abs(lo[finite] - lp[finite]).max()) if lo.size == lp.size else "size"
        row["boxes_maxabs"] = float(np.abs(bo - bp).max()) if bo.size == bp.size else "size"
        dets_o = sorted(json.load(open(os.path.join(oracle, stem + ".json")))["detections"], key=lambda d: -d["score"])
        dets_p = []
        for line in open(os.path.join(probe, stem + ".dets.txt"), encoding="utf-8"):
            f = line.rstrip("\n").split("\t")
            dets_p.append({"score": float(f[0]), "box": [float(v) for v in f[1:5]], "label": f[5] if len(f) > 5 else ""})
        dets_p.sort(key=lambda d: -d["score"])
        row["count"] = [len(dets_o), len(dets_p)]
        good = len(dets_o) == len(dets_p)
        worst_box = worst_score = 0.0
        labels_ok = True
        # Pair each oracle box with the nearest unused probe box (the answer order
        # follows the queries, and near-equal scores may swap places).
        left, pairs = list(dets_p), []
        for a in dets_o:
            if not left:
                break
            b = min(left, key=lambda d: max(abs(x - y) for x, y in zip(a["box"], d["box"])))
            left.remove(b)
            pairs.append((a, b))
        for a, b in pairs:
            worst_box = max(worst_box, max(abs(x - y) for x, y in zip(a["box"], b["box"])))
            worst_score = max(worst_score, abs(a["score"] - b["score"]))
            labels_ok = labels_ok and a["label"] == b["label"]
        row.update(box_px=worst_box, score=worst_score, labels_ok=labels_ok)
        good = good and worst_box <= 1.0 and worst_score <= 1e-3 and labels_ok
    else:
        lo, lp = o(".lowres.f32"), p(".lowres.f32")
        row["lowres_cosine"] = float(lo @ lp / (np.linalg.norm(lo) * np.linalg.norm(lp))) if lo.size == lp.size else "size"
        row["lowres_maxabs"] = float(np.abs(lo - lp).max()) if lo.size == lp.size else "size"
        row["iou_scores"] = [o(".scores.f32").tolist(), p(".scores.f32").tolist()]
        mo = np.fromfile(os.path.join(oracle, stem + ".masks.u8"), dtype=np.uint8)
        mp = np.fromfile(os.path.join(probe, stem + ".masks.u8"), dtype=np.uint8)
        n = len(row["iou_scores"][0])
        ious = []
        if mo.size == mp.size:
            for a, b in zip(mo.reshape(n, -1), mp.reshape(n, -1)):
                inter, union = np.logical_and(a, b).sum(), np.logical_or(a, b).sum()
                ious.append(float(inter / union) if union else 1.0)
        row["mask_iou"] = ious
        good = bool(ious) and min(ious) >= 0.999
    row["pass"] = bool(good)
    ok = ok and good
    print(json.dumps(row))
sys.exit(0 if ok else 1)
