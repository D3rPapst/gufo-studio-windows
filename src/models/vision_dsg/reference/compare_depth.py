"""Compare vision-dsg-probe depth outputs with make_depth_oracle.py.
Usage: compare_depth.py <oracle_dir> <probe_dir> <stem>...
Prints per stem: pixel max-abs, predicted_depth and full-size cosine and
max-abs (also relative to the oracle's range). Exit 1 when cosine < 0.9999
or the relative max-abs exceeds 1e-3."""
import json
import os
import sys

import numpy as np


def load(path):
    return np.fromfile(path, dtype="<f4").astype(np.float64)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


oracle, probe = sys.argv[1], sys.argv[2]
ok = True
for stem in sys.argv[3:]:
    meta = json.load(open(os.path.join(oracle, stem + ".json")))
    row = {"stem": stem, "net": meta["pixel_shape"][2:]}
    a, b = load(os.path.join(oracle, stem + ".pixel.f32")), load(os.path.join(probe, stem + ".pixel.f32"))
    row["pixel_maxabs"] = float(np.abs(a - b).max()) if a.size == b.size else "size %d vs %d" % (a.size, b.size)
    for part in ("pred", "full"):
        a, b = load(os.path.join(oracle, stem + "." + part + ".f32")), load(os.path.join(probe, stem + "." + part + ".f32"))
        if a.size != b.size:
            row[part] = "size %d vs %d" % (a.size, b.size)
            ok = False
            continue
        c = cosine(a, b)
        m = float(np.abs(a - b).max())
        rel = m / float(a.max() - a.min())
        row[part] = {"cosine": round(c, 8), "maxabs": m, "rel_maxabs": rel, "range": [float(a.min()), float(a.max())]}
        ok = ok and c >= 0.9999 and rel <= 1e-3
    print(json.dumps(row))
sys.exit(0 if ok else 1)
