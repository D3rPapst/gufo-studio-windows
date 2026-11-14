# One-time converter: the WD14 SwinV2 v3 tagger as dghs-imgutils loads it (deepghs/wd14_tagger_with_embeddings,
# SmilingWolf/wd-swinv2-tagger-v3/model.onnx, timm swinv2_base_window8_256 at 448 px, window 14) plus the tag list
# (SmilingWolf/wd-swinv2-tagger-v3/selected_tags.csv) -> <stem>.safetensors (F32) + <stem>.json for the native
# runtime (cpu/swinv2.cpp, /v1/vision/detect "wd14_tags").
# Every value is read off the ONNX graph by node name (the export keeps timm's module paths); constant-folded parts
# are taken as they are: per block the relative position bias (16 * sigmoid(cpb_mlp(...)) gathered, [heads, N, N]),
# the attention logit scale (exp(clamp(logit_scale))) and the shifted-window mask. The script asserts the patterns
# it relies on (input x / 127.5 - 1 on BGR, post-norm blocks, GELU-tanh, patch merging order, mean pool, sigmoid).
# Written with AI assistance (Claude Code).
#
# Usage (venv with onnx, numpy, safetensors; run with python -I):
#   python -I convert_wd14.py <model.onnx> <selected_tags.csv> <out_dir> [<stem>]
# <stem> defaults to wd-swinv2-tagger-v3.
import csv
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import onnx
from onnx import helper, numpy_helper
from safetensors.numpy import save_file


def main():
    if len(sys.argv) not in (4, 5):
        raise SystemExit("usage: convert_wd14.py <model.onnx> <selected_tags.csv> <out_dir> [<stem>]")
    onnx_path, tags_path, out_dir = sys.argv[1:4]
    stem = sys.argv[4] if len(sys.argv) == 5 else "wd-swinv2-tagger-v3"
    model = onnx.load(onnx_path)
    g = model.graph
    inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    consts = {}
    by_name = {}
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = numpy_helper.to_array(a.t)
            continue
        by_name[n.name] = n

    def node(name, op=None):
        if name not in by_name:
            raise SystemExit(f"node not found: {name}")
        n = by_name[name]
        if op is not None and n.op_type != op:
            raise SystemExit(f"{name}: {n.op_type}, expected {op}")
        return n

    def value(name):
        if name in inits:
            return inits[name]
        if name in consts:
            return consts[name]
        return None

    def attrs(n):
        return {a.name: helper.get_attribute_value(a) for a in n.attribute}

    def const_input(n):
        found = [v for v in (value(i) for i in n.input) if v is not None]
        assert len(found) == 1, (n.name, list(n.input))
        return found[0]

    tensors = OrderedDict()

    def put(name, arr):
        assert name not in tensors, name
        tensors[name] = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))

    eps_seen = set()

    def layer_norm(prefix, name, dim):
        n = node(prefix + "/LayerNormalization", "LayerNormalization")
        a = attrs(n)
        assert a.get("axis", -1) == -1, a
        eps_seen.add(round(float(a.get("epsilon", 1e-5)), 12))
        w, b = value(n.input[1]), value(n.input[2])
        assert w.shape == (dim,) and b.shape == (dim,), (prefix, w.shape, b.shape)
        put(name + ".w", w)
        put(name + ".b", b)
        return n

    def matmul(prefix, name, k, n_out, bias=True):
        mm = node(prefix + "/MatMul", "MatMul")
        w = value(mm.input[1])
        assert w is not None and w.shape == (k, n_out), (prefix, None if w is None else w.shape)
        put(name + ".w", w)  # [in, out]: row-major K x N, the B of cpu::Gemm
        if bias:
            add = node(prefix + "/Add", "Add")
            assert mm.output[0] in add.input, prefix
            b = const_input(add)
            assert b.shape == (n_out,), (prefix, b.shape)
            put(name + ".b", b)

    # Input: NHWC float BGR 0..255 -> Transpose -> / 127.5 -> - 1.
    inp = g.input[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    assert dims[1] == dims[2] and dims[3] == 3, dims
    img = dims[1]
    t0 = node("/Transpose", "Transpose")
    assert t0.input[0] == inp.name and list(attrs(t0)["perm"]) == [0, 3, 1, 2]
    div = node("/Div", "Div")
    sub = node("/Sub", "Sub")
    scale = float(value(div.input[1]))
    offset = float(value(sub.input[1]))
    assert div.input[0] == t0.output[0] and sub.input[0] == div.output[0]

    conv = node("/core_model/patch_embed/proj/Conv", "Conv")
    assert conv.input[0] == sub.output[0]
    pw, pb = value(conv.input[1]), value(conv.input[2])
    ca = attrs(conv)
    patch = int(pw.shape[2])
    assert pw.shape[1] == 3 and pw.shape[2] == pw.shape[3] == patch, pw.shape
    assert list(ca["strides"]) == [patch, patch] and list(ca.get("pads", [0, 0, 0, 0])) == [0, 0, 0, 0], ca
    embed = int(pw.shape[0])
    put("patch.w", pw)
    put("patch.b", pb)
    layer_norm("/core_model/patch_embed/norm", "patch.norm", embed)

    stages = []
    dim = embed
    res = img // patch
    s = 0
    hidden = 0
    while f"/core_model/layers/layers.{s}/blocks.0/attn/MatMul" in by_name:
        sp = f"/core_model/layers/layers.{s}"
        down = f"{sp}/downsample/reduction/MatMul" in by_name
        if down:
            tr = node(f"{sp}/downsample/Transpose", "Transpose")
            # B, H/2, 2 (dy), W/2, 2 (dx), C -> B, H/2, W/2, dx, dy, C: feature (dx * 2 + dy) * C + c
            assert list(attrs(tr)["perm"]) == [0, 1, 3, 4, 2, 5], attrs(tr)
            matmul(f"{sp}/downsample/reduction", f"s{s}.down", 4 * dim, 2 * dim, bias=False)
            dim *= 2
            res //= 2
            layer_norm(f"{sp}/downsample/norm", f"s{s}.down.norm", dim)
        else:
            assert s == 0, s
        depth = 0
        heads = None
        window = None
        shifts = []
        mask = None
        while f"{sp}/blocks.{depth}/attn/MatMul" in by_name:
            bp = f"{sp}/blocks.{depth}"
            name = f"s{s}.b{depth}"
            matmul(f"{bp}/attn", f"{name}.qkv", dim, 3 * dim)
            mul = node(f"{bp}/attn/Mul", "Mul")
            assert node(f"{bp}/attn/MatMul_1", "MatMul").output[0] in mul.input
            ls = const_input(mul)
            h = int(ls.shape[0])
            assert ls.shape == (h, 1, 1), ls.shape
            put(f"{name}.scale", ls.reshape(h))
            add1 = node(f"{bp}/attn/Add_1", "Add")
            assert mul.output[0] in add1.input
            rb = const_input(add1)
            n_tok = int(rb.shape[-1])
            assert rb.shape == (1, h, n_tok, n_tok), rb.shape
            ws = int(round(n_tok ** 0.5))
            assert ws * ws == n_tok and res % ws == 0, (ws, res)
            put(f"{name}.relbias", rb.reshape(h, n_tok, n_tok))
            if heads is None:
                heads, window = h, ws
            assert (heads, window) == (h, ws)
            node(f"{bp}/attn/softmax/Softmax", "Softmax")
            # Shifted window: torch.roll by -shift (Slice [shift:] + [:shift]) and the -100 mask.
            if f"{bp}/attn/Add_2" in by_name:
                m = const_input(node(f"{bp}/attn/Add_2", "Add"))
                nw = (res // ws) ** 2
                assert m.shape == (1, nw, 1, n_tok, n_tok), m.shape
                m = m.reshape(nw, n_tok, n_tok)
                if mask is None:
                    mask = m
                    put(f"s{s}.mask", m)
                assert np.array_equal(mask, m), bp
                sl = node(f"{bp}/Slice", "Slice")
                shift = int(value(sl.input[1]).reshape(-1)[0])
                assert 0 < shift < ws, shift
                shifts.append(shift)
            else:
                assert f"{bp}/Slice" not in by_name, bp
                shifts.append(0)
            matmul(f"{bp}/attn/proj", f"{name}.proj", dim, dim)
            n1 = layer_norm(f"{bp}/norm1", f"{name}.norm1", dim)
            # post-norm: x + norm1(attn(x)), then x + norm2(mlp(x))
            assert n1.output[0] in node(f"{bp}/Add", "Add").input
            fc1 = node(f"{bp}/mlp/fc1/MatMul", "MatMul")
            hidden = int(value(fc1.input[1]).shape[1])
            matmul(f"{bp}/mlp/fc1", f"{name}.fc1", dim, hidden)
            node(f"{bp}/mlp/act/Tanh", "Tanh")  # GELU, tanh approximation
            matmul(f"{bp}/mlp/fc2", f"{name}.fc2", hidden, dim)
            n2 = layer_norm(f"{bp}/norm2", f"{name}.norm2", dim)
            assert n2.output[0] in node(f"{bp}/Add_1", "Add").input
            depth += 1
        stages.append({"depth": depth, "dim": dim, "heads": heads, "window": window, "resolution": res,
                       "shifts": shifts, "downsample": down, "mlp_hidden": hidden})
        s += 1
    assert stages, "no stages found"
    layer_norm("/core_model/norm", "norm", dim)
    pool = node("/core_model/head/global_pool/pool/ReduceMean", "ReduceMean")
    assert list(attrs(pool)["axes"]) == [1, 2], attrs(pool)
    fc = node("/core_model/head/fc/Gemm", "Gemm")
    fa = attrs(fc)
    assert fa.get("transB", 0) == 1 and fa.get("alpha", 1.0) == 1.0 and fa.get("beta", 1.0) == 1.0, fa
    hw, hb = value(fc.input[1]), value(fc.input[2])
    classes = int(hw.shape[0])
    assert hw.shape == (classes, dim) and hb.shape == (classes,)
    put("head.w", hw.T)  # [dim, classes]
    put("head.b", hb)
    assert node("/final_act/Sigmoid", "Sigmoid").input[0] == fc.output[0]
    assert len(eps_seen) == 1, eps_seen

    with open(tags_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = [r["name"] for r in rows]
    cats = [int(r["category"]) for r in rows]
    assert len(names) == classes, (len(names), classes)
    assert set(cats) <= {0, 4, 9}, set(cats)

    os.makedirs(out_dir, exist_ok=True)
    arch = {
        "arch": "swinv2_wd14",
        "source": "deepghs/wd14_tagger_with_embeddings SmilingWolf/wd-swinv2-tagger-v3/model.onnx",
        "img_size": img,
        "input": {"order": "BGR", "scale": scale, "offset": offset},
        "patch": patch,
        "embed_dim": embed,
        "eps": eps_seen.pop(),
        "stages": stages,
        "features": dim,
        "classes": classes,
        "tags": {"names": names, "categories": cats},
    }
    save_file(tensors, os.path.join(out_dir, stem + ".safetensors"))
    with open(os.path.join(out_dir, stem + ".json"), "wb") as f:
        f.write(json.dumps(arch, ensure_ascii=False).encode("utf-8"))
    total = sum(t.size for t in tensors.values())
    print(f"{stem}: {len(tensors)} tensors, {total * 4 / 1e6:.1f} MB, stages "
          + ", ".join(f"{st['depth']}x{st['dim']}/{st['heads']}h shifts {st['shifts']}" for st in stages))


if __name__ == "__main__":
    main()
