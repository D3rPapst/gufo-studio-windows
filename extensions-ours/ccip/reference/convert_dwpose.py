# One-time converter for "pose_wholebody": DWPose dw-ll_ucoco_384.onnx (RTMPose-l, COCO-WholeBody 133 points,
# SimCC head) -> dwpose_wholebody.safetensors (every initializer and constant as F32) + dwpose_wholebody.json
# (the node list in graph order). The shape arithmetic of the head (Shape -> Slice -> Concat -> Reshape) is folded
# for the fixed input 1 x 3 x 384 x 288 into one static Reshape, so the C++ runtime executes a small, fixed op set:
# Conv, Sigmoid, Mul, Add, Div, Concat, GlobalAveragePool, HardSigmoid, MaxPool, Relu, ReduceSum, Sqrt, Clip,
# Split, Squeeze, Unsqueeze, Transpose, MatMul, Reshape, Identity. Any other op fails the conversion.
# Only onnx and numpy are needed. Written with AI assistance (Claude Code).
#
#   python -I convert_dwpose.py <dw-ll_ucoco_384.onnx> <out_dir>
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference
from safetensors.numpy import save_file

SUPPORTED = {"Conv", "Sigmoid", "Mul", "Add", "Div", "Concat", "GlobalAveragePool", "HardSigmoid", "MaxPool", "Relu",
             "ReduceSum", "Sqrt", "Clip", "Split", "Squeeze", "Unsqueeze", "Transpose", "MatMul", "Reshape", "Identity"}


def main(src, dst):
    model = onnx.load(src)
    g = model.graph
    inp = g.input[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim[1:]]
    assert dims == [3, 384, 288], dims
    # Fix the batch to 1 and infer every shape.
    inp.type.tensor_type.shape.dim[0].dim_value = 1
    model = shape_inference.infer_shapes(model)
    g = model.graph
    shapes = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        t = vi.type.tensor_type
        if t.HasField("shape"):
            shapes[vi.name] = [d.dim_value for d in t.shape.dim]
    tensors = OrderedDict()
    for t in g.initializer:
        tensors[t.name] = numpy_helper.to_array(t)
    nodes = []
    folded = set()
    for n in g.node:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        if n.op_type == "Constant":
            tensors[n.output[0]] = numpy_helper.to_array(attrs["value"])
            continue
        if n.op_type in ("Shape", "Slice") or (n.op_type == "Concat" and attrs.get("axis") == 0):
            folded.update(n.output)
            continue
        if n.op_type == "Reshape":
            # The folded target is Shape(x)[:2] ++ [-1] (flatten from axis 2), x of known shape.
            src_shape = shapes.get(n.input[0])
            assert src_shape and all(v > 0 for v in src_shape), (n.name, src_shape)
            target = shapes.get(n.output[0]) or src_shape[:2] + [int(np.prod(src_shape[2:]))]
            name = n.name + "_shape"
            tensors[name] = np.array(target, dtype=np.int64)
            ins = [n.input[0], name]
        else:
            ins = list(n.input)
        if n.op_type not in SUPPORTED:
            raise SystemExit(f"unsupported op {n.op_type} ({n.name})")
        for x in ins:
            assert x == "" or x not in folded, (n.name, x)
        a = {}
        for k, v in attrs.items():
            if isinstance(v, (list, tuple)):
                a[k] = [int(x) if isinstance(x, (int, np.integer)) else float(x) for x in v]
            elif isinstance(v, (int, np.integer)):
                a[k] = int(v)
            elif isinstance(v, float):
                a[k] = float(v)
            elif isinstance(v, bytes):
                a[k] = v.decode()
        nodes.append({"op": n.op_type, "name": n.name, "in": ins, "out": list(n.output), "attrs": a})
    used = {x for n in nodes for x in n["in"] if x}
    out_tensors = OrderedDict()
    for k, v in tensors.items():
        if k in used:
            out_tensors[k] = np.ascontiguousarray(v.astype(np.float32))
    arch = OrderedDict(model="dw-ll_ucoco_384", source=os.path.abspath(src), input={"name": g.input[0].name,
                       "shape": [1, 3, 384, 288], "mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375]},
                       outputs=[o.name for o in g.output], split_ratio=2.0, keypoints=133,
                       shapes={k: v for k, v in shapes.items() if k in [o.name for o in g.output]}, nodes=nodes)
    os.makedirs(dst, exist_ok=True)
    save_file(dict(out_tensors), os.path.join(dst, "dwpose_wholebody.safetensors"),
              metadata={"format": "gufo-pose", "arch": "dwpose_wholebody"})
    with open(os.path.join(dst, "dwpose_wholebody.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(arch, f, indent=1)
    print("dwpose", len(nodes), "nodes", len(out_tensors), "tensors", sorted({n["op"] for n in nodes}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
