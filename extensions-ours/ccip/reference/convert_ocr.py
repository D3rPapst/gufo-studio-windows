# One-time converter for the /v1/vision/detect "ocr" task: the PaddleOCR PP-OCRv4 ONNX exports that
# dghs-imgutils uses (deepghs/paddleocr det/ch_PP-OCRv4_det, rec/ch_PP-OCRv4_rec) -> safetensors + JSON.
# paddle2onnx keeps every weight in a Constant node; the float constants go to <stem>.safetensors (F32,
# by output name), the int64 constants (shapes, axes, slice bounds) and the node list (op type, inputs,
# outputs, attributes) to <stem>.json. gufo_ccip's OcrNet runs that node list natively (cpu/ocr_net.cpp);
# the script refuses ops the runtime does not implement, so nothing can be dropped silently.
# Written with AI assistance (Claude Code).
#
# Usage (venv with onnx, numpy, safetensors; run with python -I):
#   python -I convert_ocr.py <det/ch_PP-OCRv4_det/model.onnx> <out_dir> ch_PP-OCRv4_det
#   python -I convert_ocr.py <rec/ch_PP-OCRv4_rec/model.onnx> <out_dir> ch_PP-OCRv4_rec <dict.txt>
# The recognizer's dict.txt is copied to <stem>.dict.txt (bytes unchanged).
import json
import os
import shutil
import sys
from collections import OrderedDict

import numpy as np
import onnx
from onnx import helper, numpy_helper
from safetensors.numpy import save_file

# Ops implemented by cpu/ocr_net.cpp (with the attributes listed in SUPPORTED_ATTRS).
SUPPORTED = {
    "Add", "AveragePool", "BatchNormalization", "Cast", "Clip", "Concat", "Conv", "ConvTranspose", "Div",
    "GlobalAveragePool", "HardSigmoid", "MatMul", "Mul", "Pow", "ReduceMean", "Relu", "Reshape", "Resize",
    "Shape", "Sigmoid", "Slice", "Softmax", "Sqrt", "Squeeze", "Sub", "Transpose",
}
SUPPORTED_ATTRS = {
    "AveragePool": {"ceil_mode", "count_include_pad", "kernel_shape", "pads", "strides"},
    "BatchNormalization": {"epsilon", "momentum"},
    "Cast": {"to"},
    "Clip": {"min", "max"},
    "Concat": {"axis"},
    "Conv": {"dilations", "group", "kernel_shape", "pads", "strides"},
    "ConvTranspose": {"dilations", "group", "kernel_shape", "pads", "strides"},
    "HardSigmoid": {"alpha", "beta"},
    "ReduceMean": {"axes", "keepdims"},
    "Resize": {"coordinate_transformation_mode", "mode", "nearest_mode"},
    "Softmax": {"axis"},
    "Squeeze": {"axes"},
    "Transpose": {"perm"},
}


def attr_json(op, a):
    v = helper.get_attribute_value(a)
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, (list, tuple)):
        return [int(x) if isinstance(x, (int, np.integer)) else float(x) for x in v]
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    raise SystemExit(f"{op}: unsupported attribute {a.name} of type {type(v)}")


def main():
    if len(sys.argv) not in (4, 5):
        raise SystemExit(__doc__ or "usage: convert_ocr.py <model.onnx> <out_dir> <stem> [<dict.txt>]")
    path, out_dir, stem = sys.argv[1:4]
    dict_path = sys.argv[4] if len(sys.argv) == 5 else None
    model = onnx.load(path)
    g = model.graph
    opset = max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    floats = OrderedDict()
    ints = OrderedDict()
    for t in g.initializer:
        arr = numpy_helper.to_array(t)
        if arr.dtype == np.float32:
            floats[t.name] = np.ascontiguousarray(arr)
        elif arr.dtype in (np.int64, np.int32):
            ints[t.name] = arr.astype(np.int64)
        else:
            raise SystemExit(f"initializer {t.name}: dtype {arr.dtype}")
    nodes = []
    for n in g.node:
        if n.op_type == "Constant":
            assert len(n.attribute) == 1 and n.attribute[0].name == "value", n.name
            arr = numpy_helper.to_array(n.attribute[0].t)
            if arr.dtype == np.float32:
                floats[n.output[0]] = np.ascontiguousarray(arr)
            elif arr.dtype in (np.int64, np.int32):
                ints[n.output[0]] = arr.astype(np.int64)
            else:
                raise SystemExit(f"constant {n.output[0]}: dtype {arr.dtype}")
            continue
        if n.op_type not in SUPPORTED:
            raise SystemExit(f"op {n.op_type} ({n.name}) is not implemented by ocr_net.cpp")
        attrs = OrderedDict()
        for a in n.attribute:
            if a.name not in SUPPORTED_ATTRS.get(n.op_type, set()):
                raise SystemExit(f"{n.op_type} ({n.name}): attribute {a.name} not implemented")
            attrs[a.name] = attr_json(n.op_type, a)
        if n.op_type in ("Conv", "ConvTranspose"):
            assert attrs.get("dilations", [1, 1]) == [1, 1], n.name
        if n.op_type == "Resize":
            assert attrs.get("mode") == "nearest" and attrs.get("nearest_mode", "round_prefer_floor") == "floor"
            assert attrs.get("coordinate_transformation_mode") == "asymmetric"
        if n.op_type == "AveragePool":
            assert attrs.get("ceil_mode", 0) == 0 and not any(attrs.get("pads", [0])), n.name
        if n.op_type == "Cast":
            assert attrs["to"] in (1, 6, 7), n.name
        nodes.append(OrderedDict(op=n.op_type, inputs=list(n.input), outputs=list(n.output), attrs=attrs))
    inputs = [i.name for i in g.input if i.name not in floats and i.name not in ints]
    assert len(inputs) == 1 and len(g.output) == 1, (inputs, len(g.output))
    # Every float constant must be used by a node (no stray tensors) and every node input must exist.
    produced = set(inputs) | set(floats) | set(ints)
    for n in nodes:
        for i in n["inputs"]:
            if i and i not in produced:
                raise SystemExit(f"{n['op']}: input {i} is not produced before use")
        produced.update(n["outputs"])
    arch = OrderedDict(
        model=stem, source=os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path),
        opset=opset, input=inputs[0], output=g.output[0].name,
        int_constants=OrderedDict((k, OrderedDict(shape=list(v.shape), data=[int(x) for x in v.ravel()]))
                                  for k, v in ints.items()),
        nodes=nodes)
    os.makedirs(out_dir, exist_ok=True)
    save_file(dict(floats), os.path.join(out_dir, stem + ".safetensors"),
              metadata={"source": arch["source"], "model": stem})
    data = (json.dumps(arch, indent=None, separators=(",", ":")) + "\n").encode("utf-8")
    with open(os.path.join(out_dir, stem + ".json"), "wb") as f:
        f.write(data)
    if dict_path is not None:
        shutil.copyfile(dict_path, os.path.join(out_dir, stem + ".dict.txt"))
    n_params = sum(v.size for v in floats.values())
    print(f"{stem}: {len(nodes)} nodes, {len(floats)} float tensors ({n_params} values), "
          f"{len(ints)} int constants, opset {opset}")


if __name__ == "__main__":
    main()
