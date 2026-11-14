# One-time converter for the "pose" task of /v1/vision/detect: Hugging Face RT-DETR (rtdetr_r50vd_coco_o365) and
# ViTPose (vitpose_base_simple) folders -> one F32 safetensors + JSON each, read by the native CPU runtime.
# RT-DETR: every conv + BatchNorm pair is folded into conv weight + bias (eval mode: y = (x - mean) / sqrt(var + eps)
# * w + b), int64 buffers (num_batches_tracked) and the training-only denoising embedding are dropped, the "model."
# prefix goes. ViTPose: the tensors are copied as they are (all F32). The architecture values come from config.json
# and are checked against the tensor shapes. Only numpy and safetensors are needed (no torch).
# Written with AI assistance (Claude Code).
#
#   python -I convert_pose.py rtdetr  <hf_folder> <out_dir>    -> rtdetr_r50vd_coco_o365.{safetensors,json}
#   python -I convert_pose.py vitpose <hf_folder> <out_dir>    -> vitpose_base_simple.{safetensors,json}
import json
import os
import sys
from collections import OrderedDict

import numpy as np
from safetensors.numpy import load_file, save_file


def fold(tensors, conv_w, norm, eps):
    w = tensors[conv_w].astype(np.float64)
    g = tensors[norm + ".weight"].astype(np.float64)
    b = tensors[norm + ".bias"].astype(np.float64)
    m = tensors[norm + ".running_mean"].astype(np.float64)
    v = tensors[norm + ".running_var"].astype(np.float64)
    s = g / np.sqrt(v + eps)
    return (w * s[:, None, None, None]).astype(np.float32), (b - m * s).astype(np.float32)


def convert_rtdetr(src, dst):
    cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    t = load_file(os.path.join(src, "model.safetensors"))
    eps_backbone = 1e-5  # RTDetrFrozenBatchNorm2d
    eps = float(cfg.get("batch_norm_eps", 1e-5))
    out = OrderedDict()
    used = set()
    names = sorted(t)
    # conv + norm pairs
    for k in names:
        pairs = []
        if k.endswith(".convolution.weight"):
            base = k[: -len(".convolution.weight")]
            pairs.append((k, base + ".normalization", base, eps_backbone))
        elif k.endswith(".conv.weight"):
            base = k[: -len(".conv.weight")]
            pairs.append((k, base + ".norm", base, eps))
        elif (k.startswith("model.encoder_input_proj.") or k.startswith("model.decoder_input_proj.")) and k.endswith(".0.weight"):
            base = k[: -len(".0.weight")]
            pairs.append((k, base + ".1", base, eps))
        for conv_w, norm, base, e in pairs:
            assert norm + ".running_var" in t, (k, norm)
            w, b = fold(t, conv_w, norm, e)
            name = base[len("model."):]
            out[name + ".weight"] = w
            out[name + ".bias"] = b
            used.add(conv_w)
            for p in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                used.add(norm + "." + p)
    for k in names:
        if k in used or k.startswith("model.denoising_class_embed"):
            continue
        assert t[k].dtype == np.float32, (k, t[k].dtype)
        name = k[len("model."):] if k.startswith("model.") else k
        out[name] = np.ascontiguousarray(t[k])
    # shape checks of the parts the runtime relies on
    d = cfg["d_model"]
    assert out["decoder.layers.0.encoder_attn.sampling_offsets.weight"].shape == (
        cfg["decoder_attention_heads"] * cfg["num_feature_levels"] * cfg["decoder_n_points"] * 2, d)
    assert out["enc_score_head.weight"].shape == (len(cfg["id2label"]), d)
    arch = OrderedDict(
        model="rtdetr_r50vd_coco_o365", source=os.path.abspath(src),
        backbone={"depths": [3, 4, 6, 3], "hidden_sizes": [256, 512, 1024, 2048], "embedding_size": 64,
                  "out_stages": [1, 2, 3]},
        d_model=d, encoder_layers=cfg["encoder_layers"], encoder_heads=cfg["encoder_attention_heads"],
        encoder_ffn=cfg["encoder_ffn_dim"], encode_proj_layers=cfg["encode_proj_layers"],
        decoder_layers=cfg["decoder_layers"], decoder_heads=cfg["decoder_attention_heads"],
        decoder_ffn=cfg["decoder_ffn_dim"], decoder_points=cfg["decoder_n_points"],
        feature_levels=cfg["num_feature_levels"], num_queries=cfg["num_queries"],
        layer_norm_eps=cfg["layer_norm_eps"], pos_temperature=cfg["positional_encoding_temperature"],
        labels=[cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))],
        input={"size": [640, 640], "rescale": 1.0 / 255.0, "resample": "bilinear antialias"})
    write(dst, "rtdetr_r50vd_coco_o365", out, arch)


def convert_vitpose(src, dst):
    cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    pre = json.load(open(os.path.join(src, "preprocessor_config.json"), encoding="utf-8"))
    t = load_file(os.path.join(src, "model.safetensors"))
    for k, v in t.items():
        assert v.dtype == np.float32, k
    assert cfg.get("use_simple_decoder", True)
    layers = len({k.split(".")[3] for k in t if k.startswith("backbone.encoder.layer.")})
    arch = OrderedDict(
        model="vitpose_base_simple", source=os.path.abspath(src), layers=layers, hidden=768, heads=12,
        patch=16, patch_pad=2, layer_norm_eps=1e-12, scale_factor=cfg["scale_factor"],
        keypoints=[cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))],
        input={"height": pre["size"]["height"], "width": pre["size"]["width"], "mean": pre["image_mean"],
               "std": pre["image_std"], "normalize_factor": pre["normalize_factor"]})
    write(dst, "vitpose_base_simple", OrderedDict((k, np.ascontiguousarray(t[k])) for k in sorted(t)), arch)


def write(dst, stem, tensors, arch):
    os.makedirs(dst, exist_ok=True)
    save_file(dict(tensors), os.path.join(dst, stem + ".safetensors"), metadata={"format": "gufo-pose", "arch": stem})
    with open(os.path.join(dst, stem + ".json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(arch, f, indent=1)
    print(stem, len(tensors), "tensors")


if __name__ == "__main__":
    kind, src, dst = sys.argv[1:4]
    {"rtdetr": convert_rtdetr, "vitpose": convert_vitpose}[kind](src, dst)
