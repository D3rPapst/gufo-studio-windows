#!/usr/bin/env python3
"""Capture a MiniMax-H3-Image VAE *decoder* oracle for the gufo port.

The reference side is diffusers' ``AutoencoderKLMiniMaxH3Image`` (``_decode`` with
the default tiling: 256-pixel tiles, at least 64 pixels of overlap, linear blend)
and ``MiniMaxH3ImageViTDecoder2d``, transcribed here module by module so that the
script needs only torch + safetensors (diffusers 0.42 does not ship the image VAE
yet). The released *native* checkpoint ``vae/diffusion_pytorch_model.safetensors``
is converted on the fly exactly as diffusers' ``scripts/convert_minimax_h3_to_diffusers.py``
(``convert_video_vae_key``) does:

* ``attn.to_qkv`` is per-head interleaved ``[head0: q k v, head1: ...]`` and is split
  into ``to_q`` / ``to_k`` / ``to_v``,
* ``ff.w1`` stores ``[gate; up]``; diffusers' SwiGLU reads ``[up; gate]``,
* ``x_embedder`` is ``proj_in``.

So the oracle shares no code or layout assumption with the C++ port
(``src/models/minimax_h3/image_vae.*``).

``--blocks N`` keeps only the first N transformer blocks (a truncated but
otherwise identical decoder) so the C++ CPU reference can check it in seconds;
``--blocks 24`` is the full decoder.

Writes into ``--output``: ``latent.f32`` (normalized latent [24][h][w], what the
DiT emits), ``rgb.f32`` (ImageNet-normalized interleaved RGB [H][W][3] before the
pipeline's de-normalization) and ``meta.json``.

Usage (CPU only, never the GPU):

  python tools/h3/gufo-h3-image-vae-oracle.py --model-root <MiniMax-H3-Image> \
      --output <dir> [--latent-height 20 --latent-width 24 --blocks 2 --seed 7]

AI-assisted (Claude Code).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

HIDDEN = 1024
HEADS = 16
HEAD_DIM = 64
FFN = 4096
REGISTERS = 4
PATCH = 16
EPS = 1e-5


def rotary(height: int, width: int, theta: float = 100.0, rope_dim: int = 32):
    """MiniMaxH3ImageRotaryPosEmbed(dim=32, num_axes=2) on the decoder grid."""
    inv_freq = 1.0 / theta ** torch.arange(0, 1, 2 * 2 / rope_dim, dtype=torch.float32)
    grids = [2.0 * (torch.arange(0.5, size, dtype=torch.float32) / size) - 1.0 for size in (height, width)]
    position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 1)[None]
    suffix = position_ids.new_zeros((1, REGISTERS + 1, 2))
    position_ids = torch.cat([position_ids, suffix], dim=1)
    angles = 2.0 * math.pi * position_ids[:, :, :, None] * inv_freq[None, None, None, :]
    angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
    return angles.cos(), angles.sin()


def rms(x: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    x = x.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return y if weight is None else y * weight


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    rot, keep = x[..., :rotary_dim], x[..., rotary_dim:]
    first, second = rot.chunk(2, dim=-1)
    rotated = torch.cat([-second, first], dim=-1)
    return torch.cat([rot * cos + rotated * sin, keep], dim=-1)


class Weights:
    def __init__(self, path: Path):
        self.handle = safe_open(str(path), framework="pt")

    def get(self, name: str) -> torch.Tensor:
        return self.handle.get_tensor(name).float()

    def block(self, index: int) -> dict[str, torch.Tensor]:
        prefix = f"decoder.transformer_blocks.{index}."
        qkv_w = self.get(prefix + "attn.to_qkv.weight")
        qkv_b = self.get(prefix + "attn.to_qkv.bias")

        def split(tensor: torch.Tensor):
            grouped = tensor.reshape(HEADS, 3 * HEAD_DIM, *tensor.shape[1:])
            q, k, v = grouped.split(HEAD_DIM, dim=1)
            return [t.reshape(HEADS * HEAD_DIM, *tensor.shape[1:]) for t in (q, k, v)]

        (qw, kw, vw), (qb, kb, vb) = split(qkv_w), split(qkv_b)
        gate_w, up_w = self.get(prefix + "ff.w1.weight").chunk(2, dim=0)
        gate_b, up_b = self.get(prefix + "ff.w1.bias").chunk(2, dim=0)
        return {
            "norm1": self.get(prefix + "norm1.weight"),
            "norm2": self.get(prefix + "norm2.weight"),
            "to_q": (qw, qb), "to_k": (kw, kb), "to_v": (vw, vb),
            "to_out": (self.get(prefix + "attn.to_out.weight"), self.get(prefix + "attn.to_out.bias")),
            "scale1": self.get(prefix + "scale1"),
            "scale2": self.get(prefix + "scale2"),
            # diffusers SwiGLU: proj = [up; gate], hidden * silu(gate)
            "ff_proj": (torch.cat([up_w, gate_w]), torch.cat([up_b, gate_b])),
            "ff_out": (self.get(prefix + "ff.w2.weight"), self.get(prefix + "ff.w2.bias")),
        }


def decoder_forward(z: torch.Tensor, weights: Weights, blocks: list[dict]) -> torch.Tensor:
    """post_quant_conv + MiniMaxH3ImageViTDecoder2d.forward for one tile, z [1, 24, h, w]."""
    z = F.conv2d(z, weights.get("post_quant_conv.weight"), weights.get("post_quant_conv.bias"))
    batch, channels, height, width = z.shape
    x = z.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
    x = F.linear(x, weights.get("decoder.x_embedder.weight"), weights.get("decoder.x_embedder.bias"))
    patches = x.shape[1]
    registers = weights.get("decoder.register_tokens").expand(batch, -1, -1)
    x = torch.cat([x, registers, torch.zeros_like(x[:, :1, :])], dim=1)
    cos, sin = rotary(height, width)
    for block in blocks:
        h = rms(x, block["norm1"])
        q = F.linear(h, *block["to_q"]).unflatten(2, (HEADS, -1))
        k = F.linear(h, *block["to_k"]).unflatten(2, (HEADS, -1))
        v = F.linear(h, *block["to_v"]).unflatten(2, (HEADS, -1))
        q = apply_rope(rms(q, None), cos, sin)
        k = apply_rope(rms(k, None), cos, sin)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        a = a.transpose(1, 2).flatten(2, 3)
        x = x + F.linear(a, *block["to_out"]) * block["scale1"]
        h = rms(x, block["norm2"])
        hidden, gate = F.linear(h, *block["ff_proj"]).chunk(2, dim=-1)
        x = x + F.linear(hidden * F.silu(gate), *block["ff_out"]) * block["scale2"]
    x = F.layer_norm(x, (HIDDEN,), weights.get("decoder.norm_out.weight"), weights.get("decoder.norm_out.bias"), EPS)
    x = F.linear(x, weights.get("decoder.proj_out.weight"), weights.get("decoder.proj_out.bias"))[:, :patches]
    x = x.view(batch, height, width, 3, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.reshape(batch, 3, height * PATCH, width * PATCH)


def split_tiles(length: int, tile: int = 256, overlap: int = 64, ratio: int = 16):
    if tile >= length:
        return [0], [length], []
    count = math.ceil(length / tile)
    while tile * count - overlap * (count - 1) - length < 0:
        count += 1
    overlaps = [overlap] * (count - 1)
    remaining = tile * count - sum(overlaps) - length
    for i in range(remaining // ratio):
        overlaps[i % (count - 1)] += ratio
    starts = [0]
    for i in range(count - 1):
        starts.append(starts[-1] + tile - overlaps[i])
    return starts, [tile] * count, overlaps


def blend(a: torch.Tensor, b: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
    extent = min(a.shape[dim], b.shape[dim], extent)
    positions = torch.arange(extent, dtype=b.dtype)
    shape = [1] * a.ndim
    shape[dim] = extent
    weight_a = (1 - positions / extent).view(shape)
    weight_b = (positions / extent).view(shape)
    slice_a = [slice(None)] * a.ndim
    slice_a[dim] = slice(-extent, None)
    slice_b = [slice(None)] * b.ndim
    slice_b[dim] = slice(0, extent)
    blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b
    if extent == b.shape[dim]:
        return blended
    rest = [slice(None)] * b.ndim
    rest[dim] = slice(extent, None)
    return torch.cat([blended, b[tuple(rest)]], dim=dim)


def decode(z: torch.Tensor, weights: Weights, blocks: list[dict]) -> torch.Tensor:
    """AutoencoderKLMiniMaxH3Image._decode with the release's default tiling."""
    height, width = z.shape[-2] * PATCH, z.shape[-1] * PATCH
    ys, yl, yo = split_tiles(height)
    xs, xl, xo = split_tiles(width)
    rows = []
    for i_pos, i_len in zip(ys, yl):
        row = []
        for j_pos, j_len in zip(xs, xl):
            tile = z[..., i_pos // PATCH: (i_pos + i_len) // PATCH, j_pos // PATCH: (j_pos + j_len) // PATCH]
            row.append(decoder_forward(tile, weights, blocks))
        rows.append(row)
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend(rows[i - 1][j], tile, yo[i - 1], dim=-2)
            if j > 0:
                tile = blend(row[j - 1], tile, xo[j - 1], dim=-1)
            if i < len(rows) - 1:
                tile = tile[..., : -yo[i], :]
            if j < len(row) - 1:
                tile = tile[..., :, : -xo[j]]
            result_row.append(tile)
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-height", type=int, default=20)
    parser.add_argument("--latent-width", type=int, default=24)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    config = json.loads((args.model_root / "vae" / "config.json").read_text())
    mean = torch.tensor(config["latents_mean"], dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor(config["latents_std"], dtype=torch.float32).view(1, -1, 1, 1)
    weights = Weights(args.model_root / "vae" / "diffusion_pytorch_model.safetensors")
    blocks = [weights.block(index) for index in range(args.blocks)]

    rng = np.random.default_rng(args.seed)
    latent = rng.standard_normal((24, args.latent_height, args.latent_width)).astype(np.float32)
    z = torch.from_numpy(latent)[None] * std + mean
    image = decode(z, weights, blocks)[0]  # [3, H, W], ImageNet-normalized
    rgb = image.permute(1, 2, 0).contiguous().numpy().astype(np.float32)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "latent.f32").write_bytes(latent.tobytes())
    (args.output / "rgb.f32").write_bytes(rgb.tobytes())
    meta = {
        "schema": "gufo.minimax-h3-image-vae-golden.v1",
        "reference": "diffusers AutoencoderKLMiniMaxH3Image._decode (transcribed), native weights converted as "
                     "convert_minimax_h3_to_diffusers.py convert_video_vae_key",
        "blocks": args.blocks,
        "seed": args.seed,
        "latent": [24, args.latent_height, args.latent_width],
        "rgb": list(rgb.shape),
        "files": {
            "latent.f32": sha256(latent.tobytes()),
            "rgb.f32": sha256(rgb.tobytes()),
        },
        "rgb_stats": {"min": float(rgb.min()), "max": float(rgb.max()), "mean": float(rgb.mean())},
    }
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
