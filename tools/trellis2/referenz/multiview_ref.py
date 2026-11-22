#!/usr/bin/env python3
"""Reference numbers for tests/models/trellis2/flow_sampler_test.cpp and cascade_test.cpp.

Re-implements, in numpy float64 and without any upstream code, the semantics of
  * FlowEulerGuidanceIntervalSampler (rescaled times, CFG g*pos + (1-g)*neg inside the
    interval, x0 rescale with the unbiased std over all elements, sigma_min 1e-5),
  * microsoft/TRELLIS.2 PR #104 inject_sampler_multi_image: "stochastic" (view = call
    counter % views, one counter per flow) and "multidiffusion" (mean of the conditional
    predictions, then CFG / rescale with the mean as the positive prediction),
  * Trellis2ImageTo3DPipeline.sample_shape_slat_cascade coordinate quantisation
    ((c + 0.5) / 512 * (hr // 16)).int(), unique, hr -= 128 while tokens >= max and hr > 1024),
on a toy model, and prints C++ initialisers. Run: python -I multiview_ref.py
AI-assisted (Claude Code), 2026-11-22.
"""
import numpy as np

SIGMA_MIN = 1e-5


def flow_times(steps, r):
    s = 1.0 - np.arange(steps + 1) / steps
    return r * s / (1.0 + (r - 1.0) * s)


def toy_model(inp, cond, t_model, channels):
    """v[i, c] = 0.5 in[i, c] - 0.25 in[i, (c+1) % W] + cond[(i*C + c) % L] * (1 + t/1000)."""
    n, w = inp.shape
    v = np.zeros((n, channels))
    L = len(cond)
    for i in range(n):
        for c in range(channels):
            v[i, c] = 0.5 * inp[i, c] - 0.25 * inp[i, (c + 1) % w] + cond[(i * channels + c) % L] * (1.0 + t_model / 1000.0)
    return v


def cfg(pos, neg, x_t, t, g, rescale):
    pred = g * pos + (1 - g) * neg
    if rescale > 0:
        a, b = 1 - SIGMA_MIN, SIGMA_MIN + (1 - SIGMA_MIN) * t
        x0_pos = a * x_t - b * pos
        x0_cfg = a * x_t - b * pred
        ratio = x0_pos.std(ddof=1) / x0_cfg.std(ddof=1)
        x0 = rescale * x0_cfg * ratio + (1 - rescale) * x0_cfg
        pred = (a * x_t - x0) / b
    return pred


def sample(x, extra, conds, zero, steps, g, rescale, lo, hi, rt, mode):
    x = x.copy()
    channels = x.shape[1]
    times = flow_times(steps, rt)
    counter = 0
    for i in range(steps):
        t, tp = times[i], times[i + 1]
        inp = x if extra is None else np.concatenate([x, extra], axis=1)
        if len(conds) == 1 or mode == "stochastic":
            pos = toy_model(inp, conds[counter % len(conds)], 1000 * t, channels)
            counter += 1
        else:
            preds = [toy_model(inp, c, 1000 * t, channels) for c in conds]
            pos = sum(preds) / len(preds)
        if g != 1.0 and lo <= t <= hi:
            neg = toy_model(inp, zero, 1000 * t, channels)
            pos = cfg(pos, neg, x, t, g, rescale)
        x = x - (t - tp) * pos
    return x


def quantise(fine, hr, lr=512):
    q = ((fine.astype(np.float32) + 0.5) / lr * (hr // 16)).astype(np.int32)
    return np.unique(q, axis=0)


def cascade(fine, target, max_tokens):
    hr = target
    while True:
        c = quantise(fine, hr)
        if len(c) < max_tokens or hr == 1024:
            return c, hr
        hr -= 128


def fmt(a):
    return "{" + ", ".join(f"{v:.9g}f" for v in np.asarray(a).reshape(-1)) + "}"


def main():
    x0 = np.array([[0.3, -1.2], [0.8, 0.1], [-0.5, 0.7]])
    extra = np.array([[0.2], [-0.4], [0.9]])
    conds = [np.array([0.5, -0.3, 0.8, 0.1]), np.array([-0.6, 0.4, 0.2, -0.9]), np.array([0.05, 0.7, -0.2, 0.3])]
    zero = np.zeros(4)
    args = dict(steps=5, g=7.5, rescale=0.5, lo=0.6, hi=1.0, rt=3.0)
    one = sample(x0, None, conds[:1], zero, mode="stochastic", **args)
    sto = sample(x0, None, conds, zero, mode="stochastic", **args)
    mul = sample(x0, None, conds, zero, mode="multidiffusion", **args)
    tex = sample(x0, extra, conds[:2], zero, mode="multidiffusion", steps=4, g=1.0, rescale=0.0, lo=0.6, hi=0.9, rt=3.0)
    print("single:", fmt(one))
    print("stochastic:", fmt(sto))
    print("multidiffusion:", fmt(mul))
    print("tex_multidiffusion:", fmt(tex))

    # deterministic voxel list (same formula in cascade_test.cpp)
    i = np.arange(6000)
    fine = np.stack([(i * 37) % 512, (i * 101 + 7) % 512, (i * 13 + i // 7) % 512], axis=1)
    for target in (1024, 1536):
        c, hr = cascade(fine, target, 49152)
        print(f"cascade {target}: hr={hr} tokens={len(c)} first={c[0].tolist()} last={c[-1].tolist()} sum={int(c.sum())}")
    c, hr = cascade(fine, 1536, 3500)
    print(f"cascade 1536 cap 3500: hr={hr} tokens={len(c)} sum={int(c.sum())}")
    for hr in (1408, 1280, 1152):
        print(f"  quantise {hr}: tokens={len(quantise(fine, hr))}")
    c, hr = cascade(fine, 1024, 10)
    print(f"cascade 1024 cap 10: hr={hr} tokens={len(c)}")


if __name__ == "__main__":
    main()
