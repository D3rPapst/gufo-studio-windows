# YuE2 port (internal)

Native port of YuE2-3B and YuE2-Vae (m-a-p, release 0.1.6 of `yue2_infer`) into
gufo-studio-windows. User page: [docs/models/yue2.md](../models/yue2.md).

**Licence of the weights:** CC BY-NC 4.0 plus the individual-creator permission
in `MODEL_LICENSE` of the GitHub repository (commit `cbb3b060`, 16 Sep 2026).
Weights non-commercial; outputs may be monetized by individuals and creators
according to `MODEL_LICENSE`, companies need their own licence. Do not mirror
the weights on Hugging Face. The gufo code is our own and carries no weights.

## Layout

| Part | File | Reuse |
|---|---|---|
| Tokenizer | `src/models/yue2/tokenizer.cpp` | own tiktoken-rank BPE (ICU regex, NFC) |
| Prompts, chunks, sampler | `src/models/yue2/protocol.cpp` | CPU, `yue2/protocol.py` + `sampling.py` |
| AR + NAR model | `src/models/yue2/model.hip` | Qwen3 stack of ACE-Step (`ace_step/qwen3_stack.hpp`) |
| Pipeline | `src/models/yue2/pipeline.cpp` | ACE-Step `MusicNoise`, Oobleck decoder |
| VAE decoder | `src/models/ace_step/vae.hip` | `OobleckLayout::YuE2()` (6 blocks, stable-audio-tools names) |
| Server | `src/cli/serve/media/yue2_api.cpp` | `gufo serve music --yue2-vae DIR` |
| Checks | `src/models/yue2/check_main.cpp` (`gufo-yue2-check`), `tests/models/yue2/protocol_test.cpp` | |
| Oracle | `tools/yue2/oracle.py` | official modules, FP32 CPU |

`ace_step/qwen3_stack.hpp` is the layer stack that used to sit in the anonymous
namespace of `ace_step/model.hip`, moved unchanged. The Oobleck decoder takes its
block count, strides and tensor names from an `OobleckLayout`; the ACE-Step layout
(`AceStep()`, five blocks, Diffusers names) is the default and decodes as before.
An odd stride (YuE2: 5) drops the last sample of its block, so a YuE2 decode of
`T` frames has `1920 T - 64` samples, as the reference.

## Numerics

- AR: BF16 weights and GEMM operands, FP32 residual stream (as the ACE-Step
  planner). Logits of the phase slice are an FP32 matrix-vector product
  (ABC: ids `0..151848`, semantic: `151852..184620`; MUSIC_END directly
  precedes the codes).
- NAR input `vae2llm(x) + time_embedder(t) + latent_pos_embed` and the
  `llm2vae` head in FP32; the time MLP in FP32 with BF16 rounding of its two
  outputs (`LaunchMatVec`). The ODE state stays FP32 on the host (the
  reference pipeline keeps it in BF16 on CUDA).
- NAR attention: the frames see the chunk's AR prefix keys / values (prefill
  with the AR weights) and each other, at RoPE positions after the prefix;
  the KV cache rows after the prefix are rewritten by every velocity call.
- Sampling: masks, `min_tokens`, windowed penalty, temperature, top_k, top_p
  as `sampling.distribution`; the draw uses gufo's `std::mt19937_64` (not
  torch's multinomial), the acoustic noise `ace_step::MusicNoise` (not torch
  `randn`), so a seed gives a different song than the reference.

## Oracle

`tools/yue2/oracle.py` (venv with `yue2_infer` 0.1.6, torch 2.10.0 CPU,
transformers 4.57.6) dumps per stage: token ids and prefixes, ABC and semantic
prefix logits and 64 greedy tokens each, the NAR velocity at two fixed
(state, t), the 32-step latents of 64 frames from the dumped noise, and VAE
decodes. One stage per process; peak below 5 GB host RAM (unbuffered reads,
only the needed weight set cached as BF16).

```text
gufo-yue2-check --oracle DIR --model .../YuE2-3B --vae .../YuE2-Vae --stages tok,vae,ar,nar
```

Results: see "Measured" in the user page and the release notes of the branch.

## Not done

- Live preview (`/preview`) and `X-Gufo-Preview-*`: no clean estimate yet.
- Q8 AR weights, fused attention for long NAR chunks (speed).
- SheetSage2 covers (CC BY-NC without the creator permission).
- Router page integration and the music page's model switch.
