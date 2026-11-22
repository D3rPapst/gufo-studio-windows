# Inkling 2 Mini on gufo_windows (draft)

*Drafted with AI assistance (Claude Code); reviewed by a human before any publication.*

This folder is [headbouyJB/inkling2-gufo](https://github.com/headbouyJB/inkling2-gufo) (MIT, commit 8bb6cd5e, see
`UPSTREAM.txt`) ported to the Windows fork of gufo, plus a few additions. Upstream's `README.md` describes the
extension itself; this file only lists what differs here.

## Windows port

- Builds with the fork's extension API 0.3.10 (upstream asks for 0.3.20; the only feature it used from the newer
  API, shared transfer-stream leases, is left out).
- Vision `mmap` → Windows file mapping, `fseeko` → `_fseeki64`.
- Weight upload drops each copied chunk of the mapped GGUF from the working set (`VirtualUnlock`), so loading
  does not grow the process to the model size.

## Thinking effort

Inkling's chat template writes a "Thinking effort level" between 0 and 0.99 into the prompt. Per request:

| Field | Values | Notes |
|---|---|---|
| `reasoning_effort` (top level, OpenAI style) | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or a number 0–0.99 | `none` turns thinking off (the answer block is opened directly). Numbers map to the nearest named level of the template's effort map (0 → none, < 0.15 minimal, < 0.45 low, < 0.8 medium, < 0.945 high, else xhigh). |
| `chat_template_kwargs.reasoning_effort` | the same | If both fields are given they must agree, otherwise the request is rejected (HTTP 400). |
| `thinking: {"type": "disabled"}`, `chat_template_kwargs.enable_thinking: false` | | Same as `none`. |

Server default: gufo's generic `--reasoning-effort LEVEL` (`auto`, `minimal` … `max`); a request's own value
always wins. Without either, the template default is 0.9 (`high`). In a router preset: `reasoning-effort = medium`.

## DSpark speculative decoding (opt-in)

```
gufo serve llm -m Inkling-2-Mini-UD-Q4_K_XL.gguf --speculative dspark \
    --dspark-model dspark-Inkling-2-Mini-Q8_0.gguf --sessions 1
```

- Drafter on the GPU (about 0.8 GB); target verifies up to 6 tokens per step in one batched decode pass.
- Lossless: only greedy requests speculate (temperature 0, no penalties, logprobs, grammar or reasoning-off
  masks); every accepted draft token is the target's own greedy choice. Sampled requests decode exactly as
  without DSpark (no rejection sampling yet).
- An adaptive policy decides per request: on code with high acceptance it drafts 3 tokens per step (measured
  +40–50 % tokens/s with the bench tool), on free text it falls back to plain decoding.
- One session only for now (`--sessions 1`).
