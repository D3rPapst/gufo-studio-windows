# MOSS-Music music understanding

`gufo serve moss-music` runs MOSS-Music-8B-Instruct (OpenMOSS, Apache-2.0) natively:
description, lyrics, structure and questions about a music clip. The API is the one of the
Omni captioner: `POST /v1/chat/completions` with `input_audio` (WAV, base64) and a text
question; streaming, `temperature`, `top_p`, `top_k`, `seed`.

**Not in release 2.1.0.** The worker and its converter are on the fork branch
`feat/moss-music` (merged with 2.1 as `feat/moss-music-21`); the router starts it like any
other `[external/...]` worker. The production preset runs it from a separate build of that branch.

## Files

Two GGUF files: the Whisper-style audio encoder with adapter and DeepStack mergers
(architecture `moss-music`, matrices BF16, vectors F32) and the Qwen3-8B language model as a
standard llama.cpp `qwen3` GGUF in Q8_0 with tokenizer and chat template.
`tools/moss_music/convert_hf_to_gufo.py all` (on the feature branch) writes the audio
GGUF and a plain `Qwen3ForCausalLM` folder from the Hugging Face shards; llama.cpp's
`convert_hf_to_gguf.py --outtype bf16` and `llama-quantize` Q8_0 make the LLM file.

### Converted weights

Both files are published as
[D3rPapst/moss-music-8b-gufo](https://huggingface.co/D3rPapst/moss-music-8b-gufo) (10.4 GB,
Apache-2.0, licence text included; source revision `fce7f830`, regression results on the card).

```powershell
hf download D3rPapst/moss-music-8b-gufo --local-dir D:\llm-modelle\gufo\modelle\OpenMOSS-Team\moss-music-gufo
```

| File | Size | Content |
|---|---:|---|
| `moss-music-8b.audio.gguf` | 1.72 GB | audio encoder + adapters, BF16/F32, arch `moss-music` |
| `moss-music-8b.llm-Q8_0.gguf` | 8.71 GB | Qwen3-8B LLM, Q8_0, arch `qwen3`, tokenizer included |

## Router section

From the production preset of the 64 GB machine (`command` points at the build of the
feature branch):

```ini
[external/moss-music]
command = D:\llm-modelle\gufo\app\gufo-moss-2026-11-12\gufo.exe
args = serve moss-music
model = D:\llm-modelle\gufo\modelle\OpenMOSS-Team\moss-music-gufo\moss-music-8b.llm-Q8_0.gguf
audio-gguf = D:\llm-modelle\gufo\modelle\OpenMOSS-Team\moss-music-gufo\moss-music-8b.audio.gguf
context = 4096
max-tokens = 1024
served-model-name = moss-music-8b
max-request-bytes = 67108864
router-gpu-gb = 12
router-host-gb = 1
router-idle-seconds = 300
```

## Measured

Strix Halo, 64 GB: about 10.6 GB GPU at context 4096, host RAM under 1 GB while loading
(12 Nov 2026).
