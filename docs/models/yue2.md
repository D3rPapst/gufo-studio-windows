# YuE2 songs

`gufo serve music --yue2-vae DIR` runs YuE2-3B (m-a-p, HKUST) natively on HIP: style tags
and lyrics with `[Verse]` / `[Chorus]` sections to a full song with vocals, 48 kHz stereo.
The model first writes an ABC score (melody and chords), then 25 Hz music codes, then
turns them into audio latents by flow matching; the YuE2-Vae decoder makes the waveform.

**Licence of the weights:** CC BY-NC 4.0 plus the creator permission in `MODEL_LICENSE`
of the [YuE repository](https://github.com/multimodal-art-projection/YuE/blob/main/MODEL_LICENSE)
(16 Sep 2026). Weights are non-commercial; outputs may be sold and licensed by individuals
and creators according to `MODEL_LICENSE`, companies need their own licence. The Hugging
Face model card shows only CC BY-NC 4.0; keep a copy of `MODEL_LICENSE` with its commit next
to the weights. Gufo publishes no converted YuE2 weights.

## Files

Two unchanged Hugging Face repositories, no conversion:

```powershell
hf download m-a-p/YuE2-3B --local-dir D:\llm-modelle\gufo\modelle\m-a-p\YuE2-3B
hf download m-a-p/YuE2-Vae --local-dir D:\llm-modelle\gufo\modelle\m-a-p\YuE2-Vae
```

| File | Size | Content |
|---|---:|---|
| `YuE2-3B/model.safetensors` | 7.26 GB | AR and NAR weight sets, BF16 |
| `YuE2-3B/qwen.tiktoken` | 2.6 MB | text and ABC tokenizer |
| `YuE2-Vae/model.safetensors` | 0.53 GB | Oobleck VAE, FP32 (only the decoder is used) |

`--model` is the YuE2-3B folder; `--yue2-vae` defaults to `YuE2-Vae` next to it.
`YuE2-Vae-legacy` (the decoder of the paper's scores) loads the same way.

## Request

`POST /v1/audio/music` on the YuE2 worker:

| Field | Meaning |
|---|---|
| `tags` (or `prompt`, `style`) | genre, instruments, voice, mood, tempo |
| `lyrics` | sections `[Intro]`, `[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]`; may be empty |
| `language` | `de`, `en`, ... appended to the tags when they do not name it |
| `duration` | upper bound in seconds (0 = the model ends the song, at most 6 minutes); a shorter bound cuts the song |
| `seed` | integer |
| `cot` | `full` (score with chords, default), `melody`, `off` (no score) |
| `abc` | own ABC score instead of the planned one |
| `steps` | flow-matching steps (default 32) |
| `response_format`, `bitrate_kbps`, `async` | as for ACE-Step |

The job status (`GET /v1/audio/music/{id}`) carries the score in `abc`, the code count and
whether the song was cut (`truncated`). There is no live preview, no reference audio and no
repaint; use ACE-Step for those.

## Router section

Proposal, not in the production preset:

```ini
[external/yue2-3b]
command = D:\llm-modelle\gufo\app\<build>\gufo.exe
args = serve music
model = D:\llm-modelle\gufo\modelle\m-a-p\YuE2-3B
yue2-vae = D:\llm-modelle\gufo\modelle\m-a-p\YuE2-Vae
served-model-name = yue2-3b
router-gpu-gb = 12
router-host-gb = 1
router-idle-seconds = 300
```

## Measured

MEASURED
