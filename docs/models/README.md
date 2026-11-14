# Models

These are the official models: the ones Gufo releases build in and the
maintainers test on AMD Strix Halo. Each page lists the files to download, how
to run the model and what speed to expect.

<!-- BEGIN GENERATED MODEL TABLE -->
| Model | Weights / supported inputs | Guide |
| --- | --- | --- |
| DeepSeek V4 Flash | Flash 0731 mixed IQ2/Q2/Q8 GGUF; text | [Usage and modes](deepseek-v4-flash.md) |
| EmbeddingGemma 2 | BF16 safetensors; text/image to embeddings | [Usage](embeddinggemma-2.md) |
| Fish Audio Drama-3 | FP8 (F8_E4M3 + 128x128 scale_inv) MoE safetensors + DAC codec; text, optional reference voice and transcript (gufo-studio-windows) | [Usage](README.md) |
| MiniMax H3 FL2VA | Pinned safetensors; text to audiovisual output | [Presets and usage](minimax-h3.md) |
| Qwen-Image-2.1 | BF16 safetensors; generation and image editing | [Usage](qwen-image-2.1.md) |
| Qwen3-ASR 1.7B | BF16 safetensors; audio to text | [Usage](qwen3-asr.md) |
| Qwen3-TTS 12Hz 1.7B | BF16 safetensors; text/reference audio to speech | [Voice modes](qwen3-tts.md) |
| Qwen3.8 27B | Q4/Q8 GGUF; text/images | [Usage and modes](qwen3.8-27b.md) |
| Qwen3.8 Flash-Next | Sharded Q4 GGUF; text/images | [Usage and modes](qwen3.8-flash-next.md) |
| Wan 2.2 TI2V 5B and Wan 2.1 14B (Uni3C camera, VACE, InfiniteTalk) | Diffusers Wan 2.2 directory or gufo-camera/vace/talk.json bundles; text, image, audio and video conditioning | [Extension manifest (no user guide yet)](../../src/models/wan22/extension.json) |
<!-- END GENERATED MODEL TABLE -->

Fork pages outside the generated table: [Qwen3-Omni captioner](qwen3-omni-captioner.md),
[ACE-Step 1.5](ace-step-1.5.md), [MOSS-Music](moss-music.md), [TripoSG](triposg.md),
[TRELLIS.2](trellis2.md), [RIFE](rife/README.md), [Real-ESRGAN](realesrgan/README.md),
[image recognition (imgutils detectors)](vision-detect.md).

Gufo doesn't ship weights; download them from Hugging Face as each page shows.
Every speed on these pages was measured on Strix Halo with 128 GB of memory,
on the date given next to it, unless the page names the 64 GB machine.

Community extensions add other models and variants; see
[extensions](../extensions.md).

### Converted weights

Six models run in a file layout of this fork (merged BF16 safetensors or GGUF). Those files
are published ready to use under [huggingface.co/D3rPapst](https://huggingface.co/D3rPapst);
each page has a "Converted weights" section with the download command and the preset:

| Model | Repository | Size | Licence of the weights |
|---|---|---:|---|
| [ACE-Step 1.5](ace-step-1.5.md) XL (turbo / sft / base, planners 0.6B / 1.7B / 4B, text encoder, VAE) | [D3rPapst/ace-step-1.5-gufo](https://huggingface.co/D3rPapst/ace-step-1.5-gufo) | 44.9 GB | MIT; Qwen3-Embedding part Apache-2.0 |
| [MOSS-Music](moss-music.md)-8B-Instruct (audio encoder BF16 + LLM Q8_0 GGUF) | [D3rPapst/moss-music-8b-gufo](https://huggingface.co/D3rPapst/moss-music-8b-gufo) | 10.4 GB | Apache-2.0 |
| [TRELLIS.2](trellis2.md)-4B (512 path, six parts; DINOv3 encoder not included) | [D3rPapst/trellis2-gufo](https://huggingface.co/D3rPapst/trellis2-gufo) | 9.8 GB | MIT; DINOv3 under Meta's DINOv3 License, convert it yourself |
| [TripoSG](triposg.md) (one file) | [D3rPapst/triposg-gufo](https://huggingface.co/D3rPapst/triposg-gufo) | 3.92 GB | MIT |
| [Real-ESRGAN](realesrgan/README.md) compact x4 (general, wdn, anime) | [D3rPapst/realesrgan-srvgg-gufo](https://huggingface.co/D3rPapst/realesrgan-srvgg-gufo) | 12 MB | BSD-3-Clause |
| [RIFE](rife/README.md) 4.7 / 4.9 IFNet | [D3rPapst/rife-ifnet-gufo](https://huggingface.co/D3rPapst/rife-ifnet-gufo) | 43 MB | MIT |

The cards record the source revision, the converter commit, every file with its size and the
router section. The layouts are read by this fork's workers only (upstream Gufo does not load
them). Unchanged originals are not mirrored; the other pages link to the publishers' repositories.
