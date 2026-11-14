# Qwen3-TTS 12Hz 1.7B implementation notes

Developer detail behind the [public model page](../../../models/qwen3-tts.md).
[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md) · [ROCm](ROCM.md)

Native gfx1151 speech synthesis from BF16 safetensors, with an F32 waveform
decoder. Checkpoint configuration selects the variant; 0.6B, 25Hz and GGUF
checkpoints are unsupported. Official Python is an offline oracle only.

Repeated Base requests reuse the same reference's encoder features and exact
waveform-decoder state. One reference prefix is retained per runtime; changing
voices replaces it. This preserves generated audio and reduces warm latency.

## Sampling and streaming

Sampling controls and the PCM/SSE streaming formats are on the
[public model page](../../../models/qwen3-tts.md). Implementation details
behind them:

- WAV is buffered so its length header is correct; PCM is signed little-endian
  16-bit mono at 24 kHz and streams the same generation as a complete-text
  request, without sentence splitting or a server flag.
- On the `/v1/audio/speech/stream` WebSocket, `split_granularity` defaults to
  `none` (buffer text until `input.done`); `sentence` or `clause` starts
  generation earlier at text boundaries, independent of packet sizes. Splitting
  changes future context and can change prosody, so it does not promise the same
  audio as synthesizing the complete text.
- Timestamps, VAD and forced alignment are not loaded. Slow or disconnected
  consumers cancel generation; waiting requests are bounded and cancellable.
- Greedy generation can fail to reach EOS; bound diagnostic requests with
  `max_new_tokens` and see the [quality limits](QUALITY.md#known-limitations).

## Served model names

The production preset of gufo-studio-windows (`gufo-router-combo7.ini`) sets
`served-model-name` per section: `qwen3-tts` (CustomVoice),
`qwen3-tts-voicedesign` (VoiceDesign) and `qwen3-tts-clone` (Base with
registered voices). `GET /v1/models` on the router lists the names in use.
