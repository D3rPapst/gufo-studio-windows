# Transcription segments

Speech recognition (`include/gufo/ext/asr.hpp`): a runner can return timed
segments with its transcript.
- `TranscriptionSegment`: `start` and `end` in seconds from the start of the
  audio, `text`, the model's `tokens` (timestamps included), `avg_logprob` and
  `no_speech_prob`.
- `TranscriptionResult::segments`: the segments in order; empty when the model
  produces none.
- `GUFO_EXT_ASR_SEGMENTS` is defined while the field exists, so an extension
  can fill it and still build against older 0.3 revisions.

`POST /v1/audio/transcriptions` with `response_format=verbose_json` lists the
segments as OpenAI does (`id`, `start`, `end`, `text`, `tokens`,
`temperature`, `avg_logprob`, `no_speech_prob`); before, `segments` was always
empty.

Compatibility: additive. The field defaults to empty, so Qwen3-ASR, LFM3 and
other runners need no change and still answer with `"segments": []`.
