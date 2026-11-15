# Transcription words

Speech recognition (`include/gufo/ext/asr.hpp`): a runner can return the
time span of every word.
- `TranscriptionRequest::word_timestamps`: the client asked for words
  (`timestamp_granularities[]=word`). Word alignment can cost an extra pass,
  so runners compute words only when it is set.
- `TranscriptionWord`: `word` as the model wrote it (it may start with a space
  and carry adjacent punctuation), `start` and `end` in seconds from the start
  of the audio.
- `TranscriptionResult::words`: the words in order; empty when the request
  did not ask for them or the model has no word timing.
- `GUFO_EXT_ASR_WORDS` is defined while these fields exist, so an extension
  can fill them and still build against older 0.3 revisions.

`POST /v1/audio/transcriptions` with `response_format=verbose_json` and
`timestamp_granularities[]=word` lists the words as OpenAI does (`word`,
`start`, `end`; the word without surrounding spaces); before, such requests
were rejected.

Compatibility: additive. The fields default to false and empty, so Qwen3-ASR,
LFM3 and other runners need no change and answer with `"words": []`.
