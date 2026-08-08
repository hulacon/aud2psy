# aud2psy

Extract psychological and acoustic features from audio — including the audio
stream of video stimuli — for psychology and cognitive-neuroscience research.
Sibling of [word2psy](https://github.com/hulacon/word2psy) (text) and
[viz2psy](https://github.com/hulacon/viz2psy) (images/video); outputs feed
[psyquilt](https://github.com/hulacon/psyquilt) for relational matrices.

## Install

```bash
pip install -e .
```

Requires Python 3.10–3.12 and the `ffmpeg` binary on PATH (for video input
and non-wav audio). Whisper weights download from HuggingFace on first use
of `transcribe` (the `speech` VAD model needs no Whisper weights).

## Usage

```bash
# frame-level features + transcript from a film clip
aud2psy --all clip.mp4 -o scores.csv

# just the acoustic tier, finer time grid
aud2psy loudness pitch spectral onsets speech clip.wav -o scores.csv --hop 0.25

# what's available
aud2psy --list-models
```

`-o scores.csv` writes:

| File | Contents |
|------|----------|
| `scores_frames.csv` | one row per 0.5 s window (`time` = window center) with all frame-level features flat — psyquilt-ready |
| `scores_transcript.csv` | one row per Whisper segment: `text`, `onset`, `offset`, `asr_confidence`, `no_speech_prob` |
| `scores_transcript_words.csv` | word-level timestamps |
| `scores.meta.json` | provenance sidecar |

A wordless clip produces a zero-row transcript and `n_speech_segments: 0`
in the sidecar — an explicit result, not an error.

## Models

| Model | Level | Features |
|-------|-------|----------|
| `loudness` | frame | `loudness_rms`, `loudness_db` |
| `pitch` | frame | `pitch_f0` (pYIN, NaN when unvoiced), `pitch_voiced_prob` |
| `spectral` | frame | `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_flux`, `spectral_zcr` |
| `onsets` | frame | `onsets_strength`, `onsets_rate`, `onsets_tempo` |
| `speech` | frame | `speech_prob` (Silero VAD) |
| `transcribe` | segment | time-stamped transcript export (faster-whisper, default `large-v3`; `--whisper-model` to change) |

## Piping the transcript into word2psy

Verbal content is word2psy's job — `transcribe` is an export, not a feature:

```bash
aud2psy transcribe clip.mp4 -o scores.csv
word2psy --all scores_transcript.csv --text-column text -o words.csv
```

The transcript's `onset`/`offset` columns pass through into word2psy's
chunks output, so every text feature stays on the clip's timeline.

## License

MIT
