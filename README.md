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
| `scores_beats.csv` | beat/downbeat events (with the `beats` model) |
| `scores.meta.json` | provenance sidecar |

A wordless clip produces a zero-row transcript and `n_speech_segments: 0`
in the sidecar — an explicit result, not an error.

> **Word-timestamp accuracy.** Whisper-derived word timestamps are good
> to roughly the 100–200 ms level, not forced-alignment level — published
> checks put Whisper-based aligners at 84–93% of words within a 200 ms
> collar, with Montreal Forced Aligner the accuracy reference
> ([comparison](https://arxiv.org/pdf/2406.19363)). They are well suited
> to chunk-level timing, fMRI-scale regressors, and word2psy piping, but
> **not** to vocalization-onset-locked EEG/iEEG analyses or voice-key
> reaction times. For those, refine the word onsets with a forced aligner
> (e.g. MFA) before analysis.

## Models

| Model | Level | Features |
|-------|-------|----------|
| `loudness` | frame | `loudness_rms`, `loudness_db` |
| `pitch` | frame | `pitch_f0` (pYIN, NaN when unvoiced), `pitch_voiced_prob` |
| `spectral` | frame | `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_flux`, `spectral_zcr` |
| `onsets` | frame | `onsets_strength`, `onsets_rate`, `onsets_tempo` |
| `tonal` | frame | `tonal_key_clarity`, `tonal_majorness`, `tonal_chroma_entropy` (Krumhansl profiles, 3 s windows) |
| `rhythm` | frame | `rhythm_pulse_clarity`, `rhythm_beat_strength`, `rhythm_novelty` (Foote section novelty) |
| `speech` | frame | `speech_prob` (Silero VAD) |
| `clap` | frame | `clap_000`…`clap_511`: LAION-CLAP audio embeddings in a shared space with word2psy's `clap_text` (10 s windows; `--clap-model` to change checkpoint) |
| `beats` | events | beat/downbeat table (`time`, `is_downbeat`) via [beat_this](https://github.com/CPJKU/beat_this); needs `pip install "aud2psy[beats]"` |
| `transcribe` | segment | time-stamped transcript export (faster-whisper, default `large-v3`; `--whisper-model` to change) |

## Piping the transcript into word2psy

Verbal content is word2psy's job — `transcribe` is an export, not a feature:

```bash
aud2psy transcribe clip.mp4 -o scores.csv
word2psy --all scores_transcript.csv --text-column text -o words.csv
```

The transcript's `onset`/`offset` columns pass through into word2psy's
chunks output, so every text feature stays on the clip's timeline.

## Cross-modal audio–text comparison

`clap` audio embeddings share a space with word2psy's `clap_text`
(same LAION-CLAP checkpoint), so soundtrack windows can be compared
directly to captions or transcript chunks — and psyquilt pairs the two
automatically in cross mode:

```bash
aud2psy clap clip.mp4 -o audio.csv
word2psy clap_text captions.csv --text-column caption -o text.csv
psyquilt matrices audio_frames.csv text_chunks.csv -o out/
```

## Free-recall annotation

For spoken free-recall recordings, supply a wordpool (one item per line)
to get an automated annotation — the hand-scoring workflow of tools like
Penn-TotalRecall, automated:

```bash
aud2psy transcribe recall_session.wav --wordpool pool.txt -o scores.csv
```

This adds `scores_recall.csv`: one row per spoken word with
`matched_item` (exact-then-fuzzy match against the pool), `pool_index`,
`match_score`, `intrusion`, `repetition`, and `irt` (inter-response time
between successive matched recalls). The sidecar gains recall summary
counts and speech-timing metrics (latency to first word, speech rate,
pause statistics). Mind the word-timestamp caveat above: onsets are
fMRI/behavior-grade, not EEG-grade.

## License

MIT
