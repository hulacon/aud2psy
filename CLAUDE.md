# CLAUDE.md

## What this project is

aud2psy extracts numerical psychological and acoustic features from **audio**
— including the audio stream of video stimuli — mirroring
[viz2psy](../viz2psy) (images/video frames) and [word2psy](../word2psy)
(text). The audience is psychology / cognitive-neuroscience researchers
working with naturalistic stimuli (film clips, spoken narratives). Outputs
feed [psyquilt](../psyquilt) for relational matrices; see psyquilt's
CLAUDE.md "Long-term ambition" section for the movie-input plan that
motivated this package and the target stimulus set (~60 3–6-min clips, many
dialogue-free).

Design decisions mirror the siblings deliberately: same CLI feel, flat
CSV-plus-`.meta.json` outputs, registry of wrapped models with lazy imports
so `--help` stays fast, `BaseModel` with `load()`/`predict()`, one model per
module.

### Two output levels (the word2psy two-table precedent)

- **Frame-level** models produce a row-per-timepoint CSV with a `time`
  column (viz2psy video-mode analog; configurable hop). This is what
  psyquilt consumes directly — its spaces auto-detect with zero changes.
- **Segment-level** transcription produces a `text`/`onset`/`offset` CSV
  that pipes into `word2psy --text-column text` (passthrough columns carry
  timing into word2psy's chunks output). Transcription is an *export*, not
  a feature: verbal content stays word2psy's job — no duplication.

### Division of labor within the family

aud2psy owns acoustic/paralinguistic features + transcription export.
word2psy owns everything about the words themselves. CLAP (v0.2) will share
an audio–text embedding space the way viz2psy's `clip` and word2psy's
`clip_text` share OpenCLIP ViT-B-32 — psyquilt pairs such spaces via its
`COMPATIBLE_SPACES` registry.

## Architecture (`src/aud2psy/`)

*(skeleton — to be filled in as v0.1 lands; design checkpoint pending)*

## Dev environment

- `uv` for environments, **Python 3.11** (system 3.14 is not for projects).
  Recreate with `uv venv --python 3.11 && uv pip install -e ".[dev]"`.
- Deps: numpy, pandas, librosa, soundfile, faster-whisper (CTranslate2 —
  no torch in v0.1), tqdm. **ffmpeg** (system binary) required for video
  input and non-wav decoding.
- Whisper weights download from HuggingFace on first use to the HF cache.
- Tests must be offline-first with synthetic audio (sine tones, noise
  bursts, silence — known ground truth for acoustic features); transcription
  tests behind a marker.

## Working style

Ben works in explicitly approved phases: propose, wait for go-ahead at
design checkpoints, summarize finished work with concrete validation
numbers. Commit/push only when asked.

## Roadmap

1. **v0.1 — acoustic tier + transcription export** (design checkpoint
   opened Aug 2026): frame-level `loudness`, `pitch`, `spectral`, `onsets`,
   `speech` (VAD); segment-level `transcribe` (faster-whisper, word-level
   timestamps, no-speech flagging); video-audio extraction via ffmpeg;
   CLI + registry + sidecar; offline synthetic tests. Validation targets:
   a dialogue clip transcribed and piped through word2psy end-to-end with
   sensible timing; a wordless clip cleanly flagged no-speech with
   plausible loudness/onset curves; psyquilt auto-detects aud2psy spaces.

### Explicitly deferred (do not build without discussion)

- **CLAP embeddings (v0.2 flagship)** — `clap_{i:03d}` audio +
  `clap_text_{i:03d}` text in one shared space; torch dependency arrives
  here. Coordinate column naming with psyquilt's `COMPATIBLE_SPACES`.
- **Speaker diarization** (who is speaking when).
- **Prosodic-emotion models** — how a line is said vs. its content; a
  different affective signal than word2psy's text `emotion`.
- **SRT/VTT subtitle ingestion** as an alternate transcription input
  (useful for full-length commercial films where verbatim SRTs exist).
- **Music-specific features** (key, mode, harmony) beyond basic
  onset/tempo.

Prior art to consult before expanding the registry: `pliers` (Yarkoni lab
multimodal extraction), studyforrest movie annotations.
