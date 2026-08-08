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

- **`audio.py`** — input decoding. `.wav` reads via librosa/soundfile; every
  other supported extension (audio: .mp3/.m4a/.flac/.ogg; video:
  .mp4/.mov/.mkv/.webm) funnels through ffmpeg to a temp wav (`-vn -ac 1
  -ar <sr>`). Two sample rates: 22050 Hz for features (`FEATURE_SR`),
  16 kHz for Whisper/VAD (`WHISPER_SR`); the pipeline decodes once per rate
  actually needed.
- **`grid.py`** — the shared frame grid. `Grid.for_duration(duration, hop)`;
  window k covers [k·hop, (k+1)·hop), `time` = window center. Models compute
  at librosa native resolution then reduce with `Grid.average` (NaN-aware
  mean — unvoiced pyin frames excluded, all-NaN windows stay NaN) or
  `Grid.rate` (events/sec, used for onset rate).
- **`models/base.py`** — `BaseModel` with class attrs `name` + `level`.
  Frame models implement `extract(y, sr, grid) -> dict[str, np.ndarray]`
  (arrays of length `grid.n_windows`); the segment model implements
  `transcribe(y_16k, sr)`. `load()` is lazy; `unload()` frees weights.
  **Feature keys are model-prefixed** (`loudness_rms`, `spectral_centroid`)
  — the word2psy chunk-model convention; psyquilt's `PROFILE_REGISTRY`
  pattern-matches these (entries `loudness`/`pitch`/`spectral`/`onsets` +
  combined `acoustic` were added to psyquilt in Aug 2026 — coordinate any
  renaming with that registry).
- **`models/`** — frame level: `loudness` (rms, db — dB of window-mean RMS),
  `pitch` (pyin f0 in 65–2093 Hz, voiced_prob; f0 NaN when unvoiced),
  `spectral` (centroid, bandwidth, rolloff, flux = half-wave-rectified STFT
  L2 flux, zcr), `onsets` (strength, rate via `Grid.rate` on
  `onset_detect` events, framewise tempo), `speech` (`speech_prob`: real
  per-32ms Silero VAD probabilities via faster-whisper's bundled ONNX model
  — no Whisper weights; resamples 22050→16k internally). Segment level:
  `transcribe` (faster-whisper, `word_timestamps=True`, `vad_filter=True`
  against hallucination on wordless audio, CPU/int8; `asr_confidence` =
  exp(avg_logprob)). Wordless clips → zero-row transcript +
  `n_speech_segments: 0` in the sidecar: an explicit result, not an error.
- **`pipeline.py`** — `score_audio(path, models, hop, whisper_model,
  language) -> ScoreResult(frames_df, transcript_df, words_df, meta)`;
  models load/extract/unload one at a time. `save_result` writes the
  family file set for `-o scores.csv`: `scores_frames.csv` (time +
  features flat — the psyquilt-ready file), `scores_transcript.csv`
  (raw Whisper segments: segment_idx/text/onset/offset/asr_confidence/
  no_speech_prob — pipes into `word2psy --text-column text`),
  `scores_transcript_words.csv` (word-level timestamps),
  `scores.meta.json`.
- **`metadata.py`** — sidecar builder (version, input, per-model columns +
  runtimes, frames hop/n, transcription config incl. detected language and
  n_speech_segments).
- **`cli.py`** — `MODEL_REGISTRY` maps name → (module path, class name,
  description); lazy imports keep `--help`/`--list-models` fast (a test
  asserts librosa/faster_whisper/pandas are not imported for `--help`).
  Usage: `aud2psy <models...> clip.mp4 -o scores.csv`, `--all`, `--hop`
  (default **0.5 s**, 1:1 with viz2psy video frames), `--whisper-model`
  (default **large-v3**), `--language`, `--list-models`. No `-o` → frames
  CSV to stdout.
- **`tests/`** — offline synthetic suite (sine/noise/silence with
  closed-form ground truth per feature; 35 tests, ~3 s); transcription
  tests behind the `transcription` marker (deselected by default via
  `addopts`; they synthesize speech with macOS `say` and use Whisper
  `small` to stay light).

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

1. **v0.1 — acoustic tier + transcription export** (done Aug 2026):
   everything in the Architecture section above. Checkpoint decisions:
   default `--hop` 0.5 s (1:1 with viz2psy frames), default Whisper
   **large-v3** (Ben's call over `small`; ~1.4× real time on this CPU at
   int8), transcript rows = raw Whisper segments (no sentence regroup).
   Tests: 35 offline synthetic (~3 s) + 2 behind the `transcription`
   marker. Validation (all Aug 2026, synthetic stimuli):
   - *(a) dialogue*: 16.7 s two-voice `say` dialogue muxed into .mp4 →
     `--all` → all 4 turns transcribed verbatim, onsets tracking the
     0.6 s turn gaps (0 / 3.2 / 7.6 / 11.6 s); transcript piped through
     `word2psy --text-column text`: 4 chunks with onset/offset passthrough
     intact and face-valid sentiment (the flood-warning line maxes
     negative at .83). Note: Whisper computes `asr_confidence` /
     `no_speech_prob` per 30 s decode window, so clips under 30 s show
     one shared value across segments — upstream behavior.
   - *(b) wordless*: 20 s synthetic 120-BPM melody+percussion clip →
     0 transcript rows, `n_speech_segments: 0`, max speech_prob .047;
     onsets_rate 1.95/s vs 2.0 ground truth, median tempo 117.5 vs 120
     BPM, loudness tracks the synthesized crescendo (r = .99).
   - *(c) psyquilt*: `psyquilt spaces dialogue_frames.csv` detects 5
     profile spaces (loudness/pitch/spectral/onsets + 13-d acoustic).
     This needed a PROFILE_REGISTRY addition in psyquilt (committed
     there), matched by aud2psy's model-prefixed column naming — the
     original "zero changes" claim held for row identity (`time`), not
     space detection.

2. **v0.1.x — no-new-deps tier** (candidates from the Aug 2026 literature
   survey, below; per-item go-ahead still required):
   - **`tonal` model** — key clarity, mode-majorness, chroma summary,
     computed in ~3 s sliding windows emitted on the 2 Hz grid (librosa
     `chroma_cqt` + Krumhansl/Temperley profile correlation, implemented
     natively — Essentia's KeyExtractor is AGPL and its macOS arm64 wheel
     is broken). No maintained Python tool provides key clarity — a
     genuine niche.
   - **`rhythm` upgrade** — pulse clarity (tempogram peak salience) + PLP
     local tempo + a structural novelty curve (librosa recurrence matrix).
     With `tonal`, this completes the Alluri/Toiviainen MIRtoolbox canon
     of naturalistic-music-fMRI regressors
     (https://pmc.ncbi.nlm.nih.gov/articles/PMC9531138/).
   - **Recall export** — wordpool-aware post-processing of the word
     timestamps table: fuzzy match to a supplied wordpool, emit
     matched_item/intrusion/repetition + inter-response times; plus
     pause/speech-rate features from VAD + word timing. Direct ask of the
     free-recall community — the Kahana lab is building exactly this on
     WhisperX (https://github.com/pennmem/automated_annotation), and
     quail's transcription layer is a dead Google API.
3. **v0.2 — torch tier**: CLAP embeddings (flagship, below) + **`beats`
   segment-level table** via beat_this (MIT, CPU-capable,
   https://github.com/CPJKU/beat_this — do NOT use madmom: dead upstream,
   pinned to Python <3.10) + optionally a DEAM-trained valence/arousal
   probe on MERT/CLAP (DEAM's dynamic emotion annotations are natively
   2 Hz — a literal match to our default grid).

### The 2 Hz question (music survey, Aug 2026)

Musical properties (key, mode, meter, emotion) integrate over 1–60 s, but
the naturalistic-music-fMRI field's standard practice is exactly our
architecture: compute in long sliding windows, sample onto a fast frame
grid (window ≫ hop; `Grid` already supports this). Event-shaped outputs
(beats, chords, sections) go to segment-level tables — the transcript
export's `onset`/`offset` pattern. Nothing about the 2-level design needs
to change for music.

### N.B. — Whisper word-timestamp accuracy

Whisper-derived word timestamps (ours included) are **not
forced-alignment grade**: published checks put WhisperX-style alignment
at 84–93% of words within a 200 ms collar, and Montreal Forced Aligner
remains the accuracy reference (https://arxiv.org/pdf/2406.19363).
faster-whisper's cross-attention timestamps are a tier below WhisperX
(which adds wav2vec2 phoneme alignment). Fine for word2psy chunking and
fMRI-scale regressors; NOT sufficient for vocalization-onset-locked
EEG/iEEG analyses. Documented in the README; an optional alignment
refinement stage (MFA export now / wav2vec2 in the torch tier) is the
deferred fix.

### Explicitly deferred (do not build without discussion)

- **CLAP embeddings (v0.2 flagship)** — `clap_{i:03d}` audio +
  `clap_text_{i:03d}` text in one shared space; torch dependency arrives
  here. Coordinate column naming with psyquilt's `COMPATIBLE_SPACES`.
- **Word-timestamp refinement** — MFA TextGrid round-trip (CPU, no torch,
  heavy Kaldi install) or wav2vec2 alignment (torch tier). See N.B. above.
- **Verbatim/disfluency mode** — vanilla Whisper silently deletes fillers
  and repetitions; CrisperWhisper (https://arxiv.org/abs/2408.16589)
  preserves them with timed filler events. Needed before anyone uses our
  transcripts for fluency/pause-content research.
- **Speaker diarization** (who is speaking when) — pyannote via WhisperX
  is the community standard; torch + HF-gated weights.
- **Prosodic-emotion models** — how a line is said vs. its content; a
  different affective signal than word2psy's text `emotion`. openSMILE
  eGeMAPS is the standard feature set (non-OSI license — optional extra).
- **SRT/VTT subtitle ingestion** as an alternate transcription input
  (useful for full-length commercial films where verbatim SRTs exist).
- **Chords / section labeling** — chroma + key clarity capture most tonal
  variance for regressors; deep section labelers (all-in-one) are trained
  on pop-song form, a poor prior for film scores, with fragile deps.
- **Audio surprisal/expectancy** — hot 2023–26 direction, no production
  tooling; D-REX (MATLAB) could someday consume aud2psy's own exported
  pitch/loudness columns. Watch, don't wrap.
- **Chronset-style speech-onset latency** — validated voice-key
  replacement (<50 ms error) with no maintained Python equivalent; only
  relevant if single-trial RT users show up.

Prior art to consult before expanding the registry: `pliers` (Yarkoni lab
multimodal extraction), studyforrest movie annotations, CANLab's
narrative-annotation scoping review
(https://github.com/canlab/narrative_feature_annotations).
