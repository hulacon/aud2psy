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
  `onset_detect` events, framewise tempo), `tonal` (key clarity /
  majorness / chroma entropy: chroma_cqt smoothed over 3 s, correlated
  against the 24 Krumhansl–Kessler profiles; NaN on silence), `rhythm`
  (pulse clarity: beat-lag-restricted local autocorrelation of the
  **mean-removed** onset envelope — without mean removal a quasi-constant
  envelope autocorrelates everywhere and noise beats a click track —
  gated to 0 where onset activity < 0.1 log-mel-flux units, which is
  loudness-invariant; beat strength: `Grid.window_max` of the PLP curve,
  since an oscillating pulse curve's window *mean* is uninformative;
  novelty: Foote checkerboard on an MFCC cosine SSM at 0.2 s frames),
  `timbre` (MFCCs 1–13 — c0 excluded as loudness_db's job, the
  encoding-model convention — plus 7-band per-octave spectral contrast
  and spectral flatness; frontend constants shared with `spectral` so
  native frames align), `psychoacoustic` (Zwicker loudness/sharpness/
  roughness via MoSQITo — core dep, Apache-2.0, plus matplotlib only
  because mosqito 1.2.1 imports it at module load, an upstream bug;
  `input_sr = 48000`; float samples mapped as digital RMS 1.0 = 94 dB
  SPL since files carry no calibration; sharpness computed via
  `sharpness_din_from_loudness` on the one `loudness_zwtv` pass —
  bit-identical to `sharpness_din_tv`, verified — and NaN-gated below
  0.25 sone window loudness (a spectral ratio is undefined on silence);
  `psychoacoustic_fluctuation` is a **native Fastl-style estimate** —
  MoSQITo has no fluctuation strength — from 1-bark specific-loudness
  envelopes: 2 s modulation-spectrum windows at grid centers, relative
  depth with a 0.01 sone floor, H(f)=1/(f/4+4/f) weighting,
  root-sum-square over modulation bins (L1 over-counts broadband
  transients), one-point calibrated to Fastl's 4 Hz reference = 1.0;
  ~3x real time on CPU — slowest non-transcription model), `speech` (`speech_prob`: real per-32ms Silero VAD probabilities via
  faster-whisper's bundled ONNX model — no Whisper weights; resamples
  22050→16k internally), `clap` (v0.2 flagship: 512-d L2-normalized
  LAION-CLAP embeddings, default checkpoint
  `laion/larger_clap_music_and_speech`, 10 s context windows strided at
  the grid hop, batched, MPS with CPU fallback; declares `input_sr =
  48000` so the pipeline decodes at full bandwidth — transformers ≥ 5
  returns ModelOutput objects, the 512-d projection is `pooler_output`.
  **Shared space with word2psy's `clap_text`** (same checkpoint) — do not
  change checkpoint/normalization/naming without coordinating word2psy
  and psyquilt's `COMPATIBLE_SPACES`), `music_emotion` (DEAM-trained
  ridge probe applied to CLAP embeddings → `music_emotion_valence` /
  `music_emotion_arousal` in [-1, 1]; the fitted probe ships as package
  data `data/music_emotion_probe.{npz,json}` — ~4 KB coefficient matrix
  plus provenance with CV numbers — reproduced by
  `scripts/train_deam_probe.py`; embeddings are recomputed even when
  `clap` also runs, a known deferred optimization), `speech_emotion`
  (audeering `wav2vec2-large-robust-12-ft-emotion-msp-dim` →
  `speech_emotion_valence`/`_arousal`/`_dominance` in the model's native
  ~0–1 (deliberately NOT rescaled — music_emotion's [-1,1] comes from its
  DEAM training targets, not a rescale); 4 s context windows at the grid
  hop, 16 kHz, MPS w/ CPU fallback; **Silero-VAD-gated** — windows with
  mean speech prob < 0.25 are NaN, else the model hallucinates affect on
  music; the custom regression-head class from the model card needed
  `post_init()` instead of its 4.x-era `init_weights()` under
  transformers 5, verified bit-for-bit against the card's reference
  output; weights ungated but CC-BY-NC-SA research-only, documented in
  README; no new deps), `egemaps` (the 25 eGeMAPSv02 LLDs via the
  `opensmile` wrapper — optional `[egemaps]` extra (audEERING
  research-only license; also in dev deps since it's offline and
  weightless, so the tests always run); 16 kHz input shared with
  Silero; openSMILE names carry `-`/`.` so they're sanitized to
  `egemaps_*` keys with the raw mapping recorded in the sidecar; the
  15 pitch-synchronous `_sma3nz` columns get exact-0 unvoiced
  sentinel → NaN before `Grid.average` (openSMILE's own "nz"
  functional convention), then the whole voiced set — including
  formants, which LPC computes even on noise — is Silero-gated per
  grid window reusing speech_emotion's `speech_fraction` + threshold;
  the 10 general-audio columns (loudness/spectral balance/MFCC 1–4/
  flux) stay ungated. Pipeline additions for this: models may set
  `info_` for extra sidecar fields, and the embedding-pattern sidecar
  branch now requires numeric `name_NNN` suffixes (`_is_embedding`)
  so 25 named columns aren't squashed into a clap-style pattern).
  Event level: `beats`
  (beat_this,
  optional `[beats]` extra since it's a git dep — madmom is dead
  upstream; CPU inference; one row per beat with `is_downbeat`, sidecar
  gets n_beats/n_downbeats/median tempo), `diarize` (pyannote
  `speaker-diarization-community-1`, optional `[diarize]` extra — the
  weights are HF-gated (one-time license acceptance + token), so it can
  never work straight out of pip; CPU inference on an in-memory 16 kHz
  waveform so pyannote's own decoder never touches inputs; one row per
  speaker turn in `{stem}_speakers.csv` from the raw timeline, while the
  *exclusive* timeline drives speaker merging onto the transcript
  (majority overlap per segment) and words table (word-midpoint lookup —
  Whisper merges turns into one segment routinely, so the word-level
  column is the trustworthy one); `--num-speakers` forwards the
  exact-count hint; sidecar gets pipeline/n_speakers/n_turns/per-speaker
  speech time; merge helpers are pure pandas, offline-tested). Segment
  level:
  `transcribe` (faster-whisper, `word_timestamps=True`, `vad_filter=True`
  against hallucination on wordless audio, CPU/int8; `asr_confidence` =
  exp(avg_logprob)). Wordless clips → zero-row transcript +
  `n_speech_segments: 0` in the sidecar: an explicit result, not an error.
  `--verbatim` swaps the checkpoint for the official CT2 conversion of
  CrisperWhisper v1 (`nyrahealth/faster_CrisperWhisper`, CC-BY-NC-4.0,
  **en/de only** — enforced at init): fillers arrive as `[UH]`/`[UM]`
  tags (`is_filler` words column, excluded from recall matching but not
  speech_timing), repetitions/false starts kept. Three verbatim-mode
  inversions of normal behavior, all validated empirically:
  `vad_filter` **off** (Silero's resegmentation destabilizes
  CrisperWhisper, whose anti-hallucination training gives 0 segments on
  wordless audio without help); a **degenerate-loop retry** — a genuine
  "I I" in the audio stochastically seeds a `,I`×100 loop through
  temperature fallback (observed with and without explicit `language`),
  `repetition_penalty=1.1` reliably kills it, BUT CTranslate2 applies
  the penalty across the whole decode window and it suppressed a
  genuine distant repeat ("apple … apple" in a recall list), so the
  penalty fires only when `max_word_run > 6` flags a degenerate first
  pass (`degenerate_retry` in the sidecar); and word tokens arrive as
  `,word` / `.word`-after-pause (the retrained no-space-prefix
  tokenizer through CT2), so `clean_verbatim_word` strips leading
  separators (trailing punctuation is real) and segment `text` is
  rebuilt by joining cleaned words. The sidecar's `vad_filter` now
  comes from the model, not hardcoded in metadata.py. Known residue: an
  occasional 1-word junk trailing segment at speech offset (low
  asr_confidence, uninformative no_speech_prob;
  `hallucination_silence_threshold` made output worse — accepted and
  documented).
- **`pipeline.py`** — `score_audio(path, models, hop, whisper_model,
  language) -> ScoreResult(frames_df, transcript_df, words_df, meta)`;
  models load/extract/unload one at a time. `save_result` writes the
  family file set for `-o scores.csv`: `scores_frames.csv` (time +
  features flat — the psyquilt-ready file), `scores_transcript.csv`
  (raw Whisper segments: segment_idx/text/onset/offset/asr_confidence/
  no_speech_prob — pipes into `word2psy --text-column text`),
  `scores_transcript_words.csv` (word-level timestamps),
  `scores.meta.json`.
- **`recall.py`** — free-recall export (not a model; runs with
  `--wordpool` + `transcribe`): exact-then-fuzzy (difflib, threshold .8)
  matching of the word-timestamp table against a wordpool file, emitting
  `{stem}_recall.csv` (word/onset/offset/matched_item/pool_index/
  match_score/intrusion/repetition/irt; `irt` = time between successive
  *matched* recalls, the list-recall convention — intrusions don't
  advance it). `speech_timing` summary (latency to first word, speech
  rate, pauses ≥ 0.5 s) lands in the sidecar on every transcribe run.
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
  closed-form ground truth per feature; 79 tests, ~5 s); transcription
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

2. **v0.1.1 — no-new-deps tier** (done Aug 2026; motivated by the
   literature survey below): `tonal` + `rhythm` frame models (the
   Alluri/Toiviainen MIRtoolbox regressor canon,
   https://pmc.ncbi.nlm.nih.gov/articles/PMC9531138/ — no maintained
   Python tool provided key/pulse clarity; implemented natively on
   librosa, dodging Essentia's AGPL + broken macOS arm64 wheel) and the
   free-recall export (the Kahana lab is building the same thing on
   WhisperX, https://github.com/pennmem/automated_annotation). Design
   details in Architecture. Tests: 50 offline + 3 marked. Validation
   (Aug 2026, synthetic stimuli):
   - *tonal*: C-major vs C-minor triads → majorness +.12 / −.32 with key
     clarity .85/.91 vs .46 on white noise; chroma entropy 1.00 on noise.
     The A-minor-ish wordless clip lands majorness −.03 (its pitch-class
     set is genuinely major/minor-ambiguous), key clarity .51 vs .38 for
     dialogue.
   - *rhythm*: pulse clarity separates the 120-BPM wordless clip (.61)
     from dialogue (.23) and, on ground-truth signals, click track .57 >
     white noise .23 > steady tone/silence .00 (the tone case required
     mean-removing the onset envelope and gating by onset activity —
     first drafts ranked noise *above* the click track). Novelty peaks
     within one window of a known 6.0 s texture boundary at 16× baseline.
   - *recall*: 8-word spoken list (`say`, one intrusion, one repetition)
     → 8/8 words matched correctly through large-v3: 6 unique recalls,
     the intrusion caught (best pool match .364 < .8 threshold), the
     repetition flagged, IRTs on the clip timeline (mean 1.94 s) with the
     intrusion correctly not advancing IRT; sidecar speech timing:
     7 pauses ≥ .5 s, speech rate .65 wps.
   - *psyquilt*: detects 7 profile spaces on the 20-column frames CSV
     (per-model + 19-d acoustic; registry extended in psyquilt,
     committed there).
3. **v0.2 — torch tier** (done Aug 2026): `clap` frame embeddings +
   `beats` event table; torch/transformers now core deps (word2psy gained
   `clap_text`, psyquilt's COMPATIBLE_SPACES gained the clap pairs — both
   committed in their repos). The DEAM valence/arousal probe was
   deliberately held back for its own design checkpoint (dataset download
   + training/validation protocol). Validation (Aug 2026):
   - *clap*: 20 s clip → 40×512 unit-norm embeddings in 5.6 s on MPS
     (~1.7 min per 6-min clip). Cross-modal retrieval, mean-pooled clip
     embeddings vs word2psy `clap_text` captions: 3/3 diagonal — dialogue
     ↔ "two people having a conversation" .334 (next best .138), music ↔
     "a melody with a steady drum beat" .222 (.068), synthetic rain ↔
     "rain falling" .286 (.128).
   - *beats*: 120-BPM wordless clip → 41 beats, inter-beat interval
     .500 ± .000 s, median tempo 120.0 BPM exactly.
   - *psyquilt cross mode end-to-end*: `psyquilt matrices
     clap_frames.csv captions_chunks.csv` emits a 30×3
     `clap__x__clap_text__cosine` matrix.

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
EEG/iEEG analyses. Documented in the README, which is the permanent
answer: in-package refinement was **descoped** Aug 2026 (see the
deferred list) — precision-timing users should hand-verify alignments
in a forced aligner themselves.

4. **v0.3 — music_emotion probe** (done Aug 2026): design checkpoint
   decisions — probe on CLAP (not MERT), RidgeCV per dimension (the
   word2psy norms recipe), 45-s excerpts only, 2 s training stride,
   ship fitted coefficients as package data, name `music_emotion_*`
   (Ben's call over `affect_*`; `emotion_*` would collide with
   word2psy's GoEmotions profile in psyquilt), separate psyquilt profile
   NOT folded into the 19-d acoustic. Trained on 26,198 windows from
   1,746 songs (~20 min embed on MPS, cached at
   `~/.cache/aud2psy/deam/`). **Song-level 5-fold GroupKFold CV: pooled
   r = .706 valence / .838 arousal** (above the .4–.6 / .6–.7 literature
   anchors), RMSE .169/.156. **Within-song curve-tracking r is only
   .06/.11 — but diagnostics show this is mostly a dataset ceiling**:
   DEAM's median within-song annotation SD is .026–.029 vs .23–.28
   between songs, so ~99% of the dynamic variance the probe can learn is
   between-song. Interpretation for users: `music_emotion` discriminates
   affective *levels* between clips and between section-sized textures
   within a film clip; it is not a beat-to-beat affect tracker. A
   temporal model (LSTM/transformer over embedding sequences) is the
   documented upgrade path if within-excerpt dynamics ever matter.

5. **v0.4 — speaker diarization** (done Aug 2026): `diarize` via
   pyannote.audio 4.0.7 + `speaker-diarization-community-1` (CC-BY-4.0,
   supersedes the 3.1 pipeline WhisperX standardized on; better DER and
   built-in exclusive diarization, which made WhisperX unnecessary — the
   merge is ~30 lines of pandas here). Checkpoint decisions: optional
   `[diarize]` extra (gated weights mean it can't work out-of-the-box
   anyway), `--num-speakers` exact-count hint only, recall export left
   alone. Tests: 58 offline (+6 merge-logic) + 2 behind the new
   `diarization` marker. Validation (Aug 2026, synthetic stimuli):
   - *dialogue*: 14 s two-voice `say` dialogue (4 turns, 0.6 s gaps) →
     exactly 2 speakers, 4 turns, turn onsets within 0.03–0.06 s of
     ground truth, both with and without the `--num-speakers 2` hint;
     word-level speaker purity 100% (37/39 words assigned; 2 trailing
     words fall in a gap → ""), speaker↔voice_gender mapping perfectly
     1:1. Whisper merged turns 1+2 into one segment (routine), which is
     why per-word speakers matter.
   - *wordless*: 20 s melody+percussion → 0 speakers, 0 turns; silence
     likewise (explicit result, not an error).
   - *word2psy passthrough*: speaker-stamped transcript through
     `word2psy sentiment --text-column text` → `speaker` survives into
     chunks output next to onset/offset.
   - *runtime*: 8.2 s for the 14 s clip on CPU (M-series), incl. ~5 s
     pipeline load — fine for 3–6 min clips.

6. **v0.5 — speech_emotion** (done Aug 2026): dimensional vocal affect
   (Wagner et al. 2023 TPAMI model; CCC .745 arousal / .634 dominance /
   .638 valence on MSP-Podcast — the released 12-layer prune matches the
   full model). Checkpoint decisions: audeering model over emotion2vec
   (categorical would duplicate word2psy's GoEmotions) and over training
   our own probe (MSP-Podcast needs a data agreement, unlike DEAM);
   native ~0–1 scale kept; dominance included; no per-segment transcript
   summary; psyquilt gained a separate 3-d `speech_emotion` profile (no
   collision — psyquilt matches `speech_prob` exactly, there is no
   `speech_*` pattern), NOT folded into `acoustic` (the music_emotion
   precedent); family-wide naming revisit deferred until models settle.
   Tests: 62 offline (+4: windowing + VAD gating, which run offline since
   Silero ships with faster-whisper) + 1 behind `weights`. Validation
   (Aug 2026):
   - *CREMA-D* (16 clips, 4 actors × ANG/HAP/SAD/NEU, same sentence —
     lexical content controlled): arousal ANG .755 / HAP .608 ≫ NEU .295
     / SAD .266; valence HAP > ANG in 4/4 actors, arousal ANG > SAD in
     4/4; dominance peaks on ANG (.742), as theory says.
   - *rate manipulation*: same sentence at `say -r 110` vs `-r 320` →
     arousal .394 → .637.
   - *gating*: 20 s music clip → 40/40 windows NaN; dense dialogue →
     0/29 gated.
   - *per-speaker affect*: frames grouped by diarized turns give
     per-speaker VAD means on the dialogue clip (the diarize+affect
     combo recipe, now in README).
   - *psyquilt*: combined frames CSV detects `speech_emotion` as its own
     3-d profile; `acoustic` stays 5-d (registry change committed in
     psyquilt, 53 tests pass).
   - *runtime*: 4.4 s for the 14 s clip on MPS (~3× real time).

7. **v0.6 — egemaps** (done Aug 2026): the interpretable-prosody
   complement to speech_emotion (Eyben et al. 2016 eGeMAPSv02 LLDs).
   Checkpoint decisions (Ben, Aug 2026): eGeMAPS before verbatim mode;
   LLD level + `Grid.average` only, never openSMILE functionals (Grid
   is our windowing layer); VAD-gate the voiced set, general set
   ungated; sanitized `egemaps_*` naming; own 25-d psyquilt profile
   (not folded into `acoustic`); `opensmile` in both the `[egemaps]`
   extra and dev deps (offline + weightless, unlike the other extras).
   Also fixed pyproject `version` drift (was stuck at 0.3.1 while
   `__init__` advanced). Research notes: no OSI-licensed eGeMAPS
   reimplementation exists (openSMILE is the reference); Parselmouth
   is GPL-3 with Praat frozen at 6.1.38 — worse for us, not better;
   nkululeko/senselab ship opensmile as a *required* dep of
   MIT/Apache packages, so our optional extra is conservative; the
   PyPI "MIT" classifier on opensmile is a metadata bug. Tests: 69
   offline (+7: mapping sanity, sine f0 ground truth, unvoiced
   sentinel, amplitude tracking, real-VAD gating, pipeline/sidecar,
   `_is_embedding`). Validation (Aug 2026, synthetic stimuli):
   - *ground truth*: 220 Hz sine → `egemaps_f0_semitone` 35.98 vs
     12·log2(220/27.5) = 36.00; jitter < .02, HNR > 10 dB.
   - *speech*: same-sentence `say` male/female clips → median f0
     111/170 Hz (pyin: 110/172 — the v0.3.1 voice_gender anchors);
     median |f0 disagreement| vs pyin .31/.20 semitones on voiced
     frames; `egemaps_loudness` vs `loudness_rms` r = .89/.94;
     formants shift up female vs male (F1 665 vs 602, F2 1714 vs
     1642 Hz), jitter/shimmer/HNR in plausible speech ranges.
   - *gating*: 20 s 120-BPM melody+percussion clip (max speech_prob
     .15) → all 15 voiced columns NaN in all 40 windows, all 10
     general columns finite.
   - *psyquilt*: `egemaps` detected as its own 25-d profile,
     `acoustic` stays 5-d (registry change in psyquilt, 53 tests
     pass).
   - *runtime*: 1.4 s for a 5 s clip on CPU including openSMILE+VAD
     init — negligible next to the torch models.

8. **v0.7 — verbatim transcription** (done Aug 2026): `--verbatim` on
   `transcribe`, swapping the checkpoint for the official CT2
   conversion of CrisperWhisper **v1**
   (https://huggingface.co/nyrahealth/faster_CrisperWhisper —
   CC-BY-NC-4.0, English+German only, enforced at init). Checkpoint
   decision: Ben chose v1-CT2 over the July 2026 **CrisperWhisper
   2.0** (Interspeech 2026, https://arxiv.org/abs/2607.18934) — 2.0
   is better (multilingual, controllable verbatim/intended modes,
   29.6 ms word-boundary error, disfluency F1 87.8 vs v1's 64.8) but
   has **no faster-whisper path**: its own pip package with a
   forked-CTranslate2 NVIDIA-only fast path or an eager-attention
   transformers backend with no MPS branch, unverified transformers-5
   compat, gated custom non-commercial license (org moved to
   `nyralabs`). Documented upgrade path; revisit when its
   Mac/transformers-5 story is verifiable. Also rejected: post-hoc
   disfluency classifiers (vanilla Whisper deletes fillers before any
   classifier sees them), whisper-timestamped's `[*]` markers
   (position-only), Parakeet/Canary (no verbatim support). Design
   details in Architecture (vad_filter off / degenerate-loop retry /
   separator stripping — all empirically forced). Sub-question
   resolutions: tags stay verbatim in `text` + an `is_filler` words
   column; fillers excluded from recall matching but kept in
   speech_timing; `--verbatim` errors on a custom `--whisper-model`
   or non-en/de `--language`. Tests: 79 offline (+10: checkpoint
   resolution, CLI validation, filler regex, separator cleaning,
   max_word_run, scripted retry/no-retry/junk-segment fakes, recall
   exclusion) + 1 behind `weights` (~3 GB checkpoint). Validation
   (Aug 2026, synthetic `say` stimuli):
   - *fillers clip* ("So, um, I I think the, uh…"): verbatim output
     `So [UM] I I think the [UH] the meeting should [UM] start now
     because the the schedule is [UH] very tight.` — 4/4 fillers
     tagged, 2/2 repetitions kept. (Caveat: `say` articulates fillers
     so cleanly that large-v3 also kept them, as plain "um"/"uh" —
     the deletion problem this mode exists for shows on real
     conversational speech, per the paper's AMI numbers.)
   - *the stochastic loop*: ~half of unpenalized decodes of that clip
     degenerated into `,I`×112 (temperature-fallback lottery seeded
     by the genuine "I I"; language pinning irrelevant).
     `repetition_penalty=1.1` cured it in 4/4 runs ~2.5× faster — but
     also deleted the genuine distant "apple … apple" repeat in the
     recall clip, hence fallback-only. End-to-end: the retry fired on
     the fillers clip and produced the perfect transcript;
     `degenerate_retry` recorded in the sidecar.
   - *recall*: "apple um banana uh guitar apple" + 4-item pool →
     fillers excluded (n_intrusions = 1: guitar), the repeated apple
     matched + flagged repetition, IRT 3.22 s correctly spanning the
     intervening filler and intrusion.
   - *word2psy*: verbatim transcript → sentiment chunks with
     onset/offset/asr_confidence passthrough intact; `[UM]` tokens
     tokenize harmlessly.
   - *runtime*: ~18× real time on CPU per pass (~14× slower than
     large-v3 — token volume from the no-space tokenizer; greedy
     doesn't help), doubled when the degenerate retry fires. Budget
     minutes per clip. Trailing junk segments carry asr_confidence
     .33–.39 vs .90 for real speech — the documented filter handle.

9. **v0.8.0 — timbre** (done Aug 2026; first of the endgame trio from
   the Aug 2026 survey: CANLab scoping review, Giordano et al. 2023
   Nat Neurosci, pliers): MFCCs 1–13 (c0 excluded — window log-energy
   is loudness_db's job, and encoding models conventionally drop it),
   7-band per-octave spectral contrast, spectral flatness. Frontend
   constants match `spectral.py` so native frames align;
   `egemaps_mfcc1–4` coexist by design (different mel frontend — the
   loudness_rms/egemaps_loudness precedent). 21 columns, own psyquilt
   profile. Tests: 86 offline (+7). Validation (Aug 2026, synthetic,
   Linux container):
   - *flatness*: white noise .563 vs the closed-form periodogram
     expectation exp(−γ) = .5615 (within 0.3%); 440 Hz sine < 1e-3;
     silence 1.0 (floored-flat by convention).
   - *contrast*: tone's 400–800 Hz band 54.1 dB-scale depth vs 11.9 on
     white noise (4.5×), and the tone's argmax band is the band holding
     the tone.
   - *MFCC*: std < 1e-3 across windows on a stationary tone (1e-4
     measured); tone-vs-noise MFCC profile L2 distance ≫ 10.

10. **v0.9.0 — psychoacoustic** (done Aug 2026): Zwicker metrics via
    MoSQITo 1.2.1 (verified Apache-2.0; deps numpy/scipy/pyuff, 1.4 MB;
    matplotlib added to core deps only because mosqito imports it at
    module load — upstream bug, drop when fixed). Checkpoint findings:
    **MoSQITo has no fluctuation strength** (1.2.1 and master checked)
    → `psychoacoustic_fluctuation` is a native Fastl-style estimate
    (Ben's call, option B), honestly labeled non-standardized; design
    details in Architecture. Validation (Aug 2026, synthetic, Linux
    container, all now test asserts):
    - *roughness*: 1 kHz carrier, 100% AM — 1.355 asper at 70 Hz vs
      .271 at 20 Hz / .302 at 200 Hz (the Fastl peak, >4x).
    - *sharpness*: 4.33 acum on 4 kHz-high-passed noise vs .83 on
      1 kHz-low-passed (equal RMS); chained `sharpness_din_from_loudness`
      bit-identical to `sharpness_din_tv` (max |diff| = 0.0).
    - *loudness*: monotone 2.14/8.57/22.84 sone at amp .01/.1/.5; ISO
      532-1 post-offset decay visible and respected by the tests.
    - *fluctuation*: raw estimate 10.59 on Fastl's reference (4 Hz) vs
      3.13 at 0.5 Hz, 4.28 at 16 Hz, 0.63 at 70 Hz — the canonical
      contour, complementary to roughness; steady tone exactly 0; AM
      noise 22.2 > AM tone 10.6 (Fastl's published ordering). The L1→L2
      modulation-bin fix mattered: L1 read ~10 "vacil" on a mere
      amplitude step (broadband transient over-counting).
    - *runtime*: ~2.5x real time for loudness_zwtv on this CPU (~3x
      total with roughness); offline suite grows ~32 s (module-scoped
      fixtures share every mosqito call).

### Next (approved Aug 2026)

- **v0.10 — endgame trio finale**: `sound_events` (zero-shot CLAP
  prompt bank, 16 categories, raw-cosine scoring; fall back to BEATs
  (MIT) if zero-shot underperforms Ben's local validation). After this
  the registry is considered complete pending real-user demand.

### Explicitly deferred (do not build without discussion)

- **Temporal music-emotion model** — sequence model over CLAP embeddings
  for within-excerpt affect dynamics (see v0.3 finding above); only
  worth it with a use case in hand, given DEAM's within-song ceiling.
- ~~Word-timestamp refinement~~ — **descoped, do not build** (Ben, Aug
  2026). The only users needing sub-50 ms onsets (vocalization-locked
  EEG/iEEG, single-trial RT) are doing bespoke work that should
  hand-verify alignments in MFA directly; a built-in "refined" flag
  would invite trusting the pipeline for precision work it shouldn't
  carry. The family's contract is wide-sweep feature tables, honestly
  caveated — the README's word-timestamp caveat (N.B. above) stays as
  the permanent answer, pointing precision users out of scope.
- ~~Verbatim/disfluency mode~~ — **shipped in v0.7** (see roadmap).
- ~~Speaker diarization~~ — **shipped in v0.4** (see roadmap). The
  v0.3.1 `median_f0`/`voice_gender` columns remain as the no-dep
  fallback. Negative finding worth remembering: zero-shot CLAP against
  "a man/woman speaking" captions performed at chance (52%) for
  per-window gender — 10 s windows straddle dialogue turns and CLAP
  encodes speaker attributes weakly; don't reach for it for speaker
  tasks.
- ~~eGeMAPS prosodic features~~ — **shipped in v0.6** (see roadmap).
  Caveat to carry into any speech-affect docs: the wav2vec2 model's
  valence partly encodes implicit linguistic content (Wagner et al.
  2023) — not a pure prosody signal.
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
- **wav2vec2 layer-embedding export** — intermediate-layer speech
  embeddings as encoding-model features (the dominant 2025–26
  naturalistic-fMRI methods trend; nearly free since speech_emotion
  already runs the forward pass). Wait for a user with an encoding
  model in hand.
- **Spectrotemporal modulation energy** — the canonical
  auditory-cortex model (Santoro 2014 lineage), the most
  neuroscience-native feature family we lack, but no maintained Python
  tooling exists; would be bespoke scipy work. Watch.
- **Ruled out in the Aug 2026 survey** (don't re-research): Meta
  Audiobox-Aesthetics (built for filtering generative audio, no
  psychology uptake); timbral_models (unmaintained); dedicated
  laughter detectors (all stale or research-only weights — laughter is
  an AudioSet class, `sound_events` covers it); inaSpeechSegmenter for
  music presence (TensorFlow dep clashes with the torch stack —
  `sound_events` covers it too).

Prior art to consult before expanding the registry: `pliers` (Yarkoni lab
multimodal extraction), studyforrest movie annotations, CANLab's
narrative-annotation scoping review
(https://github.com/canlab/narrative_feature_annotations).
