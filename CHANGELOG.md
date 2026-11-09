# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.22.0] - 2026-11-08

Minor release. No output values change, so nothing needs re-extracting:
`aud2psy sidecar refresh` brings existing sidecars to 1.4.

### Added

- **Contract B schema 1.4 (window labels):** every column of a frame-level
  model now declares, in its `labels`, how much audio around the row its value
  reads (constellation-contracts §4.1). `window_sec` is a number of seconds
  when the column reads more than the grid step, `"grid"` when the row reads
  only its own grid window, or `"event"` when the value depends on the whole
  extent of an input event (a diarised turn, a transcribed word) that can
  reach past the row. Numeric windows also carry `window_align`: `"centred"`
  (every window in aud2psy today; clamped at clip edges) or `"trailing"`.
  Context inside an upstream model (Silero's state, pyannote's segmentation,
  Whisper's 30 s) is not counted. Segment and event models declare no window.
  `schema_version` is now `"1.4"` (1.3 is viz2psy's video framing record).
  - Windowed columns beyond the ones already declared at model level:
    `tonal` (3 s, all columns), `rhythm_pulse_clarity` (3 s),
    `rhythm_beat_strength` (8.92 s: 384 onset frames at 22.05 kHz),
    `rhythm_novelty` (4 s kernel), `onsets_tempo` (8 s autocorrelation),
    `psychoacoustic_fluctuation` (2 s). `conversation_turn_duration`,
    `conversation_switch_rate`, `speech_rate_word_duration` and
    `speech_rate_pauses` are `"event"`.
- `metadata.stamp_windows`: `sidecar refresh` stamps the model-level
  `window_sec` (the 1.0 key) from the model class, so a refreshed sidecar
  carries it exactly when every column shares one numeric window (extraction
  already writes it from the running model). `tonal` now sets it
  (3.0); before 0.22.0 tonal sidecars claimed no window.

### Changed

- `onsets` passes `ac_size=TEMPO_AC_SEC` (8.0, librosa's default) and
  `rhythm` passes `win_length=PLP_WIN_FRAMES` (384, the literal it used), so
  the declared windows and the computation share one constant. Values are
  unchanged.
- `sidecar refresh` also stamps `models.*.window_sec` (recorded in
  `refreshed[].fields`).

## [0.21.0] - 2026-11-05

Minor release. No output values change, so nothing needs re-extracting:
`aud2psy sidecar refresh` brings existing sidecars to 1.2.

### Added

- **Contract B schema 1.2 (`labels`):** every model entry in the sidecar
  carries `labels`, `{"default": {...}, "columns": {...}}`, saying what each
  column is and how its value is computed (constellation-contracts §4.1). The
  default applies to every column; a `columns` entry replaces individual keys
  for that column (key-wise merge). Every column resolves to
  `interpretability` (`nameable` / `opaque`) and `derivation` (`formula`,
  `formula-on-model`, `trained-on-labels`, `probe`, `zero-shot`, `embedding`,
  `bookkeeping`); `derived_from` names the parent checkpoint exactly where the
  derivation is `formula-on-model`, `probe` or `zero-shot`. Optional
  `source_validation` and BIDS column keys (`LongName`, `Description`,
  `Units`, `Levels`). `schema_version` is now `"1.2"`.
  - Declared on the model class (`BaseModel.labels`; vocabularies and
    `resolve_label` in `aud2psy.models.base`) and stamped by
    `metadata.stamp_labels`, filtered to the columns the run emitted.
  - All 20 models are labelled. MFCCs (`egemaps_mfcc1`-`4`,
    `timbre_mfcc_01`-`13`) and the `clap` / `ebind_audio` embeddings are
    `opaque`; every other feature column is `nameable`. `sound_events` is
    `zero-shot` and `music_emotion` a `probe`, both `derived_from` the CLAP
    checkpoint; `conversation` and `speech_rate` are `formula-on-model` from
    the diarize pipeline and the Whisper checkpoint; `speech`,
    `speech_emotion`, `beats`, `diarize` and `transcribe` are
    `trained-on-labels`; the DSP models are `formula`. Row ordinals and
    decoder confidences (`turn_idx`, `chunk_idx`, `word_idx`,
    `transcribe_asr_confidence`, `transcribe_no_speech_prob`,
    `transcribe_probability`) are `bookkeeping` and `opaque` (they describe
    the extraction, not the stimulus; the family convention).
  - `conversation` and `speech_rate` declare a `parent_model`: when the
    parent ran in the same call, `derived_from` is the checkpoint it recorded
    (e.g. `transcribe --verbatim`), not the default.
  - `source_validation` is filled only where the repository records the
    numbers: `music_emotion` (song-level cross-validation, read from the
    probe's provenance JSON) and `speech_emotion` (CCC on MSP-Podcast, as
    the module docstring cites the model's paper).
- `aud2psy sidecar refresh` writes `labels` too (JSON only, as for `nulls`)
  and re-stamps them, so a label-only patch release reaches existing
  sidecars.
- `tests/test_labels.py`: the producer-duty contract test. Every column of
  every model's inventory resolves, every `columns` key is a real column,
  values come from the vocabularies, and `derived_from` is present exactly
  where required and names a checkpoint aud2psy records. The offline models'
  inventories are checked against a real run.

### Changed

- `timbre` passes `fmin=200` to `spectral_contrast` explicitly. That is
  librosa's default, so values are unchanged; the band edges in the labels
  now follow the code.

## [0.20.0] - 2026-11-05

Minor release with a **breaking default**: the speech gate on `egemaps`'
voiced set and on `speech_emotion` now reads diarized speaker turns, and a
call that runs either model needs a turn table.

### Changed

- **Speech gate: diarized-speech coverage replaces Silero VAD by default**
  (`aud2psy.speech_gate`; CLI `--speech-gate {diarize,silero}`, API
  `speech_gate=`). A window of a gated model is now speech when pyannote
  speaker turns cover at least `GATE_MIN_SPEECH` (0.25) of it: the grid
  window for `egemaps`, the 4 s context window for `speech_emotion`.
  Before, it was speech when its mean Silero speech probability reached the
  same 0.25. Silero under-calls speech that overlaps music or background
  noise, the usual case in film audio, so those windows were NaN; the
  diarize gate keeps more of them, at the cost of firing on some non-speech
  vocal sound. **Changes output values** of `egemaps`' 15 voiced columns
  and the 3 `speech_emotion` columns (which windows are NaN); nothing else.
  - The turns come from `diarize` in the same call, or from an existing
    `{stem}_speakers.csv` via `--speakers` (`speakers_csv=`), which now
    feeds the gate as well as `conversation`. A batch needs `diarize` in the
    same call.
  - **Breaking:** a gated model with neither now fails, and the error names
    the three fixes. `--speech-gate silero` restores the previous gate
    exactly. `--all` without pyannote installed skips `egemaps` and
    `speech_emotion` (with a note), as it already skipped `conversation`.
  - In a single-file call, `diarize` now runs before the frame models, so it
    comes first in the sidecar's `models` map.
  - Sidecar: both models record `speech_gate`, `speech_gate_measure`,
    `speech_gate_min`, `speech_gate_window_sec`, `n_gated_windows` and, for
    the diarize gate, `speech_gate_turns_source`. `egemaps`'
    `vad_gate_min_speech` key is replaced by `speech_gate_min`. The `nulls`
    entries for the gated columns name the gate generically.
  - `speech_fraction` and `GATE_MIN_SPEECH` moved to `aud2psy.speech_gate`.
    `aud2psy.models.speech_emotion` still re-exports both.
- The EBind checkpoint `encord-team/ebind-full` loads at a pinned hub revision (`482831d`, `DEFAULT_REVISION`) instead of whatever `main` is. That snapshot has been `main` since 2025-11-19, so features are unchanged. viz2psy, word2psy and aud2psy pin the same revision, so the shared space cannot split when the hub moves. A non-default checkpoint still loads from `main`. The `beats` extra pins `beat_this` to `b95c8ab`, and the `ebind` extra pins the `ebind` package to `7909701`: the builds the family has been running on.

## [0.19.0] - 2026-10-15

Minor release: several fixes below change written output values, though no
column is renamed.

### Changed

- **Python floor raised to 3.11; `scikit-learn>=1.8` declared.** The
  dashboard's MDS projection uses the scikit-learn 1.8 API
  (`MDS(metric_mds=...)`), and scikit-learn 1.8 requires Python 3.11. A 3.10
  install resolved scikit-learn 1.7, where that call is a `TypeError`.
  scikit-learn was already imported directly but only arrived transitively via
  librosa. It is now a declared dependency. CI runs 3.11 and 3.12.
- **CSV `time`/`onset`/`offset` columns keep up to 10 significant digits**
  (previously 6, the `%.6g` used for feature values). Before, at `--hop 0.25`
  every frame time past 1000 s was rounded to 10 ms (1000.125 → 1000.12), and
  at the default hop every time past 10 000 s lost precision; word onsets did
  too. Applies to every saved table and to the stdout frames CSV. Feature
  columns keep `%.6g`. **Changes output values** wherever a coordinate needed
  more than 6 significant digits.
- **A time exactly on a window edge goes to the correct window at non-default
  hops.** `Grid.average`/`window_max`/`rate` truncated `time / hop`, and
  `0.3 / 0.1 == 2.9999999999999996`, so an edge time landed one window early
  at hops that are not a power of two (0.1, 0.2, 0.3 ...). This hit
  round-valued word and turn onsets in `speech_rate` and `conversation`. The
  0.5 s default and other power-of-two hops (0.25, 0.125 ...) are unaffected:
  division by them is exact. **Changes output values** only for edge times at
  non-power-of-two hops.

### Fixed

- **`--all` succeeds without the `[ebind]` extra.** It scheduled `ebind_audio`
  unconditionally, so a default install crashed with `ModuleNotFoundError`
  after the earlier models (CLAP included) had already run, writing nothing.
  It is now skipped with a note, like `beats`/`diarize`/`egemaps`; the output
  has no `ebind_*` columns.
- **Batch inputs that would overwrite each other are refused up front.** Two
  inputs sharing a file name (`a/clip.wav`, `b/clip.wav`), or manifest
  `output` values that normalize to the same stem, made the second write over
  the first. Stems are now checked before any scoring. The existing
  `output`-escape check moved ahead of scoring too.
- **`beats` or `diarize` without `-o` is refused up front.** The stdout path
  prints the frames or transcript table; an events-only run has neither, and
  crashed on `None.to_csv` after the model had run.
- **Dashboard projection failures are no longer swallowed.** A broad
  `except Exception` turned any error in a projection (such as the
  scikit-learn 1.7 `TypeError` above) into a warning and a silently missing
  clustering view. Only degenerate-input errors (`ValueError`, `LinAlgError`)
  now skip a view.

## [0.18.1] - 2026-10-02

### Fixed

- **A one-row `--inputs-from` manifest is batched like any other.** It was
  routed to the single-input path, which read `-o DIR` as a file stem
  (writing `DIR_frames.csv` beside it) and ignored the row's `output` and
  `stimulus_id`. Any manifest now takes the batch path. A caller narrowing a
  campaign to one unit hit this, and its per-cell check reported every cell
  failed. The `output` escape check and the `--words` refusal now apply to a
  one-row manifest too (`--words` would otherwise have been dropped silently,
  since the batch path has no slot for it).

## [0.18.0] - 2026-09-23

### Added

- **Contract B schema 1.1 (`nulls`):** every model entry in the sidecar
  carries `nulls`, a map from each column the model can set to NaN to
  `{"means", "when"}` (constellation-contracts §4.1). Declared on the model
  class (`BaseModel.nulls`, `{}` = never null) and stamped by
  `metadata.stamp_nulls`, filtered to the columns the run emitted. 16 models
  declare 90 columns. 44 are `undefined` (content gates): pitch_f0, egemaps
  (15 voiced), tonal, psychoacoustic sharpness, speech_emotion, speech_rate,
  conversation, sound_events (16, CLAP silence gate), music_emotion (2, same
  gate), transcribe's `median_f0`. 46 are `undefinable`: every grid-reduced
  column is NaN on a trailing window that no native frame centre reaches (a
  clip ending a few ms into its last window; at most one row per clip), found
  by `sidecar refresh` on real corpora (`base.TRAILING_WINDOW`). Gated columns
  keep `undefined` and name the trailing case in `when`. `schema_version` is
  now `"1.1"`.
- `aud2psy sidecar refresh PATH... [--dry-run]`: rewrites existing sidecars
  to 1.1 in place (JSON only, never a CSV). Each model's columns are read back
  from the family's `output` tables, and a NaN in an undeclared column refuses
  that sidecar with nothing written. Idempotent; records a `refreshed` entry.
- `tests/test_nulls.py`: the producer-duty contract test, with a fixture per
  declared condition (silence, tone, noise, silence-then-tone, gapped turn and
  word tables, an unvoiced segment), asserting in both directions.

- `aud2psy viz browse` accepts multiple scores paths, or a directory of
  per-model `*_frames.csv` files, and merges them into one all-models
  dashboard (`aud2psy.viz.merge`). Frames on the spine's exact time grid
  merge exactly; offset grids merge to the nearest spine row within half
  the sampling step. Directory-collected tables that cannot merge 1:1
  (e.g. diarization segments) are skipped with a note.

## [0.17.0] - 2026-08-25

### Added

- **`speech_rate` model:** the conversation pattern applied to
  `transcribe` — three frame-level speaking-rate features derived from
  the word timestamps: `speech_rate_words` (word onsets/s),
  `_word_duration` (time-weighted mean active-word duration, NaN where
  no word is active), `_pauses` (within-utterance silent pauses/s; a gap
  between consecutive words counts when in [0.15, 2.0) s — shorter is
  articulatory, longer is an utterance boundary). Word source is either
  `transcribe` in the same call or an existing
  `{stem}_transcript_words.csv` via the new `--words` flag (`words_csv=`
  in `score_audio`). Analytic (`checkpoint` null); the sidecar records
  `words_source`, `derived_from`, `n_words`, and the pause-gap bounds.
  In a batch, `speech_rate` requires `transcribe` in the same call.
  Whisper word timings are fine for windowed rates, not onset-locked
  latencies (see the existing timing caveat).

## [0.16.0] - 2026-08-25

### Added

- **`conversation` model:** frame-level conversation-structure features
  derived from the `diarize` turn table — `conversation_n_speakers`,
  `_speech_fraction`, `_overlap_fraction`, `_turn_rate`, `_switch_rate`,
  `_turn_duration` (NaN in silence). It consumes turns rather than audio:
  run it together with `diarize`, or point it at an existing
  `{stem}_speakers.csv` via the new `--speakers` flag
  (`speakers_csv=` in `score_audio`), which needs neither the `[diarize]`
  extra nor an HF token. Everything derives from the *raw* turn table
  (the exclusive timeline is never persisted), so the in-run and
  from-CSV routes are bit-identical. The sidecar records the turn source
  (`turns_source`), `derived_from: diarize`, and `n_turns`; `checkpoint`
  is `null` (analytic) — speaker-label provenance stays with diarize's
  own sidecar entry. In a batch, `conversation` requires `diarize` in
  the same call (one CSV cannot map onto several inputs); with `--all`
  and no pyannote installed, `conversation` is skipped alongside
  `diarize` unless `--speakers` is given.

## [0.15.1] - 2026-08-23

### Fixed

- **`transcribe` declared only one of its two frames.** It is the only model
  here that writes both a segments table and a words table, and `pipeline.py`
  set `models.transcribe.columns` to `list(transcript_df.columns)` — the
  segments frame — at all four of its sites. The words frame's own columns
  were never declared, leaving `transcribe_probability` emitted but
  undeclared in **1,060** sidecars across the MMMData feature store.

  No data was wrong: the column is correctly prefixed, psytwill attributed it
  by prefix, and it is present in the aggregates. What was wrong is the
  sidecar's account of itself — and a consumer that trusts declarations to
  decide what a column *is* would have skipped it.

  `_transcribe_columns()` now declares the union of both frames, in emitted
  order. Two tests in `test_column_prefixes.py` cover it, including the
  no-words-frame case.

  Found by the mmmdata campaign gate (`stimfeat_campaign.py verify`) once it
  was changed to check emission against the sidecar in **both** directions.
  The previous gate inspected only columns the sidecar *declared*, so an
  emitted-but-undeclared column was invisible to it by construction — the
  same blind spot, mirrored, that let `is_downbeat` through in 0.15.0.
  Existing sidecars were backfilled in place by
  `mmmdata/scripts/patch_transcribe_sidecars.py` rather than re-extracted:
  re-running Whisper to correct a provenance field is ~3.5 GPU-h.

## [0.15.0] - 2026-08-23

### Changed

- **BREAKING: `beats` and `transcribe` columns now carry their model prefix**
  (§4.1). `is_downbeat` → `beats_is_downbeat`; `text`, `asr_confidence`,
  `no_speech_prob`, `probability` → `transcribe_*`. Unprefixed feature columns
  are invisible to a consumer's resolver, which drops them onto a null model
  and then collides them across models — `movies/audio/beats` had 8,392 rows
  attributed to nothing. All 17 models were scanned with psytwill's own
  resolver; only these two were non-compliant.
- **BREAKING: `segment_idx` → `chunk_idx`.** A transcript segment *is* a
  chunk, the same grouping level word2psy emits for text. One structural
  vocabulary across extractors means a consumer needs one key rather than one
  per extractor, and it meant no new reserved column had to enter Contract B.

### Added

- **`transcribe` gains `word_idx`.** Whisper can give two words identical
  start and end times, so timings do not identify a word row — that was the
  ~158 duplicate keys `movies/audio/transcript_words` refused on.
- `test_column_prefixes.py`, ported from word2psy, which reads the declared
  column lists and so needs no weights and no audio. word2psy had this test
  and aud2psy did not, which is the whole reason these defects survived.

## [0.14.0] - 2026-08-22

### Added

- **Model-major batch scoring** — `pipeline.score_audio_batch(paths, models)`
  and multi-input support in the CLI. `score_audio` loads and unloads every
  model per file, which is right when the file is long: the weights amortise
  over minutes of audio. It is pathological when the files are short. Measured
  on 0.54 s spoken words, the nine neural models each cost **44–65 s per file**
  and essentially all of it is the load — scoring 1,000 words that way is
  ~159 GPU-hours to analyse nine minutes of audio.

  `score_audio_batch` inverts the loops so each model is loaded once and run
  across every file. Peak memory stays one model at a time, the invariant
  `BaseModel.unload` exists to protect; what grows instead is the accumulated
  feature table, bounded by (n_files × n_frames × n_features).

  Measured speedup on three 0.54 s words, against per-file scoring:
  **12.5× for the clap family**, 9.2× for `speech`+`speech_emotion`, 1.2× for
  `transcribe`. The ratio grows with batch size, since the saved load is a
  constant divided over more files.

  Per-file results are **bit-identical** to `score_audio` — verified with
  `max_abs_diff = 0` on `clap`/`music_emotion`/`sound_events`,
  `speech`/`speech_emotion`, and `transcribe` — with one deliberate exception:
  the CLAP window-embedding cache is not shared across the clap-family models
  in batch mode. Holding one cache per file alive across three model passes to
  save forward passes on audio already in memory is worth it per-file and
  pointless against a load cost amortised a thousandfold. Output values are
  unaffected.

  Batching is best for many short files. Each file is decoded once per model
  rather than once per batch (bounded memory beats a decode cache here), so for
  movie-length audio the repeated decode can outweigh the saved loads — keep
  using `score_audio` per file there.

- **`--inputs-from CSV`** — a batch manifest with a `path` column plus optional
  `stimulus_id` and `output`. `--stimulus-id` applies one value to every row,
  so a manifest is the only way to batch inputs whose `stimulus_id` differs —
  which is the normal case when one file is one stimulus.

### Changed

- The CLI splits `MODEL... INPUT...` by registry membership rather than by
  position (leading tokens naming a model are models, the rest are files).
  Positional splitting cannot express more than one input. Single-input
  invocations are unaffected.
- Batched runs record `models.<name>.batched: true` in the sidecar. Their
  `runtime_sec` is extraction only, with the load amortised across the batch,
  so it is **not** comparable to an unbatched `runtime_sec` — the flag is what
  makes that legible rather than a mysterious 50× drop.

## [0.13.2] - 2026-08-20

### Added

- **`window_sec` in the sidecar** for models using the `window >> hop`
  pattern (Contract B §4.1). A frame model that scores a long context
  window centered on each grid midpoint produces rows that overlap
  heavily, and nothing in the written metadata said so: `frames` already
  recorded `hop_sec` and `time: window center`, but not how much audio each
  row actually saw. `BaseModel.window_sec` now declares it and the pipeline
  writes it to `models.<name>.window_sec`. Declared by `clap` (10 s),
  `sound_events` and `music_emotion` (10 s, inherited from `ClapModel`),
  `speech_emotion` (4 s), `ebind_audio` (2 s); left `None` by the twelve
  models whose rows see only their own `[k·hop, (k+1)·hop)` window.

  Set it only when *every* column of a model shares one context window. A
  window belonging to a single feature — `psychoacoustic`'s
  modulation-spectrum window, for instance — stays a per-feature detail and
  is deliberately not declared here.

  Why it matters: a consumer reading a frames CSV could not tell how much
  temporal smoothing was baked into it, which silently inflates apparent
  sample size and breaks naive permutation nulls on time-resolved features.

## [Unreleased]

### Known issues

- **`transcribe --verbatim` silently drops ~30 s chunks on long
  recordings** (open, reported 2026-08-20). The CTranslate2 CrisperWhisper
  path loses whole chunk-sized spans with no gap flag, no low
  `asr_confidence`, and no error — the words table is simply missing them.
  Measured on 3 subjects of real free-recall audio (~70–110 min each):
  7 / 33 / 9 gaps, versus 0–2 for the same audio through the default
  `large-v3` path. Does not reproduce on short clips, which is why the
  v0.7 validation missed it. Workaround: run both arms and diff their
  gaps. Documented in `transcribe.py`, the README, and roadmap entry 8.

## [0.13.1] - 2026-08-18

Housekeeping release for public use — documentation, packaging metadata, and
citations; no feature-output changes.

### Added

- `ebind` optional-dependency extra (the GitHub-only `ebind` package was
  previously undeclared).
- README row and cross-modal documentation for `ebind_audio` (absent from
  the README since its 0.13.0 release); the cross-modal section now covers
  both shared spaces (CLAP and EBind).
- README documentation of the `stimulus_id` column and the Contract-B
  sidecar fields (`schema_version`, per-model `checkpoint`).
- `CITATION.cff` for citing aud2psy itself.
- README "Related packages" section (viz2psy, word2psy, psytwill) and a
  "Citing" section with full references for every model.
- Install instructions for the `git+` form (the package is not on PyPI).

### Fixed

- `psyquilt` renamed to `psytwill` everywhere (README examples, source
  comments, VALIDATION.md, CLAUDE.md) — the downstream package was renamed
  and its CLI command is `psytwill`, so the old `psyquilt matrices …`
  example no longer resolved.
- License section now links the LICENSE file and summarizes the
  third-party weight licenses in one place.

## [0.13.0] - 2026-08-17

### Added

- `ebind_audio` model: 1024-d L2-normalized soundtrack embeddings from
  EBind's ImageBind-huge audio arm projected into the Perception Encoder
  space (checkpoint `encord-team/ebind-full`, revision-pinned). Shares
  one cross-modal space with viz2psy `ebind` and word2psy `ebind_text`.
  2 s context window per grid point (ImageBind's native clip length);
  columns are fixed-width 4-digit (`ebind_audio_0000..1023`). Scope is
  naturalistic soundtracks — the AudioSet-trained encoder carries no
  lexical signal for isolated spoken words (2026-08-17 mmmdata pilot).

### Fixed

- Sidecar embedding `pattern` now reflects the actual index width
  (`ebind_audio_{NNNN}`) instead of hard-coding `{NNN}`.

## [0.12.0] - 2026-08-10

### Added

- Sidecar (`.meta.json`): `schema_version` ("1.0"), `extractor`,
  `extractor_version`, and per-model `package_version` + **`checkpoint`**
  fields per the constellation Contract B extractor-output convention
  (mmmdata-agents `docs/constellation-contracts.md` §4.1). Checkpoint is the
  exact architecture+weights identifier (e.g.
  `laion/larger_clap_music_and_speech` for the clap family, the actual
  Whisper id for transcribe, `pyannote/speaker-diarization-community-1` for
  diarize); analytic/DSP models record `null`. Checkpoint identity backs the
  cross-modal `clap_text↔clap` space guarantee asserted by psytwill. Legacy
  `aud2psy_version` key retained for one deprecation cycle.
- `BaseModel.checkpoint` class attribute (None for analytic models); models
  that already record a `checkpoint` in `info_` (sound_events) keep that as
  the authoritative value.
- `aud2psy.metadata.get_model_version()`: per-model underlying-package
  version via `importlib.metadata` (the word2psy/viz2psy family pattern).
- **`stimulus_id` column** (first column of every written table): the input
  file's stem by default, `--stimulus-id` / `save_result(...,
  stimulus_id=...)` to override (`time`/`onset` disambiguate rows within
  the stimulus), per the §4.1 identity rules.

No feature columns were renamed in this release.
