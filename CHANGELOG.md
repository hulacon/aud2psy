# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
