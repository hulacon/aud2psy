# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
