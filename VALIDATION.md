# Local validation checklist — endgame trio (v0.8.0–v0.10.1)

Status tracker for the validation that could NOT run in the
implementation container (no HuggingFace egress → no CLAP weights).
Everything not listed here was already validated with synthetic
ground-truth and is asserted in the offline suite (`pytest -q`,
104 tests): timbre closed-form anchors, all four psychoacoustic
anchors, sound_events scoring machinery, embedding-cache wiring.

Work through the boxes, fill in the numbers, then fold the results
into CLAUDE.md's roadmap entry 11 (which currently carries the
"no real-embedding numbers exist yet" caveat) and delete this file.

## 1. sound_events — synthetic target separation

- [ ] `pytest -m weights tests/test_sound_events.py` — asserts each
  synthesizable target column peaks on its own stimulus
  (water←rain, siren_alarm←siren, music←music, wind←wind,
  applause←applause, thunder←thunder, gunshot_explosion←gunshots).
- [ ] `python scripts/validate_sound_events.py` — prints the full
  stimulus × category cosine matrix + per-target rank/margin
  diagnostics ("OK"/"FAIL" per row; siren and alarm are both valid
  peaks for `siren_alarm`).

Reference magnitudes from the v0.2 CLAP retrieval validation (same
checkpoint): a good match ran ~.22–.33 cosine with next-best ~.07–.14.
Zero-shot event columns should look similar: target clearly above
controls (tone/noise/silence), margin comfortably positive. A target
that peaks on the wrong stimulus, or sits within noise of the
controls, fails.

| category (synthesizable) | target stim | target cosine | best non-target (which) | margin | verdict |
|---|---|---|---|---|---|
| water | rain | | | | |
| siren_alarm | siren (or alarm) | | | | |
| music | music | | | | |
| wind | wind | | | | |
| applause | applause | | | | |
| thunder | thunder | | | | |
| gunshot_explosion | gunshots | | | | |

## 2. sound_events — human-sound categories (real clips)

Not synthesizable; judge on real audio:
`python scripts/validate_sound_events.py clip1.wav clip2.mp4 ...`
prints each clip's top-5 category profile.

Suggested stimuli from the usual validation set: the two-voice `say`
dialogue clip (speech should top, crowd/singing should not), the 20 s
melody+percussion clip (music tops, speech low), CREMA-D ANG clips
(shouting rises vs NEU), any sitcom/laugh-track excerpt (laughter),
street-recording or trailer excerpts for vehicle/crowd/footsteps/
animals/crying. Pass = the category clearly rises on clips containing
the event relative to clips without it (within-column comparison — the
scoring is raw cosine, cross-column rank is ordinal at best).

- [ ] speech      — clip used: ............ verdict: ....
- [ ] singing     — clip used: ............ verdict: ....
- [ ] laughter    — clip used: ............ verdict: ....
- [ ] crying      — clip used: ............ verdict: ....
- [ ] shouting    — clip used: ............ verdict: ....
- [ ] crowd       — clip used: ............ verdict: ....
- [ ] footsteps   — clip used: ............ verdict: ....
- [ ] vehicle     — clip used: ............ verdict: ....
- [ ] animals     — clip used: ............ verdict: ....

**Drop rule** for any failure: delete the category from `PROMPT_BANK`
in `src/aud2psy/models/sound_events.py` — under raw-cosine scoring
nothing else changes. Remember to also update: the `len(PROMPT_BANK)
== 16` assert in `tests/test_sound_events.py`, the README models-table
row and sound_events section, and the CLAUDE.md architecture bullet.
If several categories fail broadly, the recorded fallback is BEATs
(MIT) instead of zero-shot CLAP.

## 3. Embedding cache with real weights (v0.10.1)

- [ ] `aud2psy clap music_emotion sound_events <clip> -o /tmp/x.csv`
  then check `runtime_sec` per model in the sidecar: whichever
  clap-family model runs first pays the forward pass; the other two
  should be near-instant. (Offline wiring is already tested; this is
  the belt-and-braces check with real weights.)

## 4. psychoacoustic on real stimuli (optional)

The synthetic anchors are fully validated and asserted offline
(roughness 70 Hz peak, fluctuation 4 Hz peak, sharpness on filtered
noise, loudness monotone + ISO decay). Optional face-validity pass on
the usual clips:

- [ ] wordless music clip: roughness/fluctuation nonzero and varying;
  loudness tracks the crescendo (loudness_rms did r = .99).
- [ ] dialogue clip: sharpness NaN only during true silences;
  fluctuation elevated during speech (syllabic ~4 Hz modulation is
  exactly what it measures — expected, not a bug).
- [ ] runtime on M-series: expect ≲3× real time (was ~3× on the Linux
  container CPU).

## 5. psyquilt registry + detection

- [ ] add three standalone profiles in psyquilt's PROFILE_REGISTRY
  (mirroring the egemaps/speech_emotion entries; NOT folded into
  `acoustic`, which stays 5-d):
  - `timbre` — columns matching `timbre_*` (21-d)
  - `psychoacoustic` — columns matching `psychoacoustic_*` (4-d)
  - `sound_events` — columns matching `sound_events_*` (16-d; adjust
    if categories were dropped in §2)
- [ ] `psyquilt spaces <combined_frames.csv>` detects the three new
  profiles; existing profiles unchanged; psyquilt tests pass.

## 6. Close out

- [ ] transfer the numbers above into CLAUDE.md roadmap entry 11 and
  remove its "no real-embedding numbers exist yet" caveat.
- [ ] delete this file.
