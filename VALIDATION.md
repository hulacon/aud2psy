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

> **Protocol constraint — do NOT use macOS `say` for any CLAP-based
> category** (established 11 Aug 2026, below). Synthetic TTS is
> out-of-distribution for an audio–text model trained on natural
> recordings, and it produces *inverted* results: on the two-voice
> `say` dialogue clip, `sound_events_speech` read **−0.074 and ranked
> 6/16 during speech** — every column shifts down together, so the
> section reads as a global offset rather than category evidence.
> Applying the drop rule to that evidence would have deleted a
> category that works. The same caution applies to the `say` clips
> suggested elsewhere in this checklist and, by extension, to CLAP
> retrieval checks. Human-sound categories need **real recordings**.

Suggested stimuli from the usual validation set: the 20 s
melody+percussion clip (music tops, speech low), CREMA-D ANG clips
(shouting rises vs NEU), any sitcom/laugh-track excerpt (laughter),
street-recording or trailer excerpts for vehicle/crowd/footsteps/
animals/crying. Pass = the category clearly rises on clips containing
the event relative to clips without it (within-column comparison — the
scoring is raw cosine, cross-column rank is ordinal at best).

- [x] speech      — clip used: real mic recording (38 s, Ben, 11 Aug
      2026: ~11 s speech → brief silence → ~25 s music from a phone)
      verdict: **PASS**
- [ ] singing     — clip used: ............ verdict: ....
- [ ] laughter    — clip used: ............ verdict: ....
- [ ] crying      — clip used: ............ verdict: ....
- [ ] shouting    — clip used: ............ verdict: ....
- [ ] crowd       — clip used: ............ verdict: ....
- [ ] footsteps   — clip used: ............ verdict: ....
- [ ] vehicle     — clip used: ............ verdict: ....
- [ ] animals     — clip used: ............ verdict: ....

### `speech` — PASS (11 Aug 2026, real mic recording)

Stimulus: 37.8 s phone recording, three sections by construction —
speech ~0–11 s, brief silence, music ~15–38 s. Section cores below
exclude the boundaries, since CLAP's 10 s context window smears them.

| | speech core (0–10 s) | music core (18–38 s) | Δ |
|---|---|---|---|
| `sound_events_speech` | **+0.174** | −0.031 | **+0.205** |
| `sound_events_music` | +0.201 | **+0.384** | −0.182 |

- **r(`speech_prob`, `sound_events_speech`) = +0.78** over 76 windows;
  r for `sound_events_music` = **−0.87**. Both signed correctly.
- `speech` shows the largest positive Δ of all 16 categories — no
  other column discriminates the speech section as well. In the music
  core it falls to rank 14/16.
- Rows are independent: 58 unique embedding rows / 76 windows.

Two caveats worth carrying forward. First, **cross-column rank fails
exactly as documented**: `music` outranks `speech` (+0.201 vs +0.174)
*during the speech section*, so a per-window argmax reports "music"
over clear speech. Within-column contrast is the only trustworthy
read — as the sidecar already states. Second, a clip shorter than
CLAP's 10 s window yields **one observation, not N**: a 4.3 s pilot
recording produced byte-identical `sound_events` rows across all 9
grid windows (`speech` did rank 1/16 there, margin +0.031 over
`gunshot_explosion`, but that is n=1). Validation clips should run
**≥30 s with content that changes over time**.

**Drop rule** for any failure: delete the category from `PROMPT_BANK`
in `src/aud2psy/models/sound_events.py` — under raw-cosine scoring
nothing else changes. Remember to also update: the `len(PROMPT_BANK)
== 16` assert in `tests/test_sound_events.py`, the README models-table
row and sound_events section, and the CLAUDE.md architecture bullet.
If several categories fail broadly, the recorded fallback is BEATs
(MIT) instead of zero-shot CLAP.

## 3. Embedding cache with real weights (v0.10.1)

- [x] `aud2psy clap music_emotion sound_events <clip> -o /tmp/x.csv`
  then check `runtime_sec` per model in the sidecar: whichever
  clap-family model runs first pays the forward pass; the other two
  should be near-instant. (Offline wiring is already tested; this is
  the belt-and-braces check with real weights.)

**PASS** (11 Aug 2026, M-series MPS, real weights; 37.8 s clip →
76 windows). Both orderings run, plus a single-model baseline:

| run | models in requested order | `runtime_sec` |
|---|---|---|
| A | clap, music_emotion, sound_events | **7.32**, 2.28, 2.40 |
| B | sound_events, music_emotion, clap | **7.91**, 2.22, 2.33 |
| C | sound_events alone | 7.54 |

Whichever clap-family model runs **first** pays the forward pass
(7.3–7.9 s, matching the 7.54 s standalone baseline); each subsequent
one costs ~2.2–2.4 s. That residual is per-model **weight loading**,
not embedding — exactly as the design states ("weights still load per
model"), so "near-instant" in the checklist above meant near-instant
*embedding*, not a zero-cost model. All three together cost 12.0 s
instead of ~22.6 s (3 × baseline).

Correctness, which matters more than the timing: outputs are
**bit-identical** regardless of ordering or cache use — max|diff| = 0
across all 530 columns for A vs B (clap 512-d, sound_events 16,
music_emotion 2), and 0 for the cached `sound_events` (run B) vs the
uncached single-model run (run C).

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
