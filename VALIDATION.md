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

- [x] `pytest -m weights tests/test_sound_events.py` — asserts each
  synthesizable target column peaks on its own stimulus
  (water←rain, siren_alarm←siren, music←music, wind←wind,
  applause←applause, thunder←thunder, gunshot_explosion←gunshots).
  **PASSES** post-gate (11 Aug 2026). Pre-gate it failed on `music`;
  the cause was the silence artifact, not the category — see below.
  The `gunshots` pair is commented out of the assertion pending a
  real-clip re-probe (Ben's call, 11 Aug 2026).
- [x] `python scripts/validate_sound_events.py` — prints the full
  stimulus × category cosine matrix + per-target rank/margin
  diagnostics ("OK"/"FAIL" per row; siren and alarm are both valid
  peaks for `siren_alarm`).

Reference magnitudes from the v0.2 CLAP retrieval validation (same
checkpoint): a good match ran ~.22–.33 cosine with next-best ~.07–.14.
Zero-shot event columns should look similar: target clearly above
controls (tone/noise/silence), margin comfortably positive. A target
that peaks on the wrong stimulus, or sits within noise of the
controls, fails.

Run 11 Aug 2026 (M-series MPS, `laion/larger_clap_music_and_speech`,
10 s synthetic stimuli), **after** the silence gate shipped the same
day. **6 of 7 pass**; the silence stimulus is now gated to NaN and
excluded from the ranking.

| category (synthesizable) | target stim | target cosine | best non-target (which) | margin | verdict |
|---|---|---|---|---|---|
| water | rain | 0.265 | 0.239 (noise) | +0.026 | OK |
| siren_alarm | siren (or alarm) | 0.257 | 0.184 (tone) | +0.073 | OK |
| music | music | 0.257 | 0.169 (tone) | +0.088 | OK |
| wind | wind | 0.345 | 0.147 (tone) | +0.198 | OK |
| applause | applause | 0.271 | 0.086 (tone) | +0.185 | OK |
| thunder | thunder | 0.320 | 0.298 (wind) | +0.022 | OK |
| gunshot_explosion | gunshots | 0.048 | **0.360 (thunder)** | −0.312 | **FAIL** |

Pre-gate, the same run scored 5 of 7: `music` peaked on **silence**
(0.294 vs 0.257 on the music stimulus), and silence was also the
runner-up that suppressed `applause`'s margin (+0.119 → +0.185) and
`gunshot_explosion`'s (0.309).

### The two failures — neither was a category to drop

**`music` ← silence: a digital-silence artifact affecting the whole
bank.** The silence column (literal `np.zeros`) is positive for all 16
categories and is the top stimulus for several: music 0.294,
gunshot_explosion 0.309, animals 0.208, singing 0.197, vehicle 0.196.
`music` itself is healthy — 0.257 on the music stimulus, the highest
of any *real* stimulus, and it tracked correctly on the real recording
in §2 (0.384 music core vs 0.201 speech core).

A level sweep (10 s stimuli, mean over grid windows) locates the
pathology and rules out the obvious fix:

| stimulus | RMS dBFS | cos to zeros-embedding | `music` | `gunshot_explosion` |
|---|---|---|---|---|
| digital zeros | −∞ | 1.000 | 0.294 | 0.309 |
| noise | −90 | 0.721 | 0.140 | 0.222 |
| noise | −80 | 0.639 | 0.091 | 0.180 |
| noise | −60 | 0.561 | 0.053 | 0.179 |
| noise | −30 | 0.346 | −0.040 | 0.069 |
| melody | −60 | 0.491 | 0.211 | 0.176 |
| melody | −20 | 0.327 | 0.200 | 0.162 |

**CLAP is level-invariant**: the same melody scores `music` 0.211 at
−60 dBFS and 0.200 at −20 dBFS — 0.011 of drift over 40 dB. So a
"quiet scene" gate at −50/−60 dBFS would NaN legitimately quiet
content that the model scores *correctly*, while the actual
degeneracy is confined to ≲ −80 dBFS.

**Fix shipped 11 Aug 2026** (`clap.SILENCE_DBFS = -80.0`,
`clap.silent_windows`): `sound_events` and `music_emotion` NaN every
column when the **10 s context window** — not the 0.5 s grid window,
since the context is what CLAP embeds — is below −80 dBFS RMS. The
clamping arithmetic is shared with `_embed_windows` via
`clap.window_starts` so the gate and the embedder cannot drift apart.
Sidecar records `silence_gate_dbfs` and `n_windows_silence_gated`.
This reverses `sound_events.py`'s previous docstring line "No
NaN/gating: cosine to a unit vector is defined everywhere, silence
included" (rewritten in place).

Scope decisions (Ben, 11 Aug 2026): `music_emotion` **is** gated — on
digital silence its probe returns valence −0.294 / arousal −0.245,
values that sit *inside* the range real music produces, so an ungated
reading is indistinguishable downstream (the `speech_emotion` VAD-gate
precedent). `clap` itself is **not** gated: its embedding is a valid
unit vector, silence is a real acoustic state for an embedding space
to represent, and NaN rows would punch holes in the 512-d matrix
psytwill consumes (whose NaN handling is unverified — psytwill is not
checked out on this machine).

Narrowness confirmed on real audio: the toy clip, whose quietest
section sits at −67 dBFS, gets `n_windows_silence_gated: 0` for both
models — the gate fires on degenerate input only.

**`gunshot_explosion` ← thunder: probably a stimulus problem.** It
scored 0.048 on synthetic gunshots while thunder hit 0.360 on the same
column. Synthetic gunshots are impulsive noise bursts, and
thunder-vs-explosion is a hard distinction even for humans; per the §2
protocol constraint, don't judge an impulsive category on a
synthesized probe.

- [ ] **OPEN**: re-probe `gunshot_explosion` on a real action-film
  excerpt, then decide (Ben, 11 Aug 2026: wait for the re-probe, do
  not drop). Until then the pair is commented out of the
  `weights`-marked assertion in `tests/test_sound_events.py` so the
  test suite is green on a known-open question rather than red on an
  unactioned one.

### Separate finding: edge clamping makes head/tail windows identical

`ClapModel._embed_windows` clamps the context start to
`[0, duration − 10 s]`, so **every grid center within 5 s of either
end shares one context window** — `sound_events`/`clap`/
`music_emotion` are constant across the first and last ~5 s of any
clip. Visible in the toy clip (11 identical rows from 25.25 s to the
end) and in §2 (58 unique embedding rows out of 76 windows). Correct
per the docstring's "edge-clamped", but worth stating in user-facing
docs: those rows are not independent observations, and a clip shorter
than 10 s yields exactly one.

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

## 5. psytwill registry + detection

- [ ] add three standalone profiles in psytwill's PROFILE_REGISTRY
  (mirroring the egemaps/speech_emotion entries; NOT folded into
  `acoustic`, which stays 5-d):
  - `timbre` — columns matching `timbre_*` (21-d)
  - `psychoacoustic` — columns matching `psychoacoustic_*` (4-d)
  - `sound_events` — columns matching `sound_events_*` (16-d; adjust
    if categories were dropped in §2)
- [ ] `psytwill spaces <combined_frames.csv>` detects the three new
  profiles; existing profiles unchanged; psytwill tests pass.

## 6. Close out

- [ ] transfer the numbers above into CLAUDE.md roadmap entry 11 and
  remove its "no real-embedding numbers exist yet" caveat.
- [ ] delete this file.
