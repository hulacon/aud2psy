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
| `scores_transcript.csv` | one row per Whisper segment: `text`, `onset`, `offset`, `asr_confidence`, `no_speech_prob` — plus `median_f0` and coarse `voice_gender` (M/F, 150 Hz f0 boundary) when the `pitch` model also runs |
| `scores_transcript_words.csv` | word-level timestamps |
| `scores_beats.csv` | beat/downbeat events (with the `beats` model) |
| `scores_speakers.csv` | speaker turns (with the `diarize` model): `turn_idx`, `speaker`, `onset`, `offset` |
| `scores.meta.json` | provenance sidecar |

A wordless clip produces a zero-row transcript and `n_speech_segments: 0`
in the sidecar — an explicit result, not an error.

`voice_gender` is a deliberately coarse cue (per-segment median pyin f0
against the classic 150 Hz boundary): it cannot separate two same-gender
speakers, and children's voices sit above both adult ranges. For true
speaker-identity tracking, use the `diarize` model (below).

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
| `timbre` | frame | `timbre_mfcc_01`–`13` (MFCC coefficients 1–13; c0 excluded — that's `loudness_db`'s job), `timbre_contrast_01`–`07` (per-octave spectral peak-valley contrast), `timbre_flatness` (Wiener entropy: ~1 noise, ~0 tones) — the standard encoding-model timbre regressors |
| `psychoacoustic` | frame | `psychoacoustic_loudness` (sone, ISO 532-1 time-varying Zwicker), `psychoacoustic_sharpness` (acum, DIN 45692; NaN on silence), `psychoacoustic_roughness` (asper, Daniel & Weber), `psychoacoustic_fluctuation` (Fastl-style estimate, see below) via [MoSQITo](https://github.com/Eomys/MoSQITo). Absolute values assume digital RMS 1.0 = 94 dB SPL (files carry no calibration) — relative time courses are the meaningful output. ~3× real time on CPU |
| `speech` | frame | `speech_prob` (Silero VAD) |
| `clap` | frame | `clap_000`…`clap_511`: LAION-CLAP audio embeddings in a shared space with word2psy's `clap_text` (10 s windows; `--clap-model` to change checkpoint) |
| `music_emotion` | frame | `music_emotion_valence`, `music_emotion_arousal` in [−1, 1]: DEAM-trained probe on CLAP embeddings. Discriminates affective levels between clips/sections (held-out song-level r = .71/.84); not a beat-to-beat tracker |
| `sound_events` | frame | `sound_events_speech`, `_music`, `_singing`, `_laughter`, `_crying`, `_shouting`, `_crowd`, `_applause`, `_footsteps`, `_vehicle`, `_water`, `_wind`, `_animals`, `_gunshot_explosion`, `_siren_alarm`, `_thunder`: zero-shot cosine scores of each 10 s window against a text prompt bank in the CLAP space (see below) |
| `speech_emotion` | frame | `speech_emotion_valence`, `_arousal`, `_dominance` (~0–1): vocal affect from [audeering's wav2vec2 MSP-Podcast model](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) in 4 s windows; NaN where the window isn't speech (Silero-gated). Weights are CC-BY-NC-SA (research use) |
| `egemaps` | frame | the 25 [eGeMAPS](https://audeering.github.io/opensmile/) v02 low-level descriptors via openSMILE: `egemaps_loudness`, spectral balance (`_alpha_ratio`, `_hammarberg`, `_slope_0_500`, `_slope_500_1500`, `_flux`), `_mfcc1`–`4`, and a voiced set (`_f0_semitone`, `_jitter`, `_shimmer`, `_hnr`, `_h1_h2`, `_h1_a3`, formant `_f1/f2/f3_freq/_bw/_amp`) that is NaN off-speech (Silero-gated); needs `pip install "aud2psy[egemaps]"` |
| `beats` | events | beat/downbeat table (`time`, `is_downbeat`) via [beat_this](https://github.com/CPJKU/beat_this); needs `pip install "aud2psy[beats]"` |
| `diarize` | events | speaker turn table via [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1); needs `pip install "aud2psy[diarize]"` + a HuggingFace token (see below) |
| `transcribe` | segment | time-stamped transcript export (faster-whisper, default `large-v3`; `--whisper-model` to change; `--verbatim` for disfluency-preserving CrisperWhisper mode, see below) |

## Speaker diarization

`diarize` labels who speaks when, using pyannote's open
[community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
pipeline (CC-BY-4.0). The weights are gated on HuggingFace — one-time
setup:

```bash
pip install "aud2psy[diarize]"
# 1. accept the conditions at the model page above (free)
# 2. authenticate: `huggingface-cli login`, or set HF_TOKEN
```

```bash
aud2psy diarize transcribe clip.mp4 -o scores.csv --num-speakers 2
```

This writes `scores_speakers.csv` (one row per speaker turn) and, when
`transcribe` runs in the same call, adds a `speaker` column to the
transcript (majority-overlap speaker per segment) and words table
(speaker at each word's midpoint) — so word2psy chunks inherit speaker
identity via passthrough columns. `--num-speakers` is optional; without
it the pipeline estimates the count. The sidecar records `n_speakers`,
turn counts, and per-speaker speech time. Inference runs on CPU.

## Vocal affect (speech_emotion)

`speech_emotion` scores *how* speech sounds — dimensional
arousal/dominance/valence from the audeering wav2vec2 model (CCC
.745/.634/.638 on MSP-Podcast; Wagner et al. 2023, TPAMI) — a different
affective signal than word2psy's text `emotion`, which scores what the
words say. Windows that aren't speech are NaN, so music and effects don't
produce phantom affect. Combine with `diarize` to get per-speaker affect
by grouping frame windows by speaker turn.

Two honest caveats from the source paper: the model's *valence* partly
reflects implicit linguistic content learned in fine-tuning (so it
overlaps text sentiment for English; arousal/dominance are the more
purely paralinguistic dimensions), and overlapping *human* non-speech
sound (crowds, babble) is its worst-case interference. The weights are
CC-BY-NC-SA 4.0 — free for research; commercial use needs a license from
audEERING.

## Interpretable prosody (egemaps)

Where `speech_emotion` gives model *judgments* of vocal affect, `egemaps`
gives the raw interpretable prosody and voice-quality measures those
judgments are usually explained with: the 25 eGeMAPSv02 low-level
descriptors (Eyben et al. 2016, IEEE TAC) computed by openSMILE — the
reference implementation — at its native ~10 ms hop, averaged onto the
frame grid.

```bash
pip install "aud2psy[egemaps]"
aud2psy egemaps clip.mp4 -o scores.csv
```

Two NaN conventions keep the columns honest. Pitch-dependent measures
(F0, jitter, shimmer, HNR, H1–H2, H1–A3) are NaN on unvoiced frames (the
`pitch_f0` convention). And the whole voiced set — including formants —
is NaN wherever a window isn't speech (Silero-gated, the `speech_emotion`
threshold): "jitter" measured on a violin line is not voice quality. The
general-audio columns (loudness, spectral balance, MFCCs, flux) are
computed everywhere. The sidecar records each column's original openSMILE
name.

Caveats: jitter and shimmer are sensitive to recording quality and
background sound even within speech; and openSMILE is free for academic
and other non-commercial research under audEERING's research license —
use inside commercial products needs a
[commercial license from audEERING](https://audeering.github.io/opensmile/about.html)
(the reason this is an opt-in extra).

## Psychoacoustics (psychoacoustic)

Zwicker sound-quality metrics — the classic music-fMRI regressors of the
Alluri/Toiviainen lineage, with roughness the one most often missing
from Python pipelines. Loudness (ISO 532-1), sharpness (DIN 45692), and
roughness (Daniel & Weber) are MoSQITo's reference implementations,
sampled onto the frame grid.

Two honesty notes. First, audio files carry no absolute calibration, so
samples are mapped as digital RMS 1.0 = 94 dB SPL (recorded in the
sidecar): sone/acum/asper *time courses and comparisons* are meaningful,
absolute values are not calibrated to playback level. Second,
`psychoacoustic_fluctuation` is a **native Fastl-style estimate, not a
standardized metric** — MoSQITo implements no fluctuation strength (no
maintained Python tool does). It measures 4 Hz-weighted modulation of
the Zwicker specific-loudness envelopes, calibrated so Fastl's reference
signal (1 kHz, 60 dB, 100% AM at 4 Hz) reads 1.0, and reproduces the
canonical orderings (peak at 4 Hz modulation, gone by 70 Hz where
roughness takes over; AM noise above AM tone). Treat it as a relative
regressor, not a vacil meter.

`psychoacoustic_sharpness` is NaN where a window's loudness is below
0.25 sone — sharpness is a spectral ratio, undefined on silence.

## Verbatim transcription (`--verbatim`)

Vanilla Whisper silently deletes fillers, repetitions, and false starts —
fine for content, wrong for fluency, hesitation, or pause research.
`--verbatim` swaps the transcription checkpoint for
[CrisperWhisper](https://arxiv.org/abs/2408.16589) (the official
CTranslate2 conversion of the large-v3 fine-tune):

```bash
aud2psy transcribe interview.mp4 --verbatim -o scores.csv
```

Fillers come out as bracketed tags in the text (`[UH]`, `[UM]`) and get
an `is_filler` column in the words table; repetitions and false starts
are kept. Filler tags are excluded from `--wordpool` recall matching (a
filler is not a recall attempt), but still count in the sidecar's
`speech_timing` — and `n_fillers` is recorded there too.

Decoding is accuracy-first: if a pass degenerates into a repetition loop
(a real failure mode this model has around genuine repeated words), the
clip is automatically re-decoded with a mild repetition penalty — the
penalty is never applied by default because it can suppress genuine
distant repeats, which recall research needs. The sidecar's
`degenerate_retry` field records whether the fallback fired.

Limits to know about: **English and German only** (the model's training
languages); roughly 10–15× slower than the default model on CPU (budget
minutes per clip, not seconds); word timestamps through the CTranslate2
conversion are ordinary faster-whisper grade, *not* the CrisperWhisper
paper's forced-alignment-level numbers (the retrained alignment heads
don't survive conversion — the README's word-timestamp caveat applies
unchanged); occasional one-word trailing segments with low
`asr_confidence` can appear at speech offsets. The weights are
CC-BY-NC-4.0 (research use). CrisperWhisper 2.0 (multilingual,
controllable verbatim/intended modes) currently has no faster-whisper
path and is the documented upgrade once it does.

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

## Sound-event tags (sound_events)

AudioSet-style scene/event regressors — laughter, music presence,
crowds, sirens, gunshots — with no extra weights: each category is a
small ensemble of text prompts embedded by the same CLAP checkpoint the
`clap` model uses, and every window is scored by cosine similarity.
The sidecar records the full prompt bank, so the CSV is
self-documenting.

Read the scores as *relative* tag strengths: compare a column over time
or between clips. Columns have different baselines (an artifact of
zero-shot prompting), so comparing *across* categories is ordinal at
best — and the values are similarities, not calibrated probabilities.
Two recorded limits: CLAP zero-shot performs at chance for speaker
attributes (the bank deliberately contains none), and categories should
be validated on stimuli like yours before interpretation —
`scripts/validate_sound_events.py` prints per-category separation on
synthetic target stimuli and profiles any real clips you pass it.

## Interactive dashboard (`viz browse`)

The word2psy/viz2psy dashboard, for audio: a single self-contained HTML
file with a model selector, timeseries / 2D-3D clustering / trajectory
views over the frame and transcript tables, and a click-to-open detail
viewer with a slider and prev/next browsing.

```bash
aud2psy --all clip.mp4 -o scores.csv
aud2psy viz browse scores.csv --open
```

Since the stimulus can't be pictured, it's played: the input clip is
embedded in the page (ffmpeg-encoded mono mp3, ~4 MB for a 6-minute
clip) behind play/pause controls. The overview gets a waveform strip
and a play button that follows the full stream with a moving time
cursor on the plots; clicking any point opens the detail view, whose
play button plays just that frame's hop-length window (with an
optional loop — half a second is short) or the transcript segment's
onset-offset span. Arrow keys step frames, space replays, and
scrubbing while playing keeps the audio following.

The audio comes from the input path recorded in the `.meta.json`
sidecar; pass `--audio` if the file has moved (or to add playback to
CSVs scored elsewhere). Without audio the dashboard still works,
minus playback. Plotly.js loads from CDN (the one non-offline bit,
same as the siblings).

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
