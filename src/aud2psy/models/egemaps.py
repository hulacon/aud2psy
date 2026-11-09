"""eGeMAPS prosodic / voice-quality low-level descriptors (openSMILE).

The interpretable-prosody complement to speech_emotion's model judgments:
the 25 eGeMAPSv02 LLDs (Eyben et al. 2016, IEEE TAC) — loudness, spectral
balance (alpha ratio, Hammarberg index, slopes), MFCC 1-4, and the voiced
set (F0, jitter, shimmer, HNR, H1-H2, H1-A3, formant 1-3 frequency /
bandwidth / relative amplitude) — computed by openSMILE at its native
~10 ms hop and reduced onto the grid with ``Grid.average``.

openSMILE column names carry ``-`` and ``.`` and config-internal suffixes
(``slope0-500_sma3``, ``F0semitoneFrom27.5Hz_sma3nz``), so columns are
renamed to model-prefixed CSV-friendly keys; the raw mapping is recorded
in the sidecar. Two NaN conventions:

- openSMILE's pitch-synchronous columns (the ``_sma3nz`` set) emit 0 on
  unvoiced frames; exact zeros are converted to NaN before grid averaging
  (matching openSMILE's own "nz" functional convention and the pitch
  model's unvoiced-f0 NaN).
- The voiced set is additionally **speech-gated** per grid window (the
  speech_emotion precedent; ``aud2psy.speech_gate``): "jitter" measured on
  a violin line is not voice quality. The gate is diarized-speech coverage
  by default, or Silero VAD (``speech_gate="silero"``, the pre-0.20 gate).
  The general-audio columns (loudness, spectral balance, MFCCs, flux) stay
  ungated.

opensmile is an optional extra (``pip install "aud2psy[egemaps]"``): the
bundled openSMILE binaries are under audEERING's research-only license
(free for academic/non-commercial use; commercial products need a license
from audEERING) — documented in the README. Caveat for interpretation,
from Eyben et al.: LLD reliability differs by feature — F0/loudness are
robust; jitter/shimmer are sensitive to recording conditions and
background sound even within speech regions.
"""

from __future__ import annotations

import numpy as np

from ..speech_gate import DEFAULT_SPEECH_GATE, GATE_MIN_SPEECH, SpeechGated, check_gate
from .base import BaseModel, undefined, undefinable_trailing

FEATURE_SET = "eGeMAPSv02"

# openSMILE LLD name -> aud2psy column. Order is openSMILE's own.
RAW_TO_CLEAN = {
    "Loudness_sma3": "egemaps_loudness",
    "alphaRatio_sma3": "egemaps_alpha_ratio",
    "hammarbergIndex_sma3": "egemaps_hammarberg",
    "slope0-500_sma3": "egemaps_slope_0_500",
    "slope500-1500_sma3": "egemaps_slope_500_1500",
    "spectralFlux_sma3": "egemaps_flux",
    "mfcc1_sma3": "egemaps_mfcc1",
    "mfcc2_sma3": "egemaps_mfcc2",
    "mfcc3_sma3": "egemaps_mfcc3",
    "mfcc4_sma3": "egemaps_mfcc4",
    "F0semitoneFrom27.5Hz_sma3nz": "egemaps_f0_semitone",
    "jitterLocal_sma3nz": "egemaps_jitter",
    "shimmerLocaldB_sma3nz": "egemaps_shimmer",
    "HNRdBACF_sma3nz": "egemaps_hnr",
    "logRelF0-H1-H2_sma3nz": "egemaps_h1_h2",
    "logRelF0-H1-A3_sma3nz": "egemaps_h1_a3",
    "F1frequency_sma3nz": "egemaps_f1_freq",
    "F1bandwidth_sma3nz": "egemaps_f1_bw",
    "F1amplitudeLogRelF0_sma3nz": "egemaps_f1_amp",
    "F2frequency_sma3nz": "egemaps_f2_freq",
    "F2bandwidth_sma3nz": "egemaps_f2_bw",
    "F2amplitudeLogRelF0_sma3nz": "egemaps_f2_amp",
    "F3frequency_sma3nz": "egemaps_f3_freq",
    "F3bandwidth_sma3nz": "egemaps_f3_bw",
    "F3amplitudeLogRelF0_sma3nz": "egemaps_f3_amp",
}

# the pitch-synchronous set: 0-as-unvoiced sentinel + speech gate
VOICED_RAW = tuple(n for n in RAW_TO_CLEAN if n.endswith("_sma3nz"))


class EgemapsModel(SpeechGated, BaseModel):
    name = "egemaps"
    nulls = {
        **undefinable_trailing(*(c for n, c in RAW_TO_CLEAN.items() if n not in VOICED_RAW)),
        **{RAW_TO_CLEAN[n]: undefined(
            "every frame in the grid window is unvoiced (openSMILE nz zero), or the window's "
            "speech measure under the run's speech gate (sidecar `speech_gate`: diarized-turn "
            f"coverage, or mean Silero probability) is below GATE_MIN_SPEECH ({GATE_MIN_SPEECH})",
            trailing=True)
           for n in VOICED_RAW},
    }
    level = "frame"
    labels = {
        "default": {"interpretability": "nameable", "derivation": "formula",
                    "window_sec": "grid"},
        "columns": {
            **{f"egemaps_mfcc{i}": {
                "interpretability": "opaque",
                "LongName": f"MFCC {i} (eGeMAPS)"} for i in range(1, 5)},
            "egemaps_f0_semitone": {
                "LongName": "Fundamental frequency",
                "Description": "F0 on a semitone scale from 27.5 Hz (openSMILE)",
                "Units": "semitones"},
            "egemaps_hnr": {"LongName": "Harmonics-to-noise ratio", "Units": "dB"},
            "egemaps_shimmer": {"LongName": "Shimmer (local)", "Units": "dB"},
            "egemaps_jitter": {"LongName": "Jitter (local)"},
            **{f"egemaps_f{k}_freq": {"LongName": f"Formant {k} frequency", "Units": "Hz"}
               for k in (1, 2, 3)},
            **{f"egemaps_f{k}_bw": {"LongName": f"Formant {k} bandwidth", "Units": "Hz"}
               for k in (1, 2, 3)},
        },
    }
    # eGeMAPS's reference rate; also what Silero VAD wants, and openSMILE's
    # 10 ms hop lands on integer samples here (221-sample steps at 22050)
    input_sr = 16000

    def __init__(self, speech_gate: str = DEFAULT_SPEECH_GATE):
        self.speech_gate = check_gate(speech_gate)

    def load(self) -> None:
        try:
            import opensmile
        except ImportError as exc:
            raise ImportError(
                "The egemaps model needs the optional opensmile dependency: "
                'pip install "aud2psy[egemaps]"'
            ) from exc
        self.model = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        self.load_gate()

    def unload(self) -> None:
        for attr in ("model", "vad"):
            self.__dict__.pop(attr, None)

    def extract(self, y: np.ndarray, sr: int, grid) -> dict[str, np.ndarray]:
        import opensmile

        lld = self.model.process_signal(y.astype(np.float32), sr)
        starts = lld.index.get_level_values("start").total_seconds().to_numpy()
        ends = lld.index.get_level_values("end").total_seconds().to_numpy()
        times = (starts + ends) / 2
        # per-window gate at the grid's own resolution (LLDs are local,
        # unlike speech_emotion's 4 s context windows)
        gated = self.gate_mask(y, sr, grid.centers, grid.hop)

        out: dict[str, np.ndarray] = {}
        for raw, clean in RAW_TO_CLEAN.items():
            values = lld[raw].to_numpy(dtype=float)
            if raw in VOICED_RAW:
                values = np.where(values == 0.0, np.nan, values)
            reduced = grid.average(times, values)
            if raw in VOICED_RAW:
                reduced[gated] = np.nan
            out[clean] = reduced

        self.info_ = {
            "feature_set": FEATURE_SET,
            "opensmile_version": opensmile.__version__,
            **self.gate_info(gated, grid.hop),
            "gated_columns": [RAW_TO_CLEAN[n] for n in VOICED_RAW],
            "opensmile_names": {v: k for k, v in RAW_TO_CLEAN.items()},
        }
        return out
