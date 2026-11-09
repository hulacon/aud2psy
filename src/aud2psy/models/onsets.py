"""Event structure: onset strength, onset rate, and local tempo."""

from __future__ import annotations

import numpy as np

from .base import BaseModel, undefinable_trailing

HOP_LENGTH = 512
# librosa.feature.tempo autocorrelation window (s); librosa's default, named so the
# window label and the call share one constant
TEMPO_AC_SEC = 8.0


class OnsetsModel(BaseModel):
    name = "onsets"
    nulls = undefinable_trailing("onsets_strength", "onsets_tempo")  # onsets_rate counts: never NaN
    level = "frame"
    labels = {
        "default": {"interpretability": "nameable", "derivation": "formula",
                    "window_sec": "grid"},
        "columns": {
            "onsets_strength": {
                "LongName": "Onset strength",
                "Description": "librosa onset-strength envelope (log-mel spectral flux), window mean"},
            "onsets_rate": {
                "LongName": "Onset rate",
                "Description": "Detected onsets per second in the grid window",
                "Units": "Hz"},
            "onsets_tempo": {
                "window_sec": TEMPO_AC_SEC, "window_align": "centred",
                "LongName": "Local tempo",
                "Description": "librosa frame-wise tempo estimate from the onset envelope",
                "Units": "BPM"},
        },
    }

    def extract(self, y: np.ndarray, sr: int, grid) -> dict[str, np.ndarray]:
        import librosa

        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        times = librosa.frames_to_time(np.arange(len(oenv)), sr=sr, hop_length=HOP_LENGTH)
        onset_times = librosa.onset.onset_detect(
            onset_envelope=oenv, sr=sr, hop_length=HOP_LENGTH, units="time"
        )
        tempo = librosa.feature.tempo(
            onset_envelope=oenv, sr=sr, hop_length=HOP_LENGTH, ac_size=TEMPO_AC_SEC, aggregate=None
        )
        tempo = np.asarray(tempo).ravel()
        if tempo.size == 1:  # librosa collapsed to a global estimate
            tempo = np.full_like(oenv, tempo[0])
        return {
            "onsets_strength": grid.average(times, oenv),
            "onsets_rate": grid.rate(onset_times),
            "onsets_tempo": grid.average(times, tempo),
        }
