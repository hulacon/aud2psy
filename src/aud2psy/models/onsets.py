"""Event structure: onset strength, onset rate, and local tempo."""

from __future__ import annotations

import numpy as np

from .base import BaseModel

HOP_LENGTH = 512


class OnsetsModel(BaseModel):
    name = "onsets"
    level = "frame"

    def extract(self, y: np.ndarray, sr: int, grid) -> dict[str, np.ndarray]:
        import librosa

        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        times = librosa.frames_to_time(np.arange(len(oenv)), sr=sr, hop_length=HOP_LENGTH)
        onset_times = librosa.onset.onset_detect(
            onset_envelope=oenv, sr=sr, hop_length=HOP_LENGTH, units="time"
        )
        tempo = librosa.feature.tempo(
            onset_envelope=oenv, sr=sr, hop_length=HOP_LENGTH, aggregate=None
        )
        tempo = np.asarray(tempo).ravel()
        if tempo.size == 1:  # librosa collapsed to a global estimate
            tempo = np.full_like(oenv, tempo[0])
        return {
            "onsets_strength": grid.average(times, oenv),
            "onsets_rate": grid.rate(onset_times),
            "onsets_tempo": grid.average(times, tempo),
        }
