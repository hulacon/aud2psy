"""Pitch: pYIN fundamental frequency and voicing probability.

f0 is NaN wherever pYIN judges the frame unvoiced *and* in grid windows whose
native frames were all unvoiced — silence and noise produce NaN f0, not 0.
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel

FMIN = 65.0  # ~C2, below typical male f0
FMAX = 2093.0  # ~C7, above soprano range
FRAME_LENGTH = 2048


class PitchModel(BaseModel):
    name = "pitch"
    level = "frame"

    def extract(self, y: np.ndarray, sr: int, grid) -> dict[str, np.ndarray]:
        import librosa

        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=FMIN, fmax=FMAX, sr=sr, frame_length=FRAME_LENGTH
        )
        hop = FRAME_LENGTH // 4
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
        return {
            "pitch_f0": grid.average(times, f0),  # NaN-aware: unvoiced frames excluded
            "pitch_voiced_prob": grid.average(times, voiced_prob),
        }
