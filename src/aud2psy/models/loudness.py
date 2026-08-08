"""Loudness: RMS energy and its dB equivalent."""

from __future__ import annotations

import numpy as np

from .base import BaseModel

FRAME_LENGTH = 2048
HOP_LENGTH = 512


class LoudnessModel(BaseModel):
    name = "loudness"
    level = "frame"

    def extract(self, y: np.ndarray, sr: int, grid) -> dict[str, np.ndarray]:
        import librosa

        rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH)
        rms_grid = grid.average(times, rms)
        # dB of the window-mean RMS (not the mean of native dB): full-scale
        # sine = ~-3 dB, digital silence floors at -100 dB via librosa's amin.
        db_grid = librosa.amplitude_to_db(np.nan_to_num(rms_grid, nan=0.0), ref=1.0)
        db_grid[np.isnan(rms_grid)] = np.nan
        return {"loudness_rms": rms_grid, "loudness_db": db_grid}
