"""Spectral shape: centroid, bandwidth, rolloff, flux, zero-crossing rate."""

from __future__ import annotations

import numpy as np

from .base import BaseModel

N_FFT = 2048
HOP_LENGTH = 512


class SpectralModel(BaseModel):
    name = "spectral"
    level = "frame"

    def extract(self, y: np.ndarray, sr: int, grid) -> dict[str, np.ndarray]:
        import librosa

        S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=HOP_LENGTH)

        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
        # Spectral flux: L2 norm of the half-wave-rectified frame-to-frame
        # magnitude change, 0 for the first frame.
        diff = np.maximum(0.0, np.diff(S, axis=1))
        flux = np.concatenate([[0.0], np.sqrt((diff**2).sum(axis=0))])
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
        zcr_times = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=HOP_LENGTH)

        return {
            "spectral_centroid": grid.average(times, centroid),
            "spectral_bandwidth": grid.average(times, bandwidth),
            "spectral_rolloff": grid.average(times, rolloff),
            "spectral_flux": grid.average(times, flux),
            "spectral_zcr": grid.average(zcr_times, zcr),
        }
