"""Timbre: MFCCs 1-13, spectral contrast, spectral flatness.

The most common encoding-model regressors not already in the registry
(MFCCs above all — the workhorse timbre encoding in naturalistic-fMRI
work), plus the two librosa summary shapes that complete the
Alluri/Toiviainen timbre set: per-octave spectral contrast (peak-valley
depth; low on noise-like textures) and spectral flatness (Wiener
entropy; ~1 on white noise, ~0 on pure tones).

Design decisions:

- MFCC coefficient 0 is excluded: it is window log-energy, which is the
  loudness model's job (``loudness_db``), and encoding models
  conventionally drop it. Columns are ``timbre_mfcc_01``..``_13``.
- Frontend constants (``N_FFT``/``HOP_LENGTH`` at 22050 Hz) match
  spectral.py, so the two models' frames land on the same native times.
- ``egemaps_mfcc1``-``4`` coexist by design (different frontend:
  openSMILE's 26-band 20-8000 Hz mel vs librosa's 128-band) — the
  ``loudness_rms`` vs ``egemaps_loudness`` precedent.
- No gating, no NaN: all three features are defined everywhere,
  including silence (librosa floors log/ratio inputs with ``amin``).
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel

N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 13  # coefficients 1-13; c0 (log-energy) excluded
N_CONTRAST_BANDS = 6  # librosa default -> 7 rows (6 octave sub-bands + top)


class TimbreModel(BaseModel):
    name = "timbre"
    level = "frame"

    def extract(self, y: np.ndarray, sr: int, grid) -> dict[str, np.ndarray]:
        import librosa

        S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=HOP_LENGTH)

        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=N_MFCC + 1, n_fft=N_FFT, hop_length=HOP_LENGTH
        )[1:]
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=N_CONTRAST_BANDS)
        flatness = librosa.feature.spectral_flatness(S=S)[0]

        out = {
            f"timbre_mfcc_{i + 1:02d}": grid.average(times, mfcc[i]) for i in range(N_MFCC)
        }
        for b in range(contrast.shape[0]):
            out[f"timbre_contrast_{b + 1:02d}"] = grid.average(times, contrast[b])
        out["timbre_flatness"] = grid.average(times, flatness)
        return out
