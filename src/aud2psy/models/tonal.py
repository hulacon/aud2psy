"""Tonality: key clarity, mode-majorness, and chroma entropy.

The Alluri/Toiviainen naturalistic-music-fMRI regressors, computed the way
that literature does: chroma at native resolution, smoothed over a ~3 s
sliding window (tonality integrates over seconds, not milliseconds), then
emitted on the shared 2 Hz grid. Key estimation follows the classic
Krumhansl–Schmuckler algorithm: Pearson correlation of the windowed chroma
distribution against the 24 rotated Krumhansl–Kessler major/minor profiles.

- tonal_key_clarity — best correlation across all 24 keys: how strongly
  the pitch-class distribution implies *some* key (noise/silence ≈ 0).
- tonal_majorness — best-major minus best-minor correlation: continuous
  mode estimate (positive = major-leaning).
- tonal_chroma_entropy — normalized entropy of the pitch-class
  distribution in [0, 1] (single tone → 0-ish, noise → 1-ish).

All three are NaN where the window holds no tonal energy (silence).
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel

WINDOW_SEC = 3.0
HOP_LENGTH = 512

# Krumhansl & Kessler (1982) probe-tone profiles, C-based.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _key_profiles() -> np.ndarray:
    """(24, 12) z-scored profiles: 12 major rotations then 12 minor."""
    profiles = np.array(
        [np.roll(KK_MAJOR, k) for k in range(12)] + [np.roll(KK_MINOR, k) for k in range(12)]
    )
    profiles = profiles - profiles.mean(axis=1, keepdims=True)
    return profiles / np.linalg.norm(profiles, axis=1, keepdims=True)


class TonalModel(BaseModel):
    name = "tonal"
    level = "frame"

    def extract(self, y: np.ndarray, sr: int, grid) -> dict[str, np.ndarray]:
        import librosa
        from scipy.ndimage import uniform_filter1d

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)  # (12, n)
        n = chroma.shape[1]
        times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=HOP_LENGTH)
        win = max(1, int(round(WINDOW_SEC * sr / HOP_LENGTH)))
        smooth = uniform_filter1d(chroma, size=win, axis=1, mode="nearest")

        energy = smooth.sum(axis=0)
        silent = energy < 1e-6

        # Pearson correlation of each frame's chroma against the 24 profiles
        centered = smooth - smooth.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(centered, axis=0)
        unit = centered / np.where(norms > 1e-12, norms, 1.0)
        corr = _key_profiles() @ unit  # (24, n)

        key_clarity = corr.max(axis=0)
        majorness = corr[:12].max(axis=0) - corr[12:].max(axis=0)

        p = smooth / np.where(energy > 1e-12, energy, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.nansum(p * np.log2(np.where(p > 0, p, 1.0)), axis=0) / np.log2(12)

        for arr in (key_clarity, majorness, entropy):
            arr[silent] = np.nan
        return {
            "tonal_key_clarity": grid.average(times, key_clarity),
            "tonal_majorness": grid.average(times, majorness),
            "tonal_chroma_entropy": grid.average(times, entropy),
        }
