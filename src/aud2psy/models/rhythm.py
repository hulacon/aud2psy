"""Rhythmicity: pulse clarity, beat strength, structural novelty.

Completes the Alluri/Toiviainen MIRtoolbox regressor canon alongside
`tonal`. Long-window features emitted on the shared 2 Hz grid.

- rhythm_pulse_clarity — salience of the strongest periodicity in the
  *mean-removed* onset envelope's local autocorrelation (~3 s window),
  restricted to beat-plausible lags (0.25–2 s, i.e. 30–240 BPM) and
  normalized by the zero-lag energy: ~0.6 for a metronomic click track,
  ~0.2 for arrhythmic noise. Mean removal is essential — a quasi-constant
  envelope's DC component otherwise autocorrelates at every lag. Windows
  with no meaningful onset activity (local mean onset strength <
  MIN_ONSET_ACTIVITY; onset strength is log-mel flux, hence
  loudness-invariant) get 0: no activity, no pulse.
- rhythm_beat_strength — window *max* of librosa's predominant-local-pulse
  curve (the mean of an oscillating pulse curve is uninformative).
- rhythm_novelty — Foote checkerboard novelty on a cosine self-similarity
  matrix of MFCC frames aggregated to 0.2 s: peaks where the audio
  changes texture (section boundaries, scene transitions).
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel

HOP_LENGTH = 512
WINDOW_SEC = 3.0
LAG_MIN_SEC = 0.25  # 240 BPM
LAG_MAX_SEC = 2.0  # 30 BPM
MIN_ONSET_ACTIVITY = 0.1  # log-mel flux units; real onsets are >= ~1
NOVELTY_FRAME_SEC = 0.2
NOVELTY_KERNEL_SEC = 4.0


def _checkerboard(half: int) -> np.ndarray:
    """Gaussian-tapered checkerboard kernel of side 2*half (Foote 2000)."""
    g = np.exp(-0.5 * (np.linspace(-2, 2, 2 * half) ** 2))
    taper = np.outer(g, g)
    sign = np.ones((2 * half, 2 * half))
    sign[:half, half:] = -1
    sign[half:, :half] = -1
    kernel = taper * sign
    return kernel / np.abs(kernel).sum()


class RhythmModel(BaseModel):
    name = "rhythm"
    level = "frame"

    def extract(self, y: np.ndarray, sr: int, grid) -> dict[str, np.ndarray]:
        import librosa
        from scipy.ndimage import uniform_filter1d

        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        times = librosa.frames_to_time(np.arange(len(oenv)), sr=sr, hop_length=HOP_LENGTH)
        win = max(4, int(round(WINDOW_SEC * sr / HOP_LENGTH)))

        local_mean = uniform_filter1d(oenv, size=win, mode="nearest")
        centered = oenv - local_mean
        tg = librosa.feature.tempogram(
            onset_envelope=centered, sr=sr, hop_length=HOP_LENGTH, win_length=win, norm=None
        )  # (win, n): local autocorrelation, row = lag in frames
        lag_min = max(1, int(round(LAG_MIN_SEC * sr / HOP_LENGTH)))
        lag_max = min(tg.shape[0] - 1, int(round(LAG_MAX_SEC * sr / HOP_LENGTH)))
        zero_lag = tg[0]
        peak = tg[lag_min : lag_max + 1].max(axis=0)
        pulse_clarity = np.clip(peak / np.where(zero_lag > 1e-12, zero_lag, 1.0), 0.0, 1.0)
        pulse_clarity[local_mean < MIN_ONSET_ACTIVITY] = 0.0

        plp = librosa.beat.plp(
            onset_envelope=oenv, sr=sr, hop_length=HOP_LENGTH,
            win_length=min(384, len(oenv)),
        )

        novelty, novelty_times = self._novelty(y, sr)

        return {
            "rhythm_pulse_clarity": grid.average(times, pulse_clarity),
            "rhythm_beat_strength": grid.window_max(times, plp),
            "rhythm_novelty": grid.average(novelty_times, novelty),
        }

    @staticmethod
    def _novelty(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        import librosa

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
        step = max(1, int(round(NOVELTY_FRAME_SEC * sr / HOP_LENGTH)))
        bounds = np.arange(0, mfcc.shape[1], step)
        feats = librosa.util.sync(mfcc, bounds, aggregate=np.mean)  # (13, n_coarse)
        n = feats.shape[1]
        # z-score each MFCC dim across time, then cosine self-similarity
        feats = feats - feats.mean(axis=1, keepdims=True)
        sd = feats.std(axis=1, keepdims=True)
        feats = feats / np.where(sd > 1e-12, sd, 1.0)
        norms = np.linalg.norm(feats, axis=0)
        unit = feats / np.where(norms > 1e-12, norms, 1.0)
        ssm = unit.T @ unit  # (n, n)

        half = max(2, int(round(NOVELTY_KERNEL_SEC / NOVELTY_FRAME_SEC / 2)))
        kernel = _checkerboard(half)
        padded = np.pad(ssm, half, mode="edge")
        novelty = np.array(
            [
                (kernel * padded[t : t + 2 * half, t : t + 2 * half]).sum()
                for t in range(n)
            ]
        )
        novelty = np.maximum(novelty, 0.0)
        times = (bounds + step / 2) * HOP_LENGTH / sr
        return novelty, times[:n]
