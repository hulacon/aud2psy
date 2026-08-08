"""Synthetic-audio fixtures: known ground truth, fully offline."""

import numpy as np
import pytest
import soundfile as sf

SR = 22050


def sine(freq: float, duration: float, amp: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(duration: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float32)


def white_noise(duration: float, amp: float = 0.3, sr: int = SR, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(sr * duration))).astype(np.float32)


def noise_bursts(
    burst_times: list[float],
    duration: float,
    burst_len: float = 0.1,
    amp: float = 0.5,
    sr: int = SR,
) -> np.ndarray:
    """Silence with short white-noise bursts starting at known times."""
    y = silence(duration, sr)
    rng = np.random.default_rng(1)
    n_burst = int(sr * burst_len)
    for t in burst_times:
        start = int(sr * t)
        y[start : start + n_burst] = amp * rng.standard_normal(n_burst)
    return y


@pytest.fixture
def wav_factory(tmp_path):
    """Write a signal to a temp wav and return its path."""
    counter = iter(range(1000))

    def _write(y: np.ndarray, sr: int = SR):
        path = tmp_path / f"test_{next(counter)}.wav"
        sf.write(path, y, sr)
        return path

    return _write
