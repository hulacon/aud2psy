"""The shared output grid for frame-level models.

Features are computed at librosa's native frame resolution, then averaged into
non-overlapping windows of ``hop`` seconds. Window k covers
[k*hop, (k+1)*hop); its ``time`` value is the window center.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    hop: float
    n_windows: int

    @classmethod
    def for_duration(cls, duration: float, hop: float) -> "Grid":
        if hop <= 0:
            raise ValueError(f"hop must be positive, got {hop}")
        n = max(1, int(np.ceil(round(duration / hop, 9))))
        return cls(hop=hop, n_windows=n)

    @property
    def centers(self) -> np.ndarray:
        """Window-center times — the ``time`` column of the frames CSV."""
        return (np.arange(self.n_windows) + 0.5) * self.hop

    def _index(self, times: np.ndarray) -> np.ndarray:
        """Window index of each time, clipped to the grid.

        The quotient is rounded before flooring (the ``for_duration`` guard):
        0.3 / 0.1 is 2.9999999999999996, and truncating it filed a time on a
        window edge into the previous window for any non-power-of-two hop.
        """
        return np.clip(np.floor(np.round(times / self.hop, 9)).astype(int), 0, self.n_windows - 1)

    def average(self, times: np.ndarray, values: np.ndarray) -> np.ndarray:
        """NaN-aware mean of native-resolution ``values`` in each grid window.

        Windows with no native frames, or only NaN values (e.g. unvoiced f0),
        come out NaN. Native frames past the last window edge fold into the
        last window.
        """
        times = np.asarray(times, dtype=float)
        values = np.asarray(values, dtype=float)
        if times.shape != values.shape:
            raise ValueError(f"times {times.shape} and values {values.shape} differ")
        idx = self._index(times)
        valid = ~np.isnan(values)
        sums = np.bincount(idx[valid], weights=values[valid], minlength=self.n_windows)
        counts = np.bincount(idx[valid], minlength=self.n_windows)
        out = np.full(self.n_windows, np.nan)
        np.divide(sums, counts, out=out, where=counts > 0)
        return out

    def window_max(self, times: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Max of native ``values`` in each grid window (NaN where empty).

        For oscillating curves (e.g. a pulse curve) whose window *mean* is
        uninformative — the peak is the signal.
        """
        times = np.asarray(times, dtype=float)
        values = np.asarray(values, dtype=float)
        idx = self._index(times)
        valid = ~np.isnan(values)
        out = np.full(self.n_windows, -np.inf)
        np.maximum.at(out, idx[valid], values[valid])
        out[np.isinf(out)] = np.nan
        return out

    def rate(self, event_times: np.ndarray) -> np.ndarray:
        """Events per second in each grid window (e.g. onset rate)."""
        event_times = np.asarray(event_times, dtype=float)
        if event_times.size == 0:
            return np.zeros(self.n_windows)
        idx = self._index(event_times)
        return np.bincount(idx, minlength=self.n_windows) / self.hop
