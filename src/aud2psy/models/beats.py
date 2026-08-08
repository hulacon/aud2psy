"""Beat/downbeat event table via beat_this (CPJKU, ISMIR 2024).

Event-level: one row per detected beat with its downbeat flag — the
transcript export's onset/offset pattern applied to musical events.
beat_this is an optional extra (`pip install "aud2psy[beats]"`; it is a
git dependency, not on PyPI) and the ISMIR-2024 successor to madmom,
which is unmaintained and pinned to Python < 3.10. Inference on CPU:
fast enough for 3–6 min clips, and safer than MPS for this checkpoint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseModel

BEAT_COLUMNS = ["time", "is_downbeat"]
DOWNBEAT_TOLERANCE_SEC = 0.03


class BeatsModel(BaseModel):
    name = "beats"
    level = "events"

    def load(self) -> None:
        try:
            from beat_this.inference import Audio2Beats
        except ImportError as exc:
            raise ImportError(
                "The beats model needs the optional beat_this dependency: "
                'pip install "aud2psy[beats]" (or pip install '
                "git+https://github.com/CPJKU/beat_this)"
            ) from exc
        self.model = Audio2Beats(device="cpu", dbn=False)

    def detect(self, y: np.ndarray, sr: int):
        """Returns (beats_df, info) for mono audio at any sample rate."""
        beats, downbeats = self.model(y, sr)
        beats = np.asarray(beats, dtype=float)
        downbeats = np.asarray(downbeats, dtype=float)
        is_downbeat = (
            np.abs(beats[:, None] - downbeats[None, :]).min(axis=1) < DOWNBEAT_TOLERANCE_SEC
            if len(beats) and len(downbeats)
            else np.zeros(len(beats), dtype=bool)
        )
        df = pd.DataFrame({"time": beats, "is_downbeat": is_downbeat}, columns=BEAT_COLUMNS)
        ibis = np.diff(beats)
        info = {
            "n_beats": int(len(beats)),
            "n_downbeats": int(len(downbeats)),
            "tempo_bpm_median": round(float(60.0 / np.median(ibis)), 1) if len(ibis) else None,
        }
        return df, info
