"""Conversation-structure features derived from the diarize turn table.

Frame-level, but computed from "who speaks when" rather than from the
waveform: the input is a diarize turn table (raw, possibly overlapping),
either produced in the same run or loaded from an existing
{stem}_speakers.csv — the two paths are bit-identical because everything
here derives from the raw turns alone (the exclusive timeline is never
persisted, so it is deliberately not used).

Per grid window [k*hop, (k+1)*hop):

- ``conversation_n_speakers`` — distinct speakers active at any point in
  the window (0 in silence)
- ``conversation_speech_fraction`` — fraction of the window with >= 1
  active speaker
- ``conversation_overlap_fraction`` — fraction with >= 2 concurrent
  speakers (interruptions / crosstalk)
- ``conversation_turn_rate`` — turn onsets per second
- ``conversation_switch_rate`` — onsets per second of turns whose speaker
  differs from the immediately preceding turn (onset order); the first
  turn is not a switch
- ``conversation_turn_duration`` — time-weighted mean duration (s) of the
  turns active at each instant, NaN where nobody speaks

Fractions and durations are sampled on a fine time base (``FINE_DT``) and
reduced with ``grid.average`` — the family's compute-native-then-reduce
convention; the sampling error is negligible against >= 0.5 s windows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..exceptions import Aud2PsyError
from .base import BaseModel

FINE_DT = 0.01  # seconds; activity sampling resolution

REQUIRED_TURN_COLUMNS = ["speaker", "onset", "offset"]


class ConversationModel(BaseModel):
    name = "conversation"
    level = "frame"
    checkpoint = None  # analytic — provenance lives with diarize's checkpoint

    def derive(self, turns_df: pd.DataFrame, grid) -> dict[str, np.ndarray]:
        """Compute the windowed features from a raw turn table."""
        feats = conversation_frames(turns_df, grid)
        self.info_ = {
            "derived_from": "diarize",
            "n_turns": int(len(turns_df)),
        }
        return feats


def conversation_frames(turns_df: pd.DataFrame, grid) -> dict[str, np.ndarray]:
    """Windowed conversation-structure features from a raw turn table.

    Pure pandas/numpy (offline-testable, the merge_speakers pattern).
    Every returned array has length ``grid.n_windows``.
    """
    n_windows = grid.n_windows
    total = grid.hop * n_windows
    n_fine = max(1, int(round(total / FINE_DT)))
    t_fine = (np.arange(n_fine) + 0.5) * FINE_DT

    turns = turns_df.sort_values("onset", kind="stable")
    onsets = turns["onset"].to_numpy(dtype=float)
    offsets = turns["offset"].to_numpy(dtype=float)
    speakers = turns["speaker"].astype(str).to_numpy()

    def _bins(onset: float, offset: float) -> tuple[int, int]:
        """Fine bins whose center falls in [onset, offset), clipped."""
        lo = int(np.ceil(onset / FINE_DT - 0.5))
        hi = int(np.ceil(offset / FINE_DT - 0.5))
        return max(lo, 0), min(hi, n_fine)

    # Per-speaker activity on the fine base (bool, so a speaker overlapping
    # their own turns is never double-counted), plus active-turn count and
    # duration sums for the time-weighted mean turn duration.
    activity: dict[str, np.ndarray] = {}
    turn_count = np.zeros(n_fine)
    dur_sum = np.zeros(n_fine)
    for onset, offset, speaker in zip(onsets, offsets, speakers):
        lo, hi = _bins(onset, offset)
        if hi <= lo:
            continue
        act = activity.setdefault(speaker, np.zeros(n_fine, dtype=bool))
        act[lo:hi] = True
        turn_count[lo:hi] += 1
        dur_sum[lo:hi] += offset - onset

    concurrency = np.zeros(n_fine)
    for act in activity.values():
        concurrency += act

    if activity:
        n_speakers = np.sum(
            [grid.window_max(t_fine, act.astype(float)) for act in activity.values()],
            axis=0,
        )
    else:
        n_speakers = np.zeros(n_windows)

    with np.errstate(invalid="ignore"):
        mean_dur_fine = np.where(turn_count > 0, dur_sum / np.maximum(turn_count, 1), np.nan)

    switch_onsets = onsets[1:][speakers[1:] != speakers[:-1]] if len(turns) > 1 else onsets[:0]

    return {
        "conversation_n_speakers": n_speakers,
        "conversation_speech_fraction": grid.average(t_fine, (concurrency >= 1).astype(float)),
        "conversation_overlap_fraction": grid.average(t_fine, (concurrency >= 2).astype(float)),
        "conversation_turn_rate": grid.rate(onsets),
        "conversation_switch_rate": grid.rate(switch_onsets),
        "conversation_turn_duration": grid.average(t_fine, mean_dur_fine),
    }


def load_turns_csv(path: str | Path) -> pd.DataFrame:
    """Load an existing diarize turn table ({stem}_speakers.csv).

    Accepts the file exactly as ``save_result`` writes it (a leading
    ``stimulus_id`` column and ``turn_idx`` are both fine); refuses a
    table that mixes several stimuli — conversation features are
    per-stimulus by construction.
    """
    path = Path(path)
    if not path.exists():
        raise Aud2PsyError(f"speakers table not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_TURN_COLUMNS if c not in df.columns]
    if missing:
        raise Aud2PsyError(
            f"{path} is not a diarize turn table: missing column(s) "
            f"{', '.join(missing)} (expected the {'{stem}'}_speakers.csv format: "
            f"{', '.join(REQUIRED_TURN_COLUMNS)})"
        )
    if "stimulus_id" in df.columns and df["stimulus_id"].nunique() > 1:
        raise Aud2PsyError(
            f"{path} mixes {df['stimulus_id'].nunique()} stimulus_id values; "
            "conversation features need one stimulus per table"
        )
    return df
