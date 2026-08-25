"""Speaking-rate features derived from the transcribe word timestamps.

The conversation pattern applied to `transcribe`: frame-level, but
computed from word onset/offset intervals rather than the waveform. The
input is a word-timestamp table, either produced by `transcribe` in the
same run or loaded from an existing {stem}_transcript_words.csv.

Per grid window [k*hop, (k+1)*hop):

- ``speech_rate_words`` — word onsets per second (the macroscopic
  speaking rate; silence dilutes it toward 0)
- ``speech_rate_word_duration`` — time-weighted mean duration (s) of the
  words being spoken at each instant, NaN where no word is active (the
  local articulation proxy: slow, drawn-out speech scores high)
- ``speech_rate_pauses`` — silent within-utterance pauses per second: a
  gap between consecutive words counts when it is in
  [``MIN_PAUSE_SEC``, ``MAX_PAUSE_SEC``) — shorter is articulatory, and
  longer reads as an utterance boundary, not a pause

Word timestamps are Whisper's: fine for windowed rates, not for
onset-locked latencies (see the README's timing caveat). Fractions and
durations are sampled on a fine time base and reduced with
``grid.average``/``grid.rate`` — the family convention.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..exceptions import Aud2PsyError
from .base import BaseModel

FINE_DT = 0.01  # seconds; activity sampling resolution
MIN_PAUSE_SEC = 0.15  # below: articulatory gap, not a pause
MAX_PAUSE_SEC = 2.0  # at/above: utterance boundary, not a pause

REQUIRED_WORD_COLUMNS = ["word", "onset", "offset"]


class SpeechRateModel(BaseModel):
    name = "speech_rate"
    level = "frame"
    checkpoint = None  # analytic — provenance lives with transcribe's checkpoint

    def derive(self, words_df: pd.DataFrame, grid) -> dict[str, np.ndarray]:
        """Compute the windowed features from a word-timestamp table."""
        feats = speech_rate_frames(words_df, grid)
        self.info_ = {
            "derived_from": "transcribe",
            "n_words": int(len(words_df)),
            "pause_gap_sec": [MIN_PAUSE_SEC, MAX_PAUSE_SEC],
        }
        return feats


def speech_rate_frames(words_df: pd.DataFrame, grid) -> dict[str, np.ndarray]:
    """Windowed speaking-rate features from a word-timestamp table.

    Pure pandas/numpy (offline-testable). Every returned array has
    length ``grid.n_windows``.
    """
    n_windows = grid.n_windows
    total = grid.hop * n_windows
    n_fine = max(1, int(round(total / FINE_DT)))
    t_fine = (np.arange(n_fine) + 0.5) * FINE_DT

    words = words_df.sort_values("onset", kind="stable")
    onsets = words["onset"].to_numpy(dtype=float)
    offsets = words["offset"].to_numpy(dtype=float)

    # time-weighted mean active-word duration on the fine base
    word_count = np.zeros(n_fine)
    dur_sum = np.zeros(n_fine)
    for onset, offset in zip(onsets, offsets):
        lo = max(int(np.ceil(onset / FINE_DT - 0.5)), 0)
        hi = min(int(np.ceil(offset / FINE_DT - 0.5)), n_fine)
        if hi <= lo:
            continue
        word_count[lo:hi] += 1
        dur_sum[lo:hi] += offset - onset
    with np.errstate(invalid="ignore"):
        mean_dur_fine = np.where(word_count > 0, dur_sum / np.maximum(word_count, 1), np.nan)

    # pause events: the gap between a word's offset and the next word's
    # onset, stamped at the gap start
    if len(words) > 1:
        gaps = onsets[1:] - offsets[:-1]
        is_pause = (gaps >= MIN_PAUSE_SEC) & (gaps < MAX_PAUSE_SEC)
        pause_onsets = offsets[:-1][is_pause]
    else:
        pause_onsets = onsets[:0]

    return {
        "speech_rate_words": grid.rate(onsets),
        "speech_rate_word_duration": grid.average(t_fine, mean_dur_fine),
        "speech_rate_pauses": grid.rate(pause_onsets),
    }


def load_words_csv(path: str | Path) -> pd.DataFrame:
    """Load an existing word-timestamp table ({stem}_transcript_words.csv).

    Accepts the file as ``save_result`` writes it (leading
    ``stimulus_id``, ``chunk_idx``/``word_idx``/``transcribe_probability``
    all fine); refuses a table mixing several stimuli.
    """
    path = Path(path)
    if not path.exists():
        raise Aud2PsyError(f"words table not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_WORD_COLUMNS if c not in df.columns]
    if missing:
        raise Aud2PsyError(
            f"{path} is not a word-timestamp table: missing column(s) "
            f"{', '.join(missing)} (expected the {'{stem}'}_transcript_words.csv "
            f"format: {', '.join(REQUIRED_WORD_COLUMNS)})"
        )
    if "stimulus_id" in df.columns and df["stimulus_id"].nunique() > 1:
        raise Aud2PsyError(
            f"{path} mixes {df['stimulus_id'].nunique()} stimulus_id values; "
            "speech_rate features need one stimulus per table"
        )
    return df
