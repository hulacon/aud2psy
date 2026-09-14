"""Merge standalone per-model ``*_frames.csv`` files into one dashboard table.

Each aud2psy model writes its own frames CSV (``loudness_frames.csv``,
``pitch_frames.csv``, ...), all keyed on ``stimulus_id`` and ``time``. The
dashboard renders every model it detects in a single DataFrame, so an
all-models view is a merge problem, solved here.

Frames on the spine's exact time grid (the common case: one extraction run,
one hop) merge exactly; frames on a different grid merge to the nearest
spine row within half the spine's sampling step. Tables that cannot merge
1:1 (several rows per key, e.g. speaker diarization segments) are skipped
with a note when auto-collected from a directory, and raise when named
explicitly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

KEY_COLUMNS = ("stimulus_id", "time")


def collect_frames_csvs(directory: Path) -> list[Path]:
    """The ``*_frames.csv`` files directly inside ``directory``."""
    out = [p for p in sorted(directory.glob("*_frames.csv"))
           if not p.name.startswith(".")]
    if not out:
        raise FileNotFoundError(
            f"no *_frames.csv files found in {directory}")
    return out


def _spine_time_step(spine: pd.DataFrame) -> float:
    steps = (
        spine.sort_values("time", kind="stable")
        .groupby("stimulus_id")["time"]
        .diff()
        .dropna()
    )
    return float(steps.median()) if not steps.empty else 0.0


def merge_frames(
    frames: dict[str, pd.DataFrame],
    tolerance: float | None = None,
    lenient: set[str] | None = None,
) -> pd.DataFrame:
    """Merge per-model frame DataFrames (name -> frame) into one.

    ``tolerance`` overrides the nearest-time match window (default: half
    the spine's median sampling step). Names in ``lenient`` are skipped
    with a note instead of raising when they cannot merge 1:1.
    """
    lenient = lenient or set()
    if not frames:
        raise ValueError("no input frames to merge")

    for name, df in list(frames.items()):
        missing = [k for k in KEY_COLUMNS if k not in df.columns]
        if missing:
            raise ValueError(
                f"{name} lacks {'/'.join(missing)}; frames CSVs are keyed "
                "on stimulus_id and time")
        if df.duplicated(subset=list(KEY_COLUMNS)).any():
            if name in lenient:
                print(f"skipping {Path(name).name}: several rows per "
                      "(stimulus, time), cannot merge 1:1")
                del frames[name]
                continue
            raise ValueError(
                f"{name} has several rows per (stimulus_id, time); merge "
                "would multiply rows — exclude this file")

    # Same feature column from two files would silently collide in a merge.
    owners: dict[str, str] = {}
    for name, df in frames.items():
        for col in df.columns:
            if col in KEY_COLUMNS:
                continue
            if col in owners:
                raise ValueError(
                    f"feature column {col!r} appears in both {owners[col]} "
                    f"and {name}; rename or drop one file")
            owners[col] = name

    spine_name = max(frames, key=lambda n: len(frames[n]))
    spine = (frames[spine_name].sort_values("time", kind="stable")
             .reset_index(drop=True))
    if tolerance is None:
        tolerance = _spine_time_step(spine) / 2

    spine_grid = set(zip(spine["stimulus_id"], spine["time"]))
    merged = spine
    for name, df in frames.items():
        if name == spine_name:
            continue
        if set(zip(df["stimulus_id"], df["time"])) <= spine_grid:
            merged = merged.merge(df, on=list(KEY_COLUMNS), how="left")
        else:
            incoming = df.sort_values("time", kind="stable")
            merged = pd.merge_asof(
                merged, incoming,
                on="time", by="stimulus_id",
                direction="nearest",
                tolerance=tolerance if tolerance > 0 else None,
            )
    return (merged.sort_values(["stimulus_id", "time"], kind="stable")
            .reset_index(drop=True))
