"""Tests for aud2psy.viz.merge — multi-model frames merging for the dashboard."""

import numpy as np
import pandas as pd
import pytest

from aud2psy.viz.merge import collect_frames_csvs, merge_frames


def _frames(model, times, stim="clip-a"):
    return pd.DataFrame({
        "stimulus_id": [stim] * len(times),
        "time": list(times),
        f"{model}_value": np.arange(len(times), dtype=float),
    })


def test_identical_grids_merge_exactly():
    times = [0.25, 0.75, 1.25]
    merged = merge_frames({"loudness_frames.csv": _frames("loudness", times),
                           "pitch_frames.csv": _frames("pitch", times)})
    assert len(merged) == 3
    assert merged["pitch_value"].notna().all()


def test_offset_grid_merges_to_nearest():
    merged = merge_frames(
        {"a_frames.csv": _frames("a", [0.0, 0.5, 1.0, 1.5]),
         "b_frames.csv": _frames("b", [0.25, 0.75, 1.25])})
    assert len(merged) == 4
    assert merged["b_value"].notna().all()


def test_duplicate_keys_raise_when_explicit():
    dup = pd.concat([_frames("seg", [0.0]), _frames("seg", [0.0])])
    with pytest.raises(ValueError, match="several rows per"):
        merge_frames({"seg_frames.csv": dup,
                      "pitch_frames.csv": _frames("pitch", [0.0, 0.5])})


def test_duplicate_keys_skipped_when_lenient(capsys):
    dup = pd.concat([_frames("seg", [0.0]), _frames("seg", [0.0])])
    merged = merge_frames(
        {"seg_frames.csv": dup,
         "pitch_frames.csv": _frames("pitch", [0.0, 0.5])},
        lenient={"seg_frames.csv"})
    assert "seg_value" not in merged.columns
    assert "skipping seg_frames.csv" in capsys.readouterr().out


def test_column_collision_is_an_error():
    with pytest.raises(ValueError, match="a_frames.csv.*b_frames.csv"):
        merge_frames({"a_frames.csv": _frames("x", [0.0]),
                      "b_frames.csv": _frames("x", [0.0])})


def test_missing_keys_is_an_error():
    with pytest.raises(ValueError, match="stimulus_id"):
        merge_frames({"a_frames.csv": pd.DataFrame({"time": [0.0]})})


def test_collect_finds_only_frames_csvs(tmp_path):
    for name in ("loudness_frames.csv", "pitch_frames.csv",
                 "transcribe_transcript.csv", "clip.csv", ".hidden_frames.csv"):
        (tmp_path / name).write_text("stimulus_id,time\n")
    found = [p.name for p in collect_frames_csvs(tmp_path)]
    assert found == ["loudness_frames.csv", "pitch_frames.csv"]


def test_collect_empty_dir_errors_with_path(tmp_path):
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        collect_frames_csvs(tmp_path)
