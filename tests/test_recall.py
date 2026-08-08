"""Wordpool recall annotation and speech-timing metrics (offline)."""

import numpy as np
import pandas as pd
import pytest

from aud2psy.recall import (
    annotate_recall,
    load_wordpool,
    recall_summary,
    speech_timing,
)

POOL = ["apple", "banana", "cherry", "dolphin"]


def words(*rows):
    return pd.DataFrame(rows, columns=["word", "onset", "offset"])


def test_exact_and_fuzzy_matching():
    df = words(("Apple,", 1.0, 1.4), ("bananna", 3.0, 3.6), ("table", 5.0, 5.4))
    out = annotate_recall(df, POOL)
    assert list(out["matched_item"]) == ["apple", "banana", ""]
    assert list(out["pool_index"]) == [0, 1, -1]
    assert list(out["intrusion"]) == [False, False, True]
    assert out["match_score"][0] == 1.0
    assert 0.8 <= out["match_score"][1] < 1.0


def test_repetition_and_irt():
    df = words(("apple", 1.0, 1.4), ("cherry", 4.0, 4.5), ("apple", 9.0, 9.4))
    out = annotate_recall(df, POOL)
    assert list(out["repetition"]) == [False, False, True]
    assert np.isnan(out["irt"][0])  # first response has no IRT
    assert out["irt"][1] == 3.0
    assert out["irt"][2] == 5.0


def test_intrusions_do_not_advance_irt():
    df = words(("apple", 1.0, 1.4), ("xyzzy", 2.0, 2.3), ("cherry", 6.0, 6.5))
    out = annotate_recall(df, POOL)
    assert np.isnan(out["irt"][1])  # intrusion: no IRT
    assert out["irt"][2] == 5.0  # measured from previous *matched* recall


def test_recall_summary():
    df = words(("apple", 1.0, 1.4), ("apple", 2.0, 2.4), ("table", 3.0, 3.4),
               ("dolphin", 5.0, 5.6))
    summary = recall_summary(annotate_recall(df, POOL), POOL)
    assert summary["n_pool_items"] == 4
    assert summary["n_recalled_unique"] == 2
    assert summary["n_intrusions"] == 1
    assert summary["n_repetitions"] == 1
    assert summary["mean_irt_sec"] == 2.0  # (2.0-1.0 and 5.0-2.0) -> mean 2.0


def test_speech_timing_metrics():
    df = words(("a", 0.8, 1.0), ("b", 1.1, 1.3), ("c", 2.5, 2.8), ("d", 2.9, 3.2))
    m = speech_timing(df, duration=10.0)
    assert m["n_words"] == 4
    assert m["latency_first_word_sec"] == 0.8
    assert m["n_pauses"] == 1  # only the 1.3 -> 2.5 gap (1.2 s) crosses 0.5 s
    assert m["max_pause_sec"] == 1.2
    assert m["speech_span_sec"] == pytest.approx(2.4)


def test_speech_timing_empty():
    assert speech_timing(words(), duration=5.0) == {"n_words": 0}


def test_load_wordpool(tmp_path):
    pool_file = tmp_path / "pool.txt"
    pool_file.write_text("# list 1\napple\n\nbanana\n")
    assert load_wordpool(pool_file) == ["apple", "banana"]
    empty = tmp_path / "empty.txt"
    empty.write_text("# nothing\n")
    with pytest.raises(ValueError):
        load_wordpool(empty)


def test_cli_wordpool_requires_transcribe(tmp_path):
    from aud2psy.cli import main

    pool_file = tmp_path / "pool.txt"
    pool_file.write_text("apple\n")
    with pytest.raises(SystemExit):
        main(["loudness", "in.wav", "--wordpool", str(pool_file)])
