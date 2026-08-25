"""Conversation-structure model tests.

Everything here is offline: the model is pure pandas/numpy over a turn
table, so the full pipeline path runs against a stored *_speakers.csv
without pyannote or any gated weights.
"""

import numpy as np
import pandas as pd
import pytest

from aud2psy.exceptions import Aud2PsyError
from aud2psy.grid import Grid
from aud2psy.models.conversation import conversation_frames, load_turns_csv


def turns(*rows):
    df = pd.DataFrame(rows, columns=["speaker", "onset", "offset"])
    df.insert(0, "turn_idx", range(len(df)))
    return df


# A: 0.0-1.5, B: 1.0-2.0 (0.5 s crosstalk), A again: 2.5-3.0, then silence.
DIALOGUE = turns(
    ("SPEAKER_00", 0.0, 1.5),
    ("SPEAKER_01", 1.0, 2.0),
    ("SPEAKER_00", 2.5, 3.0),
)
GRID = Grid.for_duration(4.0, 1.0)  # windows [0,1) [1,2) [2,3) [3,4)


def test_dialogue_features_hand_computed():
    feats = conversation_frames(DIALOGUE, GRID)
    np.testing.assert_allclose(feats["conversation_n_speakers"], [1, 2, 1, 0])
    np.testing.assert_allclose(feats["conversation_speech_fraction"], [1.0, 1.0, 0.5, 0.0])
    np.testing.assert_allclose(feats["conversation_overlap_fraction"], [0.0, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(feats["conversation_turn_rate"], [1.0, 1.0, 1.0, 0.0])
    # first turn is not a switch; B at 1.0 and A's return at 2.5 are
    np.testing.assert_allclose(feats["conversation_switch_rate"], [0.0, 1.0, 1.0, 0.0])
    # time-weighted mean active-turn duration; NaN in the silent window
    np.testing.assert_allclose(
        feats["conversation_turn_duration"][:3], [1.5, 1.125, 0.5]
    )
    assert np.isnan(feats["conversation_turn_duration"][3])


def test_all_arrays_have_grid_length_and_prefix():
    feats = conversation_frames(DIALOGUE, GRID)
    assert all(k.startswith("conversation_") for k in feats)
    assert all(len(v) == GRID.n_windows for v in feats.values())


def test_empty_turn_table_is_silence_not_error():
    feats = conversation_frames(turns(), GRID)
    np.testing.assert_allclose(feats["conversation_n_speakers"], np.zeros(4))
    np.testing.assert_allclose(feats["conversation_speech_fraction"], np.zeros(4))
    np.testing.assert_allclose(feats["conversation_turn_rate"], np.zeros(4))
    np.testing.assert_allclose(feats["conversation_switch_rate"], np.zeros(4))
    assert np.isnan(feats["conversation_turn_duration"]).all()


def test_same_speaker_overlap_is_not_crosstalk():
    # one speaker whose raw turns overlap: never counts as speaker overlap
    self_overlap = turns(("SPEAKER_00", 0.0, 1.0), ("SPEAKER_00", 0.5, 1.5))
    feats = conversation_frames(self_overlap, GRID)
    np.testing.assert_allclose(feats["conversation_overlap_fraction"], np.zeros(4))
    np.testing.assert_allclose(feats["conversation_n_speakers"], [1, 1, 0, 0])
    np.testing.assert_allclose(feats["conversation_switch_rate"], np.zeros(4))


def test_turns_past_grid_end_fold_into_last_window():
    late = turns(("SPEAKER_00", 3.5, 9.0))
    feats = conversation_frames(late, GRID)
    assert feats["conversation_n_speakers"][3] == 1
    assert feats["conversation_speech_fraction"][3] == pytest.approx(0.5)


# --- loading a stored *_speakers.csv ----------------------------------------


def speakers_csv(tmp_path, df, name="scores_speakers.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_load_turns_csv_accepts_saved_format(tmp_path):
    df = DIALOGUE.copy()
    df.insert(0, "stimulus_id", "clip01")  # save_result's leading column
    loaded = load_turns_csv(speakers_csv(tmp_path, df))
    assert list(loaded["speaker"]) == list(DIALOGUE["speaker"])


def test_load_turns_csv_missing_column_names_the_fix(tmp_path):
    bad = pd.DataFrame({"onset": [0.0], "offset": [1.0]})
    with pytest.raises(Aud2PsyError, match="missing column.*speaker"):
        load_turns_csv(speakers_csv(tmp_path, bad))


def test_load_turns_csv_refuses_mixed_stimuli(tmp_path):
    df = DIALOGUE.copy()
    df.insert(0, "stimulus_id", ["a", "a", "b"])
    with pytest.raises(Aud2PsyError, match="mixes 2 stimulus_id"):
        load_turns_csv(speakers_csv(tmp_path, df))


def test_load_turns_csv_missing_file(tmp_path):
    with pytest.raises(Aud2PsyError, match="not found"):
        load_turns_csv(tmp_path / "nope_speakers.csv")


# --- pipeline integration (offline: turns from CSV, no pyannote) ------------


def make_wav(tmp_path, duration=4.0):
    import soundfile as sf

    from conftest import silence

    wav = tmp_path / "clip.wav"
    sf.write(wav, silence(duration), 22050)
    return wav


def test_score_audio_conversation_from_csv(tmp_path):
    from aud2psy.pipeline import save_result, score_audio

    wav = make_wav(tmp_path)
    csv = speakers_csv(tmp_path, DIALOGUE)
    result = score_audio(wav, ["conversation"], hop=1.0, speakers_csv=csv,
                         show_progress=False)
    frames = result.frames_df
    assert list(frames["time"]) == [0.5, 1.5, 2.5, 3.5]
    np.testing.assert_allclose(frames["conversation_n_speakers"], [1, 2, 1, 0])
    meta = result.meta["models"]["conversation"]
    assert meta["turns_source"] == str(csv)
    assert meta["derived_from"] == "diarize"
    assert meta["n_turns"] == 3
    assert meta["checkpoint"] is None
    assert result.meta["frames"]["hop_sec"] == 1.0
    written = save_result(result, tmp_path / "scores.csv")
    header = written["frames"].read_text().splitlines()[0]
    assert header.startswith("stimulus_id,time,conversation_n_speakers")


def test_score_audio_conversation_composes_with_frame_models(tmp_path):
    from aud2psy.pipeline import score_audio

    wav = make_wav(tmp_path)
    csv = speakers_csv(tmp_path, DIALOGUE)
    result = score_audio(wav, ["loudness", "conversation"], hop=1.0,
                         speakers_csv=csv, show_progress=False)
    cols = list(result.frames_df.columns)
    assert "loudness_rms" in cols and "conversation_turn_rate" in cols
    assert len(result.frames_df) == 4


def test_score_audio_conversation_without_turn_source_errors(tmp_path):
    from aud2psy.pipeline import score_audio

    with pytest.raises(Aud2PsyError, match="diarize turn table"):
        score_audio(make_wav(tmp_path), ["conversation"], show_progress=False)


def test_batch_conversation_requires_diarize(tmp_path):
    from aud2psy.pipeline import score_audio_batch

    with pytest.raises(Aud2PsyError, match="needs diarize"):
        score_audio_batch([make_wav(tmp_path)], ["conversation"],
                          show_progress=False)


# --- CLI validation ---------------------------------------------------------


def test_cli_conversation_needs_diarize_or_speakers(capsys):
    from aud2psy.cli import main

    with pytest.raises(SystemExit):
        main(["conversation", "clip.wav"])
    assert "diarize turn table" in capsys.readouterr().err


def test_cli_speakers_requires_conversation(capsys):
    from aud2psy.cli import main

    with pytest.raises(SystemExit):
        main(["loudness", "clip.wav", "--speakers", "x_speakers.csv"])
    assert "requires the conversation model" in capsys.readouterr().err


def test_cli_speakers_replaces_diarize(capsys):
    from aud2psy.cli import main

    with pytest.raises(SystemExit):
        main(["conversation", "diarize", "clip.wav", "--speakers", "x_speakers.csv"])
    assert "replaces the diarize run" in capsys.readouterr().err


def test_cli_speakers_single_input_only(capsys):
    from aud2psy.cli import main

    with pytest.raises(SystemExit):
        main(["conversation", "a.wav", "b.wav", "--speakers", "x_speakers.csv"])
    assert "one turn table onto one input" in capsys.readouterr().err
