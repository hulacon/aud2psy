"""The speech gate shared by egemaps and speech_emotion (aud2psy.speech_gate).

Coverage arithmetic is exact interval math, so these assert closed-form
fractions; the CLI tests check the routing rules without scoring anything.
"""

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from aud2psy import pipeline
from aud2psy.cli import main
from aud2psy.speech_gate import (
    DEFAULT_SPEECH_GATE,
    GATE_MIN_SPEECH,
    check_gate,
    turn_coverage,
    window_starts,
)

from conftest import sine


def turns(*spans):
    return pd.DataFrame({"speaker": [f"S{i}" for i in range(len(spans))],
                         "onset": [a for a, _ in spans], "offset": [b for _, b in spans]})


def test_defaults():
    assert DEFAULT_SPEECH_GATE == "diarize" and GATE_MIN_SPEECH == 0.25
    assert check_gate("silero") == "silero"
    with pytest.raises(ValueError):
        check_gate("Silero")


def test_grid_windows_take_their_own_span():
    # hop-sized windows centred on grid centres are exactly [k*hop, (k+1)*hop)
    centers = np.array([0.25, 0.75, 1.25, 1.75])
    np.testing.assert_allclose(window_starts(centers, 0.5, 2.0), [0.0, 0.5, 1.0, 1.5])
    cov = turn_coverage(turns((0.4, 1.0)), centers, 0.5, 2.0)
    np.testing.assert_allclose(cov, [0.2, 1.0, 0.0, 0.0])


def test_overlapping_turns_count_once():
    # crosstalk: two speakers over [1, 2) must not give coverage > 1
    cov = turn_coverage(turns((0.0, 2.0), (1.0, 3.0)), np.array([2.0]), 4.0, 4.0)
    np.testing.assert_allclose(cov, [0.75])


def test_context_windows_are_edge_clamped():
    # 4 s windows in a 10 s clip: the first starts at 0, the last ends at 10
    centers = np.array([0.25, 5.0, 9.75])
    np.testing.assert_allclose(window_starts(centers, 4.0, 10.0), [0.0, 3.0, 6.0])
    cov = turn_coverage(turns((9.0, 10.0)), centers, 4.0, 10.0)
    np.testing.assert_allclose(cov, [0.0, 0.0, 0.25])


def test_clip_shorter_than_the_window_uses_the_whole_clip():
    cov = turn_coverage(turns((0.0, 1.0)), np.array([0.5, 1.5]), 4.0, 2.0)
    np.testing.assert_allclose(cov, [0.5, 0.5])


def test_empty_or_degenerate_turn_tables_cover_nothing():
    empty = pd.DataFrame(columns=["speaker", "onset", "offset"])
    centers = np.array([0.25, 0.75])
    np.testing.assert_array_equal(turn_coverage(empty, centers, 0.5, 1.0), [0.0, 0.0])
    np.testing.assert_array_equal(turn_coverage(turns((0.5, 0.5)), centers, 0.5, 1.0), [0.0, 0.0])


class TestCli:
    """Routing only: score_audio is mocked, nothing is extracted."""

    def run(self, wav_factory, models, *options):
        path = wav_factory(sine(220, 0.5))
        fake = pipeline.ScoreResult(frames_df=pd.DataFrame({"time": [0.25]}),
                                    transcript_df=None, words_df=None)
        with mock.patch.object(pipeline, "score_audio", return_value=fake) as score:
            rc = main([*options, *models, str(path)])
        return rc, score

    def test_gated_model_without_turns_is_refused(self, wav_factory, capsys):
        with pytest.raises(SystemExit):
            self.run(wav_factory, ["egemaps"])
        err = capsys.readouterr().err
        assert "--speakers" in err and "--speech-gate silero" in err

    def test_silero_gate_needs_no_turns(self, wav_factory):
        rc, score = self.run(wav_factory, ["egemaps"], "--speech-gate", "silero")
        assert rc == 0 and score.call_args.kwargs["speech_gate"] == "silero"

    def test_speakers_feeds_the_gate(self, wav_factory, tmp_path):
        csv = tmp_path / "s.csv"
        turns((0.0, 0.5)).to_csv(csv, index=False)
        rc, score = self.run(wav_factory, ["speech_emotion"], "--speakers", str(csv))
        assert rc == 0
        assert score.call_args.kwargs["speech_gate"] == "diarize"
        assert score.call_args.kwargs["speakers_csv"] == str(csv)

    def test_diarize_in_the_call_feeds_the_gate(self, wav_factory):
        rc, score = self.run(wav_factory, ["diarize", "egemaps"])
        assert rc == 0 and score.call_args.kwargs["speech_gate"] == "diarize"

    def test_speakers_with_nothing_to_feed_is_refused(self, wav_factory, tmp_path, capsys):
        csv = tmp_path / "s.csv"
        turns((0.0, 0.5)).to_csv(csv, index=False)
        with pytest.raises(SystemExit):
            self.run(wav_factory, ["egemaps"], "--speech-gate", "silero", "--speakers", str(csv))
        assert "--speakers requires" in capsys.readouterr().err


def test_batch_needs_diarize_in_the_call(wav_factory):
    from aud2psy.exceptions import Aud2PsyError

    paths = [wav_factory(sine(220, 0.5)), wav_factory(sine(330, 0.5))]
    with pytest.raises(Aud2PsyError, match="in a batch"):
        pipeline.score_audio_batch(paths, ["speech_emotion"], show_progress=False)
