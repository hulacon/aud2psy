"""Speech-rate model tests — pure pandas over word-timestamp tables,
fully offline (the conversation-test pattern applied to transcribe)."""

import numpy as np
import pandas as pd
import pytest

from aud2psy.exceptions import Aud2PsyError
from aud2psy.grid import Grid
from aud2psy.models.speech_rate import load_words_csv, speech_rate_frames


def words(*rows):
    df = pd.DataFrame(rows, columns=["word", "onset", "offset"])
    df.insert(0, "word_idx", range(len(df)))
    return df


# gaps: 0.1 s (articulatory, not a pause), 0.3 s (pause), 1.4 s (pause)
WORDS = words(
    ("hello", 0.2, 0.4),
    ("there", 0.5, 0.7),
    ("stranger", 1.0, 1.6),
    ("bye", 3.0, 3.2),
)
GRID = Grid.for_duration(4.0, 1.0)  # windows [0,1) [1,2) [2,3) [3,4)


class TestSpeechRateFrames:
    def test_hand_computed(self):
        f = speech_rate_frames(WORDS, GRID)
        np.testing.assert_allclose(f["speech_rate_words"], [2.0, 1.0, 0.0, 1.0])
        # pauses at 0.7 (0.3 s gap) and 1.6 (1.4 s gap); the 0.1 s gap is not one
        np.testing.assert_allclose(f["speech_rate_pauses"], [1.0, 1.0, 0.0, 0.0])
        # time-weighted mean active-word duration; NaN in wordless window
        np.testing.assert_allclose(f["speech_rate_word_duration"][[0, 1, 3]],
                                   [0.2, 0.6, 0.2])
        assert np.isnan(f["speech_rate_word_duration"][2])

    def test_empty_words_is_silence_not_error(self):
        f = speech_rate_frames(words(), GRID)
        np.testing.assert_allclose(f["speech_rate_words"], np.zeros(4))
        np.testing.assert_allclose(f["speech_rate_pauses"], np.zeros(4))
        assert np.isnan(f["speech_rate_word_duration"]).all()

    def test_long_gap_is_utterance_boundary_not_pause(self):
        # 2.5 s gap >= MAX_PAUSE_SEC: no pause event anywhere
        f = speech_rate_frames(words(("a", 0.0, 0.2), ("b", 2.7, 2.9)), GRID)
        np.testing.assert_allclose(f["speech_rate_pauses"], np.zeros(4))

    def test_all_prefixed_and_grid_length(self):
        f = speech_rate_frames(WORDS, GRID)
        assert all(k.startswith("speech_rate_") for k in f)
        assert all(len(v) == GRID.n_windows for v in f.values())


class TestLoadWordsCsv:
    def test_accepts_saved_format(self, tmp_path):
        df = WORDS.copy()
        df.insert(0, "stimulus_id", "clip01")
        df["transcribe_probability"] = 0.5
        path = tmp_path / "scores_transcript_words.csv"
        df.to_csv(path, index=False)
        loaded = load_words_csv(path)
        assert list(loaded["word"]) == list(WORDS["word"])

    def test_missing_column_names_the_fix(self, tmp_path):
        path = tmp_path / "bad_transcript_words.csv"
        pd.DataFrame({"onset": [0.0], "offset": [1.0]}).to_csv(path, index=False)
        with pytest.raises(Aud2PsyError, match="missing column.*word"):
            load_words_csv(path)

    def test_refuses_mixed_stimuli(self, tmp_path):
        df = WORDS.copy()
        df.insert(0, "stimulus_id", ["a", "a", "b", "b"])
        path = tmp_path / "mixed_transcript_words.csv"
        df.to_csv(path, index=False)
        with pytest.raises(Aud2PsyError, match="mixes 2 stimulus_id"):
            load_words_csv(path)


class TestPipelineIntegration:
    def make_wav(self, tmp_path, duration=4.0):
        import soundfile as sf

        from conftest import silence

        wav = tmp_path / "clip.wav"
        sf.write(wav, silence(duration), 22050)
        return wav

    def words_csv(self, tmp_path):
        path = tmp_path / "old_transcript_words.csv"
        WORDS.to_csv(path, index=False)
        return path

    def test_score_audio_from_csv(self, tmp_path):
        from aud2psy.pipeline import score_audio

        result = score_audio(self.make_wav(tmp_path), ["speech_rate"], hop=1.0,
                             words_csv=self.words_csv(tmp_path),
                             show_progress=False)
        frames = result.frames_df
        np.testing.assert_allclose(frames["speech_rate_words"], [2.0, 1.0, 0.0, 1.0])
        meta = result.meta["models"]["speech_rate"]
        assert meta["derived_from"] == "transcribe"
        assert meta["n_words"] == 4
        assert meta["checkpoint"] is None
        assert meta["words_source"].endswith("old_transcript_words.csv")

    def test_without_word_source_errors(self, tmp_path):
        from aud2psy.pipeline import score_audio

        with pytest.raises(Aud2PsyError, match="transcribe word"):
            score_audio(self.make_wav(tmp_path), ["speech_rate"],
                        show_progress=False)

    def test_batch_requires_transcribe(self, tmp_path):
        from aud2psy.pipeline import score_audio_batch

        with pytest.raises(Aud2PsyError, match="needs transcribe"):
            score_audio_batch([self.make_wav(tmp_path)], ["speech_rate"],
                              show_progress=False)


class TestCliValidation:
    def test_needs_transcribe_or_words(self, capsys):
        from aud2psy.cli import main

        with pytest.raises(SystemExit):
            main(["speech_rate", "clip.wav"])
        assert "transcribe word timestamps" in capsys.readouterr().err

    def test_words_requires_speech_rate(self, capsys):
        from aud2psy.cli import main

        with pytest.raises(SystemExit):
            main(["loudness", "clip.wav", "--words", "x_transcript_words.csv"])
        assert "requires the speech_rate model" in capsys.readouterr().err

    def test_words_replaces_transcribe(self, capsys):
        from aud2psy.cli import main

        with pytest.raises(SystemExit):
            main(["speech_rate", "transcribe", "clip.wav",
                  "--words", "x_transcript_words.csv"])
        assert "replaces the transcribe run" in capsys.readouterr().err
