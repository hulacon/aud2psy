"""Tests for the interactive HTML dashboard.

Fully offline: synthetic DataFrames and wavs, no model loading, no ffmpeg
required (wav inputs embed raw when ffmpeg is absent).
"""

import numpy as np
import pandas as pd
import pytest

from aud2psy.cli import resolve_scores_paths
from aud2psy.viz.dashboard import (
    _audio_envelope,
    _infer_hop,
    _is_scalar_col,
    create_dashboard,
    prepare_audio,
)
from aud2psy.viz.feature_config import detect_models_in_dataframe, get_model_columns
from conftest import sine


def _make_frames_df(n=40, with_embeddings=True, rng_seed=0):
    rng = np.random.RandomState(rng_seed)
    data = {
        "time": np.arange(n) * 0.5 + 0.25,
        "loudness_rms": rng.uniform(0, 0.3, n),
        "loudness_db": rng.uniform(-60, 0, n),
        "pitch_f0": rng.uniform(80, 300, n),
        "pitch_voiced_prob": rng.uniform(0, 1, n),
        "spectral_centroid": rng.uniform(500, 4000, n),
        "spectral_bandwidth": rng.uniform(500, 3000, n),
        "spectral_rolloff": rng.uniform(1000, 8000, n),
        "spectral_flux": rng.uniform(0, 5, n),
        "spectral_zcr": rng.uniform(0, 0.5, n),
        "onsets_strength": rng.uniform(0, 3, n),
        "onsets_rate": rng.uniform(0, 4, n),
        "onsets_tempo": rng.uniform(60, 180, n),
        "speech_prob": rng.uniform(0, 1, n),
        "music_emotion_valence": rng.uniform(-1, 1, n),
        "music_emotion_arousal": rng.uniform(-1, 1, n),
    }
    # unvoiced frames: NaN f0, the family convention
    data["pitch_f0"][::7] = np.nan
    for cat in ["speech", "music", "laughter", "siren_alarm"]:
        data[f"sound_events_{cat}"] = rng.uniform(0, 0.5, n)
    if with_embeddings:
        for i in range(16):
            data[f"clap_{i:03d}"] = rng.randn(n)
    return pd.DataFrame(data)


def _make_transcript_df(n=6, rng_seed=0):
    rng = np.random.RandomState(rng_seed)
    return pd.DataFrame({
        "chunk_idx": np.arange(n),
        "transcribe_text": [f"Sentence number {i}." for i in range(n)],
        "onset": np.arange(n) * 3.0,
        "offset": np.arange(n) * 3.0 + 2.5,
        "transcribe_asr_confidence": rng.uniform(0.5, 1, n),
        "transcribe_no_speech_prob": rng.uniform(0, 0.2, n),
        "speaker": ["SPEAKER_00", "SPEAKER_01"] * (n // 2),
    })


class TestFeatureDetection:
    def test_frame_models_detected(self):
        cols = _make_frames_df().columns.tolist()
        detected = detect_models_in_dataframe(cols, level="frame")
        for model in ["loudness", "pitch", "spectral", "onsets", "speech",
                      "music_emotion", "sound_events", "clap"]:
            assert model in detected
        assert "transcribe" not in detected

    def test_segment_models_detected(self):
        cols = _make_transcript_df().columns.tolist()
        assert detect_models_in_dataframe(cols, level="segment") == ["transcribe"]

    def test_speech_prefixes_do_not_cross_match(self):
        # speech_prob is exact; speech_emotion_* must not pull it in
        assert get_model_columns(["speech_prob"], "speech_emotion") == []
        assert get_model_columns(
            ["speech_emotion_valence", "speech_emotion_arousal"], "speech"
        ) == []
        detected = detect_models_in_dataframe(["speech_emotion_valence"])
        assert detected == ["speech_emotion"]


class TestScalarColDetection:
    def test_embedding_cols_excluded(self):
        assert not _is_scalar_col("clap_000")
        assert not _is_scalar_col("clap_511")

    def test_scalar_cols_included(self):
        assert _is_scalar_col("loudness_rms")
        assert _is_scalar_col("timbre_mfcc_01")  # 2-digit tail, named column
        assert _is_scalar_col("sound_events_siren_alarm")
        assert _is_scalar_col("egemaps_f0_semitone")


class TestInferHop:
    def test_sidecar_wins(self):
        df = _make_frames_df()
        assert _infer_hop(df, {"frames": {"hop_sec": 0.25}}) == 0.25

    def test_time_diff_fallback(self):
        assert _infer_hop(_make_frames_df(), None) == 0.5

    def test_default(self):
        df = pd.DataFrame({"time": [0.25]})
        assert _infer_hop(df, None) == 0.5


class TestCreateDashboard:
    def test_returns_html(self):
        html = create_dashboard(_make_frames_df(), _make_transcript_df())
        assert html.startswith("<!DOCTYPE html>")
        assert "plotly" in html.lower()

    def test_embeds_models_and_hop(self):
        html = create_dashboard(_make_frames_df(), _make_transcript_df())
        for model in ["loudness", "pitch", "spectral", "onsets", "speech",
                      "music_emotion", "sound_events", "clap", "transcribe"]:
            assert f'"{model}"' in html
        assert '"hop":0.5' in html

    def test_embedding_dims_not_in_feature_table(self):
        html = create_dashboard(_make_frames_df(), None)
        assert '"clap_000"' not in html

    def test_detail_panels_configured(self):
        html = create_dashboard(_make_frames_df(), _make_transcript_df())
        for panel in ["loudness", "pitch", "sound_events", "confidence"]:
            assert f'"id":"{panel}"' in html
        assert '"kind":"bars_sorted"' in html  # sound_events raw cosines
        assert '"kind":"bars_prob"' in html  # whisper confidence

    def test_frames_only(self):
        html = create_dashboard(_make_frames_df(), None)
        assert '"loudness"' in html
        assert '"transcribe"' not in html

    def test_transcript_only(self):
        html = create_dashboard(None, _make_transcript_df())
        assert '"transcribe"' in html
        assert "Sentence number 0." in html

    def test_no_input_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            create_dashboard(None, None)

    def test_no_models_raises(self):
        df = pd.DataFrame({"time": [0.25, 0.75, 1.25], "unrelated": [1, 2, 3]})
        with pytest.raises(ValueError, match="No aud2psy model outputs"):
            create_dashboard(df, None)

    def test_truncation(self):
        html = create_dashboard(_make_frames_df(n=40), None, max_points=20)
        assert '"frames":40' in html  # recorded original size
        assert '"time":[0.25,' in html

    def test_nan_rows_dropped_from_projection(self):
        frames = _make_frames_df()
        # Simulate VAD-gated frames: NaN rows must be dropped, not crash
        clap_cols = [c for c in frames.columns if c.startswith("clap_")]
        frames.loc[3, clap_cols] = np.nan
        frames.loc[7, clap_cols] = np.nan
        html = create_dashboard(frames, None)
        assert '"clap"' in html

    def test_no_audio_payload(self):
        html = create_dashboard(_make_frames_df(), None)
        assert '"audio":null' in html

    def test_audio_payload_embedded(self):
        audio = {"src": "data:audio/wav;base64,UklGRg==", "duration": 20.0,
                 "envelope": [0.5, 1.0, 0.25]}
        html = create_dashboard(_make_frames_df(), None, audio=audio)
        assert "data:audio/wav;base64,UklGRg==" in html
        assert '"duration":20.0' in html


class TestPrepareAudio:
    def test_wav_roundtrip(self, wav_factory):
        path = wav_factory(sine(440, 2.0))
        audio = prepare_audio(path)
        assert audio["src"].startswith(("data:audio/mpeg;base64,",
                                        "data:audio/wav;base64,"))
        assert audio["duration"] == pytest.approx(2.0, abs=0.01)
        assert 0 < len(audio["envelope"]) <= 1500
        assert max(audio["envelope"]) == pytest.approx(1.0)

    def test_envelope_shapes(self):
        assert _audio_envelope(np.zeros(1000), n_bins=10) == [0.0] * 10
        env = _audio_envelope(np.ones(1000, dtype=np.float32), n_bins=10)
        assert env == [1.0] * 10
        assert len(_audio_envelope(np.ones(5, dtype=np.float32), n_bins=100)) == 5


class TestResolveScoresPaths:
    def _touch(self, tmp_path, *names):
        for name in names:
            (tmp_path / name).write_text("x")

    def test_base_path(self, tmp_path):
        self._touch(tmp_path, "scores_frames.csv", "scores_transcript.csv",
                    "scores.meta.json")
        frames, transcript, meta = resolve_scores_paths(tmp_path / "scores.csv")
        assert frames == tmp_path / "scores_frames.csv"
        assert transcript == tmp_path / "scores_transcript.csv"
        assert meta == tmp_path / "scores.meta.json"

    def test_frames_path(self, tmp_path):
        self._touch(tmp_path, "scores_frames.csv")
        frames, transcript, meta = resolve_scores_paths(tmp_path / "scores_frames.csv")
        assert frames == tmp_path / "scores_frames.csv"
        assert transcript is None and meta is None

    def test_transcript_path(self, tmp_path):
        self._touch(tmp_path, "scores_transcript.csv")
        frames, transcript, _ = resolve_scores_paths(tmp_path / "scores_transcript.csv")
        assert frames is None
        assert transcript == tmp_path / "scores_transcript.csv"

    def test_missing(self, tmp_path):
        assert resolve_scores_paths(tmp_path / "nope.csv") == (None, None, None)
