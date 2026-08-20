"""Pipeline, CLI, and I/O layout tests (offline; frame models only)."""

import json
import subprocess
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from aud2psy.audio import input_type, load_audio
from aud2psy.cli import MODEL_REGISTRY, main
from aud2psy.exceptions import InputError
from aud2psy import pipeline
from aud2psy.pipeline import save_result, score_audio

from conftest import SR, sine

FRAME_MODELS = ["loudness", "spectral", "onsets"]


def test_score_audio_frames_shape(wav_factory):
    path = wav_factory(sine(220, 3.0))
    result = score_audio(path, FRAME_MODELS, show_progress=False)
    df = result.frames_df
    assert len(df) == 6
    np.testing.assert_allclose(df["time"], [0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
    assert list(df.columns) == [
        "time", "loudness_rms", "loudness_db",
        "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "spectral_flux", "spectral_zcr",
        "onsets_strength", "onsets_rate", "onsets_tempo",
    ]
    assert result.transcript_df is None


def test_score_audio_custom_hop(wav_factory):
    path = wav_factory(sine(220, 2.0))
    result = score_audio(path, ["loudness"], hop=0.25, show_progress=False)
    assert len(result.frames_df) == 8
    assert result.meta["frames"]["hop_sec"] == 0.25


def test_score_audio_unknown_model(wav_factory):
    path = wav_factory(sine(220, 1.0))
    with pytest.raises(KeyError):
        score_audio(path, ["nope"], show_progress=False)


def test_save_result_file_layout(wav_factory, tmp_path):
    path = wav_factory(sine(220, 2.0))
    result = score_audio(path, ["loudness"], show_progress=False)
    written = save_result(result, tmp_path / "scores.csv")
    assert written["frames"].name == "scores_frames.csv"
    assert written["meta"].name == "scores.meta.json"
    meta = json.loads(written["meta"].read_text())
    assert meta["schema_version"] == "1.0"
    assert meta["extractor"] == "aud2psy"
    assert meta["extractor_version"] == meta["aud2psy_version"]  # legacy key kept
    assert meta["input"]["duration_sec"] == 2.0
    assert meta["models"]["loudness"]["columns"] == ["loudness_rms", "loudness_db"]
    assert meta["models"]["loudness"]["package_version"] not in (None, "unknown")
    assert meta["models"]["loudness"]["checkpoint"] is None  # analytic DSP model
    assert meta["output"]["frames"]["rows"] == 4
    df = pd.read_csv(written["frames"])
    assert df.shape == (4, 4)
    assert df.columns[0] == "stimulus_id"
    assert set(df["stimulus_id"]) == {path.stem}


def test_cli_end_to_end(wav_factory, tmp_path, capsys):
    path = wav_factory(sine(220, 2.0))
    out = tmp_path / "scores.csv"
    assert main(["loudness", "spectral", str(path), "-o", str(out)]) == 0
    assert (tmp_path / "scores_frames.csv").exists()
    assert (tmp_path / "scores.meta.json").exists()
    assert not (tmp_path / "scores_transcript.csv").exists()


def test_cli_stdout_when_no_output(wav_factory, capsys):
    path = wav_factory(sine(220, 1.0))
    assert main(["loudness", str(path)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "time,loudness_rms,loudness_db"
    assert len(lines) == 3


def test_cli_rejects_unknown_model(wav_factory):
    path = wav_factory(sine(220, 1.0))
    with pytest.raises(SystemExit):
        main(["nonsense", str(path)])


def test_cli_requires_models(wav_factory):
    path = wav_factory(sine(220, 1.0))
    with pytest.raises(SystemExit):
        main([str(path)])


def test_input_type_validation(tmp_path):
    with pytest.raises(InputError):
        input_type(tmp_path / "missing.wav")
    bad = tmp_path / "notes.txt"
    bad.write_text("hi")
    with pytest.raises(InputError):
        input_type(bad)


def test_load_audio_resamples(wav_factory):
    path = wav_factory(sine(220, 1.0))
    y = load_audio(path, 16000)
    assert abs(len(y) - 16000) <= 1
    assert y.dtype == np.float32


def test_help_is_fast():
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "-m", "aud2psy.cli", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0
    # heavy deps must not be imported for --help
    for lib in ("librosa", "faster_whisper", "pandas", "mosqito", "matplotlib"):
        assert lib not in proc.stderr, f"--help imported {lib}"


def test_registry_modules_all_exist():
    import importlib

    for name, (module_path, class_name, _) in MODEL_REGISTRY.items():
        cls = getattr(importlib.import_module(module_path), class_name)
        assert cls.name == name


def test_window_sec_declared_where_context_exceeds_the_hop():
    """Contract B 4.1: a time-resolved row must state how much it saw.

    Models using the `window >> hop` pattern score a long context window
    centered on each grid midpoint, so adjacent rows overlap heavily. That
    is invisible to a consumer unless the sidecar says so.
    """
    import importlib

    expected = {
        "clap": 10.0,
        "sound_events": 10.0,  # inherits ClapModel's window
        "music_emotion": 10.0,  # same
        "ebind_audio": 2.0,
        "speech_emotion": 4.0,
    }
    for name, (module_path, class_name, _) in MODEL_REGISTRY.items():
        cls = getattr(importlib.import_module(module_path), class_name)
        assert getattr(cls, "window_sec", None) == expected.get(name), (
            f"{name}: window_sec is {getattr(cls, 'window_sec', None)}, "
            f"expected {expected.get(name)}. A model whose every column shares "
            f"one context window must declare it; one whose rows see only "
            f"their own hop window must leave it None."
        )


def test_window_sec_reaches_the_sidecar(wav_factory, tmp_path):
    """The declared window has to survive into the written metadata."""
    from aud2psy.models.base import BaseModel

    class FakeWindowed(BaseModel):
        name = "loudness"  # ride an existing registry slot
        level = "frame"
        window_sec = 7.5

        def extract(self, y, sr, grid):
            return {"loudness_rms": np.zeros(grid.n_windows)}

    path = wav_factory(sine(440, 2.0))
    with mock.patch.object(pipeline, "get_model", return_value=FakeWindowed()):
        result = score_audio(path, models=["loudness"], hop=0.5)
    assert result.meta["models"]["loudness"]["window_sec"] == 7.5
    assert result.meta["frames"]["hop_sec"] == 0.5
    assert result.meta["frames"]["time"] == "window center"


def test_music_emotion_probe_ships_and_loads():
    # offline: the committed probe artifact is well-formed
    from aud2psy.models.music_emotion import load_probe

    coef, intercept, provenance = load_probe()
    assert coef.shape == (2, 512)
    assert intercept.shape == (2,)
    assert provenance["dimensions"] == ["valence", "arousal"]
    assert provenance["checkpoint"] == "laion/larger_clap_music_and_speech"
    assert "cv" in provenance and "arousal" in provenance["cv"]


def test_annotate_voice_gender():
    from aud2psy.pipeline import annotate_voice_gender

    transcript = pd.DataFrame({
        "text": ["male line", "female line", "unvoiced"],
        "onset": [0.0, 3.0, 6.0],
        "offset": [2.0, 5.0, 7.0],
    })
    frames = pd.DataFrame({
        "time": np.arange(0.25, 7.5, 0.5),
        "pitch_f0": [115, 118, 112, np.nan,          # segment 1: male
                     np.nan, np.nan, 205, 198, 210,  # segment 2: female
                     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    })
    annotate_voice_gender(transcript, frames, hop=0.5)
    assert list(transcript["voice_gender"]) == ["M", "F", ""]
    assert transcript["median_f0"][0] == 115
    assert transcript["median_f0"][1] == 205
    assert np.isnan(transcript["median_f0"][2])
