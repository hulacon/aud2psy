"""CLAP and beats tests — behind the `weights` marker (large downloads;
warm HF cache on this machine).

Run with: pytest -m weights
"""

import numpy as np
import pytest

from aud2psy.grid import Grid

from conftest import SR, noise_bursts, sine

pytestmark = pytest.mark.weights


def test_clap_shape_norm_and_discrimination():
    from aud2psy.models.clap import ClapModel

    grid = Grid.for_duration(4.0, 0.5)
    model = ClapModel(device="cpu")
    model.load()
    tone = model.extract(sine(220, 4.0, sr=48000), 48000, grid)
    assert len(tone) == 512
    emb_tone = np.column_stack([tone[f"clap_{i:03d}"] for i in range(512)])
    assert emb_tone.shape == (8, 512)
    np.testing.assert_allclose(np.linalg.norm(emb_tone, axis=1), 1.0, atol=1e-4)
    # identical windows (steady content, edge-clamped) give identical rows
    assert np.allclose(emb_tone[0], emb_tone[-1], atol=1e-5)
    # a different sound lands elsewhere in the space
    clicks = model.extract(
        noise_bursts(list(np.arange(0, 4, 0.5)), 4.0, sr=48000), 48000, grid
    )
    emb_clicks = np.column_stack([clicks[f"clap_{i:03d}"] for i in range(512)])
    assert float(emb_tone[0] @ emb_clicks[0]) < 0.9


def test_beats_on_click_track(wav_factory):
    from aud2psy.models.beats import BeatsModel

    y = noise_bursts(list(np.arange(0.0, 8.0, 0.5)), duration=8.0, amp=0.8)
    model = BeatsModel()
    model.load()
    beats_df, info = model.detect(y, SR)
    assert info["n_beats"] >= 12
    assert abs(info["tempo_bpm_median"] - 120.0) < 3
    ibis = np.diff(beats_df["time"])
    assert np.abs(ibis - 0.5).max() < 0.06
    assert set(beats_df.columns) == {"time", "is_downbeat"}


def test_music_emotion_on_music_vs_silence():
    from aud2psy.models.music_emotion import MusicEmotionModel

    grid = Grid.for_duration(4.0, 0.5)
    model = MusicEmotionModel(device="cpu")
    model.load()
    out = model.extract(sine(220, 4.0, sr=48000), 48000, grid)
    assert set(out) == {"music_emotion_valence", "music_emotion_arousal"}
    for v in out.values():
        assert v.shape == (8,)
        assert np.isfinite(v).all()
        assert (np.abs(v) <= 1.0).all()
