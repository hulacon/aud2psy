"""ebind_audio: window geometry and naming offline; real weights gated.

The compute path (EBind forward) is stubbed for the default suite; the
`weights` marker gates one real-checkpoint smoke test, following the CLAP
convention.
"""

import numpy as np
import pytest

from aud2psy.grid import Grid
from aud2psy.models.ebind_audio import (
    EMBED_DIM,
    WINDOW_SEC,
    EBindAudioModel,
    window_starts,
)

from conftest import sine

SR = 16000
CENTERS = np.array([0.25, 0.75, 1.25])


def test_window_starts_edge_clamped():
    n = SR * 3  # 3 s of audio
    starts = window_starts(n, SR, np.array([0.0, 1.5, 3.0]))
    assert starts[0] == 0  # clamped at the head
    assert starts[1] == int((1.5 - WINDOW_SEC / 2) * SR)
    assert starts[2] == int((3.0 - WINDOW_SEC) * SR)  # clamped at the tail


def test_extract_names_and_shapes():
    model = EBindAudioModel(device="cpu")

    def fake_embed(y, sr, centers):
        return np.tile(np.eye(1, EMBED_DIM), (len(centers), 1))

    model.embed_windows = fake_embed
    y = sine(220, 2.0, sr=SR)
    grid = Grid.for_duration(len(y) / SR, 0.5)
    out = model.extract(y, SR, grid)
    assert len(out) == EMBED_DIM
    assert sorted(out)[0] == "ebind_audio_0000"
    assert sorted(out)[-1] == "ebind_audio_1023"
    assert all(len(v) == grid.n_windows for v in out.values())


@pytest.mark.weights
def test_real_embeddings_unit_norm():
    model = EBindAudioModel(device="cpu")
    model.load()
    y = sine(220, 3.0, sr=SR)
    out = model.embed_windows(y, SR, CENTERS)
    assert out.shape == (3, EMBED_DIM)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-3)
