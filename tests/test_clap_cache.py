"""The per-run CLAP embedding cache: clap-family models in one pipeline
call share a single forward pass. Fully offline — the compute path is
stubbed; only the caching layer and pipeline wiring are under test."""

import numpy as np

from aud2psy.grid import Grid
from aud2psy.models.clap import EMBED_DIM, ClapModel

from conftest import SR, sine

CENTERS = np.array([0.25, 0.75, 1.25])


def make_counting_model(counter, checkpoint="ckpt-a"):
    model = ClapModel(checkpoint=checkpoint, device="cpu")

    def fake_compute(y, sr, centers):
        counter.append(checkpoint)
        emb = np.tile(np.eye(1, EMBED_DIM), (len(centers), 1))
        return emb

    model._embed_windows = fake_compute
    return model


def test_shared_cache_computes_once():
    counter = []
    cache = {}
    a = make_counting_model(counter)
    b = make_counting_model(counter)
    a.cache_ = cache
    b.cache_ = cache
    y = np.zeros(100)
    out_a = a.embed_windows(y, 48000, CENTERS)
    out_b = b.embed_windows(y, 48000, CENTERS)
    assert len(counter) == 1
    assert out_b is out_a  # the cached array itself


def test_cache_misses_on_different_checkpoint_or_centers():
    counter = []
    cache = {}
    a = make_counting_model(counter, "ckpt-a")
    b = make_counting_model(counter, "ckpt-b")
    a.cache_ = cache
    b.cache_ = cache
    y = np.zeros(100)
    a.embed_windows(y, 48000, CENTERS)
    b.embed_windows(y, 48000, CENTERS)  # other checkpoint -> miss
    a.embed_windows(y, 48000, CENTERS * 2)  # other grid -> miss
    assert len(counter) == 3


def test_no_cache_attached_recomputes():
    counter = []
    model = make_counting_model(counter)
    y = np.zeros(100)
    model.embed_windows(y, 48000, CENTERS)
    model.embed_windows(y, 48000, CENTERS)
    assert len(counter) == 2


def test_pipeline_shares_cache_across_clap_family(monkeypatch, wav_factory):
    from aud2psy import pipeline

    counter = []

    class StubClap(ClapModel):
        def load(self):
            pass

        def unload(self):
            pass

        def _embed_windows(self, y, sr, centers):
            counter.append(self.checkpoint)
            return np.ones((len(centers), EMBED_DIM)) / np.sqrt(EMBED_DIM)

    class StubSoundEvents(StubClap):
        name = "sound_events"

        def extract(self, y, sr, grid):
            emb = self.embed_windows(y, sr, grid.centers)
            return {"sound_events_music": emb[:, 0]}

    def fake_get_model(name, **kwargs):
        return StubClap(**kwargs) if name == "clap" else StubSoundEvents()

    monkeypatch.setattr(pipeline, "get_model", fake_get_model)
    path = wav_factory(sine(220, 1.5))
    result = pipeline.score_audio(path, ["clap", "sound_events"], show_progress=False)
    assert len(counter) == 1  # one forward pass for both models
    assert "clap_000" in result.frames_df and "sound_events_music" in result.frames_df


def test_music_emotion_silence_gate(monkeypatch):
    """music_emotion NaNs digital silence (the speech_emotion precedent).
    Offline: the probe is package data, only the embedder is stubbed.

    On real weights, silence yields valence -.294 / arousal -.245 —
    inside the range genuine music produces, so an ungated value is
    indistinguishable from a real reading downstream."""
    from aud2psy.models.music_emotion import MusicEmotionModel

    model = MusicEmotionModel(device="cpu")
    model.coef, model.intercept, model.provenance = __import__(
        "aud2psy.models.music_emotion", fromlist=["load_probe"]
    ).load_probe()
    emb = np.tile(np.eye(1, EMBED_DIM), (3, 1))
    monkeypatch.setattr(
        MusicEmotionModel, "embed_windows", lambda self, y, sr, centers: emb
    )
    grid = Grid.for_duration(1.5, 0.5)

    out = model.extract(np.zeros(48000, dtype=np.float32), 48000, grid)
    assert all(np.isnan(v).all() for v in out.values())
    assert model.info_["n_windows_silence_gated"] == 3

    out = model.extract(sine(220, 1.0, sr=48000), 48000, grid)
    assert all(np.isfinite(v).all() for v in out.values())
    assert model.info_["n_windows_silence_gated"] == 0
