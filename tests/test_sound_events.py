"""sound_events: offline tests of the scoring machinery (no CLAP weights).

Real-embedding validation cannot run in CI (weights + network): the
`weights`-marked test at the bottom and scripts/validate_sound_events.py
cover it on a machine with the checkpoint available.
"""

import re

import numpy as np
import pytest

from aud2psy.grid import Grid
from aud2psy.models.sound_events import PROMPT_BANK, SoundEventsModel, ensemble_bank


def test_bank_shape_and_hygiene():
    assert len(PROMPT_BANK) == 16
    for cat, prompts in PROMPT_BANK.items():
        assert re.fullmatch(r"[a-z][a-z_]*", cat)  # CSV/psyquilt-safe keys
        assert len(prompts) >= 2
    assert "laughter" in PROMPT_BANK and "music" in PROMPT_BANK
    # the recorded negative finding: no speaker-attribute prompts
    banned = ("man ", "woman", "male", "female", " man", "gender")
    for prompts in PROMPT_BANK.values():
        for p in prompts:
            assert not any(b in p.lower() for b in banned), p


def test_ensemble_bank_math():
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])
    bank = ensemble_bank(np.stack([e1, e2, e3]), sizes=[2, 1])
    np.testing.assert_allclose(bank[0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0.0])
    np.testing.assert_allclose(bank[1], e3)
    np.testing.assert_allclose(np.linalg.norm(bank, axis=1), 1.0)


def test_extract_scores_are_cosines(monkeypatch):
    model = SoundEventsModel()
    n_cat = len(PROMPT_BANK)
    rng = np.random.default_rng(0)
    bank = rng.standard_normal((n_cat, 512))
    bank /= np.linalg.norm(bank, axis=1, keepdims=True)
    model.bank_ = bank

    # 3 windows: aligned with category 0, anti-aligned, orthogonal-ish
    emb = np.stack([bank[0], -bank[0], bank[1]])
    monkeypatch.setattr(
        SoundEventsModel, "embed_windows", lambda self, y, sr, centers: emb
    )
    grid = Grid.for_duration(1.5, 0.5)
    out = model.extract(np.zeros(16), 48000, grid)

    assert list(out) == [f"sound_events_{c}" for c in PROMPT_BANK]
    first = f"sound_events_{list(PROMPT_BANK)[0]}"
    np.testing.assert_allclose(out[first], [1.0, -1.0, bank[1] @ bank[0]], atol=1e-12)
    assert all(len(v) == 3 for v in out.values())
    assert model.info_["prompts"] is PROMPT_BANK
    assert "cosine" in model.info_["scoring"]


def test_registry_and_embedding_heuristic():
    from aud2psy.cli import MODEL_REGISTRY
    from aud2psy.pipeline import _is_embedding

    assert "sound_events" in MODEL_REGISTRY
    # 16 *named* columns must be listed in the sidecar, not squashed into
    # a clap-style numbered pattern
    names = [f"sound_events_{c}" for c in PROMPT_BANK]
    assert not _is_embedding("sound_events", names)


@pytest.mark.weights
def test_zero_shot_separation_on_synthetic_stimuli():
    """Needs the ~2 GB CLAP checkpoint — run locally, not in CI."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from validate_sound_events import SR_CLAP, synthetic_stimuli

    model = SoundEventsModel()
    model.load()
    stimuli = synthetic_stimuli()
    scores = {}
    for name, y in stimuli.items():
        grid = Grid.for_duration(len(y) / SR_CLAP, 0.5)
        out = model.extract(y, SR_CLAP, grid)
        scores[name] = {c: float(np.mean(v)) for c, v in out.items()}
    model.unload()

    # each synthesizable target's column should peak on its own stimulus
    for stim, cat in [
        ("rain", "water"),
        ("siren", "siren_alarm"),
        ("music", "music"),
        ("wind", "wind"),
        ("applause", "applause"),
        ("thunder", "thunder"),
        ("gunshots", "gunshot_explosion"),
    ]:
        col = f"sound_events_{cat}"
        best = max(scores, key=lambda s: scores[s][col])
        assert best == stim, f"{col} peaked on {best}, not {stim}: {scores[best][col]:.3f}"
