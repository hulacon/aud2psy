"""phonetics tests.

The phone table, the window reduction, chunking and gating run offline on a
stubbed recognizer. Real wav2vec2 inference (~1.3 GB weights) is behind the
`weights` marker: pytest -m weights
"""

import numpy as np
import pandas as pd
import pytest

from aud2psy.exceptions import Aud2PsyError
from aud2psy.grid import Grid
from aud2psy.models import phonetics
from aud2psy.models.phonetics import (
    CHUNK_SEC,
    COLUMNS,
    HOP_SAMPLES,
    MIN_MASS,
    SR,
    PhoneticsModel,
    load_table,
    table_matrices,
    window_features,
)
from aud2psy.models.phonetics_table import MANNERS, PLACES, build_table, segments

from conftest import silence

TABLE = load_table()
MATS = table_matrices(TABLE)
VOCAB = len(TABLE)
IDX = {t: i for i, t in enumerate(TABLE["token"])}


def row(token):
    return TABLE.loc[IDX[token]]


# ---- the shipped table ------------------------------------------------------

def test_shipped_table_is_what_the_parser_builds():
    rebuilt = build_table(list(TABLE["token"]))
    pd.testing.assert_frame_equal(
        rebuilt.drop(columns=["token", "segments", "status"]).astype(float),
        TABLE.drop(columns=["token", "segments", "status"]).astype(float),
        atol=1e-5)
    assert list(rebuilt["status"]) == list(TABLE["status"])
    assert list(rebuilt["segments"].fillna("")) == list(TABLE["segments"])


def test_table_covers_the_checkpoint_vocabulary():
    assert VOCAB == 392
    assert TABLE["token"].iloc[0] == "<pad>"  # the CTC blank is index 0
    # every token with a phone maps fully; only the non-phone tokens are unmapped
    assert set(TABLE.loc[TABLE["status"] != "mapped", "token"]) == {
        "<pad>", "<s>", "</s>", "<unk>", "??", "ʲ", "1"}


def test_row_weights_are_consistent():
    manner = TABLE[list(MANNERS)].sum(axis=1) + TABLE["unmapped"]
    np.testing.assert_allclose(manner, 1.0, atol=1e-6)
    place = TABLE[list(PLACES)].sum(axis=1) + TABLE["place_unmapped"]
    np.testing.assert_allclose(place, TABLE["consonant"], atol=1e-6)
    assert (TABLE["voiced"] <= TABLE["consonant"] + 1e-9).all()


@pytest.mark.parametrize("token,manner,place,voiced", [
    ("tʃ", "affricate", "postalveolar", 0), ("dʒ", "affricate", "postalveolar", 1),
    ("ð", "fricative", "dental", 1), ("θ", "fricative", "dental", 0),
    ("ŋ", "nasal", "velar", 1), ("ʔ", "stop", "glottal", 0), ("h", "fricative", "glottal", 0),
    ("ɾ", "stop", "alveolar", 1),  # the flapped t/d counts as a stop (DECIDED 11-11)
    ("ɹ", "liquid", "alveolar", 1), ("j", "glide", "palatal", 1),
])
def test_english_consonants(token, manner, place, voiced):
    r = row(token)
    assert r[manner] == 1.0 and r[place] == 1.0 and r["voiced"] == voiced


def test_w_is_half_labial_half_velar():
    r = row("w")
    assert r["glide"] == 1.0 and r["labial"] == 0.5 and r["velar"] == 0.5


def test_multi_segment_tokens_split_their_mass():
    r = row("əl")
    assert r["vowel"] == 0.5 and r["liquid"] == 0.5
    assert r["height"] / r["vowel"] == pytest.approx(0.5)  # schwa is mid
    r = row("eɪ")
    assert r["vowel"] == 1.0 and r["height"] == pytest.approx((0.67 + 0.83) / 2)


@pytest.mark.parametrize("token,segs", [
    ("iɜ", ["i"]),          # trailing ɜ after a vowel is Mandarin tone 3
    ("ɜː", ["ɜ"]),          # ... but the English NURSE vowel stays a vowel
    ("onɡ5", ["o", "ŋ"]), ("ts.h", ["tʂ"]), ("kh", ["k"]), ("ää", ["A"]),
    ("dZ", ["dʒ"]), ("??", None), ("1", None),
])
def test_token_parsing(token, segs):
    assert segments(token) == segs


# ---- the window reduction ---------------------------------------------------

def frames(tokens, p=1.0):
    """Posterior rows putting mass p on each token (the rest on the blank)."""
    out = np.zeros((len(tokens), VOCAB))
    for i, t in enumerate(tokens):
        if t == "<pad>":
            out[i, 0] = 1.0
        else:
            out[i, IDX[t]] = p
            out[i, 0] = 1.0 - p
    return out


def test_window_shares_rate_and_entropy():
    # window 0: s s _ z _ ; window 1: i i _ u u  (hop 0.1 s, 5 frames each)
    toks = ["s", "s", "<pad>", "z", "<pad>", "i", "i", "<pad>", "u", "u"]
    win = np.repeat([0, 1], 5)
    f = window_features(frames(toks), win, 2, 0.1, MATS)
    assert f["phonetics_fricative"][0] == 1.0 and f["phonetics_alveolar"][0] == 1.0
    assert f["phonetics_voiced"][0] == pytest.approx(1 / 3)
    assert np.isnan(f["phonetics_vowel_height"][0])  # no vowel mass
    assert f["phonetics_vowel"][1] == 1.0 and np.isnan(f["phonetics_labial"][1])
    assert f["phonetics_vowel_height"][1] == 1.0
    assert f["phonetics_vowel_backness"][1] == pytest.approx(0.5)  # i front, u back
    assert f["phonetics_vowel_rounded"][1] == pytest.approx(0.5)
    assert f["phonetics_rate"][0] == pytest.approx(2 / 0.1)  # s(+s collapsed), z
    assert f["phonetics_rate"][1] == pytest.approx(2 / 0.1)  # i, u
    assert f["phonetics_entropy"][0] == 0.0  # one-hot posteriors


def test_a_pause_inside_speech_has_rate_zero_and_no_shares():
    f = window_features(frames(["<pad>"] * 5), np.zeros(5, int), 1, 0.1, MATS)
    assert f["phonetics_rate"][0] == 0.0
    assert np.isnan(f["phonetics_stop"][0]) and np.isnan(f["phonetics_entropy"][0])


def test_shares_need_min_mass():
    # 0.15 phone mass per frame x 5 frames = .75 < MIN_MASS
    f = window_features(frames(["t"] * 5, p=0.15), np.zeros(5, int), 1, 0.1, MATS)
    assert 0.75 < MIN_MASS and np.isnan(f["phonetics_stop"][0])


def test_window_no_frame_reaches_is_nan_everywhere():
    f = window_features(frames(["t", "t"]), np.array([0, 0]), 2, 0.1, MATS)
    assert all(np.isnan(f[c][1]) for c in COLUMNS)


# ---- chunking and gating on a stubbed recognizer ----------------------------

def stub(turns=None, gate="diarize"):
    model = PhoneticsModel(device="cpu", speech_gate=gate)
    model.mats_ = MATS
    calls = []

    def posteriors(y):
        calls.append(len(y))
        n = 1 + (len(y) - 400) // HOP_SAMPLES
        return frames(["t", "<pad>"] * (n // 2) + ["t"] * (n % 2))

    model.posteriors = posteriors
    model.calls = calls
    model.turns_ = turns
    return model


def speech(on, off):
    return pd.DataFrame({"speaker": ["A"], "onset": [on], "offset": [off]})


def test_every_window_gets_its_frames_once_across_chunks():
    dur = 2.5 * CHUNK_SEC
    model = stub(speech(0.0, dur))
    grid = Grid.for_duration(dur, 0.5)
    feats = model.extract(silence(dur, sr=SR), SR, grid)
    assert model.info_["n_chunks"] == 3 and model.info_["n_chunks_run"] == 3
    # t,_ alternating at 20 ms: 12-13 onsets per 0.5 s window = 24-26 phones/s
    rate = feats["phonetics_rate"][:-1]
    assert np.all((rate >= 24) & (rate <= 26)), np.unique(rate)
    assert np.all(feats["phonetics_stop"][:-1] == 1.0)


def test_gated_windows_are_nan_and_gated_chunks_are_not_run():
    dur = 3 * CHUNK_SEC
    model = stub(speech(0.0, 5.0))  # speech only in the first chunk
    feats = model.extract(silence(dur, sr=SR), SR, Grid.for_duration(dur, 0.5))
    assert model.info_["n_chunks_run"] == 1 and len(model.calls) == 1
    assert np.isfinite(feats["phonetics_stop"][:10]).all()
    assert all(np.isnan(feats[c][10:]).all() for c in COLUMNS)
    assert model.info_["speech_gate_window_sec"] == 0.5


def test_diarize_gate_without_turns_fails_before_inference():
    model = stub(None)
    model.posteriors = lambda y: pytest.fail("inference ran before the gate check")
    with pytest.raises(Aud2PsyError, match="--speakers"):
        model.extract(silence(4.0, sr=SR), SR, Grid.for_duration(4.0, 0.5))


def test_wrong_sample_rate_is_refused():
    with pytest.raises(ValueError, match="16000"):
        stub(speech(0, 1)).extract(silence(1.0, sr=22050), 22050, Grid.for_duration(1.0, 0.5))


def test_registered_and_gated():
    from aud2psy.cli import MODEL_REGISTRY
    from aud2psy.speech_gate import GATED_MODELS

    assert "phonetics" in MODEL_REGISTRY and "phonetics" in GATED_MODELS
    assert set(phonetics.PhoneticsModel.nulls) == set(COLUMNS)


@pytest.mark.weights
def test_checkpoint_vocabulary_matches_the_table():
    model = PhoneticsModel(device="cpu")
    model.load()  # raises if the shipped table and the vocabulary disagree
    p = model.posteriors(silence(2.0, sr=SR))
    assert p.shape[1] == VOCAB and np.allclose(p.sum(axis=1), 1.0, atol=1e-4)
    model.unload()
