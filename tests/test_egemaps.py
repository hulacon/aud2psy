"""egemaps tests.

opensmile ships in the dev extra and runs fully offline (bundled binaries,
no weights, no network), so these run in the default suite; each extraction
test importorskips in case the extra is absent. VAD-bypass tests use the
FakeVAD from the speech_emotion tests to separate LLD accuracy from gating
(those run the Silero gate); the diarize-gate tests inject a turn table.
"""

import numpy as np
import pandas as pd
import pytest

from aud2psy.grid import Grid
from aud2psy.models.egemaps import RAW_TO_CLEAN, VOICED_RAW, EgemapsModel

from conftest import sine, white_noise
from test_speech_emotion import FakeVAD

SR = 16000


def make_loaded_model(vad=None):
    pytest.importorskip("opensmile")
    model = EgemapsModel(speech_gate="silero")
    model.load()
    if vad is not None:
        model.vad = vad
    return model


def test_column_mapping_sanity():
    assert len(RAW_TO_CLEAN) == 25
    cleans = list(RAW_TO_CLEAN.values())
    assert len(set(cleans)) == 25
    for clean in cleans:
        assert clean.startswith("egemaps_")
        # CSV/psytwill-friendly: no '-', '.', or capitals survive the rename
        assert clean.replace("_", "").isalnum() and clean == clean.lower()
    assert set(VOICED_RAW) == {n for n in RAW_TO_CLEAN if n.endswith("_sma3nz")}
    assert len(VOICED_RAW) == 15


def test_sine_f0_semitone_matches_ground_truth():
    model = make_loaded_model(vad=FakeVAD(lambda t: np.ones_like(t)))
    y = sine(220.0, 2.0, sr=SR)
    grid = Grid.for_duration(2.0, 0.5)
    feats = model.extract(y, SR, grid)
    assert set(feats) == set(RAW_TO_CLEAN.values())
    assert all(len(v) == grid.n_windows for v in feats.values())
    # 12 * log2(220 / 27.5) = 36.00 semitones exactly
    f0 = feats["egemaps_f0_semitone"]
    assert np.nanmedian(f0) == pytest.approx(36.0, abs=0.1)
    # a pure sine is maximally periodic: tiny jitter, strongly harmonic
    assert np.nanmedian(feats["egemaps_jitter"]) < 0.02
    assert np.nanmedian(feats["egemaps_hnr"]) > 10.0


def test_unvoiced_zero_sentinel_becomes_nan():
    model = make_loaded_model(vad=FakeVAD(lambda t: np.ones_like(t)))
    y = white_noise(2.0, sr=SR)
    feats = model.extract(y, SR, Grid.for_duration(2.0, 0.5))
    # pitch-dependent columns: all-zero sentinel rows -> NaN even ungated
    for col in ("egemaps_f0_semitone", "egemaps_jitter", "egemaps_shimmer", "egemaps_hnr"):
        assert np.isnan(feats[col]).all(), col
    # LPC formant frequency and the general-audio set still measure noise
    assert np.isfinite(feats["egemaps_f1_freq"]).all()
    assert np.isfinite(feats["egemaps_loudness"]).all()
    assert np.isfinite(feats["egemaps_mfcc1"]).all()


def test_loudness_tracks_amplitude():
    model = make_loaded_model(vad=FakeVAD(lambda t: np.ones_like(t)))
    y = np.concatenate([sine(220.0, 2.0, amp=0.05, sr=SR), sine(220.0, 2.0, amp=0.5, sr=SR)])
    feats = model.extract(y, SR, Grid.for_duration(4.0, 1.0))
    loud = feats["egemaps_loudness"]
    assert loud[3] > loud[0] * 2


def test_real_vad_gates_voiced_set_only():
    # a pure tone is not speech: Silero gates the voiced set, general set survives
    model = make_loaded_model()
    y = sine(220.0, 4.0, sr=SR)
    feats = model.extract(y, SR, Grid.for_duration(4.0, 0.5))
    for raw in VOICED_RAW:
        assert np.isnan(feats[RAW_TO_CLEAN[raw]]).all(), raw
    assert np.isfinite(feats["egemaps_loudness"]).all()
    assert np.isfinite(feats["egemaps_alpha_ratio"]).all()


def test_pipeline_integration_and_sidecar(wav_factory):
    pytest.importorskip("opensmile")
    from aud2psy.pipeline import score_audio

    path = wav_factory(sine(220.0, 3.0, sr=SR), sr=SR)
    res = score_audio(path, ["egemaps"], hop=0.5, speech_gate="silero", show_progress=False)
    assert list(res.frames_df.columns) == ["time"] + list(RAW_TO_CLEAN.values())
    meta = res.meta["models"]["egemaps"]
    # 25 named columns must be listed, not squashed into an embedding pattern
    assert meta["columns"] == list(RAW_TO_CLEAN.values())
    assert "pattern" not in meta
    assert meta["feature_set"] == "eGeMAPSv02"
    assert meta["opensmile_names"]["egemaps_jitter"] == "jitterLocal_sma3nz"
    assert meta["n_gated_windows"] == res.frames_df["egemaps_f0_semitone"].isna().sum()
    assert meta["speech_gate"] == "silero" and meta["speech_gate_min"] == 0.25
    assert "speech_gate_turns_source" not in meta


def _turns(*spans):
    return pd.DataFrame({"turn_idx": range(len(spans)), "speaker": "SPEAKER_00",
                         "onset": [a for a, _ in spans], "offset": [b for _, b in spans]})


def test_diarize_gate_follows_turn_coverage():
    # a voiced tone throughout; turns say only the first 2 s are speech
    pytest.importorskip("opensmile")
    model = EgemapsModel()  # diarize is the default gate
    model.load()
    assert not hasattr(model, "vad")  # Silero is not loaded for the diarize gate
    model.turns_ = _turns((0.0, 2.0), (2.0, 2.1))  # 2.0-2.1: 20 % of a window, below .25
    grid = Grid.for_duration(4.0, 0.5)
    feats = model.extract(sine(220.0, 4.0, sr=SR), SR, grid)
    f0 = feats["egemaps_f0_semitone"]
    assert np.isfinite(f0[:4]).all() and np.isnan(f0[4:]).all()
    assert np.isfinite(feats["egemaps_loudness"]).all()  # the general set is never gated
    assert model.info_["speech_gate"] == "diarize"
    assert model.info_["n_gated_windows"] == 4


def test_diarize_gate_without_turns_fails_with_the_fix(wav_factory):
    pytest.importorskip("opensmile")
    from aud2psy.exceptions import Aud2PsyError
    from aud2psy.pipeline import score_audio

    path = wav_factory(sine(220.0, 2.0, sr=SR), sr=SR)
    with pytest.raises(Aud2PsyError, match="speakers_csv.*speech_gate='silero'"):
        score_audio(path, ["egemaps"], show_progress=False)


def test_pipeline_diarize_gate_from_speakers_csv(wav_factory, tmp_path):
    pytest.importorskip("opensmile")
    from aud2psy.pipeline import score_audio

    path = wav_factory(sine(220.0, 3.0, sr=SR), sr=SR)
    csv = tmp_path / "clip_speakers.csv"
    _turns((1.0, 3.0)).assign(stimulus_id="clip").to_csv(csv, index=False)
    res = score_audio(path, ["egemaps"], speakers_csv=csv, show_progress=False)
    f0 = res.frames_df["egemaps_f0_semitone"]
    assert f0.isna().tolist() == [True, True, False, False, False, False]
    meta = res.meta["models"]["egemaps"]
    assert meta["speech_gate"] == "diarize"
    assert meta["speech_gate_turns_source"] == str(csv)
    assert meta["n_gated_windows"] == 2


def test_embedding_pattern_detection():
    from aud2psy.pipeline import _is_embedding

    assert _is_embedding("clap", [f"clap_{i:03d}" for i in range(512)])
    assert not _is_embedding("egemaps", list(RAW_TO_CLEAN.values()))
    assert not _is_embedding("clap", [f"clap_{i:03d}" for i in range(512)] + ["clap_extra"])
