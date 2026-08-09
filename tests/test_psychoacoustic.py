"""Closed-form-anchor checks for the psychoacoustic model.

The anchors are the textbook Zwicker/Fastl orderings: roughness peaks
near 70 Hz amplitude modulation, fluctuation strength near 4 Hz (the
complementary pair), sharpness rises with high-frequency content, and
loudness is monotone in level. Signals are short (2-3 s) because
loudness_zwtv runs ~2.5x real time on CPU — fixtures are module-scoped
so each signal is analyzed once. The fluctuation estimator also gets
direct offline unit tests on synthetic specific-loudness envelopes
(no MoSQITo in the loop).
"""

import numpy as np
import pytest

from aud2psy.grid import Grid
from aud2psy.models.psychoacoustic import (
    FLUCT_CAL,
    PsychoacousticModel,
    fluctuation_estimate,
    fluctuation_weight,
)

FS = 48000  # the model's declared input_sr

COLUMNS = [
    "psychoacoustic_loudness",
    "psychoacoustic_sharpness",
    "psychoacoustic_roughness",
    "psychoacoustic_fluctuation",
]


def extract(y, hop=0.5):
    grid = Grid.for_duration(len(y) / FS, hop)
    model = PsychoacousticModel()
    model.load()
    out = model.extract(y, FS, grid)
    return out, model


def am_tone(fc, fmod, dur, rms=0.02):
    """100% sinusoidally amplitude-modulated tone at a given RMS."""
    t = np.arange(int(FS * dur)) / FS
    y = (1 + np.sin(2 * np.pi * fmod * t)) * np.sin(2 * np.pi * fc * t)
    return y / np.sqrt(np.mean(y**2)) * rms


def tone(freq, dur, amp):
    t = np.arange(int(FS * dur)) / FS
    return amp * np.sin(2 * np.pi * freq * t)


@pytest.fixture(scope="module")
def steps():
    """1 s soft tone + 1 s loud tone + 1 s silence -> 6 windows."""
    y = np.concatenate([tone(1000, 1.0, 0.05), tone(1000, 1.0, 0.4), np.zeros(FS)])
    return extract(y)


@pytest.fixture(scope="module")
def am():
    return {fmod: extract(am_tone(1000, fmod, 2.0))[0] for fmod in (20, 70)}


@pytest.fixture(scope="module")
def am4():
    # Fastl's fluctuation reference: 1 kHz, 60 dB SPL (0.02 Pa RMS at the
    # model's 94 dB full-scale mapping), 100% AM at 4 Hz
    return extract(am_tone(1000, 4, 3.0))[0]


def test_columns_and_sidecar_info(steps):
    out, model = steps
    assert list(out) == COLUMNS
    assert all(len(v) == 6 for v in out.values())
    assert model.info_["sharpness_min_sone"] == 0.25
    assert "not standardized" in model.info_["fluctuation"]


def test_loudness_monotone_with_zwicker_decay(steps):
    out, _ = steps
    loud = out["psychoacoustic_loudness"]
    assert loud[2] > 2 * loud[0] > 0  # 8x amplitude -> much louder
    # ISO 532-1 loudness decays after offset rather than dropping to 0:
    # the first silent window still carries decay, the second is quiet
    assert loud[3] > loud[4] > loud[5]
    assert loud[5] < 0.05


def test_sharpness_nan_only_where_decay_has_faded(steps):
    out, model = steps
    sharp = out["psychoacoustic_sharpness"]
    assert np.isfinite(sharp[:4]).all()
    assert np.isnan(sharp[5])
    assert model.info_["n_sharpness_gated_windows"] == int(np.isnan(sharp).sum()) >= 1


def test_sharpness_high_vs_low_passed_noise():
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(0)
    noise = rng.standard_normal(2 * FS)
    lo = sosfilt(butter(4, 1000, "low", fs=FS, output="sos"), noise)
    hi = sosfilt(butter(4, 4000, "high", fs=FS, output="sos"), noise)
    vals = {}
    for name, y in (("lo", lo), ("hi", hi)):
        y = y / np.sqrt(np.mean(y**2)) * 0.02
        out, _ = extract(y)
        vals[name] = np.nanmean(out["psychoacoustic_sharpness"])
    # design-time measurement: 4.3 vs 0.8 acum
    assert vals["hi"] > 3 * vals["lo"]


def test_roughness_peaks_near_70hz_modulation(am):
    r70 = np.mean(am[70]["psychoacoustic_roughness"])
    r20 = np.mean(am[20]["psychoacoustic_roughness"])
    assert r70 > 3 * r20  # design-time: 1.36 vs 0.27 asper


def test_fluctuation_reference_reads_one_vacil(am4):
    mid = slice(1, -1)
    np.testing.assert_allclose(am4["psychoacoustic_fluctuation"][mid], 1.0, rtol=0.15)


def test_fluctuation_and_roughness_are_complementary(am, am4):
    # FS peaks at 4 Hz modulation and is gone by 70 Hz; roughness inverts
    f4 = np.mean(am4["psychoacoustic_fluctuation"])
    f70 = np.mean(am[70]["psychoacoustic_fluctuation"])
    assert f4 > 5 * f70
    r4 = np.mean(am4["psychoacoustic_roughness"])
    r70 = np.mean(am[70]["psychoacoustic_roughness"])
    assert r70 > 5 * r4


# --- fluctuation estimator unit tests (offline, synthetic envelopes) ---


def synthetic_n_specific(mod_hz, depth, dur=4.0, dt=0.002, bands=slice(40, 60)):
    """240-bin specific loudness: 1 sone in `bands`, sinusoidally modulated."""
    t = np.arange(int(dur / dt)) * dt
    n_spec = np.zeros((240, len(t)))
    n_spec[bands] = 1.0 + depth * np.sin(2 * np.pi * mod_hz * t)
    return n_spec


def test_fluctuation_estimate_zero_on_constant_envelope():
    n_spec = synthetic_n_specific(4.0, depth=0.0)
    out = fluctuation_estimate(n_spec, 0.002, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out, 0.0)


def test_fluctuation_estimate_prefers_4hz_and_scales_with_depth():
    centers = np.array([2.0])
    at_4 = fluctuation_estimate(synthetic_n_specific(4.0, 0.5), 0.002, centers)[0]
    at_16 = fluctuation_estimate(synthetic_n_specific(16.0, 0.5), 0.002, centers)[0]
    shallow = fluctuation_estimate(synthetic_n_specific(4.0, 0.25), 0.002, centers)[0]
    assert at_4 > 2 * at_16 > 0  # weight(16 Hz) ~= 0.47 of the 4 Hz peak
    np.testing.assert_allclose(at_4 / shallow, 2.0, rtol=0.05)  # linear in depth


def test_fluctuation_weight_shape():
    f = np.array([0.0, 0.5, 2.0, 4.0, 8.0, 70.0])
    w = fluctuation_weight(f)
    assert w[0] == 0.0
    assert w[3] == 1.0  # peak at 4 Hz
    np.testing.assert_allclose(w[2], w[4])  # symmetric in log-frequency
    assert w[1] < 0.3 and w[5] < 0.15
