"""Ground-truth checks for the timbre model on synthetic audio."""

import numpy as np

from aud2psy.grid import Grid
from aud2psy.models.timbre import TimbreModel

from conftest import SR, silence, sine, white_noise

MID = slice(1, -1)  # skip edge windows, the test_models convention


def run(y, hop=0.5):
    grid = Grid.for_duration(len(y) / SR, hop)
    model = TimbreModel()
    model.load()
    return model.extract(y, SR, grid)


def test_columns_and_length():
    out = run(white_noise(3.0))
    expected = (
        [f"timbre_mfcc_{i:02d}" for i in range(1, 14)]
        + [f"timbre_contrast_{b:02d}" for b in range(1, 8)]
        + ["timbre_flatness"]
    )
    assert list(out) == expected
    assert all(len(v) == 6 for v in out.values())


def test_flatness_noise_matches_periodogram_expectation():
    # Expected flatness of a white-noise periodogram (chi^2_2 bins) is
    # exp(-gamma) ~= 0.5615 — a closed-form anchor.
    out = run(white_noise(3.0))
    np.testing.assert_allclose(out["timbre_flatness"][MID], np.exp(-np.euler_gamma), rtol=0.05)


def test_flatness_sine_near_zero():
    out = run(sine(440, 3.0))
    assert (out["timbre_flatness"][MID] < 1e-3).all()


def test_contrast_peaks_in_tone_band():
    # 440 Hz falls in contrast band 3 (400-800 Hz octave); a pure tone's
    # peak-valley depth there dwarfs white noise's.
    tone = run(sine(440, 3.0))
    noise = run(white_noise(3.0))
    assert np.mean(tone["timbre_contrast_03"][MID]) > 3 * np.mean(noise["timbre_contrast_03"][MID])
    # and the tone's own maximum-contrast band is the one holding the tone
    band_means = [np.mean(tone[f"timbre_contrast_{b:02d}"][MID]) for b in range(1, 8)]
    assert int(np.argmax(band_means)) == 2  # 0-indexed -> band 3


def test_mfcc_stationary_tone_is_stable():
    out = run(sine(440, 3.0))
    for i in range(1, 14):
        assert np.std(out[f"timbre_mfcc_{i:02d}"][MID]) < 0.01


def test_mfcc_separates_timbres():
    # same f0 region, different spectra -> different MFCC profiles
    tone = run(sine(440, 3.0))
    noise = run(white_noise(3.0))
    t = np.array([np.mean(tone[f"timbre_mfcc_{i:02d}"][MID]) for i in range(1, 14)])
    n = np.array([np.mean(noise[f"timbre_mfcc_{i:02d}"][MID]) for i in range(1, 14)])
    assert np.linalg.norm(t - n) > 10


def test_silence_is_finite():
    out = run(silence(2.0))
    for v in out.values():
        assert np.isfinite(v).all()
    # flatness of the floored (constant) spectrum is 1 by convention
    np.testing.assert_allclose(out["timbre_flatness"], 1.0, rtol=1e-5)
