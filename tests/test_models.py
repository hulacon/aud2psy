"""Ground-truth checks for each frame-level model on synthetic audio."""

import numpy as np
import pytest

from aud2psy.grid import Grid
from aud2psy.models.loudness import LoudnessModel
from aud2psy.models.onsets import OnsetsModel
from aud2psy.models.pitch import PitchModel
from aud2psy.models.spectral import SpectralModel
from aud2psy.models.speech import SpeechModel

from conftest import SR, noise_bursts, silence, sine, white_noise


def run(model, y, sr=SR, hop=0.5):
    grid = Grid.for_duration(len(y) / sr, hop)
    model.load()
    return model.extract(y, sr, grid)


# --- loudness ---


def test_loudness_sine_rms():
    out = run(LoudnessModel(), sine(220, 3.0, amp=0.5))
    # RMS of a 0.5-amplitude sine is 0.5/sqrt(2); skip edge windows
    np.testing.assert_allclose(out["loudness_rms"][1:-1], 0.5 / np.sqrt(2), rtol=0.02)
    np.testing.assert_allclose(out["loudness_db"][1:-1], 20 * np.log10(0.5 / np.sqrt(2)), atol=0.2)


def test_loudness_silence_floors():
    out = run(LoudnessModel(), silence(2.0))
    assert (out["loudness_rms"] < 1e-6).all()
    assert (out["loudness_db"] <= -80).all()


def test_loudness_tracks_amplitude_steps():
    y = np.concatenate([sine(220, 1.0, amp=0.1), sine(220, 1.0, amp=0.4)])
    out = run(LoudnessModel(), y)
    assert out["loudness_rms"][3] > 3 * out["loudness_rms"][1]  # 4x amplitude -> ~4x RMS


# --- pitch ---


def test_pitch_pure_tone():
    out = run(PitchModel(), sine(220, 3.0))
    mid = slice(1, -1)
    np.testing.assert_allclose(out["pitch_f0"][mid], 220, rtol=0.01)
    assert (out["pitch_voiced_prob"][mid] > 0.5).all()


def test_pitch_octave_step():
    y = np.concatenate([sine(220, 1.5), sine(440, 1.5)])
    out = run(PitchModel(), y)
    np.testing.assert_allclose(out["pitch_f0"][1], 220, rtol=0.02)
    np.testing.assert_allclose(out["pitch_f0"][4], 440, rtol=0.02)


def test_pitch_silence_is_nan_and_unvoiced():
    out = run(PitchModel(), silence(2.0))
    assert np.isnan(out["pitch_f0"]).all()
    assert (out["pitch_voiced_prob"] < 0.5).all()


# --- spectral ---


def test_spectral_sine_centroid_and_zcr():
    out = run(SpectralModel(), sine(440, 3.0))
    mid = slice(1, -1)
    np.testing.assert_allclose(out["spectral_centroid"][mid], 440, rtol=0.15)
    # ZCR of a sine: 2 crossings per cycle -> 2f/sr per sample
    np.testing.assert_allclose(out["spectral_zcr"][mid], 2 * 440 / SR, rtol=0.1)


def test_spectral_noise_centroid_near_half_nyquist():
    out = run(SpectralModel(), white_noise(3.0))
    mid = slice(1, -1)
    np.testing.assert_allclose(out["spectral_centroid"][mid], SR / 4, rtol=0.15)
    assert (out["spectral_rolloff"][mid] > 0.75 * SR / 2).all()
    assert (out["spectral_bandwidth"][mid] > 2000).all()


def test_spectral_flux_high_at_burst_onsets():
    y = noise_bursts([1.0], duration=2.0)
    out = run(SpectralModel(), y)
    # window 0 is pure silence; window 1 can leak a little onset energy
    # because STFT frames centered just before 1.0 s reach into the burst
    assert out["spectral_flux"][2] > 10 * max(out["spectral_flux"][0], 1e-12)


# --- onsets ---


def test_onsets_rate_matches_bursts():
    times = [0.5, 1.5, 2.5, 3.5]
    y = noise_bursts(times, duration=4.5)
    out = run(OnsetsModel(), y)
    # 4 bursts over 4.5 s -> total events = sum(rate * hop) = 4 (+/- 1 for edge effects)
    total_events = out["onsets_rate"].sum() * 0.5
    assert 3 <= total_events <= 5
    # strength concentrates in burst windows (times fall in windows 1,3,5,7)
    burst_strength = out["onsets_strength"][[1, 3, 5, 7]].mean()
    quiet_strength = out["onsets_strength"][[0, 2, 4, 6]].mean()
    assert burst_strength > 2 * quiet_strength


def test_onsets_silence_has_no_events():
    out = run(OnsetsModel(), silence(3.0))
    assert out["onsets_rate"].sum() == 0
    assert np.allclose(out["onsets_strength"], 0)


# --- speech (Silero VAD; small bundled ONNX model, offline) ---


def test_speech_silence_prob_near_zero():
    out = run(SpeechModel(), silence(3.0))
    assert (out["speech_prob"] < 0.1).all()


def test_speech_tone_and_noise_not_speechlike():
    for y in (sine(220, 2.0), white_noise(2.0, amp=0.1)):
        out = run(SpeechModel(), y)
        assert out["speech_prob"].mean() < 0.5


def test_speech_output_covers_grid():
    out = run(SpeechModel(), white_noise(2.7))
    assert len(out["speech_prob"]) == Grid.for_duration(2.7, 0.5).n_windows
    assert not np.isnan(out["speech_prob"]).any()


# --- tonal ---


def triad(freqs, duration=6.0, sr=SR):
    t = np.arange(int(sr * duration)) / sr
    return sum(0.2 * np.sin(2 * np.pi * f * t) for f in freqs).astype(np.float32)


C_MAJOR = (261.63, 329.63, 392.0)  # C E G
C_MINOR = (261.63, 311.13, 392.0)  # C Eb G


def test_tonal_major_vs_minor():
    from aud2psy.models.tonal import TonalModel

    maj = run(TonalModel(), triad(C_MAJOR))
    mino = run(TonalModel(), triad(C_MINOR))
    assert np.nanmean(maj["tonal_majorness"]) > 0
    assert np.nanmean(mino["tonal_majorness"]) < 0
    assert np.nanmean(maj["tonal_key_clarity"]) > 0.6
    assert np.nanmean(mino["tonal_key_clarity"]) > 0.6


def test_tonal_noise_is_unclear_and_flat():
    from aud2psy.models.tonal import TonalModel

    out = run(TonalModel(), white_noise(6.0))
    assert np.nanmean(out["tonal_key_clarity"]) < 0.55
    assert np.nanmean(out["tonal_chroma_entropy"]) > 0.95  # ~uniform pitch classes


def test_tonal_silence_is_nan():
    from aud2psy.models.tonal import TonalModel

    out = run(TonalModel(), silence(4.0))
    assert np.isnan(out["tonal_key_clarity"]).all()


# --- rhythm ---


def test_rhythm_pulse_clarity_ordering():
    from aud2psy.models.rhythm import RhythmModel

    click = noise_bursts(list(np.arange(0, 6, 0.5)), duration=6.0, burst_len=0.015, amp=0.8)
    pc_click = np.nanmean(run(RhythmModel(), click)["rhythm_pulse_clarity"])
    pc_noise = np.nanmean(run(RhythmModel(), white_noise(6.0))["rhythm_pulse_clarity"])
    pc_tone = np.nanmean(run(RhythmModel(), sine(220, 6.0, amp=0.4))["rhythm_pulse_clarity"])
    assert pc_click > 2 * pc_noise  # metronomic >> arrhythmic
    assert pc_tone == 0.0  # no onset activity -> no pulse


def test_rhythm_novelty_peaks_at_section_boundary():
    from aud2psy.models.rhythm import RhythmModel
    from aud2psy.grid import Grid

    y = np.concatenate([sine(220, 6.0, amp=0.4), white_noise(6.0)])
    grid = Grid.for_duration(12.0, 0.5)
    model = RhythmModel()
    model.load()
    nov = model.extract(y, SR, grid)["rhythm_novelty"]
    peak_time = grid.centers[np.nanargmax(nov)]
    assert abs(peak_time - 6.0) <= 0.5  # peak within one window of the boundary


def test_rhythm_silence():
    from aud2psy.models.rhythm import RhythmModel

    out = run(RhythmModel(), silence(6.0))
    assert np.nanmax(out["rhythm_pulse_clarity"]) == 0.0
    assert np.nanmax(out["rhythm_beat_strength"]) == 0.0
