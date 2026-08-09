"""speech_emotion tests.

Windowing and VAD gating run offline (the Silero VAD ships with
faster-whisper — no download). Real wav2vec2 inference (~600 MB weights)
is behind the `weights` marker: pytest -m weights
"""

import numpy as np
import pytest

from aud2psy.models.speech_emotion import (
    GATE_MIN_SPEECH,
    WINDOW_SEC,
    context_windows,
    speech_fraction,
)

from conftest import silence, sine, white_noise

SR = 16000


def test_context_windows_are_edge_clamped():
    y = silence(10.0, sr=SR)
    centers = np.array([0.25, 5.0, 9.75])
    wins = context_windows(y, SR, centers, WINDOW_SEC)
    n = int(WINDOW_SEC * SR)
    assert all(len(w) == n for w in wins)
    # first window starts at 0, last ends at clip end
    assert wins[0][0] == y[0] and len(wins) == 3


def test_context_windows_short_clip_uses_whole_clip():
    y = white_noise(2.0, sr=SR)  # shorter than the 4 s context
    wins = context_windows(y, SR, np.array([0.5, 1.5]), WINDOW_SEC)
    assert all(len(w) == len(y) for w in wins)


class FakeVAD:
    """Callable mimicking faster-whisper's Silero wrapper."""

    def __init__(self, prob_fn):
        self.prob_fn = prob_fn

    def __call__(self, y, num_samples):
        n = len(y) // num_samples
        times = (np.arange(n) + 0.5) * num_samples / SR
        return self.prob_fn(times)


def test_speech_fraction_windows_average_probs():
    y = silence(8.0, sr=SR)
    # speech in the first half only
    vad = FakeVAD(lambda t: np.where(t < 4.0, 0.9, 0.0))
    frac = speech_fraction(vad, y, SR, np.array([2.0, 6.0]), WINDOW_SEC)
    assert frac[0] == pytest.approx(0.9, abs=0.02)
    assert frac[1] == pytest.approx(0.0, abs=0.02)
    assert frac[0] >= GATE_MIN_SPEECH > frac[1]


def test_real_vad_gates_nonspeech():
    from faster_whisper.vad import get_vad_model

    vad = get_vad_model()
    for y in (silence(6.0, sr=SR), sine(440.0, 6.0, sr=SR)):
        frac = speech_fraction(vad, y, SR, np.array([1.0, 3.0, 5.0]), WINDOW_SEC)
        assert (frac < GATE_MIN_SPEECH).all()


@pytest.mark.weights
def test_speech_emotion_on_speech_and_music(tmp_path):
    import shutil
    import subprocess

    import soundfile as sf

    from aud2psy.pipeline import score_audio

    if shutil.which("say") is None:
        pytest.skip("macOS `say` not available to synthesize speech")
    aiff, wav = tmp_path / "s.aiff", tmp_path / "s.wav"
    subprocess.run(["say", "-o", str(aiff), "This is a perfectly normal sentence "
                    "spoken in a perfectly normal voice."], check=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", str(aiff), "-ac", "1", "-ar", "16000", str(wav)],
        check=True, capture_output=True,
    )
    res = score_audio(wav, ["speech_emotion"])
    cols = ["speech_emotion_valence", "speech_emotion_arousal", "speech_emotion_dominance"]
    df = res.frames_df
    assert list(df.columns) == ["time"] + cols
    voiced = df[cols].dropna()
    assert len(voiced) > 0
    assert ((voiced > -0.5) & (voiced < 1.5)).all().all()  # approximately 0..1

    # pure tone: every window gated to NaN
    tone = tmp_path / "tone.wav"
    sf.write(tone, sine(440.0, 6.0, sr=16000), 16000)
    res_tone = score_audio(tone, ["speech_emotion"])
    assert res_tone.frames_df[cols].isna().all().all()
