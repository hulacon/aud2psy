"""Speech presence: per-window probability from Silero VAD.

Uses the VAD bundled with faster-whisper (small ONNX model, no Whisper weights
needed). The model emits one probability per 512 samples of 16 kHz audio
(32 ms); those are averaged into the shared grid. Input arrives at feature
rate (22050 Hz) like every frame model and is resampled internally.
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel

VAD_SR = 16000
VAD_WINDOW = 512  # samples per probability at 16 kHz


def _silero_asset_name() -> str:
    """The bundled Silero ONNX asset — the actual weights identity, which
    tracks the installed faster-whisper rather than a hardcoded string."""
    import glob
    import os

    from faster_whisper.utils import get_assets_path

    assets = sorted(glob.glob(os.path.join(get_assets_path(), "silero_vad*.onnx")))
    return os.path.basename(assets[0]) if assets else "silero-vad"


class SpeechModel(BaseModel):
    name = "speech"
    level = "frame"

    def load(self) -> None:
        from faster_whisper.vad import get_vad_model

        self.model = get_vad_model()
        self.checkpoint = _silero_asset_name()  # e.g. "silero_vad_v6.onnx"

    def extract(self, y: np.ndarray, sr: int, grid) -> dict[str, np.ndarray]:
        import librosa

        if sr != VAD_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=VAD_SR)
        y = y.astype(np.float32)
        pad = (-len(y)) % VAD_WINDOW
        if pad:
            y = np.concatenate([y, np.zeros(pad, dtype=np.float32)])
        probs = np.asarray(self.model(y, num_samples=VAD_WINDOW)).ravel()
        times = (np.arange(len(probs)) + 0.5) * VAD_WINDOW / VAD_SR
        return {"speech_prob": grid.average(times, probs)}
