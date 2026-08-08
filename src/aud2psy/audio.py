"""Input decoding: audio files directly, video soundtracks via ffmpeg.

All decoding funnels through ffmpeg except plain .wav, which librosa/soundfile
read natively. Everything is returned mono at the requested sample rate.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .exceptions import FFmpegError, InputError

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}

FEATURE_SR = 22050  # librosa feature extraction
WHISPER_SR = 16000  # faster-whisper / Silero VAD


def input_type(path: str | Path) -> str:
    """Classify an input path as "audio" or "video"; raise InputError otherwise."""
    path = Path(path)
    if not path.exists():
        raise InputError(f"Input file not found: {path}")
    ext = path.suffix.lower()
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    supported = sorted(AUDIO_EXTENSIONS | VIDEO_EXTENSIONS)
    raise InputError(f"Unsupported input extension {ext!r} (supported: {', '.join(supported)})")


def load_audio(path: str | Path, sr: int) -> np.ndarray:
    """Load any supported input as a mono float32 array at ``sr`` Hz."""
    path = Path(path)
    input_type(path)  # validates existence and extension
    if path.suffix.lower() == ".wav":
        import librosa

        y, _ = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32)
    return _ffmpeg_decode(path, sr)


def _ffmpeg_decode(path: Path, sr: int) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise FFmpegError(
            "ffmpeg not found on PATH; it is required for video input and non-wav audio"
        )
    with tempfile.TemporaryDirectory(prefix="aud2psy_") as tmpdir:
        tmp_wav = Path(tmpdir) / "decoded.wav"
        cmd = [
            "ffmpeg", "-nostdin", "-y",
            "-i", str(path),
            "-vn", "-ac", "1", "-ar", str(sr),
            "-f", "wav", str(tmp_wav),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = proc.stderr.strip().splitlines()[-8:]
            raise FFmpegError(f"ffmpeg failed on {path}:\n" + "\n".join(tail))
        import soundfile as sf

        y, file_sr = sf.read(tmp_wav, dtype="float32")
    if file_sr != sr:  # ffmpeg was told -ar sr; this would be a bug, not bad input
        raise FFmpegError(f"ffmpeg produced {file_sr} Hz, expected {sr} Hz")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.ascontiguousarray(y, dtype=np.float32)
