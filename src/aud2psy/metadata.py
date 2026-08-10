"""Sidecar JSON generation (the family .meta.json convention)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def get_model_version(model_name: str) -> str:
    """Version of the underlying package (the word2psy/viz2psy pattern)."""
    version_map = {
        "loudness": ("librosa", None),
        "pitch": ("librosa", None),
        "spectral": ("librosa", None),
        "onsets": ("librosa", None),
        "tonal": ("librosa", None),
        "rhythm": ("librosa", None),
        "timbre": ("librosa", None),
        "psychoacoustic": ("mosqito", None),
        "speech": ("faster-whisper", None),  # bundled Silero VAD
        "transcribe": ("faster-whisper", None),
        "clap": ("transformers", None),
        "music_emotion": ("transformers", None),
        "sound_events": ("transformers", None),
        "speech_emotion": ("transformers", None),
        "egemaps": ("opensmile", None),
        "beats": ("beat_this", None),
        "diarize": ("pyannote.audio", None),
    }
    pkg, fallback = version_map.get(model_name, (None, "unknown"))
    if pkg:
        try:
            from importlib.metadata import version

            return version(pkg)
        except Exception:
            pass
    return fallback or "unknown"


def build_sidecar(
    input_path: Path,
    input_type: str,
    duration: float | None,
    hop: float | None,
    n_frames: int | None,
    model_meta: dict,
    whisper_model: str | None,
    transcribe_info: dict,
    total_runtime_sec: float,
    beats_info: dict | None = None,
    diarize_info: dict | None = None,
) -> dict:
    from . import __version__

    meta: dict = {
        "schema_version": "1.0",  # Contract B §4.1 extractor output convention
        "extractor": "aud2psy",
        "extractor_version": __version__,
        "aud2psy_version": __version__,  # legacy key, one deprecation cycle
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {
            "type": input_type,
            "path": str(input_path),
            "duration_sec": round(duration, 3) if duration is not None else None,
        },
        "models": model_meta,
        "total_runtime_sec": total_runtime_sec,
    }
    if hop is not None:
        meta["frames"] = {"hop_sec": hop, "n_frames": n_frames, "time": "window center"}
    if beats_info:
        meta["beats"] = beats_info
    if diarize_info:
        meta["diarization"] = diarize_info
    if whisper_model is not None:
        meta["transcription"] = {
            "whisper_model": whisper_model,
            "compute_type": "int8",
            "word_timestamps": True,
            **transcribe_info,  # language, vad_filter, n_speech_segments, ...
        }
    return meta
