"""Sidecar JSON generation (the family .meta.json convention)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


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
) -> dict:
    from . import __version__

    meta: dict = {
        "aud2psy_version": __version__,
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
    if whisper_model is not None:
        meta["transcription"] = {
            "whisper_model": whisper_model,
            "compute_type": "int8",
            "word_timestamps": True,
            "vad_filter": True,
            **transcribe_info,  # language, language_probability, n_speech_segments
        }
    return meta
