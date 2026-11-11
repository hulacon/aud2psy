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
        "ebind_audio": ("ebind", None),
        "music_emotion": ("transformers", None),
        "sound_events": ("transformers", None),
        "speech_emotion": ("transformers", None),
        "phonetics": ("transformers", None),
        "egemaps": ("opensmile", None),
        "beats": ("beat_this", None),
        "diarize": ("pyannote.audio", None),
        "conversation": ("aud2psy", None),  # analytic; derives from diarize output
        "speech_rate": ("aud2psy", None),  # analytic; derives from transcribe output
    }
    pkg, fallback = version_map.get(model_name, (None, "unknown"))
    if pkg:
        try:
            from importlib.metadata import version

            return version(pkg)
        except Exception:
            pass
    return fallback or "unknown"


SCHEMA_VERSION = "1.4"  # Contract B §4.1 extractor output convention (1.3 is video-only)


def model_class(model_name: str) -> type:
    """The registered model's class, imported without instantiating it.

    Every model module imports only numpy/pandas at top level, so this never
    loads weights. A name outside the registry is a caller bug and raises.
    """
    import importlib

    from .cli import MODEL_REGISTRY

    module_path, class_name, _ = MODEL_REGISTRY[model_name]
    return getattr(importlib.import_module(module_path), class_name)


def declared_nulls(model_name: str) -> dict[str, dict[str, str]]:
    """The model's Contract B 1.1 ``nulls`` map, read from its class."""
    return {col: dict(entry) for col, entry in model_class(model_name).nulls.items()}


def declared_labels(model_name: str) -> dict[str, dict]:
    """The model's Contract B 1.2 ``labels`` (``default`` + ``columns``), read from its class."""
    import copy

    labels = copy.deepcopy(model_class(model_name).labels)
    return {"default": labels.get("default", {}), "columns": labels.get("columns", {})}


def stamp_nulls(model_meta: dict) -> dict:
    """Set ``nulls`` on every model entry from its class (overwriting any stale map).

    Where the entry lists its ``columns``, only declarations for columns it
    actually emitted are kept: a key must name a real column (§4.1), and some
    columns exist only in some runs (``median_f0`` needs pitch alongside
    transcribe).
    """
    for name, entry in model_meta.items():
        declared = declared_nulls(name)
        if "columns" in entry:
            emitted = set(entry["columns"])
            declared = {c: e for c, e in declared.items() if c in emitted}
        entry["nulls"] = declared
    return model_meta


def stamp_windows(model_meta: dict) -> dict:
    """Set (or remove) each model entry's model-level ``window_sec`` from its class.

    The model-level key is the 1.0 convention: present exactly when every
    column shares one numeric window (``BaseModel.window_sec``). The per-column
    truth is in ``labels`` (schema 1.4). Applied by ``sidecar refresh``;
    extraction writes the key from the running model instance. The window is
    a constant of the code, like the labels.
    """
    for name, entry in model_meta.items():
        w = model_class(name).window_sec
        if w is None:
            entry.pop("window_sec", None)
        else:
            entry["window_sec"] = float(w)
    return model_meta


def stamp_labels(model_meta: dict) -> dict:
    """Set ``labels`` on every model entry from its class (overwriting any stale map).

    As with nulls, where the entry lists its ``columns`` only the ``columns``
    overrides for columns the run emitted are kept (a key must be a real
    column). A model with a ``parent_model`` whose entry is in the same
    sidecar gets that entry's recorded checkpoint as ``derived_from``, so the
    label names the weights the parent actually ran with.
    """
    for name, entry in model_meta.items():
        labels = declared_labels(name)
        if "columns" in entry:
            emitted = set(entry["columns"])
            labels["columns"] = {c: e for c, e in labels["columns"].items() if c in emitted}
        parent = model_class(name).parent_model
        recorded = (model_meta.get(parent) or {}).get("checkpoint") if parent else None
        if recorded:
            for e in (labels["default"], *labels["columns"].values()):
                if "derived_from" in e:
                    e["derived_from"] = recorded
        entry["labels"] = labels
    return model_meta


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
        "schema_version": SCHEMA_VERSION,
        "extractor": "aud2psy",
        "extractor_version": __version__,
        "aud2psy_version": __version__,  # legacy key, one deprecation cycle
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {
            "type": input_type,
            "path": str(input_path),
            "duration_sec": round(duration, 3) if duration is not None else None,
        },
        "models": stamp_labels(stamp_nulls(model_meta)),
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
