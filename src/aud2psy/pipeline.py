"""Batch scoring: decode input once per sample rate, run models, build tables."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .audio import FEATURE_SR, WHISPER_SR, input_type, load_audio
from .grid import Grid
from .metadata import build_sidecar


@dataclass
class ScoreResult:
    frames_df: pd.DataFrame | None  # time + features; the psyquilt-ready table
    transcript_df: pd.DataFrame | None  # raw Whisper segments, for word2psy
    words_df: pd.DataFrame | None  # word-level timestamps
    meta: dict = field(default_factory=dict)


def get_model(name: str, **kwargs):
    """Instantiate a registered model by name (lazy import)."""
    from .cli import MODEL_REGISTRY

    module_path, class_name, _ = MODEL_REGISTRY[name]
    import importlib

    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**kwargs)


def score_audio(
    path: str | Path,
    models: list[str],
    hop: float = 0.5,
    whisper_model: str = "large-v3",
    language: str | None = None,
    show_progress: bool = True,
) -> ScoreResult:
    """Run the named models on one audio/video file."""
    from tqdm import tqdm

    path = Path(path)
    in_type = input_type(path)
    t0 = time.time()

    from .cli import MODEL_REGISTRY

    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        raise KeyError(f"Unknown model(s): {unknown}; see --list-models")
    frame_names = [m for m in models if m != "transcribe"]
    do_transcribe = "transcribe" in models

    y_feat = None
    grid = None
    duration = None
    frames_df = None
    model_meta: dict[str, dict] = {}
    transcribe_info: dict = {}

    if frame_names:
        y_feat = load_audio(path, FEATURE_SR)
        duration = len(y_feat) / FEATURE_SR
        grid = Grid.for_duration(duration, hop)
        columns: dict[str, np.ndarray] = {"time": grid.centers}
        for name in tqdm(frame_names, desc="models", disable=not show_progress):
            model = get_model(name)
            t_model = time.time()
            model.load()
            features = model.extract(y_feat, FEATURE_SR, grid)
            model.unload()
            for feat, values in features.items():
                columns[feat] = values
            model_meta[name] = {
                "columns": list(features),
                "runtime_sec": round(time.time() - t_model, 2),
            }
        frames_df = pd.DataFrame(columns)

    transcript_df = words_df = None
    if do_transcribe:
        y_16k = load_audio(path, WHISPER_SR)
        if duration is None:
            duration = len(y_16k) / WHISPER_SR
        model = get_model("transcribe", whisper_model=whisper_model, language=language)
        t_model = time.time()
        model.load()
        transcript_df, words_df, transcribe_info = model.transcribe(y_16k, WHISPER_SR)
        model.unload()
        model_meta["transcribe"] = {
            "columns": list(transcript_df.columns),
            "runtime_sec": round(time.time() - t_model, 2),
        }

    meta = build_sidecar(
        input_path=path,
        input_type=in_type,
        duration=duration,
        hop=hop if frame_names else None,
        n_frames=grid.n_windows if grid else None,
        model_meta=model_meta,
        whisper_model=whisper_model if do_transcribe else None,
        transcribe_info=transcribe_info,
        total_runtime_sec=round(time.time() - t0, 2),
    )
    return ScoreResult(frames_df=frames_df, transcript_df=transcript_df, words_df=words_df, meta=meta)


def save_result(result: ScoreResult, out_path: str | Path) -> dict[str, Path]:
    """Write the family-standard file set next to ``out_path``.

    -o scores.csv produces scores_frames.csv, scores_transcript.csv,
    scores_transcript_words.csv (each only if its table exists), and
    scores.meta.json.
    """
    import json

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.parent / out_path.stem
    written: dict[str, Path] = {}
    if result.frames_df is not None:
        written["frames"] = Path(f"{stem}_frames.csv")
        result.frames_df.to_csv(written["frames"], index=False, float_format="%.6g")
    if result.transcript_df is not None:
        written["transcript"] = Path(f"{stem}_transcript.csv")
        result.transcript_df.to_csv(written["transcript"], index=False, float_format="%.6g")
    if result.words_df is not None:
        written["transcript_words"] = Path(f"{stem}_transcript_words.csv")
        result.words_df.to_csv(written["transcript_words"], index=False, float_format="%.6g")
    meta_path = Path(f"{stem}.meta.json")
    result.meta["output"] = {
        kind: {"path": str(p), "rows": _n_rows(result, kind)} for kind, p in written.items()
    }
    with open(meta_path, "w") as f:
        json.dump(result.meta, f, indent=2)
    written["meta"] = meta_path
    return written


def _n_rows(result: ScoreResult, kind: str) -> int:
    df = {
        "frames": result.frames_df,
        "transcript": result.transcript_df,
        "transcript_words": result.words_df,
    }[kind]
    return len(df)
