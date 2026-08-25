"""Batch scoring: decode input once per sample rate, run models, build tables."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .audio import FEATURE_SR, WHISPER_SR, input_type, load_audio
from .grid import Grid
from .metadata import build_sidecar, get_model_version


@dataclass
class ScoreResult:
    frames_df: pd.DataFrame | None  # time + features; the psytwill-ready table
    transcript_df: pd.DataFrame | None  # raw Whisper segments, for word2psy
    words_df: pd.DataFrame | None  # word-level timestamps
    recall_df: pd.DataFrame | None = None  # wordpool-annotated recall
    beats_df: pd.DataFrame | None = None  # beat/downbeat events
    speakers_df: pd.DataFrame | None = None  # diarization speaker turns
    meta: dict = field(default_factory=dict)


VOICE_GENDER_F0_HZ = 150.0  # classic adult male/female f0 boundary


def annotate_voice_gender(transcript_df: pd.DataFrame, frames_df: pd.DataFrame, hop: float) -> None:
    """Add median_f0 + coarse voice_gender columns to transcript segments.

    Coarse M/F from per-segment median pyin f0 (median 111–114 Hz male vs
    170–172 Hz female on the validation dialogue). Deliberately uses raw
    f0 with NaN-aware medians, not a voiced_prob gate — window-averaged
    voicing probability dilutes over consonants. Limits: cannot separate
    same-gender speakers, children sit above both ranges; real speaker
    identity needs diarization (pyannote — roadmap).
    """
    import warnings

    med_f0, gender = [], []
    for _, seg in transcript_df.iterrows():
        m = (frames_df["time"] >= seg["onset"] - hop / 2) & (
            frames_df["time"] <= seg["offset"] + hop / 2
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # all-NaN slice on unvoiced segments
            f0 = float(np.nanmedian(frames_df.loc[m, "pitch_f0"])) if m.any() else np.nan
        med_f0.append(round(f0, 1) if np.isfinite(f0) else np.nan)
        gender.append("" if not np.isfinite(f0) else ("F" if f0 >= VOICE_GENDER_F0_HZ else "M"))
    transcript_df["median_f0"] = med_f0
    transcript_df["voice_gender"] = gender


def _is_embedding(name: str, feature_names: list[str]) -> bool:
    """Numbered-dimension outputs (clap_000...) vs distinct named columns."""
    import re

    return len(feature_names) > 16 and all(
        re.fullmatch(rf"{re.escape(name)}_\d+", f) for f in feature_names
    )


def get_model(name: str, **kwargs):
    """Instantiate a registered model by name (lazy import)."""
    from .cli import MODEL_REGISTRY

    module_path, class_name, _ = MODEL_REGISTRY[name]
    import importlib

    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**kwargs)


def _transcribe_columns(transcript_df, words_df) -> list:
    """Every column the `transcribe` model writes, across both its tables.

    `transcribe` is the one model here that emits two frames -- segments and
    words -- and only the segments frame used to be declared. That left
    `transcribe_probability` (words-only) emitted but undeclared in 1,060
    sidecars: correctly prefixed, so psytwill attributed it and the data is
    right, but the sidecar did not admit the column existed. A sidecar that
    under-declares is the same "declared != emitted" defect as one that
    over-declares, just in the other direction, and a gate reading only
    declarations cannot see either.
    """
    seen = list(transcript_df.columns)
    if words_df is not None:
        seen += [c for c in words_df.columns if c not in seen]
    return seen


def score_audio(
    path: str | Path,
    models: list[str],
    hop: float = 0.5,
    whisper_model: str = "large-v3",
    language: str | None = None,
    wordpool: str | Path | None = None,
    clap_model: str | None = None,
    num_speakers: int | None = None,
    speakers_csv: str | Path | None = None,
    words_csv: str | Path | None = None,
    verbatim: bool = False,
    show_progress: bool = True,
) -> ScoreResult:
    """Run the named models on one audio/video file.

    ``speakers_csv``: an existing diarize turn table ({stem}_speakers.csv)
    for the ``conversation`` model, instead of running ``diarize`` in the
    same call; when given, it is the turn source even if diarize also runs.
    ``words_csv``: likewise for ``speech_rate`` — an existing
    {stem}_transcript_words.csv instead of running ``transcribe``.
    """
    from tqdm import tqdm

    path = Path(path)
    in_type = input_type(path)
    t0 = time.time()

    from .cli import MODEL_REGISTRY
    from .exceptions import Aud2PsyError

    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        raise KeyError(f"Unknown model(s): {unknown}; see --list-models")
    frame_names = [
        m for m in models
        if m not in ("transcribe", "beats", "diarize", "conversation", "speech_rate")
    ]
    do_transcribe = "transcribe" in models
    do_beats = "beats" in models
    do_diarize = "diarize" in models
    do_conversation = "conversation" in models
    do_speech_rate = "speech_rate" in models
    if do_conversation and not do_diarize and speakers_csv is None:
        raise Aud2PsyError(
            "the conversation model derives from the diarize turn table; "
            "run it together with diarize, or pass an existing "
            "{stem}_speakers.csv via speakers_csv (CLI: --speakers)"
        )
    if do_speech_rate and not do_transcribe and words_csv is None:
        raise Aud2PsyError(
            "the speech_rate model derives from the transcribe word "
            "timestamps; run it together with transcribe, or pass an "
            "existing {stem}_transcript_words.csv via words_csv (CLI: --words)"
        )

    audio_cache: dict[int, np.ndarray] = {}

    def audio_at(sr: int) -> np.ndarray:
        if sr not in audio_cache:
            audio_cache[sr] = load_audio(path, sr)
        return audio_cache[sr]

    grid = None
    duration = None
    frames_df = None
    beats_df = None
    model_meta: dict[str, dict] = {}
    transcribe_info: dict = {}
    beats_info: dict = {}
    diarize_info: dict = {}

    if frame_names or do_beats:
        duration = len(audio_at(FEATURE_SR)) / FEATURE_SR

    if frame_names:
        grid = Grid.for_duration(duration, hop)
        columns: dict[str, np.ndarray] = {"time": grid.centers}
        clap_cache: dict = {}  # shares CLAP window embeddings across clap-family models
        for name in tqdm(frame_names, desc="models", disable=not show_progress):
            kwargs = {"checkpoint": clap_model} if name == "clap" and clap_model else {}
            model = get_model(name, **kwargs)
            if hasattr(model, "embed_windows"):
                model.cache_ = clap_cache
            t_model = time.time()
            model.load()
            model_sr = model.input_sr or FEATURE_SR
            features = model.extract(audio_at(model_sr), model_sr, grid)
            model.unload()
            for feat, values in features.items():
                columns[feat] = values
            names = list(features)
            if _is_embedding(name, names):  # record the pattern, not 512 names
                index_width = len(names[0]) - len(name) - 1  # digits after "{name}_"
                model_meta[name] = {
                    "pattern": f"{name}_{{{'N' * index_width}}}",
                    "range": [0, len(names) - 1],
                    "count": len(names),
                    "runtime_sec": round(time.time() - t_model, 2),
                }
            else:
                model_meta[name] = {
                    "columns": names,
                    "runtime_sec": round(time.time() - t_model, 2),
                }
            model_meta[name]["package_version"] = get_model_version(name)
            model_meta[name]["checkpoint"] = getattr(model, "checkpoint", None)
            window_sec = getattr(model, "window_sec", None)
            if window_sec is not None:
                model_meta[name]["window_sec"] = float(window_sec)
            info = getattr(model, "info_", None)
            if info:
                model_meta[name].update(info)  # an info_ checkpoint is authoritative
        frames_df = pd.DataFrame(columns)

    if do_beats:
        model = get_model("beats")
        t_model = time.time()
        model.load()
        beats_df, beats_info = model.detect(audio_at(FEATURE_SR), FEATURE_SR)
        model.unload()
        model_meta["beats"] = {
            "columns": list(beats_df.columns),
            "runtime_sec": round(time.time() - t_model, 2),
            "package_version": get_model_version("beats"),
            "checkpoint": getattr(model, "checkpoint", None),
        }

    speakers_df = exclusive_df = None
    if do_diarize:
        y_16k = audio_at(WHISPER_SR)
        if duration is None:
            duration = len(y_16k) / WHISPER_SR
        model = get_model("diarize", num_speakers=num_speakers)
        t_model = time.time()
        model.load()
        speakers_df, exclusive_df, diarize_info = model.diarize(y_16k, WHISPER_SR)
        model.unload()
        model_meta["diarize"] = {
            "columns": list(speakers_df.columns),
            "runtime_sec": round(time.time() - t_model, 2),
            "package_version": get_model_version("diarize"),
            "checkpoint": getattr(model, "checkpoint", None),
        }

    if do_conversation:
        from .models.conversation import load_turns_csv

        if speakers_csv is not None:
            turns_df = load_turns_csv(speakers_csv)
            turns_source = str(speakers_csv)
        else:
            turns_df = speakers_df
            turns_source = "diarize (this run)"
        if duration is None:
            duration = len(audio_at(FEATURE_SR)) / FEATURE_SR
        if grid is None:
            grid = Grid.for_duration(duration, hop)
        model = get_model("conversation")
        t_model = time.time()
        model.load()
        features = model.derive(turns_df, grid)
        model.unload()
        if frames_df is None:
            frames_df = pd.DataFrame({"time": grid.centers})
        for feat, values in features.items():
            frames_df[feat] = values
        model_meta["conversation"] = {
            "columns": list(features),
            "runtime_sec": round(time.time() - t_model, 2),
            "package_version": get_model_version("conversation"),
            "checkpoint": None,
            "turns_source": turns_source,
            **model.info_,
        }

    transcript_df = words_df = recall_df = None
    if do_transcribe:
        y_16k = audio_at(WHISPER_SR)
        if duration is None:
            duration = len(y_16k) / WHISPER_SR
        model = get_model(
            "transcribe", whisper_model=whisper_model, language=language, verbatim=verbatim
        )
        whisper_model = model.whisper_model  # verbatim swaps in the CrisperWhisper checkpoint
        t_model = time.time()
        model.load()
        transcript_df, words_df, transcribe_info = model.transcribe(y_16k, WHISPER_SR)
        model.unload()
        model_meta["transcribe"] = {
            "columns": _transcribe_columns(transcript_df, words_df),
            "runtime_sec": round(time.time() - t_model, 2),
            "package_version": get_model_version("transcribe"),
            "checkpoint": whisper_model,  # actual id, incl. the verbatim swap
        }
        from .recall import annotate_recall, load_wordpool, recall_summary, speech_timing

        transcribe_info["speech_timing"] = speech_timing(words_df, duration)
        if wordpool is not None:
            pool = load_wordpool(wordpool)
            recall_df = annotate_recall(words_df, pool)
            transcribe_info["recall"] = {
                "wordpool": str(wordpool),
                **recall_summary(recall_df, pool),
            }

    if (
        transcript_df is not None and len(transcript_df)
        and frames_df is not None and "pitch" in frame_names
    ):
        annotate_voice_gender(transcript_df, frames_df, hop)
        transcribe_info["voice_gender_threshold_hz"] = VOICE_GENDER_F0_HZ

    if transcript_df is not None and exclusive_df is not None:
        from .models.diarize import merge_speakers

        merge_speakers(transcript_df, words_df, exclusive_df)
        model_meta["transcribe"]["columns"] = _transcribe_columns(
            transcript_df, words_df)

    if do_speech_rate:
        from .models.speech_rate import load_words_csv

        if words_csv is not None:
            sr_words_df = load_words_csv(words_csv)
            words_source = str(words_csv)
        else:
            sr_words_df = words_df
            words_source = "transcribe (this run)"
        if duration is None:
            duration = len(audio_at(FEATURE_SR)) / FEATURE_SR
        if grid is None:
            grid = Grid.for_duration(duration, hop)
        model = get_model("speech_rate")
        t_model = time.time()
        model.load()
        features = model.derive(sr_words_df, grid)
        model.unload()
        if frames_df is None:
            frames_df = pd.DataFrame({"time": grid.centers})
        for feat, values in features.items():
            frames_df[feat] = values
        model_meta["speech_rate"] = {
            "columns": list(features),
            "runtime_sec": round(time.time() - t_model, 2),
            "package_version": get_model_version("speech_rate"),
            "checkpoint": None,
            "words_source": words_source,
            **model.info_,
        }

    meta = build_sidecar(
        input_path=path,
        input_type=in_type,
        duration=duration,
        hop=hop if (frame_names or do_conversation or do_speech_rate) else None,
        n_frames=grid.n_windows if grid else None,
        model_meta=model_meta,
        whisper_model=whisper_model if do_transcribe else None,
        transcribe_info=transcribe_info,
        beats_info=beats_info,
        diarize_info=diarize_info,
        total_runtime_sec=round(time.time() - t0, 2),
    )
    return ScoreResult(
        frames_df=frames_df,
        transcript_df=transcript_df,
        words_df=words_df,
        recall_df=recall_df,
        beats_df=beats_df,
        speakers_df=speakers_df,
        meta=meta,
    )


def score_audio_batch(
    paths: list[str | Path],
    models: list[str],
    hop: float = 0.5,
    whisper_model: str = "large-v3",
    language: str | None = None,
    wordpool: str | Path | None = None,
    clap_model: str | None = None,
    num_speakers: int | None = None,
    verbatim: bool = False,
    show_progress: bool = True,
) -> list[ScoreResult]:
    """Run the named models over many files, loading each model ONCE.

    ``score_audio`` loads and unloads every model per file, which is right when
    the file is long: the weights are amortised over minutes of audio. It is
    pathological when the files are short. Measured on 0.54 s spoken words, the
    nine neural models each cost 44-65 s per file and essentially all of it is
    the load -- scoring 1,000 words that way is ~159 GPU-hours to analyse nine
    minutes of audio.

    This function inverts the loops: **model-major**, so each model is loaded
    once and run across every file before being unloaded. Peak memory stays one
    model at a time, which is the invariant ``BaseModel.unload`` exists to
    protect; what grows instead is the accumulated feature table, bounded by
    (n_files x n_frames x n_features) and trivial for short files.

    Per-file results are identical to calling ``score_audio`` on each file, with
    one deliberate exception: the CLAP window-embedding cache is not shared
    across the clap-family models. Sharing it would mean holding one cache per
    file alive across three model passes, and it saves forward passes on audio
    already in memory -- worth it per-file, pointless against a load cost
    amortised a thousandfold. Output values are unaffected.

    Best suited to many short files. Each file is decoded once per model rather
    than once per batch (bounded memory beats a decode cache here), so for long
    inputs the repeated decode can outweigh the saved loads -- use
    ``score_audio`` per file for movie-length audio.
    """
    from tqdm import tqdm

    paths = [Path(p) for p in paths]
    if not paths:
        return []
    from .cli import MODEL_REGISTRY

    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        raise KeyError(f"Unknown model(s): {unknown}; see --list-models")

    frame_names = [
        m for m in models
        if m not in ("transcribe", "beats", "diarize", "conversation", "speech_rate")
    ]
    do_transcribe = "transcribe" in models
    do_beats = "beats" in models
    do_diarize = "diarize" in models
    do_conversation = "conversation" in models
    do_speech_rate = "speech_rate" in models
    if do_conversation and not do_diarize:
        from .exceptions import Aud2PsyError

        raise Aud2PsyError(
            "in a batch, the conversation model needs diarize in the same call "
            "(a single speakers_csv cannot map onto several inputs)"
        )
    if do_speech_rate and not do_transcribe:
        from .exceptions import Aud2PsyError

        raise Aud2PsyError(
            "in a batch, the speech_rate model needs transcribe in the same "
            "call (a single words_csv cannot map onto several inputs)"
        )

    n = len(paths)
    t0 = [time.time()] * n
    in_type = [input_type(p) for p in paths]
    duration: list[float | None] = [None] * n
    grids: list[Grid | None] = [None] * n
    columns: list[dict] = [{} for _ in range(n)]
    model_meta: list[dict] = [{} for _ in range(n)]
    transcribe_info: list[dict] = [{} for _ in range(n)]
    beats_info: list[dict] = [{} for _ in range(n)]
    diarize_info: list[dict] = [{} for _ in range(n)]
    beats_df: list = [None] * n
    speakers_df: list = [None] * n
    exclusive_df: list = [None] * n
    transcript_df: list = [None] * n
    words_df: list = [None] * n
    recall_df: list = [None] * n

    def _durations(sr: int) -> None:
        """Fill duration/grid on first need, from a decode at ``sr``."""
        for i, p in enumerate(paths):
            if duration[i] is None:
                duration[i] = len(load_audio(p, sr)) / sr

    if frame_names or do_beats:
        _durations(FEATURE_SR)
    if frame_names:
        for i in range(n):
            grids[i] = Grid.for_duration(duration[i], hop)
            columns[i] = {"time": grids[i].centers}

    def _record(i: int, name: str, model, feats: dict | None, t_model: float) -> None:
        if feats is not None:
            names = list(feats)
            if _is_embedding(name, names):
                index_width = len(names[0]) - len(name) - 1
                model_meta[i][name] = {
                    "pattern": f"{name}_{{{'N' * index_width}}}",
                    "range": [0, len(names) - 1],
                    "count": len(names),
                    "runtime_sec": round(time.time() - t_model, 2),
                }
            else:
                model_meta[i][name] = {
                    "columns": names,
                    "runtime_sec": round(time.time() - t_model, 2),
                }
        model_meta[i][name]["package_version"] = get_model_version(name)
        model_meta[i][name]["checkpoint"] = getattr(model, "checkpoint", None)
        window_sec = getattr(model, "window_sec", None)
        if window_sec is not None:
            model_meta[i][name]["window_sec"] = float(window_sec)
        info = getattr(model, "info_", None)
        if info:
            model_meta[i][name].update(info)
        # The load is amortised across the batch, so a per-file runtime_sec
        # here is extraction only and is NOT comparable to score_audio's.
        model_meta[i][name]["batched"] = True

    for name in tqdm(frame_names, desc="models", disable=not show_progress):
        kwargs = {"checkpoint": clap_model} if name == "clap" and clap_model else {}
        model = get_model(name, **kwargs)
        model.load()
        model_sr = model.input_sr or FEATURE_SR
        try:
            for i, p in enumerate(tqdm(paths, desc=name, leave=False,
                                       disable=not show_progress)):
                if hasattr(model, "embed_windows"):
                    model.cache_ = {}  # per-file; see docstring
                t_model = time.time()
                feats = model.extract(load_audio(p, model_sr), model_sr, grids[i])
                for feat, values in feats.items():
                    columns[i][feat] = values
                _record(i, name, model, feats, t_model)
        finally:
            model.unload()

    if do_beats:
        model = get_model("beats")
        model.load()
        try:
            for i, p in enumerate(tqdm(paths, desc="beats", leave=False,
                                       disable=not show_progress)):
                t_model = time.time()
                beats_df[i], beats_info[i] = model.detect(
                    load_audio(p, FEATURE_SR), FEATURE_SR)
                model_meta[i]["beats"] = {
                    "columns": list(beats_df[i].columns),
                    "runtime_sec": round(time.time() - t_model, 2),
                }
                _record(i, "beats", model, None, t_model)
        finally:
            model.unload()

    if do_diarize:
        _durations(WHISPER_SR)
        model = get_model("diarize", num_speakers=num_speakers)
        model.load()
        try:
            for i, p in enumerate(tqdm(paths, desc="diarize", leave=False,
                                       disable=not show_progress)):
                t_model = time.time()
                speakers_df[i], exclusive_df[i], diarize_info[i] = model.diarize(
                    load_audio(p, WHISPER_SR), WHISPER_SR)
                model_meta[i]["diarize"] = {
                    "columns": list(speakers_df[i].columns),
                    "runtime_sec": round(time.time() - t_model, 2),
                }
                _record(i, "diarize", model, None, t_model)
        finally:
            model.unload()

    if do_conversation:
        model = get_model("conversation")
        model.load()
        try:
            for i in range(n):
                t_model = time.time()
                if grids[i] is None:
                    grids[i] = Grid.for_duration(duration[i], hop)
                    columns[i]["time"] = grids[i].centers
                feats = model.derive(speakers_df[i], grids[i])
                for feat, values in feats.items():
                    columns[i][feat] = values
                _record(i, "conversation", model, feats, t_model)
                model_meta[i]["conversation"]["turns_source"] = "diarize (this run)"
        finally:
            model.unload()

    if do_transcribe:
        _durations(WHISPER_SR)
        model = get_model("transcribe", whisper_model=whisper_model,
                          language=language, verbatim=verbatim)
        whisper_model = model.whisper_model
        model.load()
        try:
            from .recall import annotate_recall, load_wordpool, recall_summary, speech_timing

            pool = load_wordpool(wordpool) if wordpool is not None else None
            for i, p in enumerate(tqdm(paths, desc="transcribe", leave=False,
                                       disable=not show_progress)):
                t_model = time.time()
                transcript_df[i], words_df[i], transcribe_info[i] = model.transcribe(
                    load_audio(p, WHISPER_SR), WHISPER_SR)
                model_meta[i]["transcribe"] = {
                    "columns": _transcribe_columns(transcript_df[i], words_df[i]),
                    "runtime_sec": round(time.time() - t_model, 2),
                }
                _record(i, "transcribe", model, None, t_model)
                model_meta[i]["transcribe"]["checkpoint"] = whisper_model
                transcribe_info[i]["speech_timing"] = speech_timing(words_df[i], duration[i])
                if pool is not None:
                    recall_df[i] = annotate_recall(words_df[i], pool)
                    transcribe_info[i]["recall"] = {
                        "wordpool": str(wordpool), **recall_summary(recall_df[i], pool)}
        finally:
            model.unload()

    if do_speech_rate:
        model = get_model("speech_rate")
        model.load()
        try:
            for i in range(n):
                t_model = time.time()
                if grids[i] is None:
                    grids[i] = Grid.for_duration(duration[i], hop)
                    columns[i]["time"] = grids[i].centers
                feats = model.derive(words_df[i], grids[i])
                for feat, values in feats.items():
                    columns[i][feat] = values
                _record(i, "speech_rate", model, feats, t_model)
                model_meta[i]["speech_rate"]["words_source"] = "transcribe (this run)"
        finally:
            model.unload()

    # Cross-model per-file post-processing, identical to score_audio's tail.
    results = []
    for i, p in enumerate(paths):
        frames = pd.DataFrame(columns[i]) if (
            frame_names or do_conversation or do_speech_rate
        ) else None
        if (transcript_df[i] is not None and len(transcript_df[i])
                and frames is not None and "pitch" in frame_names):
            annotate_voice_gender(transcript_df[i], frames, hop)
            transcribe_info[i]["voice_gender_threshold_hz"] = VOICE_GENDER_F0_HZ
        if transcript_df[i] is not None and exclusive_df[i] is not None:
            from .models.diarize import merge_speakers

            merge_speakers(transcript_df[i], words_df[i], exclusive_df[i])
            model_meta[i]["transcribe"]["columns"] = _transcribe_columns(
                transcript_df[i], words_df[i])
        meta = build_sidecar(
            input_path=p,
            input_type=in_type[i],
            duration=duration[i],
            hop=hop if (frame_names or do_conversation or do_speech_rate) else None,
            n_frames=grids[i].n_windows if grids[i] else None,
            model_meta=model_meta[i],
            whisper_model=whisper_model if do_transcribe else None,
            transcribe_info=transcribe_info[i],
            beats_info=beats_info[i],
            diarize_info=diarize_info[i],
            total_runtime_sec=round(time.time() - t0[i], 2),
        )
        results.append(ScoreResult(
            frames_df=frames,
            transcript_df=transcript_df[i],
            words_df=words_df[i],
            recall_df=recall_df[i],
            beats_df=beats_df[i],
            speakers_df=speakers_df[i],
            meta=meta,
        ))
    return results


def save_result(
    result: ScoreResult,
    out_path: str | Path,
    stimulus_id: str | None = None,
) -> dict[str, Path]:
    """Write the family-standard file set next to ``out_path``.

    -o scores.csv produces scores_frames.csv, scores_transcript.csv,
    scores_transcript_words.csv (each only if its table exists), and
    scores.meta.json.

    Every table leads with a ``stimulus_id`` column (Contract B §4.1):
    the input file's stem by default, ``stimulus_id`` to override
    (``time``/``onset`` disambiguate rows within the stimulus).
    """
    import json

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.parent / out_path.stem
    if stimulus_id is None:
        input_path = result.meta.get("input", {}).get("path")
        stimulus_id = Path(input_path).stem if input_path else None

    def _write(df, path: Path) -> None:
        if stimulus_id is not None and "stimulus_id" not in df.columns:
            df = df.copy()
            df.insert(0, "stimulus_id", stimulus_id)
        df.to_csv(path, index=False, float_format="%.6g")

    written: dict[str, Path] = {}
    if result.frames_df is not None:
        written["frames"] = Path(f"{stem}_frames.csv")
        _write(result.frames_df, written["frames"])
    if result.transcript_df is not None:
        written["transcript"] = Path(f"{stem}_transcript.csv")
        _write(result.transcript_df, written["transcript"])
    if result.words_df is not None:
        written["transcript_words"] = Path(f"{stem}_transcript_words.csv")
        _write(result.words_df, written["transcript_words"])
    if result.recall_df is not None:
        written["recall"] = Path(f"{stem}_recall.csv")
        _write(result.recall_df, written["recall"])
    if result.beats_df is not None:
        written["beats"] = Path(f"{stem}_beats.csv")
        _write(result.beats_df, written["beats"])
    if result.speakers_df is not None:
        written["speakers"] = Path(f"{stem}_speakers.csv")
        _write(result.speakers_df, written["speakers"])
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
        "recall": result.recall_df,
        "beats": result.beats_df,
        "speakers": result.speakers_df,
    }[kind]
    return len(df)
