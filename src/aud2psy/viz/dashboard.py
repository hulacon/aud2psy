"""Interactive HTML dashboard for exploring aud2psy scores.

Mirrors word2psy's ``viz browse`` dashboard (itself modeled on viz2psy's):
a single self-contained HTML file with a model selector, overview
visualizations (timeseries, 2D/3D clustering, trajectory) where every point
is a frame (or transcript segment), and a click-to-open detail viewer with
a slider and prev/next buttons.

Where viz2psy shows the image and word2psy renders the word, the stimulus
here is *audio*: the input clip is embedded in the page (ffmpeg-encoded mp3,
raw wav fallback) behind play/pause controls. The overview plots get a
play button that follows the full stream with a moving time cursor, and the
detail view's play button plays just that frame's hop-length window (or the
segment's onset-offset span), with an optional loop for short windows.

All heavy computation (projections, feature detection, audio encoding)
happens here in Python; the HTML embeds pre-computed data as JSON and
renders with Plotly.js (loaded from CDN, same version as the siblings).
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_config import (
    FEATURE_CONFIGS,
    detect_models_in_dataframe,
    get_model_columns,
)
from .projection import compute_projection

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"

# Structural columns that are never features
_FRAME_INDEX_COLS = ["time"]
_SEGMENT_INDEX_COLS = ["segment_idx", "text", "onset", "offset"]

# Detail-panel labels per frame model (order = dropdown order)
_PANEL_LABELS = {
    "loudness": "Loudness",
    "pitch": "Pitch & voicing",
    "spectral": "Spectral shape",
    "onsets": "Onsets & tempo",
    "tonal": "Tonality",
    "rhythm": "Rhythm & structure",
    "timbre": "Timbre (MFCC/contrast)",
    "psychoacoustic": "Psychoacoustics (Zwicker)",
    "speech": "Speech presence",
    "music_emotion": "Music emotion",
    "sound_events": "Sound events (zero-shot)",
    "speech_emotion": "Speech emotion",
    "egemaps": "eGeMAPS prosody",
}


def _values(series: pd.Series) -> list:
    """Series -> JSON-safe list (NaN -> None, numpy -> python)."""
    out = []
    for v in series:
        if isinstance(v, (float, np.floating)):
            out.append(None if np.isnan(v) else round(float(v), 5))
        elif isinstance(v, (int, np.integer)):
            out.append(int(v))
        elif v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(None)
        else:
            out.append(str(v))
    return out


def _feature_ranges(df: pd.DataFrame, cols: list[str]) -> dict:
    """Dataset min/max per column, for normalized detail bars."""
    ranges = {}
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        lo, hi = vals.min(), vals.max()
        if pd.notna(lo) and pd.notna(hi):
            ranges[col] = [round(float(lo), 5), round(float(hi), 5)]
    return ranges


def _projection_payload(
    df: pd.DataFrame,
    cols: list[str],
    method: str,
    n_components: int,
) -> dict | None:
    """Compute a projection, dropping rows with any NaN (e.g. VAD-gated frames).

    Returns dict with coordinate arrays, kept row indices, and axis labels,
    or None if the projection is not possible.
    """
    X = df[cols].to_numpy(dtype=float)
    valid = ~np.isnan(X).any(axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) < 3:
        return None

    try:
        X_proj, info = compute_projection(
            X[valid], method=method, n_components=n_components
        )
    except Exception as e:  # pragma: no cover - degenerate inputs
        warnings.warn(f"{method} projection failed: {e}")
        return None

    payload = {
        "x": [round(float(v), 4) for v in X_proj[:, 0]],
        "y": [round(float(v), 4) for v in X_proj[:, 1]],
        "indices": indices.tolist(),
        "xlabel": info.get("xlabel", "dim 1"),
        "ylabel": info.get("ylabel", "dim 2"),
        "method": method,
    }
    if n_components == 3:
        payload["z"] = [round(float(v), 4) for v in X_proj[:, 2]]
        payload["zlabel"] = info.get("zlabel", "dim 3")
    return payload


def _timeseries_features(df: pd.DataFrame, cols: list[str], config) -> list[str]:
    """Feature order for the timeseries view (top-k by variance if configured)."""
    if config.timeseries_mode == "top_k":
        variances = [(c, float(pd.to_numeric(df[c], errors="coerce").var())) for c in cols]
        variances.sort(key=lambda t: (np.isnan(t[1]), -t[1] if not np.isnan(t[1]) else 0))
        return [c for c, _ in variances[: config.top_k]]
    return cols


def _build_model_entry(df: pd.DataFrame, name: str, mds_max: int) -> dict | None:
    """Build the payload entry for one detected model."""
    config = FEATURE_CONFIGS[name]
    cols = get_model_columns(df.columns.tolist(), name)
    if not cols:
        return None

    n = len(df)
    entry = {
        "level": config.level,
        "description": config.description,
        "nDims": len(cols),
        "featureType": config.feature_type,
        "tsFeatures": _timeseries_features(df, cols, config) if config.timeseries else [],
        "clustering": {},
        "trajectory": None,
    }

    if config.mds_clustering and len(cols) >= 2 and n >= 3:
        cluster_method = "mds" if n <= mds_max else "pca"
        proj2d = _projection_payload(df, cols, cluster_method, 2)
        if proj2d:
            entry["clustering"]["2d"] = proj2d
        if len(cols) >= 3 and n >= 4:
            proj3d = _projection_payload(df, cols, cluster_method, 3)
            if proj3d:
                entry["clustering"]["3d"] = proj3d
        # Trajectory always uses PCA (stable, fast, ordered path is the point)
        traj = _projection_payload(df, cols, "pca", 2)
        if traj:
            entry["trajectory"] = traj

    entry["available"] = {
        "timeseries": bool(entry["tsFeatures"]),
        "clustering2d": "2d" in entry["clustering"],
        "clustering3d": "3d" in entry["clustering"],
        "trajectory": entry["trajectory"] is not None,
    }
    if not any(entry["available"].values()):
        return None
    return entry


def _is_scalar_col(col: str) -> bool:
    """True if a feature column is scalar (not an embedding dimension)."""
    tail = col.rsplit("_", 1)[-1]
    return not (len(tail) == 3 and tail.isdigit())


def _infer_hop(frames_df: pd.DataFrame | None, meta: dict | None) -> float:
    """Window size in seconds: sidecar first, then median time diff, then 0.5."""
    if meta:
        hop = (meta.get("frames") or {}).get("hop_sec")
        if hop:
            return float(hop)
    if frames_df is not None and "time" in frames_df.columns and len(frames_df) > 1:
        diffs = np.diff(frames_df["time"].to_numpy(dtype=float))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if len(diffs):
            return float(round(np.median(diffs), 6))
    return 0.5


def _audio_envelope(y: np.ndarray, n_bins: int = 1500) -> list[float]:
    """Peak-amplitude envelope for the waveform strip, normalized to [0, 1]."""
    n_bins = min(n_bins, len(y)) or 1
    edges = np.linspace(0, len(y), n_bins + 1).astype(int)
    env = np.array([
        float(np.max(np.abs(y[a:b]))) if b > a else 0.0
        for a, b in zip(edges[:-1], edges[1:])
    ])
    peak = env.max()
    if peak > 0:
        env = env / peak
    return [round(float(v), 3) for v in env]


def _encode_audio(path: Path, bitrate: str) -> tuple[bytes, str]:
    """Audio file -> (bytes, mime) for the data URI.

    ffmpeg re-encodes to mono mp3 (any supported input, small embed); if
    ffmpeg is unavailable, plain .wav files embed as-is so offline use and
    tests still work.
    """
    if shutil.which("ffmpeg") is not None:
        with tempfile.TemporaryDirectory(prefix="aud2psy_") as tmpdir:
            tmp_mp3 = Path(tmpdir) / "embed.mp3"
            cmd = [
                "ffmpeg", "-nostdin", "-y",
                "-i", str(path),
                "-vn", "-ac", "1", "-ar", "44100", "-b:a", bitrate,
                "-f", "mp3", str(tmp_mp3),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                return tmp_mp3.read_bytes(), "audio/mpeg"
            warnings.warn(
                "ffmpeg mp3 encode failed; "
                + ("falling back to raw wav embed"
                   if path.suffix.lower() == ".wav" else "audio not embedded")
            )
    if path.suffix.lower() == ".wav":
        return path.read_bytes(), "audio/wav"
    raise RuntimeError(
        f"Cannot embed {path.name}: ffmpeg is required for non-wav audio"
    )


def prepare_audio(path: str | Path, bitrate: str = "96k") -> dict:
    """Prepare the audio payload: embedded data URI + waveform envelope.

    Returns a dict for :func:`create_dashboard`'s ``audio`` parameter with
    keys ``src`` (data URI), ``duration`` (seconds), ``envelope``
    (normalized peak amplitudes for the waveform strip).
    """
    from ..audio import FEATURE_SR, load_audio

    path = Path(path)
    y = load_audio(path, FEATURE_SR)
    raw, mime = _encode_audio(path, bitrate)
    return {
        "src": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}",
        "duration": round(len(y) / FEATURE_SR, 3),
        "envelope": _audio_envelope(y),
    }


def _detail_panels(frame_cols: list[str], segment_cols: list[str]) -> dict:
    """Configure detail-view feature panels from available columns."""
    panels = {"frame": [], "segment": []}

    for name, label in _PANEL_LABELS.items():
        cols = get_model_columns(frame_cols, name)
        cols = [c for c in cols if _is_scalar_col(c)]
        if not cols:
            continue
        kind = "bars_sorted" if name == "sound_events" else "bars_norm"
        panels["frame"].append(
            {"id": name, "label": label, "kind": kind, "features": cols}
        )

    conf = [c for c in ("asr_confidence", "no_speech_prob") if c in segment_cols]
    if conf:
        panels["segment"].append(
            {"id": "confidence", "label": "Whisper confidence", "kind": "bars_prob",
             "features": conf}
        )
    return panels


def create_dashboard(
    frames_df: pd.DataFrame | None,
    transcript_df: pd.DataFrame | None = None,
    *,
    audio: dict | None = None,
    meta: dict | None = None,
    title: str = "aud2psy Dashboard",
    hop: float | None = None,
    max_points: int = 2000,
    mds_max: int = 500,
) -> str:
    """Create the interactive dashboard as a self-contained HTML string.

    Parameters
    ----------
    frames_df : pd.DataFrame or None
        Frame-level scores (``*_frames.csv``: ``time`` + feature columns).
    transcript_df : pd.DataFrame or None
        Transcript segments (``*_transcript.csv``).
    audio : dict, optional
        Output of :func:`prepare_audio`. Without it the dashboard still
        works, minus playback.
    meta : dict, optional
        Parsed ``.meta.json`` sidecar (used for hop and input info).
    title : str
        Page title.
    hop : float, optional
        Frame window size in seconds; inferred from ``meta`` or the ``time``
        column when omitted.
    max_points : int
        Truncate tables beyond this many rows (protects file size and
        projection cost).
    mds_max : int
        Use MDS for clustering up to this many rows, PCA beyond.
    """
    if frames_df is None and transcript_df is None:
        raise ValueError("Need at least one of frames_df / transcript_df.")

    truncated = {}
    if frames_df is not None and len(frames_df) > max_points:
        truncated["frames"] = len(frames_df)
        frames_df = frames_df.iloc[:max_points].reset_index(drop=True)
    if transcript_df is not None and len(transcript_df) > max_points:
        truncated["segments"] = len(transcript_df)
        transcript_df = transcript_df.iloc[:max_points].reset_index(drop=True)

    hop = hop or _infer_hop(frames_df, meta)

    models = {}
    frame_feature_cols: list[str] = []
    segment_feature_cols: list[str] = []

    payload: dict = {
        "title": title,
        "hop": hop,
        "models": models,
        "truncated": truncated,
        "audio": audio,
    }

    # --- frames table ---
    if frames_df is not None and len(frames_df):
        detected = detect_models_in_dataframe(frames_df.columns.tolist(), level="frame")
        for name in detected:
            entry = _build_model_entry(frames_df, name, mds_max)
            if entry:
                models[name] = entry
        for name in detected:
            for c in get_model_columns(frames_df.columns.tolist(), name):
                if c not in frame_feature_cols:
                    frame_feature_cols.append(c)

        scalar_cols = [c for c in frame_feature_cols if _is_scalar_col(c)]
        payload["frames"] = {
            "time": _values(frames_df["time"]),
            "features": {c: _values(frames_df[c]) for c in scalar_cols},
        }
        payload.setdefault("ranges", {}).update(
            _feature_ranges(frames_df, scalar_cols)
        )

    # --- transcript segments ---
    if transcript_df is not None and len(transcript_df):
        detected = detect_models_in_dataframe(
            transcript_df.columns.tolist(), level="segment"
        )
        for name in detected:
            entry = _build_model_entry(transcript_df, name, mds_max)
            if entry:
                models[name] = entry
        for name in detected:
            for c in get_model_columns(transcript_df.columns.tolist(), name):
                if c not in segment_feature_cols:
                    segment_feature_cols.append(c)

        extra_cols = [
            c for c in transcript_df.columns
            if c not in _SEGMENT_INDEX_COLS and c not in segment_feature_cols
        ]
        payload["segments"] = {
            "text": _values(transcript_df["text"]),
            "onset": _values(transcript_df["onset"]),
            "offset": _values(transcript_df["offset"]),
            "features": {c: _values(transcript_df[c]) for c in segment_feature_cols},
            "extra": {c: _values(transcript_df[c]) for c in extra_cols},
        }
        payload.setdefault("ranges", {}).update(
            _feature_ranges(transcript_df, segment_feature_cols)
        )

    if not models:
        raise ValueError(
            "No aud2psy model outputs detected in the provided CSV(s). "
            "Expected columns like 'loudness_rms', 'clap_000', ..."
        )

    payload["panels"] = _detail_panels(frame_feature_cols, segment_feature_cols)

    payload_json = json.dumps(payload, separators=(",", ":"))
    html = (
        _HTML_TEMPLATE
        .replace("__TITLE__", title)
        .replace("__PLOTLY_CDN__", PLOTLY_CDN)
        .replace("__PAYLOAD__", payload_json)
    )
    return html


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<script src="__PLOTLY_CDN__"></script>
<style>
* { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
body { margin: 0; padding: 20px; background: #f5f5f5; }
.container { max-width: 1100px; margin: 0 auto; }
h1 { color: #333; margin-bottom: 5px; }
.subtitle { color: #666; margin-bottom: 20px; }
.controls { display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
.control-group { background: white; padding: 15px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.control-group > label { display: block; font-weight: 600; margin-bottom: 8px; color: #333; }
select { padding: 8px 12px; font-size: 14px; border: 1px solid #ddd; border-radius: 4px; min-width: 200px; }
.viz-buttons { display: flex; gap: 8px; }
.viz-btn { padding: 8px 16px; font-size: 14px; border: 2px solid #ddd; border-radius: 4px; background: white; cursor: pointer; transition: all 0.2s; }
.viz-btn:hover { border-color: #007bff; }
.viz-btn.active { background: #007bff; color: white; border-color: #007bff; }
.viz-btn.disabled { opacity: 0.4; cursor: not-allowed; }
.sub-options { display: none; margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee; }
.sub-options.visible { display: block; }
.sub-toggle { display: inline-flex; background: #f0f0f0; border-radius: 4px; overflow: hidden; margin-right: 12px; }
.sub-toggle button { padding: 6px 12px; font-size: 12px; border: none; background: transparent; cursor: pointer; }
.sub-toggle button.active { background: #007bff; color: white; }
.toggle-label { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; font-size: 13px; }
.model-info { font-size: 12px; color: #666; margin-top: 5px; max-width: 240px; }
.plot-container { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; }
#plot { width: 100%; height: 600px; }
.warning { display: none; text-align: center; padding: 60px 20px; color: #666; }
.warning .emoji { font-size: 64px; margin-bottom: 20px; }
.warning .message { font-size: 18px; margin-bottom: 10px; }
.warning .detail { font-size: 14px; color: #999; }
.footnote { color: #999; font-size: 12px; margin-top: 12px; }

/* Audio transport (the stimulus analog: no picture, so play/pause) */
.audio-group { flex: 1; min-width: 320px; }
.audio-row { display: flex; align-items: center; gap: 12px; }
.play-btn { width: 44px; height: 44px; border-radius: 50%; border: 2px solid #007bff; background: white; color: #007bff; font-size: 17px; cursor: pointer; transition: all 0.15s; flex: none; padding: 0; line-height: 1; }
.play-btn:hover { background: #007bff; color: white; }
.play-btn.playing { background: #007bff; color: white; }
.audio-time { font-size: 13px; color: #555; min-width: 110px; font-variant-numeric: tabular-nums; }
.wave-strip { width: 100%; height: 56px; display: block; margin-top: 10px; cursor: pointer; border-radius: 4px; background: #f8f9fb; }
.audio-note { color: #999; font-size: 13px; }

/* Detail overlay (the "single image viewer" analog) */
#detail-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.45); z-index: 100; }
#detail-overlay.open { display: flex; align-items: center; justify-content: center; }
.detail-box { background: #f5f5f5; border-radius: 10px; width: min(1060px, 94vw); max-height: 92vh; overflow: auto; padding: 20px 24px; position: relative; }
.detail-close { position: absolute; top: 12px; right: 14px; border: none; background: transparent; font-size: 22px; cursor: pointer; color: #666; }
.detail-close:hover { color: #000; }
.detail-content { display: flex; gap: 20px; flex-wrap: wrap; }
.stimulus-panel { background: white; border-radius: 8px; padding: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; min-width: 300px; display: flex; flex-direction: column; }
.stimulus-time { font-size: 40px; font-weight: 700; color: #1a1a2e; text-align: center; margin: 8px 0 4px 0; font-variant-numeric: tabular-nums; }
.stimulus-text { font-size: 22px; font-weight: 600; color: #1a1a2e; text-align: center; margin: 14px 0 6px 0; line-height: 1.45; }
.stimulus-meta { text-align: center; color: #888; font-size: 13px; margin-bottom: 14px; }
.big-play-wrap { display: flex; align-items: center; justify-content: center; gap: 16px; margin: 14px 0; }
.big-play { width: 84px; height: 84px; border-radius: 50%; border: 3px solid #007bff; background: white; color: #007bff; font-size: 34px; cursor: pointer; transition: all 0.15s; padding: 0; line-height: 1; }
.big-play:hover { background: #007bff; color: white; }
.big-play.playing { background: #007bff; color: white; }
.no-audio { text-align: center; color: #999; margin: 18px 0; }
.no-audio .emoji { font-size: 56px; }
.detail-strip { width: 100%; height: 52px; display: block; margin-top: auto; cursor: pointer; border-radius: 4px; background: #f8f9fb; }
.features-panel { background: white; border-radius: 8px; padding: 16px 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1.3; min-width: 380px; }
.features-panel select { margin-bottom: 6px; }
#detail-plot { width: 100%; height: 420px; }
.detail-nav { display: flex; align-items: center; gap: 12px; margin-top: 16px; background: white; border-radius: 8px; padding: 12px 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.detail-nav button { padding: 6px 14px; font-size: 15px; border: 1px solid #ccc; border-radius: 4px; background: white; cursor: pointer; }
.detail-nav button:hover { border-color: #007bff; }
.detail-nav input[type="range"] { flex: 1; }
.detail-nav .pos { font-size: 13px; color: #555; min-width: 110px; text-align: right; }
</style>
</head>
<body>
<div class="container">
  <h1>__TITLE__</h1>
  <p class="subtitle" id="subtitle"></p>

  <div class="controls">
    <div class="control-group">
      <label>Model Output</label>
      <select id="model-select" onchange="updatePlot()"></select>
      <div id="model-info" class="model-info"></div>
    </div>

    <div class="control-group">
      <label>Visualization</label>
      <div class="viz-buttons">
        <button class="viz-btn active" data-viz="timeseries" onclick="selectViz('timeseries')">&#128200; Time Series</button>
        <button class="viz-btn" data-viz="clustering" onclick="selectViz('clustering')">&#128309; Clustering</button>
        <button class="viz-btn" data-viz="trajectory" onclick="selectViz('trajectory')">&#128640; Trajectory</button>
      </div>
      <div id="sub-timeseries" class="sub-options">
        <label class="toggle-label"><input type="checkbox" id="rolling-toggle" onchange="updatePlot()"><span>Smooth (rolling avg)</span></label>
      </div>
      <div id="sub-clustering" class="sub-options">
        <div class="sub-toggle">
          <button class="active" data-sub="2d" onclick="selectSubOption('clustering','2d')">2D</button>
          <button data-sub="3d" onclick="selectSubOption('clustering','3d')">3D</button>
        </div>
      </div>
      <div id="sub-trajectory" class="sub-options"></div>
    </div>

    <div class="control-group audio-group" id="audio-group">
      <label>Audio</label>
      <div id="audio-controls" class="audio-row" style="display:none">
        <button id="play-btn" class="play-btn" onclick="togglePlay()" title="Play / pause the full clip">&#9654;</button>
        <span id="audio-time" class="audio-time"></span>
      </div>
      <canvas id="wave-strip" class="wave-strip" style="display:none"></canvas>
      <div id="audio-note" class="audio-note" style="display:none">
        No audio embedded &mdash; rebuild with <code>--audio &lt;clip&gt;</code> to enable playback.
      </div>
    </div>
  </div>

  <div class="plot-container">
    <div id="plot"></div>
    <div id="warning" class="warning">
      <div class="emoji">&#128533;</div>
      <div class="message" id="warning-message">Not available</div>
      <div class="detail" id="warning-detail"></div>
    </div>
  </div>
  <div class="footnote" id="footnote"></div>
</div>

<div id="detail-overlay" onclick="if(event.target===this)closeDetail()">
  <div class="detail-box">
    <button class="detail-close" onclick="closeDetail()">&#10005;</button>
    <div class="detail-content">
      <div class="stimulus-panel">
        <div id="stimulus-body"></div>
        <canvas id="detail-strip" class="detail-strip" style="display:none"></canvas>
      </div>
      <div class="features-panel">
        <select id="panel-select" onchange="renderDetailPanel()"></select>
        <div id="detail-plot"></div>
      </div>
    </div>
    <div class="detail-nav">
      <button onclick="stepDetail(-1)" title="Previous (left arrow)">&#9664;</button>
      <input type="range" id="detail-slider" min="0" max="0" value="0" oninput="jumpDetail(parseInt(this.value))">
      <button onclick="stepDetail(1)" title="Next (right arrow)">&#9654;</button>
      <span class="pos" id="detail-pos"></span>
    </div>
  </div>
</div>

<audio id="player" preload="auto"></audio>

<script>
const DATA = __PAYLOAD__;

const COLORS = { bar: '#636EFA', prob: '#00CC96', sorted: '#EF553B' };
let currentViz = 'timeseries';
let subOptions = { clustering: '2d' };
let detail = { level: null, idx: 0, panel: null, loop: false };

const FRAMES = DATA.frames || null;
const SEGMENTS = DATA.segments || null;
const AUDIO = DATA.audio || null;
const HOP = DATA.hop;
const nFrames = FRAMES ? FRAMES.time.length : 0;
const nSegments = SEGMENTS ? SEGMENTS.text.length : 0;

function prettify(name) {
  return name.replace(/^(loudness|pitch|spectral|onsets|tonal|rhythm|timbre|psychoacoustic|music_emotion|sound_events|speech_emotion|egemaps)_/, '').replace(/_/g, ' ');
}
function fmtTime(t) { return (t === null || t === undefined) ? '?' : t.toFixed(2) + ' s'; }
function rowLabel(level, i) {
  return level === 'frame' ? fmtTime(FRAMES.time[i])
                           : truncate(String(SEGMENTS.text[i] || ('segment ' + i)), 24);
}
function nRows(level) { return level === 'frame' ? nFrames : nSegments; }
function tableFeatures(level) { return level === 'frame' ? FRAMES.features : SEGMENTS.features; }
function rowRange(level, i) {
  // The playable audio span of a row: hop window around the frame center,
  // or the transcript segment's onset-offset.
  if (level === 'frame') {
    const t = FRAMES.time[i];
    return [Math.max(0, t - HOP / 2), t + HOP / 2];
  }
  return [SEGMENTS.onset[i], SEGMENTS.offset[i]];
}
function rowTime(level, i) { return level === 'frame' ? FRAMES.time[i] : SEGMENTS.onset[i]; }

// ---------- init ----------
function init() {
  const sel = document.getElementById('model-select');
  const groups = { frame: [], segment: [] };
  for (const [name, m] of Object.entries(DATA.models)) groups[m.level].push(name);
  for (const [level, names] of Object.entries(groups)) {
    if (!names.length) continue;
    const og = document.createElement('optgroup');
    og.label = level === 'frame' ? 'Frame-level' : 'Segment-level';
    for (const name of names) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name + ' (' + DATA.models[name].nDims + 'd)';
      og.appendChild(opt);
    }
    sel.appendChild(og);
  }

  const parts = [];
  if (nFrames) parts.push(nFrames + ' frames (' + HOP + ' s hop)');
  if (nSegments) parts.push(nSegments + ' transcript segments');
  parts.push(Object.keys(DATA.models).length + ' model outputs');
  document.getElementById('subtitle').textContent =
    parts.join(' · ') + ' — click any point to inspect and play it.';

  if (AUDIO) {
    document.getElementById('player').src = AUDIO.src;
    document.getElementById('audio-controls').style.display = 'flex';
    document.getElementById('wave-strip').style.display = 'block';
    setupAudio();
  } else {
    document.getElementById('audio-note').style.display = 'block';
  }

  const trunc = [];
  for (const [tbl, total] of Object.entries(DATA.truncated || {}))
    trunc.push('showing first ' + (tbl === 'frames' ? nFrames : nSegments) + ' of ' + total + ' ' + tbl);
  document.getElementById('footnote').textContent = trunc.join('; ');

  document.getElementById('sub-timeseries').classList.add('visible');
  updatePlot();
  document.addEventListener('keydown', e => {
    if (!document.getElementById('detail-overlay').classList.contains('open')) return;
    if (e.key === 'Escape') closeDetail();
    if (e.key === 'ArrowLeft') { stepDetail(-1); e.preventDefault(); }
    if (e.key === 'ArrowRight') { stepDetail(1); e.preventDefault(); }
    if (e.key === ' ') { toggleDetailPlay(); e.preventDefault(); }
  });
}

// ---------- audio transport ----------
const player = document.getElementById('player');
let playRangeState = null;   // {start, end, loop} while playing a row window
let rafId = null;
let lastPlotCursor = 0;

function togglePlay() {
  if (!AUDIO) return;
  if (player.paused) { playRangeState = null; player.play(); }
  else player.pause();
}
function playRange(start, end, loop) {
  playRangeState = { start: start, end: end, loop: loop };
  player.currentTime = start;
  player.play();
}
function seek(t) {
  if (!AUDIO) return;
  playRangeState = null;
  player.currentTime = Math.max(0, Math.min(AUDIO.duration, t));
  updateAudioUI(true);
}
player.addEventListener('play', () => { cancelAnimationFrame(rafId); tick(); syncPlayButtons(); });
player.addEventListener('pause', () => { cancelAnimationFrame(rafId); syncPlayButtons(); updateAudioUI(true); });
player.addEventListener('ended', () => { playRangeState = null; syncPlayButtons(); });

function tick() {
  if (playRangeState && player.currentTime >= playRangeState.end) {
    if (playRangeState.loop) player.currentTime = playRangeState.start;
    else { player.pause(); return; }
  }
  updateAudioUI(false);
  if (!player.paused) rafId = requestAnimationFrame(tick);
}

function syncPlayButtons() {
  const playing = !player.paused;
  const main = document.getElementById('play-btn');
  main.innerHTML = (playing && !playRangeState) ? '&#10074;&#10074;' : '&#9654;';
  main.classList.toggle('playing', playing && !playRangeState);
  const big = document.getElementById('big-play');
  if (big) {
    big.innerHTML = (playing && playRangeState) ? '&#10074;&#10074;' : '&#9654;';
    big.classList.toggle('playing', playing && playRangeState);
  }
}

function updateAudioUI(force) {
  if (!AUDIO) return;
  const t = player.currentTime;
  document.getElementById('audio-time').textContent =
    t.toFixed(1) + ' / ' + AUDIO.duration.toFixed(1) + ' s';
  drawStrip(document.getElementById('wave-strip'), t, null);
  if (document.getElementById('detail-overlay').classList.contains('open'))
    drawDetailStrip(t);
  const now = performance.now();
  if (force || now - lastPlotCursor > 80) { lastPlotCursor = now; updatePlotCursor(t); }
}

function drawStrip(canvas, t, highlight) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (canvas.width !== W * dpr) { canvas.width = W * dpr; canvas.height = H * dpr; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const env = AUDIO.envelope, n = env.length, dur = AUDIO.duration;
  const playedX = (t / dur) * W;
  if (highlight) {
    ctx.fillStyle = 'rgba(255,193,7,0.35)';
    const x0 = (highlight[0] / dur) * W, x1 = (highlight[1] / dur) * W;
    ctx.fillRect(x0, 0, Math.max(1.5, x1 - x0), H);
  }
  for (let i = 0; i < n; i++) {
    const x = (i + 0.5) / n * W;
    const h = Math.max(1, env[i] * (H - 6));
    ctx.fillStyle = x <= playedX ? '#007bff' : '#b8c4d8';
    ctx.fillRect(x - 0.5, (H - h) / 2, 1, h);
  }
  ctx.fillStyle = '#e53935';
  ctx.fillRect(playedX - 1, 0, 2, H);
}

function stripSeek(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  const frac = (event.clientX - rect.left) / rect.width;
  seek(frac * AUDIO.duration);
}

function setupAudio() {
  const strip = document.getElementById('wave-strip');
  strip.addEventListener('click', e => stripSeek(strip, e));
  const dstrip = document.getElementById('detail-strip');
  dstrip.addEventListener('click', e => stripSeek(dstrip, e));
  updateAudioUI(true);
}

// The moving time cursor on the overview plot. Timeseries gets a vertical
// line (x is seconds); frame-level 2D scatter/trajectory gets a marker on
// the frame nearest the playhead (3D restyle is too slow — skipped).
let cursorState = null; // {kind: 'line'} | {kind: 'point', proj, traceIdx, level}
function updatePlotCursor(t) {
  if (!cursorState) return;
  const plotDiv = document.getElementById('plot');
  if (cursorState.kind === 'line') {
    Plotly.relayout(plotDiv, { 'shapes[0].x0': t, 'shapes[0].x1': t });
  } else {
    const idx = nearestRow(cursorState.level, t);
    const pos = cursorState.proj.indices.indexOf(idx);
    if (pos >= 0) {
      Plotly.restyle(plotDiv, { x: [[cursorState.proj.x[pos]]], y: [[cursorState.proj.y[pos]]] },
                     [cursorState.traceIdx]);
    }
  }
}
function nearestRow(level, t) {
  if (level === 'frame') {
    let best = 0, bestD = Infinity;
    for (let i = 0; i < nFrames; i++) {
      const d = Math.abs(FRAMES.time[i] - t);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }
  for (let i = 0; i < nSegments; i++)
    if (t >= SEGMENTS.onset[i] && t < SEGMENTS.offset[i]) return i;
  return -1;
}

// ---------- overview plots ----------
function selectViz(viz) {
  currentViz = viz;
  document.querySelectorAll('.viz-btn').forEach(b => b.classList.toggle('active', b.dataset.viz === viz));
  for (const v of ['timeseries', 'clustering', 'trajectory'])
    document.getElementById('sub-' + v).classList.toggle('visible', v === viz);
  updatePlot();
}
function selectSubOption(viz, sub) {
  subOptions[viz] = sub;
  document.querySelectorAll('#sub-' + viz + ' .sub-toggle button')
    .forEach(b => b.classList.toggle('active', b.dataset.sub === sub));
  updatePlot();
}

const VIZ_WARNINGS = {
  timeseries: ['Time series not available for this model',
               'High-dimensional embeddings are not interpretable as individual time series.'],
  clustering: ['Clustering not available for this model',
               'This model’s output cannot be projected to 2D/3D.'],
  trajectory: ['Trajectory not available for this model',
               'This model’s output cannot show state-space evolution.'],
};

function rollingMean(values, w) {
  const half = Math.floor(w / 2), out = [];
  for (let i = 0; i < values.length; i++) {
    let s = 0, n = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(values.length - 1, i + half); j++) {
      const v = values[j];
      if (v !== null && v !== undefined) { s += v; n++; }
    }
    out.push(n ? s / n : null);
  }
  return out;
}

function hoverText(level, i) {
  if (level === 'frame') return fmtTime(FRAMES.time[i]) + '<br>frame ' + i;
  let t = truncate(String(SEGMENTS.text[i]), 60) + '<br>' +
          fmtTime(SEGMENTS.onset[i]) + ' – ' + fmtTime(SEGMENTS.offset[i]);
  for (const [k, vals] of Object.entries(SEGMENTS.extra || {}))
    if (vals[i] !== null && vals[i] !== '') t += '<br>' + k + ': ' + vals[i];
  return t;
}

function updatePlot() {
  const modelName = document.getElementById('model-select').value;
  const model = DATA.models[modelName];
  const level = model.level;
  document.getElementById('model-info').textContent = model.nDims + ' dimensions — ' + model.description;

  document.querySelectorAll('.viz-btn').forEach(btn => {
    const v = btn.dataset.viz;
    const ok = v === 'timeseries' ? model.available.timeseries
             : v === 'clustering' ? (model.available.clustering2d || model.available.clustering3d)
             : model.available.trajectory;
    btn.classList.toggle('disabled', !ok);
  });

  const plotDiv = document.getElementById('plot');
  const warnDiv = document.getElementById('warning');
  let traces = null, layout = null;
  cursorState = null;

  if (currentViz === 'timeseries' && model.available.timeseries) {
    [traces, layout] = buildTimeseries(model, level);
  } else if (currentViz === 'clustering') {
    const key = subOptions.clustering;
    const proj = model.clustering[key];
    if (proj) [traces, layout] = buildScatter(proj, level, key === '3d', false, modelName);
  } else if (currentViz === 'trajectory' && model.available.trajectory) {
    [traces, layout] = buildScatter(model.trajectory, level, false, true, modelName);
  }

  if (!traces) {
    plotDiv.style.display = 'none';
    warnDiv.style.display = 'block';
    const [msg, det] = VIZ_WARNINGS[currentViz];
    document.getElementById('warning-message').textContent = msg;
    document.getElementById('warning-detail').textContent = det;
    return;
  }
  plotDiv.style.display = 'block';
  warnDiv.style.display = 'none';
  Plotly.react('plot', traces, layout, { responsive: true }).then(attachClickHandler);
}

function buildTimeseries(model, level) {
  const feats = model.tsFeatures;
  const featureTable = tableFeatures(level);
  const n = nRows(level);
  const x = level === 'frame' ? FRAMES.time : SEGMENTS.onset;
  const smooth = document.getElementById('rolling-toggle').checked;
  const w = Math.max(2, Math.min(10, Math.floor(n / 3)));
  const customdata = [...Array(n).keys()];

  const traces = feats.map(f => {
    let y = featureTable[f];
    if (smooth) y = rollingMean(y, w);
    return {
      type: 'scatter',
      mode: smooth ? 'lines' : 'lines+markers',
      name: prettify(f),
      x: x, y: y,
      customdata: customdata,
      marker: { size: 5, opacity: 0.75 },
      line: smooth ? { width: 3 } : { width: 1.5 },
      hovertemplate: prettify(f) + '<br>%{text}<br>value: %{y:.4g}<extra></extra>',
      text: customdata.map(i => rowLabel(level, i)),
    };
  });

  const layout = {
    title: (level === 'frame' ? 'Frame sequence — ' : 'Transcript segments — ') + currentModelName(),
    xaxis: { title: 'Time (s)' },
    yaxis: { title: 'Value' },
    hovermode: 'closest',
    margin: { t: 50, r: 30, b: 60, l: 60 },
  };
  if (AUDIO) {
    layout.shapes = [{
      type: 'line', xref: 'x', yref: 'paper',
      x0: player.currentTime, x1: player.currentTime, y0: 0, y1: 1,
      line: { color: 'rgba(229,57,53,0.75)', width: 1.5 },
    }];
    cursorState = { kind: 'line' };
  }
  return [traces, layout];
}

function currentModelName() { return document.getElementById('model-select').value; }
function truncate(s, n) { return s.length > n ? s.slice(0, n) + '…' : s; }

function buildScatter(proj, level, is3d, asTrajectory, modelName) {
  const idx = proj.indices;
  const colorVals = idx.map(i => rowTime(level, i));

  const traces = [];
  if (asTrajectory) {
    traces.push({
      type: 'scatter', mode: 'lines',
      x: proj.x, y: proj.y,
      line: { color: 'rgba(100,100,100,0.45)', width: 1 },
      hoverinfo: 'skip', showlegend: false,
    });
  }
  const main = {
    type: is3d ? 'scatter3d' : 'scatter',
    mode: 'markers',
    x: proj.x, y: proj.y,
    marker: {
      size: is3d ? 5 : 10,
      color: colorVals,
      colorscale: 'Viridis',
      colorbar: { title: 'Time (s)' },
      opacity: 0.85,
    },
    customdata: idx,
    hovertemplate: idx.map(i => hoverText(level, i) + '<extra></extra>'),
    showlegend: false,
  };
  if (is3d) main.z = proj.z;
  traces.push(main);

  if (asTrajectory && idx.length > 1) {
    traces.push({ type: 'scatter', mode: 'markers', x: [proj.x[0]], y: [proj.y[0]],
      marker: { size: 16, color: 'green', symbol: 'circle-open', line: { width: 3 } },
      name: 'Start', hovertemplate: 'Start<extra></extra>' });
    traces.push({ type: 'scatter', mode: 'markers',
      x: [proj.x[proj.x.length - 1]], y: [proj.y[proj.y.length - 1]],
      marker: { size: 16, color: 'red', symbol: 'square-open', line: { width: 3 } },
      name: 'End', hovertemplate: 'End<extra></extra>' });
  }

  // Playhead marker: the row nearest the audio position (2D only)
  if (AUDIO && !is3d) {
    const idx0 = nearestRow(level, player.currentTime);
    const pos0 = Math.max(0, proj.indices.indexOf(idx0));
    traces.push({
      type: 'scatter', mode: 'markers',
      x: [proj.x[pos0]], y: [proj.y[pos0]],
      marker: { size: 18, color: 'rgba(0,0,0,0)', symbol: 'circle-open',
                line: { width: 3, color: '#e53935' } },
      hoverinfo: 'skip', showlegend: false,
    });
    cursorState = { kind: 'point', proj: proj, traceIdx: traces.length - 1, level: level };
  }

  const methodLabel = proj.method.toUpperCase();
  const title = modelName + ' — ' + (asTrajectory
    ? 'state-space trajectory (' + methodLabel + ')'
    : methodLabel + ' ' + (is3d ? '3D' : '2D') + ' projection');
  const layout = { title: title, hovermode: 'closest', margin: { t: 50, r: 30, b: 60, l: 60 },
                   showlegend: asTrajectory };
  if (is3d) {
    layout.scene = { xaxis: { title: proj.xlabel }, yaxis: { title: proj.ylabel },
                     zaxis: { title: proj.zlabel } };
  } else {
    layout.xaxis = { title: proj.xlabel };
    layout.yaxis = { title: proj.ylabel };
  }
  return [traces, layout];
}

function attachClickHandler() {
  const plotDiv = document.getElementById('plot');
  plotDiv.on('plotly_click', data => {
    if (!data.points || !data.points.length) return;
    const p = data.points[0];
    let idx = (p.customdata !== undefined && p.customdata !== null) ? p.customdata : p.pointIndex;
    if (Array.isArray(idx)) idx = idx[0];
    const model = DATA.models[document.getElementById('model-select').value];
    if (typeof idx === 'number') openDetail(model.level, idx);
  });
}

// ---------- detail viewer ----------
function openDetail(level, idx) {
  const panels = DATA.panels[level];
  if (!panels.length) return;
  const keepPanel = detail.level === level && detail.panel;
  detail.level = level;
  detail.idx = idx;
  detail.panel = keepPanel || panels[0].id;

  const sel = document.getElementById('panel-select');
  sel.innerHTML = '';
  for (const p of panels) {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = p.label;
    if (p.id === detail.panel) opt.selected = true;
    sel.appendChild(opt);
  }
  const slider = document.getElementById('detail-slider');
  slider.max = nRows(level) - 1;
  slider.value = idx;

  document.getElementById('detail-overlay').classList.add('open');
  renderDetail();
}
function closeDetail() {
  document.getElementById('detail-overlay').classList.remove('open');
  if (playRangeState) player.pause();
}
function stepDetail(d) {
  jumpDetail(Math.min(nRows(detail.level) - 1, Math.max(0, detail.idx + d)));
}
function jumpDetail(idx) {
  const wasPlaying = !player.paused && playRangeState;
  detail.idx = idx;
  document.getElementById('detail-slider').value = idx;
  renderDetail();
  // Scrubbing while playing: keep playing the new row's window
  if (wasPlaying) {
    const [s, e] = rowRange(detail.level, idx);
    playRange(s, e, detail.loop);
  }
}

function toggleDetailPlay() {
  if (!AUDIO) return;
  if (!player.paused && playRangeState) { player.pause(); return; }
  const [s, e] = rowRange(detail.level, detail.idx);
  playRange(s, e, detail.loop);
}

function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function playControlsHtml() {
  if (!AUDIO) {
    return '<div class="no-audio"><div class="emoji">&#128263;</div>' +
           'No audio embedded &mdash; rebuild with <code>--audio</code></div>';
  }
  return '<div class="big-play-wrap">' +
         '<button id="big-play" class="big-play" onclick="toggleDetailPlay()" ' +
         'title="Play this window (space)">&#9654;</button>' +
         '<label class="toggle-label"><input type="checkbox" id="loop-toggle"' +
         (detail.loop ? ' checked' : '') +
         ' onchange="detail.loop=this.checked; if(playRangeState) playRangeState.loop=this.checked;">' +
         '<span>Loop</span></label>' +
         '</div>';
}

function renderDetail() {
  const { level, idx } = detail;
  const body = document.getElementById('stimulus-body');
  const [start, end] = rowRange(level, idx);

  if (level === 'frame') {
    body.innerHTML =
      '<div class="stimulus-time">' + fmtTime(FRAMES.time[idx]) + '</div>' +
      '<div class="stimulus-meta">frame ' + (idx + 1) + ' of ' + nFrames +
      ' · window ' + fmtTime(start) + ' – ' + fmtTime(end) + '</div>' +
      playControlsHtml();
  } else {
    const meta = ['segment ' + (idx + 1) + ' of ' + nSegments,
                  fmtTime(start) + ' – ' + fmtTime(end)];
    for (const [k, vals] of Object.entries(SEGMENTS.extra || {}))
      if (vals[idx] !== null && vals[idx] !== '') meta.push(k + ': ' + vals[idx]);
    body.innerHTML =
      '<div class="stimulus-text">' + escapeHtml(SEGMENTS.text[idx] || '(no text)') + '</div>' +
      '<div class="stimulus-meta">' + escapeHtml(meta.join(' · ')) + '</div>' +
      playControlsHtml();
  }

  if (AUDIO) {
    const dstrip = document.getElementById('detail-strip');
    dstrip.style.display = 'block';
    drawDetailStrip(player.currentTime);
  }
  syncPlayButtons();

  document.getElementById('detail-pos').textContent =
    (level === 'frame' ? 'frame ' : 'segment ') + (idx + 1) + ' / ' + nRows(level);
  renderDetailPanel();
}

function drawDetailStrip(t) {
  if (!AUDIO || detail.level === null) return;
  drawStrip(document.getElementById('detail-strip'), t, rowRange(detail.level, detail.idx));
}

function normValue(feature, v) {
  const r = DATA.ranges[feature];
  if (v === null || v === undefined || !r || r[1] <= r[0]) return null;
  return Math.max(0, Math.min(1, (v - r[0]) / (r[1] - r[0])));
}

function renderDetailPanel() {
  detail.panel = document.getElementById('panel-select').value;
  const { level, idx } = detail;
  const cfg = DATA.panels[level].find(p => p.id === detail.panel);
  const featureTable = tableFeatures(level);
  const feats = cfg.features.filter(f => featureTable[f] !== undefined);
  const raw = feats.map(f => featureTable[f][idx]);
  let traces, layout;

  if (cfg.kind === 'bars_sorted' || cfg.kind === 'bars_prob') {
    let pairs = feats.map((f, i) => [prettify(f), raw[i]]);
    if (cfg.kind === 'bars_sorted') pairs.sort((a, b) => (b[1] ?? -1) - (a[1] ?? -1));
    traces = [{
      type: 'bar', orientation: 'h',
      y: pairs.map(p => p[0]), x: pairs.map(p => p[1]),
      marker: { color: cfg.kind === 'bars_prob' ? COLORS.prob : COLORS.sorted },
      hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
    }];
    layout = {
      xaxis: cfg.kind === 'bars_prob'
        ? { title: 'Probability', range: [0, 1] }
        : { title: 'Value' },
      yaxis: { autorange: 'reversed', automargin: true },
      margin: { t: 20, r: 20, b: 50, l: 10 },
    };
  } else {
    // bars_norm: dataset-normalized values, raw in hover
    const hover = feats.map((f, i) => {
      const r = DATA.ranges[f] || [null, null];
      const v = raw[i];
      return prettify(f) + ': ' + (v === null || v === undefined ? 'n/a' : v.toPrecision(4)) +
             (r[0] !== null ? ' (dataset range ' + r[0] + ' – ' + r[1] + ')' : '');
    });
    traces = [{
      type: 'bar', orientation: 'h',
      y: feats.map(prettify),
      x: feats.map((f, i) => normValue(f, raw[i])),
      marker: { color: COLORS.bar },
      customdata: hover,
      hovertemplate: '%{customdata}<extra></extra>',
    }];
    layout = {
      xaxis: { title: 'Value (normalized to dataset range)', range: [0, 1] },
      yaxis: { autorange: 'reversed', automargin: true },
      margin: { t: 20, r: 20, b: 50, l: 10 },
    };
  }
  layout.height = Math.max(360, feats.length * 22 + 120);
  Plotly.react('detail-plot', traces, layout, { displayModeBar: false, responsive: true });
}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""
