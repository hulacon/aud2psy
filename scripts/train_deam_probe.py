"""Train the music_emotion probe on DEAM dynamic valence/arousal annotations.

One-off developer script (the fitted probe ships as package data — users
never run this). Reproduces src/aud2psy/data/music_emotion_probe.{npz,json}.

Data: DEAM (Aljanaki, Yang, & Soleymani, 2017), downloaded from
https://cvml.unige.ch/databases/DEAM/ into ~/.cache/aud2psy/deam/
(DEAM_audio.zip + DEAM_Annotations.zip, both unzipped). Dynamic
annotations are averaged-per-song valence/arousal at 2 Hz from 15 s in,
values in [-1, 1].

Protocol (design checkpoint, Aug 2026): 45-s excerpts only (duration
< 60 s), CLAP embeddings from the exact runtime pipeline
(ClapModel.embed_windows, 10 s windows) at a 2 s training stride,
RidgeCV per dimension, song-level 5-fold GroupKFold CV reported before
the final all-data fit. Embeddings are cached so refits are cheap.

Usage: .venv/bin/python scripts/train_deam_probe.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aud2psy.audio import load_audio  # noqa: E402
from aud2psy.models.clap import DEFAULT_CHECKPOINT, WINDOW_SEC, ClapModel  # noqa: E402

DEAM_DIR = Path.home() / ".cache" / "aud2psy" / "deam"
AUDIO_DIR = DEAM_DIR / "MEMD_audio"
EMB_CACHE = DEAM_DIR / "clap_embeddings_2s.npz"
OUT_DIR = Path(__file__).resolve().parents[1] / "src" / "aud2psy" / "data"
TRAIN_STRIDE_SEC = 2.0
MAX_EXCERPT_SEC = 60.0  # keep the 1744 uniform 45-s excerpts, skip full songs
SR = 48000
ALPHAS = np.logspace(-2, 4, 13)


def load_annotations() -> dict[str, pd.DataFrame]:
    averaged = next(DEAM_DIR.glob("**/annotations averaged per song"), None)
    dyn = next(averaged.glob("dynamic*"), None) if averaged else None
    if dyn is None:
        sys.exit(f"Averaged dynamic annotations not found under {DEAM_DIR}; unzip DEAM_Annotations.zip")
    out = {}
    for dim in ("valence", "arousal"):
        df = pd.read_csv(next(dyn.glob(f"**/{dim}.csv")), index_col=0)
        times = [int(re.search(r"(\d+)ms", c).group(1)) / 1000 for c in df.columns]
        df.columns = times
        out[dim] = df
    return out


def embed_dataset(ann: dict[str, pd.DataFrame]):
    """Returns X (n, 512), targets (n, 2), groups (n,) — cached on disk."""
    if EMB_CACHE.exists():
        z = np.load(EMB_CACHE)
        return z["X"], z["Y"], z["groups"]

    val, aro = ann["valence"], ann["arousal"]
    song_ids = [s for s in val.index if (AUDIO_DIR / f"{s}.mp3").exists()]
    model = ClapModel()
    model.load()
    X, Y, groups = [], [], []
    for k, song in enumerate(song_ids):
        y = load_audio(AUDIO_DIR / f"{song}.mp3", SR)
        duration = len(y) / SR
        if duration >= MAX_EXCERPT_SEC:
            continue  # full-length songs excluded by design
        cols = [
            t for t in val.columns
            if t % TRAIN_STRIDE_SEC == 0 and t <= duration - 0.25
            and np.isfinite(val.loc[song, t]) and np.isfinite(aro.loc[song, t])
        ]
        if not cols:
            continue
        emb = model.embed_windows(y, SR, np.array(cols))
        X.append(emb)
        Y.append(np.column_stack([val.loc[song, cols], aro.loc[song, cols]]))
        groups.append(np.full(len(cols), song))
        if (k + 1) % 200 == 0:
            print(f"  embedded {k + 1}/{len(song_ids)} songs")
    X = np.vstack(X)
    Y = np.vstack(Y)
    groups = np.concatenate(groups)
    np.savez_compressed(EMB_CACHE, X=X, Y=Y, groups=groups)
    return X, Y, groups


def songwise_r(y_true, y_pred, groups) -> float:
    rs = []
    for song in np.unique(groups):
        m = groups == song
        if m.sum() > 3 and np.std(y_true[m]) > 1e-9 and np.std(y_pred[m]) > 1e-9:
            rs.append(np.corrcoef(y_true[m], y_pred[m])[0, 1])
    return float(np.mean(rs))


def main():
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold

    ann = load_annotations()
    X, Y, groups = embed_dataset(ann)
    print(f"training set: {X.shape[0]} samples from {len(np.unique(groups))} songs")

    cv_report = {}
    for i, dim in enumerate(("valence", "arousal")):
        pred = np.full(len(X), np.nan)  # out-of-fold predictions, index-aligned
        for train, test in GroupKFold(n_splits=5).split(X, Y[:, i], groups):
            reg = RidgeCV(alphas=ALPHAS).fit(X[train], Y[train, i])
            pred[test] = reg.predict(X[test])
        cv_report[dim] = {
            "pooled_r": round(float(np.corrcoef(Y[:, i], pred)[0, 1]), 3),
            "rmse": round(float(np.sqrt(np.mean((Y[:, i] - pred) ** 2))), 3),
            "mean_song_r": round(songwise_r(Y[:, i], pred, groups), 3),
        }
        print(f"{dim}: {cv_report[dim]}")

    coef = np.empty((2, X.shape[1]))
    intercept = np.empty(2)
    alphas_used = []
    for i in range(2):
        reg = RidgeCV(alphas=ALPHAS).fit(X, Y[:, i])
        coef[i] = reg.coef_
        intercept[i] = reg.intercept_
        alphas_used.append(float(reg.alpha_))

    OUT_DIR.mkdir(exist_ok=True)
    np.savez(OUT_DIR / "music_emotion_probe.npz", coef=coef, intercept=intercept)
    provenance = {
        "dataset": "DEAM (Aljanaki, Yang, & Soleymani, 2017)",
        "source": "https://cvml.unige.ch/databases/DEAM/",
        "checkpoint": DEFAULT_CHECKPOINT,
        "window_sec": WINDOW_SEC,
        "train_stride_sec": TRAIN_STRIDE_SEC,
        "n_samples": int(X.shape[0]),
        "n_songs": int(len(np.unique(groups))),
        "dimensions": ["valence", "arousal"],
        "ridge_alphas": alphas_used,
        "cv": cv_report,
        "cv_protocol": "5-fold GroupKFold by song",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (OUT_DIR / "music_emotion_probe.json").write_text(json.dumps(provenance, indent=2))
    print(f"probe written to {OUT_DIR}")


if __name__ == "__main__":
    main()
