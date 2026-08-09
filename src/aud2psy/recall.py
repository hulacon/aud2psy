"""Free-recall export: wordpool matching and speech-timing metrics.

Post-processing on the transcribe model's word-timestamp table — the
automated equivalent of Penn-TotalRecall hand annotation. Not a model:
runs when `--wordpool` is given alongside `transcribe`, emitting a
`_recall.csv` with one row per spoken word.

Matching is exact-first, then fuzzy via difflib ratio (stdlib; threshold
MATCH_THRESHOLD). `irt` is the inter-response time between successive
*matched* recalls (NaN for the first recall and for intrusions) — the
list-recall literature's definition. N.B. onsets inherit faster-whisper's
word-timestamp accuracy (~100–200 ms); see the README caveat before using
them for onset-locked EEG.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

MATCH_THRESHOLD = 0.8
PAUSE_MIN_SEC = 0.5

RECALL_COLUMNS = [
    "word", "onset", "offset",
    "matched_item", "pool_index", "match_score",
    "intrusion", "repetition", "irt",
]


def _normalize(word: str) -> str:
    return re.sub(r"[^a-z']", "", word.lower())


def load_wordpool(path: str | Path) -> list[str]:
    """One item per line; blank lines and '#' comments ignored."""
    items = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            items.append(line)
    if not items:
        raise ValueError(f"Wordpool file {path} contains no items")
    return items


def annotate_recall(
    words_df: pd.DataFrame,
    wordpool: list[str],
    match_threshold: float = MATCH_THRESHOLD,
) -> pd.DataFrame:
    """One row per spoken word, annotated against the wordpool."""
    if "is_filler" in words_df.columns:
        # verbatim-mode filler tags ([UH]) are not recall attempts — they'd
        # otherwise score as intrusions and pollute the intrusion count
        words_df = words_df[~words_df["is_filler"].astype(bool)]
    pool_norm = [_normalize(item) for item in wordpool]
    rows = []
    recalled: set[int] = set()
    prev_response_onset = None
    for _, w in words_df.iterrows():
        spoken = _normalize(str(w["word"]))
        if spoken in pool_norm:
            idx, score = pool_norm.index(spoken), 1.0
        else:
            scores = [SequenceMatcher(None, spoken, p).ratio() for p in pool_norm]
            idx = int(np.argmax(scores))
            score = scores[idx]
        matched = score >= match_threshold
        repetition = matched and idx in recalled
        irt = np.nan
        if matched:
            if prev_response_onset is not None:
                irt = w["onset"] - prev_response_onset
            prev_response_onset = w["onset"]
            recalled.add(idx)
        rows.append({
            "word": w["word"],
            "onset": w["onset"],
            "offset": w["offset"],
            "matched_item": wordpool[idx] if matched else "",
            "pool_index": idx if matched else -1,
            "match_score": round(score, 3),
            "intrusion": not matched,
            "repetition": repetition,
            "irt": irt,
        })
    return pd.DataFrame(rows, columns=RECALL_COLUMNS)


def recall_summary(recall_df: pd.DataFrame, wordpool: list[str]) -> dict:
    matched = recall_df[~recall_df["intrusion"]]
    unique_items = matched.loc[~matched["repetition"], "pool_index"].nunique()
    irts = matched["irt"].dropna()
    return {
        "n_pool_items": len(wordpool),
        "n_recalled_unique": int(unique_items),
        "n_intrusions": int(recall_df["intrusion"].sum()),
        "n_repetitions": int(recall_df["repetition"].sum()),
        "match_threshold": MATCH_THRESHOLD,
        "mean_irt_sec": round(float(irts.mean()), 3) if len(irts) else None,
        "median_irt_sec": round(float(irts.median()), 3) if len(irts) else None,
    }


def speech_timing(words_df: pd.DataFrame, duration: float | None) -> dict:
    """Pause / speech-rate summary from word timestamps (any transcribe run)."""
    if len(words_df) == 0:
        return {"n_words": 0}
    onsets = words_df["onset"].to_numpy(dtype=float)
    offsets = words_df["offset"].to_numpy(dtype=float)
    gaps = onsets[1:] - offsets[:-1]
    pauses = gaps[gaps >= PAUSE_MIN_SEC]
    span = offsets[-1] - onsets[0]
    return {
        "n_words": int(len(words_df)),
        "latency_first_word_sec": round(float(onsets[0]), 3),
        "speech_span_sec": round(float(span), 3),
        "speech_rate_wps": round(float(len(words_df) / span), 3) if span > 0 else None,
        "pause_min_sec": PAUSE_MIN_SEC,
        "n_pauses": int(len(pauses)),
        "mean_pause_sec": round(float(pauses.mean()), 3) if len(pauses) else None,
        "max_pause_sec": round(float(pauses.max()), 3) if len(pauses) else None,
        "total_pause_sec": round(float(pauses.sum()), 3) if len(pauses) else None,
    }
