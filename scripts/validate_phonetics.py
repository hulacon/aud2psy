"""Validate the phonetics recognizer against a forced alignment (LibriSpeech-style).

Needs the recognizer checkpoint (~1.3 GB on first use), a LibriSpeech split
(``<root>/<speaker>/<chapter>/<utt>.flac``) and a phone-level forced alignment
of it as long-format Praat TextGrids with a ``phones`` tier in ARPAbet (the
Montreal Forced Aligner layout, ``<alignments>/<speaker>/<chapter>/<utt>.TextGrid``;
silence labelled ``sil`` / ``sp`` / ``spn`` / empty):

    python scripts/validate_phonetics.py --audio-root LibriSpeech/test-clean \\
        --alignments alignments/test-clean --out-dir v0_test-clean [--limit 200]

Writes ``segments.parquet`` (one row per reference phone: reference and
predicted manner / place / voicing), ``frames.parquet`` (frame-level
reference vs prediction), ``summary.json`` (balanced accuracy per feature at
both levels, notation-sensitive phone error rate, throughput) and prints the
summary.

Scoring:

- **segment level (primary):** the recognizer's phone mass (non-blank
  posterior) summed over the frames whose centres fall inside each reference
  phone interval; the prediction is the manner with most mass, the place with
  most consonant mass, and voiced when voiced mass is at least half the
  consonant mass. A segment with less than ``--min-mass`` phone mass counts as
  a miss (wrong on every feature). CTC emits phones as spikes on a mostly-blank
  frame sequence, so this asks "what did the recognizer hear during this
  phone", not "when did it spike".
- **frame level (reported):** each frame inside a reference phone, predicted
  from its blank-excluded posterior.
- Balanced accuracy is the mean of the true-positive and true-negative rates,
  one feature value against the rest (each manner; voicing on consonants).
- The phone error rate compares greedy-decoded segments with the reference
  mapped to IPA. espeak and ARPAbet transcribe some vowels differently (ɐ / ʌ /
  ə for AH), so it is descriptive only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aud2psy.models.phonetics import (  # noqa: E402
    SR,
    PhoneticsModel,
    frame_centres,
)
from aud2psy.models.phonetics_table import (  # noqa: E402
    MANNERS,
    PLACES,
    _features,
    segments,
)

ARPABET_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "B": "b",
    "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɚ", "EY": "eɪ", "F": "f",
    "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i", "JH": "dʒ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ",
    "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}
SILENCE = {"", "sil", "sp", "spn", "<eps>"}


def read_phones(textgrid: Path) -> list[tuple[float, float, str]]:
    """(xmin, xmax, label) intervals of the ``phones`` tier of a long-format TextGrid."""
    text = textgrid.read_text(encoding="utf-8", errors="replace")
    tiers = re.split(r'\n\s*item \[\d+\]:', text)
    for tier in tiers:
        if re.search(r'name = "phones"', tier):
            return [(float(a), float(b), lab.strip()) for a, b, lab in re.findall(
                r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"', tier)]
    raise ValueError(f"{textgrid}: no 'phones' tier")


def reference(label: str):
    """(ipa, manner, place, voiced) of an ARPAbet label; None for silence/unknown."""
    base = re.sub(r"\d", "", label).upper()
    if label.lower() in SILENCE or base not in ARPABET_IPA:
        return None
    ipa = ARPABET_IPA[base]
    if base == "AH" and label.endswith("0"):
        ipa = "ə"  # unstressed AH is schwa
    seg = segments(ipa)[0]
    manner, places, voiced, _ = _features(seg)
    place = max(places, key=places.get) if places else None
    return ipa, manner, place, voiced


def predict(mass_by_token: np.ndarray, mats) -> tuple:
    """(manner, place, voiced, phone mass) predicted from summed token mass."""
    total = float(mass_by_token @ mats["mapped"])
    manner = MANNERS[int(np.argmax([mass_by_token @ mats[m] for m in MANNERS]))]
    cons = float(mass_by_token @ mats["consonant"])
    place = PLACES[int(np.argmax([mass_by_token @ mats[p] for p in PLACES]))] if cons > 0 else None
    voiced = int(float(mass_by_token @ mats["voiced"]) >= 0.5 * cons) if cons > 0 else None
    return manner, place, voiced, total


def balanced_accuracy(ref: np.ndarray, pred: np.ndarray, value) -> float | None:
    pos, neg = ref == value, ref != value
    if not pos.any() or not neg.any():
        return None
    return float(((pred[pos] == value).mean() + (pred[neg] != value).mean()) / 2)


def edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def main(argv=None) -> int:
    import pandas as pd
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audio-root", required=True, type=Path)
    ap.add_argument("--alignments", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="score at most N utterances")
    ap.add_argument("--min-mass", type=float, default=0.05)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = PhoneticsModel(device=args.device)
    model.load()
    mats, table = model.mats_, model.table_
    blank = mats["blank"]
    utts = sorted(args.audio_root.rglob("*.flac"))[: args.limit]
    seg_rows, frame_rows = [], []
    n_err = n_ref = 0
    audio_sec = infer_sec = 0.0
    missing = 0
    for k, flac in enumerate(utts):
        rel = flac.relative_to(args.audio_root).with_suffix(".TextGrid")
        tg = args.alignments / rel
        if not tg.exists():
            missing += 1
            continue
        y, sr = sf.read(flac)
        assert sr == SR, (flac, sr)
        t0 = time.time()
        p = model.posteriors(y.astype(np.float32))
        infer_sec += time.time() - t0
        audio_sec += len(y) / SR
        centres = frame_centres(0, len(p))
        mass = p * np.where(blank, 0.0, 1.0)
        refs = []
        for a, b, label in read_phones(tg):
            r = reference(label)
            if r is None:
                continue
            refs.append(r[0])
            inside = (centres >= a) & (centres < b)
            pm, pp, pv, total = predict(mass[inside].sum(axis=0), mats)
            hit = total >= args.min_mass
            seg_rows.append({"utt": flac.stem, "onset": a, "offset": b, "label": label,
                             "ref_manner": r[1], "ref_place": r[2], "ref_voiced": r[3],
                             "pred_manner": pm if hit else None,
                             "pred_place": pp if hit else None,
                             "pred_voiced": pv if hit else None, "phone_mass": total})
            for i in np.flatnonzero(inside):
                fm, fp, fv, _ = predict(mass[i], mats)
                frame_rows.append({"utt": flac.stem, "ref_manner": r[1], "ref_place": r[2],
                                   "ref_voiced": r[3], "pred_manner": fm, "pred_place": fp,
                                   "pred_voiced": fv})
        best = p.argmax(axis=1)
        prev = np.r_[-1, best[:-1]]
        hyp = []
        for t in best[(~blank[best]) & (best != prev)]:
            hyp += (segments(table["token"].iloc[t]) or [])
        ref_segs = [s for ipa in refs for s in (segments(ipa) or [])]
        n_err += edit_distance(ref_segs, hyp)
        n_ref += len(ref_segs)
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(utts)} utterances", file=sys.stderr)

    segs, frames = pd.DataFrame(seg_rows), pd.DataFrame(frame_rows)
    segs.to_parquet(args.out_dir / "segments.parquet")
    frames.to_parquet(args.out_dir / "frames.parquet")

    def scores(df):
        out = {m: balanced_accuracy(df["ref_manner"].to_numpy(),
                                    df["pred_manner"].fillna("none").to_numpy(), m)
               for m in MANNERS}
        cons = df[df["ref_manner"] != "vowel"]
        out["voiced"] = balanced_accuracy(cons["ref_voiced"].to_numpy(),
                                          cons["pred_voiced"].fillna(-1).to_numpy(), 1)
        for p_ in PLACES:
            out[f"place_{p_}"] = balanced_accuracy(cons["ref_place"].fillna("none").to_numpy(),
                                                   cons["pred_place"].fillna("none").to_numpy(),
                                                   p_)
        return out

    summary = {
        "n_utterances": len(utts) - missing, "n_missing_alignment": missing,
        "n_segments": len(segs), "n_frames": len(frames),
        "segment_miss_rate": float(segs["pred_manner"].isna().mean()) if len(segs) else None,
        "segment_balanced_accuracy": scores(segs),
        "frame_balanced_accuracy": scores(frames),
        "phone_error_rate_notation_sensitive": n_err / n_ref if n_ref else None,
        "audio_sec": audio_sec, "inference_sec": infer_sec, "device": model.device,
        "checkpoint": model.checkpoint, "min_mass": args.min_mass,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
