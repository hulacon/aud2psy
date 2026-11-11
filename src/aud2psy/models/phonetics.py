"""Articulatory (phonetic) features of speech, from the audio alone.

A frozen phone recognizer (``facebook/wav2vec2-lv-60-espeak-cv-ft``, Xu, Baevski
& Auli 2021, arXiv 2109.11680: wav2vec 2.0 pretrained on unlabelled LibriLight
audio, fine-tuned with CTC on CommonVoice espeak phone labels; Apache-2.0) gives
a posterior over 392 phone tokens every 20 ms. A fixed lookup table
(``phonetics_table``; shipped as ``data/phonetics_phone_table.tsv``) maps each
token to manner, place, voicing and vowel quality, and each 0.5 s grid window
reports the posterior-weighted shares:

- ``phonetics_<manner>``: stop, fricative, affricate, nasal, liquid, glide and
  vowel shares of the window's phone mass (they sum to 1 over mapped mass);
- ``phonetics_<place>``: labial ... glottal shares of its consonant mass;
- ``phonetics_voiced``: the voiced share of its consonant mass;
- ``phonetics_vowel_height`` (0 open .. 1 close), ``phonetics_vowel_backness``
  (0 front .. 1 back), ``phonetics_vowel_rounded``: means over its vowel mass;
- ``phonetics_rate``: phones per second, counted from the greedy CTC decode;
- ``phonetics_entropy``: the recognizer's uncertainty (bits) over phones,
  averaged over the window's frames weighted by their phone mass.

"Phone mass" is the summed non-blank posterior: CTC spends most frames on the
blank, and a frame's non-blank probability is how much phone it carries.

**No transcript is used.** The features are a function of the waveform and the
checkpoint only, so a relation between them and transcript-derived text
features is not inherited from a shared text input. The recognizer is trained
on transcribed speech, so it encodes the phonetic content of the words by
construction; that is the feature's purpose.

Windows are speech-gated (``aud2psy.speech_gate``, the grid window itself, as
for egemaps): outside speech every column is NaN. Inside speech, a window with
less than ``MIN_MASS`` frame-equivalents of phone mass (a pause) has NaN
shares and rate 0; place / voicing need that much consonant mass and the
vowel columns that much vowel mass.

The recognizer runs over ``CHUNK_SEC`` chunks with ``MARGIN_SEC`` of context on
each side, keeping each chunk's central frames; chunks whose windows are all
gated are skipped. That context lives inside the upstream model, so each
column's window is the grid window (Contract B window labels).
"""

from __future__ import annotations

import json
from importlib import resources

import numpy as np

from ..speech_gate import DEFAULT_SPEECH_GATE, SpeechGated, check_gate
from .base import BaseModel, auto_device, undefined
from .phonetics_table import BLANK, MANNERS, PLACES, TABLE_VERSION, VOWEL_DIMS

CHECKPOINT = "facebook/wav2vec2-lv-60-espeak-cv-ft"
REVISION = "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"
TABLE_FILE = "phonetics_phone_table.tsv"
SR = 16000
HOP_SAMPLES = 320  # wav2vec2 conv stride product: one frame per 20 ms
FIELD_SAMPLES = 400  # receptive field of one frame; its centre is at +200
CHUNK_SEC = 20.0
MARGIN_SEC = 2.0
MIN_MASS = 1.0  # frame-equivalents of posterior mass a share needs

MANNER_COLS = [f"phonetics_{m}" for m in MANNERS]
PLACE_COLS = [f"phonetics_{p}" for p in PLACES]
VOWEL_COLS = ["phonetics_vowel_height", "phonetics_vowel_backness", "phonetics_vowel_rounded"]
COLUMNS = (MANNER_COLS + PLACE_COLS + ["phonetics_voiced"] + VOWEL_COLS
           + ["phonetics_rate", "phonetics_entropy"])

_GATE = ("the grid window is not speech under the run's speech gate (sidecar `speech_gate`)")
_PAUSE = (f"or the window holds less than {MIN_MASS:g} frame-equivalent of {{what}} "
          "posterior mass (a pause, or no {what2} in it)")


def _nulls() -> dict[str, dict[str, str]]:
    out = {}
    for c in MANNER_COLS + ["phonetics_entropy"]:
        out[c] = undefined(f"{_GATE}, " + _PAUSE.format(what="phone", what2="phone"),
                           trailing=True)
    for c in PLACE_COLS + ["phonetics_voiced"]:
        out[c] = undefined(f"{_GATE}, " + _PAUSE.format(what="consonant", what2="consonant"),
                           trailing=True)
    for c in VOWEL_COLS:
        out[c] = undefined(f"{_GATE}, " + _PAUSE.format(what="vowel", what2="vowel"),
                           trailing=True)
    out["phonetics_rate"] = undefined(_GATE, trailing=True)
    return out


def _labels() -> dict:
    cols = {}
    for m in MANNERS:
        cols[f"phonetics_{m}"] = {
            "LongName": f"{m.capitalize()} share",
            "Description": f"Share of the window's phone posterior mass on {m} segments",
            "Units": "proportion"}
    for p in PLACES:
        cols[f"phonetics_{p}"] = {
            "LongName": f"{p.capitalize()} place share",
            "Description": f"Share of the window's consonant posterior mass at {p} place",
            "Units": "proportion"}
    cols["phonetics_voiced"] = {
        "LongName": "Voiced consonant share",
        "Description": "Share of the window's consonant posterior mass that is voiced",
        "Units": "proportion"}
    cols["phonetics_vowel_height"] = {
        "LongName": "Vowel height",
        "Description": "Posterior-weighted vowel height, 0 open to 1 close"}
    cols["phonetics_vowel_backness"] = {
        "LongName": "Vowel backness",
        "Description": "Posterior-weighted vowel backness, 0 front to 1 back"}
    cols["phonetics_vowel_rounded"] = {
        "LongName": "Rounded vowel share",
        "Description": "Share of the window's vowel posterior mass on rounded vowels",
        "Units": "proportion"}
    cols["phonetics_rate"] = {
        "LongName": "Phone rate",
        "Description": "Phones per second in the window, counted from the greedy CTC decode",
        "Units": "1/s"}
    cols["phonetics_entropy"] = {
        "LongName": "Phone recognition entropy",
        "Description": "Entropy of the recognizer's phone posterior (blank excluded), "
                       "averaged over the window's frames weighted by phone mass",
        "Units": "bit"}
    return {
        "default": {"interpretability": "nameable", "derivation": "formula-on-model",
                    "derived_from": CHECKPOINT, "window_sec": "grid"},
        "columns": cols,
    }


def load_table():
    """The shipped lookup table, in vocabulary order."""
    import pandas as pd

    path = resources.files("aud2psy").joinpath("data", TABLE_FILE)
    with path.open("r", encoding="utf-8") as fh:
        return pd.read_csv(fh, sep="\t", keep_default_na=False, dtype={"token": str})


def table_matrices(table) -> dict[str, np.ndarray]:
    """Per-token weight vectors (length = vocabulary) used by ``window_features``."""
    mats = {c: table[c].to_numpy(dtype=float)
            for c in MANNERS + PLACES + VOWEL_DIMS + ("voiced", "consonant", "unmapped")}
    mats["mapped"] = 1.0 - mats["unmapped"]
    mats["blank"] = (table["token"] == BLANK).to_numpy()
    return mats


def frame_centres(start_sample: int, n_frames: int) -> np.ndarray:
    """Centre (s) of each 20 ms frame of a segment starting at ``start_sample``."""
    return (start_sample + np.arange(n_frames) * HOP_SAMPLES + FIELD_SAMPLES / 2) / SR


def frame_window_index(centres: np.ndarray, grid) -> np.ndarray:
    """Grid window of each frame, by its centre (s); -1 outside the grid."""
    idx = np.floor(np.asarray(centres) / grid.hop).astype(int)
    idx[(idx < 0) | (idx >= grid.n_windows)] = -1
    return idx


def window_features(probs: np.ndarray, win: np.ndarray, n_windows: int, hop: float,
                    mats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reduce frame posteriors (n_frames x vocab) to the per-window columns.

    ``win`` gives each frame's grid window (-1 = none). Frames of skipped
    (gated) chunks are absent. A window no frame reaches stays NaN in every
    column.
    """
    keep = win >= 0
    probs, win = probs[keep], win[keep]
    blank = mats["blank"]
    phone = np.where(blank, 0.0, 1.0)

    def wsum(values: np.ndarray) -> np.ndarray:
        out = np.zeros(n_windows)
        np.add.at(out, win, values)
        return out

    n_frames = wsum(np.ones(len(win)))
    token_mass = probs * phone  # non-blank posterior per token
    mapped = wsum(token_mass @ mats["mapped"])
    consonant = wsum(token_mass @ mats["consonant"])
    vowel = wsum(token_mass @ mats["vowel"])

    def share(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        out = np.full(n_windows, np.nan)
        ok = den >= MIN_MASS
        out[ok] = num[ok] / den[ok]
        return out

    cols: dict[str, np.ndarray] = {}
    for m in MANNERS:
        cols[f"phonetics_{m}"] = share(wsum(token_mass @ mats[m]), mapped)
    for p in PLACES:
        cols[f"phonetics_{p}"] = share(wsum(token_mass @ mats[p]), consonant)
    cols["phonetics_voiced"] = share(wsum(token_mass @ mats["voiced"]), consonant)
    for d, c in zip(VOWEL_DIMS, VOWEL_COLS):
        cols[c] = share(wsum(token_mass @ mats[d]), vowel)

    # greedy CTC: a phone starts where the argmax is non-blank and differs
    # from the previous frame's argmax (CTC's collapse rule)
    best = probs.argmax(axis=1)
    prev = np.r_[-1, best[:-1]]
    onset = (~blank[best]) & (best != prev)
    rate = wsum(onset.astype(float)) / hop
    rate[n_frames == 0] = np.nan
    cols["phonetics_rate"] = rate

    nb = token_mass.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = token_mass / nb[:, None]
        h = -np.where(q > 0, q * np.log2(q), 0.0).sum(axis=1)
    h[nb <= 0] = 0.0
    cols["phonetics_entropy"] = share(wsum(nb * h), wsum(nb))
    for c in COLUMNS:
        cols[c][n_frames == 0] = np.nan
    return cols


class PhoneticsModel(SpeechGated, BaseModel):
    name = "phonetics"
    level = "frame"
    checkpoint = CHECKPOINT
    input_sr = SR
    nulls = _nulls()
    labels = _labels()

    def __init__(self, device: str | None = None, speech_gate: str = DEFAULT_SPEECH_GATE):
        self.device = device or auto_device()
        self.speech_gate = check_gate(speech_gate)

    def load(self) -> None:
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        # The checkpoint's tokenizer needs `phonemizer` only to encode text;
        # decoding reads vocab.json directly, so no extra dependency.
        vocab = json.load(open(hf_hub_download(CHECKPOINT, "vocab.json", revision=REVISION),
                               encoding="utf-8"))
        tokens = sorted(vocab, key=vocab.get)
        table = load_table()
        if list(table["token"]) != tokens:
            raise RuntimeError(
                f"phonetics: the shipped table {TABLE_FILE} does not match {CHECKPOINT}'s "
                "vocabulary; regenerate it with python -m aud2psy.models.phonetics_table")
        self.table_ = table
        self.mats_ = table_matrices(table)
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(CHECKPOINT, revision=REVISION)
        model = Wav2Vec2ForCTC.from_pretrained(CHECKPOINT, revision=REVISION).eval()
        try:
            self.model = model.to(self.device)
        except Exception:  # MPS op gaps etc.
            self.device = "cpu"
            self.model = model.to(self.device)
        self.load_gate()

    def unload(self) -> None:
        for attr in ("model", "extractor", "vad"):
            self.__dict__.pop(attr, None)

    def posteriors(self, y: np.ndarray) -> np.ndarray:
        """Frame posteriors (n_frames x vocab) for one 16 kHz segment."""
        import torch

        x = self.extractor(y, sampling_rate=SR, return_tensors="pt").input_values
        with torch.no_grad():
            logits = self.model(x.to(self.device)).logits[0]
        return logits.float().softmax(-1).cpu().numpy()

    def extract(self, y: np.ndarray, sr: int, grid) -> dict[str, np.ndarray]:
        if sr != SR:
            raise ValueError(f"phonetics expects {SR} Hz audio, got {sr}")
        gated = self.gate_mask(y, sr, grid.centers, grid.hop)  # fail fast, before inference
        probs, wins, n_chunks, n_run = self.chunked_posteriors(y, grid, gated)
        cols = window_features(probs, wins, grid.n_windows, grid.hop, self.mats_)
        for c in COLUMNS:
            cols[c][gated] = np.nan
        self.info_ = {
            "revision": REVISION,
            "phone_table_version": TABLE_VERSION,
            "phone_table_file": TABLE_FILE,
            "chunk_sec": CHUNK_SEC,
            "margin_sec": MARGIN_SEC,
            "min_mass": MIN_MASS,
            "n_chunks": n_chunks,
            "n_chunks_run": n_run,
            "unmapped_mass_share": self.unmapped_share_,
            **self.gate_info(gated, grid.hop),
        }
        return {c: cols[c] for c in COLUMNS}

    def chunked_posteriors(self, y: np.ndarray, grid, gated: np.ndarray):
        """Posteriors over ``CHUNK_SEC`` chunks with ``MARGIN_SEC`` context, centres kept.

        Returns (probs, window index per frame, n_chunks, n_chunks_run). A chunk
        whose grid windows are all gated is not run.
        """
        from tqdm import tqdm

        n = len(y)
        chunk, margin = int(CHUNK_SEC * SR), int(MARGIN_SEC * SR)
        starts = list(range(0, max(n, 1), chunk))
        all_p, all_w = [], []
        n_run = 0
        unmapped = total = 0.0
        for s in tqdm(starts, desc="phonetics chunks", leave=False):
            e = min(s + chunk, n)
            w0 = int(np.floor(s / SR / grid.hop))
            w1 = min(int(np.ceil(e / SR / grid.hop)), grid.n_windows)
            if w1 <= w0 or gated[w0:w1].all():
                continue
            a, b = max(s - margin, 0), min(e + margin, n)
            seg = y[a:b]
            if len(seg) < FIELD_SAMPLES:
                continue
            p = self.posteriors(seg)
            n_run += 1
            centres = frame_centres(a, len(p))
            core = (centres >= s / SR) & (centres < e / SR)
            p = p[core]
            win = frame_window_index(centres[core], grid)
            nb = p * np.where(self.mats_["blank"], 0.0, 1.0)
            total += nb.sum()
            unmapped += (nb @ self.mats_["unmapped"]).sum()
            all_p.append(p)
            all_w.append(win)
        self.unmapped_share_ = float(unmapped / total) if total > 0 else None
        if not all_p:
            V = len(self.mats_["blank"])
            return np.zeros((0, V)), np.zeros(0, dtype=int), len(starts), 0
        return np.concatenate(all_p), np.concatenate(all_w), len(starts), n_run
