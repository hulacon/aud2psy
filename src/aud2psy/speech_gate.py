"""The speech gate shared by the speech-only frame models (egemaps' voiced set, speech_emotion, phonetics).

A gated model's columns are NaN on a window that is not speech: "jitter" on a
violin line is not voice quality, and the affect model invents affect on music.
Two gates exist; the run picks one (CLI ``--speech-gate``, API ``speech_gate=``):

- ``diarize`` (default since 0.20.0): the window is speech if pyannote speaker
  turns cover at least ``GATE_MIN_SPEECH`` of it. The turns come from the
  ``diarize`` model in the same call or from an existing ``{stem}_speakers.csv``
  (``--speakers``); a gated model with neither fails rather than silently
  falling back, so one command always gates one way.
- ``silero`` (the gate before 0.20.0): the window is speech if its mean Silero
  VAD probability is at least ``GATE_MIN_SPEECH``. Silero under-calls speech
  that overlaps music or noise, the common case in film audio, so a gated
  column there drops windows a listener would call speech.

Both read the same threshold over the same window (each model's own: the grid
window for egemaps, the 4 s context window for speech_emotion), edge-clamped
the way the model clamps its audio. Only the measure differs: covered fraction
for diarize, mean probability for Silero. The gate used, its turn source and
the number of gated windows are written into the model's sidecar entry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .exceptions import Aud2PsyError

if TYPE_CHECKING:  # pandas stays out of `aud2psy --help` (the CLI imports this module)
    import pandas as pd

SPEECH_GATES = ("diarize", "silero")
DEFAULT_SPEECH_GATE = "diarize"
GATE_MIN_SPEECH = 0.25  # min covered fraction (diarize) / mean probability (silero)
GATED_MODELS = ("egemaps", "speech_emotion", "phonetics")
VAD_WINDOW = 512  # samples per Silero probability at 16 kHz

_MEASURE = {
    "diarize": "fraction of the window covered by diarize speaker turns",
    "silero": "mean Silero VAD speech probability over the window",
}


def check_gate(speech_gate: str) -> str:
    if speech_gate not in SPEECH_GATES:
        raise ValueError(
            f"unknown speech gate {speech_gate!r}; choose one of {', '.join(SPEECH_GATES)}")
    return speech_gate


def window_starts(centers: np.ndarray, window_sec: float, duration: float) -> np.ndarray:
    """Start of each window centred on ``centers``, clamped inside [0, duration]."""
    centers = np.asarray(centers, dtype=float)
    return np.minimum(np.maximum(centers - window_sec / 2, 0.0),
                      max(duration - window_sec, 0.0))


def turn_coverage(
    turns_df: pd.DataFrame, centers: np.ndarray, window_sec: float, duration: float
) -> np.ndarray:
    """Fraction of each edge-clamped window covered by at least one speaker turn.

    Exact interval arithmetic: overlapping turns (crosstalk) are merged first,
    so two speakers at once count once. In a clip shorter than ``window_sec``
    every window is the whole clip, and the fraction is taken over the clip.
    """
    starts = window_starts(centers, window_sec, duration)
    ends = np.minimum(starts + window_sec, duration)
    span = ends - starts
    covered = np.zeros(len(starts))
    for on, off in _merged(turns_df):
        covered += np.clip(np.minimum(ends, off) - np.maximum(starts, on), 0.0, None)
    out = np.zeros(len(starts))
    np.divide(covered, span, out=out, where=span > 0)
    return out


def _merged(turns_df: pd.DataFrame) -> list[tuple[float, float]]:
    if turns_df is None or not len(turns_df):
        return []
    spans = sorted(zip(turns_df["onset"].astype(float), turns_df["offset"].astype(float)))
    out = [list(spans[0])]
    for on, off in spans[1:]:
        if on <= out[-1][1]:
            out[-1][1] = max(out[-1][1], off)
        else:
            out.append([on, off])
    return [(a, b) for a, b in out if b > a]


def speech_fraction(
    vad, y: np.ndarray, sr: int, centers: np.ndarray, window_sec: float
) -> np.ndarray:
    """Mean Silero speech probability over each edge-clamped window (the silero gate)."""
    if sr != 16000:
        raise ValueError(f"speech_fraction expects 16 kHz audio, got {sr}")
    y = y.astype(np.float32)
    pad = (-len(y)) % VAD_WINDOW
    if pad:
        y = np.concatenate([y, np.zeros(pad, dtype=np.float32)])
    probs = np.asarray(vad(y, num_samples=VAD_WINDOW)).ravel()
    times = (np.arange(len(probs)) + 0.5) * VAD_WINDOW / sr
    starts = window_starts(centers, window_sec, len(y) / sr)
    out = np.empty(len(starts))
    for i, start in enumerate(starts):
        m = (times >= start) & (times < start + window_sec)
        out[i] = probs[m].mean() if m.any() else 0.0
    return out


class SpeechGated:
    """Mixin for a frame model whose speech-only columns take the speech gate.

    The pipeline sets ``turns_`` (a raw diarize turn table) before ``extract``
    when the gate is ``diarize``; ``load_gate`` loads Silero only when the gate
    is ``silero``.
    """

    speech_gate: str = DEFAULT_SPEECH_GATE
    turns_: pd.DataFrame | None = None

    def load_gate(self) -> None:
        if self.speech_gate == "silero":
            from faster_whisper.vad import get_vad_model

            self.vad = get_vad_model()

    def gate_mask(self, y: np.ndarray, sr: int, centers: np.ndarray,
                  window_sec: float) -> np.ndarray:
        """True where the window is NOT speech under this model's gate."""
        if self.speech_gate == "silero":
            frac = speech_fraction(self.vad, y, sr, centers, window_sec)
        else:
            if self.turns_ is None:
                raise Aud2PsyError(
                    f"{self.name} gates on diarized speech (speech gate 'diarize') "
                    "but was given no turn table: run it together with the diarize "
                    "model, pass an existing {stem}_speakers.csv (CLI: --speakers), "
                    "or choose the Silero gate (CLI: --speech-gate silero)")
            frac = turn_coverage(self.turns_, centers, window_sec, len(y) / sr)
        return frac < GATE_MIN_SPEECH

    def gate_info(self, gated: np.ndarray, window_sec: float) -> dict:
        return {
            "speech_gate": self.speech_gate,
            "speech_gate_measure": _MEASURE[self.speech_gate],
            "speech_gate_min": GATE_MIN_SPEECH,
            "speech_gate_window_sec": float(window_sec),
            "n_gated_windows": int(np.asarray(gated).sum()),
        }
