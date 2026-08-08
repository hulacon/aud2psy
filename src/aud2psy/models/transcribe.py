"""Transcription export (faster-whisper) — for word2psy, not a feature.

One transcript row per raw Whisper segment (no regrouping); word-level
timestamps go to a separate words table. A wordless clip is an explicit
result: zero-row DataFrames, not an error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseModel

SEGMENT_COLUMNS = ["segment_idx", "text", "onset", "offset", "asr_confidence", "no_speech_prob"]
WORD_COLUMNS = ["segment_idx", "word", "onset", "offset", "probability"]

DEFAULT_WHISPER_MODEL = "large-v3"
COMPUTE_TYPE = "int8"  # CTranslate2 on CPU; int8 is the fast path


class TranscribeModel(BaseModel):
    name = "transcribe"
    level = "segment"

    def __init__(self, whisper_model: str = DEFAULT_WHISPER_MODEL, language: str | None = None):
        self.whisper_model = whisper_model
        self.language = language

    def load(self) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(self.whisper_model, device="cpu", compute_type=COMPUTE_TYPE)

    def transcribe(self, y: np.ndarray, sr: int):
        """Transcribe 16 kHz mono audio.

        Returns (segments_df, words_df, info_dict). asr_confidence is
        exp(avg_logprob): the segment's geometric-mean token probability.
        """
        if sr != 16000:
            raise ValueError(f"transcribe expects 16 kHz audio, got {sr}")
        segments, info = self.model.transcribe(
            y,
            language=self.language,
            word_timestamps=True,
            vad_filter=True,  # suppresses hallucinated text on wordless audio
        )
        seg_rows, word_rows = [], []
        for i, seg in enumerate(segments):
            seg_rows.append({
                "segment_idx": i,
                "text": seg.text.strip(),
                "onset": seg.start,
                "offset": seg.end,
                "asr_confidence": float(np.exp(seg.avg_logprob)),
                "no_speech_prob": seg.no_speech_prob,
            })
            for w in seg.words or []:
                word_rows.append({
                    "segment_idx": i,
                    "word": w.word.strip(),
                    "onset": w.start,
                    "offset": w.end,
                    "probability": w.probability,
                })
        segments_df = pd.DataFrame(seg_rows, columns=SEGMENT_COLUMNS)
        words_df = pd.DataFrame(word_rows, columns=WORD_COLUMNS)
        info_dict = {
            "language": info.language,
            "language_probability": round(float(info.language_probability), 4),
            "n_speech_segments": len(segments_df),
        }
        return segments_df, words_df, info_dict
