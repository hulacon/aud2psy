"""Transcription export (faster-whisper) — for word2psy, not a feature.

One transcript row per raw Whisper segment (no regrouping); word-level
timestamps go to a separate words table. A wordless clip is an explicit
result: zero-row DataFrames, not an error.

``verbatim=True`` swaps the checkpoint for the official CTranslate2
conversion of CrisperWhisper v1 (a large-v3 fine-tune; CC-BY-NC-4.0,
**English/German only**): vanilla Whisper silently deletes fillers,
repetitions, and false starts, CrisperWhisper keeps them, with fillers as
bracketed tags (``[UH]``, ``[UM]``) that get an ``is_filler`` column in
the words table. Two documented compromises versus the CrisperWhisper
paper numbers: the CT2 conversion does not carry the retrained-head DTW,
so timestamp accuracy is stock-faster-whisper grade (within the README's
existing caveat), and CrisperWhisper 2.0 (multilingual, better F1) has no
faster-whisper path yet — it's the documented upgrade path once its
Mac/transformers-5 story stabilizes.

.. warning::

   **KNOWN DEFECT (open, filed 2026-08-20): the CT2 verbatim path drops
   whole ~30 s chunks silently on long recordings.** This is data loss,
   not a quality compromise like the two above, and nothing in the output
   marks it — the words table is simply missing that span, with no gap
   flag, no low ``asr_confidence``, and no error.

   Reproduced on 3/3 subjects of real free-recall speech (~70-110 min
   each): **7 / 33 / 9** chunk-sized gaps, against **0-2** for the same
   audio through the standard ``large-v3`` path. The standard arm detects
   the verbatim arm's gaps and vice versa, so a cross-arm gap diff is the
   available workaround until this is fixed. Not reproduced on short
   synthetic clips, which is why the v0.7 validation missed it.

   Until fixed, do not use ``verbatim=True`` as the sole transcript source
   for anything longer than a few minutes where completeness matters.
   Cause not yet isolated (candidates: the ``vad_filter`` disablement on
   this path, CT2 chunk-boundary handling, or the no-space tokenizer's
   token volume interacting with segment length).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .base import BaseModel

# §4.1 column convention. Two kinds of column here and the split matters:
#
#   reserved   chunk_idx / word_idx / word / onset / offset -- structural
#              position and the stimulus's own coordinates. Bare by design;
#              consumers key on them.
#   features   everything the model *measured*, prefixed with the model name
#              so it can be attributed. Unprefixed feature columns are
#              invisible to a consumer's resolver, which silently drops them
#              onto a null model and then collides them across models.
#
# `chunk_idx` rather than `segment_idx`: a transcript segment is a chunk, the
# same grouping level word2psy emits for text, and one structural vocabulary
# across extractors means a consumer needs one key rather than one per
# extractor.
SEGMENT_COLUMNS = ["chunk_idx", "onset", "offset",
                   "transcribe_text", "transcribe_asr_confidence",
                   "transcribe_no_speech_prob"]
WORD_COLUMNS = ["chunk_idx", "word_idx", "word", "onset", "offset",
                "transcribe_probability"]

DEFAULT_WHISPER_MODEL = "large-v3"
COMPUTE_TYPE = "int8"  # CTranslate2 on CPU; int8 is the fast path

VERBATIM_CHECKPOINT = "nyrahealth/faster_CrisperWhisper"
VERBATIM_LANGUAGES = ("en", "de")  # CrisperWhisper v1's training languages
FILLER_RE = re.compile(r"^\[.+\]$")  # CrisperWhisper filler tags: [UH], [UM]


def clean_verbatim_word(raw: str) -> str:
    """Strip CrisperWhisper's leading separator artifact from a CT2 word.

    The retrained tokenizer has no space prefixes, so through CTranslate2
    every non-initial word arrives as ",word" — or ".word" after a pause.
    Leading punctuation is separator; trailing punctuation is real.
    """
    return re.sub(r"^[^\w\[]+", "", raw).strip()


MAX_WORD_RUN = 6  # longest same-word run that still counts as speech, not a loop
RETRY_REPETITION_PENALTY = 1.1


def max_word_run(words) -> int:
    """Length of the longest run of consecutive identical words."""
    best = run = 0
    prev = None
    for w in words:
        run = run + 1 if w == prev else 1
        prev = w
        best = max(best, run)
    return best


class TranscribeModel(BaseModel):
    name = "transcribe"
    level = "segment"

    def __init__(
        self,
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        language: str | None = None,
        verbatim: bool = False,
    ):
        if verbatim:
            if whisper_model != DEFAULT_WHISPER_MODEL:
                raise ValueError(
                    "--verbatim selects the CrisperWhisper checkpoint itself; "
                    "drop --whisper-model"
                )
            if language is not None and language not in VERBATIM_LANGUAGES:
                raise ValueError(
                    f"verbatim mode supports {'/'.join(VERBATIM_LANGUAGES)} only "
                    f"(CrisperWhisper training languages), got {language!r}"
                )
            whisper_model = VERBATIM_CHECKPOINT
        self.whisper_model = whisper_model
        self.checkpoint = whisper_model  # Contract B §4.1 alias of whisper_model
        self.language = language
        self.verbatim = verbatim

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
        # verbatim departures from the defaults, all validated empirically:
        # vad_filter OFF (Silero's resegmentation destabilizes
        # CrisperWhisper, whose own anti-hallucination training already
        # yields 0 segments on wordless audio — whereas vanilla Whisper
        # needs the filter for exactly that), and a degenerate-loop retry:
        # a genuine "I I" in the audio stochastically seeds a ",I"x100
        # repetition loop via temperature fallback. repetition_penalty=1.1
        # reliably kills the loop but CTranslate2 applies it across the
        # whole decode window, which suppressed a genuine distant repeat
        # ("apple ... apple" in a recall list) — so the penalty is a
        # fallback for detected-degenerate decodes, never the default.
        if not self.verbatim:
            segments_df, words_df, info = self._decode(y, vad_filter=True)
        else:
            segments_df, words_df, info = self._decode(y, vad_filter=False)
            retried = False
            if max_word_run(words_df["word"]) > MAX_WORD_RUN:
                segments_df, words_df, info = self._decode(
                    y, vad_filter=False, repetition_penalty=RETRY_REPETITION_PENALTY
                )
                retried = True
        if self.verbatim:
            words_df["is_filler"] = [
                bool(FILLER_RE.match(w)) for w in words_df["word"]
            ] if len(words_df) else []
        info_dict = {
            "language": info.language,
            "language_probability": round(float(info.language_probability), 4),
            "vad_filter": not self.verbatim,
            "n_speech_segments": len(segments_df),
        }
        if self.verbatim:
            info_dict["verbatim"] = True
            info_dict["degenerate_retry"] = retried
            if retried:
                info_dict["repetition_penalty"] = RETRY_REPETITION_PENALTY
            info_dict["n_fillers"] = int(words_df["is_filler"].sum()) if len(words_df) else 0
        return segments_df, words_df, info_dict

    def _decode(self, y: np.ndarray, **kwargs):
        """One faster-whisper pass -> (segments_df, words_df, info)."""
        segments, info = self.model.transcribe(
            y, language=self.language, word_timestamps=True, **kwargs
        )
        seg_rows, word_rows = [], []
        for seg in segments:
            words = []
            for w in seg.words or []:
                word = clean_verbatim_word(w.word) if self.verbatim else w.word.strip()
                if word:
                    words.append((word, w))
            # verbatim text is rebuilt from the cleaned words, so the
            # separator artifacts don't leak into word2psy tokenization;
            # a segment whose tokens were all separator junk is dropped
            text = " ".join(w for w, _ in words) if self.verbatim else seg.text.strip()
            if self.verbatim and not text:
                continue
            i = len(seg_rows)
            seg_rows.append({
                "chunk_idx": i,
                "onset": seg.start,
                "offset": seg.end,
                "transcribe_text": text,
                "transcribe_asr_confidence": float(np.exp(seg.avg_logprob)),
                "transcribe_no_speech_prob": seg.no_speech_prob,
            })
            # word_idx is the word's position within its segment. Whisper can
            # give two words identical start/end, so onset/offset alone do not
            # identify a word row -- 158 duplicate keys turned up in the
            # MMMData movie transcripts on exactly that.
            for j, (word, w) in enumerate(words):
                word_rows.append({
                    "chunk_idx": i,
                    "word_idx": j,
                    "word": word,
                    "onset": w.start,
                    "offset": w.end,
                    "transcribe_probability": w.probability,
                })
        segments_df = pd.DataFrame(seg_rows, columns=SEGMENT_COLUMNS)
        words_df = pd.DataFrame(word_rows, columns=WORD_COLUMNS)
        return segments_df, words_df, info
