"""Speaker diarization turn table (pyannote community-1).

Event-level: one row per speaker turn — the beats/transcript onset-offset
pattern applied to "who speaks when". Two timelines come back from the
pipeline: the raw one (turns may overlap; this is what lands in
{stem}_speakers.csv) and the exclusive one (exactly one speaker owns each
instant), which is what speaker assignment onto Whisper segments/words
uses. The merge helpers are pure pandas so they stay offline-testable.

pyannote.audio is an optional extra (`pip install "aud2psy[diarize]"`): it
pulls the lightning stack, and the community-1 weights are HF-gated (a
one-time license acceptance at the model page plus a HuggingFace token via
`huggingface-cli login` or HF_TOKEN), so the model can never work straight
out of pip anyway. CPU inference by default — MPS support in pyannote
pipelines is not yet trustworthy for this checkpoint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..exceptions import Aud2PsyError
from .base import BaseModel

TURN_COLUMNS = ["turn_idx", "speaker", "onset", "offset"]
DEFAULT_PIPELINE = "pyannote/speaker-diarization-community-1"

GATED_HELP = (
    f"could not load {DEFAULT_PIPELINE} — the weights are gated. One-time "
    "setup: accept the conditions at "
    f"https://huggingface.co/{DEFAULT_PIPELINE}, then authenticate with "
    "`huggingface-cli login` or set HF_TOKEN"
)


class DiarizeModel(BaseModel):
    name = "diarize"
    level = "events"
    checkpoint = DEFAULT_PIPELINE

    def __init__(self, num_speakers: int | None = None):
        self.num_speakers = num_speakers

    def load(self) -> None:
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise ImportError(
                "The diarize model needs the optional pyannote.audio dependency: "
                'pip install "aud2psy[diarize]"'
            ) from exc
        try:
            # huggingface_hub resolves HF_TOKEN / the stored login on its own
            self.model = Pipeline.from_pretrained(DEFAULT_PIPELINE)
        except ImportError:
            raise
        except Exception as exc:
            raise Aud2PsyError(f"{GATED_HELP} (underlying error: {exc})") from exc
        if self.model is None:  # pyannote returns None on some auth failures
            raise Aud2PsyError(GATED_HELP)

    def diarize(self, y: np.ndarray, sr: int):
        """Diarize 16 kHz mono audio.

        Returns (turns_df, exclusive_df, info): the raw turn table, the
        exclusive (non-overlapping) timeline, and sidecar info.
        """
        if sr != 16000:
            raise ValueError(f"diarize expects 16 kHz audio, got {sr}")
        import torch

        waveform = torch.from_numpy(np.ascontiguousarray(y)).float().unsqueeze(0)
        kwargs = {"num_speakers": self.num_speakers} if self.num_speakers else {}
        output = self.model({"waveform": waveform, "sample_rate": sr}, **kwargs)
        annotation = getattr(output, "speaker_diarization", output)
        exclusive = getattr(output, "exclusive_speaker_diarization", None)
        turns_df = annotation_to_turns(annotation)
        exclusive_df = annotation_to_turns(exclusive) if exclusive is not None else turns_df
        labels = sorted(annotation.labels())
        info = {
            "pipeline": DEFAULT_PIPELINE,
            "n_speakers": len(labels),
            "n_turns": len(turns_df),
            "speaker_time_sec": {
                lab: round(float(annotation.label_duration(lab)), 3) for lab in labels
            },
        }
        if self.num_speakers:
            info["num_speakers_hint"] = self.num_speakers
        return turns_df, exclusive_df, info


def annotation_to_turns(annotation) -> pd.DataFrame:
    """Flatten a pyannote Annotation into the turn table, sorted by onset."""
    rows = [
        {"speaker": label, "onset": float(seg.start), "offset": float(seg.end)}
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    rows.sort(key=lambda r: (r["onset"], r["offset"]))
    df = pd.DataFrame(rows, columns=TURN_COLUMNS[1:])
    df.insert(0, "turn_idx", range(len(df)))
    return df


def merge_speakers(
    transcript_df: pd.DataFrame,
    words_df: pd.DataFrame | None,
    exclusive_df: pd.DataFrame,
) -> None:
    """Add a `speaker` column to the transcript and words tables in place.

    Transcript segments get the majority-overlap speaker against the
    exclusive timeline (Whisper segments can straddle turn boundaries);
    words get the speaker owning their midpoint. No overlap -> "".
    """
    transcript_df["speaker"] = [
        _majority_speaker(seg["onset"], seg["offset"], exclusive_df)
        for _, seg in transcript_df.iterrows()
    ]
    if words_df is not None:
        words_df["speaker"] = [
            _speaker_at((w["onset"] + w["offset"]) / 2, exclusive_df)
            for _, w in words_df.iterrows()
        ]


def _majority_speaker(onset: float, offset: float, turns_df: pd.DataFrame) -> str:
    overlap = np.minimum(offset, turns_df["offset"]) - np.maximum(onset, turns_df["onset"])
    positive = overlap > 0
    if not positive.any():
        return ""
    per_speaker = pd.Series(overlap[positive].values).groupby(
        turns_df.loc[positive, "speaker"].values
    ).sum()
    return str(per_speaker.idxmax())


def _speaker_at(t: float, turns_df: pd.DataFrame) -> str:
    inside = (turns_df["onset"] <= t) & (t < turns_df["offset"])
    if not inside.any():
        return ""
    return str(turns_df.loc[inside, "speaker"].iloc[0])
