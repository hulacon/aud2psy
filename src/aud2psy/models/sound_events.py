"""Zero-shot sound-event tags: a text prompt bank scored against CLAP.

AudioSet-style scene/event regressors (laughter, music presence, crowd,
sirens, ...) — the last evidenced registry gap from the Aug 2026 survey
(CANLab scoping review, Giordano et al. 2023) — without new weights: each
category is a small ensemble of text prompts embedded once by the SAME
LAION-CLAP checkpoint the `clap` model uses (the `music_emotion`
subclass pattern — one weight load, prompts embedded at ``load()``), and
every 10 s audio window is scored by cosine similarity against each
category's ensemble embedding.

Scoring is **raw cosine, not softmax over the bank**, deliberately:
softmax would couple every column to the bank's composition (pruning one
category after validation would silently change all others), would force
co-occurring events to compete (dialogue over score music is the norm in
film), and CLAP's trained logit scale makes it near-one-hot, destroying
the graded time course psyquilt correlates. The cost, documented here
and in the sidecar: per-column baselines differ (some prompts sit closer
to everything), so values compare *within a column* over time and
between clips; cross-column comparison is ordinal at best. Prompt
ensembling: each prompt is embedded, the category mean is re-normalized
(the CLIP prompt-ensemble recipe), so category vectors are unit-norm and
scores are true cosines in [-1, 1].

Deliberately absent from the bank: speaker-attribute categories — CLAP
zero-shot ran AT CHANCE (52%) for per-window speaker gender (recorded
negative finding, v0.3.1 era); silence (the loudness model's job); and
affect prompts (speech_emotion / music_emotion's job).
``sound_events_speech`` is retained even though Silero `speech_prob` is
the better speech detector — it doubles as a validation cross-check and
keeps the bank's event vocabulary complete.

No NaN/gating: cosine to a unit vector is defined everywhere, silence
included. Checkpoint is fixed (not coupled to ``--clap-model``):
per-category validation is checkpoint-specific, the music_emotion
stance. Categories that fail per-category validation on real stimuli
should simply be deleted from ``PROMPT_BANK`` — with raw-cosine scoring
that changes nothing else.
"""

from __future__ import annotations

import numpy as np

from .clap import ClapModel

# category -> prompt ensemble. Keys become columns: sound_events_<key>.
PROMPT_BANK: dict[str, list[str]] = {
    "speech": ["a person speaking", "people having a conversation", "someone talking"],
    "music": ["music playing", "an instrumental piece of music", "background music"],
    "singing": ["a person singing", "a singing voice", "a vocal melody"],
    "laughter": ["a person laughing", "laughter", "people laughing together"],
    "crying": ["a person crying", "sobbing and weeping", "a baby crying"],
    "shouting": ["a person shouting", "someone yelling angrily", "a person screaming"],
    "crowd": [
        "a crowd of people",
        "the murmur of a busy crowd",
        "many people talking at once",
    ],
    "applause": [
        "an audience applauding",
        "clapping and applause",
        "people clapping their hands",
    ],
    "footsteps": [
        "footsteps",
        "the sound of someone walking",
        "footsteps on a hard floor",
    ],
    "vehicle": ["a car engine running", "traffic noise", "a vehicle driving past"],
    "water": ["rain falling", "water flowing", "a running stream of water"],
    "wind": ["wind blowing", "wind howling", "a gust of wind"],
    "animals": ["a dog barking", "birds chirping", "animal sounds"],
    "gunshot_explosion": ["a gunshot", "gunfire", "an explosion"],
    "siren_alarm": ["a siren wailing", "an emergency vehicle siren", "an alarm ringing"],
    "thunder": ["thunder rumbling", "a thunderstorm", "a thunder clap"],
}


class SoundEventsModel(ClapModel):
    name = "sound_events"
    level = "frame"

    def load(self) -> None:
        super().load()
        self.bank_ = self._embed_bank()

    def _embed_bank(self) -> np.ndarray:
        """(n_categories, 512) unit-norm prompt-ensemble embeddings."""
        import torch

        prompts = [p for ensemble in PROMPT_BANK.values() for p in ensemble]
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(
            self.device
        )
        with torch.no_grad():
            result = self.model.get_text_features(**inputs)
        # transformers >= 5 returns a ModelOutput; the projected embedding
        # is pooler_output (mirroring the audio side in clap.py)
        emb = result.pooler_output if hasattr(result, "pooler_output") else result
        emb = (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy()
        return ensemble_bank(emb, [len(v) for v in PROMPT_BANK.values()])

    def extract(self, y: np.ndarray, sr: int, grid) -> dict[str, np.ndarray]:
        audio_emb = self.embed_windows(y, sr, grid.centers)
        scores = audio_emb @ self.bank_.T
        self.info_ = {
            "checkpoint": self.checkpoint,
            "scoring": "cosine vs unit-norm prompt-ensemble embedding "
            "(compare within a column; cross-column is ordinal at best)",
            "prompts": PROMPT_BANK,
        }
        return {
            f"sound_events_{cat}": scores[:, j] for j, cat in enumerate(PROMPT_BANK)
        }


def ensemble_bank(prompt_emb: np.ndarray, sizes: list[int]) -> np.ndarray:
    """Mean per-category ensemble of unit-norm prompt embeddings,
    re-normalized (the CLIP prompt-ensemble recipe)."""
    bank = []
    start = 0
    for size in sizes:
        mean = prompt_emb[start : start + size].mean(axis=0)
        bank.append(mean / np.linalg.norm(mean))
        start += size
    return np.stack(bank)
