"""Musical valence/arousal: a DEAM-trained linear probe on CLAP embeddings.

Frame-level `music_emotion_valence` / `music_emotion_arousal` in [-1, 1],
predicted by ridge regressors fit on the DEAM dataset's dynamic 2 Hz
annotations (which natively match the default grid). The fitted probe
ships as package data (`data/music_emotion_probe.npz`, ~4 KB) with a
provenance JSON recording checkpoint, training config, and song-level CV
numbers; `scripts/train_deam_probe.py` reproduces it.

The probe is trained on music. On speech/foley it still produces values
(CLAP embeds everything), but interpret them as "how the soundtrack would
feel if heard as music" — it is a musical-affect estimate, not a general
one. When `clap` (or `sound_events`) runs in the same pipeline call, the
window embeddings are computed once and shared via the pipeline's
per-run cache (see ``ClapModel.embed_windows``).
"""

from __future__ import annotations

import json
from importlib import resources

import numpy as np

from .clap import ClapModel

PROBE_RESOURCE = "music_emotion_probe.npz"
PROVENANCE_RESOURCE = "music_emotion_probe.json"
DIMENSIONS = ["valence", "arousal"]


def load_probe():
    """Returns (coef (2, 512), intercept (2,), provenance dict)."""
    data_dir = resources.files("aud2psy") / "data"
    with resources.as_file(data_dir / PROBE_RESOURCE) as path:
        probe = np.load(path)
        coef, intercept = probe["coef"], probe["intercept"]
    provenance = json.loads((data_dir / PROVENANCE_RESOURCE).read_text())
    return coef, intercept, provenance


class MusicEmotionModel(ClapModel):
    name = "music_emotion"
    level = "frame"

    def load(self) -> None:
        super().load()
        self.coef, self.intercept, self.provenance = load_probe()

    def extract(self, y: np.ndarray, sr: int, grid) -> dict[str, np.ndarray]:
        emb = self.embed_windows(y, sr, grid.centers)
        pred = np.clip(emb @ self.coef.T + self.intercept, -1.0, 1.0)
        return {f"music_emotion_{dim}": pred[:, i] for i, dim in enumerate(DIMENSIONS)}
