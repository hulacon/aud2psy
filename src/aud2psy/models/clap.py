"""CLAP audio embeddings — the v0.2 flagship shared audio–text space.

512-d L2-normalized embeddings from LAION-CLAP (default checkpoint
`laion/larger_clap_music_and_speech`, matching the film-clip domain of
music + dialogue + sound design). The paired text encoder lives in
word2psy as `clap_text` (same checkpoint), the way viz2psy's `clip` pairs
with word2psy's `clip_text`; psytwill's COMPATIBLE_SPACES declares the
pairing. Do not change the checkpoint, the L2 normalization, or the
`clap_{i:03d}` naming without coordinating all three repos.

Frame-level: one embedding per grid window, computed from a 10 s context
window centered on the window's midpoint (edge-clamped) — the family's
window ≫ hop pattern. CLAP wants 48 kHz input, declared via `input_sr`
so the pipeline decodes at full bandwidth instead of upsampling 22050 Hz.
Runs on MPS when available (CPU fallback), batched.
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel, auto_device

DEFAULT_CHECKPOINT = "laion/larger_clap_music_and_speech"
WINDOW_SEC = 10.0
BATCH_SIZE = 8
EMBED_DIM = 512

# Below this context-window RMS the CLAP embedding collapses toward the
# digital-silence point instead of describing content: measured cosine to
# the all-zeros embedding runs .64-1.0 at <= -80 dBFS, and the whole
# sound_events bank inflates there (music .294, gunshot_explosion .309 on
# np.zeros). Deliberately NOT a "quiet scene" threshold — CLAP is
# level-invariant for real content (the same melody scores music .211 at
# -60 dBFS and .200 at -20 dBFS), so gating at a normal noise floor would
# discard values the model gets right. Real recordings sit at -60 to
# -70 dBFS; this fires on muted tracks, digital black at export
# head/tail, and gaps in edited audio. Consumed by `sound_events` and
# `music_emotion`; `clap` itself stays ungated so its 512-d matrix has no
# holes for psytwill (silence is a real acoustic state to embed).
SILENCE_DBFS = -80.0


def window_starts(n_samples: int, sr: int, centers: np.ndarray) -> list[int]:
    """Sample index where each center's 10 s context window begins,
    edge-clamped to [0, duration - WINDOW_SEC]. Shared by the embedder and
    the silence gate so the two can never drift apart."""
    half = WINDOW_SEC / 2
    duration = n_samples / sr
    return [
        int(min(max(center - half, 0.0), max(duration - WINDOW_SEC, 0.0)) * sr)
        for center in centers
    ]


def silent_windows(y: np.ndarray, sr: int, centers: np.ndarray) -> np.ndarray:
    """Bool mask over ``centers``: True where the 10 s context window's RMS
    is below ``SILENCE_DBFS`` (the embedding is degenerate, not merely
    quiet). Measured on the context window, not the grid window, because
    the context is what CLAP actually embeds."""
    n_win = int(min(len(y), WINDOW_SEC * sr))
    floor = 10 ** (SILENCE_DBFS / 20)
    mask = np.zeros(len(centers), dtype=bool)
    for i, start in enumerate(window_starts(len(y), sr, centers)):
        window = y[start : start + n_win].astype(np.float64)
        rms = float(np.sqrt(np.mean(window**2))) if window.size else 0.0
        mask[i] = rms < floor
    return mask


class ClapModel(BaseModel):
    name = "clap"
    level = "frame"
    input_sr = 48000
    window_sec = WINDOW_SEC

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None):
        self.checkpoint = checkpoint
        self.device = device or auto_device()

    def load(self) -> None:
        from transformers import ClapModel as HFClapModel, ClapProcessor

        self.processor = ClapProcessor.from_pretrained(self.checkpoint)
        model = HFClapModel.from_pretrained(self.checkpoint).eval()
        try:
            self.model = model.to(self.device)
        except Exception:  # MPS op gaps etc. — fall back rather than fail
            self.device = "cpu"
            self.model = model.to(self.device)

    def unload(self) -> None:
        self.__dict__.pop("model", None)
        self.__dict__.pop("processor", None)

    def extract(self, y: np.ndarray, sr: int, grid) -> dict[str, np.ndarray]:
        out = self.embed_windows(y, sr, grid.centers)
        return {f"clap_{i:03d}": out[:, i] for i in range(EMBED_DIM)}

    def embed_windows(self, y: np.ndarray, sr: int, centers: np.ndarray) -> np.ndarray:
        """(len(centers), 512) unit-norm embeddings of 10 s windows centered
        on ``centers`` (edge-clamped). Shared by `clap`, `music_emotion`,
        `sound_events`, and the DEAM training script.

        When the pipeline attaches a per-run dict as ``cache_``, the
        computed array is reused across the clap-family models in the same
        run (keyed by checkpoint/sample rate/centers, so a ``--clap-model``
        override on `clap` correctly misses against the fixed-checkpoint
        models). In-memory and per-run only — no disk persistence."""
        cache = getattr(self, "cache_", None)
        if cache is None:
            return self._embed_windows(y, sr, centers)
        key = (self.checkpoint, sr, centers.tobytes())
        if key not in cache:
            cache[key] = self._embed_windows(y, sr, centers)
        return cache[key]

    def _embed_windows(self, y: np.ndarray, sr: int, centers: np.ndarray) -> np.ndarray:
        import torch
        from tqdm import tqdm

        n_win = int(min(len(y), WINDOW_SEC * sr))
        windows = [y[s : s + n_win] for s in window_starts(len(y), sr, centers)]

        out = np.empty((len(centers), EMBED_DIM))
        batches = range(0, len(windows), BATCH_SIZE)
        for b in tqdm(batches, desc="clap windows", leave=False):
            chunk = windows[b : b + BATCH_SIZE]
            inputs = self.processor(
                audio=chunk, sampling_rate=sr, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                result = self.model.get_audio_features(**inputs)
            # transformers >= 5 returns a ModelOutput; the 512-d projection
            # is pooler_output (older versions returned the tensor directly)
            emb = result.pooler_output if hasattr(result, "pooler_output") else result
            emb = emb / emb.norm(dim=-1, keepdim=True)
            out[b : b + len(chunk)] = emb.cpu().numpy()
        return out
