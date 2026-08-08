"""CLAP audio embeddings — the v0.2 flagship shared audio–text space.

512-d L2-normalized embeddings from LAION-CLAP (default checkpoint
`laion/larger_clap_music_and_speech`, matching the film-clip domain of
music + dialogue + sound design). The paired text encoder lives in
word2psy as `clap_text` (same checkpoint), the way viz2psy's `clip` pairs
with word2psy's `clip_text`; psyquilt's COMPATIBLE_SPACES declares the
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


class ClapModel(BaseModel):
    name = "clap"
    level = "frame"
    input_sr = 48000

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
        and the DEAM training script."""
        import torch
        from tqdm import tqdm

        half = WINDOW_SEC / 2
        duration = len(y) / sr
        n_win = int(min(len(y), WINDOW_SEC * sr))
        starts = []
        for center in centers:
            start = min(max(center - half, 0.0), max(duration - WINDOW_SEC, 0.0))
            starts.append(int(start * sr))
        windows = [y[s : s + n_win] for s in starts]

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
