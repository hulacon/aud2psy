"""EBind audio arm — 1024-d embeddings in the cross-modal shared space.

Uses the same revision-pinned checkpoint as viz2psy's ``ebind`` (image arm)
and word2psy's ``ebind_text``, so soundtracks, images, and text live in one
1024-d space; psytwill's COMPATIBLE_SPACES declares the pairings. The
encoder is EBind's ImageBind-huge audio trunk, whose output is projected
into the Perception Encoder space by EBind's trained MLP projector. Do not
change the checkpoint, the L2 normalization, or the
``ebind_audio_{i:04d}`` naming without coordinating all three repos.

Scope: naturalistic soundtracks. The AudioSet-trained encoder hears
isolated spoken words as generic speech (no lexical signal — 2026-08-17
mmmdata pilot); word stimuli enter the shared space via word2psy's
``ebind_text`` instead.

Frame-level: one embedding per grid window, computed from a 2 s context
window centered on the window's midpoint (edge-clamped). 2 s is
ImageBind's native training clip length — deliberately tighter than
CLAP's 10 s context, trading scene context for temporal precision.
Preprocessing reuses ebind's own ``waveform2melspec`` and the
``IBAudioProcessor`` constants (16 kHz, 128 mel bins, 204 frames,
mean/std normalization) so our windows match what the trunk was trained
on. Embedding indices are fixed-width 4-digit
(``ebind_audio_0000`` .. ``ebind_audio_1023``) — a >999-d space per
contracts §4.1.
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel, auto_device

DEFAULT_CHECKPOINT = "encord-team/ebind-full"
WINDOW_SEC = 2.0
BATCH_SIZE = 32
EMBED_DIM = 1024
# EBind's config requires image+video+text at minimum; audio adds the
# ImageBind trunk + projector. Points (Uni3D) stays excluded.
_MODALITIES = ["image", "video", "text", "audio"]


def window_starts(n_samples: int, sr: int, centers: np.ndarray) -> list[int]:
    """Sample index where each center's 2 s context window begins,
    edge-clamped to [0, duration - WINDOW_SEC] (the clap pattern)."""
    half = WINDOW_SEC / 2
    duration = n_samples / sr
    return [
        int(min(max(center - half, 0.0), max(duration - WINDOW_SEC, 0.0)) * sr)
        for center in centers
    ]


class EBindAudioModel(BaseModel):
    name = "ebind_audio"
    level = "frame"
    input_sr = 16000  # ImageBind's audio sample rate
    window_sec = WINDOW_SEC

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None):
        self.checkpoint = checkpoint
        self.device = device or auto_device()

    def load(self) -> None:
        from ebind import EBindModel as _EBind
        from ebind.configuration import EBindConfig
        from ebind.models.imagebind.data import IBAudioProcessor

        config = EBindConfig(modalities=list(_MODALITIES))
        model = _EBind.from_pretrained(self.checkpoint, config=config).eval()
        self.model = model.to(self.device)
        # Source the melspec constants from the package, not copies here.
        self.ib_params = IBAudioProcessor()

    def unload(self) -> None:
        self.__dict__.pop("model", None)
        self.__dict__.pop("ib_params", None)

    def extract(self, y: np.ndarray, sr: int, grid) -> dict[str, np.ndarray]:
        out = self.embed_windows(y, sr, grid.centers)
        return {f"ebind_audio_{i:04d}": out[:, i] for i in range(EMBED_DIM)}

    def embed_windows(self, y: np.ndarray, sr: int, centers: np.ndarray) -> np.ndarray:
        """(len(centers), 1024) unit-norm embeddings of 2 s windows centered
        on ``centers`` (edge-clamped)."""
        import torch
        from torchvision import transforms
        from tqdm import tqdm

        from ebind.models.imagebind.data import waveform2melspec

        p = self.ib_params
        normalize = transforms.Normalize(mean=p.mean, std=p.std)
        n_win = int(min(len(y), WINDOW_SEC * sr))

        specs = []
        for start in window_starts(len(y), sr, centers):
            # waveform2melspec mutates its input (mean-subtraction), so hand
            # it a fresh copy; it pads/crops to target_length internally.
            win = torch.from_numpy(np.array(y[start : start + n_win], dtype=np.float32))
            spec = waveform2melspec(win.unsqueeze(0), sr, p.num_mel_bins, p.target_length)
            specs.append(normalize(spec))

        out = np.empty((len(centers), EMBED_DIM))
        batches = range(0, len(specs), BATCH_SIZE)
        for b in tqdm(batches, desc="ebind_audio windows", leave=False):
            chunk = torch.stack(specs[b : b + BATCH_SIZE]).to(self.device)
            with torch.no_grad():
                emb = self.model.forward(audio=chunk)["audio"].float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            out[b : b + len(chunk)] = emb.cpu().numpy()
        return out
