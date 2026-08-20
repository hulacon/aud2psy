"""Dimensional vocal affect (arousal / dominance / valence) from speech.

The audeering `wav2vec2-large-robust-12-ft-emotion-msp-dim` model (Wagner
et al. 2023, TPAMI — the released 12-layer prune of their best model):
CCC .745 arousal / .634 dominance / .638 valence on MSP-Podcast test.
Outputs are the model's native ~0–1 scale (nothing in aud2psy rescales
model outputs; music_emotion's [-1, 1] comes from its DEAM training
targets). Weights are ungated but CC-BY-NC-SA research-only — documented
in the README; no new pip deps (torch/transformers are core).

Frame-level: one prediction per grid window from a 4 s context window
centered on the window midpoint (edge-clamped) — the CLAP window ≫ hop
pattern at 16 kHz. Windows are VAD-gated: where the context window's
mean Silero speech probability falls below GATE_MIN_SPEECH the outputs
are NaN (the pitch/tonal convention — the model emits garbage on music
and effects). Two caveats for users, both from the source paper: valence
rides partly on implicit linguistic content learned in fine-tuning (so it
overlaps text sentiment for English speech; arousal/dominance are the
more purely paralinguistic signals), and background *human* non-speech
sound is its worst-case interference.
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel, auto_device

CHECKPOINT = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
WINDOW_SEC = 4.0
BATCH_SIZE = 8
GATE_MIN_SPEECH = 0.25  # min mean Silero prob over the context window
VAD_WINDOW = 512  # samples per Silero probability at 16 kHz


def _build_model_class():
    """The checkpoint's custom regression head (from its model card)."""
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features):
            x = self.dropout(features)
            x = torch.tanh(self.dense(x))
            x = self.dropout(x)
            return self.out_proj(x)

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            # the model card's 4.x-era init_weights() breaks transformers 5's
            # from_pretrained, which needs the statics post_init() attaches
            self.post_init()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0].mean(dim=1)
            return self.classifier(hidden)

    return EmotionModel


class SpeechEmotionModel(BaseModel):
    name = "speech_emotion"
    level = "frame"
    input_sr = 16000  # the model's native rate; also what Silero VAD wants
    window_sec = WINDOW_SEC
    checkpoint = CHECKPOINT

    def __init__(self, device: str | None = None):
        self.device = device or auto_device()

    def load(self) -> None:
        from faster_whisper.vad import get_vad_model
        from transformers import Wav2Vec2Processor

        self.processor = Wav2Vec2Processor.from_pretrained(CHECKPOINT)
        model = _build_model_class().from_pretrained(CHECKPOINT).eval()
        try:
            self.model = model.to(self.device)
        except Exception:  # MPS op gaps etc. — fall back rather than fail
            self.device = "cpu"
            self.model = model.to(self.device)
        self.vad = get_vad_model()

    def unload(self) -> None:
        for attr in ("model", "processor", "vad"):
            self.__dict__.pop(attr, None)

    def extract(self, y: np.ndarray, sr: int, grid) -> dict[str, np.ndarray]:
        # (n_windows, 3) in the checkpoint's output order: arousal, dominance, valence
        vals = self.predict_windows(y, sr, grid.centers)
        gated = speech_fraction(self.vad, y, sr, grid.centers, WINDOW_SEC) < GATE_MIN_SPEECH
        vals[gated] = np.nan
        return {
            "speech_emotion_valence": vals[:, 2],
            "speech_emotion_arousal": vals[:, 0],
            "speech_emotion_dominance": vals[:, 1],
        }

    def predict_windows(self, y: np.ndarray, sr: int, centers: np.ndarray) -> np.ndarray:
        import torch
        from tqdm import tqdm

        windows = context_windows(y, sr, centers, WINDOW_SEC)
        out = np.empty((len(centers), 3))
        for b in tqdm(range(0, len(windows), BATCH_SIZE), desc="speech_emotion windows",
                      leave=False):
            chunk = windows[b : b + BATCH_SIZE]
            inputs = self.processor(
                chunk, sampling_rate=sr, return_tensors="pt", padding=True
            ).input_values.to(self.device)
            with torch.no_grad():
                logits = self.model(inputs)
            out[b : b + len(chunk)] = logits.cpu().numpy()
        return out


def context_windows(
    y: np.ndarray, sr: int, centers: np.ndarray, window_sec: float
) -> list[np.ndarray]:
    """Edge-clamped context windows centered on ``centers`` (the CLAP pattern)."""
    half = window_sec / 2
    duration = len(y) / sr
    n_win = int(min(len(y), window_sec * sr))
    return [
        y[(s := int(min(max(c - half, 0.0), max(duration - window_sec, 0.0)) * sr)) : s + n_win]
        for c in centers
    ]


def speech_fraction(
    vad, y: np.ndarray, sr: int, centers: np.ndarray, window_sec: float
) -> np.ndarray:
    """Mean Silero speech probability over each context window."""
    if sr != 16000:
        raise ValueError(f"speech_fraction expects 16 kHz audio, got {sr}")
    y = y.astype(np.float32)
    pad = (-len(y)) % VAD_WINDOW
    if pad:
        y = np.concatenate([y, np.zeros(pad, dtype=np.float32)])
    probs = np.asarray(vad(y, num_samples=VAD_WINDOW)).ravel()
    times = (np.arange(len(probs)) + 0.5) * VAD_WINDOW / sr
    half = window_sec / 2
    duration = len(y) / sr
    out = np.empty(len(centers))
    for i, c in enumerate(centers):
        start = min(max(c - half, 0.0), max(duration - window_sec, 0.0))
        m = (times >= start) & (times < start + window_sec)
        out[i] = probs[m].mean() if m.any() else 0.0
    return out
