"""Abstract base class for aud2psy models (the viz2psy/word2psy BaseModel analog).

Subclasses set class attributes ``name`` and ``level``:

- ``level = "frame"`` — implement ``extract(y, sr, grid) -> dict[str, np.ndarray]``
  where every array has length ``grid.n_windows``. Compute at librosa's native
  resolution, then reduce with ``grid.average`` / ``grid.rate``. Feature keys
  are model-prefixed (``loudness_rms``, ``spectral_centroid``) — the word2psy
  chunk-model convention, which psytwill's profile registry pattern-matches.
- ``level = "segment"`` — implement ``transcribe(y, sr) -> (segments_df, words_df, info)``
  on 16 kHz audio.

``load()`` is called by the pipeline before extraction (lazy, so ``--help`` and
``--list-models`` never import heavy dependencies); ``unload()`` frees weights.
After extraction the pipeline records ``feature_names_`` on the instance.
"""

from __future__ import annotations


def auto_device() -> str:
    """cuda -> mps -> cpu, mirroring the word2psy/viz2psy convention."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BaseModel:
    name: str = "base"
    level: str = "frame"  # "frame", "segment", or "events"
    input_sr: int | None = None  # frame models: decode rate override (e.g. CLAP's 48 kHz)
    # Seconds of audio each grid row actually saw, when that differs from the
    # grid window itself. Models using the `window >> hop` pattern (a long
    # context window centered on each grid midpoint) set this; None means the
    # row saw only its own [k*hop, (k+1)*hop) window. Recorded per model in
    # the sidecar so a consumer can tell how much smoothing is baked in --
    # Contract B 4.1 requires time-resolved tables to state their semantics.
    # Set it only when EVERY column of the model shares one context window;
    # a window used by a single feature (psychoacoustic's modulation-spectrum
    # window, say) is a per-feature detail and does not belong here.
    window_sec: float | None = None
    # Contract B §4.1: exact architecture+weights identifier for any model
    # with learned parameters (e.g. "laion/larger_clap_music_and_speech");
    # None for analytic/DSP models. Recorded per model in the sidecar.
    checkpoint: str | None = None

    def load(self) -> None:
        """Load weights/resources. Called once before extraction."""

    def unload(self) -> None:
        """Release weights so peak memory is one model at a time."""
        self.__dict__.pop("model", None)
