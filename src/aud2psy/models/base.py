"""Abstract base class for aud2psy models (the viz2psy/word2psy BaseModel analog).

Subclasses set class attributes ``name`` and ``level``:

- ``level = "frame"`` — implement ``extract(y, sr, grid) -> dict[str, np.ndarray]``
  where every array has length ``grid.n_windows``. Compute at librosa's native
  resolution, then reduce with ``grid.average`` / ``grid.rate``. Feature keys
  are model-prefixed (``loudness_rms``, ``spectral_centroid``) — the word2psy
  chunk-model convention, which psyquilt's profile registry pattern-matches.
- ``level = "segment"`` — implement ``transcribe(y, sr) -> (segments_df, words_df, info)``
  on 16 kHz audio.

``load()`` is called by the pipeline before extraction (lazy, so ``--help`` and
``--list-models`` never import heavy dependencies); ``unload()`` frees weights.
After extraction the pipeline records ``feature_names_`` on the instance.
"""

from __future__ import annotations


class BaseModel:
    name: str = "base"
    level: str = "frame"  # "frame" or "segment"

    def load(self) -> None:
        """Load weights/resources. Called once before extraction."""

    def unload(self) -> None:
        """Release weights so peak memory is one model at a time."""
        self.__dict__.pop("model", None)
