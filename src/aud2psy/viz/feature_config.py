"""Feature-to-visualization mapping configuration.

Defines which visualization types are appropriate for each model's output,
mirroring word2psy's ``viz/feature_config.py``. Each config records the
output ``level`` ("frame" or "segment"), i.e. which of the output CSVs the
model's columns live in (``*_frames.csv`` vs ``*_transcript.csv``).
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FeatureConfig:
    """Configuration for a model's feature visualization options."""

    # Model name (matches MODEL_REGISTRY key)
    name: str

    # Human-readable description
    description: str

    # Feature type classification
    feature_type: Literal["scalar", "named_distribution", "embedding"]

    # Number of output dimensions
    n_dims: int

    # Which output table the columns live in
    level: Literal["frame", "segment"] = "frame"

    # Visualization appropriateness
    timeseries: bool = False
    mds_clustering: bool = False

    # Special handling options
    timeseries_mode: Literal["all", "top_k", "none"] = "none"
    top_k: int = 5

    # Column patterns for matching (exact names where possible so prefixed
    # families like speech_prob / speech_emotion_* never cross-match)
    column_patterns: list[str] = field(default_factory=list)


FEATURE_CONFIGS: dict[str, FeatureConfig] = {
    # --- frame-level ---
    "loudness": FeatureConfig(
        name="loudness",
        description="RMS energy and dB level",
        feature_type="named_distribution",
        n_dims=2,
        timeseries=True,
        timeseries_mode="all",
        column_patterns=["loudness_rms", "loudness_db"],
    ),
    "pitch": FeatureConfig(
        name="pitch",
        description="pYIN fundamental frequency and voicing probability",
        feature_type="named_distribution",
        n_dims=2,
        timeseries=True,
        timeseries_mode="all",
        column_patterns=["pitch_f0", "pitch_voiced_prob"],
    ),
    "spectral": FeatureConfig(
        name="spectral",
        description="Centroid, bandwidth, rolloff, flux, zero-crossing rate",
        feature_type="named_distribution",
        n_dims=5,
        timeseries=True,
        mds_clustering=True,
        timeseries_mode="all",
        column_patterns=[
            "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
            "spectral_flux", "spectral_zcr",
        ],
    ),
    "onsets": FeatureConfig(
        name="onsets",
        description="Onset strength, onset rate, local tempo",
        feature_type="named_distribution",
        n_dims=3,
        timeseries=True,
        timeseries_mode="all",
        column_patterns=["onsets_strength", "onsets_rate", "onsets_tempo"],
    ),
    "tonal": FeatureConfig(
        name="tonal",
        description="Key clarity, mode-majorness, chroma entropy",
        feature_type="named_distribution",
        n_dims=3,
        timeseries=True,
        timeseries_mode="all",
        column_patterns=["tonal_key_clarity", "tonal_majorness", "tonal_chroma_entropy"],
    ),
    "rhythm": FeatureConfig(
        name="rhythm",
        description="Pulse clarity, local pulse strength, structural novelty",
        feature_type="named_distribution",
        n_dims=3,
        timeseries=True,
        timeseries_mode="all",
        column_patterns=["rhythm_pulse_clarity", "rhythm_beat_strength", "rhythm_novelty"],
    ),
    "timbre": FeatureConfig(
        name="timbre",
        description="MFCCs 1-13, per-octave spectral contrast, spectral flatness",
        feature_type="named_distribution",
        n_dims=21,
        timeseries=True,
        mds_clustering=True,
        timeseries_mode="top_k",
        top_k=8,
        column_patterns=["timbre_mfcc_*", "timbre_contrast_*", "timbre_flatness"],
    ),
    "psychoacoustic": FeatureConfig(
        name="psychoacoustic",
        description="Zwicker loudness, sharpness, roughness, fluctuation estimate",
        feature_type="named_distribution",
        n_dims=4,
        timeseries=True,
        timeseries_mode="all",
        column_patterns=[
            "psychoacoustic_loudness", "psychoacoustic_sharpness",
            "psychoacoustic_roughness", "psychoacoustic_fluctuation",
        ],
    ),
    "speech": FeatureConfig(
        name="speech",
        description="Speech presence probability (Silero VAD)",
        feature_type="scalar",
        n_dims=1,
        timeseries=True,
        timeseries_mode="all",
        column_patterns=["speech_prob"],
    ),
    "clap": FeatureConfig(
        name="clap",
        description="512-dim CLAP audio embeddings",
        feature_type="embedding",
        n_dims=512,
        mds_clustering=True,
        column_patterns=["clap_*"],
    ),
    "music_emotion": FeatureConfig(
        name="music_emotion",
        description="Musical valence/arousal in [-1, 1] (DEAM probe on CLAP)",
        feature_type="named_distribution",
        n_dims=2,
        timeseries=True,
        timeseries_mode="all",
        column_patterns=["music_emotion_valence", "music_emotion_arousal"],
    ),
    "sound_events": FeatureConfig(
        name="sound_events",
        description="16 zero-shot scene/event tags (CLAP prompt bank)",
        feature_type="named_distribution",
        n_dims=16,
        timeseries=True,
        mds_clustering=True,
        timeseries_mode="top_k",
        top_k=8,
        column_patterns=["sound_events_*"],
    ),
    "speech_emotion": FeatureConfig(
        name="speech_emotion",
        description="Vocal arousal/dominance/valence (wav2vec2; VAD-gated)",
        feature_type="named_distribution",
        n_dims=3,
        timeseries=True,
        timeseries_mode="all",
        column_patterns=[
            "speech_emotion_valence", "speech_emotion_arousal", "speech_emotion_dominance",
        ],
    ),
    "egemaps": FeatureConfig(
        name="egemaps",
        description="25 eGeMAPS prosody/voice-quality LLDs (openSMILE)",
        feature_type="named_distribution",
        n_dims=25,
        timeseries=True,
        mds_clustering=True,
        timeseries_mode="top_k",
        top_k=8,
        column_patterns=["egemaps_*"],
    ),
    # --- segment-level (transcript table) ---
    "transcribe": FeatureConfig(
        name="transcribe",
        description="Whisper segment confidence (per 30 s decode window)",
        feature_type="named_distribution",
        n_dims=2,
        level="segment",
        timeseries=True,
        timeseries_mode="all",
        column_patterns=["asr_confidence", "no_speech_prob"],
    ),
}


def get_model_columns(df_columns: list[str], model_name: str) -> list[str]:
    """Get the columns in ``df_columns`` produced by ``model_name``."""
    config = FEATURE_CONFIGS.get(model_name)
    if not config:
        return []

    cols = []
    for pattern in config.column_patterns:
        for col in df_columns:
            if fnmatch.fnmatch(col, pattern) and col not in cols:
                cols.append(col)
    return cols


def detect_models_in_dataframe(
    df_columns: list[str],
    level: Literal["frame", "segment"] | None = None,
) -> list[str]:
    """Detect which aud2psy models produced columns in a DataFrame."""
    detected = []

    for config in FEATURE_CONFIGS.values():
        if level is not None and config.level != level:
            continue
        for pattern in config.column_patterns:
            for col in df_columns:
                if fnmatch.fnmatch(col, pattern):
                    if config.name not in detected:
                        detected.append(config.name)
                    break

    return detected
