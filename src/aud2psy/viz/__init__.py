"""Visualization tools for aud2psy output.

Mirrors the word2psy/viz2psy ``viz`` packages: an interactive single-file
HTML dashboard built from the flat CSV outputs, with Plotly.js loaded from
CDN and all computation done here in Python.
"""

from .dashboard import create_dashboard, prepare_audio

__all__ = ["create_dashboard", "prepare_audio"]
