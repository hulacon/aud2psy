"""Zwicker psychoacoustics via MoSQITo: loudness, sharpness, roughness,
plus a native fluctuation-strength estimate.

The Alluri/Toiviainen psychoacoustic regressors the registry lacked —
roughness above all. Three metrics are MoSQITo's reference
implementations (Apache-2.0, offline, weightless): time-varying Zwicker
loudness (ISO 532-1, free field, 2 ms hop), DIN 45692 sharpness computed
from that same specific-loudness output via
``sharpness_din_from_loudness`` (verified bit-identical to
``sharpness_din_tv`` — loudness is never computed twice), and
Daniel & Weber roughness (200 ms frames, 50% overlap). All reduce onto
the grid with ``Grid.average``.

Fluctuation strength is NOT in MoSQITo (checked 1.2.1 and master,
Aug 2026), so ``psychoacoustic_fluctuation`` is a native Fastl-style
**estimate, not a standardized value** (the tonal/rhythm precedent:
implement natively when no maintained tool exists). Per 1-bark band
(specific loudness summed from MoSQITo's 0.1-bark bins), the mean-removed
envelope's modulation spectrum is computed in 2 s windows centered on
grid centers (window >> hop, the tonal/clap pattern), converted to
relative modulation depth (a 0.01 sone/band floor keeps near-silent
bands from blowing up), weighted by the classic fluctuation-strength
bandpass H(f) = 1/(f/4 + 4/f) peaking at 4 Hz, root-sum-squared over
modulation frequencies (an L1 sum over-counts broadband envelope
transients relative to tonal modulation by ~sqrt(n_bins); Parseval makes
the L2 norm spread-invariant), summed over bands, and one-point
calibrated so Fastl's reference
(1 kHz tone, 60 dB SPL, 100% AM at 4 Hz) reads 1.0 "vacil". Level
dependence is deliberately dropped (the weakest part of the textbook
model, and our inputs are uncalibrated anyway). The estimate preserves
the canonical orderings — peak at 4 Hz modulation with roughness taking
over by 70 Hz, and AM noise above AM tone.

Calibration: MoSQITo wants pascals; audio files are uncalibrated floats.
Samples are mapped as digital RMS 1.0 == 94 dB SPL (1 Pa), a module
constant. Absolute sone/acum/asper/vacil values therefore depend on this
arbitrary reference — relative time courses are the meaningful output
(the loudness_rms honesty stance).

NaN convention: silence yields 0 from MoSQITo for all metrics.
``psychoacoustic_sharpness`` is a spectral ratio, undefined on silence —
it is set NaN where the window's mean loudness < ``SHARPNESS_MIN_SONE``
(the tonal NaN-on-silence precedent; a 0 would read "maximally dull" and
corrupt correlations). Loudness, roughness, and fluctuation stay
ungated: 0 is physically meaningful there.

Declares ``input_sr = 48000`` (the CLAP precedent): sharpness needs full
spectral bandwidth, and 48 kHz is MoSQITo's reference rate. Runtime is
~3x real time on CPU (loudness_zwtv dominates) — the slowest
non-transcription model; budget accordingly.
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel

FULL_SCALE_DB_SPL = 94.0  # digital RMS 1.0 -> 1 Pa; arbitrary reference
FIELD_TYPE = "free"
SHARPNESS_MIN_SONE = 0.25  # sharpness NaN below this window loudness
ROUGHNESS_OVERLAP = 0.5  # -> 100 ms roughness frame hop

FLUCT_WINDOW_SEC = 2.0  # modulation-spectrum window (0.5 Hz resolution)
FLUCT_BARK_BINS = 10  # sum 0.1-bark specific loudness into 1-bark bands
FLUCT_DEPTH_FLOOR = 0.01  # sone/band floor in the relative-depth divide
# One-point calibration: raw estimate on Fastl's reference signal
# (1 kHz tone, 60 dB SPL at our 94 dB reference mapping, 100% AM at 4 Hz)
# measured 10.591 -> scale so the reference reads 1.0 vacil.
FLUCT_CAL = 1.0 / 10.591


def fluctuation_weight(f_mod: np.ndarray) -> np.ndarray:
    """The classic FS bandpass 1/(f/4 + 4/f), normalized to peak 1 at 4 Hz."""
    w = np.zeros_like(f_mod)
    nz = f_mod > 0
    w[nz] = 2.0 / (f_mod[nz] / 4.0 + 4.0 / f_mod[nz])
    return w


def fluctuation_estimate(
    n_specific: np.ndarray, dt: float, centers: np.ndarray
) -> np.ndarray:
    """Fastl-style fluctuation estimate per grid center (see module doc)."""
    fs_env = 1.0 / dt
    n_bands = n_specific.shape[0] // FLUCT_BARK_BINS
    bands = (
        n_specific[: n_bands * FLUCT_BARK_BINS]
        .reshape(n_bands, FLUCT_BARK_BINS, -1)
        .sum(axis=1)
        * (1.0 / FLUCT_BARK_BINS)
    )
    total = bands.shape[1]
    win = min(int(round(FLUCT_WINDOW_SEC * fs_env)), total)
    window = np.hanning(win)
    freqs = np.fft.rfftfreq(win, dt)
    weights = fluctuation_weight(freqs)

    out = np.zeros(len(centers))
    if win < 8:  # degenerate clip shorter than a few envelope samples
        return out
    for i, center in enumerate(centers):
        start = int(round(center * fs_env)) - win // 2
        start = min(max(start, 0), total - win)
        seg = bands[:, start : start + win]
        mean = seg.mean(axis=1, keepdims=True)
        spectrum = np.abs(np.fft.rfft((seg - mean) * window, axis=1))
        depth = spectrum * (2.0 / window.sum()) / (mean + FLUCT_DEPTH_FLOOR)
        out[i] = np.sqrt(((depth * weights) ** 2).sum(axis=1)).sum()
    return FLUCT_CAL * out


class PsychoacousticModel(BaseModel):
    name = "psychoacoustic"
    level = "frame"
    input_sr = 48000

    def extract(self, y: np.ndarray, sr: int, grid) -> dict[str, np.ndarray]:
        from mosqito.sq_metrics import (
            loudness_zwtv,
            roughness_dw,
            sharpness_din_from_loudness,
        )

        pascals = np.asarray(y, dtype=np.float64) * (
            2e-5 * 10 ** (FULL_SCALE_DB_SPL / 20)
        )

        n, n_specific, _, t_loud = loudness_zwtv(pascals, sr, field_type=FIELD_TYPE)
        with np.errstate(invalid="ignore", divide="ignore"):  # mosqito's 0/0 on silent frames
            sharp = np.asarray(sharpness_din_from_loudness(n, n_specific), dtype=float)
        rough, _, _, t_rough = roughness_dw(pascals, sr, overlap=ROUGHNESS_OVERLAP)

        loudness = grid.average(t_loud, n)
        sharpness = grid.average(t_loud, sharp)
        sharpness[~(loudness >= SHARPNESS_MIN_SONE)] = np.nan
        dt = float(t_loud[1] - t_loud[0])
        fluctuation = fluctuation_estimate(n_specific, dt, grid.centers)

        n_gated = int(np.isnan(sharpness).sum())
        self.info_ = {
            "calibration": f"digital RMS 1.0 = {FULL_SCALE_DB_SPL:g} dB SPL (arbitrary reference)",
            "field_type": FIELD_TYPE,
            "sharpness_min_sone": SHARPNESS_MIN_SONE,
            "n_sharpness_gated_windows": n_gated,
            "fluctuation": "native Fastl-style estimate (not standardized); "
            "1.0 = 1 kHz / 60 dB / 100% AM at 4 Hz reference",
        }
        return {
            "psychoacoustic_loudness": loudness,
            "psychoacoustic_sharpness": sharpness,
            "psychoacoustic_roughness": grid.average(
                t_rough, np.atleast_1d(np.asarray(rough, dtype=float))
            ),
            "psychoacoustic_fluctuation": fluctuation,
        }
