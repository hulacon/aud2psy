"""Per-category validation of the sound_events zero-shot prompt bank.

Run on a machine with the CLAP checkpoint available (downloads ~2 GB on
first use):

    python scripts/validate_sound_events.py [real_clip.wav ...]

Prints the stimulus x category cosine matrix for a fixed set of
synthetic stimuli (each synthesizable category has a target stimulus),
per-target rank/margin diagnostics, and — for any real audio/video files
passed as arguments — their category profiles. The human-sound
categories (laughter, crying, shouting, speech, singing, footsteps,
vehicle, animals, crowd) cannot be synthesized credibly: judge those on
real clips and DELETE any category that fails to separate from
PROMPT_BANK in sound_events.py (with raw-cosine scoring that changes
nothing else).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aud2psy.grid import Grid  # noqa: E402
from aud2psy.models.sound_events import PROMPT_BANK, SoundEventsModel  # noqa: E402

SR_CLAP = 48000

# stimulus name -> the category it targets (None: control stimulus)
TARGETS = {
    "rain": "water",
    "siren": "siren_alarm",
    "alarm": "siren_alarm",
    "wind": "wind",
    "thunder": "thunder",
    "gunshots": "gunshot_explosion",
    "applause": "applause",
    "music": "music",
    "tone": None,
    "noise": None,
    "silence": None,
}

NOT_SYNTHESIZABLE = [
    "speech", "singing", "laughter", "crying", "shouting",
    "crowd", "footsteps", "vehicle", "animals",
]


def _env(t: np.ndarray, attack: float, decay: float) -> np.ndarray:
    return np.minimum(t / max(attack, 1e-4), 1.0) * np.exp(-t / decay)


def synthetic_stimuli(dur: float = 10.0, sr: int = SR_CLAP) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    n = int(dur * sr)
    t = np.arange(n) / sr
    from scipy.signal import butter, sosfilt

    def lowpass(y, fc, order=4):
        return sosfilt(butter(order, fc, "low", fs=sr, output="sos"), y)

    def norm(y, rms=0.1):
        return (y / (np.sqrt(np.mean(y**2)) + 1e-12) * rms).astype(np.float32)

    out: dict[str, np.ndarray] = {}

    # rain: dense random droplet impulses over light hiss
    drops = np.zeros(n)
    idx = rng.integers(0, n - 200, size=int(80 * dur))
    for i in idx:
        length = rng.integers(40, 200)
        drops[i : i + length] += rng.standard_normal(length) * np.exp(
            -np.arange(length) / (length / 3)
        ) * rng.uniform(0.2, 1.0)
    out["rain"] = norm(lowpass(drops, 9000) + 0.15 * rng.standard_normal(n))

    # siren: 0.4 Hz pitch wail 600-1200 Hz
    f_inst = 900 + 300 * np.sin(2 * np.pi * 0.4 * t)
    out["siren"] = norm(np.sin(2 * np.pi * np.cumsum(f_inst) / sr))

    # alarm: 1 kHz beeps, 2 Hz, 50% duty
    out["alarm"] = norm(np.sin(2 * np.pi * 1000 * t) * (np.sin(2 * np.pi * 2 * t) > 0))

    # wind: slowly-modulated low-passed noise
    slow = 1 + 0.7 * sosfilt(
        butter(2, 0.5, "low", fs=sr, output="sos"), rng.standard_normal(n)
    ) / 0.05
    out["wind"] = norm(lowpass(rng.standard_normal(n), 600) * np.abs(slow))

    # thunder: two long low rumble bursts
    y = np.zeros(n)
    for start in (0.5, 5.0):
        i = int(start * sr)
        seg = min(int(3.5 * sr), n - i)
        if seg <= 0:
            continue
        y[i : i + seg] += lowpass(rng.standard_normal(seg), 150) * _env(
            np.arange(seg) / sr, 0.08, 1.2
        )
    out["thunder"] = norm(y)

    # gunshots: sharp wideband cracks with fast decay, every ~2 s
    y = np.zeros(n)
    for start in np.arange(0.5, dur - 0.5, 2.0):
        seg = int(0.35 * sr)
        i = int(start * sr)
        y[i : i + seg] += rng.standard_normal(seg) * _env(np.arange(seg) / sr, 0.002, 0.05)
    out["gunshots"] = norm(y)

    # applause: hundreds of random short claps
    y = np.zeros(n)
    idx = rng.integers(0, n - 400, size=int(40 * dur))
    for i in idx:
        length = rng.integers(100, 400)
        y[i : i + length] += rng.standard_normal(length) * np.exp(
            -np.arange(length) / (length / 4)
        )
    out["applause"] = norm(sosfilt(butter(2, [800, 6000], "bandpass", fs=sr, output="sos"), y))

    # music: 120-BPM kick + hats + pentatonic melody (the wordless-clip recipe)
    y = np.zeros(n)
    beat = int(0.5 * sr)
    for b in range(int(dur * 2)):
        i = b * beat
        kick = int(0.12 * sr)
        y[i : i + kick] += 0.9 * np.sin(
            2 * np.pi * 55 * np.arange(kick) / sr
        ) * np.exp(-np.arange(kick) / (0.04 * sr))
        hat = int(0.03 * sr)
        j = i + beat // 2
        if j + hat < n:
            y[j : j + hat] += 0.25 * rng.standard_normal(hat) * np.exp(
                -np.arange(hat) / (0.008 * sr)
            )
    notes = [261.63, 293.66, 329.63, 392.0, 440.0]
    for b in range(int(dur * 2)):
        i = b * beat
        f = notes[int(rng.integers(0, len(notes)))]
        seg = int(0.45 * sr)
        if i + seg < n:
            y[i : i + seg] += 0.4 * np.sin(2 * np.pi * f * np.arange(seg) / sr) * np.exp(
                -np.arange(seg) / (0.3 * sr)
            )
    out["music"] = norm(y)

    out["tone"] = norm(np.sin(2 * np.pi * 440 * t))
    out["noise"] = norm(rng.standard_normal(n))
    out["silence"] = np.zeros(n, dtype=np.float32)
    return out


def score(model: SoundEventsModel, y: np.ndarray) -> dict[str, float]:
    grid = Grid.for_duration(len(y) / SR_CLAP, 0.5)
    out = model.extract(y, SR_CLAP, grid)
    return {c.removeprefix("sound_events_"): float(np.mean(v)) for c, v in out.items()}


def main(argv: list[str]) -> int:
    model = SoundEventsModel()
    print(f"loading {model.checkpoint} ...")
    model.load()

    stimuli = synthetic_stimuli()
    scores = {name: score(model, y) for name, y in stimuli.items()}

    cats = list(PROMPT_BANK)
    width = max(len(c) for c in cats)
    print(f"\n=== stimulus x category cosine matrix ===\n{'':>{width}} "
          + " ".join(f"{s:>9}" for s in scores))
    for c in cats:
        print(f"{c:>{width}} " + " ".join(f"{scores[s][c]:>9.3f}" for s in scores))

    print("\n=== per-target diagnostics (column peak should be the target stimulus) ===")
    # gated stimuli (silence) score NaN across the bank — drop them rather
    # than let NaN comparisons decide the ranking
    gated = [s for s in scores if all(np.isnan(v) for v in scores[s].values())]
    if gated:
        print(f"(silence-gated, excluded from ranking: {', '.join(gated)})")
    for stim, cat in TARGETS.items():
        if cat is None:
            continue
        ranked = sorted(
            (s for s in scores if s not in gated),
            key=lambda s: scores[s][cat],
            reverse=True,
        )
        margin = scores[ranked[0]][cat] - scores[ranked[1]][cat]
        flag = "OK " if ranked[0] == stim or (stim, ranked[0]) in (
            ("siren", "alarm"), ("alarm", "siren")) else "FAIL"
        print(f"{flag} {cat:>18} <- {stim:<9} peak={ranked[0]:<9} "
              f"({scores[ranked[0]][cat]:.3f}, margin {margin:+.3f})")

    if argv:
        print("\n=== real-clip category profiles ===")
        from aud2psy.audio import load_audio

        for path in argv:
            y = load_audio(path, SR_CLAP)
            s = score(model, y)
            top = sorted(s, key=s.get, reverse=True)[:5]
            print(f"{path}: " + ", ".join(f"{c}={s[c]:.3f}" for c in top))

    print("\nNot synthesizable — judge on real clips and delete failures "
          f"from PROMPT_BANK: {', '.join(NOT_SYNTHESIZABLE)}")
    model.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
