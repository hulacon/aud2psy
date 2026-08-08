"""Unified aud2psy CLI.

Heavy libraries are imported lazily inside main() branches so ``aud2psy --help``
and ``--list-models`` stay fast.
"""

from __future__ import annotations

import argparse
import sys

# name -> (module path, class name, description). Import lazily via pipeline.get_model.
MODEL_REGISTRY: dict[str, tuple[str, str, str]] = {
    "loudness": ("aud2psy.models.loudness", "LoudnessModel", "RMS energy and dB level"),
    "pitch": ("aud2psy.models.pitch", "PitchModel", "pYIN fundamental frequency and voicing probability"),
    "spectral": ("aud2psy.models.spectral", "SpectralModel", "Centroid, bandwidth, rolloff, flux, zero-crossing rate"),
    "onsets": ("aud2psy.models.onsets", "OnsetsModel", "Onset strength, onset rate, local tempo"),
    "tonal": ("aud2psy.models.tonal", "TonalModel", "Key clarity, mode-majorness, chroma entropy (3 s windows)"),
    "rhythm": ("aud2psy.models.rhythm", "RhythmModel", "Pulse clarity, local pulse strength, structural novelty"),
    "speech": ("aud2psy.models.speech", "SpeechModel", "Speech presence probability (Silero VAD)"),
    "clap": ("aud2psy.models.clap", "ClapModel", "512-d CLAP audio embeddings (shared space with word2psy clap_text)"),
    "music_emotion": ("aud2psy.models.music_emotion", "MusicEmotionModel", "Musical valence/arousal (DEAM-trained probe on CLAP)"),
    "beats": ("aud2psy.models.beats", "BeatsModel", "Beat/downbeat event table (beat_this; needs the [beats] extra)"),
    "transcribe": ("aud2psy.models.transcribe", "TranscribeModel", "Time-stamped transcript export for word2psy (faster-whisper)"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aud2psy",
        description=(
            "Extract psychological and acoustic features from audio, including the "
            "audio stream of video. Frame-level models write a time-by-feature CSV; "
            "transcribe exports a transcript CSV for word2psy."
        ),
        epilog='Example: aud2psy loudness pitch transcribe clip.mp4 -o scores.csv',
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        metavar="MODEL... INPUT",
        help="model names followed by one audio/video file (with --all: just the file)",
    )
    parser.add_argument("-o", "--output", metavar="PATH",
                        help="output stem, e.g. scores.csv -> scores_frames.csv, "
                             "scores_transcript.csv, scores.meta.json "
                             "(default: print frames CSV to stdout)")
    parser.add_argument("--all", action="store_true", help="run every registered model")
    parser.add_argument("--hop", type=float, default=0.5, metavar="SEC",
                        help="frame-level window size in seconds (default: 0.5, "
                             "matching viz2psy's video frame interval)")
    parser.add_argument("--whisper-model", default="large-v3", metavar="NAME",
                        help="faster-whisper model for transcribe (default: large-v3)")
    parser.add_argument("--clap-model", default=None, metavar="NAME",
                        help="CLAP checkpoint for the clap model (default: "
                             "laion/larger_clap_music_and_speech)")
    parser.add_argument("--language", default=None, metavar="CODE",
                        help="transcription language code (default: auto-detect)")
    parser.add_argument("--wordpool", default=None, metavar="PATH",
                        help="wordpool file (one item per line) for free-recall "
                             "annotation; requires the transcribe model and adds "
                             "a _recall.csv output")
    parser.add_argument("--list-models", action="store_true", help="list registered models and exit")
    parser.add_argument("--version", action="version", version=_version())
    return parser


def _version() -> str:
    from . import __version__

    return f"aud2psy {__version__}"


def list_models() -> None:
    width = max(len(n) for n in MODEL_REGISTRY)
    levels = {"transcribe": "segment", "beats": "events "}
    for name, (_, _, desc) in MODEL_REGISTRY.items():
        print(f"{name:<{width}}  {levels.get(name, 'frame  ')}  {desc}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        list_models()
        return 0

    if not args.inputs:
        parser.error("missing input file (and model names, unless --all)")
    input_path = args.inputs[-1]
    models = args.inputs[:-1]
    if args.all:
        if models:
            parser.error("--all replaces explicit model names; give just the input file")
        models = list(MODEL_REGISTRY)
        import importlib.util

        if importlib.util.find_spec("beat_this") is None:
            print(
                "note: skipping beats (optional dependency not installed; "
                'pip install "aud2psy[beats]")',
                file=sys.stderr,
            )
            models.remove("beats")
    if not models:
        parser.error("no models requested; name models before the input file or use --all")
    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)} (see --list-models)")
    if args.wordpool and "transcribe" not in models:
        parser.error("--wordpool requires the transcribe model")

    from .exceptions import Aud2PsyError
    from .pipeline import save_result, score_audio

    try:
        result = score_audio(
            input_path,
            models,
            hop=args.hop,
            whisper_model=args.whisper_model,
            language=args.language,
            wordpool=args.wordpool,
            clap_model=args.clap_model,
        )
    except Aud2PsyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        written = save_result(result, args.output)
        for kind, path in written.items():
            print(f"{kind}: {path}")
        if result.transcript_df is not None and len(result.transcript_df) == 0:
            print("note: no speech detected (transcript has 0 rows)")
    else:
        df = result.frames_df if result.frames_df is not None else result.transcript_df
        df.to_csv(sys.stdout, index=False, float_format="%.6g")
    return 0


if __name__ == "__main__":
    sys.exit(main())
