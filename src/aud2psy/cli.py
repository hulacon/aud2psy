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
    "timbre": ("aud2psy.models.timbre", "TimbreModel", "MFCCs 1-13, per-octave spectral contrast, spectral flatness"),
    "psychoacoustic": ("aud2psy.models.psychoacoustic", "PsychoacousticModel", "Zwicker loudness, sharpness, roughness (MoSQITo) + fluctuation estimate"),
    "speech": ("aud2psy.models.speech", "SpeechModel", "Speech presence probability (Silero VAD)"),
    "clap": ("aud2psy.models.clap", "ClapModel", "512-d CLAP audio embeddings (shared space with word2psy clap_text)"),
    "ebind_audio": ("aud2psy.models.ebind_audio", "EBindAudioModel", "1024-d EBind audio embeddings (ImageBind arm; shared space with viz2psy ebind / word2psy ebind_text)"),
    "music_emotion": ("aud2psy.models.music_emotion", "MusicEmotionModel", "Musical valence/arousal (DEAM-trained probe on CLAP)"),
    "sound_events": ("aud2psy.models.sound_events", "SoundEventsModel", "16 zero-shot scene/event tags (CLAP prompt bank: laughter, music, sirens, ...)"),
    "speech_emotion": ("aud2psy.models.speech_emotion", "SpeechEmotionModel", "Vocal arousal/dominance/valence (wav2vec2 MSP-Podcast; VAD-gated)"),
    "egemaps": ("aud2psy.models.egemaps", "EgemapsModel", "25 eGeMAPS prosody/voice-quality LLDs (openSMILE; needs the [egemaps] extra)"),
    "beats": ("aud2psy.models.beats", "BeatsModel", "Beat/downbeat event table (beat_this; needs the [beats] extra)"),
    "diarize": ("aud2psy.models.diarize", "DiarizeModel", "Speaker turn table (pyannote community-1; needs the [diarize] extra + HF token)"),
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
        epilog=(
            "Example: aud2psy loudness pitch transcribe clip.mp4 -o scores.csv | "
            "Visualize: aud2psy viz browse scores.csv -o browse.html --open"
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        metavar="MODEL... INPUT",
        help="model names followed by one or more audio/video files (with --all: "
             "just the files). Several files load each model ONCE and run it "
             "across all of them -- see --inputs-from for per-file stimulus ids",
    )
    parser.add_argument("-o", "--output", metavar="PATH",
                        help="output stem, e.g. scores.csv -> scores_frames.csv, "
                             "scores_transcript.csv, scores.meta.json "
                             "(default: print frames CSV to stdout)")
    parser.add_argument("--all", action="store_true", help="run every registered model")
    parser.add_argument("--stimulus-id", default=None, metavar="ID",
                        help="stimulus_id value for all output rows "
                             "(default: input file's stem)")
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
    parser.add_argument("--verbatim", action="store_true",
                        help="verbatim transcription that keeps fillers ([UH]/[UM]), "
                             "repetitions, and false starts (CrisperWhisper; "
                             "English/German only, CC-BY-NC research weights)")
    parser.add_argument("--num-speakers", type=int, default=None, metavar="N",
                        help="exact speaker count hint for the diarize model "
                             "(default: let the pipeline estimate)")
    parser.add_argument("--wordpool", default=None, metavar="PATH",
                        help="wordpool file (one item per line) for free-recall "
                             "annotation; requires the transcribe model and adds "
                             "a _recall.csv output")
    parser.add_argument("--inputs-from", metavar="CSV", default=None,
                        help="batch manifest: a CSV with a `path` column and "
                             "optional `stimulus_id` and `output` columns. The "
                             "way to batch when stimulus_id differs per file "
                             "(--stimulus-id applies one value to every row)")
    parser.add_argument("--list-models", action="store_true", help="list registered models and exit")
    parser.add_argument("--version", action="version", version=_version())
    return parser


def _version() -> str:
    from . import __version__

    return f"aud2psy {__version__}"


def list_models() -> None:
    width = max(len(n) for n in MODEL_REGISTRY)
    levels = {"transcribe": "segment", "beats": "events ", "diarize": "events "}
    for name, (_, _, desc) in MODEL_REGISTRY.items():
        print(f"{name:<{width}}  {levels.get(name, 'frame  ')}  {desc}")


def resolve_scores_paths(path) -> tuple:
    """Resolve a scores path to (frames, transcript, meta) file paths.

    Accepts the base path given to ``-o`` at scoring time (``scores.csv``)
    or either of the actual output files (``scores_frames.csv`` /
    ``scores_transcript.csv``). Missing files resolve to None.
    """
    from pathlib import Path

    path = Path(path)
    name = path.name
    if name.endswith("_frames.csv"):
        base = name[: -len("_frames.csv")]
    elif name.endswith("_transcript.csv"):
        base = name[: -len("_transcript.csv")]
    else:
        base = path.stem
    frames = path.parent / f"{base}_frames.csv"
    transcript = path.parent / f"{base}_transcript.csv"
    meta = path.parent / f"{base}.meta.json"
    return tuple(p if p.exists() else None for p in (frames, transcript, meta))


def _viz_browse(args) -> int:
    """Handle 'aud2psy viz browse'."""
    import json
    from pathlib import Path

    import pandas as pd

    from .viz.dashboard import create_dashboard, prepare_audio

    frames_path, transcript_path, meta_path = resolve_scores_paths(args.csv)
    if frames_path is None and transcript_path is None:
        print(
            f"error: no scores files found for {args.csv} "
            f"(looked for *_frames.csv / *_transcript.csv)",
            file=sys.stderr,
        )
        return 1

    frames_df = pd.read_csv(frames_path) if frames_path else None
    transcript_df = pd.read_csv(transcript_path) if transcript_path else None
    meta = json.loads(meta_path.read_text()) if meta_path else None
    found = ", ".join(str(p) for p in (frames_path, transcript_path) if p)
    print(f"Building dashboard from {found}...")

    # Audio to embed: --audio flag, else the sidecar's recorded input path
    audio = None
    audio_path = args.audio or (meta or {}).get("input", {}).get("path")
    if audio_path and Path(audio_path).exists():
        print(f"Embedding audio from {audio_path}...")
        try:
            audio = prepare_audio(audio_path)
        except Exception as exc:
            print(f"warning: could not embed audio ({exc})", file=sys.stderr)
    elif audio_path:
        print(
            f"warning: audio file not found ({audio_path}); "
            "pass --audio to enable playback",
            file=sys.stderr,
        )
    else:
        print(
            "warning: no audio path (no sidecar); pass --audio to enable playback",
            file=sys.stderr,
        )

    args_csv = Path(args.csv)
    try:
        html = create_dashboard(
            frames_df,
            transcript_df,
            audio=audio,
            meta=meta,
            title=args.title or f"aud2psy — {args_csv.stem}",
            max_points=args.max_points,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    base = args_csv.name
    for suffix in ("_frames.csv", "_transcript.csv"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    output = args.output or args_csv.parent / f"{Path(base).stem}_browse.html"
    output = Path(output)
    output.write_text(html, encoding="utf-8")
    print(f"Saved dashboard to {output}")

    if args.open:
        import webbrowser

        webbrowser.open(f"file://{output.absolute()}")
    return 0


def _viz_main(argv: list[str]) -> int:
    """Handle 'aud2psy viz ...' subcommands."""
    parser = argparse.ArgumentParser(
        prog="aud2psy viz",
        description="Visualize aud2psy output.",
    )
    sub = parser.add_subparsers(dest="viz_cmd")

    p_br = sub.add_parser(
        "browse",
        help="Interactive HTML dashboard with audio play/pause controls.",
    )
    p_br.add_argument(
        "csv",
        help="Scores path (scores.csv base, or a *_frames.csv / *_transcript.csv file).",
    )
    p_br.add_argument("-o", "--output", help="Output HTML path.")
    p_br.add_argument("--audio", metavar="PATH",
                      help="audio/video file to embed for playback "
                           "(default: the input recorded in the .meta.json sidecar)")
    p_br.add_argument("--title", help="Dashboard title.")
    p_br.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Maximum rows per table to include (default: 2000).",
    )
    p_br.add_argument(
        "--open",
        action="store_true",
        help="Open the dashboard in a browser after creation.",
    )

    args = parser.parse_args(argv)
    if args.viz_cmd is None:
        parser.print_help()
        return 1
    return _viz_browse(args)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "viz":
        return _viz_main(argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        list_models()
        return 0

    # Split MODEL... INPUT... by registry membership rather than by position:
    # leading tokens that name a model are models, everything after is a file.
    # Positional splitting cannot express more than one input.
    batch_rows: list[dict] | None = None
    if args.inputs_from:
        import csv as _csv

        with open(args.inputs_from, newline="") as f:
            batch_rows = list(_csv.DictReader(f))
        if not batch_rows:
            parser.error(f"--inputs-from {args.inputs_from} has no rows")
        if "path" not in batch_rows[0]:
            parser.error(f"--inputs-from {args.inputs_from} needs a `path` column "
                         f"(got: {', '.join(batch_rows[0])})")
        input_paths = [r["path"] for r in batch_rows]
        models = list(args.inputs)
        stray = [m for m in models if m not in MODEL_REGISTRY]
        if stray:
            parser.error(f"with --inputs-from, positionals are model names only; "
                         f"got {', '.join(stray)}")
    else:
        if not args.inputs:
            parser.error("missing input file (and model names, unless --all)")
        n_models = 0
        while n_models < len(args.inputs) and args.inputs[n_models] in MODEL_REGISTRY:
            n_models += 1
        models = list(args.inputs[:n_models])
        input_paths = list(args.inputs[n_models:])
        if not input_paths:
            # every token was a model name -- the old parse treated the last as
            # the input file, so keep that reading rather than erroring.
            input_paths = [models.pop()]
    input_path = input_paths[0]
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
        if importlib.util.find_spec("pyannote") is None:
            print(
                "note: skipping diarize (optional dependency not installed; "
                'pip install "aud2psy[diarize]")',
                file=sys.stderr,
            )
            models.remove("diarize")
        if importlib.util.find_spec("opensmile") is None:
            print(
                "note: skipping egemaps (optional dependency not installed; "
                'pip install "aud2psy[egemaps]")',
                file=sys.stderr,
            )
            models.remove("egemaps")
    if not models:
        parser.error("no models requested; name models before the input file or use --all")
    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)} (see --list-models)")
    if args.wordpool and "transcribe" not in models:
        parser.error("--wordpool requires the transcribe model")
    if args.verbatim and "transcribe" not in models:
        parser.error("--verbatim requires the transcribe model")
    if args.verbatim and args.whisper_model != "large-v3":
        parser.error("--verbatim selects the CrisperWhisper checkpoint; drop --whisper-model")
    if args.num_speakers and "diarize" not in models:
        parser.error("--num-speakers requires the diarize model")

    from .exceptions import Aud2PsyError
    from .pipeline import save_result, score_audio, score_audio_batch

    if len(input_paths) > 1:
        from pathlib import Path

        if not args.output:
            parser.error("batching several inputs needs -o DIR (stdout takes one table)")
        out_dir = Path(args.output)
        if out_dir.suffix:
            parser.error(f"-o must be a DIRECTORY when batching, got {args.output!r}")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            results = score_audio_batch(
                input_paths,
                models,
                hop=args.hop,
                whisper_model=args.whisper_model,
                language=args.language,
                wordpool=args.wordpool,
                clap_model=args.clap_model,
                num_speakers=args.num_speakers,
                verbatim=args.verbatim,
            )
        except Aud2PsyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for i, (res, src_path) in enumerate(zip(results, input_paths)):
            row = batch_rows[i] if batch_rows else {}
            stem = row.get("output") or f"{Path(src_path).stem}.csv"
            # `output` may nest (one subdirectory per condition is normal), but
            # it is a path from a file, so it must not escape the output dir.
            if Path(stem).is_absolute() or ".." in Path(stem).parts:
                parser.error(f"--inputs-from row {i}: `output` must be a relative "
                             f"path inside -o, got {stem!r}")
            sid = row.get("stimulus_id") or args.stimulus_id
            written = save_result(res, out_dir / stem, stimulus_id=sid)
            print(f"{Path(src_path).name}: {written['meta']}")
        print(f"batched {len(results)} inputs through {len(models)} model(s), "
              f"each model loaded once")
        return 0

    try:
        result = score_audio(
            input_path,
            models,
            hop=args.hop,
            whisper_model=args.whisper_model,
            language=args.language,
            wordpool=args.wordpool,
            clap_model=args.clap_model,
            num_speakers=args.num_speakers,
            verbatim=args.verbatim,
        )
    except Aud2PsyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        written = save_result(result, args.output, stimulus_id=args.stimulus_id)
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
