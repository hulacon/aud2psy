"""Structured errors, mirroring viz2psy's exceptions module."""


class Aud2PsyError(Exception):
    """Base class for all aud2psy errors."""


class InputError(Aud2PsyError):
    """Unsupported, missing, or unreadable input file."""


class FFmpegError(Aud2PsyError):
    """ffmpeg is missing or failed to decode the input."""
