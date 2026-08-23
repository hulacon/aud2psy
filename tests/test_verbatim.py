"""Verbatim-mode tests.

Checkpoint resolution, filler tagging, and recall exclusion run offline.
Real CrisperWhisper inference (~3 GB CT2 weights, ~16x real time on CPU)
is behind the `weights` marker: pytest -m weights
"""

import numpy as np
import pandas as pd
import pytest

from aud2psy.cli import main
from aud2psy.models.transcribe import (
    DEFAULT_WHISPER_MODEL,
    FILLER_RE,
    VERBATIM_CHECKPOINT,
    TranscribeModel,
)
from aud2psy.recall import annotate_recall


def test_verbatim_resolves_crisperwhisper_checkpoint():
    model = TranscribeModel(verbatim=True)
    assert model.whisper_model == VERBATIM_CHECKPOINT
    assert TranscribeModel().whisper_model == DEFAULT_WHISPER_MODEL


def test_verbatim_rejects_custom_checkpoint_and_language():
    with pytest.raises(ValueError, match="drop --whisper-model"):
        TranscribeModel(whisper_model="small", verbatim=True)
    with pytest.raises(ValueError, match="en/de"):
        TranscribeModel(language="fr", verbatim=True)
    for ok in (None, "en", "de"):
        TranscribeModel(language=ok, verbatim=True)


def test_filler_regex():
    assert FILLER_RE.match("[UH]") and FILLER_RE.match("[UM]")
    for s in ("uh", "th-", "so[UH]", "[]", "I"):
        assert not FILLER_RE.match(s), s


def test_clean_verbatim_word():
    from aud2psy.models.transcribe import clean_verbatim_word

    # comma separator, and period separator after pauses — both observed
    assert clean_verbatim_word(",word") == "word"
    assert clean_verbatim_word(".Banana") == "Banana"
    assert clean_verbatim_word(",[UM]") == "[UM]"
    assert clean_verbatim_word(".[UH]") == "[UH]"
    assert clean_verbatim_word(".Apple.") == "Apple."  # trailing punct is real
    assert clean_verbatim_word("So") == "So"
    assert clean_verbatim_word("th-") == "th-"  # cut-off marker survives
    assert FILLER_RE.match(clean_verbatim_word(".[UM]"))


def test_cli_verbatim_argument_validation(capsys):
    with pytest.raises(SystemExit):
        main(["loudness", "clip.wav", "--verbatim"])
    assert "--verbatim requires the transcribe model" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["transcribe", "clip.wav", "--verbatim", "--whisper-model", "small"])
    assert "drop --whisper-model" in capsys.readouterr().err


def test_recall_skips_filler_words():
    words = pd.DataFrame({
        "chunk_idx": [0] * 4,
        "word_idx": [0, 1, 2, 3],
        "word": ["apple", "[UM]", "banana", "[UH]"],
        "onset": [1.0, 2.0, 3.0, 4.0],
        "offset": [1.4, 2.2, 3.4, 4.2],
        "transcribe_probability": [0.9] * 4,
        "is_filler": [False, True, False, True],
    })
    recall = annotate_recall(words, ["apple", "banana", "cherry"])
    assert len(recall) == 2  # filler rows never reach the table
    assert list(recall["matched_item"]) == ["apple", "banana"]
    assert recall["intrusion"].sum() == 0
    # IRT spans the filler gap: banana onset - apple onset
    assert recall["irt"].iloc[1] == pytest.approx(2.0)


def test_max_word_run():
    from aud2psy.models.transcribe import max_word_run

    assert max_word_run([]) == 0
    assert max_word_run(["a", "b", "c"]) == 1
    assert max_word_run(["I", "I", "think"]) == 2
    assert max_word_run(["no"] * 10 + ["x"]) == 10


def _fake_decode(words):
    """Build (segments, info) shaped like faster-whisper's output."""
    from types import SimpleNamespace as NS

    seg_words = [
        NS(word=f",{w}", start=i * 0.5, end=i * 0.5 + 0.4, probability=0.9)
        for i, w in enumerate(words)
    ]
    seg = NS(text=",".join(words), start=0.0, end=len(words) * 0.5,
             avg_logprob=-0.1, no_speech_prob=0.01, words=seg_words)
    return iter([seg]), NS(language="en", language_probability=0.99)


class ScriptedWhisper:
    """First decode degenerates; the penalized retry is clean."""

    def __init__(self):
        self.calls = []

    def transcribe(self, y, **kwargs):
        self.calls.append(kwargs)
        if "repetition_penalty" in kwargs:
            return _fake_decode(["So", "[UM]", "I", "I", "think"])
        return _fake_decode(["So", "[UM]"] + ["I"] * 50)


def test_degenerate_decode_triggers_penalized_retry():
    model = TranscribeModel(verbatim=True)
    model.model = ScriptedWhisper()
    segments_df, words_df, info = model.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert len(model.model.calls) == 2
    assert model.model.calls[0].get("repetition_penalty") is None
    assert model.model.calls[0]["vad_filter"] is False
    assert model.model.calls[1]["repetition_penalty"] == 1.1
    assert list(words_df["word"]) == ["So", "[UM]", "I", "I", "think"]
    assert segments_df["transcribe_text"].iloc[0] == "So [UM] I I think"
    assert info["degenerate_retry"] is True and info["repetition_penalty"] == 1.1
    assert info["n_fillers"] == 1


def test_all_separator_segment_is_dropped():
    from types import SimpleNamespace as NS

    class JunkTailWhisper(ScriptedWhisper):
        def transcribe(self, y, **kwargs):
            self.calls.append(kwargs)
            segs, info = _fake_decode(["Apple"])
            junk = NS(text=",", start=5.0, end=6.0, avg_logprob=-1.0, no_speech_prob=0.2,
                      words=[NS(word=",", start=5.0, end=6.0, probability=0.3)])
            return iter(list(segs) + [junk]), info

    model = TranscribeModel(verbatim=True)
    model.model = JunkTailWhisper()
    segments_df, words_df, info = model.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert len(segments_df) == 1 and segments_df["transcribe_text"].iloc[0] == "Apple"
    assert info["n_speech_segments"] == 1
    assert list(words_df["word"]) == ["Apple"]


def test_clean_decode_skips_retry():
    class CleanWhisper(ScriptedWhisper):
        def transcribe(self, y, **kwargs):
            self.calls.append(kwargs)
            return _fake_decode(["So", "[UM]", "I", "I", "think"])

    model = TranscribeModel(verbatim=True)
    model.model = CleanWhisper()
    _, words_df, info = model.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert len(model.model.calls) == 1
    assert info["degenerate_retry"] is False
    assert "repetition_penalty" not in info
    assert words_df["is_filler"].tolist() == [False, True, False, False, False]


@pytest.mark.weights
def test_verbatim_end_to_end(tmp_path):
    import shutil
    import subprocess

    from aud2psy.pipeline import score_audio

    if shutil.which("say") is None:
        pytest.skip("macOS `say` not available to synthesize speech")
    aiff, wav = tmp_path / "f.aiff", tmp_path / "f.wav"
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff),
                    "So, um, I I think the, uh, the meeting should start now."], check=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", str(aiff), "-ac", "1", "-ar", "16000", str(wav)],
        check=True, capture_output=True,
    )
    res = score_audio(wav, ["transcribe"], verbatim=True)
    text = " ".join(res.transcript_df["transcribe_text"])
    assert "[UM]" in text and "[UH]" in text
    assert "I I" in text  # the repetition survives
    assert "," not in " ".join(res.words_df["word"])  # separator artifact stripped
    assert res.words_df["is_filler"].sum() >= 2
    assert res.meta["transcription"]["verbatim"] is True
    assert res.meta["transcription"]["whisper_model"] == VERBATIM_CHECKPOINT
    onsets = res.words_df["onset"].to_numpy()
    assert (np.diff(onsets) >= 0).all()
