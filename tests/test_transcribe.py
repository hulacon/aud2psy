"""Transcription tests — behind the `transcription` marker (downloads Whisper
weights on a cold cache; uses macOS `say` to synthesize known speech).

Run with: pytest -m transcription
"""

import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from aud2psy.models.transcribe import TranscribeModel
from aud2psy.pipeline import save_result, score_audio

from conftest import silence

pytestmark = pytest.mark.transcription

WHISPER_TEST_MODEL = "small"  # keep marked tests light; large-v3 is exercised in validation


def make_speech_wav(tmp_path, text="The quick brown fox jumps over the lazy dog"):
    if shutil.which("say") is None:
        pytest.skip("macOS `say` not available to synthesize speech")
    aiff = tmp_path / "speech.aiff"
    wav = tmp_path / "speech.wav"
    subprocess.run(["say", "-o", str(aiff), text], check=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", str(aiff), "-ac", "1", "-ar", "16000", str(wav)],
        check=True, capture_output=True,
    )
    return wav


def test_transcribe_synthesized_speech(tmp_path):
    wav = make_speech_wav(tmp_path)
    model = TranscribeModel(whisper_model=WHISPER_TEST_MODEL)
    model.load()
    y, sr = sf.read(wav, dtype="float32")
    segments_df, words_df, info = model.transcribe(y, sr)

    assert len(segments_df) >= 1
    text = " ".join(segments_df["transcribe_text"]).lower()
    for word in ("quick", "brown", "fox", "lazy", "dog"):
        assert word in text
    assert (segments_df["offset"] > segments_df["onset"]).all()
    assert info["language"] == "en"
    # word timing is monotone and inside the clip
    assert (np.diff(words_df["onset"]) >= 0).all()
    assert words_df["offset"].iloc[-1] <= len(y) / sr + 0.5


def test_transcribe_silence_is_zero_rows(tmp_path):
    wav = tmp_path / "silence.wav"
    sf.write(wav, silence(5.0, sr=16000), 16000)
    result = score_audio(wav, ["transcribe"], whisper_model=WHISPER_TEST_MODEL)
    assert len(result.transcript_df) == 0
    assert len(result.words_df) == 0
    assert result.meta["transcription"]["n_speech_segments"] == 0
    written = save_result(result, tmp_path / "scores.csv")
    assert written["transcript"].read_text().strip() == (
        "chunk_idx,onset,offset,transcribe_text,"
        "transcribe_asr_confidence,transcribe_no_speech_prob"
    )


def test_recall_wordpool_end_to_end(tmp_path):
    """Spoken word list -> transcribe -> wordpool annotation."""
    wav = make_speech_wav(
        tmp_path, text="apple [[slnc 800]] banana [[slnc 800]] guitar [[slnc 800]] apple"
    )
    pool_file = tmp_path / "pool.txt"
    pool_file.write_text("apple\nbanana\ncherry\ndolphin\n")
    result = score_audio(
        wav, ["transcribe"], whisper_model=WHISPER_TEST_MODEL, wordpool=pool_file
    )
    recall = result.recall_df
    assert recall is not None and len(recall) == 4
    matched = recall[~recall["intrusion"]]
    assert set(matched["matched_item"]) == {"apple", "banana"}
    assert recall["intrusion"].sum() == 1  # guitar is not in the pool
    assert recall["repetition"].sum() == 1  # second apple
    assert (matched["irt"].dropna() > 0).all()
    info = result.meta["transcription"]
    assert info["recall"]["n_recalled_unique"] == 2
    assert info["speech_timing"]["n_words"] == 4
