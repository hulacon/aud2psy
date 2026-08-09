"""Diarization tests.

The speaker-merge logic is pure pandas and runs offline against hand-built
turn tables. Anything touching real pyannote weights (HF-gated) is behind
the `diarization` marker: pytest -m diarization
"""

import shutil
import subprocess

import pandas as pd
import pytest

from aud2psy.models.diarize import merge_speakers


def turns(*rows):
    df = pd.DataFrame(rows, columns=["speaker", "onset", "offset"])
    df.insert(0, "turn_idx", range(len(df)))
    return df


def test_segment_speaker_is_majority_overlap():
    exclusive = turns(("SPEAKER_00", 0.0, 1.5), ("SPEAKER_01", 1.5, 3.0))
    transcript = pd.DataFrame({"text": ["straddles the boundary"], "onset": [0.0], "offset": [2.0]})
    merge_speakers(transcript, None, exclusive)
    assert transcript["speaker"].tolist() == ["SPEAKER_00"]


def test_segment_with_no_overlap_gets_empty_speaker():
    exclusive = turns(("SPEAKER_00", 0.0, 1.0))
    transcript = pd.DataFrame({"text": ["late"], "onset": [5.0], "offset": [6.0]})
    merge_speakers(transcript, None, exclusive)
    assert transcript["speaker"].tolist() == [""]


def test_majority_sums_across_split_turns():
    # SPEAKER_00 owns 0.6+0.6 s of the segment in two turns; SPEAKER_01 owns 0.8 s
    exclusive = turns(
        ("SPEAKER_00", 0.0, 0.6),
        ("SPEAKER_01", 0.6, 1.4),
        ("SPEAKER_00", 1.4, 2.0),
    )
    transcript = pd.DataFrame({"text": ["x"], "onset": [0.0], "offset": [2.0]})
    merge_speakers(transcript, None, exclusive)
    assert transcript["speaker"].tolist() == ["SPEAKER_00"]


def test_word_speaker_is_midpoint_owner():
    exclusive = turns(("SPEAKER_00", 0.0, 1.5), ("SPEAKER_01", 1.5, 3.0))
    transcript = pd.DataFrame({"text": ["a b"], "onset": [0.0], "offset": [3.0]})
    words = pd.DataFrame(
        {"word": ["a", "b", "gap"], "onset": [0.2, 1.4, 4.0], "offset": [0.5, 1.8, 4.2]}
    )
    merge_speakers(transcript, words, exclusive)
    # word "b" spans the boundary but its midpoint (1.6) is in SPEAKER_01's turn
    assert words["speaker"].tolist() == ["SPEAKER_00", "SPEAKER_01", ""]


def test_zero_row_tables_get_speaker_column():
    exclusive = turns()
    transcript = pd.DataFrame({"text": [], "onset": [], "offset": []})
    words = pd.DataFrame({"word": [], "onset": [], "offset": []})
    merge_speakers(transcript, words, exclusive)
    assert "speaker" in transcript.columns and "speaker" in words.columns
    assert len(transcript) == 0 and len(words) == 0


def test_cli_num_speakers_requires_diarize(capsys):
    from aud2psy.cli import main

    with pytest.raises(SystemExit):
        main(["loudness", "clip.wav", "--num-speakers", "2"])
    assert "requires the diarize model" in capsys.readouterr().err


# --- marked end-to-end tests (gated weights) --------------------------------

pytest_diarization = pytest.mark.diarization

WHISPER_TEST_MODEL = "small"


def make_dialogue_wav(tmp_path):
    """Two alternating macOS `say` voices with 0.6 s gaps — known turn order."""
    if shutil.which("say") is None:
        pytest.skip("macOS `say` not available to synthesize speech")
    lines = [
        ("Daniel", "Hello there, how are you doing today?"),
        ("Samantha", "I am doing very well, thank you for asking."),
        ("Daniel", "That is wonderful news indeed."),
        ("Samantha", "Yes, it certainly is a lovely afternoon."),
    ]
    parts = []
    for i, (voice, text) in enumerate(lines):
        aiff = tmp_path / f"turn_{i}.aiff"
        wav = tmp_path / f"turn_{i}.wav"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", str(aiff), "-ac", "1", "-ar", "16000", str(wav)],
            check=True, capture_output=True,
        )
        parts.append(wav)
    import numpy as np
    import soundfile as sf

    gap = np.zeros(int(0.6 * 16000), dtype="float32")
    audio = np.concatenate([x for p in parts for x in (sf.read(p, dtype="float32")[0], gap)])
    out = tmp_path / "dialogue.wav"
    sf.write(out, audio, 16000)
    return out


@pytest_diarization
def test_diarize_two_voice_dialogue(tmp_path):
    from aud2psy.pipeline import save_result, score_audio

    wav = make_dialogue_wav(tmp_path)
    result = score_audio(
        wav, ["diarize", "transcribe"], whisper_model=WHISPER_TEST_MODEL, num_speakers=2
    )
    speakers = result.speakers_df
    assert result.meta["diarization"]["n_speakers"] == 2
    assert len(speakers) >= 4  # at least one turn per spoken line
    assert (speakers["offset"] > speakers["onset"]).all()
    # transcript rows carry speakers, and both voices appear
    assigned = result.transcript_df["speaker"]
    assert (assigned != "").all()
    assert len(set(assigned)) == 2
    written = save_result(result, tmp_path / "scores.csv")
    assert written["speakers"].exists()
    header = written["speakers"].read_text().splitlines()[0]
    assert header == "turn_idx,speaker,onset,offset"


@pytest_diarization
def test_diarize_silence_is_zero_turns(tmp_path):
    import soundfile as sf

    from aud2psy.pipeline import score_audio

    from conftest import silence

    wav = tmp_path / "silence.wav"
    sf.write(wav, silence(5.0, sr=16000), 16000)
    result = score_audio(wav, ["diarize"])
    assert len(result.speakers_df) == 0
    assert result.meta["diarization"]["n_speakers"] == 0
