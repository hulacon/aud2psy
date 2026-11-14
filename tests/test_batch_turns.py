"""Per-file turn and word sources in a batch (``speakers_csvs`` / ``words_csvs``).

A batch loads each model once across many short files. Until 0.24.0 the
models that read another model's table (conversation and the diarize speech
gate read diarize's turns; speech_rate reads transcribe's words) could run in
a batch only with their producer in the same call, because one ``--speakers``
or ``--words`` table cannot map onto several inputs. A producer stored per
file (one ``*_speakers.csv`` per input) can: each input gets its own table.

Offline: conversation and speech_rate are pure pandas; the gated model is a
stub that records the turns it was handed.
"""

import csv
import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from aud2psy import pipeline
from aud2psy.exceptions import Aud2PsyError
from aud2psy.pipeline import score_audio, score_audio_batch

from conftest import silence

SR = 22050


def _wav(path, seconds=4.0):
    sf.write(path, silence(seconds), SR)
    return path


def _turns(path, *rows):
    df = pd.DataFrame(rows, columns=["speaker", "onset", "offset"])
    df.insert(0, "turn_idx", range(len(df)))
    df.to_csv(path, index=False)
    return path


def _words(path, *rows):
    df = pd.DataFrame(rows, columns=["word", "onset", "offset"])
    df.insert(0, "word_idx", range(len(df)))
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def two(tmp_path):
    """Two silent clips, each with its own (different) turn and word table."""
    wavs = [_wav(tmp_path / "a.wav"), _wav(tmp_path / "b.wav")]
    spk = [_turns(tmp_path / "a_speakers.csv", ("S0", 0.0, 1.5), ("S1", 1.0, 2.0)),
           _turns(tmp_path / "b_speakers.csv", ("S0", 3.0, 4.0))]
    wrd = [_words(tmp_path / "a_words.csv", ("hello", 0.2, 0.4), ("there", 0.5, 0.7)),
           _words(tmp_path / "b_words.csv", ("bye", 3.0, 3.2))]
    return wavs, spk, wrd


def test_conversation_reads_each_files_own_turns(two):
    wavs, spk, _ = two
    batch = score_audio_batch(wavs, ["loudness", "conversation"], hop=1.0,
                              speakers_csvs=spk, show_progress=False)
    for wav, s, res in zip(wavs, spk, batch):
        single = score_audio(wav, ["loudness", "conversation"], hop=1.0,
                             speakers_csv=s, show_progress=False)
        assert list(res.frames_df.columns) == list(single.frames_df.columns)
        np.testing.assert_allclose(res.frames_df.to_numpy(float),
                                   single.frames_df.to_numpy(float), equal_nan=True)
        assert res.meta["models"]["conversation"]["turns_source"] == str(s)
    # the two files' turns differ, so their features must too
    a, b = (r.frames_df["conversation_n_speakers"].to_numpy() for r in batch)
    assert not np.array_equal(a, b)


def test_speech_rate_reads_each_files_own_words(two):
    wavs, _, wrd = two
    batch = score_audio_batch(wavs, ["speech_rate"], hop=1.0,
                              words_csvs=wrd, show_progress=False)
    for wav, w, res in zip(wavs, wrd, batch):
        single = score_audio(wav, ["speech_rate"], hop=1.0, words_csv=w, show_progress=False)
        np.testing.assert_allclose(res.frames_df.to_numpy(float),
                                   single.frames_df.to_numpy(float), equal_nan=True)
        assert res.meta["models"]["speech_rate"]["words_source"] == str(w)


def test_gated_model_gets_each_files_turns(two, monkeypatch):
    """The diarize speech gate reads ``turns_``; each file must see its own."""
    wavs, spk, _ = two
    seen = []

    class Gated:
        input_sr = None
        checkpoint = "stub"

        def load(self):
            pass

        def unload(self):
            pass

        def extract(self, y, sr, grid):
            seen.append(None if self.turns_ is None else self.turns_["onset"].tolist())
            return {"egemaps_stub": np.zeros(grid.n_windows)}

    real = pipeline.get_model
    monkeypatch.setattr(pipeline, "get_model",
                        lambda name, **kw: Gated() if name == "egemaps" else real(name, **kw))
    batch = score_audio_batch(wavs, ["egemaps"], hop=1.0, speakers_csvs=spk, show_progress=False)
    assert seen == [[0.0, 1.0], [3.0]]
    for s, res in zip(spk, batch):
        assert res.meta["models"]["egemaps"]["speech_gate_turns_source"] == str(s)


def test_without_per_file_sources_the_batch_still_needs_its_producer(two):
    wavs, _, _ = two
    with pytest.raises(Aud2PsyError, match="needs diarize"):
        score_audio_batch(wavs, ["conversation"], show_progress=False)
    with pytest.raises(Aud2PsyError, match="needs transcribe"):
        score_audio_batch(wavs, ["speech_rate"], show_progress=False)


def test_one_source_per_input(two):
    wavs, spk, wrd = two
    with pytest.raises(Aud2PsyError, match="one per input"):
        score_audio_batch(wavs, ["conversation"], speakers_csvs=spk[:1], show_progress=False)
    with pytest.raises(Aud2PsyError, match="one per input"):
        score_audio_batch(wavs, ["speech_rate"], words_csvs=wrd[:1], show_progress=False)


def test_stored_turns_and_a_diarize_run_are_not_mixed(two):
    """Two turn sources in one call would leave which one fed the gate implicit."""
    wavs, spk, _ = two
    with pytest.raises(Aud2PsyError, match="diarize"):
        score_audio_batch(wavs, ["diarize", "conversation"], speakers_csvs=spk, show_progress=False)


def test_stored_words_and_a_transcribe_run_are_not_mixed(two):
    wavs, _, wrd = two
    with pytest.raises(Aud2PsyError, match="transcribe"):
        score_audio_batch(wavs, ["transcribe", "speech_rate"], words_csvs=wrd, show_progress=False)


# --- CLI: per-row `speakers` / `words` in an --inputs-from manifest ----------


def _manifest(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def test_cli_manifest_speakers_column(two, tmp_path):
    from aud2psy.cli import main

    wavs, spk, _ = two
    man = _manifest(tmp_path / "m.csv", [
        {"path": str(w), "stimulus_id": f"item{i}", "output": f"item{i}/conversation.csv",
         "speakers": str(s)} for i, (w, s) in enumerate(zip(wavs, spk))])
    out = tmp_path / "out"
    assert main(["conversation", "--inputs-from", str(man), "-o", str(out), "--hop", "1.0"]) == 0
    for i, s in enumerate(spk):
        meta = json.loads((out / f"item{i}" / "conversation.meta.json").read_text())
        assert meta["models"]["conversation"]["turns_source"] == str(s)
        rows = list(csv.DictReader(open(out / f"item{i}" / "conversation_frames.csv")))
        assert {r["stimulus_id"] for r in rows} == {f"item{i}"}


def test_cli_manifest_words_column(two, tmp_path):
    from aud2psy.cli import main

    wavs, _, wrd = two
    man = _manifest(tmp_path / "m.csv", [
        {"path": str(w), "stimulus_id": f"item{i}", "output": f"item{i}/speech_rate.csv",
         "words": str(x)} for i, (w, x) in enumerate(zip(wavs, wrd))])
    out = tmp_path / "out"
    assert main(["speech_rate", "--inputs-from", str(man), "-o", str(out), "--hop", "1.0"]) == 0
    for i, x in enumerate(wrd):
        meta = json.loads((out / f"item{i}" / "speech_rate.meta.json").read_text())
        assert meta["models"]["speech_rate"]["words_source"] == str(x)


def test_cli_manifest_column_must_be_filled_on_every_row(two, tmp_path, capsys):
    from aud2psy.cli import main

    wavs, spk, _ = two
    man = _manifest(tmp_path / "m.csv", [
        {"path": str(wavs[0]), "speakers": str(spk[0])},
        {"path": str(wavs[1]), "speakers": ""}])
    with pytest.raises(SystemExit):
        main(["conversation", "--inputs-from", str(man), "-o", str(tmp_path / "out")])
    assert "every row" in capsys.readouterr().err


def test_cli_manifest_speakers_need_a_reader(two, tmp_path, capsys):
    from aud2psy.cli import main

    wavs, spk, _ = two
    man = _manifest(tmp_path / "m.csv", [{"path": str(w), "speakers": str(s)}
                                         for w, s in zip(wavs, spk)])
    with pytest.raises(SystemExit):
        main(["loudness", "--inputs-from", str(man), "-o", str(tmp_path / "out")])
    assert "speakers" in capsys.readouterr().err


def test_cli_speakers_flag_still_refused_for_a_batch(two, tmp_path, capsys):
    """One --speakers table cannot map onto several inputs; the column can."""
    from aud2psy.cli import main

    wavs, spk, _ = two
    with pytest.raises(SystemExit):
        main(["conversation", *map(str, wavs), "--speakers", str(spk[0]), "-o", str(tmp_path / "out")])
    assert "manifest" in capsys.readouterr().err
