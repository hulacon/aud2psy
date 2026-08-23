"""Model-major batching: load each model once, run it across many files.

Fully offline -- only DSP models (no weights) plus a stubbed counting model,
so nothing here downloads or needs a GPU.
"""

import numpy as np
import pytest
import soundfile as sf

from aud2psy.pipeline import score_audio, score_audio_batch

from conftest import SR, sine, white_noise

DSP = ["loudness", "spectral"]


@pytest.fixture
def clips(tmp_path):
    """Three distinguishable 2 s clips on disk."""
    paths = []
    for i, y in enumerate((sine(220, 2.0), sine(880, 2.0), white_noise(2.0))):
        p = tmp_path / f"clip{i}.wav"
        sf.write(p, y, SR)
        paths.append(p)
    return paths


def test_batch_matches_per_file(clips):
    """The point of the refactor: identical values, not merely similar."""
    single = [score_audio(p, DSP, show_progress=False) for p in clips]
    batch = score_audio_batch(clips, DSP, show_progress=False)
    assert len(batch) == len(single)
    for a, b in zip(single, batch):
        assert list(a.frames_df.columns) == list(b.frames_df.columns)
        np.testing.assert_allclose(
            a.frames_df.to_numpy(float), b.frames_df.to_numpy(float), equal_nan=True
        )


def test_batch_sidecar_describes_its_own_file(clips):
    """Each result must carry ITS input's provenance, not the batch's first."""
    batch = score_audio_batch(clips, DSP, show_progress=False)
    for p, res in zip(clips, batch):
        assert res.meta["input"]["path"] == str(p)
        for m in DSP:
            assert res.meta["models"][m]["batched"] is True


def test_each_model_loads_once_per_batch(clips, monkeypatch):
    """The whole reason batching exists -- assert it, don't assume it."""
    import aud2psy.pipeline as pipeline

    loads = []
    real_get_model = pipeline.get_model

    def counting_get_model(name, **kwargs):
        model = real_get_model(name, **kwargs)
        real_load = model.load

        def load():
            loads.append(name)
            return real_load()

        model.load = load
        return model

    monkeypatch.setattr(pipeline, "get_model", counting_get_model)

    score_audio_batch(clips, DSP, show_progress=False)
    assert sorted(loads) == sorted(DSP), f"expected one load per model, got {loads}"

    loads.clear()
    for p in clips:
        score_audio(p, DSP, show_progress=False)
    assert len(loads) == len(DSP) * len(clips)  # the per-file cost being avoided


def test_empty_batch_is_not_an_error():
    assert score_audio_batch([], DSP, show_progress=False) == []


def test_unknown_model_rejected(clips):
    with pytest.raises(KeyError):
        score_audio_batch(clips, ["loudness", "definitely_not_a_model"], show_progress=False)


def test_cli_batch_writes_one_family_per_input(clips, tmp_path, capsys):
    from aud2psy.cli import main

    out = tmp_path / "out"
    rc = main(["loudness", *[str(p) for p in clips], "-o", str(out)])
    assert rc == 0
    for p in clips:
        assert (out / f"{p.stem}_frames.csv").exists()
        assert (out / f"{p.stem}.meta.json").exists()


def test_cli_batch_manifest_sets_per_file_stimulus_id(clips, tmp_path):
    """--stimulus-id applies one value to every row; a manifest is the way
    to give each input its own -- which is what the campaign needs."""
    import csv

    from aud2psy.cli import main

    man = tmp_path / "m.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, ["path", "stimulus_id", "output"])
        w.writeheader()
        for i, p in enumerate(clips):
            w.writerow({"path": str(p), "stimulus_id": f"word{i}", "output": f"o{i}.csv"})

    out = tmp_path / "out"
    assert main(["loudness", "--inputs-from", str(man), "-o", str(out)]) == 0
    for i in range(len(clips)):
        rows = list(csv.DictReader(open(out / f"o{i}_frames.csv")))
        assert {r["stimulus_id"] for r in rows} == {f"word{i}"}


def test_cli_batch_requires_output_dir(clips, tmp_path):
    from aud2psy.cli import main

    with pytest.raises(SystemExit):
        main(["loudness", *[str(p) for p in clips]])          # no -o
    with pytest.raises(SystemExit):
        main(["loudness", *[str(p) for p in clips], "-o", str(tmp_path / "x.csv")])  # a file


def test_single_input_still_uses_the_unbatched_path(clips, tmp_path):
    """Backwards compatibility: one file must behave exactly as before."""
    from aud2psy.cli import main

    out = tmp_path / "one.csv"
    assert main(["loudness", str(clips[0]), "-o", str(out)]) == 0
    assert (tmp_path / "one_frames.csv").exists()
    assert (tmp_path / "one.meta.json").exists()
    import json

    meta = json.load(open(tmp_path / "one.meta.json"))
    assert "batched" not in meta["models"]["loudness"]


def test_cli_batch_manifest_output_may_nest(clips, tmp_path):
    """One subdirectory per condition is the normal shape for a campaign."""
    import csv

    from aud2psy.cli import main

    man = tmp_path / "m.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, ["path", "output"])
        w.writeheader()
        for i, p in enumerate(clips):
            w.writerow({"path": str(p), "output": f"voice{i}/clip{i}.csv"})
    out = tmp_path / "out"
    assert main(["loudness", "--inputs-from", str(man), "-o", str(out)]) == 0
    for i in range(len(clips)):
        assert (out / f"voice{i}" / f"clip{i}_frames.csv").exists()


@pytest.mark.parametrize("bad", ["/etc/x.csv", "../escape.csv", "a/../../b.csv"])
def test_cli_batch_manifest_output_cannot_escape(clips, tmp_path, bad):
    import csv

    from aud2psy.cli import main

    man = tmp_path / "m.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, ["path", "output"])
        w.writeheader()
        w.writerow({"path": str(clips[0]), "output": bad})
        w.writerow({"path": str(clips[1]), "output": "ok.csv"})
    with pytest.raises(SystemExit):
        main(["loudness", "--inputs-from", str(man), "-o", str(tmp_path / "out")])
