"""Contract B §4.1 (schema 1.1) producer duty: every NaN aud2psy can emit is declared.

Each declared condition gets a fixture that fires it (silence, a pure tone,
noise, an empty turn/word table), and every fixture asserts both directions:
the NaN columns are a subset of the declared ones, and every declared key is
a real column of that model. The weighted models' fixtures sit behind the
`weights` marker like their other tests.
"""

import json

import numpy as np
import pandas as pd
import pytest

from aud2psy.cli import MODEL_REGISTRY, main
from aud2psy.grid import Grid
from aud2psy.metadata import SCHEMA_VERSION, declared_nulls, stamp_nulls
from aud2psy.models.conversation import conversation_frames
from aud2psy.models.speech_rate import speech_rate_frames
from aud2psy.pipeline import annotate_voice_gender, save_result, score_audio
from aud2psy.sidecar import refresh_sidecar

from conftest import silence, sine, white_noise

KINDS = {"undefined", "undefinable", "missing"}

# Offline frame models: the NaN-emitting ones plus the ones that must never NaN.
OFFLINE = ["loudness", "pitch", "spectral", "onsets", "tonal", "rhythm", "timbre",
           "psychoacoustic", "speech", "egemaps"]
WEIGHTED = ["clap", "music_emotion", "sound_events", "speech_emotion"]


def _signals():
    # 3 s each; the mixed one puts a tone after silence so gated and ungated
    # windows occur in one table
    return {
        "silence": silence(3.0),
        "tone": sine(220, 3.0),
        "noise": white_noise(3.0),
        "silence_then_tone": np.concatenate([silence(1.5), sine(220, 1.5)]),
        # ends 3 ms into its last 0.5 s window: no native frame centre reaches it
        "trailing_sliver": sine(220, 3.003) + white_noise(3.003, amp=0.1),
    }


def _assert_declared(frames: pd.DataFrame, models: list[str]) -> set[str]:
    declared = {c for m in models for c in declared_nulls(m)}
    nan_cols = {c for c in frames.columns if c != "time" and frames[c].isna().any()}
    assert nan_cols <= declared, f"undeclared NaN columns: {sorted(nan_cols - declared)}"
    return nan_cols


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_every_model_declares_well_formed_nulls(model):
    nulls = declared_nulls(model)
    assert isinstance(nulls, dict)
    for col, entry in nulls.items():
        assert set(entry) == {"means", "when"}, col
        assert entry["means"] in KINDS, col
        assert entry["when"].strip(), col


@pytest.mark.parametrize("name", list(_signals()))
def test_offline_frame_models_emit_only_declared_nulls(name, wav_factory):
    result = score_audio(wav_factory(_signals()[name]), OFFLINE, show_progress=False)
    nan_cols = _assert_declared(result.frames_df, OFFLINE)
    for m in OFFLINE:
        entry = result.meta["models"][m]
        assert set(entry["nulls"]) <= set(entry["columns"]), m  # every key is a real column
    if name == "silence":
        # the declared gates actually fire: silence has no F0, no tonality,
        # no sharpness, and no voice
        assert {"pitch_f0", "tonal_key_clarity", "psychoacoustic_sharpness",
                "egemaps_f0_semitone"} <= nan_cols


def test_roughness_trailing_window_is_undefinable(wav_factory):
    # 2.1 s: the fifth grid window [2.0, 2.5) holds no roughness frame centre
    result = score_audio(wav_factory(sine(220, 2.1, amp=0.3)), ["psychoacoustic"], show_progress=False)
    rough = result.frames_df["psychoacoustic_roughness"]
    assert rough.isna().tolist() == [False] * 4 + [True]
    assert declared_nulls("psychoacoustic")["psychoacoustic_roughness"]["means"] == "undefinable"


def test_undefinable_nulls_sit_only_on_the_trailing_row(wav_factory):
    y = _signals()["trailing_sliver"]
    df = score_audio(wav_factory(y), OFFLINE, show_progress=False).frames_df
    undefinable = [c for m in OFFLINE for c, e in declared_nulls(m).items()
                   if e["means"] == "undefinable"]
    assert df["loudness_rms"].iloc[-1] != df["loudness_rms"].iloc[-1]  # the sliver fires
    assert not df[undefinable].iloc[:-1].isna().any().any()  # and nothing earlier is positional


def test_sidecar_is_1_1_with_every_model_carrying_nulls(wav_factory, tmp_path):
    result = score_audio(wav_factory(sine(220, 2.0)), ["onsets", "pitch"], show_progress=False)
    written = save_result(result, tmp_path / "fam.csv")
    meta = json.loads(written["meta"].read_text())
    assert meta["schema_version"] == SCHEMA_VERSION == "1.1"
    assert "onsets_rate" not in meta["models"]["onsets"]["nulls"]  # a count: never NaN
    assert set(meta["models"]["pitch"]["nulls"]) == {"pitch_f0", "pitch_voiced_prob"}
    assert stamp_nulls({"clap": {"pattern": "clap_{NNN}"}})["clap"]["nulls"] == {}  # positive "never"


def test_conversation_gap_is_declared():
    turns = pd.DataFrame({"turn_idx": [0], "speaker": ["SPEAKER_00"], "onset": [0.0], "offset": [1.0]})
    frames = pd.DataFrame(conversation_frames(turns, Grid.for_duration(3.0, 1.0)))
    assert _assert_declared(frames, ["conversation"]) == {"conversation_turn_duration"}
    assert set(declared_nulls("conversation")) <= set(frames.columns)


def test_speech_rate_gap_is_declared():
    words = pd.DataFrame({"word_idx": [0], "word": ["hi"], "onset": [0.1], "offset": [0.4]})
    frames = pd.DataFrame(speech_rate_frames(words, Grid.for_duration(3.0, 1.0)))
    assert _assert_declared(frames, ["speech_rate"]) == {"speech_rate_word_duration"}
    assert set(declared_nulls("speech_rate")) <= set(frames.columns)


def test_transcript_median_f0_is_declared():
    transcript = pd.DataFrame({"text": ["unvoiced"], "onset": [0.0], "offset": [1.0]})
    frames = pd.DataFrame({"time": [0.25, 0.75], "pitch_f0": [np.nan, np.nan]})
    annotate_voice_gender(transcript, frames, hop=0.5)
    num = transcript.select_dtypes(include="number")
    nan_cols = {c for c in num.columns if num[c].isna().any()}
    assert nan_cols == {"median_f0"} <= set(declared_nulls("transcribe"))


@pytest.mark.weights
@pytest.mark.parametrize("name", ["silence", "tone"])
def test_weighted_models_emit_only_declared_nulls(name, wav_factory):
    result = score_audio(wav_factory(_signals()[name]), WEIGHTED, show_progress=False)
    nan_cols = _assert_declared(result.frames_df, WEIGHTED)
    if name == "silence":
        assert "sound_events_music" in nan_cols and "music_emotion_valence" in nan_cols


class TestRefresh:
    def _family_1_0(self, wav_factory, tmp_path, y):
        """A family written today, then rolled back to how a 1.0 sidecar looked."""
        result = score_audio(wav_factory(y), ["onsets", "pitch"], show_progress=False)
        written = save_result(result, tmp_path / "fam.csv")
        meta = json.loads(written["meta"].read_text())
        meta["schema_version"] = "1.0"
        for entry in meta["models"].values():
            entry.pop("nulls")
        written["meta"].write_text(json.dumps(meta, indent=2))
        return written

    def test_refresh_adds_nulls_and_never_touches_a_csv(self, wav_factory, tmp_path):
        w = self._family_1_0(wav_factory, tmp_path, silence(2.0))
        before = w["frames"].read_bytes()
        assert refresh_sidecar(w["meta"]).status == "refreshed"
        meta = json.loads(w["meta"].read_text())
        assert meta["schema_version"] == "1.1"
        assert set(meta["models"]["pitch"]["nulls"]) == {"pitch_f0", "pitch_voiced_prob"}
        assert "onsets_rate" not in meta["models"]["onsets"]["nulls"]
        assert meta["refreshed"][0]["from_schema_version"] == "1.0"
        assert w["frames"].read_bytes() == before
        assert refresh_sidecar(w["meta"]).status == "unchanged"  # idempotent

    def test_refresh_refuses_an_undeclared_nan(self, wav_factory, tmp_path):
        w = self._family_1_0(wav_factory, tmp_path, sine(220, 2.0))
        frames = pd.read_csv(w["frames"])
        frames.loc[0, "onsets_rate"] = np.nan  # a producer defect: a count cannot be NaN
        frames.to_csv(w["frames"], index=False)
        side_before = w["meta"].read_text()
        r = refresh_sidecar(w["meta"])
        assert r.status == "refused" and r.undeclared == {"onsets": ["onsets_rate"]}
        assert w["meta"].read_text() == side_before

    def test_dry_run_writes_nothing(self, wav_factory, tmp_path):
        w = self._family_1_0(wav_factory, tmp_path, silence(2.0))
        side_before = w["meta"].read_text()
        assert refresh_sidecar(w["meta"], dry_run=True).status == "refreshed"
        assert w["meta"].read_text() == side_before

    def test_cli_walks_a_directory_and_skips_other_extractors(self, wav_factory, tmp_path, capsys):
        self._family_1_0(wav_factory, tmp_path, silence(2.0))
        (tmp_path / "other.meta.json").write_text(json.dumps({"extractor": "viz2psy", "models": {}}))
        assert main(["sidecar", "refresh", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "1 refreshed" in out and "1 skipped" in out

    def test_refresh_finds_tables_beside_a_moved_sidecar(self, wav_factory, tmp_path):
        w = self._family_1_0(wav_factory, tmp_path / "a", silence(2.0))
        moved = tmp_path / "b"
        (tmp_path / "a").rename(moved)
        assert refresh_sidecar(moved / w["meta"].name).status == "refreshed"
