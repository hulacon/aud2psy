"""Contract B §4.1 (schema 1.2) producer duty: every column aud2psy emits is labelled.

Each model declares ``labels`` (``default`` + per-column overrides, merged
key-wise). These tests check, for every registered model:

- every column in the model's inventory resolves to ``interpretability`` and
  ``derivation``;
- every ``columns`` key is a real column of that model (embeddings are
  pattern-aware: they get no per-column keys);
- every value comes from the contract's vocabularies;
- ``derived_from`` is present exactly where the derivation needs it, and is
  a checkpoint string some aud2psy model actually records.

The inventories below are read from the models' own constants, and the
offline models' inventories are checked against a real run, so a column added
to a model without a label fails here.
"""

import json

import pytest

from aud2psy.cli import MODEL_REGISTRY
from aud2psy.metadata import SCHEMA_VERSION, declared_labels, model_class, stamp_labels
from aud2psy.models.base import (
    DERIVATIONS,
    INTERPRETABILITY,
    LABEL_KEYS,
    NEEDS_DERIVED_FROM,
    SOURCE_VALIDATION_KEYS,
    WINDOW_ALIGNS,
    WINDOW_TOKENS,
    resolve_label,
)
from aud2psy.pipeline import save_result, score_audio
from aud2psy.sidecar import refresh_sidecar

from conftest import sine

OFFLINE = ["loudness", "pitch", "spectral", "onsets", "tonal", "rhythm", "timbre",
           "psychoacoustic", "speech", "egemaps"]


def inventory(model: str) -> list[str]:
    """Every column the model can write (feature tables and, for the event and
    segment models, the reserved columns of their own tables too)."""
    from aud2psy.models import (beats, clap, diarize, ebind_audio, egemaps, music_emotion,
                                phonetics,
                                sound_events, timbre, transcribe)

    fixed = {
        "loudness": ["loudness_rms", "loudness_db"],
        "pitch": ["pitch_f0", "pitch_voiced_prob"],
        "spectral": ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
                     "spectral_flux", "spectral_zcr"],
        "onsets": ["onsets_strength", "onsets_rate", "onsets_tempo"],
        "tonal": ["tonal_key_clarity", "tonal_majorness", "tonal_chroma_entropy"],
        "rhythm": ["rhythm_pulse_clarity", "rhythm_beat_strength", "rhythm_novelty"],
        "timbre": ([f"timbre_mfcc_{i:02d}" for i in range(1, timbre.N_MFCC + 1)]
                   + [f"timbre_contrast_{b:02d}" for b in range(1, timbre.N_CONTRAST_BANDS + 2)]
                   + ["timbre_flatness"]),
        "psychoacoustic": ["psychoacoustic_loudness", "psychoacoustic_sharpness",
                           "psychoacoustic_roughness", "psychoacoustic_fluctuation"],
        "speech": ["speech_prob"],
        "clap": [f"clap_{i:03d}" for i in range(clap.EMBED_DIM)],
        "ebind_audio": [f"ebind_audio_{i:04d}" for i in range(ebind_audio.EMBED_DIM)],
        "music_emotion": [f"music_emotion_{d}" for d in music_emotion.DIMENSIONS],
        "sound_events": [f"sound_events_{c}" for c in sound_events.PROMPT_BANK],
        "speech_emotion": ["speech_emotion_valence", "speech_emotion_arousal",
                           "speech_emotion_dominance"],
        "egemaps": list(egemaps.RAW_TO_CLEAN.values()),
        "phonetics": list(phonetics.COLUMNS),
        "beats": list(beats.BEAT_COLUMNS),
        "diarize": list(diarize.TURN_COLUMNS),
        "conversation": ["conversation_n_speakers", "conversation_speech_fraction",
                         "conversation_overlap_fraction", "conversation_turn_rate",
                         "conversation_switch_rate", "conversation_turn_duration"],
        "speech_rate": ["speech_rate_words", "speech_rate_word_duration", "speech_rate_pauses"],
        # segments + words, plus the columns other models add in the same run:
        # is_filler (verbatim), median_f0/voice_gender (pitch), speaker (diarize)
        "transcribe": list(dict.fromkeys(
            transcribe.SEGMENT_COLUMNS + transcribe.WORD_COLUMNS
            + ["is_filler", "median_f0", "voice_gender", "speaker"])),
    }
    return fixed[model]


EMBEDDINGS = {"clap", "ebind_audio"}


def known_checkpoints() -> set[str]:
    """Every checkpoint string an aud2psy model records by default."""
    from aud2psy.models import clap, ebind_audio, transcribe

    found = {model_class(m).checkpoint for m in MODEL_REGISTRY} - {None}
    return found | {clap.DEFAULT_CHECKPOINT, ebind_audio.DEFAULT_CHECKPOINT,
                    transcribe.DEFAULT_WHISPER_MODEL, transcribe.VERBATIM_CHECKPOINT}


def test_inventory_covers_the_registry():
    assert all(inventory(m) for m in MODEL_REGISTRY)


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_labels_have_the_contract_shape(model):
    labels = model_class(model).labels
    assert set(labels) == {"default", "columns"}, f"{model}: labels need default + columns"
    assert isinstance(labels["default"], dict) and isinstance(labels["columns"], dict)


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_every_column_resolves(model):
    labels = declared_labels(model)
    for col in inventory(model):
        label = resolve_label(labels, col)
        assert label.get("interpretability") in INTERPRETABILITY, (model, col)
        assert label.get("derivation") in DERIVATIONS, (model, col)


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_column_keys_are_real_columns(model):
    columns = declared_labels(model)["columns"]
    if model in EMBEDDINGS:
        assert columns == {}, f"{model}: embedding dimensions take the default only"
    assert set(columns) <= set(inventory(model)), sorted(set(columns) - set(inventory(model)))


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_values_come_from_the_vocabularies(model):
    labels = declared_labels(model)
    for col in inventory(model):
        label = resolve_label(labels, col)
        assert set(label) <= LABEL_KEYS, (model, col, sorted(set(label) - LABEL_KEYS))
        for key in ("LongName", "Description", "Units", "TermURL"):
            if key in label:
                assert isinstance(label[key], str) and label[key].strip(), (model, col, key)
        if "Levels" in label:
            assert isinstance(label["Levels"], dict) and label["Levels"], (model, col)
        for item in label.get("source_validation", []):
            assert set(item) == SOURCE_VALIDATION_KEYS, (model, col)
            assert item["citation"].startswith(("https://", "doi:")), (model, col)
            assert isinstance(item["value"], (int, float)), (model, col)


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_derived_from_present_exactly_where_required(model):
    labels = declared_labels(model)
    checkpoints = known_checkpoints()
    for col in inventory(model):
        label = resolve_label(labels, col)
        if label["derivation"] in NEEDS_DERIVED_FROM:
            assert label.get("derived_from") in checkpoints, (model, col, label.get("derived_from"))
        else:
            assert "derived_from" not in label, (model, col)


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_bookkeeping_columns_are_opaque(model):
    """Bookkeeping describes the extraction, not the stimulus (the family convention)."""
    labels = declared_labels(model)
    for col in inventory(model):
        label = resolve_label(labels, col)
        if label["derivation"] == "bookkeeping":
            assert label["interpretability"] == "opaque", (model, col)


def test_audit_points():
    """The decided per-model labels, spot-checked."""
    from aud2psy.models import clap, diarize, transcribe

    def lab(model, col):
        return resolve_label(declared_labels(model), col)

    for i in range(1, 5):
        assert lab("egemaps", f"egemaps_mfcc{i}")["interpretability"] == "opaque"
    assert lab("egemaps", "egemaps_f0_semitone")["interpretability"] == "nameable"
    for i in range(1, 14):
        assert lab("timbre", f"timbre_mfcc_{i:02d}")["interpretability"] == "opaque"
    assert lab("timbre", "timbre_contrast_01")["interpretability"] == "nameable"
    assert lab("sound_events", "sound_events_music")["derivation"] == "zero-shot"
    assert lab("sound_events", "sound_events_music")["derived_from"] == clap.DEFAULT_CHECKPOINT
    assert lab("music_emotion", "music_emotion_valence")["derivation"] == "probe"
    assert lab("music_emotion", "music_emotion_valence")["derived_from"] == clap.DEFAULT_CHECKPOINT
    assert lab("conversation", "conversation_turn_rate")["derived_from"] == diarize.DEFAULT_PIPELINE
    assert lab("speech_rate", "speech_rate_words")["derived_from"] == transcribe.DEFAULT_WHISPER_MODEL
    assert lab("speech_emotion", "speech_emotion_arousal")["derivation"] == "trained-on-labels"
    assert lab("speech", "speech_prob")["derivation"] == "trained-on-labels"
    for m in ("clap", "ebind_audio"):
        assert lab(m, inventory(m)[0])["interpretability"] == "opaque"
        assert lab(m, inventory(m)[0])["derivation"] == "embedding"


def test_music_emotion_source_validation_matches_the_provenance():
    from aud2psy.models.music_emotion import load_probe

    *_, provenance = load_probe()
    label = resolve_label(declared_labels("music_emotion"), "music_emotion_arousal")
    values = [item["value"] for item in label["source_validation"]]
    assert values == list(provenance["cv"]["arousal"].values())


def test_offline_inventory_matches_a_real_run(wav_factory, tmp_path):
    """The static inventory is what the models emit, and the written sidecar is
    1.2 with every model's labels resolving for every emitted column."""
    pytest.importorskip("opensmile")
    result = score_audio(wav_factory(sine(220, 2.0)), OFFLINE, speech_gate="silero",
                         show_progress=False)
    written = save_result(result, tmp_path / "fam.csv")
    meta = json.loads(written["meta"].read_text())
    assert meta["schema_version"] == SCHEMA_VERSION == "1.4"
    for m in OFFLINE:
        entry = meta["models"][m]
        assert entry["columns"] == inventory(m), m
        for col in entry["columns"]:
            label = resolve_label(entry["labels"], col)
            assert {"interpretability", "derivation"} <= set(label), (m, col)


def test_stamp_keeps_only_emitted_column_keys():
    meta = stamp_labels({"transcribe": {"columns": ["chunk_idx", "onset", "offset",
                                                    "transcribe_text"]},
                         "clap": {"pattern": "clap_{NNN}"}})
    assert set(meta["transcribe"]["labels"]["columns"]) == {"chunk_idx", "onset", "offset",
                                                            "transcribe_text"}
    assert meta["clap"]["labels"] == {"default": declared_labels("clap")["default"],
                                      "columns": {}}


def test_derived_from_follows_the_parent_checkpoint_of_the_same_run():
    from aud2psy.models.transcribe import DEFAULT_WHISPER_MODEL, VERBATIM_CHECKPOINT

    cols = inventory("speech_rate")
    alone = stamp_labels({"speech_rate": {"columns": cols}})
    assert alone["speech_rate"]["labels"]["default"]["derived_from"] == DEFAULT_WHISPER_MODEL
    paired = stamp_labels({"transcribe": {"columns": ["onset"], "checkpoint": VERBATIM_CHECKPOINT},
                           "speech_rate": {"columns": cols}})
    labels = paired["speech_rate"]["labels"]
    assert {resolve_label(labels, c)["derived_from"] for c in cols} == {VERBATIM_CHECKPOINT}
    # and the class attribute is untouched
    assert declared_labels("speech_rate")["default"]["derived_from"] == DEFAULT_WHISPER_MODEL


class TestRefresh:
    def _family_1_1(self, wav_factory, tmp_path):
        """A family written today, rolled back to how a 1.1 sidecar looked."""
        result = score_audio(wav_factory(sine(220, 2.0)), ["onsets", "pitch"], show_progress=False)
        written = save_result(result, tmp_path / "fam.csv")
        meta = json.loads(written["meta"].read_text())
        meta["schema_version"] = "1.1"
        for entry in meta["models"].values():
            entry.pop("labels")
        written["meta"].write_text(json.dumps(meta, indent=2))
        return written

    def test_refresh_adds_labels_and_never_touches_a_csv(self, wav_factory, tmp_path):
        w = self._family_1_1(wav_factory, tmp_path)
        before = w["frames"].read_bytes()
        assert refresh_sidecar(w["meta"]).status == "refreshed"
        meta = json.loads(w["meta"].read_text())
        assert meta["schema_version"] == SCHEMA_VERSION
        assert meta["models"]["pitch"]["labels"] == {
            "default": declared_labels("pitch")["default"],
            "columns": declared_labels("pitch")["columns"]}
        assert "models.*.labels" in meta["refreshed"][-1]["fields"]
        assert meta["refreshed"][-1]["from_schema_version"] == "1.1"
        assert w["frames"].read_bytes() == before
        assert refresh_sidecar(w["meta"]).status == "unchanged"  # idempotent

    def test_refresh_replaces_stale_labels(self, wav_factory, tmp_path):
        w = self._family_1_1(wav_factory, tmp_path)
        refresh_sidecar(w["meta"])
        meta = json.loads(w["meta"].read_text())
        meta["models"]["onsets"]["labels"]["default"]["derivation"] = "embedding"
        w["meta"].write_text(json.dumps(meta, indent=2))
        assert refresh_sidecar(w["meta"]).status == "refreshed"
        meta = json.loads(w["meta"].read_text())
        assert meta["models"]["onsets"]["labels"]["default"]["derivation"] == "formula"


# ---- schema 1.4: per-column window labels --------------------------------------------------

def _is_frame(model: str) -> bool:
    return model_class(model).level == "frame"


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_window_labels_have_the_contract_shape(model):
    """Every column of a frame model declares its window; other levels declare none.

    A missing key would read as "no window" downstream, which is the failure the
    1.4 labels exist to prevent, so absence is never allowed on a frame model.
    """
    labels = declared_labels(model)
    for col in inventory(model):
        label = resolve_label(labels, col)
        if not _is_frame(model):
            assert "window_sec" not in label and "window_align" not in label, (model, col)
            continue
        assert "window_sec" in label, (model, col)
        w = label["window_sec"]
        if isinstance(w, str):
            assert w in WINDOW_TOKENS, (model, col, w)
            assert "window_align" not in label, (model, col)
        else:
            assert isinstance(w, (int, float)) and not isinstance(w, bool) and w > 0, (model, col, w)
            assert label.get("window_align") in WINDOW_ALIGNS, (model, col)


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_model_level_window_matches_the_columns(model):
    """The 1.0 model-level ``window_sec`` is set exactly when all columns share one number."""
    if not _is_frame(model):
        assert model_class(model).window_sec is None
        return
    labels = declared_labels(model)
    windows = {resolve_label(labels, col)["window_sec"] for col in inventory(model)}
    shared = next(iter(windows)) if len(windows) == 1 else None
    expected = shared if isinstance(shared, (int, float)) else None
    assert model_class(model).window_sec == expected, (model, windows)


def test_window_audit_points():
    """The Contract B 1.4 windows, read from the code's own constants."""
    def win(model, col):
        label = resolve_label(declared_labels(model), col)
        return label["window_sec"], label.get("window_align")

    assert win("clap", "clap_000") == (10.0, "centred")
    assert win("music_emotion", "music_emotion_valence") == (10.0, "centred")
    assert win("ebind_audio", "ebind_audio_0000") == (2.0, "centred")
    assert win("speech_emotion", "speech_emotion_arousal") == (4.0, "centred")
    assert win("tonal", "tonal_majorness") == (3.0, "centred")
    assert win("rhythm", "rhythm_pulse_clarity") == (3.0, "centred")
    assert win("rhythm", "rhythm_beat_strength") == (8.92, "centred")  # 384 frames x 512 / 22050
    assert win("rhythm", "rhythm_novelty") == (4.0, "centred")
    assert win("onsets", "onsets_tempo") == (8.0, "centred")
    assert win("onsets", "onsets_rate") == ("grid", None)
    assert win("psychoacoustic", "psychoacoustic_fluctuation") == (2.0, "centred")
    assert win("psychoacoustic", "psychoacoustic_roughness") == ("grid", None)
    assert win("conversation", "conversation_turn_duration") == ("event", None)
    assert win("conversation", "conversation_turn_rate") == ("grid", None)
    assert win("speech_rate", "speech_rate_pauses") == ("event", None)
    assert win("loudness", "loudness_db") == ("grid", None)


def test_window_constants_are_the_ones_the_code_runs():
    """The label constants are the ones passed to librosa, not copies of them."""
    import inspect

    from aud2psy.models import onsets, rhythm

    assert "ac_size=TEMPO_AC_SEC" in inspect.getsource(onsets.OnsetsModel.extract)
    assert "min(PLP_WIN_FRAMES, len(oenv))" in inspect.getsource(rhythm.RhythmModel.extract)


def test_refresh_stamps_model_level_windows(wav_factory, tmp_path):
    """A pre-1.4 tonal sidecar gains window_sec 3.0 and the per-column window labels."""
    result = score_audio(wav_factory(sine(220, 4.0)), ["tonal", "loudness"], show_progress=False)
    written = save_result(result, tmp_path / "fam.csv")
    meta = json.loads(written["meta"].read_text())
    assert meta["models"]["tonal"]["window_sec"] == 3.0
    assert "window_sec" not in meta["models"]["loudness"]
    meta["schema_version"] = "1.2"
    meta["models"]["tonal"].pop("window_sec")
    for entry in meta["models"].values():
        entry["labels"]["default"].pop("window_sec", None)
        entry["labels"]["default"].pop("window_align", None)
    written["meta"].write_text(json.dumps(meta, indent=2))
    assert refresh_sidecar(written["meta"]).status == "refreshed"
    meta = json.loads(written["meta"].read_text())
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["models"]["tonal"]["window_sec"] == 3.0
    assert meta["models"]["tonal"]["labels"]["default"]["window_align"] == "centred"
    assert meta["models"]["loudness"]["labels"]["default"]["window_sec"] == "grid"
    assert "models.*.window_sec" in meta["refreshed"][-1]["fields"]
