"""Contract B §4.1: every feature column starts with its model's prefix.

The rule exists so a consumer can attribute a column to the model that made
it -- psytwill maps columns to models by prefix and nothing else. Where
word2psy had this test, aud2psy did not, and two models shipped through
0.14.0 emitting bare names: `beats` wrote `is_downbeat`, and `transcribe`
wrote `text`, `asr_confidence`, `no_speech_prob` and `probability`.

Nothing caught it. psytwill attributed them to a null model and carried
them anyway, so `movies/audio/beats` aggregated to 8,392 rows and **zero
models**. The campaign's own §4.1 gate reported clean because it only
flags columns the sidecar *declares* as features, and these were never
declared. The failure only surfaced when the same undeclared column,
emitted by several models over one stimulus, collided on the aggregate's
key.

These tests read the models' declared column lists, so they need no
weights and no audio.
"""

import pytest

# psytwill's reserved non-feature columns (§4.1). Never prefixed: they are
# identity, the stimulus's own coordinates, or structural position.
RESERVED = {
    "stimulus_id", "filename", "filepath", "image_idx", "time", "onset",
    "offset", "chunk_idx", "chunk_label", "n_words", "word", "word_idx",
    "sentence_idx", "voice", "speaker", "turn_idx",
}


def assert_prefixed(model_name: str, keys) -> None:
    """Every non-reserved key equals the model name or starts with `name_`."""
    bad = [
        k for k in keys
        if k not in RESERVED
        and k != model_name
        and not k.startswith(f"{model_name}_")
    ]
    assert not bad, (
        f"{model_name} emits unprefixed feature columns {bad}. Contract B §4.1 "
        f"requires every feature column to start with the model's declared "
        f"prefix, or to be one of the reserved columns. Fix: rename them to "
        f"{model_name}_<name> where they are built."
    )


def test_beats_columns_are_prefixed():
    from aud2psy.models.beats import BEAT_COLUMNS

    assert_prefixed("beats", BEAT_COLUMNS)


def test_transcribe_segment_columns_are_prefixed():
    from aud2psy.models.transcribe import SEGMENT_COLUMNS

    assert_prefixed("transcribe", SEGMENT_COLUMNS)


def test_transcribe_word_columns_are_prefixed():
    from aud2psy.models.transcribe import WORD_COLUMNS

    assert_prefixed("transcribe", WORD_COLUMNS)


def test_transcribe_carries_structural_ordinals():
    """Whisper can give two words identical start/end, so timings alone do
    not identify a word row -- the ordinals have to be there."""
    from aud2psy.models.transcribe import SEGMENT_COLUMNS, WORD_COLUMNS

    assert "chunk_idx" in SEGMENT_COLUMNS
    assert "chunk_idx" in WORD_COLUMNS and "word_idx" in WORD_COLUMNS


def test_structural_vocabulary_matches_the_text_extractors():
    """One grouping vocabulary across extractors: a transcript segment is a
    chunk, the same level word2psy emits, so consumers need one key rather
    than one per extractor."""
    from aud2psy.models.transcribe import SEGMENT_COLUMNS, WORD_COLUMNS

    assert "segment_idx" not in SEGMENT_COLUMNS + WORD_COLUMNS


def test_sidecar_declares_both_transcribe_frames():
    """`transcribe` writes two tables; only the segments frame was declared.

    That left `transcribe_probability` -- words-only -- emitted but absent
    from 1,060 sidecars. The column was correctly prefixed, so psytwill
    attributed it and the aggregated data is right; what was wrong is the
    sidecar's account of it. Found 2026-08-23 by the campaign gate once it
    checked emission against the sidecar in both directions, having missed
    it (and two harder defects) while it read declarations only.
    """
    import pandas as pd

    from aud2psy.models.transcribe import SEGMENT_COLUMNS, WORD_COLUMNS
    from aud2psy.pipeline import _transcribe_columns

    declared = _transcribe_columns(
        pd.DataFrame(columns=SEGMENT_COLUMNS),
        pd.DataFrame(columns=WORD_COLUMNS),
    )
    for column in SEGMENT_COLUMNS + WORD_COLUMNS:
        assert column in declared, f"{column} emitted but not declared"
    assert "transcribe_probability" in declared
    assert len(declared) == len(set(declared)), "declared columns duplicated"


def test_transcribe_columns_tolerates_no_words_frame():
    """A transcript with no words still declares its segment columns."""
    import pandas as pd

    from aud2psy.models.transcribe import SEGMENT_COLUMNS
    from aud2psy.pipeline import _transcribe_columns

    declared = _transcribe_columns(pd.DataFrame(columns=SEGMENT_COLUMNS), None)
    assert declared == list(SEGMENT_COLUMNS)
