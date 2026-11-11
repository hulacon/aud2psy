"""The fixed phone -> articulatory-feature lookup behind the ``phonetics`` model.

The recognizer (``facebook/wav2vec2-lv-60-espeak-cv-ft``) emits 392 espeak-style
tokens: IPA phones from many languages, plus diphthongs, tone digits, length and
secondary-articulation marks, and a few ASCII espeak codes. This module parses
each token into its segments and maps every segment to

- one **manner**: stop, fricative, affricate, nasal, liquid, glide, vowel;
- for consonants, a **place** distribution over labial, labiodental, dental,
  alveolar, postalveolar, palatal, velar, glottal, and a voicing value;
- for vowels, **height** (0 open .. 1 close), **backness** (0 front .. 1 back)
  and **rounding** (0 / 1).

A token with several segments (a diphthong, ``əl``, a triphthong) splits its
mass equally across them. Affricates (``tʃ``, ``dʒ``, ``ts`` ...) are one segment.

Conventions, chosen once and fixed with the table version:

- Length, palatalisation, aspiration, nasalisation, pharyngealisation and tone
  marks do not change a segment's features.
- Taps and flaps (``ɾ``, ``ɽ``) are stops: in English ``ɾ`` is the flapped t/d.
- Trills and rhotic approximants are liquids.
- ``w`` is half labial, half velar; ``ɥ`` half labial, half palatal.
- Retroflex and alveolo-palatal places count as postalveolar; uvular as velar.
- Pharyngeal consonants keep their manner but have no place (their place mass
  is reported as unmapped).
- Tokens that carry no phone (``??``, a bare tone digit, a bare ``ʲ``, and the
  special tokens) are unmapped.

``python -m aud2psy.models.phonetics_table --vocab vocab.json --out table.tsv``
writes the shipped table; ``tests/test_phonetics.py`` checks the shipped table
equals what this parser produces.
"""

from __future__ import annotations

import unicodedata

TABLE_VERSION = "1"

MANNERS = ("stop", "fricative", "affricate", "nasal", "liquid", "glide", "vowel")
PLACES = ("labial", "labiodental", "dental", "alveolar", "postalveolar", "palatal",
          "velar", "glottal")
VOWEL_DIMS = ("height", "backness", "rounded")
SPECIAL_TOKENS = ("<pad>", "<s>", "</s>", "<unk>")
BLANK = "<pad>"

# ---- the segment inventory -------------------------------------------------
_L, _LD, _D, _A, _PA, _PAL, _V, _G = PLACES
C = {}  # symbol -> (manner, {place: weight} or {}, voiced)


def _c(symbols: str, manner: str, place, voiced: int) -> None:
    places = place if isinstance(place, dict) else ({place: 1.0} if place else {})
    for s in symbols.split():
        C[s] = (manner, places, voiced)


_c("p", "stop", _L, 0); _c("b", "stop", _L, 1)
_c("t", "stop", _A, 0); _c("d", "stop", _A, 1)
_c("ʈ", "stop", _PA, 0); _c("ɖ", "stop", _PA, 1)
_c("c", "stop", _PAL, 0); _c("ɟ", "stop", _PAL, 1)
_c("k q", "stop", _V, 0); _c("ɡ g ɢ", "stop", _V, 1)
_c("ʔ", "stop", _G, 0)
_c("ɾ", "stop", _A, 1); _c("ɽ", "stop", _PA, 1)           # taps / flaps
_c("m", "nasal", _L, 1); _c("ɱ", "nasal", _LD, 1); _c("n", "nasal", _A, 1)
_c("ɳ", "nasal", _PA, 1); _c("ɲ", "nasal", _PAL, 1); _c("ŋ ɴ", "nasal", _V, 1)
_c("ɸ", "fricative", _L, 0); _c("β", "fricative", _L, 1)
_c("f", "fricative", _LD, 0); _c("v", "fricative", _LD, 1)
_c("θ", "fricative", _D, 0); _c("ð", "fricative", _D, 1)
_c("s ɬ", "fricative", _A, 0); _c("z", "fricative", _A, 1)
_c("ʃ ʂ ɕ", "fricative", _PA, 0); _c("ʒ ʐ ʑ", "fricative", _PA, 1)
_c("ç", "fricative", _PAL, 0); _c("ʝ", "fricative", _PAL, 1)
_c("x χ", "fricative", _V, 0); _c("ɣ ʁ", "fricative", _V, 1)
_c("ħ", "fricative", None, 0); _c("ʕ", "fricative", None, 1)  # pharyngeal: no place
_c("h", "fricative", _G, 0); _c("ɦ", "fricative", _G, 1)
_c("ts", "affricate", _A, 0); _c("dz", "affricate", _A, 1)
_c("tʃ tʂ tɕ", "affricate", _PA, 0); _c("dʒ dʐ dʑ", "affricate", _PA, 1)
_c("pf", "affricate", _LD, 0)
_c("l ɫ ɹ r", "liquid", _A, 1); _c("ɭ ɻ", "liquid", _PA, 1)
_c("ʎ", "liquid", _PAL, 1); _c("ʀ", "liquid", _V, 1)
_c("j", "glide", _PAL, 1); _c("ɰ", "glide", _V, 1); _c("ʋ", "glide", _LD, 1)
_c("w", "glide", {_L: 0.5, _V: 0.5}, 1); _c("ɥ", "glide", {_L: 0.5, _PAL: 0.5}, 1)
# Czech ř (raised alveolar trill) behaves as a fricative trill. ASCII stand-ins
# ``R`` / ``F`` (voiced / voiceless), so Unicode decomposition cannot strip it.
_c("R", "fricative", _A, 1); _c("F", "fricative", _A, 0)

# height: close 1, near-close .83, close-mid .67, mid .5, open-mid .33,
# near-open .17, open 0; backness: front 0, near-front .2, central .5,
# near-back .8, back 1
V = {
    "i": (1, 0, 0), "y": (1, 0, 1), "ɨ": (1, .5, 0), "ʉ": (1, .5, 1), "ɯ": (1, 1, 0),
    "u": (1, 1, 1), "ɪ": (.83, .2, 0), "ʏ": (.83, .2, 1), "ᵻ": (.83, .5, 0),
    "ʊ": (.83, .8, 1), "e": (.67, 0, 0), "ø": (.67, 0, 1), "ɘ": (.67, .5, 0),
    "ɵ": (.67, .5, 1), "ɤ": (.67, 1, 0), "o": (.67, 1, 1), "E": (.5, 0, 0), "O": (.5, 1, 1),
    "ə": (.5, .5, 0), "ɚ": (.5, .5, 0), "ɛ": (.33, 0, 0), "œ": (.33, 0, 1),
    "ɜ": (.33, .5, 0), "ɞ": (.33, .5, 1), "ʌ": (.33, 1, 0), "ɔ": (.33, 1, 1),
    "æ": (.17, 0, 0), "ɐ": (.17, .5, 0), "a": (0, 0, 0), "ɶ": (0, 0, 1), "A": (0, .5, 0),
    "ɑ": (0, 1, 0), "ɒ": (0, 1, 1),
}
# ``E`` / ``O`` / ``A`` are internal stand-ins for e̞ / o̞ (mid) and ä (open central).

# whole-token rewrites (ASCII espeak codes, non-phone tokens)
_TOKEN_MAP = {"??": "", "1": "", "ʲ": "", "S": "ʃ", "N": "ŋ", "X": "χ", "dZ": "dʒ",
              "tS": "tʃ"}
# substring rewrites, applied in order before segmentation
_REWRITES = [
    ("ər", "ɚ"), ("nɡ", "ŋ"),                       # Mandarin erhua; final -ng
    ("ts.", "tʂ"), ("s.", "ʂ"), ("r.", "ɻ"),         # '.' marks retroflex / apical
    ("i.", "ɨ"), ("a.", "a"), ("u.", "u"),
    ("t[", "t̪"), ("d[", "d̪"),                     # '[' marks dental
    ("e̞", "E"), ("o̞", "O"), ("ä", "A"),
    ("r̝̊", "F"), ("r̝", "R"), ("i̪", "ɨ"),
]
_REWRITES = [(unicodedata.normalize("NFC", a), b) for a, b in _REWRITES]
_DENTAL = "\u032a"  # combining bridge below
# marks with no featural effect here (combining marks other than the dental
# bridge are dropped separately, after NFD)
_STRIP = set("ːˑʲʰˤᵝ^'\".:[") | set("12345")
_STOPLIKE = {s for s, (m, _, _) in C.items() if m in ("stop", "affricate")}


def segments(token: str) -> list[str] | None:
    """The token's segments as inventory symbols; None when it carries no phone.

    Unknown symbols are kept as-is (``mapped`` then reports them).
    """
    if token in SPECIAL_TOKENS:
        return None
    tok = _TOKEN_MAP.get(token, token)
    if not tok:
        return None
    tok = unicodedata.normalize("NFC", tok)
    for a, b in _REWRITES:
        tok = tok.replace(a, b)
    tok = "".join(ch for ch in tok if ch not in "12345")
    # a trailing ɜ after another vowel is espeak's Mandarin tone 3, not a vowel
    if len(tok) > 1 and tok.endswith("ɜ") and any(ch in V for ch in tok[:-1]):
        tok = tok[:-1]
    tok = unicodedata.normalize("NFD", tok)
    out: list[str] = []
    i = 0
    while i < len(tok):
        ch = tok[i]
        if ch == _DENTAL and out and out[-1] in C and C[out[-1]][0] != "vowel":
            out[-1] = out[-1] + _DENTAL  # dental variant of the previous consonant
            i += 1
            continue
        if ch in _STRIP or unicodedata.combining(ch):
            i += 1
            continue
        # aspiration written as a plain h after a stop / affricate (pinyin style)
        if ch == "h" and out and out[-1] in _STOPLIKE:
            i += 1
            continue
        two = unicodedata.normalize("NFC", tok[i:i + 2])
        if len(tok) > i + 1 and two in C:
            out.append(two)
            i += 2
            continue
        out.append(unicodedata.normalize("NFC", ch))
        i += 1
    # length written as a doubled symbol (ee, ää, cʰcʰ) is one segment
    collapsed = [s for k, s in enumerate(out) if k == 0 or s != out[k - 1]]
    return collapsed or None


def _features(seg: str):
    """(manner, places, voiced, vowel) for one segment, or None when unknown."""
    if seg.endswith(_DENTAL):
        base = C.get(seg[:-1])
        if base is None:
            return None
        return base[0], {_D: 1.0}, base[2], None
    if seg in C:
        m, p, v = C[seg]
        return m, p, v, None
    if seg in V:
        return "vowel", {}, None, V[seg]
    return None


def token_row(token: str) -> dict:
    """One table row: the token's mass split over manners, places, voicing, vowel quality.

    Every value is a weight per unit of token mass. Manner weights plus
    ``unmapped`` sum to 1. Place weights sum to the consonant share minus the
    consonant mass with no place. ``voiced`` is the voiced consonant share.
    ``height`` / ``backness`` / ``rounded`` are vowel-share-weighted values, so
    a bin's vowel quality is (sum of these) / (sum of ``vowel``).
    """
    row = {"token": token, "segments": "", "unmapped": 0.0, "consonant": 0.0,
           "place_unmapped": 0.0, "voiced": 0.0,
           **{m: 0.0 for m in MANNERS}, **{p: 0.0 for p in PLACES},
           **{d: 0.0 for d in VOWEL_DIMS}}
    segs = segments(token)
    if not segs:
        row["unmapped"] = 1.0
        row["status"] = "no_phone"
        return row
    row["segments"] = " ".join(segs)
    w = 1.0 / len(segs)
    unknown = 0
    for seg in segs:
        f = _features(seg)
        if f is None:
            row["unmapped"] += w
            unknown += 1
            continue
        manner, places, voiced, vowel = f
        row[manner] += w
        if manner == "vowel":
            for d, val in zip(VOWEL_DIMS, vowel):
                row[d] += w * val
        else:
            row["consonant"] += w
            row["voiced"] += w * voiced
            if places:
                for p, pw in places.items():
                    row[p] += w * pw
            else:
                row["place_unmapped"] += w
    row["status"] = "mapped" if unknown == 0 else ("partial" if unknown < len(segs)
                                                    else "unknown")
    return row


COLUMNS = (("token", "segments", "status") + MANNERS + ("unmapped", "consonant")
           + PLACES + ("place_unmapped", "voiced") + VOWEL_DIMS)


def build_table(tokens):
    """The lookup table for an ordered token list (the checkpoint's vocabulary)."""
    import pandas as pd

    rows = [token_row(t) for t in tokens]
    return pd.DataFrame(rows)[list(COLUMNS)]


def main(argv=None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vocab", required=True, help="the checkpoint's vocab.json")
    ap.add_argument("--out", required=True, help="TSV to write")
    args = ap.parse_args(argv)
    vocab = json.load(open(args.vocab, encoding="utf-8"))
    tokens = sorted(vocab, key=vocab.get)
    table = build_table(tokens)
    table.to_csv(args.out, sep="\t", index=False, float_format="%.6g")
    counts = table["status"].value_counts().to_dict()
    print(f"wrote {len(table)} tokens to {args.out}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
