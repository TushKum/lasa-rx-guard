"""Sound-alike scoring: the "SA" half of LASA.

Orthographic distance alone misses the failure mode this project exists to
catch. ``losec``/``lasix`` differ in 3 of 5 characters — an edit-distance
ranker treats them as barely related — yet they are one of the most notorious
dispensing confusions on record, because they are *spoken* over a phone or
across a counter and they sound nearly identical. Phonetic codes collapse
exactly that: both reduce to Soundex ``L220``.

We combine three encoders rather than trusting one, because each fails
differently:

* **Soundex** is coarse (4 chars, vowels discarded after the first letter). It
  over-collapses — plenty of unrelated names collide — so an exact match is
  strong evidence but only worth a fraction of the score, never the whole of it.
* **Double Metaphone** models English pronunciation properly and emits a second
  code for names with genuinely ambiguous readings (``warfarin`` -> ``ARFRN``
  *or* ``FRFRN``). Drug names are coined words that prescribers pronounce
  inconsistently, so that alternate reading matters.
* **Metaphone** (single) keeps full length, which makes it the only code with
  enough resolution to grade *partial* similarity by edit distance.

NYSIIS is exposed for blocking/indexing but deliberately kept out of the score:
it is tuned for American surnames and its vowel handling adds noise on coined
pharmaceutical names.

Weighting of :func:`phonetic_similarity` (sums to 1.0)
-----------------------------------------------------
    0.25  Soundex codes identical (binary)
    0.35  best Double-Metaphone code agreement, taken over the cross product of
          each name's {primary, secondary}; an exact overlap scores 1.0, and a
          near-miss gets partial credit from normalised edit distance
    0.40  normalised edit similarity of the single-Metaphone codes

The continuous terms carry the majority of the weight on purpose: a binary
Soundex hit is too blunt to rank on, but it is a useful prior. This is a
*component* score, not a verdict — Stage 2 combines it with orthographic and
n-gram signals before flagging anything.
"""

from __future__ import annotations

import argparse

import jellyfish
from metaphone import doublemetaphone

# Weights are module-level so an ablation can import and perturb them rather
# than editing the function body.
W_SOUNDEX = 0.25
W_DMETAPHONE = 0.35
W_METAPHONE = 0.40


def _normalise(s: str) -> str:
    """Lowercase and reduce to ASCII letters.

    Hyphens, digits and whitespace are dropped rather than passed through:
    jellyfish's Soundex/NYSIIS copy unknown leading characters into the code
    verbatim (``soundex("!!") == "!000"``), which would silently produce
    garbage keys. A hyphen is not pronounced anyway, so ``solu-medrol`` and
    ``solumedrol`` should and do encode identically.
    """
    return "".join(c for c in s.strip().lower() if c.isascii() and c.isalpha())


def soundex(s: str) -> str:
    n = _normalise(s)
    return jellyfish.soundex(n) if n else ""


def metaphone(s: str) -> str:
    n = _normalise(s)
    return jellyfish.metaphone(n) if n else ""


def nysiis(s: str) -> str:
    n = _normalise(s)
    return jellyfish.nysiis(n) if n else ""


def double_metaphone(s: str) -> tuple[str, str]:
    """(primary, secondary); secondary is '' when the reading is unambiguous."""
    n = _normalise(s)
    if not n:
        return "", ""
    primary, secondary = doublemetaphone(n)
    return primary, secondary


def _edit_similarity(a: str, b: str) -> float:
    """1 - Levenshtein / max(len), in [0,1]. Empty vs anything -> 0."""
    if not a or not b:
        return 0.0
    return 1.0 - jellyfish.levenshtein_distance(a, b) / max(len(a), len(b))


def phonetic_similarity(a: str, b: str) -> float:
    """Sound-alike similarity in [0,1]. See the module docstring for weights."""
    na, nb = _normalise(a), _normalise(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0

    sx = 1.0 if soundex(na) == soundex(nb) else 0.0

    codes_a = [c for c in double_metaphone(na) if c]
    codes_b = [c for c in double_metaphone(nb) if c]
    # Best agreement over the cross product: this covers "primary matches the
    # other name's secondary", which is the case Double Metaphone's alternate
    # encoding exists for.
    dm = max((_edit_similarity(x, y) for x in codes_a for y in codes_b), default=0.0)

    mp = _edit_similarity(metaphone(na), metaphone(nb))

    return W_SOUNDEX * sx + W_DMETAPHONE * dm + W_METAPHONE * mp


def phonetic_keys(name: str) -> dict[str, str]:
    """All codes for one name, for building blocking indexes.

    Comparing every OCR hypothesis against all 2000 corpus names is affordable,
    but bucketing by these keys first cuts the candidate set by an order of
    magnitude — group corpus names by, say, ``soundex`` and only score the
    names sharing a bucket with the query.
    """
    primary, secondary = double_metaphone(name)
    return {
        "soundex": soundex(name),
        "metaphone": metaphone(name),
        "nysiis": nysiis(name),
        "dmetaphone_primary": primary,
        "dmetaphone_secondary": secondary,
    }


# Documented confusions (see lasa.data.lasa_pairs for provenance) plus a control
# pair that shares no pharmacology, no spelling and no sound.
_DEMO_PAIRS: list[tuple[str, str, str]] = [
    ("dopamine", "dobutamine", "FDA tall-man"),
    ("celebrex", "celexa", "ISMP brand"),
    ("hydroxyzine", "hydralazine", "FDA tall-man"),
    ("losec", "lasix", "ISMP brand"),
    ("warfarin", "amoxicillin", "CONTROL (unrelated)"),
    ("metformin", "zolmitriptan", "CONTROL (unrelated)"),
]


def _corpus_separation(n_random: int, seed: int) -> None:
    """Score the real ground-truth pairs against random corpus pairs.

    Printed so the weighting is judged on the actual data rather than on the
    six hand-picked pairs above, which would be cherry-picking.
    """
    import random

    from lasa.common.paths import DRUG_NAMES_CSV
    from lasa.data.drug_names import load_csv
    from lasa.data.lasa_pairs import deduped_pairs

    if not DRUG_NAMES_CSV.exists():
        print(f"\n(skipping corpus separation: {DRUG_NAMES_CSV} not built)")
        return

    pairs = deduped_pairs()
    known = {frozenset((p.a, p.b)) for p in pairs}
    positives = [phonetic_similarity(p.a, p.b) for p in pairs]

    names = [d.name for d in load_csv(DRUG_NAMES_CSV)]
    rng = random.Random(seed)
    negatives: list[float] = []
    while len(negatives) < n_random:
        x, y = rng.sample(names, 2)
        if frozenset((x, y)) in known:
            continue
        negatives.append(phonetic_similarity(x, y))

    def row(label: str, v: list[float]) -> str:
        v = sorted(v)
        n = len(v)
        return (f"  {label:<22} n={n:<5} mean={sum(v) / n:.3f}  "
                f"p50={v[n // 2]:.3f}  p90={v[9 * n // 10]:.3f}  max={v[-1]:.3f}")

    print("\nCorpus-level separation")
    print(row("ISMP/FDA LASA pairs", positives))
    print(row("random corpus pairs", negatives))


def main() -> None:
    from lasa.common.seed import set_seed

    ap = argparse.ArgumentParser(
        description="Show phonetic codes and similarity for known LASA confusions."
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--random-pairs", type=int, default=2000,
                    help="random pairs sampled for the separation check")
    ap.add_argument("--no-separation", action="store_true",
                    help="skip the corpus-level separation check")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"),
                    help="score one ad-hoc pair instead of the demo set")
    args = ap.parse_args()

    seed = set_seed(args.seed)

    pairs = ([(args.pair[0], args.pair[1], "ad-hoc")] if args.pair else _DEMO_PAIRS)

    words = sorted({w for a, b, _ in pairs for w in (a, b)})
    print(f"seed={seed}")
    print(f"{'name':<14} {'soundex':<9} {'metaphone':<11} {'nysiis':<13} dmetaphone")
    print("-" * 72)
    for w in words:
        k = phonetic_keys(w)
        dm = k["dmetaphone_primary"] + (f" / {k['dmetaphone_secondary']}"
                                        if k["dmetaphone_secondary"] else "")
        print(f"{w:<14} {k['soundex']:<9} {k['metaphone']:<11} {k['nysiis']:<13} {dm}")

    print(f"\n{'pair':<28} {'sim':>6}  {'sx=':>4} {'dm':>5} {'mp':>5}   source")
    print("-" * 72)
    for a, b, src in pairs:
        na, nb = _normalise(a), _normalise(b)
        ca = [c for c in double_metaphone(na) if c]
        cb = [c for c in double_metaphone(nb) if c]
        sx = int(soundex(na) == soundex(nb))
        dm = max((_edit_similarity(x, y) for x in ca for y in cb), default=0.0)
        mp = _edit_similarity(metaphone(na), metaphone(nb))
        print(f"{a + ' / ' + b:<28} {phonetic_similarity(a, b):>6.3f}  "
              f"{sx:>4} {dm:>5.2f} {mp:>5.2f}   {src}")

    if not args.no_separation and not args.pair:
        _corpus_separation(args.random_pairs, seed)


if __name__ == "__main__":
    main()
