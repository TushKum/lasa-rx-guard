"""Character n-gram TF-IDF retrieval over the drug-name corpus.

This is the look-alike ("LA") retrieval component, and it exists because Stage 1
hands us *garbage*. A CRNN reading a handwritten chit emits ``dobutamin`` or
``hydralzine`` — strings that are not in any dictionary. Anything that needs the
query to be a known word is useless here.

Character n-grams degrade gracefully instead. Dropping a trailing ``e`` destroys
one token but leaves the other ~20 character 3-grams of ``dobutamine`` intact,
so cosine similarity stays high. ``analyzer='char_wb'`` pads each name with
spaces before extracting n-grams, which makes prefix and suffix n-grams
distinguishable from interior ones — that matters for drug names, where the INN
stem (``-olol``, ``-statin``) is a real morpheme and a shared *suffix* is far
more meaningful than the same letters appearing mid-word.

TF-IDF weighting then down-weights n-grams that half the corpus shares (``in``,
``ine``, ``ol``) and up-weights the distinctive ones, which is exactly the
behaviour you want when 2000 drug names all end in the same handful of stems.

Critical invariant: queries are **transformed, never fitted**. Refitting on the
query would change the IDF weights and make scores incomparable across calls,
and would leak OCR noise into the vocabulary. The cost is that n-grams unseen
at fit time are silently ignored — documented in :meth:`NgramIndex.similarity`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Bump when the persisted layout changes so stale .joblib files fail loudly.
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class NgramConfig:
    """Vectoriser settings.

    ``(2, 4)`` is the default range because bigrams survive heavy OCR
    corruption while 4-grams carry enough of the stem to keep unrelated names
    apart; going wider mostly inflates the vocabulary.
    """

    ngram_range: tuple[int, int] = (2, 4)
    analyzer: str = "char_wb"
    sublinear_tf: bool = True   # 1+log(tf); a repeated n-gram is not 2x evidence
    min_df: int = 1             # corpus is small — a hapax n-gram is still signal


class NgramIndex:
    """Fit once over the drug corpus, then query with arbitrary OCR output."""

    def __init__(
        self,
        ngram_range: tuple[int, int] = (2, 4),
        *,
        analyzer: str = "char_wb",
        sublinear_tf: bool = True,
        min_df: int = 1,
    ) -> None:
        self.config = NgramConfig(tuple(ngram_range), analyzer, sublinear_tf, min_df)
        self.names_: list[str] = []
        self.vectorizer_: TfidfVectorizer | None = None
        self.matrix_ = None   # scipy CSR [n_names, vocab], L2-normalised rows

    # ------------------------------------------------------------------ fit
    def fit(self, names: list[str]) -> NgramIndex:
        """Fit on the corpus. Duplicates and blanks are dropped, order kept."""
        seen: set[str] = set()
        clean: list[str] = []
        for n in names:
            k = n.strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            clean.append(k)
        if not clean:
            raise ValueError("cannot fit NgramIndex on an empty name list")

        self.vectorizer_ = TfidfVectorizer(
            analyzer=self.config.analyzer,
            ngram_range=self.config.ngram_range,
            sublinear_tf=self.config.sublinear_tf,
            min_df=self.config.min_df,
            lowercase=True,
            norm="l2",          # rows unit-length => cosine is a plain dot product
            dtype=np.float32,
        )
        self.matrix_ = self.vectorizer_.fit_transform(clean)
        self.names_ = clean
        return self

    @property
    def is_fitted(self) -> bool:
        return self.vectorizer_ is not None

    @property
    def vocabulary_size(self) -> int:
        self._check_fitted()
        return len(self.vectorizer_.vocabulary_)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("NgramIndex is not fitted — call fit(names) first")

    def _vector(self, text: str):
        """Transform one string with the *fitted* vectoriser (never refit)."""
        self._check_fitted()
        return self.vectorizer_.transform([text.strip().lower()])

    # -------------------------------------------------------------- queries
    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity in [0,1].

        Both arguments may be out-of-vocabulary. Because the vectoriser is only
        transformed, n-grams absent from the fitted vocabulary contribute
        nothing; if a string shares no known n-gram with the corpus its vector
        is all-zero and every similarity involving it is 0.0.
        """
        va, vb = self._vector(a), self._vector(b)
        if va.nnz == 0 or vb.nnz == 0:
            return 0.0
        return float(va.multiply(vb).sum())

    def topk(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-k corpus names by cosine similarity, descending.

        Zero-similarity names are omitted rather than padded in — a neighbour
        sharing no n-gram with the query is not a candidate, and returning it
        would only invite a downstream threshold to trust it.
        """
        self._check_fitted()
        if k <= 0:
            return []
        q = self._vector(query)
        if q.nnz == 0:
            return []

        sims = np.asarray((self.matrix_ @ q.T).todense()).ravel()
        k = min(k, sims.shape[0])
        # argpartition is O(n) vs O(n log n) for a full sort; the corpus is only
        # 2000 names today but this is called once per OCR hypothesis.
        cand = np.argpartition(-sims, k - 1)[:k]
        # Sort by score then name so ties are deterministic across runs/platforms.
        cand = sorted(cand, key=lambda i: (-float(sims[i]), self.names_[i]))
        return [(self.names_[i], float(sims[i])) for i in cand if sims[i] > 0.0]

    # ---------------------------------------------------------- persistence
    def save(self, path: Path) -> Path:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "format_version": _FORMAT_VERSION,
                "sklearn_version": sklearn.__version__,
                "config": asdict(self.config),
                "names": self.names_,
                "vectorizer": self.vectorizer_,
                "matrix": self.matrix_,
            },
            path,
            compress=3,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> NgramIndex:
        """Rebuild a fitted index. Use as ``idx = NgramIndex.load(p)``."""
        blob = joblib.load(Path(path))
        if blob.get("format_version") != _FORMAT_VERSION:
            raise ValueError(
                f"{path} has format_version {blob.get('format_version')}, "
                f"expected {_FORMAT_VERSION} — refit the index."
            )
        if blob.get("sklearn_version") != sklearn.__version__:
            # Pickled sklearn estimators are not version-portable; warn loudly
            # rather than fail, since minor bumps usually still load.
            print(f"warning: {path} was written with scikit-learn "
                  f"{blob.get('sklearn_version')}, running {sklearn.__version__}")
        cfg = blob["config"]
        idx = cls(
            tuple(cfg["ngram_range"]),
            analyzer=cfg["analyzer"],
            sublinear_tf=cfg["sublinear_tf"],
            min_df=cfg["min_df"],
        )
        idx.names_ = blob["names"]
        idx.vectorizer_ = blob["vectorizer"]
        idx.matrix_ = blob["matrix"]
        return idx


def build_from_corpus(csv_path: Path | None = None, **kwargs) -> NgramIndex:
    """Convenience: fit an index over ``data/processed/drug_names.csv``."""
    from lasa.common.paths import DRUG_NAMES_CSV
    from lasa.data.drug_names import load_csv

    csv_path = Path(csv_path) if csv_path else DRUG_NAMES_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Build it with: python -m lasa.data.drug_names"
        )
    return NgramIndex(**kwargs).fit([d.name for d in load_csv(csv_path)])


# OCR-plausible corruptions of real corpus names: dropped final letter,
# truncated brand name, dropped interior vowel. None of these are corpus names.
_DEMO_QUERIES: tuple[str, ...] = (
    "dobutamin", "celebra", "hydralzine", "losek", "metfornin", "amoxicillin",
)


def main() -> None:
    from lasa.common.seed import set_seed

    ap = argparse.ArgumentParser(
        description="Fit the char-n-gram index and show neighbours for noisy queries."
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--ngram-min", type=int, default=2)
    ap.add_argument("--ngram-max", type=int, default=4)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--queries", nargs="*", default=list(_DEMO_QUERIES))
    ap.add_argument("--save", type=Path, default=None,
                    help="write the fitted index to this path (joblib)")
    args = ap.parse_args()

    seed = set_seed(args.seed)
    idx = build_from_corpus(ngram_range=(args.ngram_min, args.ngram_max))

    print(f"seed={seed}  names={len(idx.names_)}  "
          f"ngram_range={idx.config.ngram_range}  vocab={idx.vocabulary_size}")

    for q in args.queries:
        hits = idx.topk(q, k=args.topk)
        in_corpus = " (in corpus)" if q.strip().lower() in set(idx.names_) else ""
        print(f"\n{q!r}{in_corpus}")
        if not hits:
            print("  <no candidate shares an n-gram with the corpus>")
        for rank, (name, score) in enumerate(hits, 1):
            print(f"  {rank}. {name:<22} {score:.3f}")

    if args.save:
        print(f"\nsaved -> {idx.save(args.save)}")


if __name__ == "__main__":
    main()
