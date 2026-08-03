"""Stage 2 — LASA confusion risk engine.

Given an OCR hypothesis, decide which drug was probably intended and how
dangerous it would be to get that wrong.

Three similarity signals are fused (weights from ``configs/risk.yaml``):

* **edit** — Jaro-Winkler + Damerau-Levenshtein. Handwriting/OCR errors are
  character-shape errors, so this is the dominant signal. Jaro-Winkler leads
  because it up-weights a shared prefix, and handwriting degrades toward the end
  of a word far more than at the start.
* **phonetic** — Double Metaphone / NYSIIS / Soundex, compared by *similarity*
  rather than equality. Catches the sound-alike half of LASA, which edit
  distance ranks too low (``celebrex``/``celexa``).
* **n-gram** — character trigram TF-IDF cosine. Robust to OCR noise and handles
  strings that are not real words at all.

Two design decisions worth stating plainly, because they shape the metrics:

1. **Documented LASA partners of the top match are force-injected into the
   candidate list** (``via="known-pair"``), even when pure similarity would not
   have retrieved them. This is deliberate: the system already knows these pairs
   are dangerous, and failing to surface a published confusable because it
   scored 4th would be indefensible in a dispensing context. Candidates carry a
   ``via`` field so the evaluation can report similarity-retrieval and
   known-pair-surfacing separately — reporting them merged would flatter the
   retrieval numbers.

2. **The traffic light keys off the ambiguity between the top two candidates,
   not off how good the best match is.** A confident, unambiguous read of a drug
   that happens to have a confusable twin elsewhere in the formulary is not an
   emergency; an unresolvable choice between two similar names is. So the bands
   apply to (a) how confusable the two drugs are with each other and (b) how
   little the OCR string separates them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path

from rapidfuzz import fuzz, process
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler

from lasa.data.drug_names import DrugName, load_csv
from lasa.data.lasa_pairs import LasaPair, deduped_pairs, pair_lookup
from lasa.risk.ngram import NgramIndex
from lasa.risk.phonetics import phonetic_similarity

RED, YELLOW, GREEN = "RED", "YELLOW", "GREEN"


@dataclass
class FusionWeights:
    edit: float = 0.45
    phonetic: float = 0.35
    ngram: float = 0.20

    def __post_init__(self) -> None:
        total = self.edit + self.phonetic + self.ngram
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"fusion weights must sum to 1.0, got {total}")


@dataclass
class Thresholds:
    red: float = 0.80
    # CALIBRATED, not guessed. configs/risk.yaml shipped a provisional 0.62;
    # measured against the 164 ground-truth pairs plus 200 no-partner controls
    # at 0.2 errors/char, 0.62 produced a 36% false-positive rate on unrelated
    # names. Raising the band to 0.78 cut that to 15.5% while recall held at
    # 97.9% (F1 89.0% -> 94.4%). A third of prescriptions raising an alert is
    # alarm fatigue, which is a well-documented way clinical alerting fails.
    yellow: float = 0.78
    # How close the top two fused scores must be before the OCR string is
    # considered unable to separate them.
    ambiguity_margin: float = 0.10
    high_alert_shift: float = 0.08     # lowers both bands for high-alert drugs
    min_ocr_confidence: float = 0.55
    # Floor on the fused score of the BEST match. Below this, nothing in the
    # formulary really resembles the reading, so the honest answer is "I don't
    # know" — which must never render as GREEN. Without this, garbage input
    # ("xyzzyqqq" -> "xyzal" at 0.51) passes as a clean, safe match.
    min_match_confidence: float = 0.65


@dataclass
class Candidate:
    name: str
    display: str
    score: float          # fused similarity to the OCR string
    edit_sim: float
    phonetic_sim: float
    ngram_sim: float
    is_known_lasa: bool = False   # documented confusable of the top candidate
    tall_man: str = ""
    high_alert: bool = False
    via: str = "similarity"       # "similarity" | "known-pair"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RiskAssessment:
    query: str
    candidates: list[Candidate]
    risk: str
    reasons: list[str] = field(default_factory=list)
    best_match: str = ""
    confidence: float = 0.0
    margin: float = 1.0
    pair_similarity: float = 0.0
    requires_confirmation: bool = False
    ocr_confidence: float | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["candidates"] = [c.as_dict() for c in self.candidates]
        return d

    def render(self) -> str:
        icon = {RED: "[RED]", YELLOW: "[YELLOW]", GREEN: "[GREEN]"}[self.risk]
        lines = [f"{icon} OCR read {self.query!r} -> best match: {self.best_match!r} "
                 f"(confidence {self.confidence:.2f}, margin {self.margin:.2f})"]
        for i, c in enumerate(self.candidates, 1):
            tm = f"  [{c.tall_man}]" if c.tall_man else ""
            flags = []
            if c.is_known_lasa:
                flags.append("KNOWN-LASA")
            if c.high_alert:
                flags.append("HIGH-ALERT")
            if c.via == "known-pair":
                flags.append("via-pair-list")
            tag = ("  " + " ".join(flags)) if flags else ""
            lines.append(f"   {i}. {c.display:<22} {c.score:.3f} "
                         f"(edit {c.edit_sim:.2f} / phon {c.phonetic_sim:.2f} / "
                         f"ngram {c.ngram_sim:.2f}){tm}{tag}")
        for r in self.reasons:
            lines.append(f"   ! {r}")
        return "\n".join(lines)


def edit_similarity(a: str, b: str, jw_w: float = 0.6, dl_w: float = 0.4) -> float:
    jw = JaroWinkler.similarity(a, b)
    dl = DamerauLevenshtein.normalized_similarity(a, b)
    return jw_w * jw + dl_w * dl


class LasaMatcher:
    def __init__(
        self,
        names: list[DrugName],
        pairs: list[LasaPair],
        weights: FusionWeights | None = None,
        thresholds: Thresholds | None = None,
        ngram_index: NgramIndex | None = None,
        prefilter_limit: int = 60,
        prefilter_cutoff: float = 50.0,
    ):
        self.names = names
        self.by_name = {d.name: d for d in names}
        self.name_list = [d.name for d in names]
        self.weights = weights or FusionWeights()
        self.thresholds = thresholds or Thresholds()
        self.prefilter_limit = prefilter_limit
        self.prefilter_cutoff = prefilter_cutoff

        self.pairs = pairs
        self.partners = pair_lookup()
        self.high_alert: set[str] = set()
        self.tall_man: dict[str, str] = {}
        for p in pairs:
            if p.high_alert():
                self.high_alert.update((p.a, p.b))
            if p.a_tall_man:
                self.tall_man[p.a] = p.a_tall_man
            if p.b_tall_man:
                self.tall_man[p.b] = p.b_tall_man

        self.ngram = ngram_index or NgramIndex().fit(self.name_list)

    @classmethod
    def from_project(cls, **kwargs) -> "LasaMatcher":
        from lasa.common.paths import DRUG_NAMES_CSV
        return cls(load_csv(DRUG_NAMES_CSV), deduped_pairs(), **kwargs)

    # ------------------------------------------------------------ scoring --

    def _score(self, query: str, name: str) -> tuple[float, float, float, float]:
        e = edit_similarity(query, name)
        p = phonetic_similarity(query, name)
        n = self.ngram.similarity(query, name)
        w = self.weights
        return (w.edit * e + w.phonetic * p + w.ngram * n), e, p, n

    def _shortlist(self, query: str) -> list[str]:
        """Cheap RapidFuzz prefilter — 2000 names is small, this is about demo
        latency, not feasibility. Falls back to the full corpus if the cutoff
        rejects everything, so a badly garbled read still gets an answer."""
        hits = process.extract(
            query, self.name_list, scorer=fuzz.WRatio,
            limit=self.prefilter_limit, score_cutoff=self.prefilter_cutoff,
        )
        shortlist = [h[0] for h in hits]
        return shortlist or list(self.name_list)

    def _make_candidate(self, query: str, name: str, via: str = "similarity") -> Candidate:
        score, e, p, n = self._score(query, name)
        d = self.by_name.get(name)
        return Candidate(
            name=name,
            display=d.display if d else name,
            score=score, edit_sim=e, phonetic_sim=p, ngram_sim=n,
            tall_man=self.tall_man.get(name, ""),
            high_alert=name in self.high_alert,
            via=via,
        )

    # ------------------------------------------------------------ matching --

    def match(self, ocr_text: str, k: int = 3,
              ocr_confidence: float | None = None) -> RiskAssessment:
        query = (ocr_text or "").strip().lower()
        if not query:
            return RiskAssessment(query=ocr_text or "", candidates=[], risk=YELLOW,
                                  reasons=["Empty OCR result — nothing to match."],
                                  requires_confirmation=True,
                                  ocr_confidence=ocr_confidence)

        scored = [self._make_candidate(query, n) for n in self._shortlist(query)]
        scored.sort(key=lambda c: -c.score)
        if not scored:
            return RiskAssessment(query=query, candidates=[], risk=YELLOW,
                                  reasons=["No candidate drug matched."],
                                  requires_confirmation=True,
                                  ocr_confidence=ocr_confidence)

        best = scored[0]
        partners = self.partners.get(best.name, set())

        # Force documented partners of the top match into the list even if
        # similarity did not retrieve them (see module docstring).
        present = {c.name for c in scored[:k]}
        injected: list[Candidate] = []
        for partner in sorted(partners):
            if partner in present or partner not in self.by_name:
                continue
            existing = next((c for c in scored if c.name == partner), None)
            cand = existing or self._make_candidate(query, partner)
            cand.via = "similarity" if existing and existing in scored[:k] else "known-pair"
            injected.append(cand)

        for c in scored:
            c.is_known_lasa = c.name in partners
        for c in injected:
            c.is_known_lasa = True

        # Injected partners are appended and NEVER truncated away. They score
        # low by construction — that is exactly why similarity missed them — so
        # merging and re-sorting into a k-sized list would silently drop them
        # again and make the injection a no-op.
        sim_top = scored[:k]
        sim_top.sort(key=lambda c: -c.score)
        injected.sort(key=lambda c: -c.score)
        top = sim_top + injected

        # The rival that drives the risk decision is the *most dangerous*
        # alternative, not merely the next-best-scoring one. A documented
        # confusable outranks a closer-but-undocumented neighbour, because the
        # published pair is evidence of real-world harm.
        alternatives = [c for c in top if c.name != best.name]
        pair_rival = next((c for c in alternatives if c.name in partners), None)
        score_rival = alternatives[0] if alternatives else None
        rival = pair_rival or score_rival

        margin = (best.score - rival.score) if rival else 1.0
        pair_sim = (self._score(best.name, rival.name)[0] if rival else 0.0)

        risk, reasons, needs_confirm = self._decide(
            best, rival, margin, pair_sim, ocr_confidence
        )

        # A close undocumented neighbour is still an ambiguity worth surfacing,
        # even when a documented pair took priority above.
        if (pair_rival and score_rival and score_rival is not pair_rival
                and (best.score - score_rival.score) < self.thresholds.ambiguity_margin):
            reasons.append(
                f"Also ambiguous with '{score_rival.name}' "
                f"(margin {best.score - score_rival.score:.2f}), not a documented pair."
            )

        return RiskAssessment(
            query=query, candidates=top, risk=risk, reasons=reasons,
            best_match=best.name, confidence=best.score, margin=margin,
            pair_similarity=pair_sim, requires_confirmation=needs_confirm,
            ocr_confidence=ocr_confidence,
        )

    def _decide(self, best: Candidate, rival: Candidate | None, margin: float,
                pair_sim: float, ocr_conf: float | None
                ) -> tuple[str, list[str], bool]:
        t = self.thresholds
        reasons: list[str] = []

        if rival is None:
            return GREEN, ["Single clear match; no confusable alternative found."], False

        high_alert = best.high_alert or rival.high_alert
        shift = t.high_alert_shift if high_alert else 0.0
        red_band = t.red - shift
        yellow_band = t.yellow - shift

        known_pair = rival.name in self.partners.get(best.name, set())
        ambiguous = margin < t.ambiguity_margin
        low_conf = (ocr_conf is not None and ocr_conf < t.min_ocr_confidence)

        risk = GREEN

        if known_pair:
            src = next((p.source for p in self.pairs
                        if {p.a, p.b} == {best.name, rival.name}), "ISMP/FDA")
            tm = ""
            if best.tall_man and rival.tall_man:
                tm = f" Recommended tall-man form: {best.tall_man} vs {rival.tall_man}."
            if ambiguous:
                risk = RED
                reasons.append(
                    f"'{best.name}' and '{rival.name}' are a {src}-listed confusable "
                    f"pair, and the OCR reading does not separate them "
                    f"(margin {margin:.2f}).{tm}"
                )
            else:
                risk = YELLOW
                reasons.append(
                    f"'{best.name}' has a {src}-listed confusable, '{rival.name}'. "
                    f"The reading favours '{best.name}' (margin {margin:.2f}), "
                    f"but confirm before dispensing.{tm}"
                )

        elif pair_sim >= red_band and ambiguous:
            risk = RED
            reasons.append(
                f"'{best.name}' and '{rival.name}' are highly similar "
                f"(similarity {pair_sim:.2f}) and the reading cannot separate them "
                f"(margin {margin:.2f}), though they are not on the published list."
            )
        elif pair_sim >= yellow_band or ambiguous:
            risk = YELLOW
            reasons.append(
                f"'{rival.name}' is a plausible alternative reading "
                f"(similarity to '{best.name}' {pair_sim:.2f}, margin {margin:.2f})."
            )

        if high_alert and risk == YELLOW:
            risk = RED
            ha = best.name if best.high_alert else rival.name
            reasons.append(
                f"Escalated: '{ha}' is an ISMP high-alert medication, where a "
                f"mix-up carries a markedly higher chance of serious harm."
            )

        if low_conf:
            if risk == GREEN:
                risk = YELLOW
            reasons.append(
                f"OCR confidence {ocr_conf:.2f} is below {t.min_ocr_confidence:.2f} — "
                f"the transcription itself may be wrong, independent of similarity."
            )

        # Weak best match: no drug in the formulary genuinely resembles the read.
        if best.score < t.min_match_confidence:
            if risk == GREEN:
                risk = YELLOW
            reasons.append(
                f"No confident formulary match — best candidate '{best.name}' scored "
                f"only {best.score:.2f} (floor {t.min_match_confidence:.2f}). Treat "
                f"this as unread rather than as '{best.name}'; re-scan or confirm "
                f"the prescription manually."
            )

        if risk == GREEN and not reasons:
            reasons.append(
                f"'{best.name}' matched clearly (margin {margin:.2f}); no documented "
                f"or high-similarity confusable nearby."
            )

        return risk, reasons, risk in (RED, YELLOW)


def main() -> None:
    ap = argparse.ArgumentParser(description="Demo the LASA risk matcher.")
    ap.add_argument("--query", nargs="*", default=None)
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    m = LasaMatcher.from_project()
    queries = args.query or [
        "dobutamin",     # high-alert, near-perfect LASA partner
        "hydralzine",    # tall-man pair
        "celebra",       # brand confusion
        "losec",         # exact, but has a documented partner
        "amoxicilin",    # simple OCR typo, no dangerous twin
        "paracetamol",   # clean, unambiguous
        "xyzzyqqq",      # garbage input
    ]
    print(f"corpus {len(m.names)} names | {len(m.pairs)} documented pairs\n")
    for q in queries:
        print(m.match(q, k=args.k).render())
        print()


if __name__ == "__main__":
    main()
