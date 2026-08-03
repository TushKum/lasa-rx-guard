"""Stage-2 evaluation against the ISMP/FDA ground truth.

Four questions, each with its decision rule stated explicitly, because these are
the numbers the report leans on:

1. **Retrieval.** Corrupt a real drug name with an OCR-like noise model. Does
   the true drug come back in the top-k?

2. **Pair flagging (precision / recall / F1).** Defined as binary
   classification over queries:

   * *positive query* — a corrupted member of a documented LASA pair. Counts as
     a true positive only when the system both raises a flag **and** surfaces
     the documented partner in the top-k. Flagging without naming the partner
     is not useful to a pharmacist, so it is not scored as a success.
   * *negative query* — a corrupted drug name with **no** documented partner. A
     flag here is a false positive.

   Reported at two operating points: flag = RED only, and flag = RED-or-YELLOW.

3. **Negative control.** The false-positive rate on unrelated names, reported
   separately. Without it, a matcher that flags everything scores perfect recall
   and looks excellent.

4. **Ablation.** edit-only vs phonetic-only vs n-gram-only vs the fusion, so
   "was combining three signals worth it?" gets an answer rather than an
   assertion.

The noise model is seeded and its parameters are printed with the results. It
is a *proxy* for OCR error, not a measurement of this project's OCR — the
end-to-end number comes from feeding real CRNN/TrOCR output through the matcher
(see `lasa.evaluation.evaluate`).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from lasa.common.paths import DRUG_NAMES_CSV, REPORTS, ensure_dirs
from lasa.common.seed import set_seed
from lasa.data.drug_names import load_csv
from lasa.data.lasa_pairs import deduped_pairs, pair_lookup
from lasa.risk.matcher import RED, YELLOW, FusionWeights, LasaMatcher

# Letter pairs that a hand-written glyph plausibly degrades into. Derived from
# the usual handwriting confusions (closed vs open loops, ascender/descender
# ambiguity, minim strings) rather than keyboard adjacency, which models a
# completely different error process.
CONFUSIONS: dict[str, str] = {
    "a": "ou", "b": "hlk", "c": "eo", "d": "clta", "e": "co", "f": "tl",
    "g": "qy", "h": "bnk", "i": "lej", "j": "iy", "k": "hb", "l": "iet",
    "m": "nw", "n": "mru", "o": "ace", "p": "qg", "q": "gp", "r": "nv",
    "s": "z", "t": "fli", "u": "vnc", "v": "uyr", "w": "mv", "x": "ky",
    "y": "gvj", "z": "s",
}
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def corrupt(name: str, error_rate: float, rng: np.random.Generator) -> str:
    """Apply ~``error_rate`` edits per character, biased to visual confusions."""
    chars = list(name)
    n_edits = max(1, int(round(error_rate * len(chars))))
    for _ in range(n_edits):
        if not chars:
            break
        op = rng.choice(["sub", "sub", "sub", "del", "ins", "swap"])  # sub-heavy
        i = int(rng.integers(0, len(chars)))
        if op == "sub":
            near = CONFUSIONS.get(chars[i].lower(), ALPHABET)
            chars[i] = near[int(rng.integers(0, len(near)))]
        elif op == "del" and len(chars) > 2:
            chars.pop(i)
        elif op == "ins":
            near = CONFUSIONS.get(chars[i].lower(), ALPHABET)
            chars.insert(i, near[int(rng.integers(0, len(near)))])
        elif op == "swap" and i < len(chars) - 1:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


@dataclass
class RetrievalResult:
    error_rate: float
    n: int
    top1: float
    top3: float
    top5: float


def eval_retrieval(matcher: LasaMatcher, names: list[str], rates: list[float],
                   seed: int, k: int = 5) -> list[RetrievalResult]:
    out: list[RetrievalResult] = []
    for rate in rates:
        rng = np.random.default_rng(seed)
        hits = {1: 0, 3: 0, 5: 0}
        for name in names:
            q = corrupt(name, rate, rng)
            cands = [c.name for c in matcher.match(q, k=k).candidates]
            for kk in (1, 3, 5):
                if name in cands[:kk]:
                    hits[kk] += 1
        n = len(names)
        out.append(RetrievalResult(rate, n, hits[1] / n, hits[3] / n, hits[5] / n))
    return out


@dataclass
class FlagResult:
    rule: str            # "RED" | "RED+YELLOW"
    error_rate: float
    tp: int
    fn: int
    fp: int
    tn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def fp_rate(self) -> float:
        d = self.fp + self.tn
        return self.fp / d if d else 0.0


def eval_flagging(matcher: LasaMatcher, rates: list[float], seed: int,
                  k: int = 5, n_negatives: int = 300) -> list[FlagResult]:
    pairs = deduped_pairs()
    lookup = pair_lookup()
    all_names = [d.name for d in matcher.names]
    # Negatives: corpus names with no documented partner at all.
    no_partner = [n for n in all_names if n not in lookup]

    results: list[FlagResult] = []
    for rate in rates:
        rng = np.random.default_rng(seed)
        counts = {"RED": [0, 0, 0, 0], "RED+YELLOW": [0, 0, 0, 0]}  # tp fn fp tn

        # --- positives: one query per pair, per direction ---
        for p in pairs:
            for src, partner in ((p.a, p.b), (p.b, p.a)):
                if src not in matcher.by_name or partner not in matcher.by_name:
                    continue
                q = corrupt(src, rate, rng)
                a = matcher.match(q, k=k)
                surfaced = partner in {c.name for c in a.candidates}
                for rule, flagged in (("RED", a.risk == RED),
                                      ("RED+YELLOW", a.risk in (RED, YELLOW))):
                    if flagged and surfaced:
                        counts[rule][0] += 1
                    else:
                        counts[rule][1] += 1

        # --- negatives: names with no documented confusable ---
        rng_neg = np.random.default_rng(seed + 1)
        picks = rng_neg.choice(len(no_partner),
                               size=min(n_negatives, len(no_partner)), replace=False)
        for i in picks:
            q = corrupt(no_partner[int(i)], rate, rng_neg)
            a = matcher.match(q, k=k)
            for rule, flagged in (("RED", a.risk == RED),
                                  ("RED+YELLOW", a.risk in (RED, YELLOW))):
                if flagged:
                    counts[rule][2] += 1
                else:
                    counts[rule][3] += 1

        for rule, (tp, fn, fp, tn) in counts.items():
            results.append(FlagResult(rule, rate, tp, fn, fp, tn))
    return results


def eval_ablation(names: list[str], rate: float, seed: int,
                  k: int = 5) -> dict[str, dict]:
    """Retrieval + pair surfacing with each signal isolated."""
    configs = {
        "edit only": FusionWeights(edit=1.0, phonetic=0.0, ngram=0.0),
        "phonetic only": FusionWeights(edit=0.0, phonetic=1.0, ngram=0.0),
        "ngram only": FusionWeights(edit=0.0, phonetic=0.0, ngram=1.0),
        "fused (0.45/0.35/0.20)": FusionWeights(),
    }
    lookup = pair_lookup()
    out: dict[str, dict] = {}
    for label, w in configs.items():
        m = LasaMatcher.from_project(weights=w)
        rng = np.random.default_rng(seed)
        top1 = top3 = 0
        for name in names:
            q = corrupt(name, rate, rng)
            cands = [c.name for c in m.match(q, k=k).candidates]
            top1 += name == (cands[0] if cands else None)
            top3 += name in cands[:3]

        # partner surfacing on the ground-truth pairs
        rng2 = np.random.default_rng(seed)
        surf = tot = 0
        for p in deduped_pairs():
            for src, partner in ((p.a, p.b), (p.b, p.a)):
                if src not in m.by_name or partner not in m.by_name:
                    continue
                q = corrupt(src, rate, rng2)
                tot += 1
                surf += partner in {c.name for c in m.match(q, k=k).candidates}

        out[label] = {
            "top1": top1 / len(names), "top3": top3 / len(names),
            "partner_surfaced": surf / tot if tot else 0.0, "n": len(names),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the Stage-2 LASA engine.")
    ap.add_argument("--rates", type=float, nargs="*", default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--n-retrieval", type=int, default=400,
                    help="corpus names sampled for the retrieval test")
    ap.add_argument("--n-negatives", type=int, default=300)
    ap.add_argument("--ablation-rate", type=float, default=0.2)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--markdown", action="store_true")
    # Threshold overrides — the values in configs/risk.yaml are explicitly
    # PROVISIONAL, so the operating point has to be chosen from measurement.
    ap.add_argument("--red", type=float, default=None)
    ap.add_argument("--yellow", type=float, default=None)
    ap.add_argument("--margin", type=float, default=None)
    args = ap.parse_args()

    ensure_dirs()
    set_seed(args.seed)

    from lasa.risk.matcher import Thresholds
    t = Thresholds()
    if args.red is not None:
        t.red = args.red
    if args.yellow is not None:
        t.yellow = args.yellow
    if args.margin is not None:
        t.ambiguity_margin = args.margin
    print(f"thresholds: red={t.red} yellow={t.yellow} margin={t.ambiguity_margin}")
    matcher = LasaMatcher.from_project(thresholds=t)

    rng = np.random.default_rng(args.seed)
    all_names = [d.name for d in load_csv(DRUG_NAMES_CSV)]
    idx = rng.choice(len(all_names), size=min(args.n_retrieval, len(all_names)),
                     replace=False)
    sample = [all_names[int(i)] for i in idx]

    print(f"corpus {len(all_names)} names | {len(deduped_pairs())} documented pairs")
    print(f"retrieval sample {len(sample)} | negatives {args.n_negatives} | k={args.k}")
    print(f"noise model: seeded, visual-confusion-biased, "
          f"~rate edits/char, ops sub(x3)/del/ins/swap\n")

    retrieval = eval_retrieval(matcher, sample, args.rates, args.seed, args.k)
    flagging = eval_flagging(matcher, args.rates, args.seed, args.k, args.n_negatives)
    ablation = eval_ablation(sample, args.ablation_rate, args.seed, args.k)

    lines: list[str] = []
    lines.append("### Top-k retrieval under simulated OCR noise\n")
    lines.append("| err/char | n | Top-1 | Top-3 | Top-5 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in retrieval:
        lines.append(f"| {r.error_rate:.1f} | {r.n} | {r.top1:.1%} | "
                     f"{r.top3:.1%} | {r.top5:.1%} |")

    lines.append("\n### LASA pair flagging vs ISMP/FDA ground truth\n")
    lines.append("| rule | err/char | TP | FN | FP | TN | Precision | Recall | F1 | FP-rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for f in flagging:
        lines.append(f"| {f.rule} | {f.error_rate:.1f} | {f.tp} | {f.fn} | {f.fp} | "
                     f"{f.tn} | {f.precision:.1%} | {f.recall:.1%} | {f.f1:.1%} | "
                     f"{f.fp_rate:.1%} |")

    lines.append(f"\n### Signal ablation (error rate {args.ablation_rate})\n")
    lines.append("| variant | Top-1 | Top-3 | Partner surfaced |")
    lines.append("|---|---:|---:|---:|")
    for label, v in ablation.items():
        lines.append(f"| {label} | {v['top1']:.1%} | {v['top3']:.1%} | "
                     f"{v['partner_surfaced']:.1%} |")

    report = "\n".join(lines)
    print(report)

    out = REPORTS / "lasa_eval.json"
    out.write_text(json.dumps({
        "config": vars(args),
        "retrieval": [vars(r) for r in retrieval],
        "flagging": [vars(f) | {"precision": f.precision, "recall": f.recall,
                                "f1": f.f1, "fp_rate": f.fp_rate} for f in flagging],
        "ablation": ablation,
    }, indent=2))
    (REPORTS / "lasa_eval.md").write_text(report)
    print(f"\nwrote {out} and {REPORTS / 'lasa_eval.md'}")


if __name__ == "__main__":
    main()
