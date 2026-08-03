"""OCR metrics: CER, WER, exact-match accuracy.

CER is computed as a *corpus-level* micro-average — total edit distance over
total reference characters — not the mean of per-sample rates. The two differ,
and the micro-average is standard in the handwriting-recognition literature
(and is what jiwer reports). Averaging per-sample rates over-weights short
words, which for a drug-name corpus would flatter the model on 'lasix' and hide
failures on 'hydrochlorothiazide'.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import editdistance


@dataclass
class OcrMetrics:
    cer: float
    wer: float
    exact_match: float
    n: int
    total_ref_chars: int
    total_char_errors: int

    def as_dict(self) -> dict:
        return asdict(self)

    def as_row(self) -> str:
        return (f"CER {self.cer:6.2%} | WER {self.wer:6.2%} | "
                f"Exact {self.exact_match:6.2%} | n={self.n}")


def _norm(s: str) -> str:
    return s.strip().lower()


def char_error_rate(refs: list[str], hyps: list[str]) -> tuple[float, int, int]:
    total_dist = sum(editdistance.eval(_norm(r), _norm(h)) for r, h in zip(refs, hyps))
    total_chars = sum(len(_norm(r)) for r in refs)
    if total_chars == 0:
        return 0.0, 0, 0
    return total_dist / total_chars, total_dist, total_chars


def word_error_rate(refs: list[str], hyps: list[str]) -> float:
    """Each sample is a single word, so WER == 1 - exact-match at word level.

    Kept as a distinct function because the deliverable asks for WER explicitly,
    and because it generalises correctly if multi-word lines are added later.
    """
    total_words = 0
    total_errors = 0
    for r, h in zip(refs, hyps):
        rw, hw = _norm(r).split(), _norm(h).split()
        total_words += len(rw)
        total_errors += editdistance.eval(rw, hw)
    if total_words == 0:
        return 0.0
    return total_errors / total_words


def exact_match(refs: list[str], hyps: list[str]) -> float:
    if not refs:
        return 0.0
    return sum(_norm(r) == _norm(h) for r, h in zip(refs, hyps)) / len(refs)


def summarize(refs: list[str], hyps: list[str]) -> OcrMetrics:
    if len(refs) != len(hyps):
        raise ValueError(f"length mismatch: {len(refs)} refs vs {len(hyps)} hyps")
    cer, dist, chars = char_error_rate(refs, hyps)
    return OcrMetrics(
        cer=cer,
        wer=word_error_rate(refs, hyps),
        exact_match=exact_match(refs, hyps),
        n=len(refs),
        total_ref_chars=chars,
        total_char_errors=dist,
    )


def markdown_table(rows: dict[str, dict[str, OcrMetrics]]) -> str:
    """rows: {model_name: {split_name: OcrMetrics}} -> a markdown comparison."""
    splits: list[str] = []
    for per_split in rows.values():
        for s in per_split:
            if s not in splits:
                splits.append(s)

    head = "| Model | Split | CER ↓ | WER ↓ | Exact ↑ | n |"
    sep = "|---|---|---:|---:|---:|---:|"
    lines = [head, sep]
    for model, per_split in rows.items():
        for split in splits:
            m = per_split.get(split)
            if m is None:
                continue
            lines.append(
                f"| {model} | {split} | {m.cer:.2%} | {m.wer:.2%} | "
                f"{m.exact_match:.2%} | {m.n} |"
            )
    return "\n".join(lines)
