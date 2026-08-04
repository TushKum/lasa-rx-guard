"""End-to-end inference: prescription image -> OCR -> LASA risk assessment.

Word segmentation here is a documented approximation, not a solved problem. It
uses horizontal projection to find text lines and then whitespace gaps to split
words — which works on a reasonably flat, reasonably horizontal photo of a chit
and degrades on skew, multi-column layouts, or a drug name written across two
lines. Real prescription layout analysis is its own research problem; this is
enough to demo the pipeline end to end, and its failures are visible in the
demo's word-crop strip rather than hidden.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from lasa.common.paths import CHECKPOINTS
from lasa.data.datasets import CRNN_HEIGHT, CRNN_MAX_WIDTH, Charset, make_ctc_collate
from lasa.risk.matcher import LasaMatcher, RiskAssessment


def find_crnn_checkpoint() -> Path | None:
    """Locate a usable CRNN checkpoint, preferring the full training one.

    ``best.pt`` carries optimizer state and is ~96 MB, so it is gitignored;
    ``crnn_inference.pt`` is the weights-only export (~32 MB) that ships with the
    repo so a cloud deploy has a working model. Checked in that order because a
    local training run should win over the committed snapshot.
    """
    base = CHECKPOINTS / "crnn_ctc"
    for name in ("best.pt", "crnn_inference.pt", "last.pt"):
        p = base / name
        if p.exists():
            return p
    return None


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class WordResult:
    text: str
    confidence: float
    box: tuple[int, int, int, int]           # x, y, w, h
    assessment: RiskAssessment | None = None


@dataclass
class PageResult:
    words: list[WordResult] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def worst_risk(self) -> str:
        order = {"GREEN": 0, "YELLOW": 1, "RED": 2}
        risks = [w.assessment.risk for w in self.words if w.assessment]
        return max(risks, key=lambda r: order[r]) if risks else "GREEN"


def segment_words(image: np.ndarray, min_area: int = 120) -> list[tuple[int, int, int, int]]:
    """Return word boxes (x, y, w, h), reading order top-to-bottom, left-to-right."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Dilate horizontally so the letters of one word merge while neighbouring
    # words stay separate. The kernel must scale with LETTER height, which is
    # estimated from the ink itself — not from image height. Those two are
    # similar on a full page but wildly different on a tight single-word crop,
    # where image height *is* letter height; scaling off the image split every
    # pre-cropped word into three fragments.
    letters, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(c)[3] for c in letters if cv2.contourArea(c) > 4]
    letter_h = float(np.median(heights)) if heights else gray.shape[0] * 0.5

    kw = max(3, int(round(letter_h * 0.8)))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, hh = cv2.boundingRect(c)
        if w * hh >= min_area and w >= 8 and hh >= 8:
            boxes.append((x, y, w, hh))
    if not boxes:
        return []

    # Group into lines by vertical overlap, then order within each line.
    boxes.sort(key=lambda b: b[1])
    lines: list[list[tuple]] = [[boxes[0]]]
    for b in boxes[1:]:
        ref = lines[-1][0]
        if b[1] < ref[1] + ref[3] * 0.6:
            lines[-1].append(b)
        else:
            lines.append([b])
    out: list[tuple[int, int, int, int]] = []
    for line in lines:
        out.extend(sorted(line, key=lambda b: b[0]))
    return out


class RxPipeline:
    def __init__(self, backend: str = "crnn", checkpoint: Path | None = None,
                 device: str = "auto", matcher: LasaMatcher | None = None):
        self.backend = backend
        self.device = _device(device)
        self.matcher = matcher or LasaMatcher.from_project()
        self.model = None
        self.charset = None
        self.processor = None
        if backend != "none":
            self._load(checkpoint)

    def _load(self, checkpoint: Path | None) -> None:
        if self.backend == "crnn":
            ckpt = checkpoint or find_crnn_checkpoint()
            if ckpt is None or not Path(ckpt).exists():
                raise FileNotFoundError(
                    f"No CRNN checkpoint under {CHECKPOINTS / 'crnn_ctc'}. "
                    f"Train one first:\n"
                    f"  python -m lasa.train.train_crnn\n"
                    f"Refusing to run with random weights — the output would look "
                    f"like a real reading but be meaningless."
                )
            from lasa.models.crnn import CRNN
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            self.charset = Charset(ck["charset"])
            m = CRNN(n_classes=len(self.charset), img_h=CRNN_HEIGHT)
            m.load_state_dict(ck["model"])
            self.model = m.to(self.device).eval()
        elif self.backend == "trocr":
            ckpt = checkpoint or CHECKPOINTS / "trocr_ft"
            if not (Path(ckpt) / "config.json").exists():
                raise FileNotFoundError(
                    f"No fine-tuned TrOCR at {ckpt}. Train one first:\n"
                    f"  python -m lasa.train.train_trocr"
                )
            from lasa.models.trocr import load_trocr
            self.model, self.processor = load_trocr(str(ckpt), self.device)
        else:
            raise ValueError(f"unknown backend {self.backend!r}")

    # ------------------------------------------------------------ reading --

    @torch.no_grad()
    def read_word(self, crop: np.ndarray) -> tuple[str, float]:
        if self.backend == "crnn":
            from lasa.models.crnn import greedy_decode
            batch = make_ctc_collate(self.charset, CRNN_HEIGHT, CRNN_MAX_WIDTH)(
                [{"image": crop, "text": "", "font": ""}]
            )
            lp = self.model(batch["images"].to(self.device)).cpu()
            text = greedy_decode(lp, self.charset, batch["input_lengths"])[0]
            # Mean per-frame max posterior over non-blank frames: a rough but
            # honest confidence. Frames the model calls blank are excluded, or
            # a long padded crop would inflate the score.
            probs = lp.softmax(-1)[:, 0, :]
            top, idx = probs.max(-1)
            keep = idx != Charset.BLANK
            conf = float(top[keep].mean()) if keep.any() else 0.0
            return text, conf
        rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        pv = self.processor(images=[rgb], return_tensors="pt").pixel_values.to(self.device)
        out = self.model.generate(pv, num_beams=4, output_scores=True,
                                  return_dict_in_generate=True)
        text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
        score = getattr(out, "sequences_scores", None)
        conf = float(np.exp(score[0].item())) if score is not None else 0.75
        return text.strip().lower(), min(max(conf, 0.0), 1.0)

    def run(self, image: np.ndarray, k: int = 3) -> PageResult:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        res = PageResult()

        t0 = time.perf_counter()
        boxes = segment_words(gray)
        res.timings_ms["segment"] = (time.perf_counter() - t0) * 1000

        t_ocr = t_risk = 0.0
        for (x, y, w, h) in boxes:
            crop = gray[y:y + h, x:x + w]
            t0 = time.perf_counter()
            text, conf = self.read_word(crop)
            t_ocr += (time.perf_counter() - t0) * 1000
            if not text:
                continue
            t0 = time.perf_counter()
            assessment = self.matcher.match(text, k=k, ocr_confidence=conf)
            t_risk += (time.perf_counter() - t0) * 1000
            res.words.append(WordResult(text, conf, (x, y, w, h), assessment))

        res.timings_ms["ocr"] = t_ocr
        res.timings_ms["risk"] = t_risk
        res.timings_ms["total"] = sum(res.timings_ms.values())
        res.timings_ms["per_word"] = (
            res.timings_ms["total"] / max(len(res.words), 1)
        )
        return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Read a prescription image and flag LASA risk.")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--backend", default="crnn", choices=["crnn", "trocr"])
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    img = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"cannot read image: {args.image}")

    pipe = RxPipeline(args.backend, args.checkpoint, args.device)
    res = pipe.run(img, k=args.k)

    print(f"{len(res.words)} word(s) detected | overall risk: {res.worst_risk}")
    for w in res.words:
        print(f"\n[{w.box}] OCR {w.text!r} (conf {w.confidence:.2f})")
        if w.assessment:
            print(w.assessment.render())
    print("\ntimings (ms): " + ", ".join(f"{k}={v:.1f}" for k, v in res.timings_ms.items()))
    print("\nAdvisory only — a pharmacist must make the final dispensing decision.")


if __name__ == "__main__":
    main()
