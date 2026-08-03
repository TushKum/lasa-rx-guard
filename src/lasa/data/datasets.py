"""Charset, Dataset and collate functions shared by CRNN and TrOCR.

Both models train on exactly the same samples in the same order for a given
seed. That is deliberate: the whole point of the comparison is the *model*, so
any difference in data ordering or split would confound the result.

The Dataset yields raw uint8 grayscale arrays and the label string. Model-
specific preprocessing (CTC height normalisation vs. the TrOCR feature
extractor's 384x384 RGB) lives in the collate functions, not here.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CRNN_HEIGHT = 32
CRNN_MAX_WIDTH = 512
# Horizontal stretch applied on top of aspect-preserving resize. See
# `resize_for_ctc` — without it ~48% of this corpus is CTC-infeasible.
WIDTH_SCALE = 2.0


# ----------------------------------------------------------------- charset --


@dataclass
class Charset:
    """Index 0 is reserved for the CTC blank, so real symbols start at 1."""

    chars: str

    BLANK: int = 0

    def __post_init__(self) -> None:
        self._to_idx = {c: i + 1 for i, c in enumerate(self.chars)}
        self._to_chr = {i + 1: c for i, c in enumerate(self.chars)}

    def __len__(self) -> int:
        return len(self.chars) + 1  # + blank

    def encode(self, text: str) -> list[int]:
        return [self._to_idx[c] for c in text if c in self._to_idx]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._to_chr.get(int(i), "") for i in ids)

    def filter(self, text: str) -> str:
        return "".join(c for c in text if c in self._to_idx)

    @classmethod
    def build(cls, texts: list[str]) -> "Charset":
        return cls("".join(sorted(set("".join(texts)))))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"chars": self.chars}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Charset":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["chars"])


# ----------------------------------------------------------------- dataset --


@dataclass
class Sample:
    path: Path
    text: str
    split: str
    source: str      # "synth" | "iam"
    font: str = ""
    # Stored pixel size, carried from the manifest so width-bucketed batching
    # can order samples without opening 48k files first.
    width: int = 0
    height: int = 0


def read_manifest(root: Path, splits: tuple[str, ...] | None = None,
                  source: str = "synth") -> list[Sample]:
    """Read a ``labels.tsv`` written by the synth or IAM pipeline."""
    labels = Path(root) / "labels.tsv"
    if not labels.exists():
        raise FileNotFoundError(
            f"{labels} not found — run `python -m lasa.data.synth` "
            f"(or `python -m lasa.data.iam`) first."
        )
    out: list[Sample] = []
    with labels.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if splits and r["split"] not in splits:
                continue
            out.append(
                Sample(Path(root) / r["path"], r["text"], r["split"],
                       source, r.get("font", ""),
                       int(r.get("width") or 0), int(r.get("height") or 0))
            )
    return out


class WordImageDataset(Dataset):
    """Grayscale word images + transcription.

    Returns ``{"image": np.uint8 [H,W], "text": str}``. No augmentation happens
    here — the synth pipeline bakes degradation into the stored files so that
    every run sees byte-identical inputs, which is what makes the CRNN/TrOCR
    comparison fair and the whole thing reproducible.
    """

    def __init__(self, samples: list[Sample], charset: Charset | None = None,
                 preload: bool = False, verbose: bool = True):
        self.samples = samples
        self.charset = charset
        self._cache: list[np.ndarray] | None = None
        if preload:
            self._preload(verbose)

    def _preload(self, verbose: bool) -> None:
        """Decode every image once into RAM.

        Training is otherwise PNG-decode bound: measured ~37 img/s from disk vs
        a >110 img/s compute ceiling on this model. The whole 48k-image train
        split is only ~450 MB of uint8, so paying the decode once at startup
        turns every later epoch into pure compute.
        """
        if verbose:
            print(f"preloading {len(self.samples)} images into RAM...", flush=True)
        cache: list[np.ndarray] = []
        for i, s in enumerate(self.samples):
            img = cv2.imread(str(s.path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"unreadable image: {s.path}")
            cache.append(img)
            if verbose and i and i % 10000 == 0:
                print(f"  {i}/{len(self.samples)}", flush=True)
        self._cache = cache
        if verbose:
            mb = sum(a.nbytes for a in cache) / 1e6
            print(f"  cached {len(cache)} images ({mb:.0f} MB)", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        if self._cache is not None:
            img = self._cache[i]
        else:
            img = cv2.imread(str(s.path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"unreadable image: {s.path}")
        text = self.charset.filter(s.text) if self.charset else s.text
        return {"image": img, "text": text, "path": str(s.path), "font": s.font}


# ---------------------------------------------------------------- collates --


def resize_for_ctc(img: np.ndarray, height: int = CRNN_HEIGHT,
                   max_width: int = CRNN_MAX_WIDTH,
                   width_scale: float = WIDTH_SCALE,
                   min_width: int = 8) -> np.ndarray:
    """Resize to the CRNN's input height, stretching the width.

    ``width_scale`` exists because plain aspect-preserving resize starves CTC.
    Images are stored at height 64; shrinking to 32 halves the width, and the
    CNN then divides width by 4, so a median 146px crop yields only 18 frames
    for a label that needs 2L+1 = 21. Measured on this corpus, aspect-preserving
    resize left **47.8%** of the training set CTC-infeasible — those samples
    return infinite loss, get zeroed by ``zero_infinity``, and contribute no
    gradient at all. The model duly collapses to predicting all blanks.

    A 2.0 stretch cancels the height reduction, restoring the frame budget.
    Horizontal distortion is harmless here and is standard practice in CRNN
    pipelines (fixed 32x100 inputs are common regardless of aspect).
    """
    h, w = img.shape
    new_w = int(round(w * (height / h) * width_scale))
    new_w = max(min_width, min(new_w, max_width))
    interp = cv2.INTER_AREA if new_w < w else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, height), interpolation=interp)


def min_feasible_width(text: str, downsample: int = 4) -> int:
    """Narrowest input width for which CTC can express ``text``.

    The lattice needs one frame per character plus one per potential blank
    separator: 2L+1 frames, hence downsample*(2L+1) pixels.
    """
    return downsample * (2 * len(text) + 1)


class CtcCollate:
    """Right-padded, width-variable CTC batches.

    ``downsample`` is the CRNN's total width stride; ``input_lengths`` is
    derived from each sample's true (unpadded) width so the loss is not told
    that padding contains signal.

    Implemented as a class rather than a closure because DataLoader workers are
    spawned (not forked) on macOS and Windows, and a nested function is not
    picklable — ``num_workers>0`` would die with
    "Can't get local object 'make_ctc_collate.<locals>.collate'".
    """

    def __init__(self, charset: Charset, height: int = CRNN_HEIGHT,
                 max_width: int = CRNN_MAX_WIDTH, downsample: int = 4):
        self.charset = charset
        self.height = height
        self.max_width = max_width
        self.downsample = downsample

    def __call__(self, batch: list[dict]) -> dict:
        charset, height, max_width = self.charset, self.height, self.max_width
        texts = [b["text"] for b in batch]

        # Hard guarantee: every sample gets at least 2L+1 frames. width_scale
        # fixes the systematic shortfall, but a short, unusually narrow crop of
        # a long name could still fall through — and an infeasible sample is not
        # a hard example, it is an invisible one (zero_infinity zeroes its
        # gradient), so it must be impossible by construction rather than rare.
        imgs = []
        for b, text in zip(batch, texts):
            floor = min_feasible_width(charset.filter(text), self.downsample)
            imgs.append(
                resize_for_ctc(b["image"], height, max_width,
                               min_width=max(8, min(floor, max_width)))
            )
        widths = [im.shape[1] for im in imgs]
        w_max = max(widths)

        # Pad with 255 (paper white) rather than 0 — zero-padding would look
        # like a solid ink block to the CNN.
        canvas = np.full((len(imgs), height, w_max), 255, np.uint8)
        for i, im in enumerate(imgs):
            canvas[i, :, : im.shape[1]] = im

        x = torch.from_numpy(canvas).float().div_(255.0).sub_(0.5).div_(0.5)
        x = x.unsqueeze(1)  # [B,1,H,W]

        encoded = [charset.encode(t) for t in texts]
        targets = torch.tensor([i for e in encoded for i in e], dtype=torch.long)
        target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
        input_lengths = torch.tensor(
            [max(1, w // self.downsample) for w in widths], dtype=torch.long
        )
        return {
            "images": x,
            "targets": targets,
            "target_lengths": target_lengths,
            "input_lengths": input_lengths,
            "texts": texts,
            "fonts": [b["font"] for b in batch],
        }


class TrocrCollate:
    """HuggingFace TrOCR: RGB pixel_values + label ids with padding masked.

    Class, not closure — see the note on :class:`CtcCollate`.
    """

    def __init__(self, processor, max_target_length: int = 32):
        self.processor = processor
        self.max_target_length = max_target_length

    def __call__(self, batch: list[dict]) -> dict:
        images = [cv2.cvtColor(b["image"], cv2.COLOR_GRAY2RGB) for b in batch]
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values

        texts = [b["text"] for b in batch]
        labels = self.processor.tokenizer(
            texts, padding="max_length", max_length=self.max_target_length,
            truncation=True, return_tensors="pt",
        ).input_ids
        # -100 is ignored by the CE loss; without this the model is rewarded
        # for predicting padding.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels, "texts": texts,
                "fonts": [b["font"] for b in batch]}


# Factory aliases — the training scripts are written against these names.
def make_ctc_collate(charset: Charset, height: int = CRNN_HEIGHT,
                     max_width: int = CRNN_MAX_WIDTH, downsample: int = 4) -> CtcCollate:
    return CtcCollate(charset, height, max_width, downsample)


def make_trocr_collate(processor, max_target_length: int = 32) -> TrocrCollate:
    return TrocrCollate(processor, max_target_length)


def build_charset_from(samples: list[Sample]) -> Charset:
    return Charset.build([s.text for s in samples])
