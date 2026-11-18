"""Turn phone photos of filled collection forms into labelled word crops.

Pipeline: detect the four ArUco markers -> homography onto the canonical form
-> crop each cell by its normalised box from forms_manifest.json -> trim to ink
-> emit labels.tsv in exactly the schema the synthetic pipeline uses, so real
data drops into training without touching the loaders.

Photo filenames carry the provenance: ``<writercode>_<formid>.jpg``, e.g.
``W03_LRG-001.jpg``. The writer code is an opaque identifier assigned on the
consent form; it is never a name. Splits are assigned BY WRITER, not by sample
-- a writer-dependent split would inflate results exactly where our synthetic
experiments showed the model is weakest.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

CANON_W, CANON_H = 2000, 2800     # canonical frame (marker-centre rectangle)
TARGET_H = 64                     # must match lasa.data.synth.TARGET_HEIGHT
# Blank detection is deliberately NOT a bare ink-fraction threshold. A short
# word written with a fine pen in a large box covers very little area -- in
# testing, genuine writing ('losec', 'cisplatin') sat at 0.0026-0.0038 while a
# 0.004 cut-off silently discarded it. We instead drop speck-sized connected
# components and then require the surviving ink to span a plausible extent.
SPECK_AREA_FRAC = 6e-5            # components smaller than this are dust/noise
MIN_INK_FRAC = 6e-4               # floor on surviving ink
MIN_SPAN_W, MIN_SPAN_H = 0.05, 0.10   # ink bbox must span this much of the cell
NAME_RE = re.compile(r"^(?P<writer>[A-Za-z0-9]+)_(?P<form>LRG-\d{3})\.(jpg|jpeg|png)$", re.I)


def _detect_once(img: np.ndarray, scale: float) -> dict:
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    # A phone photo of paper is low-contrast and slightly blurred; widen the
    # adaptive-threshold sweep rather than relying on the default window.
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 6
    params.minMarkerPerimeterRate = 0.01
    det = cv2.aruco.ArucoDetector(d, params)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    corners, ids, _ = det.detectMarkers(img)
    if ids is None:
        return {}
    return {int(i): c[0].mean(axis=0) / scale for i, c in zip(ids.flatten(), corners)}


def detect_frame(img_gray: np.ndarray, report: dict | None = None):
    """Return marker centres ordered [TL, TR, BR, BL], or None.

    Tries progressively harder: raw, contrast-equalised, then upscaled. Field
    photos are downscaled and JPEG-compressed, which costs marker contrast, and
    a single detector pass loses roughly a third of sheets.
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    variants = [(img_gray, 1.0), (clahe.apply(img_gray), 1.0),
                (clahe.apply(img_gray), 2.0), (img_gray, 2.0)]
    best: dict = {}
    for im, sc in variants:
        found = _detect_once(im, sc)
        if len(found) > len(best):
            best = found
        if {0, 1, 2, 3} <= set(best):
            break
    if report is not None:
        report["markers_found"] = sorted(best)
    if not {0, 1, 2, 3} <= set(best):
        return None
    return np.float32([best[0], best[1], best[2], best[3]])


def rectify(img_gray: np.ndarray, src: np.ndarray) -> np.ndarray:
    dst = np.float32([[0, 0], [CANON_W, 0], [CANON_W, CANON_H], [0, CANON_H]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img_gray, M, (CANON_W, CANON_H),
                               flags=cv2.INTER_CUBIC, borderValue=255)


def binarise(cell: np.ndarray) -> np.ndarray:
    # Adaptive: a phone photo has uneven illumination across the sheet.
    return cv2.adaptiveThreshold(cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 12)


def extract_cell(canon: np.ndarray, box, inset: float = 0.045):
    """Crop one cell, drop the printed rule, trim to ink. None if blank."""
    x0, y0, x1, y1 = box
    X0, Y0 = int(x0 * CANON_W), int(y0 * CANON_H)
    X1, Y1 = int(x1 * CANON_W), int(y1 * CANON_H)
    dx, dy = int((X1 - X0) * inset), int((Y1 - Y0) * inset)
    cell = canon[Y0 + dy:Y1 - dy, X0 + dx:X1 - dx]
    if cell.size == 0:
        return None

    mask = binarise(cell)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))

    # Keep only components big enough to be pen strokes; this both rejects dust
    # and stops a speck from inflating the crop's bounding box.
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cell_area = cell.shape[0] * cell.shape[1]
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= SPECK_AREA_FRAC * cell_area]
    if not keep:
        return None
    ink = np.isin(lab, keep)
    if ink.mean() < MIN_INK_FRAC:
        return None

    ys, xs = np.nonzero(ink)
    span_w = (xs.max() - xs.min() + 1) / cell.shape[1]
    span_h = (ys.max() - ys.min() + 1) / cell.shape[0]
    if span_w < MIN_SPAN_W or span_h < MIN_SPAN_H:
        return None

    m = int(0.06 * cell.shape[0])
    y0c, y1c = max(0, ys.min() - m), min(cell.shape[0], ys.max() + m + 1)
    x0c, x1c = max(0, xs.min() - m), min(cell.shape[1], xs.max() + m + 1)
    crop = cell[y0c:y1c, x0c:x1c]
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return None

    s = TARGET_H / crop.shape[0]
    return cv2.resize(crop, (max(8, int(round(crop.shape[1] * s))), TARGET_H),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def assign_splits(writers: list[str], seed: int, val_frac=0.15, test_frac=0.25) -> dict:
    """Writer-independent split. Style generalisation is the measured weakness,
    so the test set must contain hands never seen in training."""
    rng = np.random.default_rng(seed)
    w = sorted(set(writers))
    rng.shuffle(w)
    n_test = max(1, int(round(len(w) * test_frac)))
    n_val = max(1, int(round(len(w) * val_frac))) if len(w) > 2 else 0
    out = {}
    for i, name in enumerate(w):
        out[name] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def main() -> None:
    from lasa.common.paths import ROOT

    ap = argparse.ArgumentParser(description="Ingest photographed collection forms.")
    ap.add_argument("--photos", type=Path, required=True,
                    help="directory of <writer>_<formid>.jpg photos")
    ap.add_argument("--forms", type=Path, default=ROOT / "data" / "forms")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "real")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    man = json.loads((args.forms / "forms_manifest.json").read_text())
    cells_by_form = {f["form_id"]: f["cells"] for f in man["forms"]}

    photos = sorted(p for p in args.photos.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    jobs, skipped = [], []
    for p in photos:
        m = NAME_RE.match(p.name)
        if not m:
            skipped.append((p.name, "filename must be <writer>_<formid>.jpg"))
            continue
        if m["form"] not in cells_by_form:
            skipped.append((p.name, f"unknown form {m['form']}"))
            continue
        jobs.append((p, m["writer"].upper(), m["form"]))

    splits = assign_splits([w for _, w, _ in jobs], args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, stats = [], {"forms": 0, "cells": 0, "blank": 0, "undetected": 0}

    for path, writer, form in jobs:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped.append((path.name, "unreadable image")); continue
        rep: dict = {}
        src = detect_frame(img, rep)
        if src is None:
            stats["undetected"] += 1
            got = rep.get("markers_found", [])
            skipped.append((path.name,
                            f"found markers {got} of [0, 1, 2, 3] -- reshoot with the "
                            f"whole sheet in frame and even lighting"))
            continue
        canon = rectify(img, src)
        stats["forms"] += 1
        split = splits[writer]
        for c in cells_by_form[form]:
            crop = extract_cell(canon, c["box"])
            if crop is None:
                stats["blank"] += 1; continue
            rel = f"{split}/{writer}_{form}_{c['index']:02d}.png"
            out_p = args.out / rel
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), crop)
            rows.append([split, rel, c["text"], writer, "real",
                         crop.shape[1], crop.shape[0]])
            stats["cells"] += 1

    with (args.out / "labels.tsv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["split", "path", "text", "font", "tier", "width", "height"])
        w.writerows(rows)

    per_split = {}
    for r in rows:
        per_split[r[0]] = per_split.get(r[0], 0) + 1
    (args.out / "meta.json").write_text(json.dumps({
        "seed": args.seed, "source": "prompted collection forms",
        "split_policy": "writer-independent", "writers": splits,
        "counts": per_split, "stats": stats,
        "skipped": [{"file": a, "reason": b} for a, b in skipped],
    }, indent=2))

    print(f"forms rectified : {stats['forms']}/{len(jobs)}")
    print(f"crops written   : {stats['cells']}  {per_split}")
    print(f"blank cells     : {stats['blank']}")
    if skipped:
        print(f"skipped         : {len(skipped)}")
        for a, b in skipped[:5]:
            print(f"   {a}: {b}")
    print(f"-> {args.out}/labels.tsv")


if __name__ == "__main__":
    main()
