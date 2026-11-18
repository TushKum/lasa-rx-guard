"""Generate printable handwriting-collection forms.

Design rationale
----------------
We do NOT photograph real patient prescriptions. We ask prescribers to write
drug names onto a structured form. That choice is deliberate:

  * No patient data is ever created, so there is no personal health
    information to protect, redact, or justify holding.
  * Ground truth is exact. Photographing a real prescription means *guessing*
    which drug was written, which is the very task under test -- circular, and
    a source of label noise.
  * It is the standard construction. The IAM corpus was built the same way:
    writers copied prompts from the LOB corpus. The limitation (copied rather
    than composed handwriting) is known, accepted, and must be stated.

Each form carries four ArUco markers. A phone photo taken at any reasonable
angle can therefore be rectified automatically, which is what makes collection
practical in a clinic corridor rather than at a flatbed scanner.

Cell geometry is stored NORMALISED to the marker rectangle, so the ingest
script never needs to know the print resolution or paper size.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

A4_300DPI = (2480, 3508)
MARKER_PX = 220          # larger = still detectable in a downscaled phone photo
MARGIN_PX = 165          # ~14mm: clear of printer no-print zones and of
                         # being clipped when the page is shot at an angle
DICT = cv2.aruco.DICT_4X4_50

COLS, ROWS = 2, 11
CELL_PROMPT_H = 58          # printed prompt strip above each write-in box
CELL_BOX_H = 150            # the empty box the prescriber writes in


def _font(size: int, bold: bool = False):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
              else "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _markers(img: Image.Image) -> dict:
    """Stamp four ArUco markers; return their centres in pixel coords."""
    d = cv2.aruco.getPredefinedDictionary(DICT)
    W, H = img.size
    pos = {
        0: (MARGIN_PX, MARGIN_PX),                              # top-left
        1: (W - MARGIN_PX - MARKER_PX, MARGIN_PX),              # top-right
        2: (W - MARGIN_PX - MARKER_PX, H - MARGIN_PX - MARKER_PX),  # bottom-right
        3: (MARGIN_PX, H - MARGIN_PX - MARKER_PX),              # bottom-left
    }
    centres = {}
    for mid, (x, y) in pos.items():
        m = cv2.aruco.generateImageMarker(d, mid, MARKER_PX)
        img.paste(Image.fromarray(m).convert("L"), (x, y))
        centres[mid] = (x + MARKER_PX / 2, y + MARKER_PX / 2)
    return centres


def build_form(form_id: str, prompts: list[str], out_dir: Path) -> dict:
    img = Image.new("L", A4_300DPI, 255)
    dr = ImageDraw.Draw(img)
    centres = _markers(img)

    # Canonical frame: the rectangle joining marker centres.
    x0, y0 = centres[0]
    x1, y1 = centres[2]
    fw, fh = x1 - x0, y1 - y0

    f_title, f_body, f_small, f_prompt = _font(52, True), _font(34), _font(26), _font(30)
    # Anchor content to the OUTER edges of the marker bands, not the centres:
    # the markers occupy MARGIN..MARGIN+MARKER_PX, and y0/y1 are centres, so
    # laying text out from y0 puts the title underneath the top-left marker.
    band_top = MARGIN_PX + MARKER_PX
    band_bottom = img.size[1] - MARGIN_PX - MARKER_PX
    top = band_top + 40
    dr.text((x0, top), "LASA-RxGuard — handwriting sample sheet", font=f_title, fill=0)
    dr.text((x0, top + 72),
            "Please write each medicine name in the box beneath it, in your normal "
            "prescribing hand.", font=f_body, fill=60)
    dr.text((x0, top + 118),
            "Write at your usual speed. Do not print neatly for us — natural writing is "
            "what we need.", font=f_body, fill=60)

    dr.text((x0, top + 182), f"Form {form_id}", font=f_prompt, fill=0)
    dr.text((x0 + 420, top + 182), "Writer code: ____________", font=f_prompt, fill=0)
    dr.text((x0 + 1180, top + 182), "Date: ____________", font=f_prompt, fill=0)
    dr.line([(x0, top + 232), (x1, top + 232)], fill=150, width=3)

    footer_y = band_bottom - 70
    grid_top = top + 290
    grid_bottom = footer_y - 50
    col_w = fw / COLS
    cell_h = (grid_bottom - grid_top) / ROWS
    box_w = col_w - 90

    cells = []
    for i, word in enumerate(prompts[: COLS * ROWS]):
        c, r = i % COLS, i // COLS
        cx = x0 + c * col_w
        cy = grid_top + r * cell_h
        dr.text((cx + 6, cy), f"{i + 1}.  {word}", font=f_small, fill=110)
        bx0, by0 = cx, cy + CELL_PROMPT_H
        bx1, by1 = cx + box_w, cy + CELL_PROMPT_H + CELL_BOX_H
        dr.rounded_rectangle([bx0, by0, bx1, by1], radius=10, outline=175, width=3)
        # Normalised to the marker rectangle so ingest is resolution-agnostic.
        cells.append({
            "index": i, "text": word,
            "box": [(bx0 - x0) / fw, (by0 - y0) / fh,
                    (bx1 - x0) / fw, (by1 - y0) / fh],
        })

    dr.text((x0, footer_y),
            "This sheet contains no patient information. Do not write any patient "
            "name, age or identifier on it.", font=f_small, fill=90)

    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / f"{form_id}.png")
    img.convert("RGB").save(out_dir / f"{form_id}.pdf", resolution=300.0)
    return {"form_id": form_id, "n_cells": len(cells), "cells": cells}


def pick_prompts(rng: random.Random, pairs_first: list[str], general: list[str],
                 n: int, pair_frac: float = 0.6) -> list[str]:
    """Over-sample documented LASA names: those are the safety-critical reads."""
    n_pair = int(round(n * pair_frac))
    out = rng.sample(pairs_first, min(n_pair, len(pairs_first)))
    out += rng.sample(general, max(0, n - len(out)))
    rng.shuffle(out)
    return out


def main() -> None:
    from lasa.common.paths import DRUG_NAMES_CSV, ROOT
    from lasa.data.drug_names import load_csv
    from lasa.data.lasa_pairs import names_in_pairs

    ap = argparse.ArgumentParser(description="Generate handwriting collection forms.")
    ap.add_argument("--n-forms", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "forms")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    corpus = [d.name for d in load_csv(DRUG_NAMES_CSV)]
    pair_names = sorted(names_in_pairs() & set(corpus))
    general = [c for c in corpus if c not in set(pair_names)]

    per_form = COLS * ROWS
    manifest = {"seed": args.seed, "cells_per_form": per_form,
                "marker_dict": "DICT_4X4_50", "forms": []}
    for k in range(args.n_forms):
        fid = f"LRG-{k + 1:03d}"
        prompts = pick_prompts(rng, pair_names, general, per_form)
        manifest["forms"].append(build_form(fid, prompts, args.out))

    (args.out / "forms_manifest.json").write_text(json.dumps(manifest, indent=2))
    total = args.n_forms * per_form
    print(f"{args.n_forms} forms x {per_form} cells = {total} samples per writer pass")
    print(f"  LASA-pair names available: {len(pair_names)}")
    print(f"  wrote {args.out}/LRG-*.pdf (+ .png) and forms_manifest.json")


if __name__ == "__main__":
    main()
