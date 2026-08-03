"""Synthesise handwritten-style drug-name word images.

IAM gives us real handwriting but essentially no medicine names; the drug corpus
gives us the right vocabulary but no images. This module bridges the two: it
renders the drug corpus in handwriting fonts and degrades the result
(:mod:`lasa.data.augment`) until it plausibly resembles a photographed chit.

Realism comes mostly from *per-character* jitter rather than from the font.
Setting a whole string in a handwriting font still yields perfectly uniform
letterforms on a perfectly straight baseline, which a CNN learns to exploit.
Jittering each glyph's baseline, rotation, size and spacing independently — plus
a slow sinusoidal baseline drift across the word — removes that shortcut.

Splits are deliberately adversarial, so the report can separate memorisation
from genuine generalisation:

    train             train words x train fonts
    val               train words x train fonts   (different render seed)
    test_seen         train words x train fonts   (different render seed)
    test_unseen_font  train words x HELD-OUT fonts
    test_unseen_word  HELD-OUT words x train fonts
    test_hard         HELD-OUT words x HELD-OUT fonts

``test_unseen_word`` is the honest measure of character-level generalisation;
``test_seen`` is the number that matters operationally, because a real pharmacy
system searches a fixed, known formulary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from lasa.data.augment import (
    AugmentConfig,
    composite_and_degrade,
    get_config,
    ink_paper_contrast,
)

TARGET_HEIGHT = 64          # canonical stored height; loaders may resize again
MAX_WIDTH = 640


@dataclass
class JitterConfig:
    """Per-glyph irregularity. This is what makes it read as handwriting."""

    baseline_px: tuple[float, float] = (-0.055, 0.055)   # as fraction of font size
    drift_amp: tuple[float, float] = (0.0, 0.07)         # sinusoidal word-level drift
    drift_cycles: tuple[float, float] = (0.4, 1.6)
    char_rotate_deg: tuple[float, float] = (-7.0, 7.0)
    char_scale: tuple[float, float] = (0.90, 1.10)
    # Spacing stays only mildly negative. `font.getlength()` returns the
    # *advance*, but script faces draw ink well outside it, so aggressive
    # negative spacing overlaps glyphs into a different letter entirely
    # (xeloda -> "xepda"). That is label noise, not realism.
    #
    # The right way to get joined-up handwriting is `connect_p` below: an
    # explicit connecting stroke between glyphs. It reads as cursive without
    # destroying letter identity, which crushing the advance never does.
    spacing: tuple[float, float] = (-0.05, 0.10)
    p_ligature: float = 0.20      # chance of pulling two letters closer
    ligature_delta: float = 0.06

    connect_p: float = 0.0        # chance of a cursive connector per glyph gap
    connect_sag: tuple[float, float] = (0.0, 0.10)   # dip below baseline
    connect_width: tuple[float, float] = (0.030, 0.065)  # x font size


# Messiness scales with the augmentation tier. Doctor handwriting is fast and
# strongly slanted with letters running together, so the upper tiers push slant,
# per-glyph rotation/size variance, baseline drift and cursive connection --
# not raw glyph overlap.
JITTER_TIERS: dict[str, "JitterConfig"] = {
    "clean": JitterConfig(
        baseline_px=(-0.030, 0.030), drift_amp=(0.0, 0.035),
        char_rotate_deg=(-3.5, 3.5), char_scale=(0.95, 1.05),
        spacing=(-0.02, 0.08), p_ligature=0.10, ligature_delta=0.04,
        connect_p=0.10,
    ),
    "medium": JitterConfig(
        baseline_px=(-0.075, 0.075), drift_amp=(0.0, 0.105),
        drift_cycles=(0.4, 1.9), char_rotate_deg=(-10.0, 10.0),
        char_scale=(0.86, 1.15), spacing=(-0.09, 0.08),
        p_ligature=0.35, ligature_delta=0.07, connect_p=0.45,
    ),
    "hard": JitterConfig(
        baseline_px=(-0.105, 0.105), drift_amp=(0.0, 0.17),
        drift_cycles=(0.4, 2.4), char_rotate_deg=(-15.0, 15.0),
        char_scale=(0.78, 1.24), spacing=(-0.14, 0.07),
        p_ligature=0.55, ligature_delta=0.09, connect_p=0.70,
        connect_sag=(0.0, 0.16), connect_width=(0.030, 0.080),
    ),
}


def _u(rng: np.random.Generator, r: tuple[float, float]) -> float:
    return float(rng.uniform(r[0], r[1]))


def _draw_connectors(
    canvas: Image.Image,
    anchors: list[tuple[float, float, float] | None],
    font_size: int,
    rng: np.random.Generator,
    jit: JitterConfig,
) -> Image.Image:
    """Add cursive joining strokes between consecutive glyphs.

    A sagging quadratic curve from one glyph's baseline exit to the next one's
    baseline entry. This is what actually makes a word read as joined-up
    handwriting; crushing the advance width only makes letters collide.
    """
    if jit.connect_p <= 0.0:
        return canvas

    layer = Image.new("L", canvas.size, 0)
    draw = ImageDraw.Draw(layer)
    drew = False

    for a, b in zip(anchors, anchors[1:]):
        if a is None or b is None or rng.random() >= jit.connect_p:
            continue

        x0, y0 = a[1], a[2]          # exit of the left glyph
        x1, y1 = b[0], b[2]          # entry of the right glyph
        if x1 <= x0:                 # letters already overlap; nothing to join
            continue

        sag = _u(rng, jit.connect_sag) * font_size
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0 + sag
        width = max(1, int(round(_u(rng, jit.connect_width) * font_size)))

        pts = []
        for t in np.linspace(0.0, 1.0, 12):
            # quadratic Bezier through the sagged control point
            px = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
            py = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1
            pts.append((float(px), float(py)))

        draw.line(pts, fill=255, width=width, joint="curve")
        drew = True

    return ImageChops.lighter(canvas, layer) if drew else canvas


def render_ink(
    text: str,
    font_path: Path,
    font_size: int,
    rng: np.random.Generator,
    jit: JitterConfig,
) -> np.ndarray:
    """Render ``text`` glyph-by-glyph into a float32 ink mask in [0, 1]."""
    font = ImageFont.truetype(str(font_path), font_size)
    ascent, descent = font.getmetrics()
    pad = int(font_size * 0.9)

    advances = [font.getlength(c) for c in text]
    # Headroom so no transform can run off the canvas: glyphs may be scaled up
    # to 1.1x, spaced up to +10%, and displaced vertically by wobble + drift.
    est_w = int(sum(advances) * 1.35) + 3 * pad
    est_h = ascent + descent + 2 * pad
    margin = int(font_size * 0.30)
    canvas = Image.new("L", (max(est_w, 8), max(est_h + 2 * margin, 8)), 0)

    drift_amp = _u(rng, jit.drift_amp) * font_size
    drift_cycles = _u(rng, jit.drift_cycles)
    n = max(len(text), 1)

    # (entry_x, exit_x, baseline_y) per glyph; None for spaces, which break the
    # cursive connection exactly as a pen lift would.
    anchors: list[tuple[float, float, float] | None] = []

    x = float(pad)
    for i, ch in enumerate(text):
        adv = advances[i]

        if ch.isspace():
            x += adv * (1.0 + _u(rng, jit.spacing))
            anchors.append(None)
            continue

        # Render the glyph alone so it can be transformed independently. The
        # drawn origin sits at (pad, pad) inside this tile; every transform
        # below must keep that reference point trackable, or glyphs land in the
        # wrong place and fuse into different letters.
        gw = max(int(adv) + 2 * pad, 8)
        gh = est_h
        glyph = Image.new("L", (gw, gh), 0)
        ImageDraw.Draw(glyph).text((pad, pad), ch, font=font, fill=255)

        # Rotate in place, NOT with expand=True. This tile is tall and narrow,
        # so expand=True would widen it by ~gh*sin(rot) (about a full character
        # width) and silently displace the glyph. The 0.9*font_size padding is
        # ample for +-7 degrees about the glyph body, so nothing clips.
        rot = _u(rng, jit.char_rotate_deg)
        if abs(rot) > 0.1:
            glyph = glyph.rotate(
                rot, resample=Image.BILINEAR, expand=False,
                center=(pad + adv / 2.0, pad + ascent / 2.0),
            )

        scale = _u(rng, jit.char_scale)
        if abs(scale - 1.0) > 0.01:
            glyph = glyph.resize(
                (max(1, int(gw * scale)), max(1, int(gh * scale))), Image.BILINEAR
            )
        else:
            scale = 1.0

        # Baseline: independent per-glyph jitter + a slow drift across the word.
        wobble = _u(rng, jit.baseline_px) * font_size
        drift = drift_amp * math.sin(2 * math.pi * drift_cycles * (i / n))

        # After the resize the origin lives at (pad*scale, pad*scale), so offset
        # by that — not by the raw pad — to land the glyph at (x, pad+dy).
        off = pad * scale
        px = int(round(x - off))
        py = int(round(margin + pad + wobble + drift - off))

        # Blend with `lighter` so overlapping glyphs merge like joined-up ink
        # instead of one glyph's black box erasing its neighbour.
        layer = Image.new("L", canvas.size, 0)
        layer.paste(glyph, (px, py))
        canvas = ImageChops.lighter(canvas, layer)

        # Baseline sits `ascent` below the drawn origin, scaled with the glyph.
        baseline_y = py + (pad + ascent) * scale
        anchors.append((x, x + adv * scale, baseline_y))

        spacing = _u(rng, jit.spacing)
        if rng.random() < jit.p_ligature:
            spacing -= jit.ligature_delta
        x += adv * (1.0 + spacing) * scale

    canvas = _draw_connectors(canvas, anchors, font_size, rng, jit)

    ink = np.asarray(canvas, dtype=np.float32) / 255.0

    # Crop to the ink, leaving a small margin for the warps to bleed into.
    ys, xs = np.nonzero(ink > 0.08)
    if len(ys) == 0:
        return np.zeros((TARGET_HEIGHT, TARGET_HEIGHT), np.float32)
    m = int(font_size * 0.35)
    y0, y1 = max(0, ys.min() - m), min(ink.shape[0], ys.max() + m + 1)
    x0, x1 = max(0, xs.min() - m), min(ink.shape[1], xs.max() + m + 1)
    return np.ascontiguousarray(ink[y0:y1, x0:x1])


def fit_height(img: np.ndarray, height: int = TARGET_HEIGHT,
               max_width: int = MAX_WIDTH) -> np.ndarray:
    h, w = img.shape
    scale = height / h
    new_w = max(8, min(int(round(w * scale)), max_width))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, height), interpolation=interp)


# Raised alongside the messier jitter settings: heavier slant, scrawl and
# connection make strokes intrinsically harder to follow, so the *photometric*
# floor has to come up or the two effects compound into illegibility.
MIN_CONTRAST = 34.0   # 0-255 units; below this the strokes stop being readable
MAX_ATTEMPTS = 4


def synthesize(
    text: str,
    font_path: Path,
    rng: np.random.Generator,
    cfg: AugmentConfig,
    jit: JitterConfig | None = None,
    font_size_range: tuple[int, int] = (48, 82),
    height: int = TARGET_HEIGHT,
    min_contrast: float = MIN_CONTRAST,
) -> np.ndarray:
    """Render + degrade ``text``, rejecting draws that came out illegible.

    Most rejections are unlucky combinations (pale ink + dark paper + heavy
    blur), so we simply redraw at the same tier. Only if that keeps failing do
    we fall back to the clean tier — dropping the sample instead would quietly
    bias the word distribution away from long names, which degrade fastest.
    """
    # Jitter tracks the tier unless the caller overrides it, so "hard" means
    # messier handwriting AND worse imaging, not just worse imaging.
    jit = jit or JITTER_TIERS.get(cfg.name, JitterConfig())

    for attempt in range(MAX_ATTEMPTS):
        last = attempt == MAX_ATTEMPTS - 1
        active = get_config("clean") if last else cfg
        active_jit = JITTER_TIERS["clean"] if last else jit
        size = int(rng.integers(font_size_range[0], font_size_range[1] + 1))
        ink = render_ink(text, font_path, size, rng, active_jit)
        img, mask = composite_and_degrade(ink, active, rng, return_mask=True)
        if ink_paper_contrast(img, mask) >= min_contrast:
            break

    return fit_height(img, height)


# ------------------------------------------------------------------ splits --


@dataclass
class SplitSpec:
    name: str
    n_samples: int
    words: list[str]
    fonts: list[str]
    tier: str
    seed: int


def _worker(job: tuple) -> tuple[str, str, str, str, int, int] | None:
    idx, text, font_file, fonts_dir, tier, seed, out_dir, split = job
    rng = np.random.default_rng(seed)
    cfg = get_config(tier)
    try:
        img = synthesize(text, Path(fonts_dir) / font_file, rng, cfg)
    except Exception as exc:  # a glyph the font cannot render, etc.
        print(f"  ! render failed {text!r}/{font_file}: {exc}")
        return None
    rel = f"{split}/{idx:07d}.png"
    path = Path(out_dir) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return rel, text, font_file, tier, img.shape[1], img.shape[0]


def generate_split(
    spec: SplitSpec, fonts_dir: Path, out_dir: Path, workers: int = 8
) -> list[tuple]:
    rng = np.random.default_rng(spec.seed)
    words = rng.choice(spec.words, size=spec.n_samples, replace=True)
    fonts = rng.choice(spec.fonts, size=spec.n_samples, replace=True)
    seeds = rng.integers(0, 2**31 - 1, size=spec.n_samples)

    jobs = [
        (i, str(words[i]), str(fonts[i]), str(fonts_dir), spec.tier,
         int(seeds[i]), str(out_dir), spec.name)
        for i in range(spec.n_samples)
    ]

    with Pool(workers) as pool:
        rows = [r for r in pool.imap_unordered(_worker, jobs, chunksize=64) if r]
    return rows


def build_dataset(
    drug_csv: Path,
    fonts_dir: Path,
    out_dir: Path,
    n_train: int = 48000,
    n_val: int = 3000,
    n_test: int = 2000,
    heldout_font_frac: float = 0.20,
    heldout_word_frac: float = 0.10,
    tier: str = "medium",
    seed: int = 1337,
    workers: int = 8,
) -> dict:
    from lasa.data.fonts import load_manifest

    rng = np.random.default_rng(seed)

    with drug_csv.open(encoding="utf-8") as fh:
        all_words = [r["name"] for r in csv.DictReader(fh)]
    all_words = sorted(set(all_words))

    font_files = [f.filename for f in load_manifest(fonts_dir)]
    font_files.sort()

    # Deterministic, seeded holdouts.
    fperm = rng.permutation(len(font_files))
    n_hf = max(1, int(round(len(font_files) * heldout_font_frac)))
    heldout_fonts = sorted(font_files[i] for i in fperm[:n_hf])
    train_fonts = sorted(font_files[i] for i in fperm[n_hf:])

    wperm = rng.permutation(len(all_words))
    n_hw = max(1, int(round(len(all_words) * heldout_word_frac)))
    heldout_words = sorted(all_words[i] for i in wperm[:n_hw])
    train_words = sorted(all_words[i] for i in wperm[n_hw:])

    specs = [
        SplitSpec("train", n_train, train_words, train_fonts, tier, seed + 1),
        SplitSpec("val", n_val, train_words, train_fonts, tier, seed + 2),
        SplitSpec("test_seen", n_test, train_words, train_fonts, tier, seed + 3),
        SplitSpec("test_unseen_font", n_test, train_words, heldout_fonts, tier, seed + 4),
        SplitSpec("test_unseen_word", n_test, heldout_words, train_fonts, tier, seed + 5),
        SplitSpec("test_hard", n_test, heldout_words, heldout_fonts, tier, seed + 6),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[tuple] = []
    counts: dict[str, int] = {}

    for spec in specs:
        print(f"  generating {spec.name:<17} n={spec.n_samples:<6} "
              f"words={len(spec.words):<5} fonts={len(spec.fonts)}")
        rows = generate_split(spec, fonts_dir, out_dir, workers=workers)
        counts[spec.name] = len(rows)
        manifest_rows.extend((spec.name, *r) for r in rows)

    labels = out_dir / "labels.tsv"
    with labels.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["split", "path", "text", "font", "tier", "width", "height"])
        w.writerows(manifest_rows)

    meta = {
        "seed": seed,
        "tier": tier,
        "target_height": TARGET_HEIGHT,
        "counts": counts,
        "n_train_words": len(train_words),
        "n_heldout_words": len(heldout_words),
        "train_fonts": train_fonts,
        "heldout_fonts": heldout_fonts,
        "heldout_words_sample": heldout_words[:25],
        "jitter": asdict(JitterConfig()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ----------------------------------------------------------------- preview --


def make_preview(
    drug_csv: Path, fonts_dir: Path, out_png: Path, seed: int = 7,
    tiers: tuple[str, ...] = ("clean", "medium", "hard"),
    n_per_tier: int = 8,
) -> Path:
    """Contact sheet for eyeballing realism before committing to a full run."""
    from lasa.data.fonts import load_manifest
    from lasa.data.lasa_pairs import deduped_pairs

    rng = np.random.default_rng(seed)
    fonts = [f.filename for f in load_manifest(fonts_dir)]

    # Show LASA pairs adjacently — the whole point is whether *these* are
    # distinguishable once handwritten.
    pairs = deduped_pairs()
    picks = [pairs[i] for i in rng.choice(len(pairs), size=n_per_tier, replace=False)]
    words = [w for p in picks for w in (p.a, p.b)]

    rows: list[np.ndarray] = []
    label_h = 22
    cell_w, cell_h = 430, 80

    for tier in tiers:
        cfg = get_config(tier)
        band: list[np.ndarray] = []
        for i, word in enumerate(words):
            font = fonts[int(rng.integers(0, len(fonts)))]
            img = synthesize(word, Path(fonts_dir) / font, rng, cfg, height=56)
            cell = np.full((cell_h, cell_w), 255, np.uint8)
            # Scale to fit rather than crop — a cropped preview would hide
            # exactly the long-name failures we most need to inspect.
            avail_w = cell_w - 8
            h, w = img.shape
            if w > avail_w:
                s = avail_w / w
                img = cv2.resize(img, (avail_w, max(1, int(h * s))),
                                 interpolation=cv2.INTER_AREA)
                h, w = img.shape
            cell[4:4 + h, 4:4 + w] = img
            cv2.putText(cell, word, (4, cell_h - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, 0, 1, cv2.LINE_AA)
            band.append(cell)
        # 2 columns x n rows per tier
        cols = 2
        rws = [np.hstack(band[i:i + cols]) for i in range(0, len(band), cols)]
        block = np.vstack(rws)
        header = np.full((label_h, block.shape[1]), 235, np.uint8)
        cv2.putText(header, f"tier = {tier}", (6, label_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1, cv2.LINE_AA)
        rows.append(np.vstack([header, block]))

    sheet = np.vstack(rows)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), sheet)
    return out_png


def main() -> None:
    from lasa.common.paths import DRUG_NAMES_CSV, FIGURES, FONTS, SYNTH, ensure_dirs
    from lasa.common.seed import set_seed

    ap = argparse.ArgumentParser(description="Synthesise handwritten drug names.")
    ap.add_argument("--preview", action="store_true", help="only write a contact sheet")
    ap.add_argument("--n-train", type=int, default=48000)
    ap.add_argument("--n-val", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--tier", default="medium", choices=["clean", "medium", "hard"])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ensure_dirs()
    set_seed(args.seed)

    if args.preview:
        out = make_preview(DRUG_NAMES_CSV, FONTS, FIGURES / "synth_preview.png",
                           seed=args.seed)
        print(f"preview -> {out}")
        return

    meta = build_dataset(
        DRUG_NAMES_CSV, FONTS, SYNTH,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        tier=args.tier, seed=args.seed, workers=args.workers,
    )
    print(json.dumps(meta["counts"], indent=2))
    print(f"held-out fonts: {meta['heldout_fonts']}")


if __name__ == "__main__":
    main()
