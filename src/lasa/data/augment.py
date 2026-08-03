"""Degradation pipeline that turns clean font renders into plausible
photographs of handwritten prescription chits.

The point is *domain gap*. A font render is perfectly uniform: constant stroke
width, dead-straight baseline, pure white background. A phone photo of a real
chit has none of that. Every transform here targets one specific way that
assumption breaks:

    baseline wobble / per-char jitter -> a hand does not write on a ruler
    shear                             -> personal slant
    elastic distortion                -> unsteady, fast strokes
    dilate / erode                     -> pen pressure and nib width
    paper synthesis                   -> off-white stock, fibres, ruled lines
    ink modulation                    -> ballpoint skipping, pressure fade
    rotation                          -> the chit is never square to the camera
    blur (gaussian + motion)          -> focus miss, hand shake
    noise / JPEG / down-up sampling   -> cheap sensor, compressed upload

Difficulty tiers let us run an ablation ("how much does aggressive augmentation
buy us?") and report it, rather than asserting that augmentation helps.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


@dataclass
class AugmentConfig:
    """All ranges are (min, max) and sampled uniformly unless noted."""

    name: str = "medium"

    # --- geometry ---
    shear: tuple[float, float] = (-0.30, 0.30)      # slant, x-shear factor
    rotate_deg: tuple[float, float] = (-4.0, 4.0)
    elastic_alpha: tuple[float, float] = (8.0, 22.0)   # displacement magnitude
    elastic_sigma: tuple[float, float] = (4.0, 7.0)    # smoothness of the field
    p_elastic: float = 0.85

    # --- stroke ---
    stroke_delta: tuple[int, int] = (-1, 2)   # <0 erode (thin), >0 dilate (thick)
    p_stroke: float = 0.7

    # --- ink / paper ---
    ink_value: tuple[int, int] = (10, 95)     # 0=black; ballpoint is never pure black
    ink_fade: tuple[float, float] = (0.0, 0.35)  # fraction of stroke that skips
    paper_value: tuple[int, int] = (215, 252)
    paper_fibre_sigma: float = 1.4
    p_ruled_line: float = 0.25
    p_blotch: float = 0.20

    # --- optics / sensor ---
    blur_sigma: tuple[float, float] = (0.0, 1.1)
    p_motion_blur: float = 0.25
    motion_ksize: tuple[int, int] = (3, 7)
    noise_sigma: tuple[float, float] = (2.0, 9.0)
    contrast: tuple[float, float] = (0.80, 1.15)
    brightness: tuple[int, int] = (-18, 18)
    p_jpeg: float = 0.5
    jpeg_quality: tuple[int, int] = (35, 88)
    p_downscale: float = 0.35
    downscale_factor: tuple[float, float] = (0.45, 0.85)


CLEAN = AugmentConfig(
    name="clean",
    shear=(-0.12, 0.12), rotate_deg=(-1.5, 1.5),
    elastic_alpha=(3.0, 8.0), elastic_sigma=(5.0, 8.0), p_elastic=0.4,
    stroke_delta=(0, 1), p_stroke=0.3,
    ink_value=(10, 55), ink_fade=(0.0, 0.10),
    paper_value=(240, 255), p_ruled_line=0.10, p_blotch=0.05,
    blur_sigma=(0.0, 0.5), p_motion_blur=0.05,
    noise_sigma=(0.0, 3.0), p_jpeg=0.2, jpeg_quality=(70, 95),
    p_downscale=0.10,
)

# Slant is per-word, i.e. a personal writing angle rather than per-letter noise.
# Real prescriber hands run well past 20 degrees, so the shear range is wide.
MEDIUM = AugmentConfig(name="medium", shear=(-0.42, 0.42))

# "hard" means hard-but-readable. Earlier settings pushed some words below
# human legibility (faint ink on dark stock + heavy noise), which is label
# noise, not difficulty — it teaches the model to guess from length alone.
HARD = AugmentConfig(
    name="hard",
    shear=(-0.62, 0.62), rotate_deg=(-8.0, 8.0),
    elastic_alpha=(16.0, 34.0), elastic_sigma=(3.5, 6.0), p_elastic=0.95,
    stroke_delta=(-2, 3), p_stroke=0.85,
    ink_value=(10, 95), ink_fade=(0.05, 0.40),
    paper_value=(205, 250), p_ruled_line=0.35, p_blotch=0.35,
    blur_sigma=(0.2, 1.3), p_motion_blur=0.40, motion_ksize=(3, 9),
    noise_sigma=(3.0, 11.0), contrast=(0.75, 1.20), brightness=(-24, 24),
    p_jpeg=0.7, jpeg_quality=(30, 75),
    p_downscale=0.55, downscale_factor=(0.40, 0.75),
)

TIERS: dict[str, AugmentConfig] = {"clean": CLEAN, "medium": MEDIUM, "hard": HARD}


def _u(rng: np.random.Generator, lo_hi: tuple[float, float]) -> float:
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


# ---------------------------------------------------------------- geometry --


def elastic_distort(ink: np.ndarray, alpha: float, sigma: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Smooth random displacement field — the classic Simard et al. warp."""
    h, w = ink.shape
    dx = gaussian_filter(rng.uniform(-1, 1, (h, w)), sigma, mode="constant") * alpha
    dy = gaussian_filter(rng.uniform(-1, 1, (h, w)), sigma, mode="constant") * alpha
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.array([np.clip(yy + dy, 0, h - 1), np.clip(xx + dx, 0, w - 1)])
    return map_coordinates(ink, coords, order=1, mode="constant", cval=0.0)


def shear_image(ink: np.ndarray, factor: float) -> np.ndarray:
    h, w = ink.shape
    pad = int(abs(factor) * h) + 2
    canvas = np.zeros((h, w + 2 * pad), dtype=ink.dtype)
    canvas[:, pad:pad + w] = ink
    m = np.float32([[1, factor, -factor * h / 2], [0, 1, 0]])
    return cv2.warpAffine(canvas, m, (canvas.shape[1], h),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def rotate_image(ink: np.ndarray, deg: float) -> np.ndarray:
    h, w = ink.shape
    pad = int(abs(np.sin(np.deg2rad(deg))) * w) + 4
    canvas = np.zeros((h + 2 * pad, w), dtype=ink.dtype)
    canvas[pad:pad + h, :] = ink
    ch, cw = canvas.shape
    m = cv2.getRotationMatrix2D((cw / 2, ch / 2), deg, 1.0)
    return cv2.warpAffine(canvas, m, (cw, ch), flags=cv2.INTER_LINEAR, borderValue=0)


# ------------------------------------------------------------------ stroke --


def adjust_stroke(ink: np.ndarray, delta: int) -> np.ndarray:
    if delta == 0:
        return ink
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(delta) * 2 + 1,) * 2)
    return cv2.dilate(ink, k) if delta > 0 else cv2.erode(ink, k)


def modulate_ink(ink: np.ndarray, fade: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate ballpoint pressure variation and skipping.

    A low-frequency multiplicative field means a whole region of the stroke
    goes light, which is what a skipping pen actually does — unlike per-pixel
    noise, which just looks like film grain.
    """
    if fade <= 0.01:
        return ink
    h, w = ink.shape
    field = gaussian_filter(rng.random((h, w)), sigma=max(h, w) / 18.0)
    field -= field.min()
    if field.max() > 1e-6:
        field /= field.max()
    return ink * (1.0 - fade * field)


# ------------------------------------------------------------------- paper --


def make_paper(h: int, w: int, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    base = _u(rng, cfg.paper_value)
    paper = np.full((h, w), base, dtype=np.float32)

    # Fibre texture: smoothed noise, so it reads as stock rather than grain.
    fibre = gaussian_filter(rng.normal(0, 6.0, (h, w)), cfg.paper_fibre_sigma)
    paper += fibre

    # Uneven illumination — one side of the chit catches more light.
    gx = np.linspace(_u(rng, (-14, 14)), _u(rng, (-14, 14)), w)[None, :]
    gy = np.linspace(_u(rng, (-10, 10)), _u(rng, (-10, 10)), h)[:, None]
    paper += gx + gy

    if rng.random() < cfg.p_ruled_line:
        y = int(rng.integers(int(h * 0.55), max(int(h * 0.55) + 1, int(h * 0.92))))
        thick = int(rng.integers(1, 3))
        shade = _u(rng, (150, 205))
        paper[y:y + thick, :] = np.minimum(paper[y:y + thick, :], shade)

    if rng.random() < cfg.p_blotch:
        for _ in range(int(rng.integers(1, 4))):
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            rad = int(rng.integers(2, max(3, h // 6)))
            cv2.circle(paper, (cx, cy), rad, float(_u(rng, (160, 215))), -1)
        paper = gaussian_filter(paper, 1.0)

    return paper


# -------------------------------------------------------------- full chain --


def composite_and_degrade(
    ink: np.ndarray,
    cfg: AugmentConfig,
    rng: np.random.Generator,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """``ink`` is float32 in [0,1] where 1 = full ink coverage.

    Returns a uint8 grayscale image (255 = paper white). With ``return_mask``,
    also returns the *warped* ink mask, aligned to the output image — callers
    use it to measure whether the strokes survived degradation legibly.
    """
    if rng.random() < cfg.p_elastic:
        ink = elastic_distort(ink, _u(rng, cfg.elastic_alpha),
                              _u(rng, cfg.elastic_sigma), rng)

    ink = shear_image(ink, _u(rng, cfg.shear))

    if rng.random() < cfg.p_stroke:
        lo, hi = cfg.stroke_delta
        ink = adjust_stroke(ink, int(rng.integers(lo, hi + 1)))

    ink = rotate_image(ink, _u(rng, cfg.rotate_deg))
    # Geometry is final here — snapshot the mask before intensity modulation so
    # a faded stroke still counts as "where the ink is".
    mask = (ink > 0.35).astype(np.uint8)
    ink = modulate_ink(ink, _u(rng, cfg.ink_fade), rng)
    ink = np.clip(ink, 0.0, 1.0)

    h, w = ink.shape
    paper = make_paper(h, w, cfg, rng)
    ink_val = _u(rng, cfg.ink_value)

    # alpha-composite ink over paper
    img = paper * (1.0 - ink) + ink_val * ink

    # --- optics ---
    sigma = _u(rng, cfg.blur_sigma)
    if sigma > 0.05:
        img = gaussian_filter(img, sigma)

    if rng.random() < cfg.p_motion_blur:
        k = int(rng.integers(*cfg.motion_ksize)) | 1
        kern = np.zeros((k, k), np.float32)
        angle = rng.uniform(0, np.pi)
        cx = cy = k // 2
        for i in range(k):
            x = int(round(cx + (i - cx) * np.cos(angle)))
            y = int(round(cy + (i - cy) * np.sin(angle)))
            if 0 <= x < k and 0 <= y < k:
                kern[y, x] = 1.0
        s = kern.sum()
        if s > 0:
            img = cv2.filter2D(img, -1, kern / s)

    # --- sensor ---
    img = img * _u(rng, cfg.contrast) + _u(rng, cfg.brightness)
    img += rng.normal(0, _u(rng, cfg.noise_sigma), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    if rng.random() < cfg.p_downscale:
        f = _u(rng, cfg.downscale_factor)
        small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    if rng.random() < cfg.p_jpeg:
        q = int(rng.integers(*cfg.jpeg_quality))
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

    return (img, mask) if return_mask else img


def ink_paper_contrast(img: np.ndarray, mask: np.ndarray) -> float:
    """Mean paper luminance minus mean ink luminance, in 0-255 units.

    A legibility proxy: if the strokes are not measurably darker than the
    surrounding stock, no reader — human or model — can recover the word, and
    the sample is mislabelled data rather than a hard example.
    """
    ink_px = img[mask > 0]
    paper_px = img[mask == 0]
    if ink_px.size < 12 or paper_px.size < 12:
        return 0.0
    return float(paper_px.mean() - ink_px.mean())


def get_config(tier: str) -> AugmentConfig:
    if tier not in TIERS:
        raise KeyError(f"unknown tier {tier!r}; choose from {sorted(TIERS)}")
    return replace(TIERS[tier])
