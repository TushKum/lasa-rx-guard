"""Regression tests for the handwriting-collection pipeline.

The bug these exist to prevent: a bare ink-fraction threshold silently
discarded genuine writing. Short words written with a fine pen covered
0.0026-0.0038 of the cell, under a 0.004 cut-off, so real samples vanished
with no error raised.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ing = _load("ingest_forms")
mf = _load("make_forms")


def _cell(draw=None, h=130, w=950):
    """A blank cell, optionally with something drawn into it."""
    img = np.full((h, w), 245, np.uint8)
    if draw is not None:
        draw(img)
    return img


def _canon_with(cell):
    """Embed a cell into a canonical-sized frame at a known normalised box."""
    canon = np.full((ing.CANON_H, ing.CANON_W), 250, np.uint8)
    y0, x0 = 400, 200
    canon[y0:y0 + cell.shape[0], x0:x0 + cell.shape[1]] = cell
    box = [x0 / ing.CANON_W, y0 / ing.CANON_H,
           (x0 + cell.shape[1]) / ing.CANON_W, (y0 + cell.shape[0]) / ing.CANON_H]
    return canon, box


def test_blank_cell_is_dropped():
    canon, box = _canon_with(_cell())
    assert ing.extract_cell(canon, box) is None


def test_single_speck_is_dropped():
    def speck(img):
        cv2.circle(img, (400, 60), 2, 20, -1)
    canon, box = _canon_with(_cell(speck))
    assert ing.extract_cell(canon, box) is None, "a dust mote must not count as writing"


def test_faint_short_word_is_kept():
    """The regression. A short word with a thin stroke covers very little area;
    it must still survive."""
    def word(img):
        cv2.putText(img, "losec", (60, 95), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    1.6, 70, 2, cv2.LINE_AA)
    cell = _cell(word)
    ink = (cell < 160).mean()
    assert ink < 0.02, f"test fixture should be sparse, got {ink:.4f}"
    canon, box = _canon_with(cell)
    out = ing.extract_cell(canon, box)
    assert out is not None, f"genuine writing dropped (ink fraction {ink:.5f})"
    assert out.shape[0] == ing.TARGET_H


def test_crop_is_trimmed_to_ink_not_to_the_box():
    def word(img):
        cv2.putText(img, "dopamine", (60, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 40, 3)
    canon, box = _canon_with(_cell(word))
    out = ing.extract_cell(canon, box)
    assert out is not None
    # the word occupies well under half the 950px-wide cell
    assert out.shape[1] < 0.7 * 950, "crop was not trimmed to the ink"


def test_target_height_matches_synthetic_pipeline():
    from lasa.data.synth import TARGET_HEIGHT
    assert ing.TARGET_H == TARGET_HEIGHT, \
        "real and synthetic crops must share a height or the loaders disagree"


def test_writer_independent_splits():
    writers = [f"W{i:02d}" for i in range(1, 13)]
    sp = ing.assign_splits(writers, seed=1337)
    assert set(sp) == set(writers)
    by_split = {}
    for w, s in sp.items():
        by_split.setdefault(s, set()).add(w)
    seen = set()
    for s, ws in by_split.items():
        assert not (ws & seen), "a writer appears in two splits"
        seen |= ws
    assert "train" in by_split and "test" in by_split


def test_form_cells_are_inside_the_marker_frame():
    """Normalised boxes must lie within [0,1]; outside means the layout has
    drifted past the markers and rectification will crop writing away."""
    import json, tempfile
    with tempfile.TemporaryDirectory() as d:
        form = mf.build_form("LRG-999", ["dopamine"] * (mf.COLS * mf.ROWS), Path(d))
    assert form["n_cells"] == mf.COLS * mf.ROWS
    for c in form["cells"]:
        x0, y0, x1, y1 = c["box"]
        assert 0.0 <= x0 < x1 <= 1.0, f"cell {c['index']} x out of frame: {c['box']}"
        assert 0.0 <= y0 < y1 <= 1.0, f"cell {c['index']} y out of frame: {c['box']}"
