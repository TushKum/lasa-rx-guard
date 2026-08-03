"""Canonical project paths, resolved relative to the repo root.

Import these instead of hard-coding relative paths so scripts work identically
whether they are run from the repo root, from ``scripts/``, or from a Colab
cell that has ``%cd``-ed somewhere unexpected.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

DATA = ROOT / "data"
RAW = DATA / "raw"
FONTS = DATA / "fonts"
SYNTH = DATA / "synth"
IAM = DATA / "iam"
PROCESSED = DATA / "processed"

CONFIGS = ROOT / "configs"
OUTPUTS = ROOT / "outputs"
CHECKPOINTS = OUTPUTS / "checkpoints"
FIGURES = OUTPUTS / "figures"
REPORTS = OUTPUTS / "reports"
DOCS = ROOT / "docs"

# Canonical data artefacts produced by the pipeline.
DRUG_NAMES_CSV = PROCESSED / "drug_names.csv"
LASA_PAIRS_CSV = PROCESSED / "lasa_pairs.csv"
FONT_MANIFEST = FONTS / "manifest.json"


def ensure_dirs() -> None:
    for p in (RAW, FONTS, SYNTH, IAM, PROCESSED, CHECKPOINTS, FIGURES, REPORTS):
        p.mkdir(parents=True, exist_ok=True)
