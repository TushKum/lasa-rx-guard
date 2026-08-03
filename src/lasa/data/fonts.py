"""Fetch handwriting fonts used to synthesise handwritten drug-name images.

All default families come from Google Fonts and are **SIL Open Font License
1.1** or Apache-2.0 — redistributable, and identically available on Colab, so a
run there reproduces a run here. That is the whole reason we do not rely on the
macOS system handwriting fonts by default: ``Bradley Hand``/``Zapfino`` are
Apple-licensed and simply do not exist on a Colab VM, which would silently
change the training distribution between machines.

``--include-system`` opts into local macOS fonts for extra diversity. If you use
it, the manifest records it and ``results.md`` must say so, because the run is
then not reproducible on Colab.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

CSS_API = "https://fonts.googleapis.com/css"
# A UA without woff2 support makes the v1 CSS API hand back plain TTF URLs.
_LEGACY_UA = "Mozilla/5.0"

# Chosen for *stylistic spread*, not beauty: upright print, slanted print,
# looping cursive, marker, and deliberately messy scrawl. Doctor handwriting is
# mostly fast cursive, so cursive families are over-represented on purpose.
GOOGLE_HANDWRITING_FAMILIES: tuple[str, ...] = (
    # print-ish / neat
    "Caveat", "Patrick Hand", "Architects Daughter", "Coming Soon", "Handlee",
    "Kalam", "Shadows Into Light", "Indie Flower", "Gloria Hallelujah",
    "Covered By Your Grace",
    # marker / heavy
    "Permanent Marker", "Rock Salt", "Caveat Brush",
    # cursive / looping
    "Dancing Script", "Sacramento", "Cedarville Cursive", "Bad Script",
    "Marck Script", "Zeyada", "Nothing You Could Do", "La Belle Aurore",
    # fast / scrawled — closest to real prescription handwriting
    "Just Another Hand", "Reenie Beanie", "Homemade Apple", "Nanum Pen Script",
)

# Only used with --include-system. Names are macOS Supplemental font files.
MACOS_HANDWRITING = (
    "Bradley Hand Bold.ttf", "Brush Script.ttf", "Chalkduster.ttf",
    "Comic Sans MS.ttf", "SnellRoundhand.ttc", "Trattatello.ttf",
    "Chalkboard.ttc",
)
_MACOS_FONT_DIR = Path("/System/Library/Fonts/Supplemental")


@dataclass
class FontRecord:
    family: str
    filename: str
    source: str        # "google-fonts" | "macos-system"
    license: str
    sha256: str
    portable: bool     # True => available on Colab too


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _parse_css(css: str) -> dict[str, str]:
    """Pull `family -> ttf url` out of a Google Fonts CSS payload.

    The CSS is a sequence of ``@font-face { font-family: 'X'; ... src:
    url(...ttf) }`` blocks (one per unicode subset), so we split on the
    at-rule and keep the first URL seen for each family.
    """
    out: dict[str, str] = {}
    for block in css.split("@font-face"):
        fam = re.search(r"font-family:\s*'([^']+)'", block)
        url = re.search(r"url\((https://[^)\s]+\.ttf)\)", block)
        if fam and url and fam.group(1) not in out:
            out[fam.group(1)] = url.group(1)
    return out


def _fetch_css(query: str) -> str:
    # NOTE: the URL is built by hand rather than via `params=`. requests
    # percent-encodes `+` to `%2B`, and the Fonts API reads `+` as a literal
    # space in a family name — so `params=` silently drops every multi-word
    # family ("Patrick Hand", "Rock Salt", ...).
    r = requests.get(
        f"{CSS_API}?family={query}", headers={"User-Agent": _LEGACY_UA}, timeout=60
    )
    r.raise_for_status()
    return r.text


def _resolve_ttf_urls(families: tuple[str, ...]) -> dict[str, str]:
    """family -> ttf url, via the Google Fonts v1 CSS API."""
    out = _parse_css(_fetch_css("|".join(f.replace(" ", "+") for f in families)))

    # Retry stragglers individually — one bad family in a batch can otherwise
    # take the whole batch's subset blocks with it.
    for fam in families:
        if fam not in out:
            try:
                out.update(_parse_css(_fetch_css(fam.replace(" ", "+"))))
            except requests.RequestException:
                pass
    return {k: v for k, v in out.items() if k in set(families)}


def download_fonts(
    dest: Path,
    families: tuple[str, ...] = GOOGLE_HANDWRITING_FAMILIES,
    include_system: bool = False,
) -> list[FontRecord]:
    dest.mkdir(parents=True, exist_ok=True)
    records: list[FontRecord] = []

    urls = _resolve_ttf_urls(families)
    missing = [f for f in families if f not in urls]
    if missing:
        print(f"  ! Google Fonts did not resolve: {missing}")

    for family, url in sorted(urls.items()):
        fname = family.replace(" ", "") + ".ttf"
        path = dest / fname
        if not path.exists():
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        records.append(
            FontRecord(family, fname, "google-fonts", "OFL-1.1/Apache-2.0",
                       _sha256(path), portable=True)
        )

    if include_system:
        for fname in MACOS_HANDWRITING:
            src = _MACOS_FONT_DIR / fname
            if not src.exists():
                continue
            dst = dest / fname
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
            records.append(
                FontRecord(Path(fname).stem, fname, "macos-system",
                           "Apple-proprietary (local use only)",
                           _sha256(dst), portable=False)
            )

    manifest = dest / "manifest.json"
    manifest.write_text(json.dumps([asdict(r) for r in records], indent=2))
    return records


def load_manifest(dest: Path) -> list[FontRecord]:
    manifest = dest / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"No font manifest at {manifest}. Run: python -m lasa.data.fonts"
        )
    return [FontRecord(**r) for r in json.loads(manifest.read_text())]


def main() -> None:
    from lasa.common.paths import FONTS, ensure_dirs

    ap = argparse.ArgumentParser(description="Download handwriting fonts.")
    ap.add_argument("--include-system", action="store_true",
                    help="also copy macOS handwriting fonts (NOT Colab-portable)")
    args = ap.parse_args()

    ensure_dirs()
    recs = download_fonts(FONTS, include_system=args.include_system)
    portable = sum(r.portable for r in recs)
    print(f"{len(recs)} fonts -> {FONTS}  ({portable} portable / "
          f"{len(recs) - portable} local-only)")
    for r in recs:
        print(f"  {r.family:<26} {r.source:<14} {r.sha256[:12]}")


if __name__ == "__main__":
    main()
