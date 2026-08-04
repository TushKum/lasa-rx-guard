"""Capture screenshots of the running Streamlit demo for the slide deck.

Real screenshots rather than mockups: a deck that shows the actual UI is
verifiable, and a judge can reproduce it. Requires the app to already be
running (see --url) and `playwright install chromium`.

    python -m streamlit run app/streamlit_app.py --server.port 8577 &
    python scripts/capture_demo.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# Query -> (filename, expected traffic-light). These four cover every branch of
# the risk logic, which is why they are the ones demoed.
CASES = [
    ("dobutamin", "demo_red", "RED"),
    ("losec", "demo_yellow", "YELLOW"),
    ("paracetamol", "demo_green", "GREEN"),
    ("xyzzyqqq", "demo_unread", "YELLOW"),
]


def capture(url: str, out_dir: Path, width: int, height: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height},
                                device_scale_factor=2)  # retina, for print
        page.goto(url, wait_until="networkidle", timeout=60_000)
        page.wait_for_timeout(2500)

        # Switch to the Stage-2 tab, which drives the risk engine directly.
        page.get_by_role("tab").nth(1).click()
        page.wait_for_timeout(1500)

        for query, name, expect in CASES:
            box = page.get_by_role("textbox").first
            box.fill("")
            box.type(query, delay=25)
            box.press("Enter")
            page.wait_for_timeout(3000)

            body = page.inner_text("body")
            got = next((f for f in ("RED", "YELLOW", "GREEN") if f"{f} —" in body), "?")
            if got != expect:
                print(f"  ! {query}: expected {expect}, page shows {got}")

            path = out_dir / f"{name}.png"
            page.screenshot(path=str(path), full_page=True)
            written.append(path)
            print(f"  {query:<14} -> {got:<7} {path.name} "
                  f"({path.stat().st_size / 1024:.0f} KB)")

        browser.close()
    return written


def main() -> None:
    from lasa.common.paths import FIGURES

    ap = argparse.ArgumentParser(description="Screenshot the Streamlit demo.")
    ap.add_argument("--url", default="http://localhost:8577")
    ap.add_argument("--out", type=Path, default=FIGURES / "demo")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=1000)
    args = ap.parse_args()

    print(f"capturing {args.url} -> {args.out}")
    files = capture(args.url, args.out, args.width, args.height)
    print(f"\n{len(files)} screenshots written")


if __name__ == "__main__":
    main()
