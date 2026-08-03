"""Strip a training checkpoint down to what inference actually needs.

A training checkpoint carries the AdamW optimizer state (two momentum buffers
per parameter) and the scheduler state, which together are ~2x the size of the
weights. None of it is needed to run the model. Dropping it takes the CRNN
checkpoint from ~96 MB to ~32 MB — under GitHub's 50 MB warning threshold, and
a third of the download for anyone reproducing the demo.

    python scripts/export_weights.py
    python scripts/export_weights.py --half     # ~16 MB, fp16 weights
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def export(src: Path, dst: Path, half: bool = False) -> tuple[float, float]:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    state = ck["model"]
    if half:
        state = {k: (v.half() if v.is_floating_point() else v)
                 for k, v in state.items()}

    slim = {
        "model": state,
        "charset": ck["charset"],
        # Keep just enough provenance to know what this file is.
        "epoch": ck.get("epoch"),
        "best_cer": ck.get("best_cer"),
        "val_metrics": (ck.get("history") or [{}])[-1],
        "dtype": "float16" if half else "float32",
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dst)
    return src.stat().st_size / 1e6, dst.stat().st_size / 1e6


def main() -> None:
    from lasa.common.paths import CHECKPOINTS

    ap = argparse.ArgumentParser(description="Export inference-only weights.")
    ap.add_argument("--src", type=Path, default=CHECKPOINTS / "crnn_ctc" / "best.pt")
    ap.add_argument("--dst", type=Path, default=None)
    ap.add_argument("--half", action="store_true", help="cast weights to fp16")
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"no checkpoint at {args.src} — train first:\n"
                         f"  python -m lasa.train.train_crnn")
    dst = args.dst or args.src.with_name(
        "crnn_inference_fp16.pt" if args.half else "crnn_inference.pt"
    )
    before, after = export(args.src, dst, args.half)
    print(f"{args.src.name}: {before:.1f} MB -> {dst.name}: {after:.1f} MB "
          f"({100 * (1 - after / before):.0f}% smaller)")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
