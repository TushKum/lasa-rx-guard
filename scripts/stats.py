"""Bootstrap confidence intervals and between-split significance tests.

The paper previously reported point estimates only. Two claims depend on the
uncertainty around them and cannot be made honestly without it:

  1. "held-out vocabulary costs nothing" -- requires showing the seen/unseen-word
     difference is consistent with zero.
  2. "held-out style costs 6.7x more" -- requires showing that difference is not.

We resample samples (not characters) with replacement, recomputing the
corpus-level micro-average CER on each draw, which is the quantity actually
reported. Differences are computed on *paired* draws where splits share an
index space and unpaired otherwise; splits here contain different samples, so
the difference bootstrap is unpaired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import editdistance
import numpy as np
import torch
from torch.utils.data import DataLoader

from lasa.common.paths import CHECKPOINTS, DRUG_NAMES_CSV, REPORTS, SYNTH, ensure_dirs
from lasa.common.seed import set_seed
from lasa.data.datasets import (CRNN_HEIGHT, CRNN_MAX_WIDTH, Charset,
                                WordImageDataset, make_ctc_collate, read_manifest)

SPLITS = ("test_seen", "test_unseen_font", "test_unseen_word", "test_hard")


def device_of(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect(model, charset, samples, device, lexicon, batch_size=64):
    """Return per-sample (ref, greedy_hyp, lexicon_hyp)."""
    from lasa.models.crnn import greedy_decode, lexicon_constrained_decode
    ds = WordImageDataset(samples, charset)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=make_ctc_collate(charset, CRNN_HEIGHT, CRNN_MAX_WIDTH))
    refs, g, lx = [], [], []
    with torch.no_grad():
        for b in loader:
            lp = model(b["images"].to(device)).cpu()
            il = b["input_lengths"]
            g.extend(greedy_decode(lp, charset, input_lengths=il))
            lx.extend(lexicon_constrained_decode(lp, charset, lexicon,
                                                 beam_width=8, input_lengths=il))
            refs.extend(b["texts"])
    return refs, g, lx


def _cer_arrays(refs, hyps):
    """Per-sample (edit distance, reference length) -- the micro-average's parts."""
    d = np.array([editdistance.eval(r.lower(), h.lower()) for r, h in zip(refs, hyps)], float)
    n = np.array([len(r) for r in refs], float)
    return d, n


def boot_cer(refs, hyps, n_boot, rng):
    d, n = _cer_arrays(refs, hyps)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    # micro-average per draw: sum(dist) / sum(chars)
    return d[idx].sum(1) / np.maximum(n[idx].sum(1), 1.0)


def boot_em(refs, hyps, n_boot, rng):
    e = np.array([r.lower() == h.lower() for r, h in zip(refs, hyps)], float)
    idx = rng.integers(0, len(e), size=(n_boot, len(e)))
    return e[idx].mean(1)


def ci(v, lo=2.5, hi=97.5):
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def main():
    ap = argparse.ArgumentParser(description="Bootstrap CIs for Stage-1 results.")
    ap.add_argument("--ckpt", type=Path, default=CHECKPOINTS / "crnn_ctc" / "best.pt")
    ap.add_argument("--data", type=Path, default=SYNTH)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    ensure_dirs(); set_seed(args.seed)
    dev = device_of(args.device)
    rng = np.random.default_rng(args.seed)

    from lasa.data.drug_names import load_csv
    from lasa.models.crnn import CRNN
    lexicon = [d.name for d in load_csv(DRUG_NAMES_CSV)]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    charset = Charset(ck["charset"])
    model = CRNN(n_classes=len(charset), img_h=CRNN_HEIGHT)
    model.load_state_dict(ck["model"]); model = model.to(dev).eval()

    preds, out = {}, {}
    for sp in SPLITS:
        s = read_manifest(args.data, splits=(sp,))
        if args.limit:
            s = s[: args.limit]
        refs, g, lx = collect(model, charset, s, dev, lexicon)
        preds[sp] = {"refs": refs, "greedy": g, "lexicon": lx}
        row = {"n": len(refs)}
        for mode, h in (("greedy", g), ("lexicon", lx)):
            bc, be = boot_cer(refs, h, args.n_boot, rng), boot_em(refs, h, args.n_boot, rng)
            d, n = _cer_arrays(refs, h)
            row[mode] = {
                "cer": float(d.sum() / max(n.sum(), 1)), "cer_ci95": ci(bc),
                "em": float(np.mean([r.lower() == x.lower() for r, x in zip(refs, h)])),
                "em_ci95": ci(be),
            }
        out[sp] = row
        print(f"{sp:<18} n={row['n']:<5} "
              f"greedy CER {row['greedy']['cer']:.4f} "
              f"[{row['greedy']['cer_ci95'][0]:.4f}, {row['greedy']['cer_ci95'][1]:.4f}]  "
              f"lexicon CER {row['lexicon']['cer']:.4f} "
              f"[{row['lexicon']['cer_ci95'][0]:.4f}, {row['lexicon']['cer_ci95'][1]:.4f}]", flush=True)

    # --- between-split differences (unpaired: splits hold different samples) ---
    print("\nDifferences in greedy CER (positive = second split worse):")
    diffs = {}
    base = "test_seen"
    bb = boot_cer(preds[base]["refs"], preds[base]["greedy"], args.n_boot, rng)
    for sp in SPLITS:
        if sp == base:
            continue
        ob = boot_cer(preds[sp]["refs"], preds[sp]["greedy"], args.n_boot, rng)
        dv = ob - bb
        lo, hi = ci(dv)
        crosses = lo <= 0.0 <= hi
        diffs[f"{sp}_minus_{base}"] = {
            "delta": float(np.mean(dv)), "ci95": [lo, hi],
            "ci_includes_zero": bool(crosses),
        }
        print(f"  {sp:<18} delta={np.mean(dv):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{'NOT significant (CI includes 0)' if crosses else 'significant'}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "stage1_ci.json").write_text(json.dumps(
        {"config": {k: str(v) for k, v in vars(args).items()},
         "per_split": out, "differences": diffs}, indent=2))
    (REPORTS / "stage1_predictions.json").write_text(json.dumps(preds))
    print(f"\nwrote {REPORTS/'stage1_ci.json'} and stage1_predictions.json")


if __name__ == "__main__":
    main()
