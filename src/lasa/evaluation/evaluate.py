"""Stage-1 model comparison — produces the headline CRNN vs TrOCR table.

Decoding conditions are reported as separate rows, never merged. In particular
**lexicon-constrained decoding is a different task**: snapping output to the
nearest of 2000 known drug names cannot be compared like-for-like against free
transcription, even though it is the mode a real pharmacy would deploy. Folding
it into one "CER" column would be the single most misleading thing this report
could do.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from lasa.common.paths import CHECKPOINTS, DRUG_NAMES_CSV, REPORTS, SYNTH, ensure_dirs
from lasa.common.seed import set_seed
from lasa.data.datasets import (CRNN_HEIGHT, CRNN_MAX_WIDTH, Charset,
                                WordImageDataset, make_ctc_collate, read_manifest)
from lasa.evaluation.metrics import OcrMetrics, markdown_table, summarize

SPLITS = ("test_seen", "test_unseen_font", "test_unseen_word", "test_hard")


def _device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_crnn(ckpt_path: Path, device: torch.device):
    from lasa.models.crnn import CRNN
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    charset = Charset(ck["charset"])
    model = CRNN(n_classes=len(charset), img_h=CRNN_HEIGHT)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), charset, ck


def eval_crnn_split(model, charset, samples, device, lexicon, limit=None,
                    beam_width=10, batch_size=64):
    from lasa.models.crnn import (beam_search_decode, greedy_decode,
                                  lexicon_constrained_decode)
    if limit:
        samples = samples[:limit]
    ds = WordImageDataset(samples, charset)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=make_ctc_collate(charset, CRNN_HEIGHT,
                                                    CRNN_MAX_WIDTH))
    refs: list[str] = []
    fonts: list[str] = []
    out = {"greedy": [], "beam": [], "lexicon": []}
    with torch.no_grad():
        for batch in loader:
            lp = model(batch["images"].to(device)).cpu()
            il = batch["input_lengths"]
            out["greedy"].extend(greedy_decode(lp, charset, il))
            out["beam"].extend(beam_search_decode(lp, charset, il, beam_width=beam_width))
            out["lexicon"].extend(
                lexicon_constrained_decode(lp, charset, lexicon, il,
                                           beam_width=beam_width)
            )
            refs.extend(batch["texts"])
            fonts.extend(batch["fonts"])
    return refs, fonts, out


@torch.no_grad()
def eval_trocr_split(model, processor, samples, device, limit=None,
                     num_beams=4, batch_size=16):
    if limit:
        samples = samples[:limit]
    refs, hyps = [], []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        imgs = [cv2.cvtColor(cv2.imread(str(s.path), cv2.IMREAD_GRAYSCALE),
                             cv2.COLOR_GRAY2RGB) for s in chunk]
        pv = processor(images=imgs, return_tensors="pt").pixel_values.to(device)
        ids = model.generate(pv, num_beams=num_beams)
        hyps.extend(s.strip().lower()
                    for s in processor.batch_decode(ids, skip_special_tokens=True))
        refs.extend(s.text for s in chunk)
    return refs, hyps


def latency_ms(fn, n: int = 30) -> tuple[float, float]:
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts[3:])           # drop warmup
    return float(ts.mean()), float(np.percentile(ts, 95))


def breakdown(refs, hyps, fonts) -> dict:
    import editdistance
    by_font: dict[str, list[int]] = {}
    by_len: dict[str, list[int]] = {}
    for r, h, f in zip(refs, hyps, fonts):
        d = editdistance.eval(r, h)
        by_font.setdefault(f or "?", [0, 0])
        by_font[f][0] += d
        by_font[f][1] += len(r)
        bucket = "1-6" if len(r) <= 6 else "7-10" if len(r) <= 10 else "11-14" if len(r) <= 14 else "15+"
        by_len.setdefault(bucket, [0, 0])
        by_len[bucket][0] += d
        by_len[bucket][1] += len(r)
    return {
        "cer_by_font": {k: v[0] / max(v[1], 1) for k, v in sorted(by_font.items())},
        "cer_by_length": {k: v[0] / max(v[1], 1) for k, v in sorted(by_len.items())},
    }


def worst(refs, hyps, n=15) -> list[dict]:
    import editdistance
    rows = [{"ref": r, "hyp": h, "dist": editdistance.eval(r, h)}
            for r, h in zip(refs, hyps)]
    rows.sort(key=lambda d: (-d["dist"], d["ref"]))
    return rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare CRNN and TrOCR.")
    ap.add_argument("--crnn-ckpt", type=Path, default=CHECKPOINTS / "crnn_ctc" / "best.pt")
    ap.add_argument("--trocr-ckpt", type=Path, default=CHECKPOINTS / "trocr_ft")
    ap.add_argument("--data", type=Path, default=SYNTH)
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--beam-width", type=int, default=10)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    device = _device(args.device)

    from lasa.data.drug_names import load_csv
    lexicon = [d.name for d in load_csv(DRUG_NAMES_CSV)]

    have_crnn = args.crnn_ckpt.exists()
    have_trocr = (args.trocr_ckpt / "config.json").exists()
    if not have_crnn and not have_trocr:
        print("No trained checkpoints found.\n"
              f"  expected CRNN : {args.crnn_ckpt}\n"
              f"  expected TrOCR: {args.trocr_ckpt}\n\n"
              "Train them first:\n"
              "  python -m lasa.train.train_crnn\n"
              "  python -m lasa.train.train_trocr")
        raise SystemExit(1)
    if not have_crnn:
        print(f"! CRNN checkpoint missing ({args.crnn_ckpt}) — skipping that arm.")
    if not have_trocr:
        print(f"! TrOCR checkpoint missing ({args.trocr_ckpt}) — skipping that arm.")

    table: dict[str, dict[str, OcrMetrics]] = {}
    details: dict = {"device": str(device), "splits": args.splits, "models": {}}

    if have_crnn:
        model, charset, ck = load_crnn(args.crnn_ckpt, device)
        n_params = sum(p.numel() for p in model.parameters())
        size_mb = args.crnn_ckpt.stat().st_size / 1e6
        print(f"CRNN  : {n_params:,} params | ckpt {size_mb:.1f} MB | {device}")
        for split in args.splits:
            samples = read_manifest(args.data, splits=(split,))
            refs, fonts, outs = eval_crnn_split(model, charset, samples, device,
                                                lexicon, args.limit, args.beam_width)
            for mode in ("greedy", "beam", "lexicon"):
                key = f"CRNN ({mode})"
                table.setdefault(key, {})[split] = summarize(refs, outs[mode])
            details["models"].setdefault("crnn", {}).setdefault("breakdown", {})[split] = \
                breakdown(refs, outs["greedy"], fonts)
            details["models"]["crnn"].setdefault("worst", {})[split] = \
                worst(refs, outs["greedy"])
            print(f"  {split}: " + " | ".join(
                f"{m} CER {summarize(refs, outs[m]).cer:.2%}" for m in outs))

        one = WordImageDataset(read_manifest(args.data, splits=(args.splits[0],))[:1],
                               charset)[0]
        coll = make_ctc_collate(charset, CRNN_HEIGHT, CRNN_MAX_WIDTH)
        b = coll([one])
        mean_ms, p95_ms = latency_ms(lambda: model(b["images"].to(device)))
        details["models"]["crnn"].update(
            params=n_params, ckpt_mb=size_mb,
            latency_ms={"mean": mean_ms, "p95": p95_ms, "device": str(device),
                        "batch_size": 1, "decode": "excluded"},
            train_summary=json.loads((args.crnn_ckpt.parent / "summary.json").read_text())
            if (args.crnn_ckpt.parent / "summary.json").exists() else None,
        )
        print(f"  latency (fwd only, bs=1, {device}): {mean_ms:.1f} ms mean / "
              f"{p95_ms:.1f} ms p95")

    if have_trocr:
        from lasa.models.trocr import load_trocr
        model_t, proc = load_trocr(str(args.trocr_ckpt), device, num_beams=args.num_beams)
        n_params = sum(p.numel() for p in model_t.parameters())
        size_mb = sum(f.stat().st_size for f in args.trocr_ckpt.rglob("*")
                      if f.is_file()) / 1e6
        print(f"TrOCR : {n_params:,} params | ckpt {size_mb:.1f} MB | {device}")
        for split in args.splits:
            samples = read_manifest(args.data, splits=(split,))
            refs, hyps = eval_trocr_split(model_t, proc, samples, device,
                                          args.limit, args.num_beams)
            table.setdefault(f"TrOCR (beam {args.num_beams})", {})[split] = \
                summarize(refs, hyps)
            fonts = [s.font for s in (samples[:args.limit] if args.limit else samples)]
            details["models"].setdefault("trocr", {}).setdefault("breakdown", {})[split] = \
                breakdown(refs, hyps, fonts)
            details["models"]["trocr"].setdefault("worst", {})[split] = worst(refs, hyps)
            print(f"  {split}: CER {summarize(refs, hyps).cer:.2%}")

        s0 = read_manifest(args.data, splits=(args.splits[0],))[:1]
        img = [cv2.cvtColor(cv2.imread(str(s0[0].path), cv2.IMREAD_GRAYSCALE),
                            cv2.COLOR_GRAY2RGB)]
        pv = proc(images=img, return_tensors="pt").pixel_values.to(device)
        mean_ms, p95_ms = latency_ms(
            lambda: model_t.generate(pv, num_beams=args.num_beams), n=12)
        details["models"]["trocr"].update(
            params=n_params, ckpt_mb=size_mb,
            latency_ms={"mean": mean_ms, "p95": p95_ms, "device": str(device),
                        "batch_size": 1, "decode": f"beam {args.num_beams} included"},
        )
        print(f"  latency (generate, bs=1, {device}): {mean_ms:.1f} ms mean / "
              f"{p95_ms:.1f} ms p95")

    md = markdown_table(table)
    print("\n" + md)
    (REPORTS / "ocr_comparison.md").write_text(md)
    (REPORTS / "ocr_comparison.json").write_text(json.dumps(
        {"table": {m: {s: v.as_dict() for s, v in d.items()} for m, d in table.items()},
         "details": details}, indent=2))
    print(f"\nwrote {REPORTS/'ocr_comparison.md'} and {REPORTS/'ocr_comparison.json'}")


if __name__ == "__main__":
    main()
