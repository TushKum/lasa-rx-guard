"""Fine-tune TrOCR on the synthetic handwritten drug-name corpus.

A plain manual loop rather than ``Seq2SeqTrainer``. The Trainer buys checkpoint
plumbing and distributed support we do not need here, and costs visibility into
exactly what runs on which device — which matters because this project has to
behave on CUDA (Colab T4), MPS (local Apple Silicon) and CPU, and the MPS path
has enough sharp edges already. The loop is short enough to audit.

Trains on the *same* samples in the same order as the CRNN for a given seed, so
any difference in the reported metrics is attributable to the model rather than
to the data pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

from lasa.common.paths import CHECKPOINTS, SYNTH, ensure_dirs
from lasa.common.seed import set_seed
from lasa.data.datasets import WordImageDataset, make_trocr_collate, read_manifest
from lasa.evaluation.metrics import summarize
from lasa.models.trocr import DEFAULT_MODEL, auto_device, load_trocr


@torch.no_grad()
def evaluate(model, processor, loader, device, num_beams: int,
             max_batches: int | None = None) -> tuple:
    model.eval()
    refs: list[str] = []
    hyps: list[str] = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        ids = model.generate(batch["pixel_values"].to(device), num_beams=num_beams)
        hyps.extend(s.strip().lower()
                    for s in processor.batch_decode(ids, skip_special_tokens=True))
        refs.extend(batch["texts"])
    model.train()
    return summarize(refs, hyps), refs, hyps


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune TrOCR.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="use microsoft/trocr-small-handwritten for low VRAM / MPS")
    ap.add_argument("--data", type=Path, default=SYNTH)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-train", type=int, default=20000,
                    help="seeded subsample; a full 48k epoch does not fit a T4 slot")
    ap.add_argument("--max-eval", type=int, default=500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=110.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, default=CHECKPOINTS / "trocr_ft")
    args = ap.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = auto_device(args.device)

    train_samples = read_manifest(args.data, splits=("train",))[: args.max_train]
    val_samples = read_manifest(args.data, splits=("val",))[: args.max_eval]

    model, processor = load_trocr(args.model, device, num_beams=args.num_beams)
    collate = make_trocr_collate(processor)

    train_loader = DataLoader(
        WordImageDataset(train_samples), batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.workers,
    )
    val_loader = DataLoader(
        WordImageDataset(val_samples), batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.workers,
    )

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(total_steps, 1),
        pct_start=args.warmup_ratio, anneal_strategy="linear",
    )
    # fp16 GradScaler is CUDA-only; MPS/CPU run fp32.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model       : {args.model} ({n_params/1e6:.0f}M params)")
    print(f"device      : {device} | amp fp16: {use_amp}")
    print(f"train/val   : {len(train_samples)} / {len(val_samples)}")
    print(f"steps/epoch : {steps_per_epoch} (batch {args.batch_size} "
          f"x accum {args.grad_accum})\n")

    best_cer = float("inf")
    history: list[dict] = []
    t_start = time.perf_counter()
    stop_reason = "completed"

    for epoch in range(args.epochs):
        t_epoch = time.perf_counter()
        running = 0.0
        n = 0
        opt.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(pixel_values=batch["pixel_values"].to(device),
                            labels=batch["labels"].to(device))
                loss = out.loss / args.grad_accum

            scaler.scale(loss).backward() if use_amp else loss.backward()
            running += out.loss.item()
            n += 1

            if (i + 1) % args.grad_accum == 0:
                if use_amp:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if use_amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch < total_steps - 1:
                    sched.step()

            if i % 100 == 0:
                print(f"  e{epoch:02d} step {i:5d}/{len(train_loader)} "
                      f"loss={running / max(n,1):.4f} lr={sched.get_last_lr()[0]:.2e}",
                      flush=True)

        metrics, _, _ = evaluate(model, processor, val_loader, device, args.num_beams)
        dt = time.perf_counter() - t_epoch
        elapsed = (time.perf_counter() - t_start) / 60.0
        print(f"epoch {epoch:02d} | loss {running/max(n,1):.4f} | {metrics.as_row()} "
              f"| {dt:.0f}s | elapsed {elapsed:.1f}m", flush=True)

        history.append({"epoch": epoch, "train_loss": running / max(n, 1),
                        "seconds": dt, **metrics.as_dict()})
        (args.out / "history.json").write_text(json.dumps(history, indent=2))

        if metrics.cer < best_cer:
            best_cer = metrics.cer
            model.save_pretrained(args.out)
            processor.save_pretrained(args.out)
            print(f"  * new best val CER {best_cer:.4%} -> {args.out}", flush=True)

        if epoch == 0:
            print(f"  projected total: {dt * args.epochs / 60:.0f} min "
                  f"for {args.epochs} epochs")
        if args.max_minutes and elapsed > args.max_minutes:
            stop_reason = f"hit --max-minutes ({args.max_minutes})"
            print(f"  stopping: {stop_reason}")
            break

    total = (time.perf_counter() - t_start) / 60.0
    print(f"\ndone ({stop_reason}) in {total:.1f} min | best val CER {best_cer:.4%}")
    (args.out / "summary.json").write_text(json.dumps({
        "model": args.model, "best_val_cer": best_cer, "minutes": total,
        "epochs_run": len(history), "stop_reason": stop_reason,
        "device": str(device), "params": n_params, "amp_fp16": use_amp,
        "max_train": args.max_train, "seed": args.seed,
    }, indent=2))


if __name__ == "__main__":
    main()
