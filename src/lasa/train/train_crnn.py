"""Train the from-scratch CRNN (CNN + BiLSTM + CTC).

Two implementation details that are not obvious and that dominate runtime:

**CTC on Apple Silicon.** ``aten::_ctc_loss`` is not implemented for the MPS
backend (torch 2.8), and ``PYTORCH_ENABLE_MPS_FALLBACK=1`` does not rescue it —
it raises at dispatch. Running everything on CPU instead is catastrophic:
measured on an M2, 15 img/s CPU-only vs 111 img/s with the conv/LSTM stack on
MPS and only the loss hopping to CPU. So on MPS we keep the model on device and
compute CTC on CPU. ``log_probs.cpu()`` stays in the autograd graph, so
gradients flow back correctly; the cost is one sync per step, which the 7x
throughput win pays for many times over. On CUDA none of this applies and the
loss runs on device.

**Width bucketing.** Word crops vary from ~30px to ~210px wide at height 32.
Batching them randomly pads everything to the batch maximum, so most of the
convolution runs on blank padding. Sorting within a shuffled pool before
forming batches keeps similar widths together and cuts that waste, without the
full-sort determinism/bias problems of sorting the whole epoch.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

from lasa.common.paths import CHECKPOINTS, SYNTH, ensure_dirs
from lasa.common.seed import set_seed, worker_init_fn
from lasa.data.datasets import (
    CRNN_HEIGHT,
    CRNN_MAX_WIDTH,
    Charset,
    WordImageDataset,
    make_ctc_collate,
    read_manifest,
)
from lasa.evaluation.metrics import summarize
from lasa.models.crnn import CRNN, count_parameters, greedy_decode


# ------------------------------------------------------------------ device --


@dataclass
class DevicePlan:
    device: torch.device
    ctc_on_cpu: bool
    amp: bool

    def describe(self) -> str:
        bits = [str(self.device)]
        if self.ctc_on_cpu:
            bits.append("CTC on CPU (MPS lacks aten::_ctc_loss)")
        if self.amp:
            bits.append("AMP fp16")
        return " | ".join(bits)


def plan_device(name: str = "auto", amp: bool = True) -> DevicePlan:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    dev = torch.device(name)
    return DevicePlan(
        device=dev,
        ctc_on_cpu=(dev.type == "mps"),
        amp=(amp and dev.type == "cuda"),
    )


# ----------------------------------------------------------------- sampler --


class BucketBatchSampler(Sampler[list[int]]):
    """Width-bucketed batching with a constant *pixel* budget per batch.

    Holding the sample count constant is what makes bucketing dangerous: the
    sampler sorts by width, so the widest bucket receives a full batch of
    near-maximum-width crops while the narrowest receives a full batch of tiny
    ones. Activation memory scales with ``count x width``, so the widest batch
    can be several times the median. That is exactly how the first full run died
    — 128 samples x ~512 px allocated 15.07 GiB and OOM'd MPS at epoch 6, and a
    16 GB T4 would have gone the same way.

    So the budget is ``batch_size x reference_width`` pixels, and each batch
    takes as many samples as fit under it. Wide batches get fewer samples, narrow
    batches get more. Peak memory becomes roughly flat across buckets, and
    throughput improves as a side effect because narrow batches stop being
    padded out to a fixed small count.
    """

    def __init__(self, widths: list[int], batch_size: int, seed: int = 1337,
                 pool_multiplier: int = 20, drop_last: bool = False,
                 max_batch_size: int | None = None):
        self.widths = np.asarray(widths)
        self.batch_size = batch_size
        self.seed = seed
        self.pool = batch_size * pool_multiplier
        self.drop_last = drop_last
        # Reference width = median, so the budget matches what a fixed batch of
        # `batch_size` typical samples would have used.
        self.reference_width = max(1, int(np.median(self.widths)))
        self.pixel_budget = batch_size * self.reference_width
        # Never exceed 2x the nominal count, or a batch of very narrow crops
        # could balloon enough to hurt gradient-noise scale.
        self.max_batch_size = max_batch_size or batch_size * 2
        self.epoch = 0
        self._cached: list[list[int]] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._cached = None

    def _build(self) -> list[list[int]]:
        if self._cached is not None:
            return self._cached
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.widths))
        batches: list[list[int]] = []

        for start in range(0, len(order), self.pool):
            pool = order[start:start + self.pool]
            pool = pool[np.argsort(self.widths[pool], kind="stable")]

            batch: list[int] = []
            widest = 0
            for idx in pool.tolist():
                w = int(self.widths[idx])
                new_widest = max(widest, w)
                # Would adding this sample blow the budget?
                if batch and ((len(batch) + 1) * new_widest > self.pixel_budget
                              or len(batch) + 1 > self.max_batch_size):
                    batches.append(batch)
                    batch, widest = [idx], w
                else:
                    batch.append(idx)
                    widest = new_widest
            if batch and not (self.drop_last and len(batch) < 2):
                batches.append(batch)

        order2 = rng.permutation(len(batches))
        self._cached = [batches[i] for i in order2]
        return self._cached

    def __iter__(self):
        yield from self._build()

    def __len__(self) -> int:
        return len(self._build())


# ---------------------------------------------------------------- training --


def ctc_step(model: CRNN, batch: dict, ctc: nn.CTCLoss, plan: DevicePlan) -> torch.Tensor:
    images = batch["images"].to(plan.device, non_blocking=True)
    log_probs = model(images)                        # [T, B, C]

    if plan.ctc_on_cpu:
        # .cpu() is differentiable — autograd routes the gradient back to MPS.
        log_probs = log_probs.cpu()
        targets = batch["targets"]
    else:
        targets = batch["targets"].to(plan.device, non_blocking=True)

    # Clamp so a crop narrower than its transcript cannot make CTC infeasible;
    # zero_infinity then keeps one bad sample from NaN-ing the whole batch.
    input_lengths = torch.minimum(
        batch["input_lengths"], torch.full_like(batch["input_lengths"], log_probs.shape[0])
    )
    return ctc(log_probs, targets, input_lengths, batch["target_lengths"])


@torch.no_grad()
def evaluate(model: CRNN, loader: DataLoader, charset: Charset,
             plan: DevicePlan, max_batches: int | None = None) -> tuple:
    model.eval()
    refs: list[str] = []
    hyps: list[str] = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        log_probs = model(batch["images"].to(plan.device)).cpu()
        hyps.extend(greedy_decode(log_probs, charset, batch["input_lengths"]))
        refs.extend(batch["texts"])
    model.train()
    return summarize(refs, hyps), refs, hyps


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the CRNN + CTC model.")
    ap.add_argument("--data", type=Path, default=SYNTH)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--preload", action="store_true",
                    help="decode the whole split into RAM once (much faster; "
                         "forces --workers 0)")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--val-batches", type=int, default=None,
                    help="cap val batches per epoch (full val at the end regardless)")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop cleanly once wall-clock exceeds this")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, default=CHECKPOINTS / "crnn_ctc")
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # --- data ---
    train_samples = read_manifest(args.data, splits=("train",))
    val_samples = read_manifest(args.data, splits=("val",))
    if args.limit_train:
        train_samples = train_samples[: args.limit_train]

    charset = Charset.build([s.text for s in train_samples])
    charset.save(args.out / "charset.json")

    # Preloading and worker processes are mutually exclusive: DataLoader workers
    # are spawned on macOS, so each would receive its own pickled copy of the
    # ~450 MB cache. In-RAM data needs no workers anyway.
    workers = 0 if args.preload else args.workers
    train_ds = WordImageDataset(train_samples, charset, preload=args.preload)
    val_ds = WordImageDataset(val_samples, charset, preload=args.preload)
    collate = make_ctc_collate(charset, height=CRNN_HEIGHT, max_width=CRNN_MAX_WIDTH)

    # Stored widths are at height 64; at CRNN_HEIGHT they scale by the same
    # factor, so relative order — all bucketing needs — is preserved.
    widths = [max(1, s.width or 128) for s in train_samples]
    sampler = BucketBatchSampler(widths, args.batch_size, seed=args.seed)

    train_loader = DataLoader(
        train_ds, batch_sampler=sampler, collate_fn=collate,
        num_workers=workers, worker_init_fn=worker_init_fn,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate,
        num_workers=workers, worker_init_fn=worker_init_fn,
    )

    # --- model ---
    plan = plan_device(args.device)
    model = CRNN(n_classes=len(charset), img_h=CRNN_HEIGHT,
                 hidden=args.hidden, dropout=args.dropout).to(plan.device)
    ctc = nn.CTCLoss(blank=Charset.BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.98))
    # Pixel-budget batching makes the per-epoch batch count vary slightly with
    # the shuffle, so pad total_steps by 5%. OneCycleLR raises if stepped past
    # total_steps, and losing a run at epoch 14 to a scheduler assertion would
    # be an absurd way to fail.
    total_steps = int(args.epochs * len(sampler) * 1.05) + 1
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.15, div_factor=25.0, final_div_factor=1000.0,
    )

    start_epoch = 0
    best_cer = float("inf")
    history: list[dict] = []
    if args.resume and args.resume.exists():
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best_cer = ck.get("best_cer", float("inf"))
        history = ck.get("history", [])
        print(f"resumed from {args.resume} @ epoch {start_epoch}")

    print(f"device      : {plan.describe()}")
    print(f"charset     : {len(charset)} classes -> {charset.chars!r}")
    print(f"train/val   : {len(train_ds)} / {len(val_ds)}")
    print(f"params      : {count_parameters(model):,}")
    print(f"steps/epoch : {len(sampler)}\n")

    t_start = time.perf_counter()
    stop_reason = "completed"

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        t_epoch = time.perf_counter()
        running = 0.0
        n_batches = 0

        for i, batch in enumerate(train_loader):
            opt.zero_grad(set_to_none=True)
            loss = ctc_step(model, batch, ctc, plan)
            if not torch.isfinite(loss):
                # Skip rather than poison the weights; report if it recurs.
                print(f"  ! non-finite loss at epoch {epoch} step {i}, skipping")
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if sched.last_epoch < total_steps - 1:
                sched.step()

            running += loss.item()
            n_batches += 1
            if i % 100 == 0:
                print(f"  e{epoch:02d} step {i:4d}/{len(sampler)} "
                      f"loss={running / max(n_batches, 1):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e}")

        train_loss = running / max(n_batches, 1)
        metrics, _, _ = evaluate(model, val_loader, charset, plan, args.val_batches)
        dt = time.perf_counter() - t_epoch
        elapsed = (time.perf_counter() - t_start) / 60.0

        print(f"epoch {epoch:02d} | loss {train_loss:.4f} | {metrics.as_row()} "
              f"| {dt:.0f}s | elapsed {elapsed:.1f}m")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "seconds": dt,
            **metrics.as_dict(),
        })
        (args.out / "history.json").write_text(json.dumps(history, indent=2))

        ck = {
            "model": model.state_dict(), "optim": opt.state_dict(),
            "sched": sched.state_dict(), "epoch": epoch, "best_cer": best_cer,
            "charset": charset.chars, "history": history,
            "args": vars(args) | {"data": str(args.data), "out": str(args.out),
                                  "resume": str(args.resume)},
        }
        torch.save(ck, args.out / "last.pt")
        if metrics.cer < best_cer:
            best_cer = metrics.cer
            ck["best_cer"] = best_cer
            torch.save(ck, args.out / "best.pt")
            print(f"  * new best val CER {best_cer:.4%} -> {args.out / 'best.pt'}")

        if epoch == start_epoch:
            projected = dt * args.epochs / 60.0
            print(f"  projected total: {projected:.0f} min for {args.epochs} epochs")
        if args.max_minutes and elapsed > args.max_minutes:
            stop_reason = f"hit --max-minutes ({args.max_minutes})"
            print(f"  stopping: {stop_reason}")
            break

    total = (time.perf_counter() - t_start) / 60.0
    print(f"\ndone ({stop_reason}) in {total:.1f} min | best val CER {best_cer:.4%}")
    (args.out / "summary.json").write_text(json.dumps({
        "best_val_cer": best_cer, "minutes": total, "epochs_run": len(history),
        "stop_reason": stop_reason, "device": str(plan.device),
        "ctc_on_cpu": plan.ctc_on_cpu, "params": count_parameters(model),
    }, indent=2))


if __name__ == "__main__":
    main()
