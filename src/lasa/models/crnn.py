"""CRNN (VGG-style CNN + BiLSTM + CTC) trained from scratch.

This is the "small model, no pretraining" arm of the Stage-1 comparison against
a fine-tuned TrOCR. It exists to answer a concrete question for the write-up:
how much of TrOCR's accuracy comes from 300M+ pretrained parameters versus the
task itself? A ~8M-parameter CRNN that fits in minutes-per-epoch on a free T4
is the honest baseline for that.

Geometry contract
-----------------
The backbone must map a 32-pixel-high input to a height of exactly 1 and reduce
width by exactly 4, because :func:`lasa.data.datasets.make_ctc_collate` computes
``input_lengths = width // 4`` and hands that straight to ``nn.CTCLoss``. If the
real factor drifted, the loss would either silently ignore real frames or claim
frames that do not exist. Height pooling is therefore 2/2/2/2 while width
pooling happens only in the first two blocks; the final ``(H,1)`` convolution
collapses height without touching width. Nested floor division makes this exact
for odd widths too: ``⌊⌊W/2⌋/2⌋ == ⌊W/4⌋``. :func:`measure_width_downsample`
checks it empirically rather than trusting the arithmetic.

Devices
-------
Forward and backward run on CUDA, MPS and CPU. One caveat for whoever writes the
trainer: as of torch 2.8 ``aten::_ctc_loss`` is *not* implemented on MPS
(NotImplementedError), so on Apple silicon the loss has to be taken on CPU
tensors (or ``PYTORCH_ENABLE_MPS_FALLBACK=1`` set). CUDA and CPU are unaffected.

Decoding
--------
Three conditions, deliberately kept separate:

* :func:`greedy_decode` — argmax per frame, collapse repeats, drop blanks.
* :func:`beam_search_decode` — CTC prefix beam search, which sums the
  probability of every alignment that produces the same label string instead of
  scoring one path.
* :func:`lexicon_constrained_decode` — beam search snapped to the pharmacy
  formulary. A real dispensing counter has a fixed drug list, so this is the
  operationally relevant number. It is also an *easier task* — the output space
  shrinks from |charset|^L to ~2000 strings — so it MUST be reported as its own
  condition and never folded into the free-decoding CER.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import editdistance
import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:  # avoid pulling cv2 in at inference time
    from lasa.data.datasets import Charset

# Must match ``downsample`` in lasa.data.datasets.make_ctc_collate.
WIDTH_DOWNSAMPLE = 4
NEG_INF = -float("inf")


# ------------------------------------------------------------------ model --


@dataclass
class CRNNConfig:
    """Architecture hyperparameters, stored alongside checkpoints.

    Kept separate from the module so a checkpoint can be reloaded without
    remembering which flags the training run used.
    """

    n_classes: int
    img_h: int = 32
    n_channels: int = 1
    hidden: int = 256
    dropout: float = 0.1

    def build(self) -> "CRNN":
        return CRNN(self.n_classes, self.img_h, self.n_channels,
                    self.hidden, self.dropout)


def _conv_bn_relu(in_c: int, out_c: int) -> list[nn.Module]:
    # bias=False because the BatchNorm that follows has its own shift.
    return [
        nn.Conv2d(in_c, out_c, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    ]


class CRNN(nn.Module):
    """CNN feature extractor -> 2-layer BiLSTM -> per-frame class posteriors."""

    def __init__(self, n_classes: int, img_h: int = 32, n_channels: int = 1,
                 hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        if img_h % 16 != 0:
            raise ValueError(
                f"img_h must be divisible by 16 (four height poolings), got {img_h}"
            )
        self.n_classes = n_classes
        self.img_h = img_h
        self.hidden = hidden

        c1, c2, c3, c4 = 64, 128, 256, 512
        self.cnn = nn.Sequential(
            # --- width /2, height /2
            *_conv_bn_relu(n_channels, c1),
            *_conv_bn_relu(c1, c1),
            nn.MaxPool2d(2, 2),
            # --- width /4, height /4
            *_conv_bn_relu(c1, c2),
            *_conv_bn_relu(c2, c2),
            nn.MaxPool2d(2, 2),
            # --- height only from here on: width must stay at /4
            *_conv_bn_relu(c2, c3),
            *_conv_bn_relu(c3, c3),
            nn.MaxPool2d((2, 1), (2, 1)),
            *_conv_bn_relu(c3, c4),
            *_conv_bn_relu(c4, c4),
            nn.MaxPool2d((2, 1), (2, 1)),
            # Collapse the remaining height in one shot. A (h,1) kernel with no
            # padding leaves the width axis untouched, unlike the classic CRNN's
            # 2x2 final conv which shifts T by one and breaks the //4 contract.
            nn.Conv2d(c4, c4, kernel_size=(img_h // 16, 1), bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

        self.rnn = nn.LSTM(
            input_size=c4, hidden_size=hidden, num_layers=2,
            bidirectional=True, dropout=dropout if dropout > 0 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> log-probabilities [T, B, n_classes] (T-first)."""
        if x.dim() != 4:
            raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
        if x.size(2) != self.img_h:
            raise ValueError(f"expected height {self.img_h}, got {x.size(2)}")

        f = self.cnn(x)                       # [B, 512, 1, W/4]
        if f.size(2) != 1:
            raise RuntimeError(
                f"backbone left height {f.size(2)}, expected 1 — geometry broken"
            )
        f = f.squeeze(2).permute(2, 0, 1)     # [T, B, 512]
        f, _ = self.rnn(f)
        f = self.dropout(f)
        logits = self.fc(f)
        # log_softmax in fp32: under AMP the fp16 version underflows and CTCLoss
        # returns inf/nan a few hundred steps in.
        return logits.float().log_softmax(2)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_width_downsample(model: CRNN, widths: Sequence[int] = (32, 64, 127, 320),
                             device: torch.device | str = "cpu") -> int:
    """Run real forward passes and return the observed width stride.

    Raises if any width disagrees with ``W // WIDTH_DOWNSAMPLE``. Called by
    :func:`main` and by the unit test so the collate contract is checked against
    behaviour, not against a comment. Leaves the model on ``device``.
    """
    was_training = model.training
    model = model.to(device).eval()
    try:
        with torch.no_grad():
            for w in widths:
                x = torch.zeros(1, 1, model.img_h, w, device=device)
                t = model(x).size(0)
                expected = w // WIDTH_DOWNSAMPLE
                if t != expected:
                    raise AssertionError(
                        f"width {w}: T={t}, expected {expected} "
                        f"(downsample must be exactly {WIDTH_DOWNSAMPLE})"
                    )
    finally:
        model.train(was_training)
    return WIDTH_DOWNSAMPLE


# --------------------------------------------------------------- decoding --


def _to_numpy(log_probs: torch.Tensor) -> np.ndarray:
    return log_probs.detach().float().cpu().numpy()


def _lengths(log_probs: torch.Tensor,
             input_lengths: torch.Tensor | Sequence[int] | None) -> list[int]:
    t_max, batch = log_probs.shape[0], log_probs.shape[1]
    if input_lengths is None:
        return [t_max] * batch
    lens = [int(v) for v in input_lengths]
    if len(lens) != batch:
        raise ValueError(f"input_lengths has {len(lens)} entries, batch is {batch}")
    return [max(1, min(t_max, v)) for v in lens]


@torch.no_grad()
def greedy_decode(log_probs: torch.Tensor, charset: "Charset",
                  input_lengths: torch.Tensor | Sequence[int] | None = None
                  ) -> list[str]:
    """Best-path decode: argmax per frame, collapse repeats, drop blanks.

    ``input_lengths`` should be passed whenever the batch came from the padded
    CTC collate — the frames past a sample's true width are pure padding and
    happily emit characters.
    """
    blank = getattr(charset, "BLANK", 0)
    ids = log_probs.detach().argmax(dim=2).cpu().numpy()  # [T, B]
    lens = _lengths(log_probs, input_lengths)

    out: list[str] = []
    for b in range(ids.shape[1]):
        prev = -1
        kept: list[int] = []
        for k in ids[: lens[b], b]:
            k = int(k)
            if k != prev and k != blank:
                kept.append(k)
            prev = k
        out.append(charset.decode(kept))
    return out


def _logaddexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _prefix_beam_search(logp: np.ndarray, beam_width: int, blank: int,
                        prune_k: int) -> list[tuple[tuple[int, ...], float]]:
    """CTC prefix beam search over one sample's [T, C] log-probabilities.

    Returns ``(label_ids, log_score)`` sorted best-first, where the score is the
    summed probability of *all* alignments collapsing to that label sequence —
    which is the quantity greedy decoding approximates with a single path.

    Each beam carries two scores: probability of the prefix ending in a blank
    and of it ending in a non-blank. That split is what lets the algorithm know
    whether the next repeated character extends the label ("aa") or merges into
    the previous one ("a").

    Only the top ``prune_k`` symbols per frame are considered. With a ~30-symbol
    charset the discarded mass is negligible and it keeps the loop linear in T.
    """
    n_classes = logp.shape[1]
    prune_k = max(2, min(prune_k, n_classes))

    empty: tuple[int, ...] = ()
    beams: dict[tuple[int, ...], list[float]] = {empty: [0.0, NEG_INF]}

    for t in range(logp.shape[0]):
        row = logp[t]
        if prune_k < n_classes:
            cand = np.argpartition(-row, prune_k - 1)[:prune_k]
        else:
            cand = np.arange(n_classes)
        cand_syms = [int(c) for c in cand if int(c) != blank]
        p_blank_t = float(row[blank])

        nxt: dict[tuple[int, ...], list[float]] = {}

        def slot(prefix: tuple[int, ...]) -> list[float]:
            s = nxt.get(prefix)
            if s is None:
                s = [NEG_INF, NEG_INF]
                nxt[prefix] = s
            return s

        for prefix, (p_b, p_nb) in beams.items():
            p_tot = _logaddexp(p_b, p_nb)

            # Emit blank -> label sequence unchanged, path now ends in blank.
            s = slot(prefix)
            s[0] = _logaddexp(s[0], p_tot + p_blank_t)

            last = prefix[-1] if prefix else None
            for c in cand_syms:
                p_c = float(row[c])
                if c == last:
                    # Repeat with no blank in between: merges into the existing
                    # character, so the label sequence does not grow.
                    s = slot(prefix)
                    s[1] = _logaddexp(s[1], p_nb + p_c)
                    # Genuine double letter only if a blank separated them.
                    ext = slot(prefix + (c,))
                    ext[1] = _logaddexp(ext[1], p_b + p_c)
                else:
                    ext = slot(prefix + (c,))
                    ext[1] = _logaddexp(ext[1], p_tot + p_c)

        # Prune to the beam width by total probability. Sorting on the tuple as
        # a secondary key keeps the result deterministic across runs.
        ranked = sorted(nxt.items(),
                        key=lambda kv: (-_logaddexp(kv[1][0], kv[1][1]), kv[0]))
        beams = {k: v for k, v in ranked[:beam_width]}

    scored = [(k, _logaddexp(v[0], v[1])) for k, v in beams.items()]
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return scored


@torch.no_grad()
def beam_search_decode(log_probs: torch.Tensor, charset: "Charset",
                       beam_width: int = 10,
                       input_lengths: torch.Tensor | Sequence[int] | None = None
                       ) -> list[str]:
    """CTC prefix beam search. Pure numpy/PyTorch, no external decoder."""
    blank = getattr(charset, "BLANK", 0)
    arr = _to_numpy(log_probs)
    lens = _lengths(log_probs, input_lengths)

    out: list[str] = []
    for b in range(arr.shape[1]):
        hyps = _prefix_beam_search(arr[: lens[b], b, :], beam_width, blank,
                                   prune_k=max(2 * beam_width, 8))
        out.append(charset.decode(list(hyps[0][0])) if hyps else "")
    return out


def _normalised_edit_distance(a: str, b: str) -> float:
    # Normalised so a 1-char slip on "hct" is not treated the same as a 1-char
    # slip on "hydrochlorothiazide" when ranking formulary candidates.
    return editdistance.eval(a, b) / max(len(a), len(b), 1)


@torch.no_grad()
def lexicon_constrained_decode(log_probs: torch.Tensor, charset: "Charset",
                               lexicon: Sequence[str], beam_width: int = 10,
                               input_lengths: torch.Tensor | Sequence[int] | None = None
                               ) -> list[str]:
    """Beam search, then snap the output to the nearest formulary entry.

    The snap uses the *expected* normalised edit distance under the beam
    posterior rather than the distance to the single best hypothesis: if the
    model is torn between "lasix" and "lanoxin", both votes should count when
    choosing the formulary entry. That also avoids inventing a free weighting
    hyperparameter between acoustic score and edit distance.

    Report this separately from free decoding — constraining the output to ~2000
    known strings is a different, easier task, and mixing it into the headline
    CER would overstate the recogniser.
    """
    blank = getattr(charset, "BLANK", 0)
    arr = _to_numpy(log_probs)
    lens = _lengths(log_probs, input_lengths)

    entries = [w for w in lexicon if w]
    if not entries:
        return beam_search_decode(log_probs, charset, beam_width, input_lengths)
    keys = [w.strip().lower() for w in entries]
    exact = {k: entries[i] for i, k in enumerate(keys)}

    out: list[str] = []
    for b in range(arr.shape[1]):
        hyps = _prefix_beam_search(arr[: lens[b], b, :], beam_width, blank,
                                   prune_k=max(2 * beam_width, 8))
        if not hyps:
            out.append(entries[0])
            continue

        texts = [charset.decode(list(ids)).strip().lower() for ids, _ in hyps]
        if texts[0] in exact:  # fast path: already on the formulary
            out.append(exact[texts[0]])
            continue

        scores = np.array([s for _, s in hyps], dtype=np.float64)
        post = np.exp(scores - scores.max())
        post /= post.sum()

        best_i, best_cost = 0, float("inf")
        for i, key in enumerate(keys):
            cost = 0.0
            for w, text in zip(post, texts):
                if w < 1e-6:  # negligible vote, skip the edit-distance call
                    continue
                cost += w * _normalised_edit_distance(text, key)
                if cost >= best_cost:
                    break
            if cost < best_cost:  # strict: ties go to the earlier entry
                best_i, best_cost = i, cost
        out.append(entries[best_i])
    return out


# ------------------------------------------------------------------- main --


def _auto_device(name: str = "auto") -> torch.device:
    """CUDA on Colab, MPS on Apple silicon, CPU everywhere else."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    from lasa.common.paths import DRUG_NAMES_CSV
    from lasa.common.seed import set_seed
    from lasa.data.datasets import CRNN_HEIGHT, Charset

    ap = argparse.ArgumentParser(description="Smoke-test the CRNN architecture.")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--beam-width", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    device = _auto_device(args.device)

    # Real charset and formulary when the corpus is built; a plausible stand-in
    # otherwise, so this script never depends on the data pipeline having run.
    lexicon: list[str] = []
    if DRUG_NAMES_CSV.exists():
        from lasa.data.drug_names import load_csv

        lexicon = [d.name for d in load_csv(DRUG_NAMES_CSV)]
        charset = Charset.build(lexicon)
        source = f"{DRUG_NAMES_CSV.name} ({len(lexicon)} names)"
    else:
        charset = Charset("abcdefghijklmnopqrstuvwxyz0123456789 -")
        source = "fallback charset (drug_names.csv not built)"

    model = CRNNConfig(n_classes=len(charset), img_h=CRNN_HEIGHT,
                       hidden=args.hidden, dropout=args.dropout).build().to(device)
    model.eval()

    x = torch.randn(args.batch, 1, CRNN_HEIGHT, args.width, device=device)
    with torch.no_grad():
        log_probs = model(x)

    factor = measure_width_downsample(model, device=device)
    print(f"charset          : {source}, n_classes={len(charset)}")
    print(f"device           : {device}")
    print(f"input            : {tuple(x.shape)}")
    print(f"output log_probs : {tuple(log_probs.shape)}  (T, B, n_classes)")
    print(f"parameters       : {count_parameters(model):,}")
    print(f"width downsample : {factor} (measured; collate expects {WIDTH_DOWNSAMPLE})")

    # Untrained -> the strings below are noise. Printed only to prove the three
    # decoders run end to end on a real forward pass.
    lens = torch.full((args.batch,), log_probs.size(0), dtype=torch.long)
    print(f"greedy   (random init): {greedy_decode(log_probs, charset, lens)}")
    print(f"beam k={args.beam_width} (random init): "
          f"{beam_search_decode(log_probs, charset, args.beam_width, lens)}")
    if lexicon:
        snapped = lexicon_constrained_decode(log_probs, charset, lexicon,
                                             args.beam_width, lens)
        print(f"lexicon  (random init): {snapped}")


if __name__ == "__main__":
    main()
