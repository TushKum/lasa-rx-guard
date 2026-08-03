"""Real handwriting from the IAM database, cut down to word-level images.

The synth pipeline (:mod:`lasa.data.synth`) has the right vocabulary but fake
ink: a font outline, however hard we jitter it, is still an outline. IAM is the
exact opposite — genuine pen-on-paper from hundreds of writers, but essentially
no medicine names in it. Mixing the two lets the recogniser pick up real stroke
and paper statistics from IAM while learning the formulary from synth.

IAM ships at *line* level (a whole sentence per image) and this project reads
single words, so lines have to be cut up. We do that with a vertical-projection
whitespace search on the binarised line, then pair the crops positionally with
the whitespace tokens of the transcript.

That pairing is only trusted when the crop count EXACTLY matches the token
count. Anything else — one word split in two, two words fused, a speck of
scanner noise counted as a word — shifts every later pairing on that line and
mislabels real handwriting. Wrong labels on real ink are far more damaging than
having fewer samples, so misaligned lines are dropped wholesale and counted;
``meta.json`` carries the discard rate so the report can quote it honestly.

Two sources, in order of preference:

    --local-words-dir   the official ``words.tgz`` + ``words.txt`` from
                        fki.tic.heia-fr.ch. Preferred whenever it is present,
                        because IAM's own word boxes are ground truth and need
                        no projection guessing at all (discard rate ~0).
                        Registration-gated, so it cannot be the default.
    HuggingFace         ``Teklia/IAM-line`` — public, ungated, MIT, no token
                        needed, which is what makes a Colab run reproducible
                        for anyone marking this project.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from lasa.data.synth import TARGET_HEIGHT, fit_height

HF_DATASET = "Teklia/IAM-line"
# Our split name -> the dataset's. Teklia ships 6482/976/2915 lines, i.e. the
# published IAM partition rather than an ad-hoc reshuffle.
HF_SPLIT_MAP = {"train": "train", "val": "validation", "test": "test"}

SPLITS = ("train", "val", "test")

# Manifest columns are shared with synth so `read_manifest` can concatenate the
# two sources. IAM has writers, not fonts, and carries no synthetic degradation.
FONT_TAG = "iam"
TIER_TAG = "real"

# This project recognises drug names: alphabetic, never one character long, and
# never longer than the longest INN (~"trastuzumab deruxtecan" territory).
MIN_TOKEN_LEN = 2
MAX_TOKEN_LEN = 24

# Random-split fractions, used only when no official partition is available.
RANDOM_SPLIT_FRACS = (0.80, 0.10, 0.10)


@dataclass
class SegmentConfig:
    """Vertical-projection word segmentation.

    Thresholds are fractions of the line's *core* height — the dense band of
    rows carrying the x-height letters — rather than of the image height. The HF
    renders are all resized to exactly 128 px, so image height carries no
    information about letter size: a line with tall ascenders has small letters
    in the same 128 px. Word spacing scales with letter size, so the core band
    is the only stable yardstick. See :func:`core_height`.

    ``gap_ratio`` is the one parameter that matters, and it is a genuine
    trade-off rather than a knob to turn up for yield: too small and joined-up
    cursive is chopped mid-word, too large and short words fuse with their
    neighbour. Either way the line fails the count check and is dropped, so a
    bad value costs samples, never label correctness.

    The default was picked by sweeping against a 500-line prefix (300 train +
    200 validation) of ``Teklia/IAM-line``, maximising the exact-alignment rate.
    The optimum is writer-dependent — that prefix's train writers peaked near
    0.70 and its validation writers near 0.42 — so no single value serves every
    hand, and roughly 60% of lines are lost whatever we choose. That ceiling is
    the price of inferring word boxes; ``--local-words-dir`` avoids it entirely.
    """

    gap_ratio: float = 0.65        # whitespace run wide enough to separate words
    core_frac: float = 0.50        # ink fraction defining the core band
    min_ink_px: int = 8            # below this a component is scanner noise
    pad_ratio: float = 0.05        # margin kept around each crop
    max_aspect: float = 40.0       # sanity guard on absurdly wide crops
    # Post-alignment plausibility band on width-per-character (see
    # `_width_consistent`). Equal counts alone do not prove equal *ordering*.
    wpc_lo: float = 0.40
    wpc_hi: float = 2.50


@dataclass
class BuildStats:
    """Counters behind the headline discard rate."""

    lines_seen: int = 0
    lines_aligned: int = 0
    tokens_total: int = 0          # tokens on aligned lines
    tokens_kept: int = 0           # survivors of the alphabetic filter
    rejects: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejects[reason] = self.rejects.get(reason, 0) + 1

    def merge(self, other: "BuildStats") -> "BuildStats":
        self.lines_seen += other.lines_seen
        self.lines_aligned += other.lines_aligned
        self.tokens_total += other.tokens_total
        self.tokens_kept += other.tokens_kept
        for k, v in other.rejects.items():
            self.rejects[k] = self.rejects.get(k, 0) + v
        return self

    @property
    def lines_discarded(self) -> int:
        return self.lines_seen - self.lines_aligned

    @property
    def discard_rate(self) -> float:
        return self.lines_discarded / self.lines_seen if self.lines_seen else 0.0

    def as_dict(self) -> dict:
        return {
            "lines_seen": self.lines_seen,
            "lines_aligned": self.lines_aligned,
            "lines_discarded": self.lines_discarded,
            "discard_rate": round(self.discard_rate, 4),
            "discard_reasons": dict(sorted(self.rejects.items())),
            "tokens_on_aligned_lines": self.tokens_total,
            "tokens_kept_after_filter": self.tokens_kept,
        }


# ------------------------------------------------------------ segmentation --


def binarise(line: np.ndarray) -> np.ndarray:
    """Otsu threshold a grayscale line into a uint8 mask where 255 is ink."""
    blur = cv2.GaussianBlur(line, (3, 3), 0)
    _, mask = cv2.threshold(blur, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return mask


def core_height(mask: np.ndarray, frac: float = 0.50) -> int:
    """Height of the band that carries the bulk of the ink, in pixels.

    Ascenders and descenders are sparse rows; the x-height body is dense. Taking
    the rows whose ink count reaches ``frac`` of the peak isolates that body,
    which is a far better proxy for letter size — and therefore for how wide a
    word gap should be — than the image height.
    """
    prof = mask.sum(axis=1).astype(np.float64)
    peak = prof.max()
    if peak <= 0:
        return mask.shape[0]
    band = np.flatnonzero(prof >= frac * peak)
    return max(4, int(band[-1] - band[0] + 1))


def _column_runs(mask: np.ndarray, gap_px: int) -> list[tuple[int, int]]:
    """Half-open [x0, x1) spans of ink, merging gaps narrower than ``gap_px``."""
    cols = mask.max(axis=0) > 0
    if not cols.any():
        return []

    # Sentinel-pad so a run touching either border still shows a rising and a
    # falling edge; without it the first/last word silently disappears.
    padded = np.concatenate(([False], cols, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)

    merged: list[tuple[int, int]] = []
    for a, b in zip(starts.tolist(), ends.tolist()):
        if merged and a - merged[-1][1] < gap_px:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def segment_line(gray: np.ndarray, cfg: SegmentConfig | None = None
                 ) -> list[np.ndarray]:
    """Cut a grayscale line image into word crops, left to right.

    Crops are tightened to their own ink bounding box (not just horizontally),
    because synth stores tightly-cropped words; leaving IAM's full 128 px
    ascender/descender band on a word like "one" would make it render tiny after
    height normalisation and hand the model a spurious cue.
    """
    cfg = cfg or SegmentConfig()
    mask = binarise(gray)
    core = core_height(mask, cfg.core_frac)
    gap_px = max(2, int(round(cfg.gap_ratio * core)))
    pad = max(1, int(round(cfg.pad_ratio * gray.shape[0])))

    crops: list[np.ndarray] = []
    for x0, x1 in _column_runs(mask, gap_px):
        band = mask[:, x0:x1]
        if int(np.count_nonzero(band)) < cfg.min_ink_px:
            continue  # dust, not a word
        rows = np.flatnonzero(band.max(axis=1) > 0)
        y0 = max(0, int(rows[0]) - pad)
        y1 = min(gray.shape[0], int(rows[-1]) + 1 + pad)
        cx0 = max(0, x0 - pad)
        cx1 = min(gray.shape[1], x1 + pad)
        crop = gray[y0:y1, cx0:cx1]
        if crop.shape[0] < 2 or crop.shape[1] < 2:
            continue
        if crop.shape[1] / crop.shape[0] > cfg.max_aspect:
            continue
        crops.append(np.ascontiguousarray(crop))
    return crops


def _width_consistent(pairs: list[tuple[np.ndarray, str]],
                      cfg: SegmentConfig) -> bool:
    """Second opinion on an alignment that already passed the count check.

    Matching counts do not prove matching *order*. A stray ink blot can occupy
    one slot and shove every later crop one token to the right while the totals
    still agree — observed in the wild on IAM, e.g. a 18 px speck landing on the
    token "well" and pushing the real "well" onto "chosen".

    One hand wrote the whole line at one scale, so width-per-character should be
    roughly constant across it. Any token wildly off the line's median means the
    pairing has slipped. Punctuation and single letters are excluded: a comma is
    one character of almost no width and would fire on every line.
    """
    wpc = [c.shape[1] / len(t) for c, t in pairs if clean_token(t)]
    if len(wpc) < 2:
        return True
    median = float(np.median(wpc))
    if median <= 0:
        return False
    return all(cfg.wpc_lo <= w / median <= cfg.wpc_hi for w in wpc)


def align_line_verbose(gray: np.ndarray, text: str,
                       cfg: SegmentConfig | None = None
                       ) -> tuple[list[tuple[np.ndarray, str]] | None, str]:
    """:func:`align_line` plus the reason, so the builder can tabulate rejects."""
    cfg = cfg or SegmentConfig()
    tokens = text.split()
    if not tokens:
        return None, "empty_text"
    crops = segment_line(gray, cfg)
    if len(crops) != len(tokens):
        return None, "count_mismatch"
    pairs = list(zip(crops, tokens))
    if not _width_consistent(pairs, cfg):
        return None, "width_inconsistent"
    return pairs, "ok"


def align_line(gray: np.ndarray, text: str, cfg: SegmentConfig | None = None
               ) -> list[tuple[np.ndarray, str]] | None:
    """Pair crops with transcript tokens, or ``None`` if the pairing is suspect.

    Punctuation is counted here even though it is filtered out later: IAM
    tokenises ``"Exchange ."`` as two tokens and the pen wrote two marks, so
    including it is what keeps the positional pairing meaningful.
    """
    return align_line_verbose(gray, text, cfg)[0]


def clean_token(token: str) -> str | None:
    """Lowercased alphabetic token, or ``None`` if it is not usable."""
    t = token.strip().lower()
    if not t.isascii() or not t.isalpha():
        return None
    if not (MIN_TOKEN_LEN <= len(t) <= MAX_TOKEN_LEN):
        return None
    return t


# ------------------------------------------------------------------ output --


class ManifestWriter:
    """Accumulates rows and saves height-normalised PNGs under ``out_dir``."""

    COLUMNS = ("split", "path", "text", "font", "tier", "width", "height")

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.rows: list[tuple] = []
        # Pre-seeded so an empty split reports 0 rather than vanishing from the
        # manifest counts — a missing key reads as "not built", not "empty".
        self._n: dict[str, int] = {s: 0 for s in SPLITS}

    def add(self, split: str, img: np.ndarray, text: str) -> None:
        idx = self._n[split]
        self._n[split] = idx + 1
        img = fit_height(img, TARGET_HEIGHT)
        rel = f"{split}/{idx:07d}.png"
        path = self.out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
        self.rows.append(
            (split, rel, text, FONT_TAG, TIER_TAG, img.shape[1], img.shape[0])
        )

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._n)

    def flush(self, meta: dict) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with (self.out_dir / "labels.tsv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(self.COLUMNS)
            w.writerows(self.rows)
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta


# ----------------------------------------------------------- HuggingFace --


def _hf_lines(split: str, max_lines: int | None, streaming: bool
              ) -> Iterator[tuple[np.ndarray, str]]:
    """Yield ``(grayscale line, transcript)`` from the HF dataset, in row order.

    Imported lazily so the segmentation helpers stay usable — and testable —
    without ``datasets`` installed or a network connection.
    """
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split=HF_SPLIT_MAP[split], streaming=streaming)
    for i, row in enumerate(ds):
        if max_lines is not None and i >= max_lines:
            break
        img = row["image"]
        yield np.asarray(img.convert("L"), dtype=np.uint8), row["text"]


def build_from_hf(
    out_dir: Path,
    max_lines: int | None = None,
    seed: int = 1337,
    cfg: SegmentConfig | None = None,
    streaming: bool | None = None,
) -> dict:
    """Segment ``Teklia/IAM-line`` into words and write the manifest.

    ``streaming`` defaults to True whenever ``max_lines`` is set: a smoke test
    should not pull 265 MB of parquet to look at 200 lines. Streaming yields the
    dataset's own row order, so a capped run reads a deterministic *prefix*
    rather than a random sample — fine for verification, not for a real build.

    ``seed`` only lands in ``meta.json`` here. Nothing on this path is random:
    the splits come from the dataset and the segmentation is deterministic, so
    the same commit always produces the same files.
    """
    cfg = cfg or SegmentConfig()
    if streaming is None:
        streaming = max_lines is not None

    writer = ManifestWriter(out_dir)
    stats = {s: BuildStats() for s in SPLITS}

    for split in SPLITS:
        st = stats[split]
        for gray, text in _hf_lines(split, max_lines, streaming):
            st.lines_seen += 1
            pairs, reason = align_line_verbose(gray, text, cfg)
            if pairs is None:
                st.reject(reason)
                continue
            st.lines_aligned += 1
            st.tokens_total += len(pairs)
            for crop, token in pairs:
                clean = clean_token(token)
                if clean is None:
                    continue
                st.tokens_kept += 1
                writer.add(split, crop, clean)
        print(f"  {split:<6} lines={st.lines_seen:<6} aligned={st.lines_aligned:<6} "
              f"discard={st.discard_rate:.1%}  words={writer.counts.get(split, 0)}")

    overall = BuildStats()
    for s in SPLITS:
        overall.merge(stats[s])

    meta = {
        "source": "huggingface:" + HF_DATASET,
        "source_kind": "line-level, segmented by vertical projection",
        "seed": seed,
        "target_height": TARGET_HEIGHT,
        "max_lines_per_split": max_lines,
        "streaming": streaming,
        "split_strategy": "dataset_provided",
        "split_note": (
            "Splits come from the HF dataset's own train/validation/test "
            "(6482/976/2915 lines), i.e. a published IAM partition rather than "
            "a reshuffle, and every word inherits its parent line's split so no "
            "line straddles splits. The dataset exposes no writer ids, so "
            "writer-independence could NOT be verified programmatically here — "
            "treat it as inherited from upstream, not as proven."
        ),
        "segmentation": asdict(cfg),
        "token_filter": {
            "alphabetic_ascii_only": True,
            "lowercased": True,
            "min_len": MIN_TOKEN_LEN,
            "max_len": MAX_TOKEN_LEN,
        },
        "counts": writer.counts,
        "overall": overall.as_dict(),
        "per_split": {s: stats[s].as_dict() for s in SPLITS},
    }
    return writer.flush(meta)


# ---------------------------------------------------------- local words.tgz --


# IAM's official writer-independent partition ships as these files (the "large
# writer independent text line recognition task" archive).
_OFFICIAL_SPLIT_FILES = {
    "train": ("trainset.txt",),
    "val": ("validationset1.txt", "validationset2.txt"),
    "test": ("testset.txt",),
}


@dataclass
class WordEntry:
    word_id: str
    path: Path
    text: str

    @property
    def form_id(self) -> str:
        # a01-000u-00-03 -> a01-000u
        return "-".join(self.word_id.split("-")[:2])

    @property
    def line_id(self) -> str:
        # a01-000u-00-03 -> a01-000u-00
        return "-".join(self.word_id.split("-")[:3])


def _find_file(root: Path, name: str) -> Path | None:
    """Locate ``name`` near ``root``; IAM's archives unpack in several layouts.

    ``--local-words-dir`` may point at the folder holding ``words/`` or at
    ``words/`` itself, and the ascii metadata may sit loose or under ``ascii/``.
    Explicit candidates are tried first; the recursive scan is a last resort
    because ``words/`` contains ~115k files.
    """
    parent = root.parent
    for base in (root, root / "ascii", parent, parent / "ascii"):
        if (base / name).is_file():
            return base / name
    for base in (root, parent):
        hit = next(iter(sorted(base.glob(f"*/{name}"))), None)
        if hit is not None:
            return hit
    return next(iter(sorted(root.rglob(name))), None)


def _word_image_path(root: Path, word_id: str) -> Path | None:
    """``a01-000u-00-03`` -> ``<root>/[words/]a01/a01-000u/<id>.png``."""
    parts = word_id.split("-")
    rel = Path(parts[0]) / "-".join(parts[:2]) / f"{word_id}.png"
    for base in (root / "words", root):
        p = base / rel
        if p.is_file():
            return p
    return None


def parse_words_txt(words_txt: Path, root: Path) -> tuple[list[WordEntry], int]:
    """Parse IAM's ``words.txt``; returns entries and the count dropped as err.

    Line format (from IAM's own header)::

        a01-000u-00-00 ok 154 1 408 768 27 51 AT A
        id  seg_result graylevel n_components x y w h tag transcription
    """
    entries: list[WordEntry] = []
    n_err = 0
    for raw in words_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        # Fewer than 10 fields means the transcription is missing; falling back
        # to the last field would label the image with its grammatical tag.
        if len(parts) < 10:
            continue
        word_id, seg = parts[0], parts[1]
        if seg == "err":
            # IAM itself flags these boxes as bad segmentations — exactly the
            # label noise the projection path works so hard to avoid.
            n_err += 1
            continue
        path = _word_image_path(root, word_id)
        if path is None:
            continue
        entries.append(WordEntry(word_id, path, " ".join(parts[9:])))
    return entries, n_err


def _official_split_map(root: Path) -> dict[str, str] | None:
    """line_id -> split, from IAM's writer-independent partition files."""
    mapping: dict[str, str] = {}
    for split, names in _OFFICIAL_SPLIT_FILES.items():
        found = False
        for name in names:
            f = _find_file(root, name)
            if f is None:
                continue
            found = True
            for line in f.read_text(encoding="utf-8").split():
                mapping[line.strip()] = split
        if not found:
            return None
    return mapping or None


def _writer_map(root: Path) -> dict[str, str] | None:
    """form_id -> writer_id, from ``forms.txt`` if it was downloaded."""
    f = _find_file(root, "forms.txt")
    if f is None:
        return None
    out: dict[str, str] = {}
    for raw in f.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out or None


def _grouped_random_split(groups: list[str], seed: int) -> dict[str, str]:
    """Assign whole groups (writers or forms) to splits, never single samples.

    Splitting individual words would put the same hand — often the same word —
    on both sides of the train/test line and inflate every number downstream.
    """
    rng = np.random.default_rng(seed)
    uniq = sorted(set(groups))
    perm = rng.permutation(len(uniq))
    n_train = int(round(len(uniq) * RANDOM_SPLIT_FRACS[0]))
    n_val = int(round(len(uniq) * RANDOM_SPLIT_FRACS[1]))
    out: dict[str, str] = {}
    for rank, i in enumerate(perm):
        if rank < n_train:
            split = "train"
        elif rank < n_train + n_val:
            split = "val"
        else:
            split = "test"
        out[uniq[int(i)]] = split
    return out


def build_from_local_words(
    words_dir: Path,
    out_dir: Path,
    max_lines: int | None = None,
    seed: int = 1337,
) -> dict:
    """Build from the official ``words.tgz`` + ``words.txt``.

    No projection guessing here — IAM's boxes are ground truth, so the discard
    rate is 0 by construction and the only losses are the ``err`` segmentations
    IAM already flags plus the alphabetic filter.
    """
    words_dir = Path(words_dir)
    words_txt = _find_file(words_dir, "words.txt")
    if words_txt is None:
        raise FileNotFoundError(
            f"no words.txt under {words_dir} — expected the ascii/ archive "
            f"alongside the extracted words/ tree"
        )

    entries, n_err = parse_words_txt(words_txt, words_dir)
    if not entries:
        raise RuntimeError(f"{words_txt} parsed, but no word images resolved "
                           f"under {words_dir}")

    official = _official_split_map(words_dir)
    if official is not None:
        strategy = "iam_official_writer_independent"
        note = ("IAM's published writer-independent partition files "
                "(trainset/validationset1+2/testset).")

        def split_of(e: WordEntry) -> str | None:
            return official.get(e.line_id)
    else:
        writers = _writer_map(words_dir)
        group_of = (lambda e: writers.get(e.form_id, e.form_id)) if writers \
            else (lambda e: e.form_id)
        strategy = ("seeded_random_by_writer" if writers
                    else "seeded_random_by_form")
        note = ("No official split files found. "
                + ("Grouped by writer id from forms.txt, so writer-independent."
                   if writers else
                   "forms.txt was absent, so grouping falls back to form id: a "
                   "writer who filled in several forms CAN appear on both sides, "
                   "which inflates results relative to a true writer-independent "
                   "split. Say so in the report if this path is used."))
        assign = _grouped_random_split([group_of(e) for e in entries], seed)

        def split_of(e: WordEntry) -> str | None:
            return assign.get(group_of(e))

    # `max_lines` means lines here too: cap by distinct line ids per split so a
    # smoke test costs the same either way.
    seen_lines: dict[str, set[str]] = {s: set() for s in SPLITS}

    writer_out = ManifestWriter(out_dir)
    stats = {s: BuildStats() for s in SPLITS}
    n_unreadable = 0
    n_filtered = 0

    for e in entries:
        split = split_of(e)
        if split is None:
            continue
        lines = seen_lines[split]
        if e.line_id not in lines:
            if max_lines is not None and len(lines) >= max_lines:
                continue
            lines.add(e.line_id)
            stats[split].lines_seen += 1
            stats[split].lines_aligned += 1
        stats[split].tokens_total += 1

        clean = clean_token(e.text)
        if clean is None:
            n_filtered += 1
            continue
        img = cv2.imread(str(e.path), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0 or min(img.shape) < 2:
            n_unreadable += 1
            continue
        stats[split].tokens_kept += 1
        writer_out.add(split, img, clean)

    for s in SPLITS:
        print(f"  {s:<6} lines={stats[s].lines_seen:<6} "
              f"words={writer_out.counts.get(s, 0)}")

    overall = BuildStats()
    for s in SPLITS:
        overall.merge(stats[s])

    meta = {
        "source": f"local:{words_txt}",
        "source_kind": "official IAM word boxes (ground-truth segmentation)",
        "seed": seed,
        "target_height": TARGET_HEIGHT,
        "max_lines_per_split": max_lines,
        "split_strategy": strategy,
        "split_note": note,
        "segmentation": None,
        "token_filter": {
            "alphabetic_ascii_only": True,
            "lowercased": True,
            "min_len": MIN_TOKEN_LEN,
            "max_len": MAX_TOKEN_LEN,
        },
        "counts": writer_out.counts,
        "overall": overall.as_dict(),
        "per_split": {s: stats[s].as_dict() for s in SPLITS},
        "dropped": {
            "iam_err_segmentations": n_err,
            "non_alphabetic_or_length": n_filtered,
            "unreadable_images": n_unreadable,
        },
    }
    return writer_out.flush(meta)


# -------------------------------------------------------------------- cli --


def main() -> None:
    from lasa.common.paths import IAM, ensure_dirs
    from lasa.common.seed import set_seed

    ap = argparse.ArgumentParser(
        description="Build word-level IAM handwriting data (labels.tsv + PNGs)."
    )
    ap.add_argument("--max-lines", type=int, default=None,
                    help="cap lines read PER SPLIT; default reads everything")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--local-words-dir", type=Path, default=None,
                    help="directory holding the official words.tgz extraction "
                         "and words.txt; preferred over the HF download because "
                         "its segmentation is ground truth")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output directory (default: {IAM})")
    ap.add_argument("--no-stream", action="store_true",
                    help="force the full ~265 MB parquet download instead of "
                         "streaming (HF path only; this is already the default "
                         "when --max-lines is not given)")
    args = ap.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    out = Path(args.out) if args.out else IAM

    if args.local_words_dir is not None:
        print(f"source: local IAM words at {args.local_words_dir}")
        meta = build_from_local_words(args.local_words_dir, out,
                                      max_lines=args.max_lines, seed=args.seed)
    else:
        print(f"source: {HF_DATASET} (public, ungated)")
        meta = build_from_hf(out, max_lines=args.max_lines, seed=args.seed,
                             streaming=False if args.no_stream else None)

    print(json.dumps(meta["counts"], indent=2))
    print(f"split strategy : {meta['split_strategy']}")
    print(f"discard rate   : {meta['overall']['discard_rate']:.1%} of lines")
    print(f"words written  : {sum(meta['counts'].values())} -> {out}")


if __name__ == "__main__":
    main()
