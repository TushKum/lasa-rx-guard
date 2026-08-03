"""Regression tests for the parts that broke silently during development.

Each test here corresponds to a real bug that shipped and was caught by
measurement rather than by an exception — those are exactly the ones worth
pinning, because none of them raised.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lasa.data.augment import get_config, ink_paper_contrast
from lasa.data.datasets import (CRNN_HEIGHT, Charset, WordImageDataset,
                                make_ctc_collate, min_feasible_width,
                                read_manifest, resize_for_ctc)
from lasa.data.lasa_pairs import deduped_pairs, names_in_pairs
from lasa.evaluation.metrics import summarize
from lasa.models.crnn import CRNN, greedy_decode
from lasa.risk.matcher import GREEN, RED, YELLOW, LasaMatcher

CHARS = "abcdefghijklmnopqrstuvwxyz"


# ------------------------------------------------------------------ charset --


def test_charset_roundtrip_and_blank_reserved():
    cs = Charset(CHARS)
    assert cs.BLANK == 0
    assert 0 not in cs.encode("dopamine")          # blank index never emitted
    assert cs.decode(cs.encode("dopamine")) == "dopamine"
    assert len(cs) == len(CHARS) + 1


def test_charset_filters_unknown_symbols():
    cs = Charset(CHARS)
    assert cs.filter("Dopa-mine 5") == "opamine"   # uppercase/punct/digits dropped


# ------------------------------------------------------------------ collate --


def test_ctc_collate_shapes_and_white_padding():
    cs = Charset(CHARS)
    coll = make_ctc_collate(cs)
    batch = [
        {"image": np.full((64, 120), 200, np.uint8), "text": "abc", "font": "f"},
        {"image": np.full((64, 40), 200, np.uint8), "text": "de", "font": "f"},
    ]
    out = coll(batch)
    assert out["images"].shape[:2] == (2, 1)
    assert out["images"].shape[2] == CRNN_HEIGHT
    assert out["target_lengths"].tolist() == [3, 2]
    # Padding must be paper-white (+1 after normalisation), never 0/ink-black:
    # zero-padding reads to the CNN as a solid ink block.
    assert out["images"].max().item() == pytest.approx(1.0, abs=1e-5)


def test_input_lengths_use_true_width_not_padded_width():
    cs = Charset(CHARS)
    coll = make_ctc_collate(cs)
    out = coll([
        {"image": np.full((64, 400), 200, np.uint8), "text": "abcdefgh", "font": ""},
        {"image": np.full((64, 60), 200, np.uint8), "text": "ab", "font": ""},
    ])
    assert out["input_lengths"][0] > out["input_lengths"][1], \
        "a narrow sample must not inherit the padded batch width"


@pytest.mark.parametrize("text,width", [("hydrochlorothiazide", 59), ("lasix", 59),
                                        ("dihydroxypropyltrimonium", 80)])
def test_no_sample_is_ctc_infeasible(text, width):
    """CTC needs >= 2L+1 frames. Infeasible samples do not error — `zero_infinity`
    zeroes them — so they silently vanish from training. This once removed 47.8%
    of the corpus and collapsed the model to all-blanks."""
    cs = Charset(CHARS)
    coll = make_ctc_collate(cs)
    out = coll([{"image": np.full((64, width), 200, np.uint8), "text": text, "font": ""}])
    assert out["input_lengths"][0] >= 2 * out["target_lengths"][0] + 1


def test_resize_stretches_width_enough_for_long_names():
    img = np.full((64, 146), 200, np.uint8)        # corpus median width
    out = resize_for_ctc(img, CRNN_HEIGHT)
    assert out.shape[0] == CRNN_HEIGHT
    assert out.shape[1] >= 146, "aspect-preserving resize starves CTC of frames"


def test_min_feasible_width_matches_ctc_requirement():
    assert min_feasible_width("abc", downsample=4) == 4 * 7


# --------------------------------------------------------------------- CRNN --


def test_crnn_width_downsample_is_exactly_four():
    """The CNN's width stride is an implicit contract with
    `make_ctc_collate(downsample=4)`. If either side drifts, CTC alignment
    breaks quietly rather than raising."""
    m = CRNN(n_classes=28, img_h=CRNN_HEIGHT).eval()
    for w in (32, 64, 127, 256):
        with torch.no_grad():
            t = m(torch.zeros(1, 1, CRNN_HEIGHT, w)).shape[0]
        assert t == w // 4, f"width {w}: got T={t}, expected {w // 4}"


def test_greedy_decode_collapses_repeats_and_strips_blanks():
    cs = Charset(CHARS)
    a, b = cs.encode("a")[0], cs.encode("b")[0]
    # frames: a a blank a b b  ->  "aab"
    seq = [a, a, cs.BLANK, a, b, b]
    lp = torch.full((len(seq), 1, len(cs)), -20.0)
    for t, idx in enumerate(seq):
        lp[t, 0, idx] = 0.0
    assert greedy_decode(lp, cs)[0] == "aab"


# ------------------------------------------------------------------ metrics --


def test_cer_wer_exact_on_hand_computed_example():
    m = summarize(["dopamine"], ["dobutamine"])
    # dopamine -> dobutamine is 3 edits over 8 reference chars
    assert m.total_char_errors == 3
    assert m.cer == pytest.approx(3 / 8)
    assert m.exact_match == 0.0
    assert m.wer == pytest.approx(1.0)


def test_cer_is_corpus_micro_average_not_mean_of_rates():
    """Micro-average = total errors / total chars. The mean of per-sample rates
    would be (1/2 + 1/10)/2 = 0.30 here; the correct value is 2/12."""
    m = summarize(["ab", "abcdefghij"], ["xb", "xbcdefghij"])
    assert m.cer == pytest.approx(2 / 12)


def test_metrics_handle_empty_hypothesis():
    m = summarize(["lasix"], [""])
    assert m.cer == pytest.approx(1.0)
    assert m.exact_match == 0.0


# ---------------------------------------------------------------- augment --


def test_ink_paper_contrast_separates_ink_from_paper():
    img = np.full((32, 64), 240, np.uint8)
    img[10:20, 10:40] = 20
    mask = np.zeros_like(img)
    mask[10:20, 10:40] = 1
    assert ink_paper_contrast(img, mask) > 200
    assert ink_paper_contrast(np.full((32, 64), 240, np.uint8), mask) == pytest.approx(0, abs=1)


def test_augment_tiers_increase_in_severity():
    clean, hard = get_config("clean"), get_config("hard")
    assert hard.shear[1] > clean.shear[1]
    assert hard.elastic_alpha[1] > clean.elastic_alpha[1]
    assert hard.noise_sigma[1] > clean.noise_sigma[1]


# ------------------------------------------------------------ LASA ground truth --


def test_every_ground_truth_name_is_in_the_corpus():
    """A Stage-2 score against a database missing the ground truth is meaningless."""
    m = LasaMatcher.from_project()
    missing = names_in_pairs() - set(m.by_name)
    assert not missing, f"{len(missing)} ground-truth names absent from corpus: {sorted(missing)[:5]}"


def test_pairs_are_deduplicated_and_symmetric():
    pairs = deduped_pairs()
    keys = {frozenset((p.a, p.b)) for p in pairs}
    assert len(keys) == len(pairs)
    assert all(p.a != p.b for p in pairs)


# ----------------------------------------------------------------- matcher --


@pytest.mark.parametrize("query,partner", [
    ("dobutamin", "dopamine"),
    ("celebrex", "celexa"),
    ("hydroxyzine", "hydralazine"),
])
def test_known_pairs_are_flagged_and_partner_surfaced(query, partner):
    m = LasaMatcher.from_project()
    a = m.match(query, k=3)
    assert a.risk in (RED, YELLOW), f"{query} -> {a.risk}"
    assert partner in {c.name for c in a.candidates}, \
        f"{partner} missing from candidates for {query}"


def test_injected_partner_is_never_truncated_away():
    """Documented partners score low by construction — that is why similarity
    missed them — so merging and re-sorting into a k-sized list silently drops
    them and makes the injection a no-op."""
    m = LasaMatcher.from_project()
    a = m.match("losec", k=3)
    assert "lasix" in {c.name for c in a.candidates}


def test_unambiguous_drug_is_green():
    m = LasaMatcher.from_project()
    assert m.match("paracetamol", k=3).risk == GREEN


def test_garbage_input_is_never_green():
    """A weak best match means 'I do not know', which must not read as safe."""
    m = LasaMatcher.from_project()
    a = m.match("xyzzyqqq", k=3)
    assert a.risk != GREEN
    assert any("no confident" in r.lower() for r in a.reasons)


def test_high_alert_escalates_to_red():
    m = LasaMatcher.from_project()
    a = m.match("vincristin", k=3)
    assert a.risk == RED
    assert any("high-alert" in r.lower() for r in a.reasons)


def test_empty_query_does_not_crash():
    m = LasaMatcher.from_project()
    assert m.match("", k=3).risk in (RED, YELLOW, GREEN)
