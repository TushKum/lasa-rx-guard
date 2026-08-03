# LASA-RxGuard

Handwritten prescription OCR + look-alike/sound-alike (LASA) drug-confusion risk
flagging.

Drug names that look or sound alike are a well-documented source of dispensing
errors — *DOPamine* / *DOBUTamine*, *hydrOXYzine* / *hydrALAZINE*, *Celebrex* /
*Celexa*. Handwriting makes it worse. This project reads a handwritten drug name
from a prescription and, before dispensing, flags when the reading is
dangerously ambiguous.

Two stages:

**Stage 1 — OCR.** A CRNN (CNN + BiLSTM + CTC) trained from scratch, compared
against a fine-tuned `microsoft/trocr-base-handwritten`. Reported side by side
on CER, WER and exact-match.

**Stage 2 — LASA risk.** The OCR string is matched against a 2,000-name drug
database using edit distance, phonetic encoding (Soundex / Metaphone / Double
Metaphone) and character n-gram TF-IDF cosine similarity. Returns the top-3
candidates plus a RED / YELLOW / GREEN flag, validated against published
ISMP/FDA confused-drug-name pairs.

> **This is a hackathon prototype, not a validated medical device.** It has not
> been clinically evaluated and must not be used for real dispensing decisions.
> The intended role is advisory, with a pharmacist making the final call.

---

## Quickstart

```bash
git clone <this-repo> && cd lasa-rx-guard
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e . --no-deps
```

> On Apple Silicon, make sure the venv uses a **native arm64** interpreter. An
> Anaconda `python3` is often x86_64 under Rosetta, which cannot install
> `onnxruntime` and disables MPS. Check with
> `.venv/bin/python -c "import sysconfig; print(sysconfig.get_platform())"` —
> you want `macosx-11.0-arm64`, not `macosx-10.16-x86_64`.

Build the data, then train:

```bash
python -m lasa.data.drug_names --size 2000   # RxNorm + NLEM + LASA ground truth
python -m lasa.data.fonts                    # 25 OFL handwriting fonts
python -m lasa.data.synth --preview          # eyeball the samples first
python -m lasa.data.synth                    # full dataset
python -m lasa.train.train_crnn --config configs/crnn.yaml
python -m lasa.train.train_trocr --config configs/trocr.yaml
python -m lasa.evaluation.evaluate
python -m lasa.risk.evaluate_lasa --markdown
```

Or run the whole thing: `scripts/run_all.sh` (add `--smoke` for a fast pass).

Demo app:

```bash
streamlit run app/streamlit_app.py
```

For the free-Colab-T4 path, use `notebooks/colab_train.ipynb`.

---

## Data

| Asset | Count | Source |
|---|---|---|
| Drug names | 2,000 | RxNorm `IN` + `BN` via the RxNav REST API, filtered; NLEM India 2022; LASA ground-truth names |
| LASA pairs | 164 unique | ISMP "List of Confused Drug Names" + FDA/ISMP tall-man lettering list |
| Handwriting fonts | 25 | Google Fonts, all OFL-1.1 / Apache-2.0 |
| Real handwriting | IAM | HuggingFace `Teklia/IAM-line` (public, ungated) |

**Drug-name filtering.** Raw RxNorm `IN` contains ~14.6k concepts, most of which
are IUPAC chemical strings and allergenic extracts rather than anything a
prescriber writes. `lasa/data/drug_names.py` keeps only drug-shaped tokens and
then ranks by LASA-ground-truth membership → NLEM → recognised INN stem
(`-olol`, `-pril`, `-statin`, …) → everything else. All 272 distinct names that
appear in the LASA pair list are guaranteed present, because a Stage-2 score
against a database that is missing the ground truth would be meaningless.

**LASA pair provenance.** `lasa/data/lasa_pairs.py` is a hand-curated subset of
publicly published ISMP/FDA pairs, not a verbatim copy of ISMP's full table. To
use the official list, download it and run
`python scripts/ingest_ismp.py --csv <file>`; everything downstream reads the
merged CSV, so the evaluation improves automatically. **Any results table must
state which source was used.**

**IAM word segmentation.** `Teklia/IAM-line` is line-level, so word crops are
inferred from whitespace gaps. A line only contributes samples when the number
of detected crops exactly matches the number of transcript tokens — otherwise
the alignment is a guess and would inject label noise. The discard rate is
recorded in `data/iam/meta.json`. If you have the registration-gated official
`words.tgz`, pass `--local-words-dir` to use its ground-truth segmentation
instead.

### Synthetic handwriting

IAM has real handwriting but essentially no medicine names; the drug corpus has
the right vocabulary but no images. `lasa/data/synth.py` bridges the gap.

Realism comes from **per-glyph** jitter, not from the font. Setting a whole
string in a handwriting font still yields uniform letterforms on a straight
baseline, which a CNN learns to exploit. Each glyph gets independent baseline
wobble, rotation, scale and spacing, plus a slow sinusoidal baseline drift and
explicit cursive connecting strokes across the word. Degradation
(`lasa/data/augment.py`) then models the imaging path: paper stock and fibre
texture, uneven illumination, ruled lines, ink pressure and skipping, shear,
elastic distortion, motion blur, sensor noise, JPEG artefacts.

Three tiers — `clean`, `medium` (default), `hard` — scale both the handwriting
messiness and the imaging degradation together.

A **legibility guard** rejects draws whose ink-to-paper contrast falls below a
threshold and redraws them. Without it, aggressive augmentation silently
produces samples no human could read, which is mislabelled data rather than
difficulty: it puts a floor on achievable CER and teaches the model to guess
from word length.

### Splits are deliberately adversarial

| Split | Words | Fonts | Measures |
|---|---|---|---|
| `train` / `val` | train | train | — |
| `test_seen` | train | train | operational accuracy on a known formulary |
| `test_unseen_font` | train | **held out** | robustness to an unfamiliar hand |
| `test_unseen_word` | **held out** | train | genuine character-level generalisation |
| `test_hard` | **held out** | **held out** | both at once |

`test_seen` is the number that matters operationally, since a real pharmacy
searches a fixed formulary. `test_unseen_word` is the honest measure of whether
the model learned to read rather than to memorise. Reporting only one of them
would be misleading, so `results.md` reports all four.

---

## Stage 2: how the risk flag is decided

Three similarity signals are fused (weights in `configs/risk.yaml`):

1. **Edit distance** — catches visual/typographic confusion.
2. **Phonetic** — Soundex, Metaphone, Double Metaphone. Catches verbal-order
   confusion, which is how many real LASA errors happen.
3. **Character n-gram TF-IDF cosine** — robust to OCR noise, and handles
   out-of-vocabulary strings (OCR output frequently is not a real word).

| Flag | Meaning |
|---|---|
| 🔴 **RED** | The top match has a known ISMP/FDA confusable scoring close behind it, or the top-2 are a known pair within the margin, or a high-alert drug is involved in the ambiguity. |
| 🟡 **YELLOW** | Top-2 are close but not a documented pair, or overall confidence is low. |
| 🟢 **GREEN** | One clear match, no known confusable nearby. |

`reasons` are plain clinical language, not bare scores — a pharmacist has to act
on the output.

Stage 2 is evaluated on top-1/3/5 retrieval under a seeded OCR-like noise model,
precision/recall/F1 for pair flagging, a **negative control** (unrelated names
must not be flagged — without it, a matcher that flags everything would score
perfect recall), and an ablation of each signal against the fusion.

---

## Reproducibility

- Every entrypoint calls `lasa.common.seed.set_seed` (default `1337`); DataLoader
  workers are seeded via `worker_init_fn`.
- Augmentation is **baked into the stored image files**, not applied on the fly,
  so CRNN and TrOCR train on byte-identical inputs. Any difference between them
  is the model, not the data.
- Font downloads are SHA-256 pinned in `data/fonts/manifest.json`. Default fonts
  are all Google Fonts and therefore identical on Colab; the optional
  `--include-system` macOS fonts are marked non-portable and would break that
  guarantee.
- The resolved config is written next to every checkpoint.

## Layout

```
configs/            YAML for data, CRNN, TrOCR, risk thresholds
src/lasa/
  common/           seeding, paths, config loading
  data/             drug corpus, LASA pairs, fonts, synthesis, augmentation,
                    IAM loader, torch datasets + CTC/TrOCR collates
  models/           CRNN (+ greedy / beam / lexicon-constrained decoding), TrOCR
  train/            train_crnn.py, train_trocr.py
  evaluation/       CER/WER/exact metrics, side-by-side comparison harness
  risk/             phonetics, n-gram index, LASA matcher, Stage-2 evaluation
  infer/            end-to-end image → OCR → risk pipeline
app/                Streamlit demo
notebooks/          Colab T4 training notebook
docs/               results.md, deployment.md
```

## Licence / attribution

Fonts are OFL-1.1 / Apache-2.0 (see `data/fonts/manifest.json`). RxNorm is a
product of the U.S. National Library of Medicine. LASA pairs are derived from
publicly published ISMP and FDA safety references; ISMP's full list is
redistributed under their own terms and is not bundled here.
