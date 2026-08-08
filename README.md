# LASA-RxGuard

**Reads messy doctor handwriting, then warns the pharmacist when two medicines have dangerously similar names.**

---

## The simple version

Some medicines have names that look almost the same.

> **DOPamine** and **DOBUTamine**
> **Celebrex** and **Celexa**
> **Losec** and **Lasix**

They are *completely different drugs*. Giving someone the wrong one can seriously hurt them.

Now imagine that name is written by hand, in a hurry, on a small piece of paper. The pharmacist has a few seconds to decide which one it says.

**That is the problem this project tries to help with.**

### How it works, in two steps

**Step 1 — Read the handwriting.**
A computer program looks at the photo of the written word and guesses the letters. It gets about 97 out of every 100 letters right.

**Step 2 — Check if it is dangerous.**
The program compares its guess against a list of 2,000 real medicine names, three different ways:

- Is it **spelled** almost the same?
- Does it **sound** almost the same when you say it out loud?
- Do the **letter patterns** look similar?

Then it shows a traffic light:

| | What it means |
|---|---|
| 🔴 **RED** | Stop. Two dangerous look-alike medicines both match. Check before giving it out. |
| 🟡 **YELLOW** | Careful. There is another medicine it could be. |
| 🟢 **GREEN** | Looks clear. Only one medicine matches. |

And it always explains *why* in normal words — not just a score. A warning nobody understands is a warning everybody ignores.

### One important thing

**This is a student project, not a real medical tool.** It has never been tested in a real pharmacy. A real pharmacist must always make the final decision. It is meant to *help* a human, never to replace one.

---

## Try it yourself

```bash
git clone https://github.com/TushKum/lasa-rx-guard && cd lasa-rx-guard
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e . --no-deps
.venv/bin/python -m streamlit run app/streamlit_app.py
```

Open the **"Type a drug name"** tab and try typing these, one at a time:

| Type this | You should see |
|---|---|
| `dobutamin` | 🔴 RED — because DOPamine is a known dangerous twin |
| `losec` | 🟡 YELLOW — because Lasix sounds almost identical |
| `paracetamol` | 🟢 GREEN — nothing else is close |
| `xyzzyqqq` | 🟡 YELLOW — "I don't know what this is", never a green light |

That last one matters. If the program is confused, it says so. It never pretends.

> On Apple Silicon, make sure your venv uses a **native arm64** Python. Anaconda's is often x86_64 under Rosetta, which can't install `onnxruntime` and disables MPS.
> Check: `.venv/bin/python -c "import sysconfig; print(sysconfig.get_platform())"` → you want `macosx-11.0-arm64`.

---

## What it scored

**Reading handwriting** — character error rate, lower is better:

| Test | Result | What it tells you |
|---|---|---|
| Familiar words + handwriting | 2.91% | Everyday accuracy |
| **Words it had never seen** | **2.83%** | It genuinely learned to read, not memorise |
| **Handwriting it had never seen** | **19.45%** | New handwriting styles are the weak point |
| Picking from the known medicine list | **0.48%** | 98.75% of whole names exactly right |

**The main finding:** new *words* cost almost nothing. New *handwriting* costs nearly 7× more error. So to improve this, collect more **handwriting styles**, not more medicine names.

**Spotting dangerous pairs:**

| | |
|---|---|
| Catches known dangerous pairs | **97.9%** |
| Correct when it warns | **91.2%** |
| False alarms | **15.5%** |

The false-alarm number was originally **36%** — one prescription in three. That is how safety systems fail in real life: when most alerts are noise, staff learn to ignore all of them, including the real ones. Retuning cut it to 15.5% **without missing any more dangerous pairs**.

---

## Honest limitations

- **All the handwriting is computer-generated.** Real doctors' handwriting is messier, and the results above would likely be worse on it. This is the biggest weakness.
- **The second model didn't finish training.** A larger model (TrOCR) ran out of computing power. It is reported as a failed run, not as "our model won".
- **The dangerous-pairs list is a subset** — 164 pairs from published ISMP/FDA safety lists, not the complete official table.
- **Never clinically tested.** Not a medical device.

---

## Technical detail

<details>
<summary><strong>Architecture, data provenance, method</strong> — click to expand</summary>

### Stage 1 — OCR
A CRNN (CNN + BiLSTM + CTC), 8,379,740 parameters, trained from scratch, compared against a fine-tuned `microsoft/trocr-base-handwritten`. Reported on CER, WER and exact-match.

### Stage 2 — LASA risk
Fuses edit distance (0.45), phonetic encoding — Soundex / Metaphone / Double Metaphone (0.35), and character n-gram TF-IDF cosine (0.20). Weights in `configs/risk.yaml`, ablated in `docs/results.md`.

Documented partners of the top match are **force-injected** into the candidate list even when similarity misses them. Candidates carry a `via` field so retrieval and pair-surfacing are reported separately rather than merged.

### Data

| Asset | Count | Source |
|---|---|---|
| Drug names | 2,000 | RxNorm `IN`+`BN` via RxNav; NLEM India 2022 |
| LASA pairs | 164 | ISMP Confused Drug Names; FDA/ISMP tall-man list |
| Fonts | 25 | Google Fonts, OFL-1.1 / Apache-2.0, SHA-256 pinned |
| Synthetic images | 59,000 | Generated, seeded |
| Real handwriting | 22,506 words | `Teklia/IAM-line`, 69.1% of lines discarded on alignment |

Raw RxNorm `IN` contains ~14.6k concepts, mostly IUPAC strings and allergenic extracts. `lasa/data/drug_names.py` keeps drug-shaped tokens, then ranks by LASA-ground-truth membership → NLEM → INN stem → remainder. All 272 ground-truth names are guaranteed present; a Stage-2 score against a database missing the ground truth would be meaningless.

The LASA pair list is a curated subset of published ISMP/FDA material. To use the official list: `python scripts/ingest_ismp.py --csv <file>`. Results must state which source was used.

### Synthesis
Realism comes from **per-glyph** jitter, not the font: independent baseline wobble, rotation, scale and spacing, sinusoidal drift, and explicit cursive connectors. Degradation models the imaging path — paper texture, uneven illumination, ruled lines, ink pressure, shear, elastic distortion, motion blur, sensor noise, JPEG artefacts. Three tiers scale handwriting messiness and imaging degradation together.

A **legibility guard** rejects and redraws any sample whose ink-to-paper contrast falls below threshold. Without it, aggressive augmentation silently produces samples no human can read — mislabelled data that caps achievable CER and teaches the model to guess from word length.

### Splits

| Split | Words | Fonts | Measures |
|---|---|---|---|
| `train` / `val` | train | train | — |
| `test_seen` | train | train | operational accuracy |
| `test_unseen_font` | train | **held out** | robustness to an unfamiliar hand |
| `test_unseen_word` | **held out** | train | character-level generalisation |
| `test_hard` | **held out** | **held out** | both |

### Reproducibility
Every entrypoint calls `set_seed` (default 1337); DataLoader workers seeded via `worker_init_fn`. Augmentation is baked into stored files, so CRNN and TrOCR train on byte-identical inputs — any difference is the model, not the data. Fonts SHA-256 pinned. Resolved config written beside every checkpoint.

### Layout
```
configs/     YAML for data, CRNN, TrOCR, risk thresholds
src/lasa/
  common/    seeding, paths, config
  data/      corpus, LASA pairs, fonts, synthesis, augmentation, IAM, datasets
  models/    CRNN (greedy / beam / lexicon decoding), TrOCR
  train/     train_crnn.py, train_trocr.py
  evaluation/ metrics, comparison harness
  risk/      phonetics, n-gram index, matcher, Stage-2 evaluation
  infer/     image → OCR → risk pipeline
app/         Streamlit demo
notebooks/   Colab + Kaggle training notebooks
scripts/     run_all.sh, export_weights.py, capture_demo.py, build_deck.js
docs/        results.md, deployment.md
```

</details>

---

## Run the whole thing

```bash
./scripts/run_all.sh          # full pipeline (hours, needs a GPU for TrOCR)
./scripts/run_all.sh --smoke  # tiny sizes, minutes, proves the wiring
```

On a free GPU: `notebooks/colab_train.ipynb` or `notebooks/kaggle_train.ipynb`. Both work whether this repo is public or private.

Tests: `.venv/bin/python -m pytest tests/ -q` — 26 regression tests, each pinning a bug that raised no exception.

## Licence / attribution

Fonts are OFL-1.1 / Apache-2.0 (`data/fonts/manifest.json`). RxNorm is a product of the U.S. National Library of Medicine. LASA pairs derive from publicly published ISMP and FDA safety references; ISMP's full list is redistributed under their own terms and is not bundled here.
