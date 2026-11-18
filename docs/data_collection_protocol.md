# Real handwriting collection protocol

**Purpose.** Everything LASA-RxGuard reports is measured on synthetic
handwriting. The paper shows that held-out *writing style* costs 7.4× the error
rate while held-out vocabulary costs nothing measurable, so the domain gap to
real prescriber handwriting is the single largest source of uncertainty in the
work. This protocol closes it.

---

## 1. The central design decision

**We do not photograph patient prescriptions. We ask prescribers to write drug
names onto a structured form.**

This is not a convenience. It is the difference between a feasible student
study and an infeasible one:

| | Prompted forms (this protocol) | Photographing real prescriptions |
|---|---|---|
| Patient data created | **None** | Names, ages, diagnoses — all PHI |
| Ground-truth labels | **Exact** (we chose the prompt) | Guessed — and guessing which drug was written *is the task under test* |
| Ethics burden | Consent from prescribers only | Patient consent or waiver, redaction, DPDP obligations |
| Realistic to obtain | Yes | Rarely, for a student project |

The precedent is standard: the IAM corpus was built this way, with writers
copying prompts from the LOB corpus.

**The limitation this creates, and which must be stated in any write-up:**
copied handwriting is not composed handwriting. A prescriber copying a printed
word may write more carefully than when writing at speed during a consultation.
The instruction text on the form explicitly counters this ("write at your usual
speed; do not print neatly for us"), but the effect cannot be eliminated. Treat
results as an *optimistic* estimate of real-world performance.

---

## 2. Ethics and approvals

**Get Institutional Ethics Committee (IEC) approval before collecting
anything.** Do not start and seek approval afterwards; data collected without
prior approval generally cannot be used or published.

- Governing guidance: **ICMR National Ethical Guidelines for Biomedical and
  Health Research Involving Human Participants (2017)**.
- Your institution's IEC will have its own proforma. This protocol, the
  participant information sheet and the consent form are written to be attached
  to it.
- Because no patient data is collected and the intervention is "write some
  words on paper", most IECs will treat this as **minimal risk**. Say so in the
  application, and say why.

**Personal data.** Handwriting is potentially identifying. Under the **Digital
Personal Data Protection Act 2023**, treat the samples as personal data and
pseudonymise:

- Each prescriber receives an opaque **writer code** (`W01`, `W02`, …).
- The dataset contains **only** the code. No name, registration number,
  hospital, or specialty appears in any image or file.
- Signed consent forms (which do carry a name) are stored **separately** from
  the image data, under lock or encryption, accessible only to the named
  investigator.
- The linking list (code ↔ name) is the only bridge. Store it with the consent
  forms, never in the repository.

---

## 3. Who to recruit

- **Eligible:** any currently practising prescriber (MBBS or above), any
  specialty, any seniority.
- **Exclude:** anyone unable to give informed consent; anyone who wants their
  handwriting published identifiably.
- **Target: 30–50 writers.** Style diversity is what we are short of, not
  sample count. Thirty different hands is worth far more than three thousand
  samples from three hands.
- Recruit across seniority (interns, residents, consultants) and across
  settings (OPD, ward, private clinic). Junior and senior hands differ.

**What you can realistically expect.** 30 writers × 2 forms × 22 cells ≈ **1,320
samples**. That is far too small to train from scratch (we use 48,000 synthetic
images). Its value is:

1. **As a test set** — the honest measurement of the real-world domain gap.
   This is the primary purpose.
2. **For fine-tuning** — adapting the synthetic-trained model to real hands.

Do not describe it as a training corpus.

---

## 4. Materials

Generate the forms:

```bash
python scripts/make_forms.py --n-forms 20
```

This writes `data/forms/LRG-0NN.pdf` (print these), matching `.png` previews,
and `forms_manifest.json`, which records each cell's prompt and its position
normalised to the four ArUco markers. **Keep the manifest** — without it the
photographs cannot be labelled.

Each A4 sheet holds **22 cells**. Prompts over-sample documented ISMP/FDA
confusable names (~60%), because those are the reads that matter for safety;
the remainder are drawn from the general corpus for diversity.

Print at **100% scale, A4, no "fit to page"**. Rescaling moves the markers
relative to the cells and breaks rectification.

---

## 5. Collection session (about 6 minutes per writer)

1. Explain the study using the participant information sheet. Answer questions.
2. Obtain **written consent** before anything is written.
3. Assign the next unused writer code. Record it on the consent form, *not* on
   the sample sheet in any identifying way — write only the code in the
   "Writer code" field.
4. Hand over **2 forms** (3 if willing) and a normal ballpoint. Let them use
   their own pen if they prefer — pen variation is useful signal.
5. Ask them to write at normal prescribing speed. Repeat that neatness is not
   wanted.
6. Confirm no patient information has been written anywhere on the sheet.

**Do not** correct, coach, or ask anyone to rewrite a word because it is
illegible. Illegible writing is the phenomenon under study.

---

## 6. Capture

Photograph each completed sheet with a phone:

- The **whole sheet in frame**, including all four corner markers.
- Flat surface, even light. **No flash** — glare erases strokes.
- Avoid casting your own shadow across the page.
- Moderate angle is fine; the markers correct perspective.

Name each file `<writercode>_<formid>.jpg`, e.g. `W03_LRG-001.jpg`.

Then ingest:

```bash
python scripts/ingest_forms.py --photos path/to/photos --out data/real
```

This rectifies each sheet, crops each cell, drops blank cells, and writes
`labels.tsv` in exactly the schema the synthetic pipeline uses — so real data
loads through the existing loaders with no code changes. Splits are assigned
**by writer**, so no hand appears in both training and test.

**Run the ingest before the prescriber leaves, if possible.** Any sheet
reported as skipped can be re-photographed on the spot. Re-shooting later means
tracking the person down again.

---

## 7. Data handling

| | |
|---|---|
| Stored | Cropped word images, writer code, prompt text, form ID, date |
| Never stored | Names, registration numbers, hospital, specialty, patient data |
| Consent forms | Separately, locked or encrypted, investigator access only |
| Retention | Per IEC ruling; state a definite period in the application |
| Withdrawal | A writer may withdraw at any time; delete all crops bearing their code and say so in writing |

If the dataset is ever released publicly, release **only** the cropped word
images and labels — never the full sheets, which carry handwritten free text
in the header fields.

---

## 8. Quality checks before you call it done

- [ ] Every photograph ingested, or re-shot until it did
- [ ] ≥30 distinct writer codes
- [ ] Spot-check 30 random crops against their labels — the label is the
      *prompt*, so a mismatch means a geometry or manifest error, which would
      corrupt the whole batch
- [ ] `data/real/meta.json` shows no writer appearing in two splits
- [ ] No patient information on any sheet (check before photographing, and
      again before archiving)
- [ ] Consent form on file for every writer code present in the data

---

## 9. What to do with the result

```bash
# 1. The honest number: synthetic-trained model, real handwriting
python -m lasa.evaluation.evaluate --data data/real --splits test

# 2. Fine-tune on real train split, re-evaluate on held-out writers
python -m lasa.train.train_crnn --data data/real --resume outputs/checkpoints/crnn_ctc/best.pt
```

Report **both**: the zero-shot number measures the domain gap, which is the
scientifically interesting quantity; the fine-tuned number measures what a
deployed system could achieve. Reporting only the second hides the gap.
