# Deployment considerations — pharmacy-counter device

Target: a low-cost box sitting next to the dispensing counter in a clinic or
retail pharmacy, ideally in a setting where connectivity is unreliable.

Numbers marked **measured** come from real runs on this project's hardware
(Apple M2, MPS). Numbers marked **target** or **TBD** are budgets, not results —
they must be replaced with measurements on the actual device before anyone
quotes them.

---

## 1. What actually has to run

| Stage | Cost driver | Notes |
|---|---|---|
| Word segmentation | OpenCV morphology | Negligible; pure CPU, sub-millisecond at counter resolutions |
| OCR | The model | Dominates end-to-end latency |
| LASA matching | 2,000-name search | Bounded and cheap — see §4 |

The two OCR arms have very different deployment profiles, and that difference —
not the raw accuracy gap — is the real finding of the comparison.

| | CRNN (from scratch) | TrOCR (fine-tuned base) |
|---|---|---|
| Parameters | **8,379,740 (measured)** | ~334M |
| Checkpoint on disk | ~32 MB fp32 (measured) | ~1.3 GB fp32 |
| Architecture | CNN + BiLSTM + CTC, single forward pass | ViT encoder + autoregressive transformer decoder |
| Decoding | One pass, then a cheap CTC collapse | One decoder forward **per output token**, ×beam width |
| Realistic target hardware | Raspberry Pi 5 / low-end x86 mini-PC | Needs a GPU, or a server |

An 8.4M-parameter single-pass model and a 334M-parameter autoregressive one are
not in the same deployment class. For a counter device with no GPU, the CRNN is
the only one of the two that is plausibly deployable as-is.

## 2. ONNX export — what converts cleanly and what does not

**CRNN: straightforward.** Conv + BiLSTM + Linear all have stable ONNX
operators. Export with a dynamic width axis so variable-length word crops still
work:

```python
torch.onnx.export(
    model, dummy,                       # dummy: [1, 1, 32, W]
    "crnn.onnx", opset_version=17,
    input_names=["image"], output_names=["log_probs"],
    dynamic_axes={"image": {0: "batch", 3: "width"},
                  "log_probs": {0: "time", 1: "batch"}},
)
```

Two things to verify rather than assume: that ONNX Runtime's LSTM kernel matches
PyTorch's numerically (compare log-probs on a fixed batch, not just output
strings — a small drift can flip a close CTC decision), and that the dynamic
width axis really is dynamic (export with one width, run with three).

**TrOCR: this is the hard part, and it is worth being blunt about it.** A
`VisionEncoderDecoderModel` does not export to a single usable ONNX graph by
naively tracing `.generate()` — generation is a Python loop with a KV cache,
beam bookkeeping and early stopping. The workable path is to export **three**
graphs (encoder, decoder-with-past, decoder-init) and reimplement the search
loop on the host, which is what `optimum.onnxruntime` does:

```bash
optimum-cli export onnx -m outputs/checkpoints/trocr_ft --task image-to-text-with-past trocr_onnx/
```

Expect to spend real time on the KV-cache plumbing and on matching beam search
exactly. Anyone who claims "we just exported TrOCR to ONNX" and reports one
latency number has usually exported the encoder only.

## 3. Quantisation

**Dynamic INT8** is the first thing to try for the CRNN — it needs no
calibration set and targets exactly the layers that dominate:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("crnn.onnx", "crnn.int8.onnx", weight_type=QuantType.QInt8)
```

Expected: ~4× smaller weights (≈32 MB → ≈8 MB) and a meaningful CPU speedup on
the Linear/LSTM layers. **The accuracy cost must be measured, not assumed** —
re-run `lasa.evaluation.evaluate` against the quantised model and report CER
side by side with fp32. For this task specifically, quantisation error is more
dangerous than the headline CER suggests: the failure mode that matters is a
*confusable* substitution, and a model that degrades from `dopamine` to
`dobutamine` is far worse than one that degrades to `dopamlne`. Report the
change in **Stage-2 RED/YELLOW recall** after quantisation, not just CER.

Static INT8 (with a calibration set drawn from the training split) is worth
trying for the convolutional trunk if dynamic is not enough. For TrOCR, INT8 on
the decoder tends to hurt more, because errors compound autoregressively.

## 4. Latency budget

**Target: < 300 ms per word**, end to end, on the counter device. That is roughly
the threshold at which a pharmacist experiences the check as instant rather than
as a wait, and it leaves headroom for a 3–5 word prescription line.

Where the time goes:

| Stage | Budget | Status |
|---|---|---|
| Segmentation | < 10 ms | Measured negligible on M2; **TBD on target hardware** |
| CRNN forward (fp32, batch 1) | < 150 ms | **Measured on M2** — see `outputs/reports/ocr_comparison.json`; **TBD on Pi 5** |
| CTC greedy decode | < 5 ms | Measured < 1 ms |
| Lexicon-constrained decode | < 60 ms | Measured ~26 ms/sample on CPU at beam 10 |
| LASA matching | < 20 ms | RapidFuzz prefilter over 2,000 names, then fused scoring on ~60 survivors |

The LASA stage is intentionally cheap and **does not need a GPU or a model** —
which matters, because it is the stage that can still run when OCR is degraded
or unavailable (see §6).

Beam search is the obvious latency lever: greedy is ~26× faster than beam-10 for
the CRNN. Whether beam is worth its cost is an empirical question answered in
`docs/results.md`, not a default.

## 5. Offline operation

The system must work with no network:

- **Privacy.** A prescription is health data. Under India's DPDP Act 2023 it is
  personal data whose processing needs a lawful basis; shipping images to a
  cloud OCR API turns every dispensing event into a data-transfer event with a
  third party. On-device inference avoids that class of problem entirely rather
  than mitigating it.
- **Connectivity.** The rural PHC and small-town pharmacy settings where this is
  most useful are exactly where connectivity is least reliable. A system that
  degrades to unusable when the link drops will be switched off.
- **Latency.** No round trip.

Everything needed ships with the device: model weights, the drug corpus CSV, the
LASA pair CSV, and the fitted n-gram index (`NgramIndex.save`). Total footprint
with an INT8 CRNN is on the order of tens of megabytes.

Updates (new drugs, revised ISMP list) should be a signed data-file drop that
does not require re-training — the corpus and pair list are plain CSVs read at
startup precisely so this is possible.

## 6. Degradation and the fallback mode

The single most important design property: **the system must fail loudly, not
silently**.

- If OCR confidence is below threshold, the pipeline does **not** present a
  best guess as fact. `LasaMatcher` already returns YELLOW with an explicit
  "treat this as unread" reason when the best formulary match scores below the
  floor. Garbage in must never render as GREEN.
- If OCR is unavailable entirely (model failed to load, image unreadable), the
  Stage-2 engine still works from typed input. A pharmacist typing what they
  think the name says and getting "that has a documented confusable" is still
  most of the safety value.
- RED is a *block-and-confirm* gate, not a warning banner to be clicked past.

## 7. Human-in-the-loop and alarm fatigue

The system advises; the pharmacist decides. This is not a disclaimer, it is a
design constraint that shaped the thresholds.

The YELLOW band was recalibrated from a provisional 0.62 to **0.78** precisely
because 0.62 flagged **36% of unrelated names** — a third of prescriptions
raising an alert. Alarm fatigue is a well-documented way clinical alerting
systems fail: when most alerts are noise, staff learn to dismiss all of them,
including the real ones. The recalibration cut false positives to **15.5%** with
recall unchanged at 97.9%. Both operating points are reported in
`docs/results.md` rather than only the flattering one.

Every alert carries a plain-language reason and, where published, the tall-man
form (`DOBUTamine` vs `DOPamine`) — because tall-man lettering *is* the
recommended mitigation, and an unexplained warning gets dismissed reflexively.

## 8. Regulatory and safety position

**This is an unvalidated prototype, not a medical device.**

Software intended to support clinical decisions falls under India's Medical
Device Rules 2017 (CDSCO), which took a risk-based classification approach to
software as a medical device; equivalent regimes exist elsewhere (EU MDR,
FDA CDS guidance). Anything resembling real deployment would need, at minimum:

- A clinical evaluation on **real** prescriptions from the target setting — not
  synthetic renders and not IAM. Everything reported here is on synthetic data;
  the domain gap to genuine prescriber handwriting is unmeasured and is the
  largest single source of uncertainty in this project.
- A prospective study of whether the alerts change dispensing behaviour and
  whether they cause alarm fatigue in practice.
- Formal risk management (ISO 14971), quality management (ISO 13485), and a
  software lifecycle process (IEC 62304).
- Versioning and audit of the drug corpus and LASA pair list, with provenance
  for each entry.
- A defined human-factors design for how RED is surfaced and overridden, with
  overrides logged.

Until all of that exists, the honest framing is: a demonstration that the
approach is technically feasible, on synthetic data, with the error modes
measured and stated.
