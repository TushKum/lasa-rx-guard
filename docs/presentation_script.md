# 4-minute presentation script — two presenters

Audience: an ML professor. Assume CTC, CRNN and transformers need no
explanation. What earns credit here is **methodology and honesty**, not the
accuracy number.

Total ≈ 565 spoken words ≈ 4:00 at a normal pace. **T** = Tushit, **B** = Bhavik.

---

## [0:00–0:45] T — The problem
*On screen: DOPamine / DOBUTamine side by side, handwritten*

Some drug names are so alike that regulators publish them as hazards.
DOPamine and DOBUTamine. Both ISMP high-alert — confusing them is a
cardiovascular dosing error, not a typo.

The published mitigation is Tall Man lettering: capitalise the letters that
differ. But handwriting destroys that. There is no controlled capitalisation on
a handwritten chit, and the pharmacist has seconds.

Two research lines touch this problem and never meet. Handwriting recognition
reads text well — but its benchmarks contain no drug vocabulary, and it never
asks whether a confident reading is dangerous. Drug-name confusion research
measures exactly that danger — but on typed names, at drug-approval time.

We couple the two.

## [0:45–1:45] B — Method
*On screen: two-stage diagram, then the four-split table*

Two stages. Stage one reads the word. Stage two asks whether that reading is
safe.

On data: public handwriting corpora have no medicine names, and drug registries
have no images. So we generated 59,000 handwritten drug-name images — 2,000
names from RxNorm and India's essential medicines list, 25 open-licence
handwriting fonts, per-glyph jitter, and a simulated imaging path.

The part we would defend hardest is the split design. A single test set cannot
distinguish a model that reads from one that memorises. So we held out
vocabulary and writing style on **independent axes** — four test sets.

Stage one is a CRNN with CTC loss, 8.4 million parameters, trained from
scratch. Stage two scores the reading against the formulary on three signals —
edit distance, phonetic encoding, character n-grams — validated against 164
documented ISMP and FDA confusable pairs.

## [1:45–2:55] T — Results and the finding that mattered
*On screen: the CER-by-split bar chart*

On seen conditions: 2.61% character error rate. Constrained to the known
formulary — which is what a pharmacy actually has — 0.18%, and 99.5% of whole
names exactly right.

But the number that taught us something is this. Words the model had never
seen: 3.00%. Handwriting it had never seen: 19.29%. **Seven point four times
worse.**

We tested that rather than eyeballing it. Bootstrap, 5,000 resamples. The
vocabulary difference is plus 0.40 percentage points, confidence interval minus
0.02 to plus 0.81 — it includes zero. The style difference is plus 16.68,
interval 15.77 to 17.60.

So the model genuinely learned to read characters. More drug names buys us
nothing. More handwriting styles is the entire game. That changed what we do
next.

One debugging note you may appreciate: 47.8% of our training set was silently
invisible to CTC — fewer encoder frames than the label required, so zero loss
and zero gradient, with no error raised. It presented as a hyperparameter
problem.

## [2:55–4:00] B — What didn't work, and what's next
*On screen: ablation table, then the collection form*

Now the parts that did not work, because we think they matter more.

Our three-signal fusion **loses to plain edit distance** — on top-1, on top-3,
and on surfacing the documented partner. We report that. We keep the phonetic
channel, but the honest reason is that our noise model is biased against it by
construction: it perturbs characters toward visually similar glyphs, which is
correct for OCR error and systematically rewards spelling. Sound-alike confusion
comes from spoken orders, and we did not model that — so we cannot claim the
phonetic channel earns its weight.

Second, our transformer baseline did not converge — 96% error. That is a compute
budget failure, not a verdict on TrOCR, and we claim no comparison.

Third, all our handwriting is synthetic. That is the real limitation. So we have
specified and validated the collection protocol — prompted forms, no patient
data, exact labels — and that is the next phase.

Thank you.

---

## Anticipated questions

**"Why not just fine-tune a transformer?"**
We tried; it did not converge on the compute we had, and we report it as a
failed run. On deployment grounds a CRNN is also the better fit: 8.4M parameters
and a single forward pass at 3.8 ms, versus 334M and autoregressive decoding
that does not export cleanly to ONNX.

**"Isn't 2.61% on synthetic data meaningless?"**
On its own, yes. Which is why the headline finding is the *ratio*, not the rate:
unseen style costs 7.4× unseen vocabulary. That relationship should survive the
domain shift even though the absolute number will not.

**"If edit distance wins, why keep the fusion?"**
Strictly, under this evaluation we should not — and the paper says so. The
defence is that our noise model cannot reward a phonetic signal by
construction. Settling it needs a verbal-error corpus, which we do not have.

**"How do you know your splits are really disjoint?"**
Verified programmatically over the 59,000-row manifest, and the assertion is a
regression test, not a one-off check.

**"What's the false-alarm rate?"**
At the advisory tier, 20.67% at 0.2 errors per character — which we treat as a
finding, not a footnote. Clinicians already override 49–96% of drug safety
alerts, so a system interrupting one prescription in five is a safety problem of
its own. The block-and-confirm tier runs at 0.33%.

---

## Delivery notes

- **Do not read this.** Learn the four beats; the numbers are the only parts
  worth saying verbatim.
- Hand over cleanly — the switch is at 0:45, 1:45 and 2:55. Rehearse the seams.
- If you are running long, cut the CTC debugging note at 2:40. Keep every
  negative result.
- Have the Streamlit demo open on a second tab. If asked, type `dobutamin`
  live — it is faster than describing it.
