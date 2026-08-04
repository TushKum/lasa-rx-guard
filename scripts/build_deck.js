/**
 * Build the LASA-RxGuard submission deck.
 *
 *   node scripts/build_deck.js
 *
 * Every figure here is measured — see docs/results.md. Nothing is illustrative.
 */
const pptxgen = require("pptxgenjs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "outputs", "figures");

// Clinical navy ground; the triage trio is the motif and carries every
// risk-related mark in the deck.
const INK = "0A1A2F", NAVY = "12315B", STEEL = "4A6285";
const ICE = "E8EDF3", MIST = "F4F6F9", WHITE = "FFFFFF", SLATE = "5A6B80";
const RED = "A8232B", AMBER = "8A5D00", GREEN = "1B6047";
const RED_T = "F7E9E9", AMBER_T = "FAF2E0", GREEN_T = "E6F0EA";

const SERIF = "Cambria", SANS = "Calibri";
const W = 13.333, H = 7.5, M = 0.62;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "LASA-RxGuard";
pres.title = "LASA-RxGuard — Preventing look-alike drug dispensing errors";

/* ── helpers ─────────────────────────────────────────────────────────── */

// Triage dot — the deck's repeated motif.
function dot(s, x, y, color, d = 0.17) {
  s.addShape(pres.ShapeType.ellipse,
    { x, y, w: d, h: d, fill: { color }, line: { width: 0 } });
}

function head(s, kicker, title, opts = {}) {
  s.addText(kicker.toUpperCase(), {
    x: M, y: 0.42, w: 9, h: 0.26, fontFace: SANS, fontSize: 11, bold: true,
    color: opts.dark ? "8FB3DC" : NAVY, charSpacing: 2.2, margin: 0,
  });
  s.addText(title, {
    x: M, y: 0.74, w: opts.w || 11.4, h: 0.9, fontFace: SERIF, fontSize: 34,
    bold: true, color: opts.dark ? WHITE : INK, margin: 0,
  });
}

function foot(s, text, page, dark) {
  s.addText(text, {
    x: M, y: H - 0.62, w: 10.6, h: 0.4, fontFace: SANS, fontSize: 9,
    color: dark ? "7E93AC" : SLATE, margin: 0, valign: "top",
  });
  s.addText(String(page), {
    x: W - M - 0.5, y: H - 0.62, w: 0.5, h: 0.3, fontFace: SANS, fontSize: 9,
    color: dark ? "7E93AC" : SLATE, align: "right", margin: 0,
  });
}

// Soft card — tint + shadow, never an edge stripe.
function card(s, x, y, w, h, fill = MIST) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.04, fill: { color: fill }, line: { width: 0 },
    shadow: { type: "outer", angle: 90, blur: 8, offset: 1, color: "9AA9BC", opacity: 0.22 },
  });
}

function stat(s, x, y, w, value, label, desc, color = NAVY) {
  s.addText(value, { x, y, w, h: 0.72, fontFace: SERIF, fontSize: 40, bold: true,
    color, margin: 0 });
  s.addText(label.toUpperCase(), { x, y: y + 0.72, w, h: 0.26, fontFace: SANS,
    fontSize: 10, bold: true, color: SLATE, charSpacing: 1.6, margin: 0 });
  if (desc) s.addText(desc, { x, y: y + 1.02, w, h: 0.95, fontFace: SANS,
    fontSize: 11.5, color: STEEL, margin: 0, valign: "top" });
}

const TBL = {
  fontFace: SANS, fontSize: 12.5, color: INK, border: { type: "solid", pt: 0.5, color: "D3DBE4" },
  autoPage: false, valign: "middle",
};
const TH = { fill: { color: INK }, color: WHITE, bold: true, fontSize: 10.5, align: "right" };
const th = (t, first) => ({ text: t.toUpperCase(), options: { ...TH, align: first ? "left" : "right" } });
const td = (t, o = {}) => ({ text: t, options: { align: "right", ...o } });
const tdl = (t, o = {}) => ({ text: t, options: { align: "left", ...o } });

/* ── 1 · title (dark) ────────────────────────────────────────────────── */
{
  const s = pres.addSlide();
  s.background = { color: INK };
  s.addText("LASA-RxGuard", { x: M, y: 0.5, w: 8, h: 0.3, fontFace: SANS,
    fontSize: 11, bold: true, color: "8FB3DC", charSpacing: 3, margin: 0 });

  s.addText("Preventing look-alike\ndrug dispensing errors", {
    x: M, y: 1.9, w: 8.6, h: 2.1, fontFace: SERIF, fontSize: 50, bold: true,
    color: WHITE, lineSpacing: 52, margin: 0 });
  s.addText("Handwritten prescription recognition with a look-alike / sound-alike confusion risk engine, validated against published ISMP and FDA confused-drug-name pairs.",
    { x: M, y: 4.15, w: 7.4, h: 0.9, fontFace: SANS, fontSize: 14.5,
      color: "AFC2D6", margin: 0 });

  // Motif introduced on the cover: the three states the system emits.
  [["RED", RED], ["YELLOW", AMBER], ["GREEN", GREEN]].forEach(([lab, c], i) => {
    const y = 1.95 + i * 0.62;
    dot(s, 10.15, y + 0.07, c, 0.2);
    s.addText(lab, { x: 10.5, y, w: 2.4, h: 0.34, fontFace: SANS, fontSize: 13,
      bold: true, color: WHITE, margin: 0 });
  });
  s.addText("Triage output", { x: 10.15, y: 1.55, w: 2.6, h: 0.26, fontFace: SANS,
    fontSize: 9.5, bold: true, color: "7E93AC", charSpacing: 1.6, margin: 0 });

  s.addText([
    { text: "ML Bubble 2026 · AIT Pune", options: { breakLine: true } },
    { text: "TE-BE · Design & Solve (Advanced)", options: { breakLine: true } },
    { text: "github.com/TushKum/lasa-rx-guard", options: {} },
  ], { x: M, y: 5.6, w: 8, h: 1, fontFace: SANS, fontSize: 12, color: "AFC2D6",
       lineSpacing: 19, margin: 0 });
  s.addText("Advisory prototype. Not a validated medical device.", {
    x: W - M - 5.2, y: H - 0.7, w: 5.2, h: 0.3, fontFace: SANS, fontSize: 9.5,
    color: "7E93AC", align: "right", margin: 0, italic: true });
  s.addNotes("Two drugs, one unreadable word. The system reads the handwritten name, then flags when that reading cannot safely distinguish a documented dangerous pair.");
}

/* ── 2 · problem ─────────────────────────────────────────────────────── */
{
  const s = pres.addSlide();
  head(s, "01 · The problem", "Two drugs, one unreadable word");

  s.addText("Certain drug names look or sound so alike that regulators publish them as known hazards. Handwriting removes the last cue separating them — and the pharmacist is the final check before the patient.",
    { x: M, y: 1.85, w: 5.4, h: 1.3, fontFace: SANS, fontSize: 15, color: INK,
      lineSpacing: 23, margin: 0 });

  card(s, M, 3.3, 5.4, 1.75, RED_T);
  dot(s, M + 0.3, 3.62, RED, 0.19);
  s.addText("DOPamine  ·  DOBUTamine", { x: M + 0.62, y: 3.52, w: 4.6, h: 0.36,
    fontFace: SANS, fontSize: 15, bold: true, color: RED, margin: 0 });
  s.addText("Both ISMP high-alert medications. Confusing them is a cardiovascular dosing error, not a typo. The capitalisation shown is the published mitigation.",
    { x: M + 0.62, y: 3.94, w: 4.5, h: 0.95, fontFace: SANS, fontSize: 11.5,
      color: "6B2D31", lineSpacing: 16, margin: 0 });

  s.addTable([
    [th("Documented pair", true), th("Source"), th("High-alert")],
    [tdl("DOPamine / DOBUTamine"), td("FDA tall-man", { color: SLATE }), td("Yes", { bold: true, color: RED })],
    [tdl("hydrOXYzine / hydrALAZINE"), td("FDA tall-man", { color: SLATE }), td("—", { color: SLATE })],
    [tdl("Celebrex / Celexa / Cerebyx"), td("ISMP", { color: SLATE }), td("—", { color: SLATE })],
    [tdl("vinBLAStine / vinCRIStine"), td("FDA tall-man", { color: SLATE }), td("Yes", { bold: true, color: RED })],
    [tdl("Losec / Lasix"), td("ISMP", { color: SLATE }), td("—", { color: SLATE })],
    [tdl("Ground-truth pairs used", { bold: true }), td("ISMP + FDA", { color: SLATE }),
     td("164", { bold: true, color: NAVY })],
  ], { x: 6.5, y: 1.85, w: 6.2, colW: [3.1, 1.75, 1.35], rowH: 0.42, ...TBL,
       fill: { color: WHITE } });

  s.addText("Losec / Lasix scores 0.52 on spelling but 0.81 phonetically — identical Soundex L220. An orthographic-only matcher ranks it far down the list.",
    { x: 6.5, y: 5.2, w: 6.2, h: 0.7, fontFace: SANS, fontSize: 11.5,
      color: STEEL, italic: true, lineSpacing: 16, margin: 0 });

  foot(s, "Source: ISMP List of Confused Drug Names; FDA/ISMP tall-man lettering list. Curated subset of 164 unique pairs covering 272 distinct names.", 2);
}

/* ── 3 · architecture ────────────────────────────────────────────────── */
{
  const s = pres.addSlide();
  head(s, "02 · Architecture", "Recognition, then risk — deliberately separable");

  const boxes = [
    ["Stage 1 · Read", "CRNN — convolutional trunk, bidirectional LSTM, CTC loss. Trained from scratch, 8,379,740 parameters."],
    ["Stage 2 · Match", "Reading scored against a 2,000-name formulary by three independent similarity signals."],
    ["Fusion", "Edit 0.45 · phonetic 0.35 · character n-gram 0.20. Weights ablated, not assumed."],
    ["Output", "Top-3 candidates, a triage flag, and a plain-language reason a pharmacist can act on."],
  ];
  boxes.forEach(([t, d], i) => {
    const x = M + i * 3.11;
    card(s, x, 1.95, 2.86, 2.15);
    s.addText(String(i + 1).padStart(2, "0"), { x: x + 0.26, y: 2.14, w: 1, h: 0.34,
      fontFace: SERIF, fontSize: 17, bold: true, color: "AEBECE", margin: 0 });
    s.addText(t, { x: x + 0.26, y: 2.5, w: 2.36, h: 0.34, fontFace: SANS,
      fontSize: 14.5, bold: true, color: NAVY, margin: 0 });
    s.addText(d, { x: x + 0.26, y: 2.86, w: 2.36, h: 1.1, fontFace: SANS,
      fontSize: 11.5, color: STEEL, lineSpacing: 16, margin: 0 });
  });

  card(s, M, 4.45, 12.09, 1.5, ICE);
  s.addText("The two stages are independent by design.", { x: M + 0.35, y: 4.68,
    w: 11.4, h: 0.32, fontFace: SANS, fontSize: 14.5, bold: true, color: INK, margin: 0 });
  s.addText("Stage 2 carries no dependency on the deep-learning stack at all — so when recognition confidence collapses, or the model fails to load, the risk engine still runs from a typed name. Degradation is graceful rather than total.",
    { x: M + 0.35, y: 5.04, w: 11.3, h: 0.75, fontFace: SANS, fontSize: 12.5,
      color: STEEL, lineSpacing: 17, margin: 0 });

  foot(s, "Fusion weights in configs/risk.yaml; per-signal ablation on slide 10.", 3);
}

/* ── 4 · data ────────────────────────────────────────────────────────── */
{
  const s = pres.addSlide();
  head(s, "03 · Corpus", "Built, not borrowed");
  s.addText("Public handwriting corpora contain almost no medicine names; drug registries contain no images. The corpus bridges the two — and every asset is traceable.",
    { x: M, y: 1.68, w: 8.2, h: 0.6, fontFace: SANS, fontSize: 13.5, color: STEEL,
      lineSpacing: 19, margin: 0 });

  s.addTable([
    [th("Asset", true), th("Count"), th("Origin"), th("Note")],
    [tdl("Drug names"), td("2,000", { bold: true }), tdl("RxNorm (RxNav API), NLEM India 2022", { color: SLATE }),
     tdl("14,648 raw ingredients filtered", { color: SLATE })],
    [tdl("LASA pairs"), td("164", { bold: true }), tdl("ISMP, FDA tall-man list", { color: SLATE }),
     tdl("272 names, all present in corpus", { color: SLATE })],
    [tdl("Handwriting fonts"), td("25", { bold: true }), tdl("Google Fonts, OFL / Apache-2.0", { color: SLATE }),
     tdl("SHA-256 pinned, machine-identical", { color: SLATE })],
    [tdl("Synthetic images"), td("59,000", { bold: true }), tdl("Generated", { color: SLATE }),
     tdl("Per-glyph jitter, then degradation", { color: SLATE })],
    [tdl("Real handwriting"), td("22,506", { bold: true }), tdl("IAM via Teklia/IAM-line", { color: SLATE }),
     tdl("69.1% of lines discarded on alignment", { color: SLATE })],
  ], { x: M, y: 2.45, w: 12.09, colW: [2.2, 1.25, 4.3, 4.34], rowH: 0.44, ...TBL,
       fill: { color: WHITE } });

  s.addImage({ path: path.join(FIG, "synth_preview.png"), x: M, y: 5.35, w: 3.0, h: 1.42,
    sizing: { type: "crop", w: 3.0, h: 1.42, x: 0, y: 0 } });
  s.addText("Generated samples span three difficulty tiers. A legibility guard rejects any draw a human could not read — mislabelled data caps achievable error rate regardless of model.",
    { x: 3.85, y: 5.4, w: 8.85, h: 0.85, fontFace: SANS, fontSize: 12,
      color: STEEL, lineSpacing: 17, margin: 0 });

  foot(s, "Sources: U.S. National Library of Medicine RxNorm; MoHFW NLEM 2022; Google Fonts; Teklia/IAM-line.", 4);
}

/* ── 5 · splits ──────────────────────────────────────────────────────── */
{
  const s = pres.addSlide();
  head(s, "04 · Method", "Four test splits, built to expose memorisation");
  s.addText("A single test set cannot distinguish a model that reads from a model that memorises. Vocabulary and writing style are held out on independent axes.",
    { x: M, y: 1.68, w: 8.6, h: 0.6, fontFace: SANS, fontSize: 13.5, color: STEEL,
      lineSpacing: 19, margin: 0 });

  s.addTable([
    [th("Split", true), th("Vocabulary"), th("Writing style"), th("Question it settles")],
    [tdl("test_seen"), td("Trained", { color: SLATE }), td("Trained", { color: SLATE }),
     tdl("Operational accuracy on a known formulary", { color: SLATE })],
    [tdl("test_unseen_font"), td("Trained", { color: SLATE }), td("Held out", { bold: true, color: NAVY }),
     tdl("Robustness to an unfamiliar hand", { color: SLATE })],
    [tdl("test_unseen_word"), td("Held out", { bold: true, color: NAVY }), td("Trained", { color: SLATE }),
     tdl("Genuine character-level generalisation", { color: SLATE })],
    [tdl("test_hard"), td("Held out", { bold: true, color: NAVY }), td("Held out", { bold: true, color: NAVY }),
     tdl("Both simultaneously", { color: SLATE })],
  ], { x: M, y: 2.45, w: 12.09, colW: [2.6, 2.0, 2.2, 5.29], rowH: 0.46, ...TBL,
       fill: { color: WHITE } });

  [["0", "Word overlap", "Between train and test_unseen_word. Verified, not assumed."],
   ["0", "Font overlap", "test_hard is disjoint on both axes simultaneously."],
   ["1337", "Seed", "Holdouts, augmentation and ordering all reproducible."]]
    .forEach(([v, l, d], i) => stat(s, M + i * 4.1, 4.9, 3.7, v, l, d));

  foot(s, "Disjointness verified programmatically across the 59,000-row manifest; the assertion is retained in the test suite.", 5);
}

/* ── 6 · stage 1 results ─────────────────────────────────────────────── */
{
  const s = pres.addSlide();
  head(s, "05 · Stage 1 results", "Recognition accuracy by split and decoder");

  s.addTable([
    [th("Decoder", true), th("test_seen"), th("unseen font"), th("unseen word"), th("hard"), th("Exact (seen)")],
    [tdl("Greedy"), td("2.91%"), td("19.45%"), td("2.83%"), td("22.71%"), td("81.75%")],
    [tdl("Beam, width 8"), td("2.80%"), td("19.09%"), td("2.85%"), td("22.46%"), td("82.25%")],
    [tdl("Lexicon-constrained", { bold: true }), td("0.48%", { bold: true, color: NAVY }),
     td("5.23%", { bold: true, color: NAVY }), td("0.08%", { bold: true, color: NAVY }),
     td("8.16%", { bold: true, color: NAVY }), td("98.75%", { bold: true, color: NAVY })],
  ], { x: M, y: 1.9, w: 12.09, colW: [3.09, 1.8, 1.8, 1.8, 1.8, 1.8], rowH: 0.5, ...TBL,
       fill: { color: WHITE } });

  s.addText("Character error rate, corpus-level micro-average. n = 400 per split.",
    { x: M, y: 4.0, w: 8, h: 0.3, fontFace: SANS, fontSize: 11, color: SLATE,
      italic: true, margin: 0 });

  const notes = [
    ["Beam search does not pay", "0.11pp of CER for roughly 26× the decode cost. Greedy is the correct default at the counter."],
    ["Lexicon is a different task", "Snapping to the nearest known name is what a pharmacy deploys — reported on its own row, never merged into free-decoding CER."],
    ["Second model unresolved", "A fine-tuned TrOCR arm did not converge within available compute, and is reported as a failed run rather than a comparison."],
  ];
  notes.forEach(([t, d], i) => {
    const x = M + i * 4.1;
    card(s, x, 4.45, 3.7, 1.85);
    s.addText(t, { x: x + 0.26, y: 4.66, w: 3.2, h: 0.5, fontFace: SANS, fontSize: 13,
      bold: true, color: NAVY, margin: 0 });
    s.addText(d, { x: x + 0.26, y: 5.16, w: 3.2, h: 1.0, fontFace: SANS, fontSize: 11,
      color: STEEL, lineSpacing: 15, margin: 0 });
  });

  foot(s, "Measured on the epoch-5 checkpoint; the run ended at epoch 6 on a GPU memory fault, documented in docs/results.md.", 6);
}

/* ── 7 · principal finding (chart) ───────────────────────────────────── */
{
  const s = pres.addSlide();
  head(s, "06 · Principal finding", "The constraint is handwriting style, not vocabulary", { w: 8.4 });

  s.addChart(pres.ChartType.bar, [{
    name: "Character error rate",
    labels: ["Seen words", "Unseen words", "Unseen fonts", "Both held out"],
    values: [2.91, 2.83, 19.45, 22.71],
  }], {
    x: 6.3, y: 1.85, w: 6.35, h: 4.05,
    barDir: "bar", chartColors: [NAVY, NAVY, RED, RED],
    varyColors: true, barGapWidthPct: 45,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.00"%"',
    dataLabelFontFace: SANS, dataLabelFontSize: 11.5, dataLabelColor: INK,
    catAxisLabelFontFace: SANS, catAxisLabelFontSize: 12, catAxisLabelColor: INK,
    valAxisLabelFontFace: SANS, valAxisLabelFontSize: 10, valAxisLabelColor: SLATE,
    valAxisMaxVal: 27, valGridLine: { color: "E4E9EF", size: 0.5 },
    catGridLine: { style: "none" }, showLegend: false, showTitle: false,
  });

  s.addText("Unseen drug names cost the model nothing. Unseen handwriting costs it almost seven times its error rate.",
    { x: M, y: 1.95, w: 5.3, h: 1.15, fontFace: SERIF, fontSize: 21, bold: true,
      color: INK, lineSpacing: 27, margin: 0 });
  s.addText("The model genuinely learned to read characters rather than memorise an 1,800-word list — so the return on adding vocabulary is close to zero, while the return on adding writers is large.\n\nThis also sets expectations for real prescriber handwriting, which sits further from the training distribution than any held-out font.",
    { x: M, y: 3.2, w: 5.3, h: 1.9, fontFace: SANS, fontSize: 12.5, color: STEEL,
      lineSpacing: 18, margin: 0 });

  card(s, M, 5.25, 5.3, 0.95, ICE);
  s.addText("Actionable consequence: collect writers, not words.", {
    x: M + 0.3, y: 5.48, w: 4.8, h: 0.5, fontFace: SANS, fontSize: 13.5, bold: true,
    color: NAVY, margin: 0 });

  foot(s, "Seen vs unseen vocabulary differs by 0.08pp — within run-to-run noise. Seen vs unseen style differs by 16.54pp.", 7);
}

/* ── 8 · live demo, RED ──────────────────────────────────────────────── */
{
  const s = pres.addSlide();
  head(s, "07 · Live demo", "What the pharmacist actually sees", { w: 7.6 });

  s.addImage({ path: path.join(FIG, "demo", "demo_red.png"),
    x: 5.55, y: 1.62, w: 7.15, h: 4.55 });

  s.addText("Streamlit application, running locally against the trained model. This is the real interface, not a mockup.",
    { x: M, y: 1.72, w: 4.6, h: 0.75, fontFace: SANS, fontSize: 12.5, color: STEEL,
      lineSpacing: 18, margin: 0 });

  card(s, M, 2.62, 4.6, 2.5, RED_T);
  dot(s, M + 0.3, 2.92, RED, 0.19);
  s.addText("RED", { x: M + 0.62, y: 2.82, w: 3.6, h: 0.34, fontFace: SANS,
    fontSize: 15, bold: true, color: RED, charSpacing: 1.4, margin: 0 });
  s.addText("The reading “dobutamin” resolves to dobutamine at 0.96 confidence. DOPamine is an FDA-listed confusable and both are ISMP high-alert, so the flag escalates automatically and names the tall-man mitigation.",
    { x: M + 0.62, y: 3.2, w: 3.75, h: 1.75, fontFace: SANS, fontSize: 11.5,
      color: "6B2D31", lineSpacing: 16, margin: 0 });

  s.addText("Every alert carries a plain-language reason. An unexplained warning gets dismissed reflexively at a busy counter.",
    { x: M, y: 5.3, w: 4.6, h: 0.8, fontFace: SANS, fontSize: 11.5, color: STEEL,
      italic: true, lineSpacing: 16, margin: 0 });

  foot(s, "app/streamlit_app.py — screenshots captured programmatically via scripts/capture_demo.py.", 8);
}

/* ── 9 · live demo, the other three ──────────────────────────────────── */
{
  const s = pres.addSlide();
  head(s, "07 · Live demo", "Three states, one rule each");

  const shots = [
    ["demo_yellow.png", "YELLOW", AMBER, AMBER_T, "“losec” is an exact match — but Lasix is a documented confusable, surfaced by the pair list. Spelling similarity alone (0.52) would have ranked it out of the top three."],
    ["demo_green.png", "GREEN", GREEN, GREEN_T, "“paracetamol” resolves cleanly with a 0.39 margin and no documented or high-similarity confusable nearby. No interruption."],
    ["demo_unread.png", "UNREAD", AMBER, AMBER_T, "“xyzzyqqq” matches nothing in the formulary at 0.51. Reported as unread rather than as its nearest neighbour — a weak match must never render as safe."],
  ];
  shots.forEach(([f, lab, c, tint, desc], i) => {
    const x = M + i * 4.1;
    s.addImage({ path: path.join(FIG, "demo", f), x, y: 1.78, w: 3.7, h: 2.35 });
    card(s, x, 4.28, 3.7, 1.95, tint);
    dot(s, x + 0.26, 4.53, c, 0.17);
    s.addText(lab, { x: x + 0.54, y: 4.44, w: 2.9, h: 0.32, fontFace: SANS,
      fontSize: 13.5, bold: true, color: c, charSpacing: 1.2, margin: 0 });
    s.addText(desc, { x: x + 0.26, y: 4.84, w: 3.2, h: 1.28, fontFace: SANS,
      fontSize: 10.5, color: INK, lineSpacing: 14, margin: 0 });
  });

  foot(s, "All four demo states verified programmatically: each query asserted against its expected traffic-light before capture.", 9);
}

/* ── 10 · stage 2 metrics + ablation ─────────────────────────────────── */
{
  const s = pres.addSlide();
  head(s, "08 · Stage 2 results", "Two operating points, both reported");

  s.addTable([
    [th("Measure", true), th("Advisory"), th("Block")],
    [tdl("Precision"), td("91.2%"), td("96.8%")],
    [tdl("Recall"), td("97.9%", { bold: true, color: NAVY }), td("18.3%")],
    [tdl("F1"), td("94.4%", { bold: true, color: NAVY }), td("30.8%")],
    [tdl("False-positive rate"), td("15.5%"), td("1.0%", { bold: true, color: NAVY })],
    [tdl("Top-3 retrieval", { bold: true }), td("100%", { bold: true, color: NAVY }), td("—", { color: SLATE })],
  ], { x: M, y: 1.9, w: 5.6, colW: [2.6, 1.5, 1.5], rowH: 0.44, ...TBL, fill: { color: WHITE } });

  s.addText("Block is the conservative stop-and-confirm gate. Advisory catches nearly everything at a higher interruption rate. Top-3 retrieval holds at 100% up to one wrong character in five.",
    { x: M, y: 4.75, w: 5.6, h: 0.95, fontFace: SANS, fontSize: 12, color: STEEL,
      lineSpacing: 17, margin: 0 });

  s.addText("Signal ablation — is the fusion justified?", { x: 6.55, y: 1.9, w: 6.2,
    h: 0.34, fontFace: SANS, fontSize: 14, bold: true, color: INK, margin: 0 });
  s.addTable([
    [th("Configuration", true), th("Top-1"), th("Top-3"), th("Partner")],
    [tdl("Edit distance only"), td("99.2%", { bold: true, color: NAVY }), td("100%"), td("99.7%")],
    [tdl("Phonetic only"), td("83.2%"), td("97.6%"), td("87.5%")],
    [tdl("Character n-gram only"), td("90.4%"), td("98.0%"), td("92.4%")],
    [tdl("Fused 0.45/0.35/0.20", { bold: true }), td("96.8%"), td("100%", { bold: true, color: NAVY }),
     td("97.9%", { bold: true, color: NAVY })],
  ], { x: 6.55, y: 2.34, w: 6.17, colW: [2.75, 1.14, 1.14, 1.14], rowH: 0.44, ...TBL,
       fill: { color: WHITE } });

  card(s, 6.55, 4.6, 6.17, 1.55, ICE);
  s.addText("Reported as measured, including where unflattering.", { x: 6.85, y: 4.8,
    w: 5.6, h: 0.3, fontFace: SANS, fontSize: 12.5, bold: true, color: NAVY, margin: 0 });
  s.addText("Edit distance alone wins on top-1. The phonetic channel is carried because sound-alike confusion is a distinct clinical error mode — celebrex/celexa scores 0.83 on spelling but 0.38 phonetically. The channels disagree, which is why both are needed.",
    { x: 6.85, y: 5.12, w: 5.6, h: 0.95, fontFace: SANS, fontSize: 11, color: STEEL,
      lineSpacing: 15, margin: 0 });

  foot(s, "Scored against 164 ISMP/FDA pairs with 300 no-partner controls, seeded noise at 0.2 errors per character. A true positive requires both a flag and the partner surfaced.", 10);
}

/* ── 11 · calibration ────────────────────────────────────────────────── */
{
  const s = pres.addSlide();
  head(s, "09 · Calibration", "An alert nobody trusts is worse than no alert", { w: 8.6 });

  s.addChart(pres.ChartType.bar, [
    { name: "False alarms on unrelated names", labels: ["0.62 provisional", "0.70", "0.78 adopted"], values: [36.0, 18.5, 15.5] },
    { name: "Recall", labels: ["0.62 provisional", "0.70", "0.78 adopted"], values: [97.9, 97.9, 97.9] },
  ], {
    x: 6.3, y: 1.9, w: 6.4, h: 3.55,
    barDir: "col", chartColors: [RED, NAVY], barGapWidthPct: 55,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0"%"',
    dataLabelFontFace: SANS, dataLabelFontSize: 10, dataLabelColor: INK,
    catAxisLabelFontFace: SANS, catAxisLabelFontSize: 10.5, catAxisLabelColor: INK,
    valAxisLabelFontFace: SANS, valAxisLabelFontSize: 10, valAxisLabelColor: SLATE,
    valAxisMaxVal: 110, valGridLine: { color: "E4E9EF", size: 0.5 },
    catGridLine: { style: "none" }, showLegend: true, legendPos: "b",
    legendFontFace: SANS, legendFontSize: 10.5, showTitle: false,
  });

  s.addText("The initial threshold flagged 36% of unrelated prescriptions. Recalibration cut false alarms by more than half — with no loss of recall.",
    { x: M, y: 1.95, w: 5.3, h: 1.3, fontFace: SERIF, fontSize: 20, bold: true,
      color: INK, lineSpacing: 26, margin: 0 });
  s.addText("Alarm fatigue is a documented failure mode of clinical alerting: when most alerts are noise, staff learn to dismiss all of them — including the true ones.\n\nA system that interrupts one prescription in three does not get used as designed, whatever its recall. Recall is identical across all three thresholds, so the provisional value bought nothing and cost a great deal of trust.",
    { x: M, y: 3.35, w: 5.3, h: 2.4, fontFace: SANS, fontSize: 12.5, color: STEEL,
      lineSpacing: 18, margin: 0 });

  foot(s, "Calibrated against the ground-truth pair set with no-partner controls; adopted operating point recorded in configs/risk.yaml.", 11);
}

/* ── 12 · deployment ─────────────────────────────────────────────────── */
{
  const s = pres.addSlide();
  head(s, "10 · Deployment", "Fits the counter, not the data centre");

  [["3.8 ms", "Per word, measured", "Roughly 80× inside a 300 ms target. Decoding, not inference, is the latency risk."],
   ["34 MB", "Shipped weights", "Optimizer state stripped; approximately 8 MB after INT8 quantisation."],
   ["0", "Network calls", "Runs offline. Prescriptions never leave the premises — a property, not a policy."],
   ["8.4 M", "Parameters", "A single forward pass, viable on a Raspberry Pi 5 class device."]]
    .forEach(([v, l, d], i) => stat(s, M + i * 3.11, 2.0, 2.8, v, l, d));

  const rails = [
    ["Export path is honest about difficulty", "The recogniser exports to ONNX cleanly with a dynamic width axis. A transformer alternative would not — its autoregressive decoder needs three separate graphs and a reimplemented search loop."],
    ["Quantisation must be re-measured, not assumed", "The dangerous failure is not a higher error rate but a confusable substitution. Report the change in risk-flag recall after quantising, not only CER."],
  ];
  rails.forEach(([t, d], i) => {
    const x = M + i * 6.2;
    card(s, x, 4.35, 5.89, 1.85);
    s.addText(t, { x: x + 0.3, y: 4.58, w: 5.3, h: 0.34, fontFace: SANS, fontSize: 13,
      bold: true, color: NAVY, margin: 0 });
    s.addText(d, { x: x + 0.3, y: 4.94, w: 5.3, h: 1.1, fontFace: SANS, fontSize: 11.5,
      color: STEEL, lineSpacing: 16, margin: 0 });
  });

  foot(s, "Latency measured at batch size 1 on Apple M2 (MPS), forward pass only; target-hardware figures remain to be measured. Full analysis in docs/deployment.md.", 12);
}

/* ── 13 · limitations ────────────────────────────────────────────────── */
{
  const s = pres.addSlide();
  head(s, "11 · Limitations", "What these numbers do not establish");

  const left = [
    ["Everything is synthetic.", "The gap between degraded font renders and genuine prescriber handwriting is unmeasured, and is the largest single source of uncertainty here."],
    ["The second model is unresolved.", "The transformer arm did not converge on available compute. No claim of superiority is made in either direction."],
    ["IAM word crops are inferred.", "69.1% of lines were discarded on alignment, and writer-independence could not be verified from the mirror used."],
  ];
  const right = [
    ["The pair list is a curated subset.", "164 pairs from published ISMP and FDA material, not the complete official table. Recall against the full list is untested."],
    ["Thresholds are corpus-specific.", "Calibrated with a synthetic noise model; real OCR error distributions would require recalibration."],
    ["Not a medical device.", "No clinical evaluation, no prospective study of whether alerts change dispensing behaviour. Advisory only."],
  ];
  [left, right].forEach((col, ci) => {
    col.forEach(([t, d], i) => {
      const x = M + ci * 6.2, y = 1.95 + i * 1.42;
      dot(s, x, y + 0.09, ci === 0 ? RED : AMBER, 0.15);
      s.addText(t, { x: x + 0.3, y, w: 5.5, h: 0.32, fontFace: SANS, fontSize: 13.5,
        bold: true, color: INK, margin: 0 });
      s.addText(d, { x: x + 0.3, y: y + 0.34, w: 5.5, h: 0.95, fontFace: SANS,
        fontSize: 11.5, color: STEEL, lineSpacing: 16, margin: 0 });
    });
  });

  foot(s, "Software supporting clinical decisions falls under CDSCO Medical Device Rules 2017 in India. Clinical evaluation, ISO 14971 risk management and IEC 62304 lifecycle process would all precede deployment.", 13);
}

/* ── 14 · close (dark) ───────────────────────────────────────────────── */
{
  const s = pres.addSlide();
  s.background = { color: INK };
  head(s, "12 · Artefacts", "Every number regenerates from one command", { dark: true });

  s.addTable([
    [tdl("Full pipeline", { color: WHITE }), tdl("scripts/run_all.sh", { color: "8FB3DC" })],
    [tdl("Results and limitations", { color: WHITE }), tdl("docs/results.md", { color: "8FB3DC" })],
    [tdl("Deployment analysis", { color: WHITE }), tdl("docs/deployment.md", { color: "8FB3DC" })],
    [tdl("GPU training notebook", { color: WHITE }), tdl("notebooks/kaggle_train.ipynb", { color: "8FB3DC" })],
    [tdl("Demonstration app", { color: WHITE }), tdl("app/streamlit_app.py", { color: "8FB3DC" })],
    [tdl("Regression tests", { color: WHITE, bold: true }), tdl("26 passing", { color: WHITE, bold: true })],
  ], { x: M, y: 2.05, w: 6.0, colW: [3.2, 2.8], rowH: 0.44, fontFace: SANS,
       fontSize: 12.5, fill: { color: "16273C" },
       border: { type: "solid", pt: 0.5, color: "24374F" }, valign: "middle" });

  s.addText("Six defects were found that raised no exception", { x: 7.1, y: 2.05,
    w: 5.6, h: 0.34, fontFace: SANS, fontSize: 14, bold: true, color: WHITE, margin: 0 });
  s.addText("The most consequential silently removed 47.8% of the training set from the loss — the encoder emitted fewer frames than the label required, those samples contributed no gradient, and the model collapsed to predicting nothing. It presented as a hyperparameter problem.\n\nEach is now pinned by a regression test, because a failure that stays quiet is the one that reaches a patient.",
    { x: 7.1, y: 2.5, w: 5.6, h: 2.6, fontFace: SANS, fontSize: 12, color: "AFC2D6",
      lineSpacing: 17, margin: 0 });

  s.addText("github.com/TushKum/lasa-rx-guard", { x: M, y: 5.55, w: 7, h: 0.4,
    fontFace: SANS, fontSize: 15, bold: true, color: "8FB3DC", margin: 0 });
  s.addText("Advisory prototype. Not a validated medical device; not for clinical use.",
    { x: M, y: 5.98, w: 8, h: 0.3, fontFace: SANS, fontSize: 11, color: "7E93AC",
      italic: true, margin: 0 });
  foot(s, "Seeded end-to-end; the 59,000-image corpus regenerates byte-identically.", 14, true);
}

const out = path.join(ROOT, "outputs", "LASA-RxGuard.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("wrote " + out));
