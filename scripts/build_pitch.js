/**
 * Build the LASA-RxGuard PITCH deck — 10 slides.
 *
 *   node scripts/build_pitch.js
 *
 * Distinct from build_deck.js (14-slide technical review). This one leads with
 * the problem, shows the product working, and closes on an ask. Every number is
 * measured; see docs/results.md.
 */
const pptxgen = require("pptxgenjs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const DEMO = path.join(ROOT, "outputs", "figures", "demo");

const INK = "0A1A2F", NAVY = "12315B", STEEL = "4A6285";
const ICE = "E8EDF3", MIST = "F4F6F9", WHITE = "FFFFFF", SLATE = "5A6B80";
const RED = "A8232B", AMBER = "8A5D00", GREEN = "1B6047";
const RED_T = "F7E9E9", AMBER_T = "FAF2E0", GREEN_T = "E6F0EA";
const SERIF = "Cambria", SANS = "Calibri";
const W = 13.333, H = 7.5, M = 0.62;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "LASA-RxGuard — pitch";

const dot = (s, x, y, c, d = 0.18) => s.addShape(pres.ShapeType.ellipse,
  { x, y, w: d, h: d, fill: { color: c }, line: { width: 0 } });

function card(s, x, y, w, h, fill = MIST) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.04, fill: { color: fill }, line: { width: 0 },
    shadow: { type: "outer", angle: 90, blur: 8, offset: 1, color: "9AA9BC", opacity: 0.2 },
  });
}
function head(s, kicker, title, o = {}) {
  s.addText(kicker.toUpperCase(), { x: M, y: 0.42, w: 9, h: 0.26, fontFace: SANS,
    fontSize: 11, bold: true, color: o.dark ? "8FB3DC" : NAVY, charSpacing: 2.2, margin: 0 });
  s.addText(title, { x: M, y: 0.74, w: o.w || 11.4, h: 0.9, fontFace: SERIF,
    fontSize: 34, bold: true, color: o.dark ? WHITE : INK, margin: 0 });
}
function foot(s, t, p, dark) {
  s.addText(t, { x: M, y: H - 0.6, w: 10.5, h: 0.36, fontFace: SANS, fontSize: 9,
    color: dark ? "7E93AC" : SLATE, margin: 0 });
  s.addText(String(p), { x: W - M - 0.5, y: H - 0.6, w: 0.5, h: 0.3, fontFace: SANS,
    fontSize: 9, color: dark ? "7E93AC" : SLATE, align: "right", margin: 0 });
}
function stat(s, x, y, w, v, l, d, c = NAVY) {
  s.addText(v, { x, y, w, h: 0.78, fontFace: SERIF, fontSize: 42, bold: true, color: c, margin: 0 });
  s.addText(l.toUpperCase(), { x, y: y + 0.78, w, h: 0.26, fontFace: SANS, fontSize: 10,
    bold: true, color: SLATE, charSpacing: 1.6, margin: 0 });
  if (d) s.addText(d, { x, y: y + 1.08, w, h: 0.9, fontFace: SANS, fontSize: 11.5,
    color: STEEL, lineSpacing: 16, margin: 0 });
}

/* 1 — hook (dark) */
{
  const s = pres.addSlide();
  s.background = { color: INK };
  s.addText("LASA-RxGuard", { x: M, y: 0.5, w: 8, h: 0.3, fontFace: SANS, fontSize: 11,
    bold: true, color: "8FB3DC", charSpacing: 3, margin: 0 });
  s.addText("Two drugs.\nOne unreadable word.", { x: M, y: 1.9, w: 8.4, h: 2.2,
    fontFace: SERIF, fontSize: 52, bold: true, color: WHITE, lineSpacing: 56, margin: 0 });
  s.addText("A pharmacist has seconds to decide which medicine a handwritten prescription says. Some names are almost identical — and getting it wrong hurts people.",
    { x: M, y: 4.25, w: 7.2, h: 1, fontFace: SANS, fontSize: 15, color: "AFC2D6",
      lineSpacing: 22, margin: 0 });
  [["RED", RED], ["YELLOW", AMBER], ["GREEN", GREEN]].forEach(([l, c], i) => {
    const y = 2.05 + i * 0.66;
    dot(s, 10.3, y + 0.06, c, 0.22);
    s.addText(l, { x: 10.7, y, w: 2.3, h: 0.34, fontFace: SANS, fontSize: 13.5,
      bold: true, color: WHITE, margin: 0 });
  });
  s.addText("We build the second opinion.", { x: M, y: 5.55, w: 9, h: 0.45,
    fontFace: SERIF, fontSize: 21, bold: true, color: "8FB3DC", margin: 0 });
  foot(s, "ML Bubble 2026 · AIT Pune · TE-BE track · Advisory prototype, not a medical device.", 1, true);
  s.addNotes("Open on the pair: DOPamine and DOBUTamine. Both high-alert. Completely different drugs.");
}

/* 2 — problem */
{
  const s = pres.addSlide();
  head(s, "The problem", "Names that look alike get dispensed alike");
  const pairs = [
    ["DOPamine", "DOBUTamine", "Both high-alert. A cardiovascular dosing error, not a typo.", RED, RED_T],
    ["Celebrex", "Celexa", "An arthritis drug and an antidepressant.", AMBER, AMBER_T],
    ["Losec", "Lasix", "Spelling similarity 0.52. Sound similarity 0.81.", AMBER, AMBER_T],
  ];
  pairs.forEach(([a, b, why, c, tint], i) => {
    const x = M + i * 4.1;
    card(s, x, 1.9, 3.7, 2.35, tint);
    s.addText(a, { x: x + 0.3, y: 2.15, w: 3.1, h: 0.4, fontFace: SERIF, fontSize: 21,
      bold: true, color: c, margin: 0 });
    s.addText("vs", { x: x + 0.3, y: 2.58, w: 3.1, h: 0.26, fontFace: SANS, fontSize: 11,
      color: c, margin: 0 });
    s.addText(b, { x: x + 0.3, y: 2.84, w: 3.1, h: 0.4, fontFace: SERIF, fontSize: 21,
      bold: true, color: c, margin: 0 });
    s.addText(why, { x: x + 0.3, y: 3.35, w: 3.1, h: 0.8, fontFace: SANS, fontSize: 11.5,
      color: INK, lineSpacing: 16, margin: 0 });
  });
  s.addText("Regulators publish these pairs precisely because people keep confusing them. Handwriting removes the last cue that separates them — and the pharmacist is the final check before the patient.",
    { x: M, y: 4.6, w: 7.6, h: 1, fontFace: SANS, fontSize: 14.5, color: INK,
      lineSpacing: 22, margin: 0 });
  card(s, 8.5, 4.6, 4.2, 1.4, ICE);
  s.addText("164", { x: 8.85, y: 4.78, w: 3.5, h: 0.6, fontFace: SERIF, fontSize: 34,
    bold: true, color: NAVY, margin: 0 });
  s.addText("documented dangerous pairs used as our ground truth", { x: 8.85, y: 5.36,
    w: 3.5, h: 0.5, fontFace: SANS, fontSize: 11.5, color: STEEL, lineSpacing: 15, margin: 0 });
  foot(s, "Source: ISMP List of Confused Drug Names; FDA/ISMP tall-man lettering list.", 2);
}

/* 3 — solution */
{
  const s = pres.addSlide();
  head(s, "The solution", "Read it, then question it");
  [["01", "Read the handwriting", "A neural network trained from scratch turns the written word into letters. 8.4M parameters, small enough to run on a counter device."],
   ["02", "Check every look-alike", "The reading is compared to 2,000 real medicine names three ways: how it is spelled, how it sounds, and its letter patterns."],
   ["03", "Show a traffic light", "Top-3 candidates and a RED / YELLOW / GREEN flag — always with a plain-language reason, never a bare score."]]
    .forEach(([n, t, d], i) => {
      const x = M + i * 4.1;
      card(s, x, 1.95, 3.7, 2.6);
      s.addText(n, { x: x + 0.3, y: 2.18, w: 1, h: 0.4, fontFace: SERIF, fontSize: 20,
        bold: true, color: "AEBECE", margin: 0 });
      s.addText(t, { x: x + 0.3, y: 2.62, w: 3.1, h: 0.4, fontFace: SANS, fontSize: 15,
        bold: true, color: NAVY, margin: 0 });
      s.addText(d, { x: x + 0.3, y: 3.06, w: 3.1, h: 1.35, fontFace: SANS, fontSize: 11.5,
        color: STEEL, lineSpacing: 16, margin: 0 });
    });
  card(s, M, 4.9, 12.09, 1.3, ICE);
  s.addText("The two halves are independent.", { x: M + 0.35, y: 5.08, w: 11.4, h: 0.32,
    fontFace: SANS, fontSize: 14.5, bold: true, color: INK, margin: 0 });
  s.addText("If the camera fails or the model will not load, a pharmacist can still type what they think it says and get the same safety check. It degrades gracefully instead of going dark.",
    { x: M + 0.35, y: 5.42, w: 11.3, h: 0.6, fontFace: SANS, fontSize: 12.5, color: STEEL,
      lineSpacing: 17, margin: 0 });
  foot(s, "Stage 2 has no dependency on the deep-learning stack — by design.", 3);
}

/* 4 — the product, RED */
{
  const s = pres.addSlide();
  head(s, "The product", "It runs today", { w: 7.4 });
  s.addImage({ path: path.join(DEMO, "demo_red.png"), x: 5.5, y: 1.6, w: 7.2, h: 4.6 });
  s.addText("A working web app, not a prototype sketch. Upload a prescription photo, or type what the OCR might have read.",
    { x: M, y: 1.72, w: 4.6, h: 0.8, fontFace: SANS, fontSize: 13, color: STEEL,
      lineSpacing: 18, margin: 0 });
  card(s, M, 2.65, 4.6, 2.75, RED_T);
  dot(s, M + 0.3, 2.95, RED, 0.2);
  s.addText("RED — do not dispense", { x: M + 0.64, y: 2.86, w: 3.8, h: 0.34,
    fontFace: SANS, fontSize: 14, bold: true, color: RED, margin: 0 });
  s.addText("“dobutamin” resolves to dobutamine at 0.96 confidence. DOPamine is an FDA-listed confusable and both are high-alert — so the warning escalates automatically and names the recommended tall-man spelling.",
    { x: M + 0.3, y: 3.3, w: 4.05, h: 1.95, fontFace: SANS, fontSize: 11.5,
      color: "6B2D31", lineSpacing: 16, margin: 0 });
  s.addText("Every alert explains itself. An unexplained warning gets clicked past.",
    { x: M, y: 5.6, w: 4.6, h: 0.6, fontFace: SANS, fontSize: 11.5, color: STEEL,
      italic: true, lineSpacing: 16, margin: 0 });
  foot(s, "streamlit run app/streamlit_app.py — screenshots captured programmatically from the live app.", 4);
}

/* 5 — the other three states */
{
  const s = pres.addSlide();
  head(s, "The product", "Three answers, and one it refuses to guess");
  [["demo_yellow.png", "YELLOW", AMBER, AMBER_T, "“losec” is exact — but Lasix is a documented twin, surfaced from the pair list even though spelling alone ranks it far down."],
   ["demo_green.png", "GREEN", GREEN, GREEN_T, "“paracetamol” is clear. Nothing else is close. No interruption, because false alarms are what kill trust."],
   ["demo_unread.png", "UNREAD", AMBER, AMBER_T, "“xyzzyqqq” matches nothing. It says so, rather than showing its nearest guess. Confusion never renders as green."]]
    .forEach(([f, lab, c, tint, d], i) => {
      const x = M + i * 4.1;
      s.addImage({ path: path.join(DEMO, f), x, y: 1.78, w: 3.7, h: 2.35 });
      card(s, x, 4.28, 3.7, 1.95, tint);
      dot(s, x + 0.26, 4.53, c, 0.17);
      s.addText(lab, { x: x + 0.54, y: 4.44, w: 2.9, h: 0.32, fontFace: SANS,
        fontSize: 13.5, bold: true, color: c, charSpacing: 1.2, margin: 0 });
      s.addText(d, { x: x + 0.26, y: 4.84, w: 3.2, h: 1.28, fontFace: SANS,
        fontSize: 10.5, color: INK, lineSpacing: 14, margin: 0 });
    });
  foot(s, "All four states verified programmatically — each query asserted against its expected flag before capture.", 5);
}

/* 6 — does it work */
{
  const s = pres.addSlide();
  head(s, "Does it work?", "Measured, not claimed");
  [["97%", "Letters read correctly", "On handwriting it has seen before."],
   ["98.75%", "Whole names correct", "When choosing from a known pharmacy stock list."],
   ["97.9%", "Dangerous pairs caught", "Of 164 documented ISMP / FDA pairs."],
   ["15.5%", "False alarms", "Down from 36% after retuning, with no loss of recall.", AMBER]]
    .forEach(([v, l, d, c], i) => stat(s, M + i * 3.11, 1.95, 2.85, v, l, d, c || NAVY));

  card(s, M, 4.35, 12.09, 1.85, ICE);
  s.addText("The number we are proudest of is the one that went down.", { x: M + 0.35,
    y: 4.58, w: 11.4, h: 0.34, fontFace: SANS, fontSize: 14.5, bold: true, color: NAVY, margin: 0 });
  s.addText("Our first version flagged one prescription in three. That is how safety systems fail in the real world — when most alerts are noise, staff learn to dismiss all of them, including the true ones. Retuning cut false alarms by more than half while catching exactly as many dangerous pairs.",
    { x: M + 0.35, y: 4.95, w: 11.3, h: 1.1, fontFace: SANS, fontSize: 12.5, color: STEEL,
      lineSpacing: 18, margin: 0 });
  foot(s, "Character error rate 2.91% greedy / 0.48% lexicon-constrained. Full tables in docs/results.md.", 6);
}

/* 7 — what we learned */
{
  const s = pres.addSlide();
  head(s, "What we learned", "The bottleneck was not what we expected", { w: 8.6 });

  s.addChart(pres.ChartType.bar, [{
    name: "Error rate", labels: ["Familiar words", "New words", "New handwriting", "Both new"],
    values: [2.91, 2.83, 19.45, 22.71],
  }], {
    x: 6.3, y: 1.9, w: 6.4, h: 3.9, barDir: "bar", chartColors: [NAVY, NAVY, RED, RED],
    varyColors: true, barGapWidthPct: 45, showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: '0.00"%"', dataLabelFontFace: SANS, dataLabelFontSize: 11.5,
    dataLabelColor: INK, catAxisLabelFontFace: SANS, catAxisLabelFontSize: 12,
    catAxisLabelColor: INK, valAxisLabelFontFace: SANS, valAxisLabelFontSize: 10,
    valAxisLabelColor: SLATE, valAxisMaxVal: 27, valGridLine: { color: "E4E9EF", size: 0.5 },
    catGridLine: { style: "none" }, showLegend: false, showTitle: false,
  });
  s.addText("Medicine names it had never seen cost us nothing. Handwriting it had never seen cost us seven times more.",
    { x: M, y: 1.98, w: 5.3, h: 1.35, fontFace: SERIF, fontSize: 20, bold: true, color: INK,
      lineSpacing: 26, margin: 0 });
  s.addText("That tells us the model genuinely learned to read letters rather than memorise a word list — and it tells us exactly where to spend the next month.\n\nWe do not need more medicine names. We need more people's handwriting.",
    { x: M, y: 3.5, w: 5.3, h: 1.8, fontFace: SANS, fontSize: 12.5, color: STEEL,
      lineSpacing: 18, margin: 0 });
  card(s, M, 5.4, 5.3, 0.85, ICE);
  s.addText("Collect writers, not words.", { x: M + 0.3, y: 5.6, w: 4.8, h: 0.45,
    fontFace: SANS, fontSize: 14, bold: true, color: NAVY, margin: 0 });
  foot(s, "Character error rate by test split, greedy decoding, n = 400 per split.", 7);
}

/* 8 — reproducible: github + notebook */
{
  const s = pres.addSlide();
  head(s, "Everything is open", "Clone it, run it, check us");

  s.addImage({ path: path.join(DEMO, "notebook.png"), x: 7.05, y: 1.72, w: 5.65, h: 4.3 });

  const items = [
    ["Code on GitHub", "github.com/TushKum/lasa-rx-guard", "Every script, config and test. 26 regression tests, each pinning a bug that raised no exception."],
    ["One-click training", "notebooks/colab_train.ipynb", "Opens in Google Colab or Kaggle on a free GPU. Builds the data, trains, evaluates — top to bottom."],
    ["The whole pipeline", "scripts/run_all.sh", "Seeded end to end. The 59,000-image dataset regenerates byte-for-byte identically."],
  ];
  items.forEach(([t, code, d], i) => {
    const y = 1.85 + i * 1.5;
    dot(s, M, y + 0.1, NAVY, 0.16);
    s.addText(t, { x: M + 0.32, y, w: 5.9, h: 0.32, fontFace: SANS, fontSize: 14.5,
      bold: true, color: INK, margin: 0 });
    s.addText(code, { x: M + 0.32, y: y + 0.34, w: 5.9, h: 0.28, fontFace: "Courier New",
      fontSize: 11.5, color: NAVY, margin: 0 });
    s.addText(d, { x: M + 0.32, y: y + 0.64, w: 5.9, h: 0.75, fontFace: SANS,
      fontSize: 11.5, color: STEEL, lineSpacing: 16, margin: 0 });
  });
  foot(s, "Fonts SHA-256 pinned; drug corpus and ground-truth pair list versioned in the repository.", 8);
}

/* 9 — honest limits */
{
  const s = pres.addSlide();
  head(s, "What we are not claiming", "The honest slide");
  [["All our handwriting is computer-generated.", "Real doctors write worse. Our numbers would drop on real prescriptions, and we have not measured by how much. This is our biggest weakness.", RED],
   ["Our second model did not finish.", "A larger transformer ran out of compute. We report it as a failed run, not as “our model won”.", AMBER],
   ["The pair list is a subset.", "164 pairs from published safety lists, not the complete official table.", AMBER],
   ["It has never been in a pharmacy.", "No clinical testing. It advises; a pharmacist decides. Always.", RED]]
    .forEach(([t, d, c], i) => {
      const x = M + (i % 2) * 6.2, y = 1.95 + Math.floor(i / 2) * 2.05;
      card(s, x, y, 5.89, 1.75);
      dot(s, x + 0.3, y + 0.3, c, 0.16);
      s.addText(t, { x: x + 0.62, y: y + 0.2, w: 5.0, h: 0.34, fontFace: SANS,
        fontSize: 13.5, bold: true, color: INK, margin: 0 });
      s.addText(d, { x: x + 0.62, y: y + 0.58, w: 5.0, h: 1.0, fontFace: SANS,
        fontSize: 11.5, color: STEEL, lineSpacing: 16, margin: 0 });
    });
  s.addText("We would rather be trusted than impressive.", { x: M, y: 6.05, w: 9, h: 0.4,
    fontFace: SERIF, fontSize: 16, bold: true, color: NAVY, italic: true, margin: 0 });
  foot(s, "Software supporting clinical decisions falls under CDSCO Medical Device Rules 2017 in India.", 9);
}

/* 10 — close (dark) */
{
  const s = pres.addSlide();
  s.background = { color: INK };
  s.addText("What happens next", { x: M, y: 0.42, w: 9, h: 0.3, fontFace: SANS,
    fontSize: 11, bold: true, color: "8FB3DC", charSpacing: 2.2, margin: 0 });
  s.addText("A second pair of eyes,\nat every counter.", { x: M, y: 1.5, w: 8.4, h: 1.8,
    fontFace: SERIF, fontSize: 42, bold: true, color: WHITE, lineSpacing: 48, margin: 0 });

  [["Next 30 days", "Finish the second model on a free GPU, and gather real prescription handwriting from a local pharmacy."],
   ["The device", "34 MB today, ~8 MB compressed. Runs offline on a ₹5,000 board — no cloud, no patient data leaving the shop."],
   ["The ask", "Access to real, anonymised handwritten prescriptions. That single input is worth more to us than any model change."]]
    .forEach(([t, d], i) => {
      const x = M + i * 4.1;
      s.addShape(pres.ShapeType.roundRect, { x, y: 3.55, w: 3.7, h: 1.95, rectRadius: 0.04,
        fill: { color: "16273C" }, line: { width: 0 } });
      s.addText(t, { x: x + 0.3, y: 3.78, w: 3.1, h: 0.32, fontFace: SANS, fontSize: 13.5,
        bold: true, color: "8FB3DC", margin: 0 });
      s.addText(d, { x: x + 0.3, y: 4.14, w: 3.1, h: 1.2, fontFace: SANS, fontSize: 11.5,
        color: "AFC2D6", lineSpacing: 16, margin: 0 });
    });

  s.addText("github.com/TushKum/lasa-rx-guard", { x: M, y: 5.85, w: 8, h: 0.4,
    fontFace: SANS, fontSize: 15, bold: true, color: WHITE, margin: 0 });
  s.addText("Advisory prototype. Not a validated medical device; not for clinical use.",
    { x: M, y: 6.28, w: 8.5, h: 0.3, fontFace: SANS, fontSize: 10.5, color: "7E93AC",
      italic: true, margin: 0 });
  foot(s, "ML Bubble 2026 · Army Institute of Technology, Pune", 10, true);
}

const out = path.join(ROOT, "outputs", "LASA-RxGuard-Pitch.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("wrote " + out));
