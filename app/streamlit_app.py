"""LASA-RxGuard demo.

    streamlit run app/streamlit_app.py

Works before any model is trained: the Stage-2 risk engine can be driven by
typing a drug name directly, so the LASA half is demonstrable while OCR training
is still running.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lasa.common.paths import CHECKPOINTS, SYNTH  # noqa: E402
from lasa.risk.matcher import LasaMatcher  # noqa: E402

RISK_STYLE = {
    "RED": ("#b3261e", "#fdeceb", "⛔ RED — do not dispense without confirmation"),
    "YELLOW": ("#8a6100", "#fff8e1", "⚠️ YELLOW — check before dispensing"),
    "GREEN": ("#1b5e20", "#eaf5ea", "✅ GREEN — no confusable match found"),
}

st.set_page_config(page_title="LASA-RxGuard", page_icon="💊", layout="wide")


@st.cache_resource
def get_matcher() -> LasaMatcher:
    return LasaMatcher.from_project()


@st.cache_resource
def get_pipeline(backend: str):
    from lasa.infer.pipeline import RxPipeline
    return RxPipeline(backend=backend, matcher=get_matcher())


def risk_banner(risk: str, reasons: list[str]) -> None:
    fg, bg, label = RISK_STYLE[risk]
    st.markdown(
        f"""<div style="background:{bg};border-left:8px solid {fg};padding:14px 18px;
        border-radius:6px;margin:8px 0;">
        <div style="color:{fg};font-size:20px;font-weight:700;">{label}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    for r in reasons:
        st.markdown(f"- {r}")


def candidate_table(assessment) -> None:
    rows = []
    for i, c in enumerate(assessment.candidates, 1):
        flags = []
        if c.is_known_lasa:
            flags.append("ISMP/FDA pair")
        if c.high_alert:
            flags.append("high-alert")
        if c.via == "known-pair":
            flags.append("surfaced via pair list")
        rows.append({
            "#": i,
            "Drug": c.display,
            "Tall-man": c.tall_man or "—",
            "Score": round(c.score, 3),
            "Edit": round(c.edit_sim, 2),
            "Phonetic": round(c.phonetic_sim, 2),
            "N-gram": round(c.ngram_sim, 2),
            "Flags": ", ".join(flags) or "—",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ header --

st.title("💊 LASA-RxGuard")
st.caption("Handwritten prescription OCR + look-alike/sound-alike drug-confusion risk flagging")

st.error(
    "**Prototype — not a medical device.** This is an unvalidated hackathon "
    "project. It has not been clinically evaluated and must not be used for real "
    "dispensing decisions. Output is advisory; a qualified pharmacist makes the "
    "final call.",
    icon="🚨",
)

matcher = get_matcher()
tab_ocr, tab_name, tab_about = st.tabs(
    ["📷 Prescription image", "⌨️ Type a drug name", "ℹ️ About"]
)

# ------------------------------------------------------------- image path --

with tab_ocr:
    crnn_ready = (CHECKPOINTS / "crnn_ctc" / "best.pt").exists()
    trocr_ready = (CHECKPOINTS / "trocr_ft" / "config.json").exists()

    available = [b for b, ok in (("crnn", crnn_ready), ("trocr", trocr_ready)) if ok]
    if not available:
        st.warning(
            "No trained OCR checkpoint found yet, so image reading is disabled.\n\n"
            "Train one with:\n"
            "```\npython -m lasa.train.train_crnn\n```\n"
            "The **Type a drug name** tab works right now and demonstrates the "
            "Stage-2 risk engine without OCR.",
            icon="⏳",
        )
    else:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            backend = st.selectbox("OCR model", available,
                                   format_func=lambda b: {"crnn": "CRNN (CNN+BiLSTM+CTC)",
                                                          "trocr": "TrOCR (fine-tuned)"}[b])
        with col_b:
            k = st.slider("Candidates to show", 3, 8, 3)

        uploaded = st.file_uploader("Prescription image", type=["png", "jpg", "jpeg"])
        sample_paths = sorted((SYNTH / "test_seen").glob("*.png"))[:8] if SYNTH.exists() else []
        use_sample = None
        if sample_paths:
            if st.checkbox("…or use a sample word image from the test set"):
                use_sample = st.selectbox("Sample", sample_paths,
                                          format_func=lambda p: p.name)

        img = None
        if uploaded is not None:
            img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8),
                               cv2.IMREAD_GRAYSCALE)
        elif use_sample is not None:
            img = cv2.imread(str(use_sample), cv2.IMREAD_GRAYSCALE)

        if img is not None:
            st.image(img, caption="Input", use_container_width=True, clamp=True)
            with st.spinner("Reading…"):
                pipe = get_pipeline(backend)
                res = pipe.run(img, k=k)

            if not res.words:
                st.warning("No words detected. Try a tighter crop or better contrast.")
            else:
                st.subheader(f"Overall: {res.worst_risk}")
                cols = st.columns(min(len(res.words), 5))
                for i, w in enumerate(res.words[:5]):
                    x, y, ww, hh = w.box
                    with cols[i]:
                        st.image(img[y:y + hh, x:x + ww], clamp=True,
                                 caption=f"{w.text} ({w.confidence:.2f})")

                for w in res.words:
                    st.markdown("---")
                    st.markdown(f"### OCR: `{w.text}`  ·  confidence {w.confidence:.2f}")
                    risk_banner(w.assessment.risk, w.assessment.reasons)
                    candidate_table(w.assessment)

                t = res.timings_ms
                st.caption(
                    "Latency — " + " · ".join(f"{k}: {v:.0f} ms" for k, v in t.items())
                )

# -------------------------------------------------------------- name path --

with tab_name:
    st.markdown("Drive the Stage-2 engine directly — type what OCR *might* have read.")
    c1, c2 = st.columns([2, 1])
    with c1:
        query = st.text_input("OCR hypothesis", value="dobutamin")
    with c2:
        conf = st.slider("Simulated OCR confidence", 0.0, 1.0, 0.85, 0.05)
    k2 = st.slider("Candidates", 3, 8, 3, key="k2")

    st.caption("Try: dobutamin · hydralzine · celebra · losec · vincristin · xyzzyqqq")

    if query.strip():
        a = matcher.match(query, k=k2, ocr_confidence=conf)
        risk_banner(a.risk, a.reasons)
        st.markdown(f"**Best match:** `{a.best_match}` · confidence {a.confidence:.3f} "
                    f"· margin {a.margin:.3f}")
        candidate_table(a)

with tab_about:
    st.markdown(
        f"""
**Stage 1 — OCR.** A CRNN (CNN + BiLSTM + CTC) trained from scratch, compared
against a fine-tuned `microsoft/trocr-base-handwritten`, on synthetic
handwritten drug names plus IAM handwriting.

**Stage 2 — LASA risk.** The reading is matched against a
{len(matcher.names)}-name drug database by fusing edit distance (0.45),
phonetic encoding (0.35) and character n-gram TF-IDF cosine (0.20).
{len(matcher.pairs)} documented ISMP/FDA confused-name pairs are used as ground
truth, and documented partners of the top match are always surfaced even when
similarity alone would have missed them.

**Traffic light.** RED = the reading cannot separate a documented or
highly-similar pair (escalated automatically for ISMP high-alert drugs).
YELLOW = a plausible alternative exists, or confidence is low. GREEN = one clear
match. A weak best match never reads as GREEN — "I don't know" is reported as
YELLOW, not as safe.

Tall-man lettering (DOBUTamine vs DOPamine) is shown where published, because
that capitalisation *is* the recommended mitigation.
"""
    )
