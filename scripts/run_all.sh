#!/usr/bin/env bash
# Reproduce the whole project end to end.
#
#   ./scripts/run_all.sh            full run (hours; needs a GPU for TrOCR)
#   ./scripts/run_all.sh --smoke    tiny sizes, minutes, proves the wiring
#
# Every stage is seeded. Re-running with the same flags reproduces the same
# artefacts on the same hardware.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3

SMOKE=0
[ "${1:-}" = "--smoke" ] && SMOKE=1

if [ "$SMOKE" = "1" ]; then
  N_TRAIN=1200; N_VAL=200; N_TEST=200
  CRNN_EPOCHS=2; TROCR_EPOCHS=1; TROCR_MAX_TRAIN=120; TROCR_MODEL=microsoft/trocr-small-handwritten
  EVAL_LIMIT="--limit 100"; LASA_N=60
  echo ">>> SMOKE MODE — sizes are tiny; the resulting metrics are NOT results."
else
  N_TRAIN=48000; N_VAL=3000; N_TEST=2000
  CRNN_EPOCHS=15; TROCR_EPOCHS=3; TROCR_MAX_TRAIN=20000; TROCR_MODEL=microsoft/trocr-base-handwritten
  EVAL_LIMIT=""; LASA_N=400
fi

step() { echo; echo "=========== $* ==========="; }

step "1/7 drug corpus + LASA pair ground truth"
$PY -m lasa.data.drug_names --size 2000

step "2/7 handwriting fonts"
$PY -m lasa.data.fonts

step "3/7 synthetic dataset"
$PY -m lasa.data.synth --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" \
    --tier medium --seed 1337
$PY -m lasa.data.synth --preview

step "4/7 train CRNN"
$PY -m lasa.train.train_crnn --epochs "$CRNN_EPOCHS" --batch-size 128 \
    --preload --out outputs/checkpoints/crnn_ctc

step "5/7 fine-tune TrOCR"
$PY -m lasa.train.train_trocr --model "$TROCR_MODEL" --epochs "$TROCR_EPOCHS" \
    --max-train "$TROCR_MAX_TRAIN" --out outputs/checkpoints/trocr_ft

step "6/7 Stage-1 comparison"
# shellcheck disable=SC2086
$PY -m lasa.evaluation.evaluate $EVAL_LIMIT

step "7/7 Stage-2 LASA evaluation"
$PY -m lasa.risk.evaluate_lasa --n-retrieval "$LASA_N" --markdown

echo
echo "Done. Reports in outputs/reports/:"
ls -1 outputs/reports/
echo
echo "Demo:  streamlit run app/streamlit_app.py"
