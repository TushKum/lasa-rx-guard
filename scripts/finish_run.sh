#!/usr/bin/env bash
# Wait for the in-flight CRNN run, then fine-tune TrOCR, then produce both
# reports. Sized for local Apple Silicon (MPS), NOT for the headline result:
# TrOCR-small on a 3k subset is what this machine can do in reasonable time.
# The Colab notebook runs trocr-base on 20k for 3 epochs.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "waiting for CRNN training to finish..."
while pgrep -f "lasa.train.train_crnn" > /dev/null; do sleep 30; done
echo "CRNN done:"
grep -E "^done|best val CER" outputs/reports/train_crnn.log | tail -2

echo
echo "=========== fine-tuning TrOCR (small, local budget) ==========="
$PY -m lasa.train.train_trocr \
    --model microsoft/trocr-small-handwritten \
    --epochs 2 --batch-size 8 --grad-accum 2 --lr 5e-5 \
    --max-train 3000 --max-eval 300 --num-beams 4 --workers 0 \
    --max-minutes 75 --out outputs/checkpoints/trocr_ft \
    > outputs/reports/train_trocr.log 2>&1
tail -4 outputs/reports/train_trocr.log

echo
echo "=========== Stage-1 comparison ==========="
$PY -m lasa.evaluation.evaluate --limit 600 > outputs/reports/eval_stage1.log 2>&1
tail -30 outputs/reports/eval_stage1.log

echo
echo "=========== Stage-2 LASA evaluation ==========="
$PY -m lasa.risk.evaluate_lasa --n-retrieval 400 --n-negatives 300 --markdown \
    > outputs/reports/eval_stage2.log 2>&1
tail -35 outputs/reports/eval_stage2.log

echo
echo "ALL DONE"
