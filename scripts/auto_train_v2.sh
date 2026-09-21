#!/bin/bash
# Autonomous v2 gate -> sweep -> training trigger. Launched once; polls the
# generation log, and when the corpus pipeline exits cleanly with
# DATA GATE: PASS: (1) short hyperparameter screening on MPS, (2) full
# 500-epoch training with the winning config + early stopping. No
# confirmation needed. Owner commits results in the morning.
#
# Usage: caffeinate -i nohup bash scripts/auto_train_v2.sh > artifacts/corpus_v2/autotrain.log 2>&1 &
set -u
LOG="artifacts/corpus_v2/generation.log"
OUT="artifacts/forecaster_v2"
SWEEP="artifacts/sweep_v2"

echo "[autotrain] watching ${LOG} (poll 120s)"
while true; do
  if grep -q "PIPELINE_EXIT=0" "${LOG}" 2>/dev/null; then
    echo "[autotrain] pipeline exit 0 detected"
    break
  fi
  if grep -q "PIPELINE_EXIT=" "${LOG}" 2>/dev/null; then
    echo "[autotrain] pipeline exited NONZERO -- aborting, needs human. tail:"
    tail -n 20 "${LOG}"
    exit 2
  fi
  sleep 120
done

if ! grep -q "DATA GATE: PASS" "${LOG}" 2>/dev/null; then
  echo "[autotrain] gate did not pass -- aborting, needs human. tail:"
  tail -n 20 "${LOG}"
  exit 3
fi

echo "[autotrain] gate PASS -- hyperparameter screening (3 configs x 2 epochs, MPS)"
export PYTHONPATH="${PYTHONPATH:-.}"
CACHE_DIR="artifacts/corpus_v2/.window_cache"
python3 -u scripts/sweep_forecaster.py artifacts/corpus_v2 "${SWEEP}" \
  --epochs 2 --device mps --window-cache "${CACHE_DIR}" \
  --configs '[{"width": 64, "lr": 0.001, "weight_decay": 0.0001, "dropout": 0.15}, {"width": 64, "lr": 0.002, "weight_decay": 0.0001, "dropout": 0.1}, {"width": 32, "lr": 0.001, "weight_decay": 0.0003, "dropout": 0.2}]' \
  > "${SWEEP}.log" 2>&1
echo "SWEEP_EXIT:$?" >> "${SWEEP}.log"

if [ ! -f "${SWEEP}/sweep_report.json" ]; then
  echo "[autotrain] sweep produced no report -- aborting, needs human. tail:"
  tail -n 20 "${SWEEP}.log"
  exit 4
fi

WINNER_CFG=$(python3 -c "
import json
rep = json.load(open('${SWEEP}/sweep_report.json'))
w = next(r for r in rep['ranking'] if r['name'] == rep['winner'])
c = w['config']
print(f\"--width {c['width']} --lr {c['lr']} --weight-decay {c['weight_decay']} --dropout {c['dropout']}\")
")
echo "[autotrain] sweep winner flags: ${WINNER_CFG}"

mkdir -p "${OUT}"
echo "[autotrain] starting full 500-epoch joint GNN-Transformer training on MPS"
# shellcheck disable=SC2086
CACHE_DIR="${CACHE_DIR:-artifacts/corpus_v2/.window_cache}"
python3 -u -m src.learning.train_forecaster artifacts/corpus_v2 "${OUT}" \
  --epochs 500 --batch-size 2 --device mps \
  ${WINNER_CFG} \
  --heads 4 --layers 2 --scheduler cosine \
  --patience 30 --min-delta 1e-4 --grad-clip 1.0 \
  --window-cache "${CACHE_DIR}" \
  > "${OUT}/training.log" 2>&1
echo "TRAIN_EXIT:$?" >> "${OUT}/training.log"
echo "[autotrain] done"
