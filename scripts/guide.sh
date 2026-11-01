#!/usr/bin/env bash
# Runs the project workflow. Used by the VS Code tasks.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
export HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
CONFIGS=(distilroberta_utterance distilroberta_context3 distilroberta_context3_speakers)
SEEDS=(42 7 123)

say() { printf '\n\033[1;36m%s\033[0m\n' "$*"; }

case "${1:-}" in
  check)
    say "Running tests"
    "$PYTHON" -m pytest
    ;;
  practice)
    say "Smoke test: 320 lines, 1 round"
    "$PYTHON" -m meld_emotion.train --config configs/distilroberta_context3_speakers.json --limit 320 --epochs 1
    ;;
  train)
    mkdir -p reports/logs
    exec > >(tee -a "reports/logs/train_$(date +%Y%m%d_%H%M%S).log") 2>&1
    say "Training ${#CONFIGS[@]} designs across ${#SEEDS[@]} seeds"
    command -v caffeinate >/dev/null && caffeinate -d -i -m -s -w $$ &
    total=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} )); count=0
    # Every design gets a seed before any design gets the next one, so an early stop still compares all three.
    for seed in "${SEEDS[@]}"; do
      for config in "${CONFIGS[@]}"; do
        count=$((count + 1))
        say "Run $count of $total: $config, seed $seed"
        "$PYTHON" -m meld_emotion.train --config "configs/$config.json" --seed "$seed" --device cpu
      done
    done
    ;;
  train-upgrade)
    mkdir -p reports/logs
    exec > >(tee -a "reports/logs/train_upgrade_$(date +%Y%m%d_%H%M%S).log") 2>&1
    say "Training roberta_context3 across ${#SEEDS[@]} seeds"
    command -v caffeinate >/dev/null && caffeinate -d -i -m -s -w $$ &
    for seed in "${SEEDS[@]}"; do
      say "roberta_context3, seed $seed"
      "$PYTHON" -m meld_emotion.train --config configs/roberta_context3.json --seed "$seed" --device cpu
    done
    ;;
  compare)
    say "Comparing validation results"
    "$PYTHON" -m meld_emotion.compare
    ;;
  final)
    # The test split is read once.
    if [ -f reports/test_ledger.json ]; then
      echo "The final test already ran. See reports/test_ledger.json." >&2
      exit 1
    fi
    mkdir -p reports/logs
    exec > >(tee -a "reports/logs/final_test_$(date +%Y%m%d_%H%M%S).log") 2>&1
    say "Final test: roberta_context3"
    "$PYTHON" -m meld_emotion.evaluate --experiment artifacts/roberta_context3 --split test --final --device cpu
    ;;
  *)
    echo "Usage: scripts/guide.sh {check|practice|train|train-upgrade|compare|final}" >&2
    exit 2
    ;;
esac
