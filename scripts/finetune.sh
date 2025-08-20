#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/path/to/AutoFineTuner/AutoHPO"
SAVE_DIR="$PROJECT_ROOT/outputs"
TARGET="main.py"
REFACTORED="refactored.py"
CONDA_ENV="AItxt"

python -m AutoFineTuner.engine.engine refactor \
  --proj-path "$PROJECT_ROOT" \
  --save-dir "$SAVE_DIR" \
  --target-name "$TARGET" \
  --refactored-name "$REFACTORED" \
  --conda-env "$CONDA_ENV"
