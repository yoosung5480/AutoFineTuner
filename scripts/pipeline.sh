#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/path/to/AutoFineTuner/AutoHPO"
SAVE_DIR="$PROJECT_ROOT/outputs"
TARGET="main.py"
REFACTORED="refactored.py"
CONDA_ENV="AItxt"
USER_PROMPT="이틀 내에 작업을 완료하고싶어. 데이터는 10~20만 문장."

python -m AutoFineTuner.engine.engine pipeline \
  --proj-path "$PROJECT_ROOT" \
  --save-dir "$SAVE_DIR" \
  --target-name "$TARGET" \
  --refactored-name "$REFACTORED" \
  --conda-env "$CONDA_ENV" \
  --user-prompt "$USER_PROMPT"
