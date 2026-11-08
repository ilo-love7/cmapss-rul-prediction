#!/usr/bin/env bash
# N2 全流水线：校验 → EDA → 基线 → 深度 → 误差分析 → 汇总图
set -e
PY="${PY:-python}"

"$PY" scripts/00_verify_data.py
"$PY" scripts/01_eda.py
"$PY" scripts/02_baselines.py
"$PY" scripts/03_train_lstm.py
"$PY" scripts/04_error_analysis.py
"$PY" scripts/05_make_figures.py
"$PY" -m pytest tests -q
echo "ALL DONE"
