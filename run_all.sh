#!/usr/bin/env bash
# N2 全流水线：校验 → EDA → 基线 → 深度 → 误差分析 → 汇总图 → v1.1 深化（滚动/集成/迁移）→ v1.2 深化（分数选模/注意力/校准）
set -e
cd "$(dirname "$0")"
PY="${PY:-python}"

"$PY" scripts/00_verify_data.py
"$PY" scripts/01_eda.py
"$PY" scripts/02_baselines.py
"$PY" scripts/03_train_lstm.py
"$PY" scripts/04_error_analysis.py
"$PY" scripts/05_make_figures.py
"$PY" scripts/06_rolling.py
"$PY" scripts/07_ensemble.py
"$PY" scripts/08_cross_transfer.py
"$PY" scripts/09_score_aware.py
"$PY" scripts/10_attention.py
"$PY" scripts/11_calibration.py
"$PY" -m pytest tests -q
echo "ALL DONE"
