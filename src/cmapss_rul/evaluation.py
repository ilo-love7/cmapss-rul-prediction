"""评估：RMSE 与 PHM08 scoring function（晚预测惩罚更重）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray,
                early_penalty: float = 13.0, late_penalty: float = 10.0) -> float:
    """PHM08 竞赛评分：早预测 exp(-d/13)-1，晚预测 exp(d/10)-1，d = pred - true。

    晚预测（d>0，高估 RUL，更危险）惩罚更重。
    """
    d = np.asarray(y_pred, dtype=np.float64).reshape(-1) - np.asarray(y_true, dtype=np.float64).reshape(-1)
    s = np.where(d < 0, np.exp(-d / early_penalty) - 1.0, np.exp(d / late_penalty) - 1.0)
    return float(np.sum(s))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    d = y_pred - y_true
    return {
        "rmse": rmse(y_true, y_pred),
        "phm08": phm08_score(y_true, y_pred),
        "mae": float(np.mean(np.abs(d))),
        "mean_error": float(np.mean(d)),           # >0 偏晚（高估 RUL）
        "late_frac": float(np.mean(d > 0)),
        "n": int(y_true.size),
    }


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def summarize_per_subset(results: dict[str, dict]) -> dict:
    """按子集汇总：macro RMSE/PHM08。"""
    rmses = [v["rmse"] for v in results.values()]
    scores = [v["phm08"] for v in results.values()]
    return {
        "per_subset": results,
        "macro_rmse": float(np.mean(rmses)),
        "macro_phm08": float(np.mean(scores)),
    }
