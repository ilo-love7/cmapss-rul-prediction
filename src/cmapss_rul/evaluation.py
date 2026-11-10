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


# ---------------------------------------------------------------- 校准工具

def bias_correction(y_val: np.ndarray, pred_val: np.ndarray,
                    pred_test: np.ndarray) -> tuple[np.ndarray, float]:
    """验证集偏差后验校正：把测试预测平移 −mean(pred_val − y_val)。

    返回 (校正后的测试预测, 施加的偏移量)。纯函数。
    """
    shift = float(np.mean(np.asarray(pred_val) - np.asarray(y_val)))
    return np.asarray(pred_test, dtype=np.float64) - shift, -shift


def conformal_intervals(y_val: np.ndarray, pred_val: np.ndarray,
                        pred_test: np.ndarray, alpha: float = 0.1
                        ) -> tuple[np.ndarray, np.ndarray, float]:
    """分裂式共形预测（对称残差分位数）：

        q = quantile(|pred_val − y_val|, ceil((n+1)(1−alpha))/n)
        区间 = [pred_test − q, pred_test + q]

    独立同分布假设下测试覆盖率 ≥ 1−alpha。返回 (lo, hi, q)。
    """
    y_val = np.asarray(y_val, dtype=np.float64).reshape(-1)
    pred_val = np.asarray(pred_val, dtype=np.float64).reshape(-1)
    pred_test = np.asarray(pred_test, dtype=np.float64).reshape(-1)
    n = len(y_val)
    if n == 0:
        raise ValueError("验证集为空")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha 必须在 (0,1)")
    resid = np.abs(pred_val - y_val)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q = float(np.quantile(resid, level))
    return pred_test - q, pred_test + q, q


def interval_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    """区间覆盖率与平均宽度。"""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)
    if not (len(y_true) == len(lo) == len(hi)):
        raise ValueError("长度不一致")
    inside = (y_true >= lo) & (y_true <= hi)
    return {
        "coverage": float(inside.mean()),
        "mean_width": float(np.mean(hi - lo)),
        "n": int(len(y_true)),
    }
