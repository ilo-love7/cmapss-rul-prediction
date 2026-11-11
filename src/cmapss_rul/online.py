"""在线后处理：不确定性门控（选择性预测）+ 轨迹单调修复 + 自洽重标定。

v1.1 的滚动预测是"离线全轨迹评估"（报告 §5.4）。本模块给在线化三件套：

  门控       不确定性分数超阈值 → 弃预测/报警；风险-覆盖曲线 + AURC 量化
  单调修复   piecewise RUL 的物理约束 −1 ≤ r_t − r_{t−1} ≤ 0，因果前向裁剪
  自洽重标定 RUL_t + t ≡ T（失效时刻）→ 滑窗中位数估 T̂，无标签在线校偏

全部纯函数，输入为单引擎滚动序列。
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- 门控 / 选择性预测


def risk_coverage(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray,
                  n_grid: int = 50) -> dict:
    """选择性预测曲线：按不确定性分数从低到高逐步接受样本。

    scores 越大越不可信；阈值 τ 接受 score ≤ τ 的点。返回：
      coverage  保留比例（0~1，含 0 与 1 两端）
      risk      保留点的 RMSE（risk）
      score_at  对应阈值
      aurc      风险-覆盖曲线下面积（越小越好；完美排序时接近 ∫最小风险）
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = y_true.size
    if not (y_pred.size == scores.size == n) or n == 0:
        raise ValueError("y_true/y_pred/scores 长度须一致且非空")
    order = np.argsort(scores, kind="stable")
    err_sorted = (y_pred[order] - y_true[order]) ** 2
    cum = np.cumsum(err_sorted)
    # 接受前 k 个（k=1..n）的风险
    ks = np.arange(1, n + 1)
    risk_all = np.sqrt(cum / ks)
    cov_all = ks / n
    # 下采样网格（含全接受端点）
    idx = np.unique(np.concatenate([np.linspace(0, n - 1, n_grid).astype(int), [n - 1]]))
    coverage = cov_all[idx]
    risk = risk_all[idx]
    aurc = float(np.trapezoid(np.concatenate([[0.0], risk]),
                              np.concatenate([[0.0], coverage])))
    return {
        "coverage": coverage,
        "risk": risk,
        "score_at": scores[order][idx],
        "aurc": aurc,
        "n": int(n),
    }


def coverage_at_risk(rc: dict, risk_max: float) -> float:
    """给定可接受风险上限，返回最大可保留比例（线性插值）。"""
    risk, cov = np.asarray(rc["risk"]), np.asarray(rc["coverage"])
    below = np.where(risk <= risk_max)[0]
    if below.size == 0:
        return 0.0
    return float(cov[below[-1]])


def risk_at_coverage(rc: dict, cov_target: float) -> float:
    """给定保留比例，返回该点风险（线性插值取最近格点）。"""
    cov, risk = np.asarray(rc["coverage"]), np.asarray(rc["risk"])
    i = int(np.argmin(np.abs(cov - cov_target)))
    return float(risk[i])


# ---------------------------------------------------------------- 单调修复


def causal_monotone_repair(preds: np.ndarray, cap: float = 125.0) -> np.ndarray:
    """因果前向修复：piecewise RUL 满足 −1 ≤ r_t − r_{t−1} ≤ 0 且 0 ≤ r ≤ cap。

    r'_t = clip(r_t, r'_{t−1} − 1, r'_{t−1})，随后截断到 [0, cap]。
    在线可实现（只看历史）；封顶平台（r=cap）保持不变。
    """
    p = np.asarray(preds, dtype=np.float64).reshape(-1).copy()
    if p.size == 0:
        return p
    p[0] = np.clip(p[0], 0.0, cap)
    for t in range(1, p.size):
        lo = max(0.0, p[t - 1] - 1.0)
        hi = min(cap, p[t - 1])
        p[t] = np.clip(p[t], lo, hi)
    return p


# ---------------------------------------------------------------- 自洽重标定


def failure_time_track(preds: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    """T 的逐点观测：pred + cycle（真值下恒为失效时刻）。"""
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    t = np.asarray(cycles, dtype=np.float64).reshape(-1)
    if p.size != t.size:
        raise ValueError("preds/cycles 长度不一致")
    return p + t


def self_consistent_recal(preds: np.ndarray, cycles: np.ndarray,
                          window: int = 20, cap: float | None = 125.0) -> np.ndarray:
    """无标签在线重标定：T̂_t = 滑窗 median(pred_τ + τ)，pred'_t = T̂_t − t。

    利用 RUL_t + t ≡ T 的时序恒等式校正慢漂移偏差；window 个最近观测做中位数
    （抗离群）。cap 给定时截断到 [0, cap]；None 时只截下界 0。
    """
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    t = np.asarray(cycles, dtype=np.float64).reshape(-1)
    if p.size != t.size:
        raise ValueError("preds/cycles 长度不一致")
    if window < 1:
        raise ValueError("window ≥ 1")
    T = failure_time_track(p, t)
    out = np.empty_like(p)
    for i in range(p.size):
        lo = max(0, i - window + 1)
        that = float(np.median(T[lo:i + 1]))
        out[i] = that - t[i]
    out = np.maximum(out, 0.0)
    if cap is not None:
        out = np.minimum(out, float(cap))
    return out


def ema_smooth(preds: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """指数滑动平均（可选的额外平滑）。alpha 越大越信新观测。"""
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha ∈ (0,1]")
    out = np.empty_like(p)
    out[0] = p[0]
    for t in range(1, p.size):
        out[t] = alpha * p[t] + (1 - alpha) * out[t - 1]
    return out
