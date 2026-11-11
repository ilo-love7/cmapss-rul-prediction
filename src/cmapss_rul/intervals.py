"""区间构造：pinball + 共形二段缩放（CQR / 乘性缩放 / Mondrian 分层）。

v1.2 发现 pinball [q05,q95] 最窄但覆盖不可靠（FD003 0.810）、共形-对称可靠但宽
（报告 §3.12/§5.8）。本模块把两者结合：pinball 宽度提供逐点尺度（协变量自适应），
共形只负责校准因子。全部为纯函数。

方法族
  1. cqr_interval           Romano 2019 加成式共形分位数回归：区间整体膨胀/收缩 Q
  2. scaled_interval        乘性缩放：残差除以区间半宽（sym 单因子 / asym 双因子）
  3. mondrian_conformal     按协变量（如预测 RUL 桶）分层取分位数（分层共形）
"""
from __future__ import annotations

import numpy as np


def _check_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha 必须在 (0,1)")


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """分裂式共形分位数：level = ceil((n+1)(1−alpha))/n（有限样本校正）。"""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = s.size
    if n == 0:
        raise ValueError("校准分数为空")
    _check_alpha(alpha)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, level))


# ---------------------------------------------------------------- CQR（加成式）


def cqr_nonconformity(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """CQR 不符合度：s = max(lo − y, y − hi)，y 在区间内时为负。"""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)
    return np.maximum(lo - y, y - hi)


def cqr_interval(y_cal: np.ndarray, lo_cal: np.ndarray, hi_cal: np.ndarray,
                 lo_test: np.ndarray, hi_test: np.ndarray,
                 alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray, float]:
    """加成式共形分位数回归：[lo − Q, hi + Q]，Q 可为负（收缩）。

    返回 (lo_test_new, hi_test_new, Q)。
    """
    scores = cqr_nonconformity(y_cal, lo_cal, hi_cal)
    q = conformal_quantile(scores, alpha)
    lo_test = np.asarray(lo_test, dtype=np.float64)
    hi_test = np.asarray(hi_test, dtype=np.float64)
    return lo_test - q, hi_test + q, q


# ---------------------------------------------------------------- 乘性缩放


def normalized_residuals(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                         eps: float = 1e-8) -> np.ndarray:
    """逐点归一化残差 r = (y − c) / h_side：中位数头为中心 c，按落侧取半宽。

    y ≥ c 时除以右半宽 hi−c，否则除以左半宽 c−lo。区间退化（半宽≈0）时
    回退到 eps 防除零——此时 r 绝对值大，共形会自然放大该侧。
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)
    c = (lo + hi) / 2.0
    h = np.where(y >= c, hi - c, c - lo)
    h = np.maximum(h, eps)
    return (y - c) / h


def scaled_interval(y_cal: np.ndarray, lo_cal: np.ndarray, hi_cal: np.ndarray,
                    lo_test: np.ndarray, hi_test: np.ndarray,
                    alpha: float = 0.1, asymmetric: bool = False,
                    eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, dict]:
    """pinball + 共形乘性缩放（二段式）：

    sym：λ = quantile(|r|, 1−alpha)，区间 = c ± λ·h（两侧同因子）；
    asym：r 的 α/2 与 1−α/2 分位数分别缩放左右半宽（PHM08 更怕晚预测时可用）。

    返回 (lo, hi, params)。params 含 lam / lam_lo / lam_hi。
    """
    _check_alpha(alpha)
    r_cal = normalized_residuals(y_cal, lo_cal, hi_cal, eps=eps)
    lo_test = np.asarray(lo_test, dtype=np.float64)
    hi_test = np.asarray(hi_test, dtype=np.float64)
    c = (lo_test + hi_test) / 2.0
    h_lo = np.maximum(c - lo_test, eps)
    h_hi = np.maximum(hi_test - c, eps)

    if not asymmetric:
        lam = conformal_quantile(np.abs(r_cal), alpha)
        return c - lam * h_lo, c + lam * h_hi, {"lam": lam}
    # 双侧：校准集 r 的低/高分位（有限样本校正按 α/2）
    lam_lo = conformal_quantile(-r_cal, alpha / 2)   # 左侧收缩因子（正数）
    lam_hi = conformal_quantile(r_cal, alpha / 2)
    return c - lam_lo * h_lo, c + lam_hi * h_hi, {"lam_lo": lam_lo, "lam_hi": lam_hi}


# ---------------------------------------------------------------- Mondrian 分层


def mondrian_quantiles(scores_cal: np.ndarray, strata_cal: np.ndarray,
                       alpha: float) -> dict:
    """按层取共形分位数。层内样本过少（< 5）时回退到全局分位数。"""
    s = np.asarray(scores_cal, dtype=np.float64).reshape(-1)
    st = np.asarray(strata_cal).reshape(-1)
    if s.size != st.size:
        raise ValueError("分数与层标签长度不一致")
    global_q = conformal_quantile(s, alpha)
    out = {}
    for k in np.unique(st):
        m = st == k
        out[k.item() if hasattr(k, "item") else k] = (
            conformal_quantile(s[m], alpha) if int(m.sum()) >= 5 else global_q)
    return out


def stratum_lookup(q_map: dict, strata_test: np.ndarray,
                   fallback: float | None = None) -> np.ndarray:
    """把每层分位数查表到测试点上（未见层回退 fallback，缺省取已有分位数最小值）。"""
    st = np.asarray(strata_test).reshape(-1)
    if fallback is None:
        fallback = float(min(q_map.values())) if q_map else 0.0
    out = np.empty(st.size, dtype=np.float64)
    for k in np.unique(st):
        m = st == k
        key = k.item() if hasattr(k, "item") else k
        out[m] = q_map.get(key, fallback)
    return out


def mondrian_interval(y_cal: np.ndarray, pred_cal: np.ndarray, pred_test: np.ndarray,
                      strata_cal: np.ndarray, strata_test: np.ndarray,
                      alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray, dict]:
    """分层共形-对称：残差 |pred − y| 按协变量层取分位数，逐点加宽。

    层 = 预测 RUL 分桶（测试时可用的协变量）。返回 (lo, hi, q_map)。
    """
    pred_cal = np.asarray(pred_cal, dtype=np.float64).reshape(-1)
    y_cal = np.asarray(y_cal, dtype=np.float64).reshape(-1)
    pred_test = np.asarray(pred_test, dtype=np.float64).reshape(-1)
    resid = np.abs(pred_cal - y_cal)
    q_map = mondrian_quantiles(resid, strata_cal, alpha)
    q_test = stratum_lookup(q_map, strata_test)
    return pred_test - q_test, pred_test + q_test, {str(k): float(v) for k, v in q_map.items()}
