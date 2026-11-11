"""v1.3 新增：pinball+共形二段缩放区间（CQR / 乘性缩放 / Mondrian）。"""
import numpy as np
import pytest

from cmapss_rul.intervals import (
    conformal_quantile,
    cqr_interval,
    cqr_nonconformity,
    mondrian_interval,
    mondrian_quantiles,
    normalized_residuals,
    scaled_interval,
    stratum_lookup,
)


# ---------------- conformal_quantile ----------------

def test_conformal_quantile_finite_sample_level():
    # n=9, alpha=0.1 → level = ceil(10*0.9)/9 = 1.0 → 取最大值
    s = np.arange(9, dtype=float)
    assert conformal_quantile(s, 0.1) == pytest.approx(8.0)


def test_conformal_quantile_rejects_bad_alpha():
    with pytest.raises(ValueError):
        conformal_quantile(np.ones(5), 0.0)
    with pytest.raises(ValueError):
        conformal_quantile(np.ones(5), 1.0)
    with pytest.raises(ValueError):
        conformal_quantile(np.array([]), 0.1)


# ---------------- CQR ----------------

def test_cqr_nonconformity_signs():
    y = np.array([5.0, 0.0, 10.0])
    lo = np.array([4.0, 4.0, 4.0])
    hi = np.array([6.0, 6.0, 6.0])
    s = cqr_nonconformity(y, lo, hi)
    assert s[0] == pytest.approx(-1.0)      # 区间内 → 负
    assert s[1] == pytest.approx(4.0)       # 低于下界
    assert s[2] == pytest.approx(4.0)       # 高于上界


def test_cqr_interval_zero_when_consistent():
    # 校准集全部落在区间正中 → 不符合度全为负且相等，Q=该值 → 区间收缩后仍覆盖
    y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    lo = np.zeros(5)
    hi = np.full(5, 10.0)
    lo_t, hi_t, q = cqr_interval(y, lo, hi, np.array([0.0]), np.array([10.0]), alpha=0.1)
    assert q == pytest.approx(-5.0)
    assert lo_t[0] == pytest.approx(5.0)
    assert hi_t[0] == pytest.approx(5.0)


def test_cqr_interval_inflates_when_cal_misses():
    # 校准集有样本在区间外 → Q>0 → 测试区间变宽
    y = np.array([5.0, 5.0, 5.0, 5.0, 20.0])
    lo = np.zeros(5)
    hi = np.full(5, 10.0)
    lo_t, hi_t, q = cqr_interval(y, lo, hi, np.array([0.0]), np.array([10.0]), alpha=0.1)
    assert q > 0
    assert lo_t[0] < 0.0 and hi_t[0] > 10.0


# ---------------- 乘性缩放 ----------------

def test_normalized_residuals_sides():
    y = np.array([5.0, 8.0, 2.0])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([10.0, 10.0, 10.0])
    r = normalized_residuals(y, lo, hi)
    # c=5，h=5 两侧同 → r = (y−5)/5
    assert r[0] == pytest.approx(0.0)
    assert r[1] == pytest.approx(0.6)
    assert r[2] == pytest.approx(-0.6)


def test_scaled_interval_sym_factor():
    # 校准集 y 恒在 c+3h/... 构造 r=|0.5| 全部 → lam=0.5
    lo = np.zeros(6)
    hi = np.full(6, 10.0)          # c=5, h=5
    y = np.full(6, 7.5)            # r = 2.5/5 = 0.5
    lo_t, hi_t, p = scaled_interval(y, lo, hi, np.array([0.0]), np.array([10.0]),
                                    alpha=0.1, asymmetric=False)
    assert p["lam"] == pytest.approx(0.5)
    assert lo_t[0] == pytest.approx(2.5) and hi_t[0] == pytest.approx(7.5)


def test_scaled_interval_asym_two_factors():
    # 校准集 y 分布不对称：一半在 c+4（r=0.8），一半在 c−1（r=−0.2）
    lo = np.zeros(10)
    hi = np.full(10, 10.0)
    y = np.concatenate([np.full(5, 9.0), np.full(5, 4.0)])
    lo_t, hi_t, p = scaled_interval(y, lo, hi, np.array([0.0]), np.array([10.0]),
                                    alpha=0.1, asymmetric=True)
    assert p["lam_hi"] == pytest.approx(0.8, abs=0.05)
    assert p["lam_lo"] == pytest.approx(0.2, abs=0.05)
    assert hi_t[0] > lo_t[0]


def test_scaled_interval_width_scales_with_pinball_width():
    # 逐点半宽不同 → 区间宽度随之缩放（协变量自适应）
    y = np.array([5.0] * 5)
    lo = np.array([0.0] * 5)
    hi = np.array([10.0, 2.0, 20.0, 4.0, 6.0])
    lo_t, hi_t, _ = scaled_interval(y, lo, hi,
                                    np.array([0.0, 0.0, 0.0]),
                                    np.array([10.0, 2.0, 20.0]), alpha=0.1)
    w = hi_t - lo_t
    assert w[2] > w[0] > w[1]     # 半宽 10 > 5 > 1 的次序


# ---------------- Mondrian ----------------

def test_mondrian_quantiles_per_stratum_and_fallback():
    s = np.concatenate([np.full(10, 1.0), np.full(10, 5.0)])
    st = np.concatenate([np.zeros(10), np.ones(10)])
    q = mondrian_quantiles(s, st, 0.1)
    assert q[0] == pytest.approx(1.0) and q[1] == pytest.approx(5.0)
    # 小层（<5）回退全局
    s2 = np.array([1.0, 1.0, 9.0, 9.0, 9.0, 9.0, 9.0])
    st2 = np.array([0, 0, 1, 1, 1, 1, 1])
    q2 = mondrian_quantiles(s2, st2, 0.1)
    assert q2[0] == q2[1]        # 层 0 只有 2 个样本 → 回退全局分位数


def test_stratum_lookup_unseen_stratum_uses_min():
    q_map = {"a": 2.0, "b": 8.0}
    out = stratum_lookup(q_map, np.array(["a", "b", "z"]))
    assert out[0] == 2.0 and out[1] == 8.0 and out[2] == 2.0


def test_mondrian_interval_shapes_and_keys():
    y = np.array([1.0] * 8 + [5.0] * 8)
    pred = np.zeros(16)
    st = np.array([0] * 8 + [1] * 8)
    lo, hi, q_map = mondrian_interval(y, pred, np.zeros(4), st, np.array([0, 1, 0, 1]),
                                      alpha=0.1)
    assert lo.shape == hi.shape == (4,)
    assert set(q_map.keys()) == {"0", "1"}
    assert (hi > lo).all()
