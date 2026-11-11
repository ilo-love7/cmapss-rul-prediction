"""v1.3 新增：在线门控（风险-覆盖）/ 单调修复 / 自洽重标定。"""
import numpy as np
import pytest

from cmapss_rul.online import (
    causal_monotone_repair,
    coverage_at_risk,
    ema_smooth,
    failure_time_track,
    risk_at_coverage,
    risk_coverage,
    self_consistent_recal,
)


# ---------------- 风险-覆盖 ----------------

def test_risk_coverage_perfect_ranking_beats_reversed():
    n = 60
    y = np.zeros(n)
    pred = np.concatenate([np.full(n // 2, 1.0), np.full(n // 2, 10.0)])   # 前半误差小
    good_scores = np.concatenate([np.full(n // 2, 0.1), np.full(n // 2, 0.9)])
    bad_scores = good_scores[::-1]
    rc_good = risk_coverage(y, pred, good_scores)
    rc_bad = risk_coverage(y, pred, bad_scores)
    assert rc_good["aurc"] < rc_bad["aurc"]          # 排序好 → 风险-覆盖曲线更优
    assert rc_good["risk"][-1] == pytest.approx(rc_bad["risk"][-1])   # 全接受风险相同


def test_risk_coverage_shapes_and_monotone_cov():
    y = np.arange(10, dtype=float)
    pred = y + 0.5
    rc = risk_coverage(y, pred, np.linspace(0, 1, 10), n_grid=5)
    assert rc["coverage"].shape == rc["risk"].shape == rc["score_at"].shape
    assert np.all(np.diff(rc["coverage"]) > 0)       # 严格递增
    assert rc["n"] == 10


def test_risk_coverage_rejects_mismatch():
    with pytest.raises(ValueError):
        risk_coverage(np.zeros(5), np.zeros(4), np.zeros(5))
    with pytest.raises(ValueError):
        risk_coverage(np.array([]), np.array([]), np.array([]))


def test_coverage_at_risk_and_risk_at_coverage():
    y = np.zeros(10)
    pred = np.linspace(1, 10, 10)                    # 误差递增
    scores = np.linspace(0, 1, 10)                   # 同序 → 接受低分=低误差
    rc = risk_coverage(y, pred, scores)
    cov_1 = coverage_at_risk(rc, risk_max=2.0)       # 只能接受前几个小误差点
    cov_9 = coverage_at_risk(rc, risk_max=100.0)
    assert cov_1 < cov_9
    assert cov_9 == pytest.approx(1.0)
    assert risk_at_coverage(rc, 1.0) >= risk_at_coverage(rc, 0.3)


# ---------------- 单调修复 ----------------

def test_monotone_repair_flat_cap_unchanged():
    p = np.full(20, 125.0)
    out = causal_monotone_repair(p, cap=125.0)
    assert np.allclose(out, p)


def test_monotone_repair_clips_violations():
    # 中间一个向上的尖峰（r_t > r_{t−1}）与一个过陡下跌（< −1/cycle）都该被修
    p = np.array([10.0, 20.0, 9.0, 3.0, 2.0])
    out = causal_monotone_repair(p, cap=125.0)
    d = np.diff(out)
    assert np.all(d <= 1e-12)          # 非增
    assert np.all(d >= -1.0 - 1e-12)   # 每周期最多降 1
    assert out[0] == 10.0              # 首点不动
    assert out[1] == pytest.approx(10.0)   # 尖峰被压回
    assert out[2] == pytest.approx(9.0)
    assert out[3] == pytest.approx(8.0)    # 过陡下跌被放缓（每周期最多 −1）
    assert out[4] == pytest.approx(7.0)


def test_monotone_repair_bounds():
    p = np.array([-5.0, 200.0, 199.0])
    out = causal_monotone_repair(p, cap=125.0)
    assert np.all(out >= 0.0) and np.all(out <= 125.0)


def test_monotone_repair_empty():
    assert causal_monotone_repair(np.array([])).size == 0


# ---------------- 自洽重标定 ----------------

def test_failure_time_track_constant_for_consistent():
    cycles = np.arange(5)
    preds = 30.0 - cycles
    T = failure_time_track(preds, cycles)
    assert np.allclose(T, 30.0)


def test_self_consistent_recal_constant_bias_indistinguishable():
    """恒定偏置 = 另一个失效时刻 → 不变（如实记录的方法局限，非 bug）。"""
    cycles = np.arange(1, 41, dtype=float)
    true = 60.0 - cycles
    biased = true + 7.0
    out = self_consistent_recal(biased, cycles, window=10, cap=None)
    assert np.allclose(out, biased)


def test_self_consistent_recal_smooths_noise():
    """T 观测带噪声 → 滑窗中位数降方差（围绕真值的抖动被压）。"""
    cycles = np.arange(1, 41, dtype=float)
    true = 60.0 - cycles
    rng = np.random.default_rng(0)
    noisy = true + rng.normal(0.0, 3.0, size=40)
    out = self_consistent_recal(noisy, cycles, window=10, cap=None)
    assert np.std(out - true) < np.std(noisy - true)


def test_self_consistent_recal_removes_outlier_spike():
    """单点野值（预测跳 30）被中位数滤掉。"""
    cycles = np.arange(1, 21, dtype=float)
    true = 50.0 - cycles
    p = true.copy()
    p[10] += 30.0                       # 野值
    out = self_consistent_recal(p, cycles, window=10, cap=None)
    assert abs(out[10] - true[10]) < abs(p[10] - true[10])


def test_self_consistent_recal_window1_identity():
    cycles = np.arange(1, 6, dtype=float)
    preds = np.array([10.0, 9.0, 8.5, 7.0, 6.0])
    out = self_consistent_recal(preds, cycles, window=1, cap=None)
    assert np.allclose(out, preds)           # T̂=T_t → pred'=pred


def test_self_consistent_recal_respects_cap_and_nonneg():
    cycles = np.arange(1, 11, dtype=float)
    preds = np.full(10, 200.0)               # 巨大高估
    out = self_consistent_recal(preds, cycles, window=5, cap=125.0)
    assert np.all(out <= 125.0) and np.all(out >= 0.0)


def test_self_consistent_recal_rejects_mismatch():
    with pytest.raises(ValueError):
        self_consistent_recal(np.zeros(3), np.zeros(4))


# ---------------- EMA ----------------

def test_ema_smooth_constant_unchanged():
    p = np.full(8, 42.0)
    assert np.allclose(ema_smooth(p, 0.3), p)


def test_ema_smooth_rejects_bad_alpha():
    with pytest.raises(ValueError):
        ema_smooth(np.zeros(3), alpha=0.0)
