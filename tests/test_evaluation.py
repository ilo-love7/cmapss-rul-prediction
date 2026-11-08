"""评估指标单测：PHM08 的不对称性是核心。"""
import numpy as np
import pytest

from cmapss_rul.evaluation import evaluate, phm08_score, rmse, summarize_per_subset


def test_rmse_basic():
    assert rmse([0, 0, 0], [3, 4, 0]) == pytest.approx(np.sqrt((9 + 16 + 0) / 3))


def test_phm08_asymmetric_late_penalized_more():
    y = np.array([100.0])
    early = phm08_score(y, y - 10)   # 预测 90，早 10
    late = phm08_score(y, y + 10)    # 预测 110，晚 10
    assert late > early > 0
    assert late == pytest.approx(np.exp(1.0) - 1.0)
    assert early == pytest.approx(np.exp(10 / 13) - 1.0)


def test_phm08_perfect_zero():
    y = np.array([10.0, 50.0, 125.0])
    assert phm08_score(y, y) == pytest.approx(0.0)


def test_phm08_sum_over_units():
    y = np.array([100.0, 100.0])
    s = phm08_score(y, np.array([110.0, 90.0]))
    assert s == pytest.approx(phm08_score(y[:1], y[:1] + 10) + phm08_score(y[1:], y[1:] - 10))


def test_evaluate_fields():
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([12.0, 18.0, 33.0])
    m = evaluate(y, p)
    assert m["n"] == 3
    assert m["late_frac"] == pytest.approx(2 / 3)
    assert m["mean_error"] == pytest.approx((2 - 2 + 3) / 3)
    assert m["rmse"] == pytest.approx(rmse(y, p))


def test_summarize_per_subset():
    res = {
        "FD001": {"rmse": 12.0, "phm08": 100.0},
        "FD002": {"rmse": 20.0, "phm08": 300.0},
    }
    s = summarize_per_subset(res)
    assert s["macro_rmse"] == pytest.approx(16.0)
    assert s["macro_phm08"] == pytest.approx(200.0)
