"""v1.2 新增：分位数损失 / 校准工具 / 注意力模型形状。"""
import numpy as np
import pytest
import torch

from cmapss_rul.evaluation import (bias_correction, conformal_intervals,
                                   interval_coverage, phm08_score)
from cmapss_rul.models import (AttentionRegressor, QUANTILES, SeqRegressor,
                               TransformerRegressor, pinball_loss)


# ---------------- pinball ----------------

def test_pinball_loss_perfect_median():
    # 中位数头预测真实值时，pinball（q=0.5）应为 0
    y = torch.tensor([1.0, 2.0, 3.0])
    pred = y.unsqueeze(1)                      # (3, 1) 每个样本预测 = 其真值
    loss = pinball_loss(pred, y, quantiles=(0.5,))
    assert float(loss) < 1e-6


def test_pinball_loss_asymmetric():
    # 低分位（q=0.1）下，高估比低估更贵：q*e vs (1-q)*|e|
    y = torch.tensor([1.0])
    pred_hi = torch.tensor([[2.0]])   # 高估 +1
    pred_lo = torch.tensor([[0.0]])   # 低估 −1
    l_hi = pinball_loss(pred_hi, y, quantiles=(0.1,))
    l_lo = pinball_loss(pred_lo, y, quantiles=(0.1,))
    assert float(l_hi) == pytest.approx(0.9, abs=1e-5)
    assert float(l_lo) == pytest.approx(0.1, abs=1e-5)
    assert float(l_hi) > float(l_lo)


# ---------------- 校准 ----------------

def test_bias_correction_centers():
    y = np.array([10.0, 20.0, 30.0])
    pred_val = np.array([12.0, 22.0, 32.0])          # 系统性 +2
    pred_test = np.array([5.0, 15.0])
    corrected, shift = bias_correction(y, pred_val, pred_test)
    assert shift == pytest.approx(-2.0)
    assert corrected[0] == pytest.approx(3.0)
    # 校正后验证集残差均值为 0
    assert (pred_val + shift - y).mean() == pytest.approx(0.0)


def test_conformal_coverage_synthetic():
    rng = np.random.default_rng(0)
    resid = rng.normal(0, 2.0, size=2000)
    y_val = rng.normal(50, 10, 1000)
    pred_val = y_val + resid[:1000]
    y_test = rng.normal(50, 10, 1000)
    pred_test = y_test + resid[1000:]
    lo, hi, q = conformal_intervals(y_val, pred_val, pred_test, alpha=0.1)
    cov = interval_coverage(y_test, lo, hi)
    assert cov["coverage"] == pytest.approx(0.90, abs=0.05)
    assert q > 0


def test_conformal_validation():
    with pytest.raises(ValueError):
        conformal_intervals(np.array([]), np.array([]), np.array([1.0]))
    with pytest.raises(ValueError):
        conformal_intervals(np.array([1.0]), np.array([1.0]), np.array([1.0]), alpha=1.5)


def test_interval_coverage_shape_guard():
    with pytest.raises(ValueError):
        interval_coverage(np.array([1.0, 2.0]), np.array([0.0]), np.array([3.0, 4.0]))


# ---------------- 模型形状 ----------------

def test_seq_regressor_multidim_output():
    m = SeqRegressor(n_features=4, hidden=8, layers=1, out_dim=3)
    x = torch.randn(5, 10, 4)
    y = m(x)
    assert y.shape == (5, 3)
    m1 = SeqRegressor(n_features=4, hidden=8, layers=1, out_dim=1)
    assert m1(x).shape == (5,)


def test_attention_regressor_shape_and_finite():
    m = AttentionRegressor(n_features=4, hidden=8, layers=1)
    x = torch.randn(5, 10, 4)
    y = m(x)
    assert y.shape == (5,)
    assert torch.isfinite(y).all()


def test_transformer_regressor_shape():
    m = TransformerRegressor(n_features=4, d_model=16, nhead=2, layers=1, max_len=16)
    x = torch.randn(5, 10, 4)
    y = m(x)
    assert y.shape == (5,)
    assert torch.isfinite(y).all()


def test_quantiles_constant():
    assert QUANTILES == (0.05, 0.5, 0.95)
