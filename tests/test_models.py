"""模型单测：PHM08 损失的不对称性 + 训练环冒烟。"""
import numpy as np
import pytest
import torch

from cmapss_rul.models import (
    SeqRegressor, phm08_loss, predict, train_regressor,
    constant_baseline, linear_hi_rul, similarity_rul,
)


def test_phm08_loss_asymmetric():
    target = torch.tensor([100.0])
    late = phm08_loss(torch.tensor([110.0]), target)
    early = phm08_loss(torch.tensor([90.0]), target)
    assert float(late) > float(early) > 0
    assert float(late) == pytest.approx(np.exp(1.0) - 1.0, rel=1e-5)


def test_phm08_loss_differentiable():
    p = torch.tensor([110.0], requires_grad=True)
    loss = phm08_loss(p, torch.tensor([100.0]))
    loss.backward()
    assert p.grad is not None and float(p.grad) > 0  # 高估 RUL 时梯度为正（往回拉）


def test_seq_regressor_forward_shapes():
    for cell in ("lstm", "gru"):
        m = SeqRegressor(n_features=7, hidden=16, layers=1, cell=cell)
        x = torch.randn(4, 30, 7)
        out = m(x)
        assert out.shape == (4,)


def test_train_regressor_smoke():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 10, 5)).astype(np.float64)
    y = rng.uniform(0, 100, size=64)
    Xv = rng.normal(size=(16, 10, 5)).astype(np.float64)
    yv = rng.uniform(0, 100, size=16)
    res = train_regressor(X, y, Xv, yv, cell="gru", loss_name="mse",
                          hidden=16, layers=1, dropout=0.0,
                          batch=32, epochs=3, patience=3, seed=0)
    assert res.best_epoch >= 1
    preds = predict(res.model, Xv)
    assert preds.shape == (16,)
    assert np.all(np.isfinite(preds))


def test_constant_baseline():
    p = constant_baseline(np.array([10.0, 200.0]), cap=125)
    assert list(p) == [125.0, 125.0]


def test_linear_hi_and_similarity_shapes():
    rng = np.random.default_rng(1)
    train_series = [rng.normal(size=(80, 4)) for _ in range(5)]
    train_total = np.array([80, 90, 70, 85, 75], dtype=float)
    test_tail = rng.normal(size=(3, 30, 4))
    p_sim = similarity_rul(train_series, train_total, test_tail, cap=125)
    assert p_sim.shape == (3,)
    assert np.all((p_sim >= 0) & (p_sim <= 125))

    test_series = [rng.normal(size=(60, 4)) for _ in range(3)]
    p_lin = linear_hi_rul(test_series, train_series, cap=125)
    assert p_lin.shape == (3,)
    assert np.all((p_lin >= 0) & (p_lin <= 125))

    # 无上升趋势 → 常数封顶预测
    flat = [np.ones((40, 4)) for _ in range(2)]
    assert np.all(linear_hi_rul(flat, cap=125) == 125.0)
