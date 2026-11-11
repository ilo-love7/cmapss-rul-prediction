"""v1.3 新增：域适应对齐损失（CORAL/MMD）与 few-shot 微调。"""
import numpy as np
import pytest
import torch

from cmapss_rul.adaptation import coral_loss, few_shot_finetune, mmd_rbf, train_da
from cmapss_rul.models import SeqRegressor


# ---------------- CORAL ----------------

def test_coral_zero_when_covariance_equal():
    g = torch.Generator().manual_seed(0)
    fs = torch.randn(50, 8, generator=g)
    ct = fs[torch.randperm(50, generator=g)]        # 同分布同协方差
    assert float(coral_loss(fs, ct)) == pytest.approx(0.0, abs=1e-6)


def test_coral_positive_when_covariance_differs():
    fs = torch.randn(50, 4) * 1.0
    ft = torch.randn(50, 4) * 3.0
    assert float(coral_loss(fs, ft)) > 0.0


def test_coral_symmetric_and_tiny_batch():
    fs = torch.randn(20, 3)
    ft = torch.randn(30, 3) * 2
    assert float(coral_loss(fs, ft)) == pytest.approx(float(coral_loss(ft, fs)))
    assert float(coral_loss(fs[:1], ft)) == 0.0       # n<2 无二阶信息


def test_coral_shape_guard():
    with pytest.raises(ValueError):
        coral_loss(torch.randn(5, 3), torch.randn(5, 4))


# ---------------- MMD ----------------

def test_mmd_zero_for_identical_samples():
    fs = torch.randn(30, 5)
    assert float(mmd_rbf(fs, fs.clone())) == pytest.approx(0.0, abs=1e-5)


def test_mmd_positive_for_mean_shift():
    fs = torch.randn(40, 3)
    ft = fs + 2.0
    assert float(mmd_rbf(fs, ft)) > 0.01


def test_mmd_symmetric():
    fs = torch.randn(25, 4)
    ft = torch.randn(35, 4) + 0.5
    assert float(mmd_rbf(fs, ft)) == pytest.approx(float(mmd_rbf(ft, fs)))


def test_mmd_tiny_batch_zero():
    assert float(mmd_rbf(torch.randn(1, 3), torch.randn(10, 3))) == 0.0


# ---------------- train_da / few-shot ----------------

def _toy(n=40, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 10, d)).astype(np.float32)
    y = X[:, -1, 0] * 2.0 + 1.0
    return X, y


def test_train_da_none_smoke():
    X, y = _toy()
    res = train_da(X, y, align="none", hidden=8, layers=1, epochs=2, batch=16, seed=0)
    assert res.model is not None and len(res.history["train_loss"]) >= 1


@pytest.mark.parametrize("align", ["coral", "mmd"])
def test_train_da_align_smoke(align):
    X, y = _toy(seed=1)
    Xt, _ = _toy(n=30, seed=2)
    res = train_da(X, y, Xt, align=align, align_weight=0.5, hidden=8, layers=1,
                   epochs=2, batch=16, seed=0)
    assert res.best_val < 1e6        # 跑完且有最优值


def test_train_da_requires_target_for_align():
    X, y = _toy()
    with pytest.raises(ValueError):
        train_da(X, y, None, align="coral", hidden=8, layers=1, epochs=1)
    with pytest.raises(ValueError):
        train_da(X, y, np.zeros((0, 10, 3)), align="mmd", hidden=8, layers=1, epochs=1)


def test_train_da_val_early_stop_path():
    X, y = _toy(n=40, seed=4)
    Xv, yv = _toy(n=16, seed=5)
    res = train_da(X, y, Xv, align="mmd", X_val=Xv, y_val=yv,
                   hidden=8, layers=1, epochs=2, batch=16, seed=0)
    assert res.model is not None
    with pytest.raises(ValueError):
        train_da(X, y, Xv, align="none", X_val=Xv, y_val=None, hidden=8,
                 layers=1, epochs=1)      # 只给一半


def test_train_da_rejects_bad_align():
    X, y = _toy()
    with pytest.raises(ValueError):
        train_da(X, y, align="dann", hidden=8, layers=1, epochs=1)


def test_few_shot_finetune_copies_model():
    X, y = _toy(n=20, seed=3)
    model = SeqRegressor(3, hidden=8, layers=1)
    ft = few_shot_finetune(model, X, y, epochs=2, lr=1e-2, batch=8, seed=0)
    assert ft is not model
    p0 = list(model.parameters())[0].clone()
    _ = few_shot_finetune(model, X, y, epochs=1, lr=1e-2, batch=8, seed=0)
    assert torch.equal(list(model.parameters())[0], p0)   # 原模型不动


def test_encode_shape():
    model = SeqRegressor(3, hidden=8, layers=1)
    h = model.encode(torch.randn(5, 10, 3))
    assert h.shape == (5, 8)
