"""模型：常数/线性外推/相似度匹配基线 + LSTM/GRU 回归。

深度模型输入 (batch, window, d) 归一化传感器序列，输出窗口末步 RUL。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- baselines


def constant_baseline(y_test: np.ndarray, cap: float = 125.0) -> np.ndarray:
    """把所有预测设为封顶 RUL——piecewise 标签下的朴素下界。"""
    return np.full_like(np.asarray(y_test, dtype=np.float64), cap)


def linear_hi_rul(test_series: list[np.ndarray], train_series: list[np.ndarray] | None = None,
                  cap: float = 125.0, tail: int = 30) -> np.ndarray:
    """线性健康指标外推（趋势法）：对每引擎末段 HI 做线性外推到失效阈值。

    HI = 该时刻归一化传感器均值；失效阈值 = 训练引擎末时刻 HI 均值
    （不传 train_series 时用各测试引擎自身的末窗 HI 作为阈值参考）。
    test_series: 每个测试引擎的 (cycles, d) 归一化传感器序列。
    """
    if train_series is not None and len(train_series):
        fail_level = float(np.mean([s[-1].mean() for s in train_series]))
    else:
        fail_level = float(np.mean([s[-1].mean() for s in test_series]))

    out = np.zeros(len(test_series))
    for i, s in enumerate(test_series):
        hi = s.mean(axis=1)
        k = min(tail, len(hi))
        x = np.arange(k, dtype=np.float64)
        slope = float(np.polyfit(x, hi[-k:], 1)[0]) if k >= 2 else 0.0
        if slope <= 1e-6:      # 未见上升趋势 → 常数预测
            out[i] = cap
        else:
            out[i] = float(np.clip((fail_level - hi[-1]) / slope, 0.0, cap))
    return out


def similarity_rul(train_series: list[np.ndarray], train_total: np.ndarray,
                   test_tail: np.ndarray, cap: float = 125.0,
                   n_ref: int = 10) -> np.ndarray:
    """相似度匹配 RUL：把测试末窗与训练引擎的滑动窗比相关，加权剩余寿命。

    train_series: 每个训练引擎的 (cycles, d) 传感器序列（已归一化）
    train_total:  每个训练引擎的总寿命（cycle 数）
    test_tail:    (n_test, window, d) 每个测试引擎的末窗
    """
    test_tail = np.asarray(test_tail, dtype=np.float64)
    n_test, W, _ = test_tail.shape
    preds = np.zeros(n_test)

    # 预提取训练引擎的所有滑窗
    train_windows = []
    for tr, tot in zip(train_series, train_total):
        n = tr.shape[0]
        if n < W:
            train_windows.append(None)
            continue
        from numpy.lib.stride_tricks import sliding_window_view
        win = sliding_window_view(tr, W, axis=0)          # (n-W+1, d, W)
        win = np.transpose(win, (0, 2, 1))                 # (m, W, d)
        w0 = win - win.mean(axis=1, keepdims=True)
        wn = np.linalg.norm(w0, axis=(1, 2)) + 1e-8
        rem = np.maximum(tot - (np.arange(n - W + 1) + W), 0.0)
        train_windows.append((w0, wn, rem))

    for i in range(n_test):
        tt = test_tail[i]
        tt0 = tt - tt.mean()
        tn = np.linalg.norm(tt0) + 1e-8
        best_score, best_rem = [], []
        for j, packed in enumerate(train_windows):
            if packed is None:
                continue
            w0, wn, rem = packed
            corr = (w0 * tt0).sum(axis=(1, 2)) / (wn * tn)
            k = int(np.argmax(corr))
            best_score.append(float(corr[k]))
            best_rem.append(float(rem[k]))
        if not best_score:
            preds[i] = cap
            continue
        scores = np.asarray(best_score)
        rems = np.asarray(best_rem)
        ok = scores > 0.0
        if not ok.any():
            preds[i] = cap
            continue
        idx = np.where(ok)[0]
        top = idx[np.argsort(scores[idx])[::-1][:n_ref]]
        w = scores[top]
        w = w / (w.sum() + 1e-8)
        preds[i] = float(np.clip(np.sum(w * rems[top]), 0.0, cap))
    return preds


# ---------------------------------------------------------------- torch 模型

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SeqRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.2, cell: str = "lstm", bidirectional: bool = False):
        super().__init__()
        self.bidirectional = bidirectional
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * (2 if bidirectional else 1), hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        h = out[:, -1, :]  # 末时间步
        return self.head(h).squeeze(-1)


def phm08_loss(pred: torch.Tensor, target: torch.Tensor,
               early: float = 13.0, late: float = 10.0) -> torch.Tensor:
    """可微的 PHM08 评分作为损失：晚预测（pred>target）权重更陡。"""
    d = pred - target
    s = torch.where(d < 0, torch.exp(-d / early) - 1.0, torch.exp(d / late) - 1.0)
    return s.mean()


@dataclass
class TrainResult:
    model: nn.Module
    history: dict
    best_epoch: int
    best_val: float


def train_regressor(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                    cell: str = "lstm", loss_name: str = "mse",
                    hidden: int = 128, layers: int = 2, dropout: float = 0.2,
                    lr: float = 1e-3, weight_decay: float = 1e-4,
                    batch: int = 128, epochs: int = 40, patience: int = 8,
                    seed: int = 0, device: str | None = None,
                    bidirectional: bool = False, verbose: bool = False) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = torch.tensor(X, dtype=torch.float32)
    ytr = torch.tensor(y, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=512, shuffle=False)

    model = SeqRegressor(X.shape[2], hidden=hidden, layers=layers, dropout=dropout,
                         cell=cell, bidirectional=bidirectional).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = (lambda p, t: phm08_loss(p, t)) if loss_name == "phm08" else nn.MSELoss()

    history = {"train_loss": [], "val_rmse": []}
    best_state, best_val, best_epoch, bad = None, float("inf"), -1, 0

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.detach()) * len(xb)
        sched.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(model(xb.to(device)).cpu().numpy())
                trues.append(yb.numpy())
        vrmse = float(np.sqrt(np.mean((np.concatenate(preds) - np.concatenate(trues)) ** 2)))
        history["train_loss"].append(running / len(Xtr))
        history["val_rmse"].append(vrmse)
        if verbose:
            print(f"  ep{ep:02d} loss={running/len(Xtr):.4f} val_rmse={vrmse:.3f}")
        if vrmse < best_val - 1e-5:
            best_val, best_epoch, bad = vrmse, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model=model, history=history, best_epoch=best_epoch, best_val=best_val)


def predict(model: nn.Module, X: np.ndarray, device: str | None = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.tensor(X[i:i + 512], dtype=torch.float32).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)
