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
                 dropout: float = 0.2, cell: str = "lstm", bidirectional: bool = False,
                 out_dim: int = 1):
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
            nn.Linear(hidden, out_dim),
        )

    def encode(self, x):
        """窗口表征（末步隐层，投影前）——域适应对齐用。"""
        out, _ = self.rnn(x)
        return out[:, -1, :]

    def forward(self, x):
        out, _ = self.rnn(x)
        h = out[:, -1, :]  # 末时间步
        y = self.head(h)
        return y.squeeze(-1) if y.shape[-1] == 1 else y


class AttentionRegressor(nn.Module):
    """LSTM/GRU + 加性注意力池化：对全部时间步加权，而非只取末步。"""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 1,
                 dropout: float = 0.2, cell: str = "lstm", attn_dim: int = 64):
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.attn_w = nn.Linear(hidden, attn_dim)
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)                       # (B, T, H)
        score = self.attn_v(torch.tanh(self.attn_w(out))).squeeze(-1)  # (B, T)
        alpha = torch.softmax(score, dim=1)
        ctx = (alpha.unsqueeze(-1) * out).sum(dim=1)                   # (B, H)
        return self.head(ctx).squeeze(-1)


class TransformerRegressor(nn.Module):
    """紧凑 Transformer 编码器 + 末步读出。d_model=64、2 层，参数量与 LSTM 相当。"""

    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4,
                 layers: int = 2, dropout: float = 0.2, max_len: int = 64):
        super().__init__()
        self.inp = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        T = x.shape[1]
        h = self.inp(x) + self.pos[:, :T, :]
        h = self.enc(h)
        return self.head(h[:, -1, :]).squeeze(-1)


def phm08_loss(pred: torch.Tensor, target: torch.Tensor,
               early: float = 13.0, late: float = 10.0) -> torch.Tensor:
    """可微的 PHM08 评分作为损失：晚预测（pred>target）权重更陡。"""
    d = pred - target
    s = torch.where(d < 0, torch.exp(-d / early) - 1.0, torch.exp(d / late) - 1.0)
    return s.mean()


QUANTILES = (0.05, 0.5, 0.95)


def pinball_loss(pred_q: torch.Tensor, target: torch.Tensor,
                 quantiles: tuple[float, ...] = QUANTILES) -> torch.Tensor:
    """分位数损失（pinball）。pred_q (B, Q)，target (B,)。"""
    e = target.unsqueeze(-1) - pred_q                       # (B, Q)
    q = torch.tensor(quantiles, dtype=pred_q.dtype, device=pred_q.device)
    return torch.maximum(q * e, (q - 1.0) * e).mean()


@dataclass
class TrainResult:
    model: nn.Module
    history: dict
    best_epoch: int
    best_val: float


def build_model(arch: str, n_features: int, hidden: int = 128, layers: int = 2,
                dropout: float = 0.2, cell: str = "lstm",
                bidirectional: bool = False, out_dim: int = 1) -> nn.Module:
    """arch: rnn | attn | transformer。"""
    if arch == "attn":
        return AttentionRegressor(n_features, hidden=hidden, layers=layers, dropout=dropout, cell=cell)
    if arch == "transformer":
        return TransformerRegressor(n_features, d_model=hidden, layers=max(1, min(layers, 3)),
                                    dropout=dropout)
    return SeqRegressor(n_features, hidden=hidden, layers=layers, dropout=dropout,
                        cell=cell, bidirectional=bidirectional, out_dim=out_dim)


def train_regressor(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                    cell: str = "lstm", loss_name: str = "mse",
                    hidden: int = 128, layers: int = 2, dropout: float = 0.2,
                    lr: float = 1e-3, weight_decay: float = 1e-4,
                    batch: int = 128, epochs: int = 40, patience: int = 8,
                    seed: int = 0, device: str | None = None,
                    bidirectional: bool = False, verbose: bool = False,
                    select_by: str = "rmse", arch: str = "rnn",
                    quantiles: tuple[float, ...] = QUANTILES) -> TrainResult:
    """select_by: rmse（默认，与 v1.0/v1.1 一致）| score（验证集 PHM08 分数）。
    loss_name: mse | phm08 | pinball（pinball 时输出 Q 个分位数头）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dim = len(quantiles) if loss_name == "pinball" else 1

    Xtr = torch.tensor(X, dtype=torch.float32)
    ytr = torch.tensor(y, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=512, shuffle=False)

    model = build_model(arch, X.shape[2], hidden=hidden, layers=layers, dropout=dropout,
                        cell=cell, bidirectional=bidirectional, out_dim=out_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if loss_name == "phm08":
        loss_fn = lambda p, t: phm08_loss(p, t)          # noqa: E731
    elif loss_name == "pinball":
        loss_fn = lambda p, t: pinball_loss(p, t, quantiles)  # noqa: E731
    else:
        loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_rmse": [], "val_score": []}
    best_state, best_val, best_epoch, bad = None, float("inf"), -1, 0

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.detach()) * len(xb)
        sched.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb.to(device))
                if loss_name == "pinball":
                    out = out[:, quantiles.index(0.5)] if 0.5 in quantiles else out.mean(dim=-1)
                preds.append(out.cpu().numpy())
                trues.append(yb.numpy())
        pv = np.concatenate(preds)
        tv = np.concatenate(trues)
        vrmse = float(np.sqrt(np.mean((pv - tv) ** 2)))
        vscore = float(np.sum(np.where(pv - tv < 0, np.exp(-(pv - tv) / 13.0) - 1.0,
                                       np.exp((pv - tv) / 10.0) - 1.0)))
        history["train_loss"].append(running / len(Xtr))
        history["val_rmse"].append(vrmse)
        history["val_score"].append(vscore)
        metric = vrmse if select_by == "rmse" else vscore
        if verbose:
            print(f"  ep{ep:02d} loss={running/len(Xtr):.4f} val_rmse={vrmse:.3f} val_score={vscore:.1f}")
        if metric < best_val - 1e-5:
            best_val, best_epoch, bad = metric, ep, 0
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
