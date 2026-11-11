"""域适应：CORAL / MMD 表征对齐 + few-shot 目标域微调。

v1.1 测出零样本迁移缺口 +8.2~+20.8 RMSE 且系统性偏晚（报告 §5.3），但没有
缩小缺口的手段。本模块提供三种（报告 §5.3 未做清单）：

  CORAL  二阶统计对齐（Sun et al. 2016）：源/目标隐层协方差差的 Frobenius 范数
  MMD    多带宽 RBF 最大均值差异：源/目标隐层分布的均值嵌入距离
  few-shot  k 台目标域引擎标签微调（N1 式适配）

损失函数为纯 torch 张量函数（可单测）；训练循环与 train_regressor 同口径。
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .models import TrainResult, build_model


# ---------------------------------------------------------------- 对齐损失


def coral_loss(fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
    """CORAL：||Cov(fs) − Cov(ft)||_F² / (4 d²)。fs/ft 形状 (n, d)。

    协方差用 1/n（有偏）估计；任一侧样本 < 2 时返回 0（无二阶信息）。
    """
    if fs.ndim != 2 or ft.ndim != 2 or fs.shape[1] != ft.shape[1]:
        raise ValueError("fs/ft 须为同列数的 (n, d) 张量")
    ns, nt = fs.shape[0], ft.shape[0]
    if ns < 2 or nt < 2:
        return fs.new_zeros(())
    d = fs.shape[1]
    cs = (fs.t() @ fs) / ns
    ct = (ft.t() @ ft) / nt
    return ((cs - ct) ** 2).sum() / (4.0 * d * d)


def mmd_rbf(fs: torch.Tensor, ft: torch.Tensor,
            bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)) -> torch.Tensor:
    """多带宽 RBF-MMD²（有偏估计，非负）。

    MMD² = Σ_b [ E k(xs,xs') + E k(xt,xt') − 2 E k(xs,xt) ] / |B|
    带宽以特征各维标准差的中位数为尺度（中位数启发式），相对量纲无关。
    """
    if fs.ndim != 2 or ft.ndim != 2 or fs.shape[1] != ft.shape[1]:
        raise ValueError("fs/ft 须为同列数的 (n, d) 张量")
    ns, nt = fs.shape[0], ft.shape[0]
    if ns < 2 or nt < 2:
        return fs.new_zeros(())
    with torch.no_grad():
        scale = torch.median(torch.std(torch.cat([fs, ft], dim=0), dim=0))
        scale = torch.clamp(scale, min=1e-6)
    out = fs.new_zeros(())
    for bw in bandwidths:
        sigma = bw * scale
        kss = _rbf(fs, fs, sigma).mean()
        ktt = _rbf(ft, ft, sigma).mean()
        kst = _rbf(fs, ft, sigma).mean()
        out = out + (kss + ktt - 2.0 * kst)
    return out / len(bandwidths)


def _rbf(a: torch.Tensor, b: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """RBF 核矩阵 exp(−||a_i − b_j||² / (2σ²))。"""
    d2 = torch.cdist(a, b, p=2) ** 2
    return torch.exp(-d2 / (2.0 * sigma * sigma))


# ---------------------------------------------------------------- 训练


def train_da(X_src: np.ndarray, y_src: np.ndarray,
             X_tgt_unlab: np.ndarray | None = None,
             align: str = "none", align_weight: float = 1.0,
             align_batch: int = 256,
             X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
             hidden: int = 128, layers: int = 2, dropout: float = 0.2,
             lr: float = 1e-3, weight_decay: float = 1e-4,
             batch: int = 128, epochs: int = 40, patience: int = 8,
             seed: int = 0, device: str | None = None,
             verbose: bool = False) -> TrainResult:
    """源域 MSE 训练 + 可选表征对齐（align: none | coral | mmd）。

    对齐项在每个 batch 上取：loss = MSE(src) + align_weight · align(h_s, h_t)，
    h 为 SeqRegressor.encode 的窗口表征。目标 batch 从 X_tgt_unlab 有放回抽样。
    早停：传 X_val/y_val 时按验证集 RMSE（与 train_regressor 同口径），
    否则按源域训练集 RMSE。
    """
    if align not in ("none", "coral", "mmd"):
        raise ValueError("align 只能是 none/coral/mmd")
    if align != "none" and (X_tgt_unlab is None or len(X_tgt_unlab) == 0):
        raise ValueError(f"align={align} 需要非空目标域无标签样本")
    if (X_val is None) != (y_val is None):
        raise ValueError("X_val/y_val 须同时给出")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = torch.tensor(X_src, dtype=torch.float32)
    ytr = torch.tensor(y_src, dtype=torch.float32)
    Xt = (torch.tensor(X_tgt_unlab, dtype=torch.float32)
          if X_tgt_unlab is not None else None)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)

    model = build_model("rnn", X_src.shape[2], hidden=hidden, layers=layers,
                        dropout=dropout, cell="lstm").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    align_fn = {"coral": coral_loss, "mmd": mmd_rbf, "none": None}[align]
    gen = torch.Generator().manual_seed(seed)

    history = {"train_loss": [], "val_rmse": [], "val_score": []}
    best_state, best_val, best_epoch, bad = None, float("inf"), -1, 0

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if align_fn is not None:
                n = xb.shape[0]
                idx = torch.randint(len(Xt), (min(align_batch, len(Xt)),), generator=gen)
                xt = Xt[idx].to(device)
                hs = model.encode(xb)
                ht = model.encode(xt)
                loss = loss + align_weight * align_fn(hs, ht)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.detach()) * len(xb)
        sched.step()
        # 早停指标：验证集（若有）否则源域训练集
        model.eval()
        with torch.no_grad():
            if X_val is not None:
                pv = model(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy()
                tv = np.asarray(y_val, dtype=np.float64)
            else:
                pv = model(Xtr.to(device)).cpu().numpy()
                tv = y_src
        vrmse = float(np.sqrt(np.mean((pv - tv) ** 2)))
        vscore = float(np.sum(np.where(pv - tv < 0, np.exp(-(pv - tv) / 13.0) - 1.0,
                                       np.exp((pv - tv) / 10.0) - 1.0)))
        history["train_loss"].append(running / len(Xtr))
        history["val_rmse"].append(vrmse)
        history["val_score"].append(vscore)
        if verbose:
            print(f"  ep{ep:02d} loss={running / len(Xtr):.4f} rmse={vrmse:.3f}")
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


def few_shot_finetune(model: nn.Module, X: np.ndarray, y: np.ndarray,
                      epochs: int = 20, lr: float = 5e-4,
                      batch: int = 64, device: str | None = None,
                      seed: int = 0) -> nn.Module:
    """few-shot 目标域微调：全参数小学习率微调，返回微调后的模型（原对象不变）。

    传入已训模型（如源域模型），在 k 台目标引擎的窗口上继续训练。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ft = copy.deepcopy(model).to(device)
    opt = torch.optim.AdamW(ft.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    Xtr = torch.tensor(X, dtype=torch.float32)
    ytr = torch.tensor(y, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=min(batch, len(Xtr)),
                        shuffle=True)
    for _ in range(epochs):
        ft.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(ft(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(ft.parameters(), 5.0)
            opt.step()
    ft.eval()
    return ft
