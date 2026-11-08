"""06 滚动预测：测试引擎每周期连续预测 RUL，替代"只评末窗"。

回应报告 §5.4：真实场景要每周期都出一个 RUL，误差随时序变化。
- 对每个测试引擎滑窗到每个周期，预测 RUL
- 真值（未封顶）= RUL_FDxxx + (末周期 - 当前周期)；同时给封顶口径
- 按真值 RUL 分桶给 RMSE，看"早期被低估 / 末期准"是否成立

复用 03 已训权重（预处理可确定性重算），无需重训。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmapss_rul import load_all, load_config, save_json
from cmapss_rul.io import iter_units
from cmapss_rul.models import SeqRegressor, predict
from cmapss_rul.plots import save, setup_style
from cmapss_rul.preprocessing import (
    Scaler, assign_conditions, build_sequences, fit_conditions, fit_scaler,
    op_settings, piecewise_rul, select_sensors, sensor_matrix,
)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
MODELS = ROOT / "results" / "models"

BUCKETS = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, 200), (200, 1e9)]
BUCKET_LABELS = ["0-25", "25-50", "50-75", "75-100", "100-125", "125-200", ">200"]
MODEL_FILES = [("lstm_mse", "LSTM_MSE"), ("lstm_phm08", "LSTM_PHM08"), ("gru_mse", "GRU_MSE")]


def load_model2(stem: str, n_features: int, subset: str, cfg: dict) -> SeqRegressor:
    cell = "lstm" if stem.startswith("lstm") else "gru"
    h = cfg["lstm"]
    model = SeqRegressor(n_features, hidden=int(h["hidden"]), layers=int(h["layers"]),
                         dropout=float(h["dropout"]), cell=cell)
    path = MODELS / f"{stem}_{subset}.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def prepare_source(train: np.ndarray, cfg: dict):
    """确定性重算源域预处理（与 prepare_subset 前半一致，seed=0）。"""
    tr_s = sensor_matrix(train)
    keep = select_sensors(tr_s, std_min=float(cfg["sensor_std_min"]))
    km, tr_lab = fit_conditions(op_settings(train), n_clusters=int(cfg["n_conditions"]), seed=0)
    n_clusters = max(int(cfg["n_conditions"]), 1) if km is not None else 1
    scaler = fit_scaler(tr_s[:, keep], tr_lab, n_clusters=n_clusters)
    Xtr_s = scaler.transform(tr_s[:, keep], tr_lab)
    ytr = np.zeros(len(train))
    for u, rows in iter_units(train):
        idx = np.where(train[:, 0] == u)[0]
        ytr[idx] = piecewise_rul(rows[:, 1], cap=float(cfg["rul_cap"]))
    return keep, km, scaler, Xtr_s, ytr


def transform_target(test: np.ndarray, keep, km, scaler: Scaler) -> np.ndarray:
    te_lab = assign_conditions(km, op_settings(test)) if km is not None else np.zeros(len(test), dtype=int)
    return scaler.transform(sensor_matrix(test)[:, keep], te_lab)


def rolling_windows(S: np.ndarray, window: int) -> np.ndarray:
    """(n, d) → (n-window+1, window, d)，只取满窗。"""
    n = S.shape[0]
    return np.stack([S[t - window + 1:t + 1] for t in range(window - 1, n)])


def main() -> None:
    setup_style()
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    data = load_all(raw)
    cap = float(cfg["rul_cap"])
    window = int(cfg["window"])
    out: dict = {}

    for name in cfg["subsets"]:
        train, test, rul = data[name].train, data[name].test, data[name].rul
        keep, km, scaler, _, _ = prepare_source(train, cfg)
        Xte_s = transform_target(test, keep, km, scaler)
        rul_end = np.asarray(rul, dtype=np.float64).reshape(-1)
        test_units = np.unique(test[:, 0]).astype(int)

        # 真值轨迹（未封顶）
        truths_raw, preds = {k: [] for _, k in MODEL_FILES}, {k: [] for _, k in MODEL_FILES}
        models = {k: load_model2(stem, len(keep), name, cfg) for stem, k in MODEL_FILES}
        traj_store = {}  # 供画图：前 3 个引擎

        for i, u in enumerate(test_units):
            S = Xte_s[test[:, 0] == u]
            n = S.shape[0]
            if n < window:
                continue
            Xw = rolling_windows(S, window)           # 每窗末周期 t = window-1 .. n-1
            ts = np.arange(window - 1, n)
            true_raw = rul_end[i] + (n - 1 - ts)      # 未封顶真值
            for key, m in models.items():
                p = np.clip(predict(m, Xw), 0.0, cap)
                preds[key].append(p)
                truths_raw[key].append(true_raw)
            if u in [int(x) for x in test_units[:3]]:
                traj_store[int(u)] = (ts, true_raw, {k: preds[k][-1] for k in preds})

        # 指标
        entry = {"n_units": len(test_units), "window": window, "models": {}}
        for key in preds:
            p = np.concatenate(preds[key])
            t_raw = np.concatenate(truths_raw[key])
            t_cap = np.minimum(t_raw, cap)
            rmse_raw = float(np.sqrt(np.mean((p - t_raw) ** 2)))
            rmse_cap = float(np.sqrt(np.mean((p - t_cap) ** 2)))
            by_bucket = {}
            for (lo, hi), lab in zip(BUCKETS, BUCKET_LABELS):
                m = (t_raw >= lo) & (t_raw < hi)
                by_bucket[lab] = float(np.sqrt(np.mean((p[m] - t_raw[m]) ** 2))) if m.any() else None
            entry["models"][key] = {
                "rmse_raw_truth": rmse_raw,
                "rmse_capped_truth": rmse_cap,
                "bias_raw_truth": float(np.mean(p - t_raw)),
                "n_windows": int(len(p)),
                "rmse_by_bucket": by_bucket,
            }
            print(f"{name} {key}: rmse_raw={rmse_raw:.2f} rmse_cap={rmse_cap:.2f} "
                  f"bias_raw={np.mean(p - t_raw):+.2f} n={len(p)}")
            for lab, v in by_bucket.items():
                if v is not None:
                    print(f"    [{lab}] rmse={v:.2f}")

        out[name] = entry

        # 图：轨迹（真值 vs 3 模型预测）
        if traj_store:
            fig, axes = plt.subplots(len(traj_store), 1, figsize=(9, 2.6 * len(traj_store)), sharex=False)
            if len(traj_store) == 1:
                axes = [axes]
            colors = {"LSTM_MSE": "#4C78A8", "LSTM_PHM08": "#E45756", "GRU_MSE": "#9E7BAC"}
            for ax, (u, (ts, t_raw, pp)) in zip(axes, traj_store.items()):
                ax.plot(ts, t_raw, color="black", linewidth=2.0, label="真实 RUL（未封顶）")
                for k, p in pp.items():
                    ax.plot(ts, p, color=colors[k], linewidth=1.3, label=k)
                ax.axhline(cap, color="gray", linestyle="--", linewidth=0.8)
                ax.set_title(f"{name} 测试引擎 #{u}（n={len(ts) + window - 1} 周期）")
                ax.set_ylabel("RUL")
                ax.legend(fontsize=7, loc="upper right")
            axes[-1].set_xlabel("测试轨迹内的周期序号")
            fig.tight_layout()
            save(fig, FIG / f"fig12_rolling_trajectory_{name.lower()}.png")

    # 汇总图：RMSE 按真值桶（每子集一条线，LSTM_PHM08）
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors4 = {"FD001": "#4C78A8", "FD002": "#F2A93B", "FD003": "#72B7B2", "FD004": "#E45756"}
    for name in cfg["subsets"]:
        vals = [out[name]["models"]["LSTM_PHM08"]["rmse_by_bucket"].get(l) for l in BUCKET_LABELS]
        vals = [v if v is not None else np.nan for v in vals]
        ax.plot(BUCKET_LABELS, vals, marker="o", color=colors4[name], label=name)
    ax.set_xlabel("该周期的真实 RUL（未封顶）")
    ax.set_ylabel("RMSE")
    ax.set_title("滚动预测误差随寿命位置的变化（LSTM_PHM08）")
    ax.legend()
    fig.tight_layout()
    save(fig, FIG / "fig12_rolling_rmse_by_bucket.png")

    save_json({"results": out}, ROOT / "results" / "metrics" / "rolling_predictions.json")
    print("rolling_predictions.json saved")


if __name__ == "__main__":
    main()
