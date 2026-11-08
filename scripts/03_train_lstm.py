"""03 深度模型：LSTM/GRU × {MSE, PHM08 损失}，4 个子集。

评估协议与基线一致：训练集内部按引擎 80/20 划分验证；
测试集每引擎取末窗预测 RUL，与 RUL_FDxxx 对齐计算 RMSE + PHM08。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmapss_rul import evaluate, load_all, load_config, prepare_subset, save_json
from cmapss_rul.models import predict, train_regressor
from cmapss_rul.plots import scatter_pred, setup_style
from cmapss_rul.preprocessing import train_val_split_units

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
MODELS = ROOT / "results" / "models"

# (cell, loss, label)
CONFIGS = [
    ("lstm", "mse", "LSTM_MSE"),
    ("lstm", "phm08", "LSTM_PHM08"),
    ("gru", "mse", "GRU_MSE"),
]


def main() -> None:
    setup_style()
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    data = load_all(raw)
    cap = float(cfg["rul_cap"])
    h = cfg["lstm"]
    seeds = list(cfg["seeds"])[:2]   # 深度模型 2 个种子
    out: dict[str, dict] = {}

    for name in cfg["subsets"]:
        t0 = time.time()
        d = prepare_subset(data[name].train, data[name].test, data[name].rul,
                           window=cfg["window"], rul_cap=cap,
                           std_min=cfg["sensor_std_min"],
                           n_conditions=cfg["n_conditions"],
                           cluster=cfg["cluster_op_settings"])
        tr_m, va_m = train_val_split_units(d["meta"], val_frac=cfg["val_frac_units"], seed=0)
        print(f"\n===== {name}: 训练 {tr_m.sum()} 序列 / 验证 {va_m.sum()} / 测试 {len(d['y_test'])} =====")
        out[name] = {}

        for cell, loss_name, label in CONFIGS:
            runs = []
            for seed in seeds:
                t1 = time.time()
                res = train_regressor(
                    d["X"][tr_m], d["y"][tr_m], d["X"][va_m], d["y"][va_m],
                    cell=cell, loss_name=loss_name,
                    hidden=int(h["hidden"]), layers=int(h["layers"]),
                    dropout=float(h["dropout"]), lr=float(h["lr"]),
                    weight_decay=float(h["weight_decay"]),
                    batch=int(h["batch"]), epochs=int(h["epochs"]),
                    patience=int(h["patience"]), seed=seed, verbose=False,
                )
                pred = predict(res.model, d["X_test"])
                m = evaluate(d["y_test"], np.clip(pred, 0.0, cap))
                m.update({"seed": seed, "best_epoch": res.best_epoch,
                          "best_val_rmse": res.best_val, "train_s": round(time.time() - t1, 1)})
                runs.append(m)
                print(f"  {label} seed={seed}: rmse={m['rmse']:.2f} phm08={m['phm08']:.1f} "
                      f"bias={m['mean_error']:+.2f} ep={res.best_epoch} ({m['train_s']}s)")
                if seed == seeds[0]:
                    MODELS.mkdir(parents=True, exist_ok=True)
                    import torch
                    torch.save(res.model.state_dict(), MODELS / f"{cell}_{loss_name}_{name}.pt")
                    if name in ("FD001", "FD004"):
                        scatter_pred(d["y_test"], np.clip(pred, 0.0, cap),
                                     f"{name} {label} 预测 vs 真实",
                                     FIG / f"fig08_scatter_{name.lower()}_{label.lower()}.png", cap=cap)

            keys = ["rmse", "phm08", "mae", "mean_error", "late_frac"]
            out[name][label] = {
                "runs": runs,
                "mean": {k: float(np.mean([r[k] for r in runs])) for k in keys},
                "std": {k + "_std": float(np.std([r[k] for r in runs])) for k in keys},
            }
            mu = out[name][label]["mean"]
            print(f"  → {label} mean: rmse={mu['rmse']:.2f}±{out[name][label]['std']['rmse_std']:.2f} "
                  f"phm08={mu['phm08']:.1f}")

        print(f"  ({time.time()-t0:.0f}s)")

    save_json({"results": out}, ROOT / "results" / "metrics" / "deep.json")

    # 汇总曲线：各子集最优模型 val 曲线
    from cmapss_rul.plots import grouped_bars
    for metric, title, ylab in [("rmse", "深度模型 RMSE（2 seeds 均值）", "RMSE"),
                                ("phm08", "深度模型 PHM08 score（2 seeds 均值）", "score")]:
        groups = {n: {lab: out[n][lab]["mean"][metric] for lab in out[n]} for n in out}
        grouped_bars(groups, title, ylab, FIG / f"fig07_deep_{metric}.png")
    print("deep.json saved")


if __name__ == "__main__":
    main()
