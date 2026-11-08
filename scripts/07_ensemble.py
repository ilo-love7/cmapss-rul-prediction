"""07 模型集成与不确定性：三模型平均 vs 单模型，预测分歧作不确定性代理。

回应报告 §5.2（没做集成）与 §5.5（点估计没有不确定性）。
- 集成 = LSTM_MSE / LSTM_PHM08 / GRU_MSE 三个已训模型的预测平均
- 不确定性代理 = 三模型预测的样本间 std（模型分歧）
- 校验：Spearman(std, |err|)、按 std 三分位看误差是否随不确定性上升
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmapss_rul import evaluate, load_all, load_config, save_json
from cmapss_rul.models import SeqRegressor, predict
from cmapss_rul.plots import grouped_bars, save, setup_style
from cmapss_rul.preprocessing import (
    assign_conditions, fit_conditions, fit_scaler, op_settings,
    piecewise_rul, select_sensors, sensor_matrix,
)
from cmapss_rul.io import iter_units
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
MODELS = ROOT / "results" / "models"
STEMS = [("lstm_mse", "LSTM_MSE"), ("lstm_phm08", "LSTM_PHM08"), ("gru_mse", "GRU_MSE")]


def load_model2(stem: str, n_features: int, subset: str, cfg: dict) -> SeqRegressor:
    cell = "lstm" if stem.startswith("lstm") else "gru"
    h = cfg["lstm"]
    model = SeqRegressor(n_features, hidden=int(h["hidden"]), layers=int(h["layers"]),
                         dropout=float(h["dropout"]), cell=cell)
    model.load_state_dict(torch.load(MODELS / f"{stem}_{subset}.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model


def prepare_eval(train, test, cfg):
    tr_s = sensor_matrix(train)
    keep = select_sensors(tr_s, std_min=float(cfg["sensor_std_min"]))
    km, tr_lab = fit_conditions(op_settings(train), n_clusters=int(cfg["n_conditions"]), seed=0)
    n_clusters = max(int(cfg["n_conditions"]), 1) if km is not None else 1
    scaler = fit_scaler(tr_s[:, keep], tr_lab, n_clusters=n_clusters)
    te_lab = assign_conditions(km, op_settings(test)) if km is not None else np.zeros(len(test), dtype=int)
    Xte_s = scaler.transform(sensor_matrix(test)[:, keep], te_lab)

    window = int(cfg["window"])
    cap = float(cfg["rul_cap"])
    rul_end = np.asarray(getattr(test, "rul", None), dtype=np.float64).reshape(-1) if False else None
    return keep, km, scaler, Xte_s, window, cap


def last_windows(test, Xte_s, window):
    """每引擎末窗（短轨迹左补零），与 prepare_subset 口径一致。"""
    Xs = []
    for u in np.unique(test[:, 0]).astype(int):
        S = Xte_s[test[:, 0] == u]
        if S.shape[0] >= window:
            Xs.append(S[-window:])
        else:
            pad = np.zeros((window - S.shape[0], S.shape[1]))
            Xs.append(np.vstack([pad, S]))
    return np.asarray(Xs, dtype=np.float64)


def main() -> None:
    setup_style()
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    data = load_all(raw)
    cap = float(cfg["rul_cap"])
    out: dict = {}

    for name in cfg["subsets"]:
        train, test, rul = data[name].train, data[name].test, data[name].rul
        keep, km, scaler, Xte_s, window, _ = prepare_eval(train, test, cfg)
        Xte = last_windows(test, Xte_s, window)
        y_true = np.minimum(np.asarray(rul, dtype=np.float64).reshape(-1), cap)

        preds = {}
        for stem, label in STEMS:
            m = load_model2(stem, len(keep), name, cfg)
            preds[label] = np.clip(predict(m, Xte), 0.0, cap)

        ensemble = np.mean(np.stack(list(preds.values())), axis=0)
        std = np.std(np.stack(list(preds.values())), axis=0)
        abs_err = np.abs(ensemble - y_true)

        entry = {"n_test": int(len(y_true)), "models": {}, "ensemble": {}, "uncertainty": {}}
        for label, p in preds.items():
            entry["models"][label] = evaluate(y_true, p)
        entry["ensemble"] = evaluate(y_true, ensemble)

        # 不确定性代理有效性：Spearman(std, |err|) + 三分位
        rho, pval = spearmanr(std, abs_err)
        qs = np.quantile(std, [1 / 3, 2 / 3])
        terciles = {}
        for i, (lo, hi, lab) in enumerate([
            (0.0, qs[0], "low"), (qs[0], qs[1], "mid"), (qs[1], np.inf, "high")]):
            m = ((std >= lo) & (std < hi)) if i < 2 else (std >= lo)
            terciles[lab] = {
                "n": int(m.sum()),
                "mean_std": float(std[m].mean()) if m.any() else None,
                "mae": float(abs_err[m].mean()) if m.any() else None,
                "rmse": float(np.sqrt(np.mean((ensemble[m] - y_true[m]) ** 2))) if m.any() else None,
            }
        entry["uncertainty"] = {
            "spearman_rho": float(rho), "spearman_p": float(pval),
            "mean_std": float(std.mean()), "terciles": terciles,
        }

        print(f"\n=== {name} (n={len(y_true)}) ===")
        for label in list(preds) + ["ensemble"]:
            r = entry["models"].get(label) or entry["ensemble"]
            print(f"  {label:<12} rmse={r['rmse']:6.2f} phm08={r['phm08']:10.1f} bias={r['mean_error']:+.2f}")
        print(f"  uncertainty: rho={rho:.3f} (p={pval:.1e}) mean_std={std.mean():.3f}")
        for lab, v in terciles.items():
            if v["mae"] is None:
                print(f"    {lab:<5} n={v['n']:4d} (空)")
            else:
                print(f"    {lab:<5} n={v['n']:4d} std={v['mean_std']:.3f} mae={v['mae']:.2f} rmse={v['rmse']:.2f}")

        out[name] = entry

    # 图 1：单模型 vs 集成 RMSE
    groups = {n: dict(out[n]["models"], Ensemble=out[n]["ensemble"]) for n in out}
    grouped_bars({n: {k: v["rmse"] for k, v in g.items()} for n, g in groups.items()},
                 "单模型 vs 集成：RMSE", "RMSE", FIG / "fig13_ensemble_rmse.png")
    grouped_bars({n: {k: v["phm08"] for k, v in g.items()} for n, g in groups.items()},
                 "单模型 vs 集成：PHM08 score", "score", FIG / "fig13_ensemble_phm08.png")

    # 图 2：不确定性三分位 MAE
    fig, ax = plt.subplots(figsize=(8, 4.0))
    labels = ["low", "mid", "high"]
    x = np.arange(len(out))
    w = 0.25
    for j, lab in enumerate(labels):
        vals = [out[n]["uncertainty"]["terciles"][lab]["mae"] or 0.0 for n in out]
        ax.bar(x + (j - 1) * w, vals, w, label=f"{lab} std")
    ax.set_xticks(x)
    ax.set_xticklabels(list(out.keys()))
    ax.set_ylabel("MAE")
    ax.set_title("不确定性（模型分歧）三分位 → 误差是否随之上升")
    ax.legend()
    fig.tight_layout()
    save(fig, FIG / "fig13_uncertainty_terciles.png")

    save_json({"results": out}, ROOT / "results" / "metrics" / "ensemble_uncertainty.json")
    print("\nensemble_uncertainty.json saved")


if __name__ == "__main__":
    main()
