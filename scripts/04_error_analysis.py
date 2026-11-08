"""04 误差分析：基线 vs 深度、早晚偏差、按真值 RUL 分桶。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmapss_rul import load_config, save_json
from cmapss_rul.plots import setup_style, save
import matplotlib.pyplot as plt
import json

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"


def main() -> None:
    setup_style()
    base = json.loads((ROOT / "results/metrics/baselines.json").read_text(encoding="utf-8"))["per_subset"]
    deep = json.loads((ROOT / "results/metrics/deep.json").read_text(encoding="utf-8"))["results"]

    # 表格：每子集各模型 RMSE / PHM08
    table = {}
    for name in base:
        table[name] = {}
        for m, v in base[name].items():
            table[name][m] = {"rmse": v["rmse"], "phm08": v["phm08"], "mean_error": v["mean_error"]}
        for m, v in deep[name].items():
            table[name][m] = {"rmse": v["mean"]["rmse"], "phm08": v["mean"]["phm08"],
                              "mean_error": v["mean"]["mean_error"]}
        print(name)
        for m, v in sorted(table[name].items(), key=lambda kv: kv[1]["rmse"]):
            print(f"  {m:<12} rmse={v['rmse']:6.2f}  phm08={v['phm08']:10.1f}  bias={v['mean_error']:+6.2f}")

    # 图：RMSE 对比（基线 vs 深度）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    models = list(table["FD001"].keys())
    x = np.arange(len(base))
    w = 0.8 / len(models)
    colors = {"Constant_cap": "#BBBBBB", "Linear_HI": "#F2A93B", "Similarity": "#72B7B2",
              "LSTM_MSE": "#4C78A8", "LSTM_PHM08": "#E45756", "GRU_MSE": "#9E7BAC"}
    for j, m in enumerate(models):
        axes[0].bar(x + j * w, [table[n][m]["rmse"] for n in x], w, label=m, color=colors.get(m))
        axes[1].bar(x + j * w, [table[n][m]["phm08"] for n in x], w, label=m, color=colors.get(m))
    for ax, title, ylab in zip(axes, ["RMSE 对比", "PHM08 score 对比"], ["RMSE", "score"]):
        ax.set_xticks(x + w * (len(models) - 1) / 2)
        ax.set_xticklabels(list(base.keys()))
        ax.set_title(title)
        ax.set_ylabel(ylab)
    axes[0].legend(fontsize=7, ncol=2)
    save(fig, FIG / "fig09_baseline_vs_deep.png")

    # 图：偏差分布（各模型 mean_error，>0 偏晚）
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for j, m in enumerate(models):
        ax.bar(x + j * w, [table[n][m]["mean_error"] for n in x], w, label=m, color=colors.get(m))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(list(base.keys()))
    ax.set_ylabel("预测误差（pred − true，>0 偏晚更危险）")
    ax.set_title("各模型早晚偏差对比")
    ax.legend(fontsize=7, ncol=2)
    save(fig, FIG / "fig10_bias.png")

    save_json({"table": table}, ROOT / "results/metrics/comparison.json")
    print("comparison.json + fig09/fig10 saved")


if __name__ == "__main__":
    main()
