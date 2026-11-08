"""05 汇总图：hero 图（RMSE 排行 + 要点标注）。"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmapss_rul.plots import setup_style, save
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"


def main() -> None:
    setup_style()
    table = json.loads((ROOT / "results/metrics/comparison.json").read_text(encoding="utf-8"))["table"]

    # hero：宏平均 RMSE 排行条
    models = list(table["FD001"].keys())
    macro = {m: float(np.mean([table[n][m]["rmse"] for n in table])) for m in models}
    order = sorted(macro, key=macro.get)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colors = {"Constant_cap": "#BBBBBB", "Linear_HI": "#F2A93B", "Similarity": "#72B7B2",
              "LSTM_MSE": "#4C78A8", "LSTM_PHM08": "#E45756", "GRU_MSE": "#9E7BAC"}
    y = np.arange(len(order))
    ax.barh(y, [macro[m] for m in order], color=[colors.get(m) for m in order])
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    for i, m in enumerate(order):
        ax.text(macro[m] + 0.4, i, f"{macro[m]:.1f}", va="center", fontsize=9)
    ax.set_xlabel("四子集宏平均 RMSE（越低越好）")
    ax.set_title("CMAPSS 涡扇 RUL 预测 · 模型总览（N2）")
    save(fig, ROOT / "docs/figures/hero.png")
    save(fig, FIG / "fig11_hero.png")
    print("hero saved:", {m: round(macro[m], 2) for m in order})


if __name__ == "__main__":
    main()
