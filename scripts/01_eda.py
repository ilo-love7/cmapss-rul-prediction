"""01 EDA：寿命分布、piecewise 标签示意、传感器退化轨迹、工况聚类。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmapss_rul import load_config, load_all, piecewise_rul
from cmapss_rul.io import iter_units
from cmapss_rul.preprocessing import fit_conditions, op_settings, select_sensors, sensor_matrix
from cmapss_rul.plots import CLASS_COLORS, grouped_bars, line_plot, save, scatter_pred, setup_style

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"


def main() -> None:
    setup_style()
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    data = load_all(raw)

    # fig01 寿命分布
    lifes = {}
    for name, s in data.items():
        lifes[name] = np.array([rows[:, 1].max() for _, rows in iter_units(s.train)])
        print(f"{name}: train 寿命 min={lifes[name].min():.0f} median={np.median(lifes[name]):.0f} max={lifes[name].max():.0f}")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, v in lifes.items():
        ax.hist(v, bins=25, alpha=0.55, label=name, color=CLASS_COLORS[name])
    ax.set_xlabel("训练引擎寿命（cycle）")
    ax.set_ylabel("引擎数")
    ax.set_title("四子集训练引擎寿命分布（run-to-failure）")
    ax.legend()
    save(fig, FIG / "fig01_life_distribution.png")

    # fig02 piecewise RUL 标签示意
    u, rows = next(iter_units(data["FD001"].train))
    cyc = rows[:, 1]
    rul = piecewise_rul(cyc, cap=cfg["rul_cap"])
    line_plot({"piecewise RUL 标签": rul},
              f"FD001 引擎 {u}（寿命 {cyc.max():.0f} 周期）的 RUL 标签",
              "cycle", "RUL 标签", FIG / "fig02_piecewise_rul.png")

    # fig03 FD001 传感器退化轨迹（保留传感器的跨引擎均值）
    tr = data["FD001"].train
    X = sensor_matrix(tr)
    keep = select_sensors(X, std_min=cfg["sensor_std_min"])
    norm = (X[:, keep] - X[:, keep].mean(0)) / (X[:, keep].std(0) + 1e-8)
    # 按「距失效周期数」对齐后平均（0 = 失效时刻）
    max_life = int(max(r[:, 1].max() for _, r in iter_units(tr)))
    buckets: list[list[np.ndarray]] = [[] for _ in range(max_life)]
    for u_id, rows in iter_units(tr):
        life = int(rows[:, 1].max())
        seg = norm[tr[:, 0] == u_id]
        for t in range(life):
            buckets[life - 1 - t].append(seg[t])
    acc = np.full((max_life, len(keep)), np.nan)
    for i, b in enumerate(buckets):
        if b:
            acc[i] = np.mean(np.stack(b), axis=0)
    pick = list(range(0, len(keep), max(1, len(keep) // 6)))[:6]
    series = {f"传感器{keep[j]+1}": acc[::-1][:, j] for j in pick}
    line_plot(series, "FD001 退化轨迹（跨引擎平均，按寿命末端对齐）",
              "距失效的周期数（倒序）", "归一化传感器值", FIG / "fig03_sensor_trends.png")

    # fig04 FD002 工况聚类散点（前两个工况设定）
    op = op_settings(data["FD002"].train)
    _, lab = fit_conditions(op, n_clusters=cfg["n_conditions"])
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    sc = ax.scatter(op[:, 0], op[:, 1], c=lab, cmap="tab10", s=6, alpha=0.6)
    ax.set_xlabel("工况设定 1")
    ax.set_ylabel("工况设定 2")
    ax.set_title("FD002 六工况聚类（KMeans on 3 个工况设定）")
    save(fig, FIG / "fig04_op_clusters.png")

    # 概要
    from cmapss_rul import save_json
    save_json({
        "life_stats": {k: {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}
                       for k, v in lifes.items()},
        "kept_sensors_FD001": (keep + 1).tolist(),
    }, ROOT / "results" / "metrics" / "eda_summary.json")
    print("EDA 完成 →", FIG)


if __name__ == "__main__":
    main()
