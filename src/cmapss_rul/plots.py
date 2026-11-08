"""绘图风格统一（与 N1 一致）。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_COLORS = {
    "FD001": "#4C78A8",
    "FD002": "#F2A93B",
    "FD003": "#72B7B2",
    "FD004": "#E45756",
    "baseline": "#9E9E9E",
    "deep": "#4C78A8",
}


def setup_style() -> None:
    for fam in ("Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"):
        try:
            matplotlib.font_manager.findfont(fam, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [fam]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25


def save(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def line_plot(series: dict[str, np.ndarray], title: str, xlabel: str, ylabel: str,
              path: str | Path, style: dict | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, ys in series.items():
        color, lab = (style or {}).get(name, (None, name))
        ax.plot(range(1, len(ys) + 1), ys, label=lab or name, color=color, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    save(fig, path)


def grouped_bars(groups: dict[str, dict[str, float]], title: str, ylabel: str,
                 path: str | Path, colors: dict | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = list(next(iter(groups.values())).keys())
    x = np.arange(len(groups))
    w = 0.8 / len(models)
    for j, m in enumerate(models):
        vals = [g[m] for g in groups.values()]
        ax.bar(x + j * w, vals, w, label=m,
               color=(colors or {}).get(m), edgecolor="white")
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(list(groups.keys()))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    save(fig, path)


def scatter_pred(y_true, y_pred, title: str, path: str | Path, cap: float = 125.0) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(y_true, y_pred, s=14, alpha=0.75, color=CLASS_COLORS["deep"])
    lim = [0, max(cap, float(np.max(y_true)) + 5)]
    ax.plot(lim, lim, "--", color="#999999", linewidth=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("真实 RUL")
    ax.set_ylabel("预测 RUL")
    ax.set_title(title)
    save(fig, path)
