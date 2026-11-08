"""02 基线：常数封顶 / 线性 HI 外推 / 相似度匹配。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmapss_rul import evaluate, load_all, load_config, prepare_subset, save_json, summarize_per_subset
from cmapss_rul.models import constant_baseline, linear_hi_rul, similarity_rul
from cmapss_rul.plots import CLASS_COLORS, grouped_bars, scatter_pred, setup_style

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"


def main() -> None:
    setup_style()
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    data = load_all(raw)
    cap = float(cfg["rul_cap"])

    out: dict[str, dict] = {}
    for name in cfg["subsets"]:
        t0 = time.time()
        d = prepare_subset(data[name].train, data[name].test, data[name].rul,
                           window=cfg["window"], rul_cap=cap,
                           std_min=cfg["sensor_std_min"],
                           n_conditions=cfg["n_conditions"],
                           cluster=cfg["cluster_op_settings"])
        y = d["y_test"]

        preds = {
            "Constant_cap": constant_baseline(y, cap=cap),
            "Linear_HI": linear_hi_rul(d["test_series"], d["train_series"], cap=cap),
            "Similarity": similarity_rul(d["train_series"], d["train_totals"],
                                         d["X_test"], cap=cap),
        }
        out[name] = {m: evaluate(y, p) for m, p in preds.items()}
        for m, p in preds.items():
            e = out[name][m]
            print(f"{name} {m:<12} rmse={e['rmse']:6.2f}  phm08={e['phm08']:9.1f}  "
                  f"bias={e['mean_error']:+6.2f}  late%={e['late_frac']*100:4.1f}")
        scatter_pred(y, preds["Linear_HI"], f"{name} 线性 HI 外推", FIG / f"fig05_scatter_{name.lower()}_linear.png", cap=cap)
        print(f"  ({time.time()-t0:.1f}s)")

    save_json({"per_subset": out}, ROOT / "results" / "metrics" / "baselines.json")

    # 汇总图：RMSE 与 PHM08 分组柱状
    for metric, title, ylab in [("rmse", "基线 RMSE（越低越好）", "RMSE"),
                                ("phm08", "基线 PHM08 score（越低越好）", "score")]:
        groups = {n: {m: v[metric] for m, v in out[n].items()} for n in out}
        grouped_bars(groups, title, ylab, FIG / f"fig06_baselines_{metric}.png")

    macro = {m: float(np.mean([out[n][m]["rmse"] for n in out])) for m in out["FD001"]}
    print("macro rmse:", {k: round(v, 2) for k, v in macro.items()})
    save_json({"macro_rmse": macro}, ROOT / "results" / "metrics" / "baselines_macro.json")


if __name__ == "__main__":
    main()
