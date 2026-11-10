"""10 · 轻量注意力模型 vs LSTM：注意力池化 / 紧凑 Transformer。

问题：末时间步读出（v1.0 的 SeqRegressor）丢掉了退化趋势的时间结构；
注意力池化或自注意编码是否能在同预算下改进 RMSE/PHM08？

对照引用 results/metrics/deep.json 的 LSTM_MSE（同协议同种子，未重跑）；
新模型：LSTM+加性注意力池化（attn）、紧凑 Transformer（d=64、2 层）。
主战场 FD001/FD004 × 2 种子；FD002/003 × 1 种子验证方向。

产出
  results/metrics/attention.json
  results/figures/fig22_attention.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss_rul import evaluate, load_all, load_config, prepare_subset, save_json  # noqa: E402
from cmapss_rul.models import build_model, predict, train_regressor  # noqa: E402
from cmapss_rul.preprocessing import train_val_split_units  # noqa: E402

ARCHS = [("attn", "LSTM+Attn"), ("transformer", "Transformer")]
REF_FILE = ROOT / "results" / "metrics" / "deep.json"


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> int:
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    data = load_all(raw)
    cap = float(cfg["rul_cap"])
    h = cfg["lstm"]
    ref = json.loads(REF_FILE.read_text(encoding="utf-8"))["results"]
    out: dict = {"reference": {}, "models": {}}

    plan = {"FD001": [0, 1], "FD004": [0, 1], "FD002": [0], "FD003": [0]}

    for name, seeds in plan.items():
        t0 = time.time()
        d = prepare_subset(data[name].train, data[name].test, data[name].rul,
                           window=cfg["window"], rul_cap=cap,
                           std_min=cfg["sensor_std_min"],
                           n_conditions=cfg["n_conditions"],
                           cluster=cfg["cluster_op_settings"])
        tr_m, va_m = train_val_split_units(d["meta"], val_frac=cfg["val_frac_units"], seed=0)
        print(f"\n===== {name}: 训练 {tr_m.sum()} / 验证 {va_m.sum()} / 测试 {len(d['y_test'])} =====")
        # 引用 v1.0 同协议 LSTM_MSE
        r = ref[name]["LSTM_MSE"]["mean"]
        out["reference"][name] = {k: r[k] for k in ("rmse", "phm08", "mae", "mean_error", "late_frac")}
        print(f"  [ref] LSTM_MSE: rmse={r['rmse']:.2f} score={r['phm08']:.0f}")
        out["models"][name] = {}

        for arch, label in ARCHS:
            runs = []
            for seed in seeds:
                t1 = time.time()
                res = train_regressor(
                    d["X"][tr_m], d["y"][tr_m], d["X"][va_m], d["y"][va_m],
                    arch=arch, loss_name="mse", select_by="rmse",
                    hidden=int(h["hidden"]), layers=int(h["layers"]),
                    dropout=float(h["dropout"]), lr=float(h["lr"]),
                    weight_decay=float(h["weight_decay"]),
                    batch=int(h["batch"]), epochs=int(h["epochs"]),
                    patience=int(h["patience"]), seed=seed, verbose=False,
                )
                pred = np.clip(predict(res.model, d["X_test"]), 0.0, cap)
                m = evaluate(d["y_test"], pred)
                n_params = count_params(res.model)
                m.update({"seed": seed, "best_epoch": res.best_epoch,
                          "n_params": n_params, "train_s": round(time.time() - t1, 1)})
                runs.append(m)
                print(f"  {label} seed={seed}: rmse={m['rmse']:.2f} score={m['phm08']:.0f} "
                      f"bias={m['mean_error']:+.2f} params={n_params} ({m['train_s']}s)")
            keys = ["rmse", "phm08", "mae", "mean_error", "late_frac"]
            out["models"][name][label] = {
                "mean": {k: float(np.mean([r[k] for r in runs])) for k in keys},
                "std": {k: float(np.std([r[k] for r in runs])) for k in keys},
                "n_params": runs[0]["n_params"],
                "runs": runs,
            }
        print(f"  ({time.time() - t0:.0f}s)")
        save_json({"reference": out["reference"], "models": out["models"]},
                  ROOT / "results" / "metrics" / "attention.json")

    print("\n=== 对比（测试 RMSE / PHM08）===")
    for name in plan:
        r = out["reference"][name]
        print(f"  {name}: ref LSTM {r['rmse']:.2f}/{r['phm08']:.0f}  " + "  ".join(
            f"{lab} {out['models'][name][lab]['mean']['rmse']:.2f}/"
            f"{out['models'][name][lab]['mean']['phm08']:.0f}"
            for _, lab in ARCHS))
    save_json({"reference": out["reference"], "models": out["models"]},
              ROOT / "results" / "metrics" / "attention.json")

    # 图
    import matplotlib.pyplot as plt
    from cmapss_rul.plots import setup_style
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    subs = list(plan)
    x = np.arange(len(subs))
    width = 0.25
    series = [("LSTM（ref）", [out["reference"][s]["rmse"] for s in subs], "#4C78A8")]
    for k, (arch, lab) in enumerate(ARCHS):
        series.append((lab, [out["models"][s][lab]["mean"]["rmse"] for s in subs],
                       ("#E45756", "#F2A93B")[k]))
    for k, (lab, vals, col) in enumerate(series):
        axes[0].bar(x + (k - 1) * width, vals, width, label=lab, color=col)
    axes[0].set_xticks(x); axes[0].set_xticklabels(subs)
    axes[0].set_ylabel("RMSE"); axes[0].set_title("注意力模型 vs LSTM：RMSE")
    axes[0].legend(fontsize=7)

    series2 = [("LSTM（ref）", [out["reference"][s]["phm08"] for s in subs], "#4C78A8")]
    for k, (arch, lab) in enumerate(ARCHS):
        series2.append((lab, [out["models"][s][lab]["mean"]["phm08"] for s in subs],
                        ("#E45756", "#F2A93B")[k]))
    for k, (lab, vals, col) in enumerate(series2):
        axes[1].bar(x + (k - 1) * width, vals, width, label=lab, color=col)
    axes[1].set_xticks(x); axes[1].set_xticklabels(subs)
    axes[1].set_yscale("log"); axes[1].set_ylabel("PHM08 score（log）")
    axes[1].set_title("注意力模型 vs LSTM：PHM08 score")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "fig22_attention.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("图 → results/figures/fig22_attention.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
