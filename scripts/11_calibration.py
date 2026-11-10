"""11 · 不确定性校准：预测区间的覆盖率。

问题：v1.1 的集成 std 只是"分歧代理"（Spearman 0.26），不是校准过的区间。
本脚本在**同一个基础模型**（09 存的 phm08_score 检查点）上比较四种区间：

  A naive     pred ± 1.96·RMSE_val   （正态假设，未校准）
  B 共形-对称  分裂式共形：残差分位数 q，区间 pred ± q
  C 共形-非对称 上/下残差各自取分位数（PHM08 更怕晚预测，下界可以更紧）
  D 分位数回归  3 头 pinball（0.05/0.5/0.95），区间 [q05, q95]

评价：90% 名义覆盖率（目标 0.90±0.05）+ 平均区间宽度。基础模型点预测沿用
09 的 phm08_score 单元格（FD001/FD004 2 种子、FD002/003 1 种子）。

产出
  results/metrics/calibration.json
  results/figures/fig23_calibration.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss_rul import evaluate, load_all, load_config, prepare_subset, save_json  # noqa: E402
from cmapss_rul.evaluation import (conformal_intervals, interval_coverage,  # noqa: E402
                                   rmse)
from cmapss_rul.models import predict, train_regressor  # noqa: E402
from cmapss_rul.preprocessing import train_val_split_units  # noqa: E402

MODELS = ROOT / "results" / "models"
NOMINAL = 0.90
ALPHA = 1 - NOMINAL


def asymmetric_conformal(y_val, pred_val, pred_test, alpha):
    """上/下残差分开取分位数的非对称共形区间。"""
    resid = np.asarray(pred_val) - np.asarray(y_val)
    over = np.maximum(resid, 0.0)          # 高估（危险侧）
    under = np.maximum(-resid, 0.0)        # 低估
    n = len(resid)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha / 2)) / n)   # 两侧各 alpha/2
    q_over = float(np.quantile(over, level))
    q_under = float(np.quantile(under, level))
    pred_test = np.asarray(pred_test)
    return pred_test - q_under, pred_test + q_over, (q_under, q_over)


def main() -> int:
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    data = load_all(raw)
    cap = float(cfg["rul_cap"])
    h = cfg["lstm"]
    plan = {"FD001": [0, 1], "FD004": [0, 1], "FD002": [0], "FD003": [0]}
    out: dict = {}

    for name, seeds in plan.items():
        t0 = time.time()
        d = prepare_subset(data[name].train, data[name].test, data[name].rul,
                           window=cfg["window"], rul_cap=cap,
                           std_min=cfg["sensor_std_min"],
                           n_conditions=cfg["n_conditions"],
                           cluster=cfg["cluster_op_settings"])
        tr_m, va_m = train_val_split_units(d["meta"], val_frac=cfg["val_frac_units"], seed=0)
        print(f"\n===== {name} =====")
        out[name] = {}

        # ---- 基础模型：09 存的 phm08_score 检查点（seed=0）----
        ckpt = MODELS / f"scoreaware_phm08_score_{name}_s0.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"缺 {ckpt.name}，先运行 scripts/09_score_aware.py")
        from cmapss_rul.models import build_model
        model = build_model("rnn", d["X"].shape[2], hidden=int(h["hidden"]),
                            layers=int(h["layers"]), dropout=float(h["dropout"]), cell="lstm")
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        pred_val = np.clip(predict(model, d["X"][va_m]), 0.0, cap)
        pred_test = np.clip(predict(model, d["X_test"]), 0.0, cap)
        y_val, y_test = d["y"][va_m], d["y_test"]
        base = evaluate(y_test, pred_test)
        print(f"  基础模型 phm08_score: rmse={base['rmse']:.2f} score={base['phm08']:.0f}")

        methods = {}

        # A naive ±1.96·RMSE_val
        s = rmse(y_val, pred_val)
        lo, hi = pred_test - 1.96 * s, pred_test + 1.96 * s
        methods["naive_1.96rmse"] = {
            "interval": interval_coverage(y_test, lo, hi), "param": {"rmse_val": s}}

        # B 共形-对称
        lo, hi, q = conformal_intervals(y_val, pred_val, pred_test, alpha=ALPHA)
        methods["conformal_sym"] = {
            "interval": interval_coverage(y_test, lo, hi), "param": {"q": q}}

        # C 共形-非对称
        lo, hi, (qu, qo) = asymmetric_conformal(y_val, pred_val, pred_test, alpha=ALPHA)
        methods["conformal_asym"] = {
            "interval": interval_coverage(y_test, lo, hi), "param": {"q_under": qu, "q_over": qo}}

        # D 分位数回归（pinball 3 头）
        for seed in seeds:
            t1 = time.time()
            res = train_regressor(
                d["X"][tr_m], d["y"][tr_m], d["X"][va_m], d["y"][va_m],
                cell="lstm", loss_name="pinball", select_by="rmse",
                hidden=int(h["hidden"]), layers=int(h["layers"]),
                dropout=float(h["dropout"]), lr=float(h["lr"]),
                weight_decay=float(h["weight_decay"]),
                batch=int(h["batch"]), epochs=int(h["epochs"]),
                patience=int(h["patience"]), seed=seed, verbose=False,
            )
            out_q = predict(res.model, d["X_test"])
            if out_q.ndim == 1:      # 防御：单头回退
                raise RuntimeError("pinball 模型输出应为 3 列")
            q05, q50, q95 = [np.clip(out_q[:, i], 0.0, cap) for i in range(3)]
            cov = interval_coverage(y_test, q05, q95)
            m = {"interval": cov,
                 "median_point": evaluate(y_test, q50),
                 "seed": seed, "train_s": round(time.time() - t1, 1)}
            key = f"pinball_q05_95_s{seed}"
            methods[key] = m
            print(f"  pinball seed={seed}: cov={cov['coverage']:.3f} "
                  f"width={cov['mean_width']:.1f} rmse={m['median_point']['rmse']:.2f} "
                  f"({m['train_s']}s)")
            if seed == seeds[0]:
                torch.save(res.model.state_dict(),
                           MODELS / f"pinball_{name}_s{seed}.pt")

        for k, m in methods.items():
            iv = m["interval"]
            print(f"  {k:18s} cov={iv['coverage']:.3f} width={iv['mean_width']:.1f}")
        out[name] = {"base": base, "methods": methods}
        print(f"  ({time.time() - t0:.0f}s)")
        save_json({"nominal": NOMINAL, "results": out},
                  ROOT / "results" / "metrics" / "calibration.json")

    save_json({"nominal": NOMINAL, "results": out},
              ROOT / "results" / "metrics" / "calibration.json")

    # 图
    import matplotlib.pyplot as plt
    from cmapss_rul.plots import setup_style
    setup_style()
    labels = ["naive_1.96rmse", "conformal_sym", "conformal_asym", "pinball_q05_95_s0"]
    pretty = ["naive±1.96σ", "共形-对称", "共形-非对称", "分位数回归"]
    subs = list(plan)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(subs))
    width = 0.2
    for k, (lab, p) in enumerate(zip(labels, pretty)):
        covs = []
        for s in subs:
            # pinball 取 seed0
            m = out[s]["methods"].get(lab) or out[s]["methods"]["pinball_q05_95_s0"]
            covs.append(m["interval"]["coverage"])
        axes[0].bar(x + (k - 1.5) * width, covs, width, label=p)
    axes[0].axhline(NOMINAL, color="k", ls="--", lw=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(subs)
    axes[0].set_ylim(0.5, 1.02); axes[0].set_ylabel("90% 区间覆盖率")
    axes[0].set_title("覆盖率（虚线=名义 0.90）")
    axes[0].legend(fontsize=7)

    for k, (lab, p) in enumerate(zip(labels, pretty)):
        widths = []
        for s in subs:
            m = out[s]["methods"].get(lab) or out[s]["methods"]["pinball_q05_95_s0"]
            widths.append(m["interval"]["mean_width"])
        axes[1].bar(x + (k - 1.5) * width, widths, width, label=p)
    axes[1].set_xticks(x); axes[1].set_xticklabels(subs)
    axes[1].set_ylabel("平均区间宽度（cycles）")
    axes[1].set_title("区间宽度（越窄越好，前提覆盖达标）")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "fig23_calibration.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("图 → results/figures/fig23_calibration.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
