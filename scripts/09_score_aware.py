"""09 · PHM08 分数感知训练与选模：损失 × 选模准则矩阵 + 后验偏差校正。

v1.0/v1.1 的早停一律按验证 RMSE 选检查点。本脚本回答两个问题：

  A 选模准则   同一训练损失下，按验证 RMSE vs 按验证 PHM08 分数选检查点，
             测试分数会不会更好？矩阵 = {MSE 损失, PHM08 损失} × {RMSE 选模, 分数选模}
  B 偏差校正   在验证集上估一个标量偏移把偏差清零，再平移到测试预测——
             这是零训练成本的 PHM08 分数改善手段

主战场 FD001（单工况）与 FD004（六工况最复杂）× 2 种子；FD002/003 只跑
两个"自然"对角（MSE+RMSE、PHM08+分数）× 1 种子验证方向。

产出
  results/metrics/score_aware.json
  results/figures/fig21_score_aware.png
  results/models/scoreaware_{cell}_{subset}_s{seed}.pt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss_rul import evaluate, load_all, load_config, prepare_subset, save_json  # noqa: E402
from cmapss_rul.evaluation import bias_correction  # noqa: E402
from cmapss_rul.models import predict, train_regressor  # noqa: E402
from cmapss_rul.preprocessing import train_val_split_units  # noqa: E402

MODELS = ROOT / "results" / "models"
CELLS = [("mse", "rmse"), ("mse", "score"), ("phm08", "rmse"), ("phm08", "score")]
DIAG = [("mse", "rmse"), ("phm08", "score")]


def main() -> int:
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    data = load_all(raw)
    cap = float(cfg["rul_cap"])
    h = cfg["lstm"]
    out: dict = {"A_selection": {}, "B_bias_correction": {}}

    plan = {
        "FD001": (CELLS, [0, 1]),
        "FD004": (CELLS, [0, 1]),
        "FD002": (DIAG, [0]),
        "FD003": (DIAG, [0]),
    }

    for name, (cells, seeds) in plan.items():
        t0 = time.time()
        d = prepare_subset(data[name].train, data[name].test, data[name].rul,
                           window=cfg["window"], rul_cap=cap,
                           std_min=cfg["sensor_std_min"],
                           n_conditions=cfg["n_conditions"],
                           cluster=cfg["cluster_op_settings"])
        tr_m, va_m = train_val_split_units(d["meta"], val_frac=cfg["val_frac_units"], seed=0)
        print(f"\n===== {name}: 训练 {tr_m.sum()} / 验证 {va_m.sum()} / 测试 {len(d['y_test'])} =====")
        out["A_selection"][name] = {}
        out["B_bias_correction"][name] = {}

        for loss_name, select_by in cells:
            label = f"{loss_name}_{select_by}"
            runs = []
            for seed in seeds:
                t1 = time.time()
                res = train_regressor(
                    d["X"][tr_m], d["y"][tr_m], d["X"][va_m], d["y"][va_m],
                    cell="lstm", loss_name=loss_name, select_by=select_by,
                    hidden=int(h["hidden"]), layers=int(h["layers"]),
                    dropout=float(h["dropout"]), lr=float(h["lr"]),
                    weight_decay=float(h["weight_decay"]),
                    batch=int(h["batch"]), epochs=int(h["epochs"]),
                    patience=int(h["patience"]), seed=seed, verbose=False,
                )
                pred = np.clip(predict(res.model, d["X_test"]), 0.0, cap)
                m = evaluate(d["y_test"], pred)
                # 后验偏差校正（偏移量只从验证集估）
                pv = np.clip(predict(res.model, d["X"][va_m]), 0.0, cap)
                pred_bc, shift = bias_correction(d["y"][va_m], pv, pred)
                m_bc = evaluate(d["y_test"], np.clip(pred_bc, 0.0, cap))
                m.update({"seed": seed, "best_epoch": res.best_epoch,
                          "val_best": res.best_val, "train_s": round(time.time() - t1, 1)})
                runs.append({"raw": m, "bias_corrected": m_bc, "shift": shift})
                print(f"  {label} seed={seed}: rmse={m['rmse']:.2f} score={m['phm08']:.0f} "
                      f"bias={m['mean_error']:+.2f} → 校正 rmse={m_bc['rmse']:.2f} "
                      f"score={m_bc['phm08']:.0f} (shift={shift:+.2f}, ep={res.best_epoch})")
                if seed == seeds[0]:
                    MODELS.mkdir(parents=True, exist_ok=True)
                    import torch
                    torch.save(res.model.state_dict(),
                               MODELS / f"scoreaware_{label}_{name}_s{seed}.pt")
            keys = ["rmse", "phm08", "mae", "mean_error", "late_frac"]
            agg = {
                "mean": {k: float(np.mean([r["raw"][k] for r in runs])) for k in keys},
                "mean_bc": {k: float(np.mean([r["bias_corrected"][k] for r in runs])) for k in keys},
                "runs": runs,
            }
            out["A_selection"][name][label] = agg
            out["B_bias_correction"][name][label] = {
                "mean_shift": float(np.mean([r["shift"] for r in runs])),
                "score_raw": agg["mean"]["phm08"], "score_bc": agg["mean_bc"]["phm08"],
                "score_improve_frac": float((agg["mean"]["phm08"] - agg["mean_bc"]["phm08"])
                                            / max(agg["mean"]["phm08"], 1e-9)),
            }
        print(f"  ({time.time() - t0:.0f}s)")
        save_json({"A_selection": out["A_selection"], "B_bias_correction": out["B_bias_correction"]},
                  ROOT / "results" / "metrics" / "score_aware.json")

    # 汇总打印
    print("\n=== 选模准则对比（测试 PHM08 score，越低越好）===")
    for name in plan:
        row = out["A_selection"][name]
        print(f"  {name}: " + "  ".join(
            f"{lab}={row[lab]['mean']['phm08']:.0f}" for lab in row))
    print("\n=== 偏差校正（score_raw → score_bc）===")
    for name in plan:
        row = out["B_bias_correction"][name]
        print(f"  {name}: " + "  ".join(
            f"{lab} {r['score_raw']:.0f}→{r['score_bc']:.0f}" for lab, r in row.items()))
    save_json({"A_selection": out["A_selection"], "B_bias_correction": out["B_bias_correction"]},
              ROOT / "results" / "metrics" / "score_aware.json")

    # 图
    import matplotlib.pyplot as plt
    from cmapss_rul.plots import setup_style
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    subs = list(plan)
    x = np.arange(len(subs))
    width = 0.2
    labs_all = ["mse_rmse", "mse_score", "phm08_rmse", "phm08_score"]
    for k, lab in enumerate(labs_all):
        vals = [out["A_selection"][s][lab]["mean"]["phm08"] if lab in out["A_selection"][s]
                else np.nan for s in subs]
        ax = axes[0]
        ax.bar(x + (k - 1.5) * width, vals, width, label=lab)
    axes[0].set_xticks(x); axes[0].set_xticklabels(subs)
    axes[0].set_yscale("log"); axes[0].set_ylabel("测试 PHM08 score（log）")
    axes[0].set_title("A 损失 × 选模准则")
    axes[0].legend(fontsize=7)

    all_raw, all_bc = [], []
    for k, lab in enumerate(labs_all):
        raw_v = [out["B_bias_correction"][s][lab]["score_raw"] if lab in out["B_bias_correction"][s]
                 else np.nan for s in subs]
        bc_v = [out["B_bias_correction"][s][lab]["score_bc"] if lab in out["B_bias_correction"][s]
                else np.nan for s in subs]
        all_raw += [v for v in raw_v if v == v]
        all_bc += [v for v in bc_v if v == v]
        ax = axes[1]
        ax.scatter(raw_v, bc_v, s=30, label=lab)
    lim = max(all_raw + all_bc) * 1.05
    axes[1].plot([0, lim], [0, lim], "k--", lw=0.7)
    axes[1].set_xlim(0, lim); axes[1].set_ylim(0, lim)
    axes[1].set_xlabel("校正前 score"); axes[1].set_ylabel("校正后 score")
    axes[1].set_title("B 偏差校正：对角线下=改善")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "fig21_score_aware.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("图 → results/figures/fig21_score_aware.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
