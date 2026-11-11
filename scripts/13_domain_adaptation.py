"""13 · 跨子集域适应：CORAL / MMD 对齐 + few-shot 微调 vs 零样本迁移。

回应报告 §5.3（迁移缺口 +8.2~+20.8 RMSE 已量化但未缩小）。协议与 08 一致：
源域三件套（传感器/工况/归一化）只用源域数据，目标域仅做变换；在 v1.1 的三对
迁移上比较四种缩小缺口的手段（同 harness 重跑 plain 作内部基线）：

  plain    零样本（= v1.1 §3.9 协议，同 harness 重跑）
  coral    训练时 CORAL 对齐源/目标隐层协方差（无标签目标训练窗）
  mmd      训练时 RBF-MMD 对齐隐层分布
  fewshot  plain 后用 k 台目标域训练引擎（k=5）标签微调

口径：四方法统一走 train_da（MSE 任务损失 + val RMSE 早停），保证内部可比；
v1.1 的 phm08 损失迁移数字作外部参照（v11_plain_ref）。
参照：oracle = 目标域自训练（复用 v1.1 JSON）。
产出
  results/metrics/v13_domain_adaptation.json
  results/figures/fig25_domain_adaptation.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss_rul import evaluate, load_all, load_config, save_json  # noqa: E402
from cmapss_rul.adaptation import few_shot_finetune, train_da  # noqa: E402
from cmapss_rul.models import predict  # noqa: E402
from cmapss_rul.plots import grouped_bars, setup_style  # noqa: E402
from cmapss_rul.preprocessing import (  # noqa: E402
    assign_conditions, build_sequences, fit_conditions, fit_scaler,
    op_settings, piecewise_rul, select_sensors, sensor_matrix,
    train_val_split_units,
)
from cmapss_rul.io import iter_units  # noqa: E402

PAIRS = [("FD001", "FD003"), ("FD002", "FD004"), ("FD001", "FD002")]
K_FEWSHOT = 5
SEED = 0


def prepare(train, test, rul, cfg, keep=None, km=None, scaler=None):
    """与 08 完全一致的迁移预处理（源域拟合 / 目标域变换）。"""
    tr_s = sensor_matrix(train)
    te_s = sensor_matrix(test)
    if keep is None:
        keep = select_sensors(tr_s, std_min=float(cfg["sensor_std_min"]))
    if km is None:
        km, tr_lab = fit_conditions(op_settings(train), n_clusters=int(cfg["n_conditions"]), seed=0)
        n_clusters = max(int(cfg["n_conditions"]), 1) if km is not None else 1
        scaler = fit_scaler(tr_s[:, keep], tr_lab, n_clusters=n_clusters)
    else:
        tr_lab = assign_conditions(km, op_settings(train))
    te_lab = assign_conditions(km, op_settings(test)) if km is not None else np.zeros(len(test), dtype=int)
    Xtr_s = scaler.transform(tr_s[:, keep], tr_lab)
    Xte_s = scaler.transform(te_s[:, keep], te_lab)

    cap = float(cfg["rul_cap"])
    ytr = np.zeros(len(train))
    for u, rows in iter_units(train):
        idx = np.where(train[:, 0] == u)[0]
        ytr[idx] = piecewise_rul(rows[:, 1], cap=cap)
    X, y, meta = build_sequences(Xtr_s, ytr, train[:, 0], window=int(cfg["window"]))

    window = int(cfg["window"])
    Xs = []
    for u in np.unique(test[:, 0]).astype(int):
        S = Xte_s[test[:, 0] == u]
        if S.shape[0] >= window:
            Xs.append(S[-window:])
        else:
            pad = np.zeros((window - S.shape[0], S.shape[1]))
            Xs.append(np.vstack([pad, S]))
    y_true = np.minimum(np.asarray(rul, dtype=np.float64).reshape(-1), cap)
    return {"X": X, "y": y, "meta": meta, "X_test": np.asarray(Xs), "y_test": y_true}


def run_method(d_src, cfg, method: str, d_tgt_train=None, X_val=None, y_val=None):
    """四种方法统一走 train_da（同早停/同超参），fewshot 额外微调。"""
    h = cfg["lstm"]
    t0 = time.time()
    align = {"plain": "none", "coral": "coral", "mmd": "mmd", "fewshot": "none"}[method]
    res = train_da(
        d_src["X"], d_src["y"],
        X_tgt_unlab=(d_tgt_train["X"] if align != "none" else None),
        align=align, align_weight=1.0,
        X_val=X_val, y_val=y_val,
        hidden=int(h["hidden"]), layers=int(h["layers"]), dropout=float(h["dropout"]),
        lr=float(h["lr"]), weight_decay=float(h["weight_decay"]),
        batch=int(h["batch"]), epochs=int(h["epochs"]), patience=int(h["patience"]),
        seed=SEED, verbose=False,
    )
    model = res.model
    if method == "fewshot":
        # k 台目标引擎的全部窗口（目标域标注，仅 fewshot 可用）
        meta = d_tgt_train["meta"]
        units = np.unique(meta[:, 0])[:K_FEWSHOT]
        m = np.isin(meta[:, 0], units)
        model = few_shot_finetune(model, d_tgt_train["X"][m], d_tgt_train["y"][m],
                                  epochs=20, lr=5e-4, batch=64, seed=SEED)
    return model, round(time.time() - t0, 1)


def main() -> int:
    setup_style()
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    data = load_all(raw)
    cap = float(cfg["rul_cap"])
    v11 = json.loads((ROOT / "results" / "metrics" / "cross_transfer.json").read_text(
        encoding="utf-8"))["results"]
    out: dict = {"k_fewshot": K_FEWSHOT, "seed": SEED, "results": {}}

    for src, tgt in PAIRS:
        key = f"{src}->{tgt}"
        print(f"\n===== {key} =====")
        d_src = prepare(data[src].train, data[src].test, data[src].rul, cfg)
        # 目标域用源域三件套变换：训练窗（对齐/微调用）+ 测试末窗（评估用）
        d_tgt_xf = prepare(data[tgt].train, data[tgt].test, data[tgt].rul, cfg,
                           **_src_kit(data[src].train, cfg))
        tr_m, va_m = train_val_split_units(d_src["meta"], val_frac=float(cfg["val_frac_units"]),
                                           seed=0)
        entry: dict = {"source": src, "target": tgt, "methods": {}}

        for method in ("plain", "coral", "mmd", "fewshot"):
            model, train_s = run_method(
                {"X": d_src["X"][tr_m], "y": d_src["y"][tr_m], "meta": d_src["meta"][tr_m]},
                cfg, method,
                d_tgt_train=d_tgt_xf,
                X_val=d_src["X"][va_m], y_val=d_src["y"][va_m],
            )
            pred = np.clip(predict(model, d_tgt_xf["X_test"]), 0.0, cap)
            m = evaluate(d_tgt_xf["y_test"], pred)
            entry["methods"][method] = {**m, "train_s": train_s}
            print(f"  {method:7s} rmse={m['rmse']:.2f} phm08={m['phm08']:.1f} "
                  f"bias={m['mean_error']:+.2f} ({train_s}s)")

        entry["v11_plain_ref"] = v11[key]["transfer"]
        entry["oracle"] = v11[key]["target_oracle"]
        entry["gap_rmse"] = {
            m: round(entry["methods"][m]["rmse"] - entry["oracle"]["rmse"], 2)
            for m in entry["methods"]}
        out["results"][key] = entry
        save_json(out, ROOT / "results" / "metrics" / "v13_domain_adaptation.json")
        print(f"  oracle rmse={entry['oracle']['rmse']:.2f} | "
              f"缺口 plain={entry['gap_rmse']['plain']:+.2f} "
              f"coral={entry['gap_rmse']['coral']:+.2f} "
              f"mmd={entry['gap_rmse']['mmd']:+.2f} "
              f"fewshot={entry['gap_rmse']['fewshot']:+.2f}")

    # 图：各方法目标域 RMSE（横线=oracle）
    groups = {k: {m: v["methods"][m]["rmse"] for m in v["methods"]}
              for k, v in out["results"].items()}
    grouped_bars(groups, "域适应：目标域 RMSE（越低越好）", "RMSE",
                 ROOT / "results" / "figures" / "fig25_domain_adaptation.png")
    print("图 → results/figures/fig25_domain_adaptation.png")
    return 0


def _src_kit(train_src, cfg):
    """源域三件套拟合（供目标域变换）。与 prepare(keep=None...) 等价，显式化。"""
    tr_s = sensor_matrix(train_src)
    keep = select_sensors(tr_s, std_min=float(cfg["sensor_std_min"]))
    km, tr_lab = fit_conditions(op_settings(train_src),
                                n_clusters=int(cfg["n_conditions"]), seed=0)
    n_clusters = max(int(cfg["n_conditions"]), 1) if km is not None else 1
    scaler = fit_scaler(tr_s[:, keep], tr_lab, n_clusters=n_clusters)
    return {"keep": keep, "km": km, "scaler": scaler}


if __name__ == "__main__":
    raise SystemExit(main())
