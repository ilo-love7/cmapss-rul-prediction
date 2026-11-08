"""08 跨子集迁移：源域训练 → 目标域测试，零目标样本。

回应报告 §5.3（未覆盖跨子集迁移）。设计：
- FD001→FD003（单工况↔单工况，故障模式部分重叠）
- FD002→FD004（六工况↔六工况）
- FD001→FD002（单→六工况，最难）
协议：传感器筛选/工况聚类/归一化统计全部只用源域；目标域仅做末窗预测。
对照：同协议源域自测（上界）与目标域自训练（oracle）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmapss_rul import evaluate, load_all, load_config, save_json
from cmapss_rul.io import iter_units
from cmapss_rul.models import predict, train_regressor
from cmapss_rul.plots import grouped_bars, setup_style
from cmapss_rul.preprocessing import (
    assign_conditions, build_sequences, fit_conditions, fit_scaler,
    op_settings, piecewise_rul, select_sensors, sensor_matrix, train_val_split_units,
)

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [("FD001", "FD003"), ("FD002", "FD004"), ("FD001", "FD002")]


def prepare(train, test, rul, cfg, keep=None, km=None, scaler=None):
    """源域：keep/km/scaler=None 时拟合；目标域：传入源域三件套直接变换。"""
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
    return {"X": X, "y": y, "meta": meta, "X_test": np.asarray(Xs), "y_test": y_true,
            "keep": keep, "km": km, "scaler": scaler}


def train_on(d, cfg, seed=0):
    tr_m, va_m = train_val_split_units(d["meta"], val_frac=float(cfg["val_frac_units"]), seed=0)
    h = cfg["lstm"]
    t0 = time.time()
    res = train_regressor(
        d["X"][tr_m], d["y"][tr_m], d["X"][va_m], d["y"][va_m],
        cell="lstm", loss_name="phm08",
        hidden=int(h["hidden"]), layers=int(h["layers"]), dropout=float(h["dropout"]),
        lr=float(h["lr"]), weight_decay=float(h["weight_decay"]),
        batch=int(h["batch"]), epochs=int(h["epochs"]), patience=int(h["patience"]),
        seed=seed, verbose=False,
    )
    return res, round(time.time() - t0, 1)


def main() -> None:
    setup_style()
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    data = load_all(raw)
    out: dict = {}

    for src, tgt in PAIRS:
        print(f"\n===== {src} → {tgt} =====")
        t0 = time.time()
        # 源域三件套（传感器/工况/归一化）只用源域数据
        d_src = prepare(data[src].train, data[src].test, data[src].rul, cfg)
        # 目标域：只做变换与末窗
        d_tgt = prepare(data[tgt].train, data[tgt].test, data[tgt].rul, cfg,
                        keep=d_src["keep"], km=d_src["km"], scaler=d_src["scaler"])

        res, train_s = train_on(d_src, cfg)
        pred_src = np.clip(predict(res.model, d_src["X_test"]), 0.0, float(cfg["rul_cap"]))
        pred_tgt = np.clip(predict(res.model, d_tgt["X_test"]), 0.0, float(cfg["rul_cap"]))
        m_src = evaluate(d_src["y_test"], pred_src)
        m_tgt = evaluate(d_tgt["y_test"], pred_tgt)

        # oracle：目标域自训练（参照上界，不是迁移）
        d_tgt_full = prepare(data[tgt].train, data[tgt].test, data[tgt].rul, cfg)
        res_o, train_s_o = train_on(d_tgt_full, cfg)
        pred_o = np.clip(predict(res_o.model, d_tgt_full["X_test"]), 0.0, float(cfg["rul_cap"]))
        m_o = evaluate(d_tgt_full["y_test"], pred_o)

        entry = {
            "source": src, "target": tgt,
            "n_keep_sensors": int(len(d_src["keep"])),
            "train_s": train_s,
            "source_self": m_src, "transfer": m_tgt, "target_oracle": m_o,
            "transfer_gap_rmse": round(m_tgt["rmse"] - m_o["rmse"], 2),
            "oracle_train_s": train_s_o,
        }
        out[f"{src}->{tgt}"] = entry
        print(f"  传感器 {len(d_src['keep'])} 个 | 训练 {train_s}s")
        print(f"  源域自测   rmse={m_src['rmse']:.2f} phm08={m_src['phm08']:.1f} bias={m_src['mean_error']:+.2f}")
        print(f"  迁移到目标 rmse={m_tgt['rmse']:.2f} phm08={m_tgt['phm08']:.1f} bias={m_tgt['mean_error']:+.2f}")
        print(f"  目标域自训 rmse={m_o['rmse']:.2f} phm08={m_o['phm08']:.1f} bias={m_o['mean_error']:+.2f}")
        print(f"  迁移缺口（transfer − oracle）rmse: {entry['transfer_gap_rmse']:+.2f}  ({time.time()-t0:.0f}s)")

    # 图：迁移 vs oracle vs 源域自测
    groups = {k: {"源域自测": v["source_self"]["rmse"],
                  "迁移到目标": v["transfer"]["rmse"],
                  "目标域自训": v["target_oracle"]["rmse"]} for k, v in out.items()}
    grouped_bars(groups, "跨子集迁移：RMSE（迁移 vs oracle）", "RMSE", ROOT / "results" / "figures" / "fig14_transfer_rmse.png")
    groups_p = {k: {"源域自测": v["source_self"]["phm08"],
                    "迁移到目标": v["transfer"]["phm08"],
                    "目标域自训": v["target_oracle"]["phm08"]} for k, v in out.items()}
    grouped_bars(groups_p, "跨子集迁移：PHM08 score", "score", ROOT / "results" / "figures" / "fig14_transfer_phm08.png")

    save_json({"results": out}, ROOT / "results" / "metrics" / "cross_transfer.json")
    print("\ncross_transfer.json saved")


if __name__ == "__main__":
    main()
