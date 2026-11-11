"""12 · pinball + 共形二段缩放：补上 v1.2 欠覆盖的方法族。

回应报告 §3.12/§5.8（两处点名未做）：pinball [q05,q95] 最窄但 FD003 覆盖 0.810，
文献解法是"pinball + 共形缩放"二段式。本脚本在同一基础模型上比较六种 90% 区间：

  A pinball_raw    v1.2 原始 [q05, q95]
  B conformal_sym  v1.2 共形-对称（点模型）参照
  C cqr_add        CQR 加成式：区间整体膨胀/收缩 Q（Romano 2019）
  D scaled_sym     乘性缩放：残差/半宽，单因子
  E scaled_asym    乘性缩放：左右半宽各一因子
  F mondrian       分层共形：按预测 RUL 桶分层取分位数（协变量自适应）

pinball 复用 11 存的检查点（无需重训）；基础点模型用 09 的 phm08_score 检查点。
产出
  results/metrics/v13_two_stage_intervals.json
  results/figures/fig24_two_stage_intervals.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss_rul import load_all, load_config, prepare_subset, save_json  # noqa: E402
from cmapss_rul.evaluation import conformal_intervals, interval_coverage  # noqa: E402
from cmapss_rul.intervals import (cqr_interval, mondrian_interval,  # noqa: E402
                                  scaled_interval)
from cmapss_rul.models import build_model, predict  # noqa: E402
from cmapss_rul.preprocessing import train_val_split_units  # noqa: E402

MODELS = ROOT / "results" / "models"
NOMINAL = 0.90
ALPHA = 1 - NOMINAL
# 预测 RUL 分桶（Mondrian 层）——绝对边界，校准/测试一致
STRATA_EDGES = (0.0, 40.0, 85.0, 126.0)
STRATA_LABELS = ("0-40", "40-85", "85+")


def to_strata(pred: np.ndarray) -> np.ndarray:
    p = np.asarray(pred, dtype=np.float64)
    return np.digitize(p, STRATA_EDGES[1:-1], right=False)   # 0,1,2


def load_ckpt(kind: str, name: str, n_features: int, cfg: dict):
    h = cfg["lstm"]
    if kind == "pinball":
        model = build_model("rnn", n_features, hidden=int(h["hidden"]),
                            layers=int(h["layers"]), dropout=float(h["dropout"]),
                            cell="lstm", out_dim=3)
        path = MODELS / f"pinball_{name}_s0.pt"
    else:
        model = build_model("rnn", n_features, hidden=int(h["hidden"]),
                            layers=int(h["layers"]), dropout=float(h["dropout"]),
                            cell="lstm", out_dim=1)
        path = MODELS / f"scoreaware_phm08_score_{name}_s0.pt"
    if not path.exists():
        raise FileNotFoundError(f"缺 {path.name}，先运行 09/11")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def main() -> int:
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    data = load_all(raw)
    cap = float(cfg["rul_cap"])
    out: dict = {"nominal": NOMINAL, "strata_edges": list(STRATA_EDGES), "results": {}}

    for name in cfg["subsets"]:
        t0 = time.time()
        d = prepare_subset(data[name].train, data[name].test, data[name].rul,
                           window=cfg["window"], rul_cap=cap,
                           std_min=cfg["sensor_std_min"],
                           n_conditions=cfg["n_conditions"],
                           cluster=cfg["cluster_op_settings"])
        tr_m, va_m = train_val_split_units(d["meta"], val_frac=cfg["val_frac_units"], seed=0)
        y_val, y_test = d["y"][va_m], d["y_test"]
        n_feat = d["X"].shape[2]
        print(f"\n===== {name} =====")

        # pinball 3 头
        pm = load_ckpt("pinball", name, n_feat, cfg)
        q_val = np.clip(predict(pm, d["X"][va_m]), 0.0, cap)
        q_test = np.clip(predict(pm, d["X_test"]), 0.0, cap)
        q05v, q50v, q95v = q_val[:, 0], q_val[:, 1], q_val[:, 2]
        q05t, q50t, q95t = q_test[:, 0], q_test[:, 1], q_test[:, 2]

        # 基础点模型（09 phm08_score）
        bm = load_ckpt("base", name, n_feat, cfg)
        pred_val = np.clip(predict(bm, d["X"][va_m]), 0.0, cap)
        pred_test = np.clip(predict(bm, d["X_test"]), 0.0, cap)

        methods: dict = {}

        # A pinball raw
        methods["pinball_raw"] = {"interval": interval_coverage(y_test, q05t, q95t),
                                  "param": {}}
        # B conformal_sym（点模型）
        lo, hi, q = conformal_intervals(y_val, pred_val, pred_test, alpha=ALPHA)
        methods["conformal_sym"] = {"interval": interval_coverage(y_test, lo, hi),
                                    "param": {"q": q}}
        # C CQR 加成
        lo, hi, q = cqr_interval(y_val, q05v, q95v, q05t, q95t, alpha=ALPHA)
        methods["cqr_add"] = {"interval": interval_coverage(y_test, lo, hi),
                              "param": {"Q": q}}
        # D 乘性对称
        lo, hi, p = scaled_interval(y_val, q05v, q95v, q05t, q95t, alpha=ALPHA,
                                    asymmetric=False)
        methods["scaled_sym"] = {"interval": interval_coverage(y_test, lo, hi), "param": p}
        # E 乘性非对称
        lo, hi, p = scaled_interval(y_val, q05v, q95v, q05t, q95t, alpha=ALPHA,
                                    asymmetric=True)
        methods["scaled_asym"] = {"interval": interval_coverage(y_test, lo, hi), "param": p}
        # F Mondrian 分层（点模型残差，预测 RUL 分桶）
        lo, hi, qmap = mondrian_interval(y_val, pred_val, pred_test,
                                         to_strata(pred_val), to_strata(pred_test),
                                         alpha=ALPHA)
        m_iv = interval_coverage(y_test, lo, hi)
        # 逐层覆盖诊断
        st_test = to_strata(pred_test)
        per_strata = {}
        for s, lab in enumerate(STRATA_LABELS):
            m = st_test == s
            if m.any():
                inside = (y_test[m] >= lo[m]) & (y_test[m] <= hi[m])
                per_strata[lab] = {"coverage": float(inside.mean()), "n": int(m.sum())}
        methods["mondrian"] = {"interval": m_iv, "param": {"q_map": qmap},
                               "per_strata": per_strata}

        # pinball 宽度分布（诊断：欠覆盖是否集中在窄区间点）
        pw = q95t - q05t
        inside_pb = (y_test >= q05t) & (y_test <= q95t)
        methods["pinball_raw"]["width_by_cover"] = {
            "covered_width_mean": float(pw[inside_pb].mean()) if inside_pb.any() else None,
            "missed_width_mean": float(pw[~inside_pb].mean()) if (~inside_pb).any() else None,
            "miss_frac": float((~inside_pb).mean()),
        }

        for k, m in methods.items():
            iv = m["interval"]
            print(f"  {k:14s} cov={iv['coverage']:.3f} width={iv['mean_width']:.1f}")
        out["results"][name] = {"n_test": int(len(y_test)), "methods": methods}
        save_json(out, ROOT / "results" / "metrics" / "v13_two_stage_intervals.json")
        print(f"  ({time.time() - t0:.0f}s)")

    # 图：覆盖率 vs 宽度
    import matplotlib.pyplot as plt
    from cmapss_rul.plots import setup_style
    setup_style()
    labels = ["pinball_raw", "conformal_sym", "cqr_add", "scaled_sym",
              "scaled_asym", "mondrian"]
    pretty = ["pinball原始", "共形-对称", "CQR加成", "缩放-对称", "缩放-非对称", "分层共形"]
    subs = list(out["results"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(subs))
    width = 0.13
    for k, (lab, ps) in enumerate(zip(labels, pretty)):
        covs = [out["results"][s]["methods"][lab]["interval"]["coverage"] for s in subs]
        axes[0].bar(x + (k - 2.5) * width, covs, width, label=ps)
        wds = [out["results"][s]["methods"][lab]["interval"]["mean_width"] for s in subs]
        axes[1].bar(x + (k - 2.5) * width, wds, width, label=ps)
    axes[0].axhline(NOMINAL, color="k", ls="--", lw=0.8)
    axes[0].axhline(0.85, color="r", ls=":", lw=0.6)
    axes[0].axhline(0.95, color="r", ls=":", lw=0.6)
    axes[0].set_xticks(x); axes[0].set_xticklabels(subs)
    axes[0].set_ylim(0.5, 1.02); axes[0].set_ylabel("90% 区间覆盖率")
    axes[0].set_title("覆盖率（虚线=名义 0.90，点线=±0.05）")
    axes[0].legend(fontsize=6.5, ncol=2)
    axes[1].set_xticks(x); axes[1].set_xticklabels(subs)
    axes[1].set_ylabel("平均宽度（cycles）")
    axes[1].set_title("区间宽度（前提：覆盖达标）")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "fig24_two_stage_intervals.png",
                dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("图 → results/figures/fig24_two_stage_intervals.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
