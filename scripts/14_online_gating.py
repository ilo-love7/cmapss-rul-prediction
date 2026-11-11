"""14 · 在线门控与重标定：滚动预测的在线化后处理。

回应报告 §5.4（滚动预测是"离线全轨迹评估"，不是在线系统）。在 v1.1 全寿命
滚动预测（§3.7）之上加三件套：

  门控       集成 std / pinball 宽度作不确定性分数 → 风险-覆盖曲线 + AURC
  单调修复   piecewise RUL 物理约束 −1 ≤ r_t − r_{t−1} ≤ 0 的因果前向裁剪
  自洽重标定 pred_t + t ≡ T → 滑窗中位数估 T̂（去漂移/野值；恒定偏置无效，如实记录）

点预测 = LSTM_MSE/LSTM_PHM08/GRU_MSE 集成均值（§3.8 约定）；pinball 只出宽度分数。
产出
  results/metrics/v13_online_gating.json
  results/figures/fig26_online_gating.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss_rul import evaluate, load_all, load_config, save_json  # noqa: E402
from cmapss_rul.models import SeqRegressor, build_model, predict  # noqa: E402
from cmapss_rul.online import (  # noqa: E402
    causal_monotone_repair, coverage_at_risk, risk_at_coverage,
    risk_coverage, self_consistent_recal,
)
from cmapss_rul.plots import setup_style  # noqa: E402
from cmapss_rul.preprocessing import (  # noqa: E402
    assign_conditions, fit_conditions, fit_scaler, op_settings,
    select_sensors, sensor_matrix,
)

FIG = ROOT / "results" / "figures"
MODELS = ROOT / "results" / "models"
STEMS = [("lstm_mse", "LSTM_MSE"), ("lstm_phm08", "LSTM_PHM08"), ("gru_mse", "GRU_MSE")]


def prepare_source(train: np.ndarray, cfg: dict):
    tr_s = sensor_matrix(train)
    keep = select_sensors(tr_s, std_min=float(cfg["sensor_std_min"]))
    km, tr_lab = fit_conditions(op_settings(train), n_clusters=int(cfg["n_conditions"]), seed=0)
    n_clusters = max(int(cfg["n_conditions"]), 1) if km is not None else 1
    scaler = fit_scaler(tr_s[:, keep], tr_lab, n_clusters=n_clusters)
    Xtr_s = scaler.transform(tr_s[:, keep], tr_lab)
    return keep, km, scaler, Xtr_s


def transform_target(test: np.ndarray, keep, km, scaler) -> np.ndarray:
    te_lab = assign_conditions(km, op_settings(test)) if km is not None else np.zeros(len(test), dtype=int)
    return scaler.transform(sensor_matrix(test)[:, keep], te_lab)


def rolling_windows(S: np.ndarray, window: int) -> np.ndarray:
    n = S.shape[0]
    return np.stack([S[t - window + 1:t + 1] for t in range(window - 1, n)])


def load_model(stem: str, n_features: int, subset: str, cfg: dict) -> SeqRegressor:
    cell = "lstm" if stem.startswith("lstm") else "gru"
    h = cfg["lstm"]
    model = SeqRegressor(n_features, hidden=int(h["hidden"]), layers=int(h["layers"]),
                         dropout=float(h["dropout"]), cell=cell)
    model.load_state_dict(torch.load(MODELS / f"{stem}_{subset}.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_pinball(n_features: int, subset: str, cfg: dict):
    h = cfg["lstm"]
    model = build_model("rnn", n_features, hidden=int(h["hidden"]),
                        layers=int(h["layers"]), dropout=float(h["dropout"]),
                        cell="lstm", out_dim=3)
    model.load_state_dict(torch.load(MODELS / f"pinball_{subset}_s0.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()
    return model


def postprocess_variants(pred: np.ndarray, cycles: np.ndarray, cap: float) -> dict:
    """单引擎因果后处理四变体。"""
    mono = causal_monotone_repair(pred, cap=cap)
    recal = self_consistent_recal(pred, cycles, window=20, cap=cap)
    both = causal_monotone_repair(
        self_consistent_recal(pred, cycles, window=20, cap=cap), cap=cap)
    return {"raw": pred, "monotone": mono, "recal": recal, "both": both}


def main() -> int:
    setup_style()
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    data = load_all(raw)
    cap = float(cfg["rul_cap"])
    window = int(cfg["window"])
    out: dict = {"results": {}}

    for name in cfg["subsets"]:
        t0 = time.time()
        train, test, rul = data[name].train, data[name].test, data[name].rul
        keep, km, scaler, _ = prepare_source(train, cfg)
        Xte_s = transform_target(test, keep, km, scaler)
        rul_end = np.asarray(rul, dtype=np.float64).reshape(-1)
        test_units = np.unique(test[:, 0]).astype(int)

        models = {k: load_model(stem, len(keep), name, cfg) for stem, k in STEMS}
        pb = load_pinball(len(keep), name, cfg)

        # 按引擎收集滚动轨迹
        unit_store = []       # (true_raw, cycles, mean_pred, std, width)
        viol_raw_u, viol_both_u = [], []
        for i, u in enumerate(test_units):
            S = Xte_s[test[:, 0] == u]
            n = S.shape[0]
            if n < window:
                continue
            Xw = rolling_windows(S, window)
            ts = np.arange(window - 1, n)                 # 每窗末周期
            true_raw = rul_end[i] + (n - 1 - ts)          # 未封顶真值
            preds3 = np.stack([np.clip(predict(m, Xw), 0.0, cap)
                               for m in models.values()])  # (3, T)
            mean_p = preds3.mean(axis=0)
            std_p = preds3.std(axis=0)
            q = np.clip(predict(pb, Xw), 0.0, cap)
            width = q[:, 2] - q[:, 0]
            unit_store.append((true_raw, ts.astype(float), mean_p, std_p, width))
            # 引擎内约束违反率（−1 ≤ Δr ≤ 0）
            viol_raw_u.append(_viol_frac(mean_p))
            viol_both_u.append(_viol_frac(
                causal_monotone_repair(
                    self_consistent_recal(mean_p, ts.astype(float), window=20, cap=cap),
                    cap=cap)))

        # 后处理（按引擎因果）+ 拼接
        variants = {k: [] for k in ("raw", "monotone", "recal", "both")}
        truths, stds, widths = [], [], []
        for true_raw, ts, mean_p, std_p, width in unit_store:
            v = postprocess_variants(mean_p, ts, cap)
            for k in variants:
                variants[k].append(v[k])
            truths.append(true_raw)
            stds.append(std_p)
            widths.append(width)
        truth = np.concatenate(truths)
        std_all = np.concatenate(stds)
        width_all = np.concatenate(widths)
        var_all = {k: np.concatenate(v) for k, v in variants.items()}

        entry = {"n_units": len(unit_store), "n_windows": int(truth.size), "postprocess": {}}
        for k, p in var_all.items():
            m = evaluate(truth, p)                        # 未封顶真值
            m_cap = evaluate(np.minimum(truth, cap), p)
            entry["postprocess"][k] = {
                "rmse_raw_truth": m["rmse"], "phm08_raw": m["phm08"],
                "mean_error": m["mean_error"],
                "rmse_capped_truth": m_cap["rmse"],
            }
            print(f"  {name} {k:9s} rmse_raw={m['rmse']:.2f} score={m['phm08']:.0f} "
                  f"bias={m['mean_error']:+.2f}")

        # 门控：两种不确定性分数的风险-覆盖
        gates = {}
        for score_name, sc in (("ensemble_std", std_all), ("pinball_width", width_all)):
            rc = risk_coverage(truth, var_all["raw"], sc)
            gates[score_name] = {
                "aurc": rc["aurc"],
                "risk_at_cov80": risk_at_coverage(rc, 0.8),
                "risk_at_cov90": risk_at_coverage(rc, 0.9),
                "cov_at_risk15": coverage_at_risk(rc, 15.0),
                "spearman": float(_spearman(sc, np.abs(var_all["raw"] - truth))),
            }
            print(f"  {name} gate[{score_name}] aurc={rc['aurc']:.2f} "
                  f"risk@80%={gates[score_name]['risk_at_cov80']:.2f} "
                  f"ρ={gates[score_name]['spearman']:+.3f}")
        entry["gating"] = gates

        # 修复幅度诊断：违反 −1≤Δr≤0 的窗格比例（引擎内，raw vs both）
        viol_raw = float(np.mean(viol_raw_u))
        viol_both = float(np.mean(viol_both_u))
        entry["constraint_violation"] = {"raw": viol_raw, "both": viol_both}
        print(f"  viol raw={viol_raw:.3f} both={viol_both:.3f} ({time.time() - t0:.0f}s)")

        out["results"][name] = entry
        save_json(out, ROOT / "results" / "metrics" / "v13_online_gating.json")

    # 图：后处理 RMSE 对比 + 风险-覆盖示意（FD001）
    import matplotlib.pyplot as plt
    subs = list(out["results"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    keys = ["raw", "monotone", "recal", "both"]
    pretty = ["原始", "单调修复", "自洽重标定", "两者叠加"]
    x = np.arange(len(subs))
    width = 0.2
    for k, (lab, ps) in enumerate(zip(keys, pretty)):
        vals = [out["results"][s]["postprocess"][lab]["rmse_raw_truth"] for s in subs]
        axes[0].bar(x + (k - 1.5) * width, vals, width, label=ps)
    axes[0].set_xticks(x); axes[0].set_xticklabels(subs)
    axes[0].set_ylabel("RMSE（未封顶真值）")
    axes[0].set_title("在线后处理效果")
    axes[0].legend(fontsize=8)

    # 门控指标条形展示（逐子集 AURC）
    gm = ["ensemble_std", "pinball_width"]
    labels_g = ["集成std", "pinball宽度"]
    xg = np.arange(len(subs))
    for j, (g, lab) in enumerate(zip(gm, labels_g)):
        vals = [out["results"][s]["gating"][g]["aurc"] for s in subs]
        axes[1].bar(xg + (j - 0.5) * 0.35, vals, 0.35, label=lab)
    axes[1].set_xticks(xg); axes[1].set_xticklabels(subs)
    axes[1].set_ylabel("AURC（越小越好）")
    axes[1].set_title("门控分数质量：风险-覆盖曲线下面积")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig26_online_gating.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("图 → results/figures/fig26_online_gating.png")
    return 0


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def _viol_frac(p: np.ndarray) -> float:
    """单引擎预测序列违反 −1 ≤ Δr ≤ 0 的相邻对比例。"""
    d = np.diff(np.asarray(p, dtype=np.float64))
    if d.size == 0:
        return 0.0
    bad = (d > 1e-9) | (d < -1.0 - 1e-9)
    return float(bad.mean())


if __name__ == "__main__":
    raise SystemExit(main())
