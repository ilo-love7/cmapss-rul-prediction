"""15 · v1.3 汇总表：三实验关键指标 markdown。

读 12/13/14 的 JSON，产出 results/v13_summary_tables.md（报告 §3.13–3.15 的表格源）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    d12 = json.loads((ROOT / "results" / "metrics" / "v13_two_stage_intervals.json")
                     .read_text(encoding="utf-8"))
    d13 = json.loads((ROOT / "results" / "metrics" / "v13_domain_adaptation.json")
                     .read_text(encoding="utf-8"))
    d14 = json.loads((ROOT / "results" / "metrics" / "v13_online_gating.json")
                     .read_text(encoding="utf-8"))

    lines = ["# N2 v1.3 汇总表（E14–E16）", ""]

    # ---- E14 区间 ----
    lines += ["## E14 pinball + 共形二段缩放（90% 区间）", "",
              "| 子集 | pinball原始 | 共形-对称 | CQR加成 | 缩放-对称 | 缩放-非对称 | 分层共形 |",
              "|---|---|---|---|---|---|---|"]
    for s, e in d12["results"].items():
        cells = []
        for m in ("pinball_raw", "conformal_sym", "cqr_add", "scaled_sym",
                  "scaled_asym", "mondrian"):
            iv = e["methods"][m]["interval"]
            cells.append(f"{iv['coverage']:.3f} / {iv['mean_width']:.1f}")
        lines.append("| " + s + " | " + " | ".join(cells) + " |")
    lines += ["", "（单元格 = 覆盖率 / 平均宽度 cycles；目标 0.90±0.05。）", ""]

    # ---- E15 域适应 ----
    lines += ["## E15 域适应（目标域 RMSE / 迁移缺口）", "",
              "| 迁移对 | plain | coral | mmd | fewshot | oracle | v1.1参照 |",
              "|---|---|---|---|---|---|---|"]
    for k, e in d13["results"].items():
        ms = e["methods"]
        cells = [f"{ms[m]['rmse']:.2f}" for m in ("plain", "coral", "mmd", "fewshot")]
        lines.append("| " + k + " | " + " | ".join(cells)
                     + f" | {e['oracle']['rmse']:.2f} | {e['v11_plain_ref']['rmse']:.2f} |")
    lines += ["", "缺口（vs oracle RMSE）：", "",
              "| 迁移对 | plain | coral | mmd | fewshot |",
              "|---|---|---|---|---|"]
    for k, e in d13["results"].items():
        g = e["gap_rmse"]
        lines.append(f"| {k} | {g['plain']:+.2f} | {g['coral']:+.2f} | "
                     f"{g['mmd']:+.2f} | {g['fewshot']:+.2f} |")
    lines += [""]

    # ---- E16 在线 ----
    lines += ["## E16 在线后处理（未封顶真值 RMSE）与门控", "",
              "| 子集 | 原始 | 单调修复 | 自洽重标定 | 两者叠加 | 违反率raw | 违反率both |",
              "|---|---|---|---|---|---|---|"]
    for s, e in d14["results"].items():
        pp = e["postprocess"]
        cells = [f"{pp[m]['rmse_raw_truth']:.2f}"
                 for m in ("raw", "monotone", "recal", "both")]
        cv = e["constraint_violation"]
        lines.append("| " + s + " | " + " | ".join(cells)
                     + f" | {cv['raw']:.3f} | {cv['both']:.3f} |")
    lines += ["", "门控（风险-覆盖 AURC / ρ(score,|err|)）：", "",
              "| 子集 | 集成std AURC | 集成std ρ | pinball宽度 AURC | pinball宽度 ρ |",
              "|---|---|---|---|---|"]
    for s, e in d14["results"].items():
        g = e["gating"]
        lines.append(f"| {s} | {g['ensemble_std']['aurc']:.2f} | "
                     f"{g['ensemble_std']['spearman']:+.3f} | "
                     f"{g['pinball_width']['aurc']:.2f} | "
                     f"{g['pinball_width']['spearman']:+.3f} |")
    lines += [""]

    out = ROOT / "results" / "v13_summary_tables.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("".join(l + "\n" for l in lines))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
