"""00 数据校验：确认 CMAPSS 12 个 txt 存在、形状与 RUL 对齐。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmapss_rul import load_config, load_all, save_json


def main() -> None:
    cfg = load_config()
    raw = Path(cfg["data_root"]) / cfg["raw_subdir"]
    files = []
    for fd in ("FD001", "FD002", "FD003", "FD004"):
        for kind in ("train", "test", "RUL"):
            p = raw / f"{kind}_{fd}.txt"
            assert p.exists(), f"缺少 {p}"
            files.append(p.name)
    data = load_all(raw)
    summary = {}
    for name, s in data.items():
        summary[name] = {
            "train_shape": list(s.train.shape),
            "test_shape": list(s.test.shape),
            "n_train_units": len(s.train_units),
            "n_test_units": len(s.test_units),
            "rul_rows": int(s.rul.size),
            "rul_max": float(s.rul.max()),
        }
        print(f"{name}: train {s.train.shape} ({len(s.train_units)} 引擎) | "
              f"test {s.test.shape} ({len(s.test_units)} 引擎) | RUL {s.rul.size} 行")
    save_json({"files": files, "subsets": summary},
              Path(__file__).resolve().parents[1] / "results" / "metrics" / "data_check.json")
    print("data_check.json saved")


if __name__ == "__main__":
    main()
