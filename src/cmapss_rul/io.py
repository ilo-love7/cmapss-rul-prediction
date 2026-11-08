"""CMAPSS txt 数据加载。

每行 26 列：unit_id, cycle, op_setting 1-3, sensor 1-21（空格分隔）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_COLS = 26
N_OP = 3
N_SENSORS = 21


@dataclass
class Subset:
    """一个 FD 子集的完整数据。"""

    name: str
    train: np.ndarray          # (n, 26) run-to-failure
    test: np.ndarray           # (n, 26) 截断轨迹
    rul: np.ndarray            # (n_test_units,) 测试集真实 RUL

    @property
    def train_units(self) -> list[int]:
        return sorted(set(self.train[:, 0].astype(int).tolist()))

    @property
    def test_units(self) -> list[int]:
        return sorted(set(self.test[:, 0].astype(int).tolist()))


def load_subset(raw_dir: str | Path, name: str) -> Subset:
    """加载 FD001–FD004 之一。"""
    raw = Path(raw_dir)
    train = np.loadtxt(raw / f"train_{name}.txt")
    test = np.loadtxt(raw / f"test_{name}.txt")
    rul = np.loadtxt(raw / f"RUL_{name}.txt").reshape(-1)
    if train.shape[1] != N_COLS or test.shape[1] != N_COLS:
        raise ValueError(f"{name}: 期望 26 列，实际 {train.shape[1]}/{test.shape[1]}")
    n_test_units = len(np.unique(test[:, 0]))
    if n_test_units != len(rul):
        raise ValueError(f"{name}: 测试引擎数 {n_test_units} != RUL 行数 {len(rul)}")
    return Subset(name=name, train=train, test=test, rul=rul)


def load_all(raw_dir: str | Path, names=("FD001", "FD002", "FD003", "FD004")) -> dict[str, Subset]:
    return {n: load_subset(raw_dir, n) for n in names}


def split_xy(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n,26) -> (unit_id+cycle, 传感器 21 列)。"""
    return arr[:, :2], arr[:, 2 + N_OP:]


def unit_series(arr: np.ndarray, unit: int) -> np.ndarray:
    """取单引擎的 (cycles, 26) 按 cycle 升序。"""
    rows = arr[arr[:, 0] == unit]
    return rows[np.argsort(rows[:, 1])]


def iter_units(arr: np.ndarray):
    for u in np.unique(arr[:, 0]):
        yield int(u), unit_series(arr, u)
