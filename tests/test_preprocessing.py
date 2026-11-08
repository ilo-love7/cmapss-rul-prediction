"""io 与预处理的单测（合成数据，不依赖 CMAPSS 文件）。"""
import numpy as np
import pytest

from cmapss_rul.io import N_COLS, split_xy, unit_series
from cmapss_rul.preprocessing import (
    build_sequences, piecewise_rul, select_sensors, sensor_matrix,
)


def test_split_xy_shapes():
    arr = np.random.rand(50, N_COLS)
    head, sensors = split_xy(arr)
    assert head.shape == (50, 2)
    assert sensors.shape == (50, 21)
    assert sensor_matrix(arr).shape == (50, 21)


def test_unit_series_sorted():
    arr = np.zeros((6, N_COLS))
    arr[:, 0] = 7
    arr[:, 1] = [3, 1, 2, 6, 4, 5]
    s = unit_series(arr, 7)
    assert list(s[:, 1]) == [1, 2, 3, 4, 5, 6]


def test_piecewise_rul_cap_and_zero():
    cycles = np.arange(1, 201)  # 200 周期寿命
    rul = piecewise_rul(cycles, cap=125)
    assert rul[0] == 125          # 早期封顶
    assert rul[-1] == 0           # 失效时刻为 0
    assert rul[74] == 125         # 200-75=125 仍在封顶区
    assert rul[150] == 49         # 200-151=49 线性段
    assert np.all(np.diff(rul) <= 1e-9)  # 单调不增
    assert np.all(rul >= 0)


def test_piecewise_rul_short_life():
    cycles = np.arange(1, 31)
    rul = piecewise_rul(cycles, cap=125)
    assert rul[0] == 29           # 寿命 30 < cap，全段线性
    assert rul[-1] == 0


def test_select_sensors_drops_constant():
    X = np.random.rand(100, 5)
    X[:, 2] = 3.14                 # 常数列
    X[:, 4] = 518.67               # 常数列
    keep = select_sensors(X, std_min=0.01)
    assert 2 not in keep and 4 not in keep
    assert set(keep) == {0, 1, 3}


def test_build_sequences_no_cross_unit():
    sensors = np.random.rand(20, 4)
    labels = np.linspace(9, 0, 20)
    unit_ids = np.array([1] * 12 + [2] * 8)
    X, y, meta = build_sequences(sensors, labels, unit_ids, window=5, stride=1)
    # 引擎1: 12-5+1=8 窗；引擎2: 8-5+1=4 窗
    assert X.shape == (12, 5, 4)
    assert y.shape == (12,)
    assert set(meta[:, 0]) == {1, 2}
    assert (meta[:, 0] == 1).sum() == 8
    assert (meta[:, 0] == 2).sum() == 4
