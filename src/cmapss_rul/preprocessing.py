"""预处理：传感器筛选、工况归一化、piecewise RUL 标签、序列切窗。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import N_OP, N_SENSORS, iter_units


def op_settings(arr: np.ndarray) -> np.ndarray:
    return arr[:, 2:2 + N_OP]


def sensor_matrix(arr: np.ndarray) -> np.ndarray:
    return arr[:, 2 + N_OP:2 + N_OP + N_SENSORS]


def select_sensors(X: np.ndarray, std_min: float = 0.01) -> np.ndarray:
    """返回保留传感器的列索引：剔除近常数列。"""
    sd = X.std(axis=0)
    keep = np.where(sd >= std_min)[0]
    return keep if keep.size else np.arange(X.shape[1])


def fit_conditions(op: np.ndarray, n_clusters: int = 6, seed: int = 0) -> tuple:
    """对工况设定聚类（FD002/4 六工况）。返回 (kmeans|None, labels)。

    判定：工况设定四舍五入到 1 位小数后唯一行 ≤1 视为单工况
    （FD001/3 的设定值带 ±0.009 噪声，不能按 spread/unique 原始值判定）。
    """
    uniq = np.unique(np.round(op, 1), axis=0)
    spread = op.max(axis=0) - op.min(axis=0)
    if len(uniq) <= 1 or float(spread.max()) < 0.05:
        return None, np.zeros(len(op), dtype=int)
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    lab = km.fit_predict(op)
    return km, lab


def assign_conditions(km, op: np.ndarray) -> np.ndarray:
    if km is None:
        return np.zeros(len(op), dtype=int)
    return km.predict(op)


@dataclass
class Scaler:
    """按工况簇的 z-score 统计量。"""

    mu: np.ndarray        # (k, d)
    sd: np.ndarray        # (k, d)

    def transform(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        out = X.astype(np.float64).copy()
        for k in range(self.mu.shape[0]):
            m = labels == k
            if m.any():
                out[m] = (out[m] - self.mu[k]) / self.sd[k]
        return out


def fit_scaler(X: np.ndarray, labels: np.ndarray, n_clusters: int) -> Scaler:
    d = X.shape[1]
    mu = np.zeros((n_clusters, d))
    sd = np.zeros((n_clusters, d))
    for k in range(n_clusters):
        m = labels == k
        if m.any():
            mu[k] = X[m].mean(axis=0)
            s = X[m].std(axis=0)
            sd[k] = np.where(s < 1e-8, 1.0, s)
        else:
            mu[k] = 0.0
            sd[k] = 1.0
    return Scaler(mu=mu, sd=sd)


def piecewise_rul(cycles: np.ndarray, cap: float = 125.0) -> np.ndarray:
    """piecewise 线性 RUL：早期常数 cap，末段线性降到 0。

    cycles 为单引擎的 cycle 序列（升序），返回同形状 RUL 标签。
    末段斜率取 1（cycle 每 +1，RUL -1）。
    """
    cycles = np.asarray(cycles, dtype=np.float64)
    total = cycles.max()
    rul = np.minimum(total - cycles, cap)
    return np.maximum(rul, 0.0)


def build_sequences(sensors: np.ndarray, labels: np.ndarray, unit_ids: np.ndarray,
                    window: int = 30, stride: int = 1):
    """把长表切成 (n_seq, window, d) 样本，目标为窗口末步的 RUL 标签。

    按引擎切窗，不跨引擎拼接。返回 (X, y, meta) meta=(unit, end_cycle)。
    """
    Xs, ys, metas = [], [], []
    for u in np.unique(unit_ids):
        m = unit_ids == u
        S = sensors[m]
        y = labels[m]
        cyc = np.asarray(y.index if hasattr(y, "index") else range(len(y)))
        n = S.shape[0]
        for end in range(window - 1, n, stride):
            Xs.append(S[end - window + 1:end + 1])
            ys.append(y[end])
            metas.append((int(u), int(end)))
    X = np.asarray(Xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    meta = np.asarray(metas, dtype=np.int64)
    return X, y, meta


def prepare_subset(train: np.ndarray, test: np.ndarray, rul: np.ndarray,
                   window: int = 30, stride: int = 1, rul_cap: float = 125.0,
                   std_min: float = 0.01, n_conditions: int = 6,
                   cluster: bool = True, seed: int = 0) -> dict:
    """端到端预处理一个子集，返回训练序列与测试末窗 + 真实 RUL。"""
    tr_s, te_s = sensor_matrix(train), sensor_matrix(test)
    keep = select_sensors(tr_s, std_min=std_min)

    km = None
    if cluster:
        km, tr_lab = fit_conditions(op_settings(train), n_clusters=n_conditions, seed=seed)
    else:
        tr_lab = np.zeros(len(train), dtype=int)
    te_lab = assign_conditions(km, op_settings(test)) if km is not None else np.zeros(len(test), dtype=int)

    # 归一化统计只来自训练集
    scaler = fit_scaler(tr_s[:, keep], tr_lab, n_clusters=max(n_conditions, 1) if km is not None else 1)
    Xtr_s = scaler.transform(tr_s[:, keep], tr_lab)
    Xte_s = scaler.transform(te_s[:, keep], te_lab)

    # 标签：按引擎 piecewise RUL
    ytr = np.zeros(len(train))
    for u, rows in iter_units(train):
        idx = np.where(train[:, 0] == u)[0]
        ytr[idx] = piecewise_rul(rows[:, 1], cap=rul_cap)

    X, y, meta = build_sequences(Xtr_s, ytr, train[:, 0], window=window, stride=stride)

    # 测试：每引擎取最后一个窗口，目标 = 该引擎真实 RUL（封顶后）
    Xs = []
    test_units = np.unique(test[:, 0]).astype(int)
    y_true = np.minimum(rul.reshape(-1), rul_cap)
    for i, u in enumerate(test_units):
        rows = test[test[:, 0] == u]
        S = Xte_s[test[:, 0] == u]
        if S.shape[0] >= window:
            Xs.append(S[-window:])
        else:  # 短轨迹：左侧零填充
            pad = np.zeros((window - S.shape[0], S.shape[1]))
            Xs.append(np.vstack([pad, S]))
    Xte = np.asarray(Xs, dtype=np.float64)
    tr_series = per_unit_series(Xtr_s, train[:, 0])
    te_series = per_unit_series(Xte_s, test[:, 0])
    train_totals = np.array([len(s) for s in tr_series], dtype=np.float64)
    return {
        "X": X, "y": y, "meta": meta,
        "X_test": Xte, "y_test": y_true,
        "test_units": test_units, "keep_sensors": keep,
        "n_conditions": int(scaler.mu.shape[0]),
        "train_series": tr_series, "test_series": te_series,
        "train_totals": train_totals,
    }


def train_val_split_units(meta: np.ndarray, val_frac: float = 0.2, seed: int = 0):
    """按引擎划分训练/验证（防泄漏）。meta=(unit, end_cycle)。返回 (tr_mask, va_mask)。"""
    units = np.unique(meta[:, 0])
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_val = max(1, int(round(len(units) * val_frac)))
    val_units = set(units[:n_val].tolist())
    va = np.isin(meta[:, 0], list(val_units))
    return ~va, va


def per_unit_series(sensors: np.ndarray, unit_ids: np.ndarray) -> list[np.ndarray]:
    """把 (n, d) 传感器矩阵按引擎拆成 [(cycles, d), ...]（保持行序）。"""
    out = []
    for u in np.unique(unit_ids):
        out.append(sensors[unit_ids == u])
    return out
