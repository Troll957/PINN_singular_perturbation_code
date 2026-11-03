"""低差异序列与选点规则（Case 4 用的是与 1d/2d 完全相同的几条规则）。"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def halton(n: int, dim: int = 2, bases: Sequence[int] = (2, 3, 5),
           offset: int = 0) -> np.ndarray:
    """Halton 低差异序列, 返回 (n, dim) 的数组, 取值在 [0,1)。"""
    out = np.empty((n, dim), dtype=float)
    for d in range(dim):
        b = bases[d]
        idx = np.arange(offset + 1, offset + n + 1, dtype=np.int64)
        f = 1.0
        res = np.zeros(n, dtype=float)
        while idx.max() > 0:
            f /= b
            res += f * (idx % b)
            idx //= b
        out[:, d] = res
    return out


def select_topm(values: np.ndarray, m: int) -> np.ndarray:
    """返回 ``values`` 中最大的 m 个元素的下标 (降序)。"""
    m = int(min(m, values.size))
    if m <= 0:
        return np.empty(0, dtype=np.int64)
    idx = np.argpartition(-values, m - 1)[:m]
    return idx[np.argsort(-values[idx])]


def select_distribution(values: np.ndarray, m: int, beta: float = 1.0,
                        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """按 ``|values|^{2*beta}`` 的权重有放回抽样 m 个下标（RAD 规则）。"""
    rng = rng or np.random.default_rng()
    w = np.abs(values) ** (2.0 * beta)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return rng.choice(values.size, size=m, replace=True)
    return rng.choice(values.size, size=m, replace=True, p=w / s)
