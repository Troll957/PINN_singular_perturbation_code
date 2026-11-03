"""点集、低差异序列与 k 近邻求积权重。

与主包 ``spp_pinn`` 中 ``空间2d`` 的实现一致，这里只保留 case5 用到的部分，
使本目录自成一体（不依赖上层包的路径）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
# 低差异序列
# ----------------------------------------------------------------------
def halton(n: int, dim: int = 3, bases: Sequence[int] = (2, 3, 5, 7),
           offset: int = 0) -> np.ndarray:
    """Halton 低差异序列，返回 ``(n, dim)``，取值在 ``[0,1)``。"""
    if dim > len(bases):
        raise ValueError(f"halton 最多支持 {len(bases)} 维")
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


# ----------------------------------------------------------------------
# 选点
# ----------------------------------------------------------------------
def select_topm(values: np.ndarray, m: int) -> np.ndarray:
    """返回 ``values`` 中最大的 ``m`` 个元素的下标（降序）。"""
    m = int(min(m, values.size))
    if m <= 0:
        return np.empty(0, dtype=np.int64)
    idx = np.argpartition(-values, m - 1)[:m]
    return idx[np.argsort(-values[idx])]


# ----------------------------------------------------------------------
# k 近邻求积权重
# ----------------------------------------------------------------------
def knn_quadrature_weights(coords: np.ndarray, measure: float,
                           n_neighbors: int = 4, chunk: int = 2048) -> np.ndarray:
    r"""k 近邻体积估计出的求积权重，满足 :math:`\sum_i\omega_i=\text{measure}`。

    若 :math:`\bar r_i` 是 :math:`z_i` 到其 :math:`k` 个最近邻的平均距离，则

    .. math::

        \omega_i=\text{measure}\cdot
        \frac{\bar r_i^{D}}{\sum_j\bar r_j^{D}},

    其中 :math:`D` 是点集所在流形的维数（内部点 3、面 2）。
    """
    P = np.ascontiguousarray(coords, dtype=float)
    N, D = P.shape
    if N <= 1:
        return np.full(N, measure)
    k = int(min(max(1, n_neighbors), N - 1))

    rbar = np.empty(N)
    for s0 in range(0, N, chunk):
        c = P[s0:s0 + chunk]
        d2 = ((c[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        part = np.partition(d2, k, axis=1)[:, 1:k + 1]
        rbar[s0:s0 + chunk] = np.sqrt(part).mean(axis=1)

    w = rbar ** D
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0:
        return np.full(N, measure / N)
    return w / tot * measure


# ----------------------------------------------------------------------
# 配点集
# ----------------------------------------------------------------------
@dataclass
class Points2D:
    """二维空间 + 时间的配点集。

    * ``interior`` : ``(N,3)``，区域内部点；
    * ``ic``       : ``(N,3)``，``t=0`` 上的点；
    * ``bc``       : ``{(axis, value): (N,3)}``，四个 Dirichlet 面上的点。
    """

    T: float = 1.0
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.interior = np.zeros((0, 3))
        self.ic = np.zeros((0, 3))
        self.bc: Dict[Tuple[int, float], np.ndarray] = {}

    # ---------------- 初始化 ----------------
    def initialise(self, n_int: int, n_ic: int, n_bc: int,
                   faces: Sequence[Tuple[int, float]], method: str = "halton") -> None:
        """``n_bc`` 是**每个面**上的点数。"""
        if method == "halton":
            u = halton(n_int, 3)
        else:
            u = self.rng.random((n_int, 3))
        self.interior = self._to_box(u)

        X = self.rng.random((n_ic, 2))
        self.ic = np.column_stack([X, np.zeros(n_ic)])

        for axis, value in faces:
            X = self.rng.random((n_bc, 2))
            X[:, axis] = value
            self.bc[(axis, value)] = np.column_stack(
                [X, self.rng.random(n_bc) * self.T])

    def _to_box(self, u: np.ndarray) -> np.ndarray:
        Z = u.copy()
        Z[:, 2] *= self.T
        return Z

    # ---------------- 增点 ----------------
    def add_interior(self, pts: np.ndarray) -> int:
        pts = np.atleast_2d(pts)
        self.interior = np.concatenate([self.interior, pts], axis=0)
        return pts.shape[0]

    def add_ic(self, X: np.ndarray) -> int:
        X = np.atleast_2d(X)
        pts = np.column_stack([X, np.zeros(X.shape[0])])
        self.ic = np.concatenate([self.ic, pts], axis=0)
        return pts.shape[0]

    def add_bc(self, face: Tuple[int, float], pts: np.ndarray) -> int:
        pts = np.atleast_2d(pts)
        old = self.bc.get(face, np.zeros((0, 3)))
        self.bc[face] = np.concatenate([old, pts], axis=0)
        return pts.shape[0]

    # ---------------- 查询 ----------------
    @property
    def n_interior(self) -> int:
        return self.interior.shape[0]

    @property
    def n_ic(self) -> int:
        return self.ic.shape[0]

    @property
    def n_bc(self) -> int:
        return sum(a.shape[0] for a in self.bc.values())

    @property
    def total(self) -> int:
        return self.n_interior + self.n_ic + self.n_bc

    def candidate_pool(self, m: int) -> np.ndarray:
        """区域内均匀候选池（三维时空均匀）。"""
        return self._to_box(self.rng.random((m, 3)))

    # ---------------- 统计 ----------------
    def layer_fraction(self, problem, factor: float = 4.0) -> float:
        """内部点中落在薄层内的比例（真值层位）。"""
        if self.n_interior == 0:
            return float("nan")
        return float(np.mean(problem.inside_layer(self.interior, factor=factor)))

    def ridge_fraction(self, problem, factor: float = 4.0) -> float:
        """内部点中落在"脊"内（x 与 y 两个方向都在层内）的比例。"""
        if self.n_interior == 0:
            return float("nan")
        return float(np.mean(problem.ridge_weight(self.interior, factor=factor)))
