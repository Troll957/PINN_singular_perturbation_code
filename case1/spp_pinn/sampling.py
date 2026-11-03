"""残差驱动的自适应采样点集 (meshless)。

本模块只维护**点**，不划分单元。三组点分别是

* ``interior`` : 区域内部的 PDE 配点;
* ``ic``       : 初值线 $t=0$ 上的配点;
* ``bc``       : 边界 $x=0$、$x=1$ 上的配点。

自适应增点策略
--------------
设当前配点集为 $X_k$，候选池为 $C_k\\subset\\Omega$。训练后用残差
$|r(z;\\theta)|$ 作为"该处还差多少"的后验度量，然后

* ``residual``     : 贪心取 $|r|$ 最大的 $m$ 个候选点（RAR, residual-based
  adaptive refinement）;
* ``distribution`` : 按 $\\rho(z)\\propto|r(z)|^{2\\beta}$ 抽样 $m$ 个点（RAD）;
* ``uniform``      : 从**同一个**候选池中**均匀随机**地取 $m$ 个点（对照组，
  用于分离"增加点数"与"把点放在哪里"这两种效应）;
* ``fixed``        : 不增点。

初值/边值点用约束违反量本身的残差作同样的处理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
# 低差异序列
# ----------------------------------------------------------------------
def halton(n: int, dim: int = 2, bases: Sequence[int] = (2, 3, 5),
           offset: int = 0) -> np.ndarray:
    """Halton 低差异序列, 返回 (n, dim) 的数组, 取值在 [0,1)。

    初始配点用低差异序列而不是伪随机数, 可以避免初始点成团或留下大片空洞。
    """
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
    """返回 ``values`` 中最大的 m 个元素的下标 (降序)。"""
    m = int(min(m, values.size))
    if m <= 0:
        return np.empty(0, dtype=np.int64)
    idx = np.argpartition(-values, m - 1)[:m]
    return idx[np.argsort(-values[idx])]


def select_distribution(values: np.ndarray, m: int, beta: float = 1.0,
                        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """按 ``|values|^{2*beta}`` 的权重有放回抽样 m 个下标。"""
    rng = rng or np.random.default_rng()
    w = np.abs(values) ** (2.0 * beta)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return rng.choice(values.size, size=m, replace=True)
    p = w / s
    return rng.choice(values.size, size=m, replace=True, p=p)


def select_separated(points: np.ndarray, values: np.ndarray, m: int,
                     existing: Optional[np.ndarray] = None,
                     min_dist: float = 0.0) -> np.ndarray:
    """贪心选取残差最大的点, 同时保证与已有点/已选点的距离不小于 ``min_dist``。

    用均匀网格做空间哈希, 使距离检查接近 O(1)。若因距离约束选不满 m 个,
    则放宽约束把剩余名额按残差补齐。
    """
    idx_sorted = np.argsort(-values)
    if min_dist <= 0.0:
        return idx_sorted[:int(min(m, values.size))]

    cell = float(min_dist)
    grid: dict = {}

    def _key(q):
        return (int(np.floor(q[0] / cell)), int(np.floor(q[1] / cell)))

    def _insert(q):
        k = _key(q)
        grid.setdefault(k, []).append(q)

    def _far_enough(q):
        kx, ky = _key(q)
        for i in (kx - 1, kx, kx + 1):
            for j in (ky - 1, ky, ky + 1):
                for p in grid.get((i, j), ()):
                    if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < min_dist ** 2:
                        return False
        return True

    if existing is not None:
        for q in existing:
            _insert(q)

    chosen = []
    for i in idx_sorted:
        if len(chosen) >= m:
            break
        q = points[i]
        if _far_enough(q):
            chosen.append(i)
            _insert(q)
    if len(chosen) < m:                      # 放宽约束补齐
        rest = [i for i in idx_sorted if i not in set(chosen)]
        chosen.extend(rest[:m - len(chosen)])
    return np.asarray(chosen[:m], dtype=np.int64)


# ----------------------------------------------------------------------
# 配点集
# ----------------------------------------------------------------------
@dataclass
class SampleStats:
    n_interior: int = 0
    n_ic: int = 0
    n_bc: int = 0
    n_added: int = 0

    @property
    def total(self) -> int:
        return self.n_interior + self.n_ic + self.n_bc


class CollocationSet:
    """PDE 配点的容器。

    Parameters
    ----------
    T : float
        终止时间。
    seed : int
        随机种子。
    """

    def __init__(self, T: float = 1.0, seed: int = 0):
        self.T = float(T)
        self.rng = np.random.default_rng(seed)
        self.interior = np.zeros((0, 2), dtype=float)
        self.ic = np.zeros((0, 2), dtype=float)
        self.bc = np.zeros((0, 2), dtype=float)

    # ---------------- 初始化 ----------------
    def initialise(self, n_interior: int, n_ic: int, n_bc: int,
                   method: str = "halton") -> None:
        """初始采样: 内部点用 Halton 低差异序列或均匀随机, 初/边值点均匀。"""
        if method == "halton":
            u = halton(n_interior, 2, offset=int(self.rng.integers(0, 10_000)))
        elif method == "uniform":
            u = self.rng.random((n_interior, 2))
        else:
            raise KeyError(f"未知初始采样方式 {method!r}")
        self.interior = np.column_stack([u[:, 0], u[:, 1] * self.T])

        xi = self.rng.random(n_ic)
        self.ic = np.column_stack([xi, np.zeros_like(xi)])

        tb = self.rng.random(n_bc)
        left = np.column_stack([np.zeros_like(tb), tb * self.T])
        right = np.column_stack([np.ones_like(tb), tb * self.T])
        self.bc = np.concatenate([left, right], axis=0)

    # ---------------- 增点 ----------------
    def add_interior(self, pts: np.ndarray) -> int:
        self.interior = np.concatenate([self.interior, np.atleast_2d(pts)], axis=0)
        return len(pts)

    def add_ic(self, x: np.ndarray) -> int:
        pts = np.column_stack([np.asarray(x).reshape(-1), np.zeros(len(x))])
        self.ic = np.concatenate([self.ic, pts], axis=0)
        return len(pts)

    def add_bc(self, pts: np.ndarray) -> int:
        self.bc = np.concatenate([self.bc, np.atleast_2d(pts)], axis=0)
        return len(pts)

    # ---------------- 查询 ----------------
    @property
    def n_interior(self) -> int:
        return self.interior.shape[0]

    @property
    def n_ic(self) -> int:
        return self.ic.shape[0]

    @property
    def n_bc(self) -> int:
        return self.bc.shape[0]

    @property
    def total(self) -> int:
        return self.n_interior + self.n_ic + self.n_bc

    def stats(self) -> SampleStats:
        return SampleStats(self.n_interior, self.n_ic, self.n_bc)

    # ---------------- 诊断 ----------------
    def density_profile(self, y: np.ndarray, x_max: float = 0.2,
                        n_bins: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        """内部点在 $x$ 方向的密度剖面 (只统计 $x\\le x_{\\max}$ 的部分)。

        返回 ``(bin_center, count)``, 供报告里画"点是否聚在层内"使用。
        """
        m = self.interior[:, 0] <= x_max
        cnt, edges = np.histogram(self.interior[m, 0], bins=n_bins,
                                 range=(0.0, x_max))
        return 0.5 * (edges[:-1] + edges[1:]), cnt.astype(float)

    def candidate_pool(self, m: int, method: str = "random") -> np.ndarray:
        """生成候选池 (在 $\\Omega$ 内均匀)。"""
        if method == "halton":
            u = halton(m, 2, offset=int(self.rng.integers(0, 1_000_000)))
        else:
            u = self.rng.random((m, 2))
        return np.column_stack([u[:, 0], u[:, 1] * self.T])


# ----------------------------------------------------------------------
# Voronoi (离散) 求积权重
# ----------------------------------------------------------------------
def voronoi_weights_2d(points: np.ndarray, extent=(0.0, 1.0, 0.0, 1.0),
                       n: int = 256, chunk: int = 768) -> np.ndarray:
    """离散 Voronoi 权重: 每个采样点"代表"的面积。

    把 $\\Omega$ 划成 $n\\times n$ 个小格, 每个小格归给离它最近的采样点;
    点 $i$ 的权重就是它分到的小格总面积。于是

    $$\\sum_i \\omega_i\\,g(z_i)\\approx\\int_\\Omega g(z)\\,\\mathrm dz$$

    对**任意**点分布都成立 (只要 $g$ 在 Voronoi 胞内变化不大)。这正是
    "自适应加点但不改变目标泛函"所需要的权重: 点变密的地方胞自动变小,
    因此损失始终是 $\\|r\\|_{L^2(\\Omega)}^2$ 的相容估计, 而不是被隐式
    重新加权的 $\\int r^2\\rho$。
    """
    x0, x1, t0, t1 = extent
    gx = x0 + (np.arange(n) + 0.5) * (x1 - x0) / n
    gt = t0 + (np.arange(n) + 0.5) * (t1 - t0) / n
    GX, GT = np.meshgrid(gx, gt, indexing="ij")
    grid = np.stack([GX.reshape(-1), GT.reshape(-1)], axis=1)
    cell_area = ((x1 - x0) / n) * ((t1 - t0) / n)

    idx = np.empty(grid.shape[0], dtype=np.int64)
    for k in range(0, grid.shape[0], chunk):
        c = grid[k:k + chunk]
        d2 = ((c[:, None, :] - points[None, :, :]) ** 2).sum(-1)
        idx[k:k + chunk] = d2.argmin(axis=1)
    return np.bincount(idx, minlength=points.shape[0]) * cell_area


def voronoi_weights_1d(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """一维 Voronoi 权重 (即相邻点间距的一半之和), 权重之和为 ``hi - lo``。

    等价于把区间按中点划分给各点。
    """
    v = np.sort(np.asarray(values, dtype=float).reshape(-1))
    if v.size == 0:
        return np.zeros(0)
    edges = np.empty(v.size + 1)
    edges[1:-1] = 0.5 * (v[:-1] + v[1:])
    edges[0], edges[-1] = lo, hi
    w = np.diff(edges)
    # 把权重写回原顺序
    order = np.argsort(np.asarray(values, dtype=float).reshape(-1))
    out = np.empty(v.size)
    out[order] = w
    return out
