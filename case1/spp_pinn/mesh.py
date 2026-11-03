"""各向异性 h-自适应网格 (quadtree / 二分细化)。

设计思路
--------
把 (x, t) 区域剖分成矩形单元 (cell), 每个单元内部放置 ``order x order``
个 Gauss-Legendre 求积点作为配点。训练一个轮次之后:

1. 在每个配点上计算 PDE 残差 r(x, t);
2. 按单元聚合成后验误差指示子
   ``eta_K = sqrt( sum_q w_q r(x_q, t_q)^2 )``, 即残差的单元 L2 范数;
3. 用 Doerfler (bulk) 标记法选出贡献了总指示子 ``theta`` 比例的单元;
4. 对这些单元做**各向异性二分**: 根据方向插值误差估计
   ``h_x^2 * int_K |u_xx|`` 与 ``h_t^2 * int_K |u_tt|`` 的大小,
   决定沿 x 还是沿 t 一分为二。

这样就能"在残差大的地方自动加密网格", 且不会在解光滑的方向浪费配点。

边值/初值点同样来自网格边界单元 (1 维 Gauss 点), 因此细化同样会
自动加密边界层/角点附近的初边值配点。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss


@dataclass
class MeshStats:
    """网格状态的简单快照。"""

    n_cells: int = 0
    n_points: int = 0
    min_hx: float = 0.0
    min_ht: float = 0.0
    max_level: int = 0
    n_marked: int = 0


class AdaptiveMesh2D:
    """矩形区域上的各向异性自适应网格。

    Parameters
    ----------
    order : int
        每个方向上的 Gauss-Legendre 求积阶数, 单元配点数为 ``order**2``。
    nx, nt : int
        初始均匀网格在两个方向上的单元数。
    extent : tuple
        ``(x0, x1, t0, t1)``。
    min_size : float
        单元最小边长, 小于该尺寸的单元不再细化 (防止无限加密)。
    max_level : int
        最大细化层数。
    """

    def __init__(
        self,
        order: int = 2,
        nx: int = 8,
        nt: int = 8,
        extent: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        min_size: float = 1e-6,
        max_level: int = 24,
    ):
        self.order = int(order)
        self.extent = tuple(float(v) for v in extent)
        self.min_size = float(min_size)
        self.max_level = int(max_level)
        self.nodes, self.weights = leggauss(self.order)

        x0, x1, t0, t1 = self.extent
        xs = np.linspace(x0, x1, nx + 1)
        ts = np.linspace(t0, t1, nt + 1)
        cells = []
        for i in range(nx):
            for j in range(nt):
                cells.append([xs[i], xs[i + 1], ts[j], ts[j + 1], 0.0])
        self.cells = np.asarray(cells, dtype=float)
        # 细化历史 (每轮记录一次)
        self.history: list = []

    # ------------------------------------------------------------------
    # 基本信息
    # ------------------------------------------------------------------
    @property
    def n_cells(self) -> int:
        return self.cells.shape[0]

    @property
    def n_points(self) -> int:
        return self.n_cells * self.order ** 2

    def hx(self) -> np.ndarray:
        return self.cells[:, 1] - self.cells[:, 0]

    def ht(self) -> np.ndarray:
        return self.cells[:, 3] - self.cells[:, 2]

    def stats(self) -> MeshStats:
        hx, ht = self.hx(), self.ht()
        return MeshStats(
            n_cells=self.n_cells,
            n_points=self.n_points,
            min_hx=float(hx.min()),
            min_ht=float(ht.min()),
            max_level=int(self.cells[:, 4].max()),
        )

    # ------------------------------------------------------------------
    # 配点 (体单元 Gauss 求积)
    # ------------------------------------------------------------------
    def quadrature(self):
        """返回单元内部的所有 Gauss 配点。

        Returns
        -------
        pts : (N, 2) float64
        weights : (N,) float64
            求积权重, 满足 ``weights.sum() == |Omega|``。
        cell_id : (N,) int
            每个配点所属单元编号。
        """
        q = self.order
        nd, wt = self.nodes, self.weights
        hx, ht = self.hx(), self.ht()
        xm = 0.5 * (self.cells[:, 0] + self.cells[:, 1])
        tm = 0.5 * (self.cells[:, 2] + self.cells[:, 3])

        XI, ETA = np.meshgrid(nd, nd, indexing="ij")          # (q, q)
        W = np.outer(wt, wt)                                   # (q, q)

        X = xm[:, None, None] + 0.5 * hx[:, None, None] * XI[None]
        Tt = tm[:, None, None] + 0.5 * ht[:, None, None] * ETA[None]
        Wq = W[None] * (0.25 * hx * ht)[:, None, None]

        pts = np.stack([X.reshape(-1), Tt.reshape(-1)], axis=1)
        weights = Wq.reshape(-1)
        cell_id = np.repeat(np.arange(self.n_cells), q * q)
        return pts, weights, cell_id

    # ------------------------------------------------------------------
    # 边界配点 (初值 t=t0, 左右边界 x=x0 / x=x1)
    # ------------------------------------------------------------------
    def boundary_quadrature(self, order: int | None = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        q = self.order if order is None else int(order)
        nd, wt = leggauss(q)
        x0, x1, t0, t1 = self.extent
        tol = 1e-12

        ic_pts, ic_w = [], []
        left_pts, left_w = [], []
        right_pts, right_w = [], []

        for xa, xb, ta, tb, _ in self.cells:
            hx, ht = xb - xa, tb - ta
            if abs(ta - t0) < tol:                      # t = t0 初值边
                xm = 0.5 * (xa + xb)
                ic_pts.append(np.stack([xm + 0.5 * hx * nd, np.full(q, t0)], axis=1))
                ic_w.append(wt * 0.5 * hx)
            if abs(xa - x0) < tol:                      # x = x0 左边界
                tm = 0.5 * (ta + tb)
                left_pts.append(np.stack([np.full(q, x0), tm + 0.5 * ht * nd], axis=1))
                left_w.append(wt * 0.5 * ht)
            if abs(xb - x1) < tol:                      # x = x1 右边界
                tm = 0.5 * (ta + tb)
                right_pts.append(np.stack([np.full(q, x1), tm + 0.5 * ht * nd], axis=1))
                right_w.append(wt * 0.5 * ht)

        def _cat(ps, ws):
            if not ps:
                return np.zeros((0, 2)), np.zeros(0)
            return np.concatenate(ps, axis=0), np.concatenate(ws)

        return {
            "ic": _cat(ic_pts, ic_w),
            "left": _cat(left_pts, left_w),
            "right": _cat(right_pts, right_w),
        }

    # ------------------------------------------------------------------
    # 后验误差指示子与标记
    # ------------------------------------------------------------------
    def cell_indicators(self, r2: np.ndarray, weights: np.ndarray, cell_id: np.ndarray,
                        normalize: bool = False) -> np.ndarray:
        """残差型单元指示子。

        ``normalize=False`` (默认): ``eta_K = ||r||_{L2(K)} = sqrt(sum_q w_q r_q^2)``,
        即残差的单元 L2 范数。

        ``normalize=True``: ``eta_K = ||r||_{L2(K)} / sqrt(|K|)``, 即残差在单元内的
        RMS 值 (残差密度)。注意 ``||r||_{L2(K)}`` 随单元面积趋于 0 而趋于 0, 因此在
        被细化过的网格上它会被"单元尺寸"主导, 反而与真实单元误差呈负相关;
        密度型指示子消除了这一偏差。
        """
        eta2 = np.bincount(cell_id, weights=r2 * weights, minlength=self.n_cells)
        eta2 = np.maximum(eta2, 0.0)
        if normalize:
            eta2 = eta2 / np.maximum(self.cell_areas(), 1e-300)
        return np.sqrt(eta2)

    def directional_estimates(
        self,
        abs_uxx: np.ndarray,
        abs_utt: np.ndarray,
        weights: np.ndarray,
        cell_id: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """方向插值误差估计 (用于选择二分方向)。

        ``ex = h_x^2 * int_K |u_xx|``, ``et = h_t^2 * int_K |u_tt|``。
        """
        ix = np.bincount(cell_id, weights=abs_uxx * weights, minlength=self.n_cells)
        it = np.bincount(cell_id, weights=abs_utt * weights, minlength=self.n_cells)
        ex = self.hx() ** 2 * ix
        et = self.ht() ** 2 * it
        return ex, et

    def refinable(self) -> np.ndarray:
        """还可以继续细化的单元。"""
        return ((self.hx() > 2.0 * self.min_size)
                | (self.ht() > 2.0 * self.min_size)) & (self.cells[:, 4] < self.max_level)

    def dorfler_mark(self, eta: np.ndarray, theta: float = 0.5) -> np.ndarray:
        """Doerfler (bulk) 标记: 最小的单元集合使得
        ``sum_{marked} eta_K^2 >= theta^2 * sum_K eta_K^2``。"""
        eta = np.asarray(eta, dtype=float)
        mask = np.zeros(eta.size, dtype=bool)
        if eta.size == 0 or not np.any(eta > 0):
            return mask
        order = np.argsort(-eta)
        cum = np.cumsum(eta[order] ** 2)
        total = cum[-1]
        k = int(np.searchsorted(cum, theta ** 2 * total) + 1)
        mask[order[:k]] = True
        mask &= self.refinable()
        return mask

    def maximum_mark(self, eta: np.ndarray, beta: float = 0.25) -> np.ndarray:
        """最大值策略: 标记所有 ``eta_K >= beta * max_K eta_K`` 的单元。

        相比于 Doerfler 标记, 当指示子高度集中在少数单元时 (奇异摄动问题的
        典型情形), 最大值策略能一次性选出整条边界层单元列, 细化效率高得多。
        """
        eta = np.asarray(eta, dtype=float)
        if eta.size == 0 or eta.max() <= 0:
            return np.zeros(eta.size, dtype=bool)
        mask = (eta >= beta * eta.max()) & (eta > 0)
        mask &= self.refinable()
        return mask

    def mark(self, eta: np.ndarray, strategy: str = "max", param: float = 0.25,
             max_marked: Optional[int] = None) -> np.ndarray:
        """统一的标记接口。

        Parameters
        ----------
        strategy : {'max', 'dorfler', 'mix'}
            * ``max``     : 最大值策略 ``eta_K >= param * max(eta)``;
            * ``dorfler`` : Doerfler bulk 标记;
            * ``mix``     : 两者并集 (默认)。指示子高度集中时由最大值策略
              选出整条层单元列, 指示子接近平坦时由 Doerfler 保证不会全选。
        max_marked : int, optional
            每轮最多细化的单元数 (按 eta 从大到小取), 用于限制网格增长。
        """
        eta = np.asarray(eta, dtype=float)
        if strategy == "max":
            mask = self.maximum_mark(eta, param)
        elif strategy == "dorfler":
            mask = self.dorfler_mark(eta, param)
        elif strategy == "mix":
            mask = self.maximum_mark(eta, param) | self.dorfler_mark(eta, max(param, 0.5))
        else:
            raise KeyError(f"未知标记策略 {strategy!r}")
        if max_marked is not None and mask.sum() > max_marked:
            idx = np.flatnonzero(mask)
            keep = idx[np.argsort(-eta[idx])[:max_marked]]
            mask = np.zeros_like(mask)
            mask[keep] = True
        return mask

    # ------------------------------------------------------------------
    # 细化
    # ------------------------------------------------------------------
    def refine(self, marked: np.ndarray, direction: np.ndarray | None = None) -> int:
        """对被标记的单元做二分 (或四等分)。

        Parameters
        ----------
        marked : (n_cells,) bool
        direction : (n_cells,) 字符串数组, 取值 ``'x'``/``'t'``/``'both'``。
            缺省时按最长边方向二分。

        Returns
        -------
        int
            实际被细化的单元个数。
        """
        if direction is None:
            direction = np.where(self.hx() >= self.ht(), "x", "t")
        new_cells = []
        n_refined = 0
        for cell, mk, d in zip(self.cells, marked, direction):
            xa, xb, ta, tb, lev = cell
            can_x = (xb - xa) > 2.0 * self.min_size and lev < self.max_level
            can_t = (tb - ta) > 2.0 * self.min_size and lev < self.max_level
            if not mk:
                new_cells.append(cell)
                continue
            split_x = d in ("x", "both") and can_x
            split_t = d in ("t", "both") and can_t
            if not split_x and not split_t:
                new_cells.append(cell)
                continue
            n_refined += 1
            xm = 0.5 * (xa + xb)
            tm = 0.5 * (ta + tb)
            if split_x and split_t:
                new_cells += [
                    [xa, xm, ta, tm, lev + 1], [xm, xb, ta, tm, lev + 1],
                    [xa, xm, tm, tb, lev + 1], [xm, xb, tm, tb, lev + 1],
                ]
            elif split_x:
                new_cells += [[xa, xm, ta, tb, lev + 1], [xm, xb, ta, tb, lev + 1]]
            else:
                new_cells += [[xa, xb, ta, tm, lev + 1], [xa, xb, tm, tb, lev + 1]]
        self.cells = np.asarray(new_cells, dtype=float)
        return n_refined

    def choose_directions(self, ex: np.ndarray, et: np.ndarray, mode: str = "auto") -> np.ndarray:
        """根据方向误差估计选择二分方向。"""
        n = self.n_cells
        if mode == "x":
            return np.full(n, "x", dtype=object)
        if mode == "t":
            return np.full(n, "t", dtype=object)
        if mode == "both":
            return np.full(n, "both", dtype=object)
        if mode != "auto":
            raise KeyError(f"未知方向模式 {mode!r}")
        # 方向误差大的一侧优先细分; 两个方向都饱和时取较长边
        d = np.where(ex >= et, "x", "t").astype(object)
        degenerate = (ex + et) <= 1e-30
        longer = np.where(self.hx() >= self.ht(), "x", "t").astype(object)
        d[degenerate] = longer[degenerate]
        return d

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------
    def cell_centers(self) -> np.ndarray:
        return np.stack([
            0.5 * (self.cells[:, 0] + self.cells[:, 1]),
            0.5 * (self.cells[:, 2] + self.cells[:, 3]),
        ], axis=1)

    def cell_areas(self) -> np.ndarray:
        return self.hx() * self.ht()

    def cell_samples(self, n: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """在每个单元内取 ``n x n`` 个均匀点 (用于估计单元真实误差)。

        Returns
        -------
        pts : (n_cells * n^2, 2)
        cell_id : (n_cells * n^2,)
        weights : (n_cells * n^2,)  权重和等于区域面积
        """
        s = (np.arange(n) + 0.5) / n
        hx, ht = self.hx(), self.ht()
        xa = self.cells[:, 0]
        ta = self.cells[:, 2]
        nc = self.n_cells
        SX = np.broadcast_to(s[None, :, None], (nc, n, n))
        ST = np.broadcast_to(s[None, None, :], (nc, n, n))
        X = xa[:, None, None] + hx[:, None, None] * SX
        Tt = ta[:, None, None] + ht[:, None, None] * ST
        w = (hx * ht)[:, None, None] * np.full((1, n, n), 1.0 / (n * n))
        pts = np.stack([X.reshape(-1), Tt.reshape(-1)], axis=1)
        cell_id = np.repeat(np.arange(self.n_cells), n * n)
        return pts, cell_id, w.reshape(-1)
