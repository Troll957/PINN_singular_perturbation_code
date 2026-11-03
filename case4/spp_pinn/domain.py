r"""Case 4 的区域：一个**非矩形、非凸**的三瓣"花瓣"区域。

区域由水平集给出

.. math::
    \Omega=\{(x,y)\in\mathbb R^2:\ p(x,y)<0\},\qquad
    p(x,y)=x^2+y^2-1-a\,(x^3-3xy^2),\qquad a=0.30 .

用极坐标写 $x=r\cos\theta,\ y=r\sin\theta$，注意到 $x^3-3xy^2=r^3\cos 3\theta$，

.. math::
    p=r^2-1-a\,r^3\cos 3\theta ,

所以 $\partial\Omega$ 是

.. math::
    r^2-a\,r^3\cos 3\theta=1 ,

一条半径在 $r\approx0.89$（$\cos3\theta=-1$）与 $r\approx1.36$
（$\cos3\theta=1$）之间起伏的光滑闭曲线。它有三条向外的"花瓣"和三处向内的
"腰"，因而**既不是矩形也不是圆**：极坐标变换不能把它变成矩形，而任何
Sishkin/Bakhvalov 型张量网格都需要沿这条曲线做贴体坐标变换。

几何量的选取
------------
* 水平集函数本身就用 $p$，不做归一化。层坐标取 :math:`\psi=-p>0`（区域内为正，
  边界上为零）。因为 $|\nabla p|\approx2.4$ 在 $\partial\Omega$ 上近似为常数，
  $\psi$ 与到边界的真实距离只差一个 $O(1)$ 因子，层宽仍是 $O(\sqrt\varepsilon)$。
* :math:`x^3-3xy^2=\operatorname{Re}(x+iy)^3` 是**调和多项式**，故
  :math:`\Delta\psi\equiv-4` 是常数，这使解析右端项 f 非常紧凑（见 case4.py）。
* 到边界的近似有符号距离用一阶展开 :math:`d\approx\psi/|\nabla\psi|`，
  只用于"层内点占比"这个诊断量，不进入方程。
"""

from __future__ import annotations

import numpy as np

A_WAVE = 0.30          # 花瓣振幅
R_MAX = 1.45           # 外接盒半宽（> max r(θ)，用于采样与评估网格）


# ----------------------------------------------------------------------
# 水平集与几何量（numpy）
# ----------------------------------------------------------------------
def lev(x, y):
    """水平集函数 p：内部 p<0。"""
    return x * x + y * y - 1.0 - A_WAVE * (x ** 3 - 3.0 * x * y * y)


def grad_lev(x, y):
    """∇p。"""
    px = 2.0 * x - A_WAVE * (3.0 * x * x - 3.0 * y * y)
    py = 2.0 * y + A_WAVE * (6.0 * x * y)
    return px, py


def psi(x, y):
    """层坐标 ψ = -p ≥ 0（区域内）。"""
    return -lev(x, y)


def inside(X: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """(N,2) 空间点是否落在区域内（``margin`` 为正时要求严格内点）。"""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return lev(X[:, 0], X[:, 1]) < -margin


def sdf(X: np.ndarray) -> np.ndarray:
    """到边界的一阶近似距离 ψ/|∇ψ|（内部为负号的相反数：区域内为正）。"""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    x, y = X[:, 0], X[:, 1]
    px, py = grad_lev(x, y)
    g = np.hypot(px, py)
    return psi(x, y) / np.maximum(g, 1e-12)


# ----------------------------------------------------------------------
# 边界：按弧长均匀采样
# ----------------------------------------------------------------------
def radius(theta: np.ndarray, newton: int = 40) -> np.ndarray:
    """给定极角 θ，求边界半径 r(θ)：解 r² - a r³ cos3θ = 1 的最小正根。

    用 Newton 迭代，从 r=1 出发；该三次多项式在 r∈(0,2) 内只有这一个根。
    """
    th = np.asarray(theta, dtype=float)
    c3 = np.cos(3.0 * th)
    r = np.ones_like(th)
    for _ in range(newton):
        f = r * r - 1.0 - A_WAVE * r ** 3 * c3
        df = 2.0 * r - 3.0 * A_WAVE * r * r * c3
        step = f / np.where(np.abs(df) < 1e-12, 1e-12, df)
        r = r - step
        if np.max(np.abs(step)) < 1e-14:
            break
    return r


def boundary_points(theta: np.ndarray) -> np.ndarray:
    """θ → ∂Ω 上的点 (N,2)。"""
    th = np.asarray(theta, dtype=float)
    r = radius(th)
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


def outward_normal(X: np.ndarray) -> np.ndarray:
    """(N,2) 边界点上的单位外法向 ∇p/|∇p|。"""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    px, py = grad_lev(X[:, 0], X[:, 1])
    g = np.hypot(px, py)
    return np.column_stack([px / g, py / g])


_THETA_DENSE = np.linspace(0.0, 2.0 * np.pi, 40001)
_XB_DENSE = boundary_points(_THETA_DENSE)
_SEG = np.hypot(np.diff(_XB_DENSE[:, 0]), np.diff(_XB_DENSE[:, 1]))
_CUM = np.concatenate([[0.0], np.cumsum(_SEG)])
PERIMETER = float(_CUM[-1])
AREA = float(0.5 * np.sum(
    _XB_DENSE[:-1, 0] * _XB_DENSE[1:, 1] - _XB_DENSE[1:, 0] * _XB_DENSE[:-1, 1]))


def boundary_sample(n: int, rng: np.random.Generator,
                    uniform: bool = False) -> np.ndarray:
    """在 ∂Ω 上按**弧长**均匀地取 n 个点。

    ``uniform=True`` 时用等间距（用于评估网格），否则在每段内随机抖动，
    避免点集是规则的而让 kNN 权重退化。
    """
    n = int(n)
    if uniform:
        s = (np.arange(n) + 0.5) / n * PERIMETER
    else:
        s = (np.arange(n) + rng.random(n)) / n * PERIMETER
    # 弧长 → θ（在稠密采样上插值）
    th = np.interp(s, _CUM, _THETA_DENSE)
    return boundary_points(th)


def bounding_box(pad: float = 0.0):
    """区域的外接盒 ``(lo, hi)``，形状 (2,)，两维相同。"""
    hi = float(np.abs(_XB_DENSE).max()) + pad
    return np.array([-hi, -hi]), np.array([hi, hi])
