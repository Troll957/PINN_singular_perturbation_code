"""奇异摄动抛物型问题的算例 (manufactured solutions)。

方程形式::

    u_t - eps * u_xx + p(x) * u_x + q(x) * u = f(x, t),
    (x, t) in (0, 1) x (0, T],       0 < eps << 1

边界/初值条件::

    u(0, t) = g_L(t),  u(1, t) = g_R(t),  u(x, 0) = u0(x)

这里给出三个具有闭式解析解的算例, 右端项 f 由解析解反代得到,
因此可以精确计算数值误差:

* ``left``     : 边界层位于 x = 0  (与工作区 test2.py 中的算例一致)
* ``right``    : 边界层位于 x = 1  (检验自适应网格能否自动定位层)
* ``interior`` : 内层位于 x = 1/2 (p(x) 在 x = 1/2 处有零点)

所有方法同时支持 numpy 数组与 torch.Tensor 输入。
"""

from __future__ import annotations

import numpy as np
import torch

PI = 3.14159265358979323846


def _xp(*arrays):
    """根据输入类型返回数组后端 (torch 或 numpy)。"""
    for a in arrays:
        if isinstance(a, torch.Tensor):
            return torch
    return np


class ParabolicSPP:
    """奇异摄动抛物型问题基类。"""

    name = "base"
    layer = "none"

    def __init__(self, eps: float, T: float = 1.0):
        if eps <= 0:
            raise ValueError("eps 必须为正")
        self.eps = float(eps)
        self.T = float(T)

    # ------------------------------------------------------------------
    # 方程系数
    # ------------------------------------------------------------------
    def p(self, x):
        xp = _xp(x)
        return 1.0 + x

    def q(self, x):
        xp = _xp(x)
        return xp.ones_like(x)

    @property
    def sqrt_eps(self) -> float:
        return float(np.sqrt(self.eps))

    # ------------------------------------------------------------------
    # 解析解 / 右端项 / 定解条件
    # ------------------------------------------------------------------
    def exact(self, x, t):
        raise NotImplementedError

    def forcing(self, x, t):
        raise NotImplementedError

    def ic(self, x):
        """初值 u(x, 0)。"""
        xp = _xp(x)
        return self.exact(x, xp.zeros_like(x))

    def bc_left(self, t):
        """左边界值 u(0, t)。"""
        xp = _xp(t)
        return self.exact(xp.zeros_like(t), t)

    def bc_right(self, t):
        """右边界值 u(1, t)。"""
        xp = _xp(t)
        return self.exact(xp.ones_like(t), t)

    def __repr__(self):  # pragma: no cover - 仅用于日志
        return f"{type(self).__name__}(eps={self.eps:g}, T={self.T:g})"


class LeftBoundaryLayer(ParabolicSPP):
    """边界层位于 x = 0。

    解析解::

        u(x, t) = (1 - exp(-x / sqrt(eps))) * cos(pi t)

    代入方程 (p = 1 + x, q = 1) 得::

        f = -pi (1 - e) sin(pi t) + (1 + (1 + x) / sqrt(eps) * e) cos(pi t),
        e = exp(-x / sqrt(eps))
    """

    name = "left"
    layer = "x=0"

    def exact(self, x, t):
        xp = _xp(x, t)
        e = xp.exp(-x / self.sqrt_eps)
        return (1.0 - e) * xp.cos(PI * t)

    def forcing(self, x, t):
        xp = _xp(x, t)
        s = self.sqrt_eps
        e = xp.exp(-x / s)
        return (-PI * (1.0 - e) * xp.sin(PI * t)
                + (1.0 + (1.0 + x) / s * e) * xp.cos(PI * t))


class RightBoundaryLayer(ParabolicSPP):
    """边界层位于 x = 1 (镜像算例)。

    解析解::

        u(x, t) = (1 - exp(-(1 - x) / sqrt(eps))) * cos(pi t)

    代入方程 (p = 1 + x, q = 1) 得::

        f = -pi (1 - e) sin(pi t) + (1 - (1 + x) / sqrt(eps) * e) cos(pi t),
        e = exp(-(1 - x) / sqrt(eps))
    """

    name = "right"
    layer = "x=1"

    def exact(self, x, t):
        xp = _xp(x, t)
        e = xp.exp(-(1.0 - x) / self.sqrt_eps)
        return (1.0 - e) * xp.cos(PI * t)

    def forcing(self, x, t):
        xp = _xp(x, t)
        s = self.sqrt_eps
        e = xp.exp(-(1.0 - x) / s)
        return (-PI * (1.0 - e) * xp.sin(PI * t)
                + (1.0 - (1.0 + x) / s * e) * xp.cos(PI * t))


class InteriorLayer(ParabolicSPP):
    """内层位于 x = 1/2 (对流系数 p(x) = x - 1/2 在 x = 1/2 变号)。

    解析解::

        u(x, t) = (1 - exp(-(x - 1/2)^2 / (2 eps))) * cos(pi t)

    代入方程 (p = x - 1/2, q = 1), 令 y = x - 1/2, s = y^2 / (2 eps)::

        phi    = 1 - e^{-s}
        phi_x  = (y / eps) e^{-s}
        phi_xx = (1 / eps - y^2 / eps^2) e^{-s}

        -eps phi_xx + y phi_x + phi = 1 - 2 e^{-s} + 2 y^2 / eps * e^{-s}

    即::

        f = -pi (1 - e^{-s}) sin(pi t)
            + (1 - 2 e^{-s} + 2 y^2 / eps * e^{-s}) cos(pi t)
    """

    name = "interior"
    layer = "x=1/2"

    def p(self, x):
        xp = _xp(x)
        return x - 0.5

    def exact(self, x, t):
        xp = _xp(x, t)
        y = x - 0.5
        return (1.0 - xp.exp(-y * y / (2.0 * self.eps))) * xp.cos(PI * t)

    def forcing(self, x, t):
        xp = _xp(x, t)
        y = x - 0.5
        e = xp.exp(-y * y / (2.0 * self.eps))
        return (-PI * (1.0 - e) * xp.sin(PI * t)
                + (1.0 - 2.0 * e + 2.0 * y * y / self.eps * e) * xp.cos(PI * t))


_PROBLEMS = {
    "left": LeftBoundaryLayer,
    "right": RightBoundaryLayer,
    "interior": InteriorLayer,
}


def make_problem(name: str, eps: float, T: float = 1.0) -> ParabolicSPP:
    """按名称创建算例。"""
    if name not in _PROBLEMS:
        raise KeyError(f"未知算例 {name!r}, 可选: {sorted(_PROBLEMS)}")
    return _PROBLEMS[name](eps=eps, T=T)
