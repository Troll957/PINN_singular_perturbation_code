"""二维空间 + 时间的奇异摄动**反源**问题（制造解）。

两个算例
--------
``moving``（默认，推荐）
    源项是一条**随对流向下游扫过区域的薄脊**，未知量是它的轨迹
    :math:`\\phi=(A,s_0,v)`：振幅、初始位置、速度。

``static``
    源项是一条**固定在** :math:`x=x_s` 的薄脊，未知量
    :math:`\\phi=(A,x_s,y_s,\\rho)`。作为对照（静止层）。

公共形式
--------
.. math::

    \\mathcal L_\\varepsilon u:=u_t-\\varepsilon\\Delta u
    +\\mathbf b\\cdot\\nabla u+q\\,u=f,
    \\qquad (x,y)\\in\\Omega=(0,1)^2,\\ t\\in(0,T],

``b=(b1,b2)``、``q`` 为常数，``0 < eps << 1``。

moving 算例的制造解
-------------------
记 :math:`s(t)=s_0+vt`、:math:`z=x-s(t)`、:math:`C=\\cos\\pi t`、:math:`S=\\sin\\pi t`，

.. math::

    u^*(x,y,t)=A\\,\\underbrace{e^{-z^2/(2\\varepsilon)}}_{G}
    \\,\\underbrace{(1-e^{-y/\\sqrt\\varepsilon})}_{P}\\,C ,

则源项有闭式（已用 sympy 验证 :math:`\\mathcal L_\\varepsilon u^*\\equiv f`）：

.. math::

    f^*=A\\,G\\,C\\Bigl\\{P\\Bigl[\\frac{(v-b_1)z-z^2}{\\varepsilon}+q+1\\Bigr]
        +Q\\Bigr\\}-\\pi A\\,G\\,P\\,S,
    \\qquad Q=e^{-y/\\sqrt\\varepsilon}.

推导（把 :math:`u^*` 代入算子，两边除以 :math:`A G C`）::

    u_t      /(AGC) = v z/eps            （因为 G_t = v (z/eps) G）
    -eps u_xx/(AGC) = -(z^2/eps) + 1
    b1 u_x   /(AGC) = -b1 z/eps
    q u      /(AGC) = q
    -eps u_yy/(AGC) = Q                  （P''=-Q/eps，单独一项）

当 :math:`v=b_1` 时 :math:`u_t` 与 :math:`b\\cdot\\nabla u` 中的
:math:`z/\\varepsilon` 项相消——这正是正文 Case 3 取 ``b=(1,0)`` 且层速为 1 的原因。
本算例允许 :math:`v\\ne b_1`，此时那一项保留，是"源速"可辨识的来源。

static 算例的制造解
-------------------
.. math::

    u^*=A\\,e^{-(x-x_s)^2/(2\\varepsilon)}\\,e^{-(y-y_s)^2/(2\\rho^2)}\\,C,
    \\qquad
    f^*=A G H\\Bigl[\\Bigl(-\\frac{z^2+b_1z}{\\varepsilon}
      +1-\\frac{\\varepsilon w^2}{\\rho^4}+\\frac{\\varepsilon}{\\rho^2}
      -\\frac{b_2w}{\\rho^2}+q\\Bigr)C-\\pi S\\Bigr].

注意
----
* 源项都由 :math:`u^*` 反代得到，所以反演时 :math:`f,g,u_0` 是**给定的数据**
  （用真值参数生成），只对 :math:`\\phi` 寻优。
* ``A`` 默认限制为正（物理上是"源"），这同时避免了振幅变号造成的伪极小。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

PI = math.pi

# 所有可能出现的参数（按算例取子集）
PARAM_NAMES_ALL: Tuple[str, ...] = ("A", "s0", "v", "xs", "ys", "rho")

PARAM_LABELS: Dict[str, str] = {
    "A": r"$A$",
    "s0": r"$s_0$",
    "v": r"$v$",
    "xs": r"$x_s$",
    "ys": r"$y_s$",
    "rho": r"$\rho$",
}

DEFAULT_BOUNDS: Dict[str, Optional[Tuple[float, float]]] = {
    "A": (0.05, 10.0),
    "s0": (-0.20, 1.20),
    "v": (0.0, 2.0),
    "xs": (0.05, 0.95),
    "ys": (0.05, 0.95),
    "rho": (0.02, 0.50),
}


def _xp(*arrays):
    for a in arrays:
        if isinstance(a, torch.Tensor):
            return torch
    return np


# ======================================================================
# 1. 可训练的源项参数
# ======================================================================


class SourceParams(torch.nn.Module):
    """源项参数，用 sigmoid 重参数化到各自的物理区间内。

    ``names`` 决定这个算例有哪些参数（moving: ``("A","s0","v")``，
    static: ``("A","xs","ys","rho")``）。
    """

    def __init__(self, names: Sequence[str], init: Dict[str, float],
                 bounds: Optional[Dict[str, Optional[Tuple[float, float]]]] = None,
                 trainable: Optional[Sequence[str]] = None):
        super().__init__()
        self.names = tuple(names)
        self.bounds = dict(DEFAULT_BOUNDS if bounds is None else bounds)
        trainable = self.names if trainable is None else tuple(trainable)
        self.raw = torch.nn.ParameterDict()
        for k in self.names:
            lo, hi = self._bound(k)
            v = float(init[k])
            if lo is None:
                raw = v
            else:
                s = (v - lo) / (hi - lo)
                s = min(max(s, 1e-4), 1.0 - 1e-4)
                raw = math.log(s / (1.0 - s))
            self.raw[k] = torch.nn.Parameter(
                torch.tensor(raw, dtype=torch.float32),
                requires_grad=(k in trainable))

    def _bound(self, k: str):
        b = self.bounds.get(k)
        return (None, None) if b is None else (float(b[0]), float(b[1]))

    def values(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k in self.names:
            p = self.raw[k]
            lo, hi = self._bound(k)
            out[k] = p if lo is None else lo + (hi - lo) * torch.sigmoid(p)
        return out

    @torch.no_grad()
    def numpy_values(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.values().items()}

    def trainable_names(self) -> Tuple[str, ...]:
        return tuple(k for k in self.names if self.raw[k].requires_grad)


# ======================================================================
# 2. 真值
# ======================================================================


@dataclass
class MovingTruth:
    """moving 算例的真值：振幅 / 初始位置 / 速度。"""

    A: float = 2.0
    s0: float = 0.15
    v: float = 0.70

    def as_dict(self) -> Dict[str, float]:
        return {"A": float(self.A), "s0": float(self.s0), "v": float(self.v)}


@dataclass
class StaticTruth:
    """static 算例的真值。"""

    A: float = 2.0
    xs: float = 0.30
    ys: float = 0.35
    rho: float = 0.08

    def as_dict(self) -> Dict[str, float]:
        return {"A": float(self.A), "xs": float(self.xs),
                "ys": float(self.ys), "rho": float(self.rho)}


# 兼容旧名字
TruthParams = StaticTruth


# ======================================================================
# 3. 问题基类
# ======================================================================


class BaseInverse2D:
    """二维空间 + 时间、带未知源项的奇异摄动抛物问题。"""

    dim = 2
    name = "base"
    param_names: Tuple[str, ...] = ()

    def __init__(self, eps: float, b: Tuple[float, float] = (1.0, 0.0),
                 q: float = 1.0, T: float = 1.0, truth=None):
        if eps <= 0:
            raise ValueError("eps 必须为正")
        self.eps = float(eps)
        self.b1, self.b2 = float(b[0]), float(b[1])
        self.q = float(q)
        self.T = float(T)
        self.truth = truth if truth is not None else self.default_truth()

    # ---------- 基本信息 ----------
    @staticmethod
    def default_truth():
        raise NotImplementedError

    @property
    def sqrt_eps(self) -> float:
        return float(np.sqrt(self.eps))

    @property
    def truth_dict(self) -> Dict[str, float]:
        return self.truth.as_dict()

    def faces(self) -> Sequence[Tuple[int, float]]:
        return [(0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0)]

    # ---------- 层的位置（静止算例是常数） ----------
    def layer_curve(self, t):
        """层心的 x 位置 :math:`s(t)`（数组）。"""
        raise NotImplementedError

    # ---------- 数据 ----------
    def exact(self, Z, P: Optional[Dict] = None):
        P = self.truth_dict if P is None else P
        u, _ = self.profiles(Z[:, 0:1], Z[:, 1:2], Z[:, 2:3], P)
        return u

    def forcing(self, Z, P: Dict):
        _, f = self.profiles(Z[:, 0:1], Z[:, 1:2], Z[:, 2:3], P)
        return f

    def ic(self, X, P: Optional[Dict] = None):
        P = self.truth_dict if P is None else P
        if isinstance(X, torch.Tensor):
            zero = torch.zeros((X.shape[0], 1), dtype=X.dtype, device=X.device)
        else:
            zero = np.zeros((X.shape[0], 1))
        u, _ = self.profiles(X[:, 0:1], X[:, 1:2], zero, P)
        return u

    def bc(self, Z, P: Optional[Dict] = None):
        return self.exact(Z, P)

    # ---------- 诊断（一律用真值） ----------
    def inside_layer(self, Z, P: Optional[Dict] = None,
                     factor: float = 4.0) -> np.ndarray:
        Z = np.asarray(Z)
        s = np.asarray(self.layer_curve(Z[:, 2]), dtype=float)
        return np.abs(Z[:, 0] - s) <= factor * self.sqrt_eps

    def ridge_weight(self, Z, P: Optional[Dict] = None,
                     factor: float = 4.0) -> np.ndarray:
        """连续版的"层内程度"，与 ``inside_layer`` 相同（保留接口）。"""
        return self.inside_layer(Z, P, factor)

    def source_error(self, P_hat: Dict, n: int = 161, nt: int = 21) -> Dict[str, float]:
        xs = np.linspace(0.0, 1.0, n)
        ts = np.linspace(0.0, self.T, nt)
        X, Y, Tt = np.meshgrid(xs, xs, ts, indexing="ij")
        Z = np.stack([X.ravel(), Y.ravel(), Tt.ravel()], axis=1)
        f_true = np.asarray(self.forcing(Z, self.truth_dict), dtype=float).ravel()
        f_hat = np.asarray(self.forcing(Z, P_hat), dtype=float).ravel()
        err = f_hat - f_true
        den = np.linalg.norm(f_true)
        return {"src_rel_l2": float(np.linalg.norm(err) / den) if den > 0 else float("nan"),
                "src_max_err": float(np.abs(err).max()),
                "src_max": float(np.abs(f_true).max())}

    # ---------- 灵敏度（解析） ----------
    def sensitivity_matrix(self, sensors: np.ndarray,
                           P: Optional[Dict] = None) -> np.ndarray:
        raise NotImplementedError


# ======================================================================
# 4. moving：运动的源（推荐）
# ======================================================================


class MovingSource2D(BaseInverse2D):
    r"""源项是一条以速度 :math:`v` 扫过区域的薄脊。

    .. math::

        u^*=A\,e^{-(x-s(t))^2/(2\varepsilon)}\,(1-e^{-y/\sqrt\varepsilon})\cos\pi t,
        \qquad s(t)=s_0+vt,

    .. math::

        f^*=A\,G\,C\Bigl\{P\Bigl[\frac{(v-b_1)z-z^2}{\varepsilon}+q+1\Bigr]+Q\Bigr\}
            -\pi A\,G\,P\,S .

    * :math:`t=0` 时层在 :math:`x=s_0`，:math:`t=T` 时到 :math:`x=s_0+vT`；
    * 取 :math:`v=b_1=1` 时就是正文 Case 3 的移动层（此时 :math:`z/\varepsilon`
      项相消）；这里允许 :math:`v\ne b_1`，那一项正是"速度"可辨识的来源；
    * 沿 :math:`y=0` 还有一条固定宽度 :math:`\sqrt\varepsilon` 的壁面层
      （来自 :math:`P`），它不随时间移动。
    """

    name = "moving"
    param_names = ("A", "s0", "v")

    def __init__(self, eps: float, b=(1.0, 0.0), q: float = 1.0,
                 T: float = 1.0, truth=None, wall: bool = False,
                 yc: float = 0.5, rho_y: float = 0.20):
        super().__init__(eps=eps, b=b, q=q, T=T, truth=truth)
        # wall=True 时沿 y=0 还有一条 O(sqrt(eps)) 的壁面层（正文 Case 3 的设定）；
        # wall=False 时 y 方向是光滑剖面（推荐：此时唯一的层就是移动层）。
        self.wall = bool(wall)
        self.yc = float(yc)
        self.rho_y = float(rho_y)

    @staticmethod
    def default_truth():
        return MovingTruth()

    # ---------- y 方向剖面 H 与 -eps H''/H ----------
    def y_profile(self, Y):
        xp = _xp(Y)
        if self.wall:
            Q = xp.exp(-Y / self.sqrt_eps)
            H = 1.0 - Q
            hlog2 = -Q / (self.eps * H)          # H''/H
        else:
            w = Y - self.yc
            r2 = self.rho_y * self.rho_y
            H = xp.exp(-w * w / (2.0 * r2))
            hlog2 = w * w / (r2 * r2) - 1.0 / r2
        return H, hlog2

    # ---------- 层的位置 ----------
    def layer_curve(self, t):
        tr = self.truth
        return float(tr.s0) + float(tr.v) * np.asarray(t, dtype=float)

    def trajectory(self, t, P: Optional[Dict] = None):
        """给定参数下的轨迹（作图用）。"""
        P = self.truth_dict if P is None else P
        return float(P["s0"]) + float(P["v"]) * np.asarray(t, dtype=float)

    # ---------- 剖面 ----------
    def profiles(self, X, Y, t, P: Dict):
        xp = _xp(X, Y, t)
        eps, b1, q = self.eps, self.b1, self.q
        A, s0, v = P["A"], P["s0"], P["v"]

        z = X - (s0 + v * t)
        G = xp.exp(-z * z / (2.0 * eps))
        H, hlog2 = self.y_profile(Y)
        C = xp.cos(PI * t)
        S = xp.sin(PI * t)

        u = A * G * H * C
        f = (A * G * C * (H * (((v - b1) * z - z * z) / eps + q + 1.0
                               - eps * hlog2))
             - PI * A * G * H * S)
        return u, f

    # ---------- 灵敏度 ----------
    def sensitivity_matrix(self, sensors: np.ndarray,
                           P: Optional[Dict] = None) -> np.ndarray:
        r"""数据对 :math:`(A,s_0,v)` 的灵敏度。

        .. math::

            \partial_Au=GPC,\quad
            \partial_{s_0}u=A\frac{z}{\varepsilon}GPC,\quad
            \partial_{v}u=At\frac{z}{\varepsilon}GPC .
        """
        P = self.truth_dict if P is None else P
        A, s0, v = (float(P[k]) for k in self.param_names)
        eps = self.eps
        z = sensors[:, 0] - (s0 + v * sensors[:, 2])
        G = np.exp(-z * z / (2.0 * eps))
        H, _ = self.y_profile(sensors[:, 1])
        C = np.cos(PI * sensors[:, 2])
        base = A * G * H * C
        return np.column_stack([
            G * H * C,
            base * z / eps,
            base * sensors[:, 2] * z / eps,
        ])


# ======================================================================
# 5. static：静止的源（对照）
# ======================================================================


class StaticSource2D(BaseInverse2D):
    r"""源项是一条固定在 :math:`x=x_s` 的二维薄脊（上一版算例，保留作对照）。"""

    name = "static"
    param_names = ("A", "xs", "ys", "rho")

    @staticmethod
    def default_truth():
        return StaticTruth()

    def layer_curve(self, t):
        return np.full(np.shape(np.asarray(t, dtype=float)), float(self.truth.xs))

    def profiles(self, X, Y, t, P: Dict):
        xp = _xp(X, Y, t)
        eps, b1, b2, q = self.eps, self.b1, self.b2, self.q
        A, xs, ys, rho = P["A"], P["xs"], P["ys"], P["rho"]

        z = X - xs
        w = Y - ys
        rho2 = rho * rho
        G = xp.exp(-z * z / (2.0 * eps))
        H = xp.exp(-w * w / (2.0 * rho2))
        C = xp.cos(PI * t)
        S = xp.sin(PI * t)

        u = A * G * H * C
        B = (-(z * z + b1 * z) / eps + 1.0
             - eps * w * w / (rho2 * rho2) + eps / rho2
             - b2 * w / rho2 + q)
        f = A * G * H * (B * C - PI * S)
        return u, f

    def sensitivity_matrix(self, sensors: np.ndarray,
                           P: Optional[Dict] = None) -> np.ndarray:
        P = self.truth_dict if P is None else P
        A, xs, ys, rho = (float(P[k]) for k in self.param_names)
        eps = self.eps
        z = sensors[:, 0] - xs
        w = sensors[:, 1] - ys
        G = np.exp(-z * z / (2.0 * eps))
        H = np.exp(-w * w / (2.0 * rho * rho))
        C = np.cos(PI * sensors[:, 2])
        base = A * G * H * C
        return np.column_stack([
            G * H * C,
            base * z / eps,
            base * w / (rho * rho),
            base * w * w / (rho ** 3),
        ])


# 兼容旧名字
InverseSource2D = StaticSource2D

_PROBLEMS = {"moving": MovingSource2D, "static": StaticSource2D}


def make_problem(name: str = "moving", eps: float = 1e-3,
                 b: Tuple[float, float] = (1.0, 0.0), q: float = 1.0,
                 T: float = 1.0, truth=None, **kw) -> BaseInverse2D:
    if name not in _PROBLEMS:
        raise KeyError(f"未知算例 {name!r}，可选 {sorted(_PROBLEMS)}")
    if name != "moving":
        kw = {}
    return _PROBLEMS[name](eps=eps, b=b, q=q, T=T, truth=truth, **kw)


# ======================================================================
# 6. Cramér–Rao 下界
# ======================================================================


def crlb(problem: BaseInverse2D, sensors: np.ndarray, sigma: float,
         P: Optional[Dict] = None) -> Dict[str, float]:
    """给定数据与噪声时，各参数能达到的最好精度（Fisher 信息的下界）。"""
    P = problem.truth_dict if P is None else P
    J = problem.sensitivity_matrix(sensors, P)
    F = J.T @ J / max(sigma * sigma, 1e-30)
    try:
        cov = np.linalg.inv(F)
        sd = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        sd = np.full(J.shape[1], np.inf)
    out = {"cond_fisher": float(np.linalg.cond(F))}
    for i, k in enumerate(problem.param_names):
        out[f"crlb_{k}"] = float(sd[i])
        out[f"crlb_rel_{k}"] = float(sd[i] / max(abs(float(P[k])), 1e-12))
    return out


def sensitivity_matrix(problem: BaseInverse2D, sensors: np.ndarray,
                       P: Optional[Dict] = None) -> np.ndarray:
    """向后兼容的包装。"""
    return problem.sensitivity_matrix(sensors, P)
