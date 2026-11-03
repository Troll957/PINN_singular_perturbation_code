"""把方法推广到 d 维空间 + 时间的自包含实现（教程用模板）。

本模块用**一个文件**给出高维版本所需的三处推广，便于对照阅读：

1. **问题定义**（:class:`NDProblem`）：区域为 $(0,1)^d\\times(0,T]$，
   算子为 $u_t-\\varepsilon\\Delta u+b(x)\\cdot\\nabla u+q(x)u=f$。
   解、右端项、初值、边值全部以"整点坐标" $(N,d+1)$ 数组为输入。
2. **点集**（:class:`CollocationSetND`）：内部点、初值面 $t=0$、
   以及 $2d$ 个边界面 $\\{x_j=0\\}$、$\\{x_j=1\\}$。
3. **求积权重**（:func:`knn_quadrature_weights`）：$d$ 维时离散 Voronoi
   的 $n^{d}$ 网格不再可行，改用 **k 近邻体积估计**
   $\\omega_i\\propto r_k(z_i)^{D}$（$D=d+1$），再做归一化。

数值算例是二维角点层问题：$u^*=\\varphi(x)\\varphi(y)\\cos(\\pi t)$，
$\\varphi(z)=1-e^{-z/\\sqrt{\\varepsilon}}$，层同时贴在 $x=0$、$y=0$ 以及角点 $(0,0)$。

用法::

    arch -arm64 .venv/bin/python demo_2d.py --eps 1e-3
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .model import MLP
from .problems import PI, _xp


def _ones_rows(Z, ncols):
    """在 ``Z`` 所在的设备与后端上创建形状 ``(N, ncols)`` 的全 1 张量。

    直接写 ``xp.ones(...)`` 会用 torch 的默认设备 (CPU), 导致 GPU 上
    ``reaction(Z) * u`` 出现 "found at least two devices" 错误。
    """
    if isinstance(Z, torch.Tensor):
        return torch.ones((Z.shape[0], ncols), dtype=Z.dtype, device=Z.device)
    return np.ones((Z.shape[0], ncols))
from .sampling import select_distribution, select_topm

# ======================================================================
# 1. 问题定义
# ======================================================================


class NDProblem:
    """d 维空间 + 时间的抛物型奇异摄动问题基类。

    约定：所有方法的输入 ``Z`` 形状为 ``(N, d+1)``，前 ``d`` 列是空间坐标
    $x_1,\\dots,x_d$，最后一列是时间 $t$。这样做的原因是网络、求积权重、
    边界处理都可以**不区分空间与时间**地统一处理。
    """

    dim: int = 1
    name: str = "base"

    def __init__(self, eps: float, T: float = 1.0):
        self.eps = float(eps)
        self.T = float(T)

    @property
    def sqrt_eps(self) -> float:
        return float(np.sqrt(self.eps))

    # ---------- 算子系数 ----------
    def convection(self, Z):
        """对流系数 $b(x)\\in\\mathbb R^d$，返回与 ``Z`` 同后端、形状 ``(N, d)``。"""
        xp = _xp(Z)
        return _ones_rows(Z, self.dim) if hasattr(Z, "shape") else 1.0

    def reaction(self, Z):
        """反应系数 $q(x)$，返回形状 ``(N, 1)``。"""
        return _ones_rows(Z, 1)

    # ---------- 问题数据 ----------
    def exact(self, Z):
        raise NotImplementedError

    def forcing(self, Z):
        raise NotImplementedError

    def ic(self, X):
        """初值 $u(x,0)$，``X`` 为 ``(N, d)`` 空间坐标。"""
        raise NotImplementedError

    def bc(self, Z, axis: int, value: float):
        """边界 $x_{\\text{axis}}=\\text{value}$ 上的 Dirichlet 数据。

        注意 ``Z`` 是**完整的** $(N,d+1)$ 点（含时间列）：边界数据一般依赖
        $t$，若只传空间坐标就会把 $u^*|_{t=0}$ 当成全部时刻的边值。
        """
        raise NotImplementedError

    def faces(self) -> List[Tuple[int, float]]:
        """区域的全部 Dirichlet 面。"""
        return [(j, v) for j in range(self.dim) for v in (0.0, 1.0)]

    # ---------- 层内判定 ----------
    def inside_layer(self, Z, factor: float = 4.0) -> np.ndarray:
        """判定点是否落在层内 (布尔数组)。

        默认认为层贴在第一维的左边界上。**层随时间移动**的算例必须重写
        这个方法, 否则 "层内点占比" 这个诊断量会用错参考位置。
        """
        Z = np.asarray(Z)
        return Z[:, 0] <= factor * self.sqrt_eps


class CornerLayer2D(NDProblem):
    r"""二维角点层算例（$d=2$）。

    .. math::
        u^*(x,y,t)=\varphi(x)\varphi(y)\cos(\pi t),\qquad
        \varphi(z)=1-e^{-z/\sqrt{\varepsilon}},

    取 $b=(1+x,\,1+y)$、$q\equiv1$。代入

    .. math::
        \mathcal L u=u_t-\varepsilon\Delta u+b\cdot\nabla u+qu

    得（记 $s=\sqrt\varepsilon$，$\varphi_x=\varphi(x)$ 等）

    .. math::
        f=-\pi\varphi_x\varphi_y\sin(\pi t)
        +\Bigl[e^{-x/s}\varphi_y+\varphi_x e^{-y/s}
        +\tfrac{1+x}{s}e^{-x/s}\varphi_y
        +\tfrac{1+y}{s}\varphi_x e^{-y/s}
        +\varphi_x\varphi_y\Bigr]\cos(\pi t).

    解在 $x=0$、$y=0$ 两条边上有宽度 $O(\sqrt\varepsilon)$ 的层，
    并在角点 $(0,0)$ 处叠加成角点层。
    """

    dim = 2
    name = "corner2d"

    def convection(self, Z):
        xp = _xp(Z)
        x, y = Z[:, 0:1], Z[:, 1:2]
        return xp.concatenate([1.0 + x, 1.0 + y], axis=1)

    def reaction(self, Z):
        return _ones_rows(Z, 1)

    @staticmethod
    def _phi(x, eps):
        xp = _xp(x)
        return 1.0 - xp.exp(-x / float(np.sqrt(eps)))

    def exact(self, Z):
        xp = _xp(Z)
        px = self._phi(Z[:, 0:1], self.eps)
        py = self._phi(Z[:, 1:2], self.eps)
        return (px * py * xp.cos(PI * Z[:, 2:3])).reshape(-1)

    def forcing(self, Z):
        xp = _xp(Z)
        eps, s = self.eps, self.sqrt_eps
        x, y, t = Z[:, 0:1], Z[:, 1:2], Z[:, 2:3]
        px, py = self._phi(x, eps), self._phi(y, eps)
        ex, ey = xp.exp(-x / s), xp.exp(-y / s)
        c, sn = xp.cos(PI * t), xp.sin(PI * t)
        return (-PI * px * py * sn
                + (ex * py + px * ey
                   + (1.0 + x) / s * ex * py
                   + (1.0 + y) / s * px * ey
                   + px * py) * c).reshape(-1)

    def ic(self, X):
        xp = _xp(X)
        return (self._phi(X[:, 0:1], self.eps)
                * self._phi(X[:, 1:2], self.eps)).reshape(-1)

    def bc(self, Z, axis: int, value: float):
        # 本算例用精确解给出 Dirichlet 数据（制造解的标准做法）
        return self.exact(Z)


class MovingLayer2D(NDProblem):
    r"""二维移动层算例（$d=2$，层的位置随时间变化）。

    层心沿 $x_1$ 方向以单位速度平移，

    .. math::
        s(t)=t,\qquad z=x_1-s(t)=x_1-t,

    并取对流系数 $b=(1,0)$，使层恰好被对流输运（于是 $u_t$ 与
    $b\cdot\nabla u$ 中的 $z/\varepsilon$ 项相消，见下文）。制造解为

    .. math::
        u^*(x,t)=G(z)\,P(x_2)\cos(\pi t),\qquad
        G(z)=e^{-z^2/(2\varepsilon)},\qquad
        P(x_2)=1-e^{-x_2/\sqrt{\varepsilon}},

    取 $q\equiv1$。这里 $G$ 是**双侧**高斯型剖面：$z<0$ 时仍有定义且光滑，
    因此层可以在区域内平移，而不像单侧剖面 $1-e^{-z/\sqrt\varepsilon}$
    那样只适用于 $z\ge0$。

    代入 :math:`\mathcal L u=u_t-\varepsilon\Delta u+b\cdot\nabla u+qu`。记
    $s=\sqrt\varepsilon$、$C=\cos(\pi t)$、$S=\sin(\pi t)$，由
    $G'=-zG/\varepsilon$、$G''=(z^2/\varepsilon^2-1/\varepsilon)G$、
    $P'=e^{-x_2/s}/s$、$P''=-e^{-x_2/s}/\varepsilon$ 可得

    .. math::
        u_t=\frac{z}{\varepsilon}GPC-\pi GPS,\qquad
        b\cdot\nabla u=-\frac{z}{\varepsilon}GPC,

    二者相加后 $z/\varepsilon$ 项相消，只剩 $-\pi GPS$；再由

    .. math::
        \varepsilon\Delta u=\Bigl(\frac{z^2}{\varepsilon}-1\Bigr)GPC
        -G\,e^{-x_2/s}C

    得

    .. math::
        f=-\pi GPS+GC\Bigl[2P-\frac{z^2}{\varepsilon}P+e^{-x_2/s}\Bigr].

    .. note::
       $s(t)=t$ 与 $b_1=1$ 必须一致。若取更一般的 $s(t)$，则
       $b_1=\dot s(t)$ 需要随时间变化，而 $\mathbf b$ 在此问题中是
       $x$ 的函数，不能依赖于 $t$。

    解在 $t=0$ 时是 $x_1=0$ 处的边界层，随 $t$ 增大向内平移，
    到 $t=1$ 时到达 $x_1=1$。因此它同时考察了层随时间的移动与
    $x_2=0$ 方向固定的层。
    """

    dim = 2
    name = "moving"

    def convection(self, Z):
        xp = _xp(Z)
        one = _ones_rows(Z, 1)
        return xp.concatenate([one, one * 0.0], axis=1)

    def reaction(self, Z):
        return _ones_rows(Z, 1)

    # ---------- 层心位置 ----------
    def layer_center(self, t):
        """层心 $s(t)=t$（与 $x_1$ 同单位）。"""
        return np.asarray(t, dtype=float)

    @staticmethod
    def _gauss(z, eps):
        xp = _xp(z)
        return xp.exp(-z * z / (2.0 * eps))

    @staticmethod
    def _phi(x, eps):
        xp = _xp(x)
        return 1.0 - xp.exp(-x / float(np.sqrt(eps)))

    def exact(self, Z):
        xp = _xp(Z)
        x1, x2, t = Z[:, 0:1], Z[:, 1:2], Z[:, 2:3]
        g = self._gauss(x1 - t, self.eps)
        p = self._phi(x2, self.eps)
        return (g * p * xp.cos(PI * t)).reshape(-1)

    def forcing(self, Z):
        xp = _xp(Z)
        eps, s = self.eps, self.sqrt_eps
        x1, x2, t = Z[:, 0:1], Z[:, 1:2], Z[:, 2:3]
        z = x1 - t
        g = self._gauss(z, eps)
        p = self._phi(x2, eps)
        c, sn = xp.cos(PI * t), xp.sin(PI * t)
        return (-PI * g * p * sn
                + g * c * (2.0 * p - (z * z / eps) * p
                           + xp.exp(-x2 / s))).reshape(-1)

    def ic(self, X):
        xp = _xp(X)
        return (self._gauss(X[:, 0:1], self.eps)
                * self._phi(X[:, 1:2], self.eps)).reshape(-1)

    def bc(self, Z, axis: int, value: float):
        return self.exact(Z)

    def inside_layer(self, Z, factor: float = 4.0) -> np.ndarray:
        r"""层心在 $x_1=s(t)=t$, 故判据为 $|x_1-t|\le\text{factor}\sqrt\varepsilon$。"""
        Z = np.asarray(Z)
        return np.abs(Z[:, 0] - Z[:, 2]) <= factor * self.sqrt_eps


_PROBLEMS_ND = {"corner2d": CornerLayer2D, "moving": MovingLayer2D}


def make_problem_nd(name: str, eps: float, T: float = 1.0) -> NDProblem:
    if name not in _PROBLEMS_ND:
        raise KeyError(f"未知算例 {name!r}，可选 {sorted(_PROBLEMS_ND)}")
    return _PROBLEMS_ND[name](eps=eps, T=T)


# ======================================================================
# 2. k 近邻求积权重
# ======================================================================


def knn_quadrature_weights(coords: np.ndarray, measure: float,
                           n_neighbors: int = 4, chunk: int = 2048) -> np.ndarray:
    r"""用 k 近邻半径估计求积权重，返回权重满足 $\sum_i\omega_i=\text{measure}$。

    设 $r_1\le r_2\le\cdots\le r_k$ 是 $z_i$ 到其 $k$ 个最近邻的距离。若局部
    点密度为 $\rho$，则半径 $r$ 的球内约有 $\rho\,\Omega_D r^D$ 个点，故

    .. math::
        \rho\ \approx\ \frac{k}{\Omega_D\,\bar r_i^{D}},\qquad
        \bar r_i=\Bigl(\frac1k\sum_{j=1}^{k}r_j\Bigr),

    其中 $D$ 是度量维数、$\Omega_D$ 是 $D$ 维单位球体积。单点"占有"的体积为
    $1/\rho$，把常数 $k,\Omega_D$ 约掉并强行归一化，得

    .. math::
        \boxed{\ \omega_i=\text{measure}\cdot
        \frac{\bar r_i^{\,D}}{\sum_j \bar r_j^{\,D}}\ }

    这是 Voronoi 胞体积的一个**与维数无关**的估计。高维时离散 Voronoi 的
    $n^D$ 网格不可行（$n=256,d=4$ 已是 $4\times10^9$ 个格点），而 kNN 的
    代价只有 $O(N^2)$。取 $k$ 个近邻距离的平均 $\bar r_i$ 而不是第 $k$ 近邻 $r_k$，
    可以抑制单点间距涨落；实测 $k=4$ 在精度与稳健性之间最平衡
    （见 `check_problems_nd.py` 的求积误差对比）。

    Parameters
    ----------
    coords : (N, D)
        用于度量距离的坐标。对内部点取全部 $d+1$ 列；对面 $x_j=\text{const}$
        取除第 $j$ 列以外的 $d$ 列（即把面参数化）。
    measure : float
        该点集所在流形的测度：内部点为 $|\Omega|=T$，初值面为 $1$，
        边界面为 $T$。
    """
    P = np.ascontiguousarray(coords, dtype=float)
    N, D = P.shape
    if N <= 1:
        return np.full(N, measure)
    k = int(min(max(1, n_neighbors), N - 1))

    rbar = np.empty(N)
    for s0 in range(0, N, chunk):
        c = P[s0:s0 + chunk]                                  # (c, D)
        d2 = ((c[:, None, :] - P[None, :, :]) ** 2).sum(-1)   # (c, N)
        # 第 0 近邻是自身（距离 0），故取 partition 的第 1..k 个
        part = np.partition(d2, k, axis=1)[:, 1:k + 1]
        rbar[s0:s0 + chunk] = np.sqrt(part).mean(axis=1)

    w = rbar ** D
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0:
        return np.full(N, measure / N)
    return w / tot * measure


# ======================================================================
# 3. 高维配点集
# ======================================================================


class CollocationSetND:
    """$d$ 维空间 + 时间的配点集。

    * ``interior`` : $(N, d+1)$，内部点
    * ``ic``       : $(N, d+1)$，$t=0$ 上的点
    * ``bc``       : ``{(axis, value): (N, d+1)}``，各 Dirichlet 面上的点
    """

    def __init__(self, dim: int, T: float = 1.0, seed: int = 0):
        self.dim = int(dim)
        self.D = self.dim + 1
        self.T = float(T)
        self.rng = np.random.default_rng(seed)
        self.interior = np.zeros((0, self.D))
        self.ic = np.zeros((0, self.D))
        self.bc: Dict[Tuple[int, float], np.ndarray] = {}

    # ---------------- 初始化 ----------------
    def initialise(self, n_interior: int, n_ic: int, n_bc: int,
                   faces: Sequence[Tuple[int, float]], method: str = "halton") -> None:
        """``n_bc`` 是**每个面**上的初始点数。"""
        from .sampling import halton
        if method == "halton":
            u = halton(n_interior, self.D)
        else:
            u = self.rng.random((n_interior, self.D))
        self.interior = self._to_box(u)

        u = self.rng.random((n_ic, self.dim))
        self.ic = np.column_stack([u, np.zeros(n_ic)])

        for axis, value in faces:
            u = self.rng.random((n_bc, self.dim))
            X = u.copy()
            X[:, axis] = value
            self.bc[(axis, value)] = np.column_stack([X, self.rng.random(n_bc) * self.T])

    def _to_box(self, u: np.ndarray) -> np.ndarray:
        """把 $[0,1]^{d+1}$ 的样本映射到 $[0,1]^d\\times[0,T]$。"""
        Z = u.copy()
        Z[:, -1] *= self.T
        return Z

    # ---------------- 增点 ----------------
    def add_interior(self, pts: np.ndarray) -> int:
        self.interior = np.concatenate([self.interior, np.atleast_2d(pts)], axis=0)
        return len(pts)

    def add_ic(self, X: np.ndarray) -> int:
        X = np.atleast_2d(X)
        pts = np.column_stack([X, np.zeros(X.shape[0])])
        self.ic = np.concatenate([self.ic, pts], axis=0)
        return len(pts)

    def add_bc(self, face: Tuple[int, float], pts: np.ndarray) -> int:
        pts = np.atleast_2d(pts)
        self.bc[face] = np.concatenate([self.bc.get(face, np.zeros((0, self.D))), pts], axis=0)
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
        return sum(a.shape[0] for a in self.bc.values())

    @property
    def total(self) -> int:
        return self.n_interior + self.n_ic + self.n_bc

    def candidate_pool(self, m: int) -> np.ndarray:
        r"""$\Omega$ 内均匀候选池；高维时纯随机即可（Halton 在高维会退化）。"""
        return self._to_box(self.rng.random((m, self.D)))

    def layer_fraction(self, eps: float, factor: float = 4.0,
                       problem=None) -> float:
        """内部点中落在层内的比例。

        传入 ``problem`` 时由 ``problem.inside_layer`` 决定判据, 这样层随时间
        移动的算例也能得到正确的统计量; 不传时退化为默认判据
        $x_1\\le\\text{factor}\\sqrt\\varepsilon$。
        """
        if problem is not None:
            return float(np.mean(problem.inside_layer(self.interior, factor)))
        thr = factor * float(np.sqrt(eps))
        return float(np.mean(self.interior[:, 0] <= thr))

    def corner_fraction(self, eps: float, factor: float = 4.0) -> float:
        """内部点中**所有**空间坐标都 $\\le\\text{factor}\\sqrt\\varepsilon$ 的比例。"""
        thr = factor * float(np.sqrt(eps))
        return float(np.mean(np.all(self.interior[:, :self.dim] <= thr, axis=1)))


# ======================================================================
# 4. 求解器
# ======================================================================


def pick_device(name: str = "auto") -> torch.device:
    """选择计算设备。

    Parameters
    ----------
    name : str
        ``"auto"``（默认）按 **CUDA > CPU** 自动选（理由见下）；
        也可以显式指定 ``"cuda"`` / ``"cuda:0"`` / ``"mps"`` / ``"cpu"``。

    Notes
    -----
    显式要求 ``"cuda"`` 但机器上没有可用 CUDA 时会**直接报错**，而不是悄悄退回
    CPU——否则你以为在用显卡，其实一直在用 CPU，白白浪费机时。
    """
    if name != "auto":
        dev = torch.device(name)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "指定了 cuda，但 torch.cuda.is_available() 为 False。逐条排查：\n"
                "  1) 装的是不是 CUDA 版 torch：\n"
                "     python -c \"import torch; print(torch.__version__, torch.version.cuda)\"\n"
                "     若 version.cuda 是 None，说明装成了 CPU 版，要重装 CUDA wheel。\n"
                "  2) 驱动是否正常：nvidia-smi\n"
                "  3) 是否被环境变量屏蔽：echo $CUDA_VISIBLE_DEVICES\n"
                "  4) 驱动版本是否低于 torch 要求的 CUDA runtime 版本。"
            )
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    # 注意: 这里**不自动选 MPS**。本项目的网络很小 (3x64) 且要用二阶自动微分,
    # 在 Apple Silicon 上实测 MPS 比 CPU 慢约 5 倍 (kernel 启动开销占主导)。
    # 静默选一个更慢的设备不如不选; 需要时显式传 --device mps。
    return torch.device("cpu")


def describe_device(dev: torch.device) -> str:
    """一行设备描述，打印在日志开头，方便确认到底跑在哪。"""
    if dev.type == "cuda":
        i = 0 if dev.index is None else dev.index
        try:
            return f"cuda:{i} ({torch.cuda.get_device_name(i)})"
        except Exception:                                   # noqa: BLE001
            return f"cuda:{i}"
    if dev.type == "mps":
        return "mps (Apple Silicon GPU)"
    return "cpu"


def device_benchmark(device: str = "auto", n: int = 2000, epochs: int = 200,
                     verbose: bool = True) -> float:
    """粗略测一下当前设备跑一步训练要多久（秒/步）。

    小网络 + 少量点时，显卡往往**不比 CPU 快**：每个 epoch 的算子很小，
    kernel 启动开销占主导。建议正式跑之前先用它对比 ``cpu`` 与 ``cuda``。
    """
    import time as _time
    from .model import MLP
    dev = pick_device(device)
    torch.manual_seed(0)
    net = MLP(hidden=(64, 64, 64), in_dim=2, T=1.0).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.rand(n, 1, device=dev, requires_grad=True)
    t = torch.rand(n, 1, device=dev, requires_grad=True)
    Z = torch.cat([x, t], dim=1)
    for _ in range(10):                                     # 预热
        opt.zero_grad(set_to_none=True)
        u = net(Z)
        g = torch.autograd.grad(u, Z, torch.ones_like(u), create_graph=True,
                                retain_graph=True)[0]
        uxx = torch.autograd.grad(g[:, 0:1], Z, torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0][:, 0:1]
        loss = (uxx ** 2).mean()
        loss.backward()
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = _time.time()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        u = net(Z)
        g = torch.autograd.grad(u, Z, torch.ones_like(u), create_graph=True,
                                retain_graph=True)[0]
        uxx = torch.autograd.grad(g[:, 0:1], Z, torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0][:, 0:1]
        loss = (uxx ** 2).mean()
        loss.backward()
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (_time.time() - t0) / epochs
    if verbose:
        print(f"[benchmark] {describe_device(dev)}  {n} 点, {dt*1000:.2f} ms/步")
    return dt



@dataclass
class NDConfig:
    problem: str = "corner2d"
    eps: float = 1e-3
    T: float = 1.0
    # 网络
    hidden: tuple = (64, 64, 64)
    activation: str = "tanh"
    # 采样
    n_int0: int = 1024
    n_ic0: int = 128
    n_bc0: int = 64          # 每个面
    init_method: str = "halton"
    # 自适应增点
    rounds: int = 8
    epochs_per_round: int = 800
    add_per_round: int = 150
    strategy: str = "residual"       # residual | distribution | uniform | fixed
    candidate_pool: int = 20000
    beta: float = 1.0
    add_ic_per_round: int = 16
    add_bc_per_round: int = 8        # 每个面
    # 权重
    weighting: str = "knn"           # knn | mean
    knn_neighbors: int = 4
    # 优化
    lr: float = 2e-3
    lr_schedule: str = "cosine"
    lr_gamma: float = 0.9
    weight_mode: str = "init"
    loss_norm: str = "l2"            # 'l2' | 'l1'
    w_ic: Optional[float] = None
    w_bc: Optional[float] = None
    # 其他
    seed: int = 0
    device: str = "auto"
    verbose: bool = True
    eval_n: int = 61
    callback: object = None

    def to_dict(self):
        d = asdict(self)
        d["hidden"] = tuple(self.hidden)
        d["callback"] = None
        return d


class AdaptivePINNND:
    """$d$ 维空间 + 时间的自适应配点 PINN。"""

    def __init__(self, cfg: NDConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = pick_device(cfg.device)
        self.problem = make_problem_nd(cfg.problem, cfg.eps, cfg.T)
        self.dim, self.D = self.problem.dim, self.problem.dim + 1

        self.points = CollocationSetND(self.dim, T=cfg.T, seed=cfg.seed)
        self.points.initialise(cfg.n_int0, cfg.n_ic0, cfg.n_bc0,
                               self.problem.faces(), cfg.init_method)
        self.model = MLP(hidden=tuple(cfg.hidden), in_dim=self.D,
                         T=cfg.T, activation=cfg.activation).to(self.device)
        self.val_pts = self.points.candidate_pool(20000)
        self.history: List[Dict] = []
        self.w_ic = 1.0 if cfg.w_ic is None else float(cfg.w_ic)
        self.w_bc = 1.0 if cfg.w_bc is None else float(cfg.w_bc)
        self.weights_fixed = cfg.weight_mode == "fixed"
        self.final_points = self.points.interior
        self.final_abs_res = np.zeros(0)

    # ---------------- 网络与残差 ----------------
    def _u(self, Z: torch.Tensor) -> torch.Tensor:
        return self.model(Z)

    def pde_residual(self, Z: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        """计算 $r=u_t-\\varepsilon\\Delta u+b\\cdot\\nabla u+qu-f$。"""
        prob = self.problem
        Z = Z.clone().detach().requires_grad_(True)
        u = self._u(Z)
        ones = torch.ones_like(u)
        g = torch.autograd.grad(u, Z, ones, create_graph=True, retain_graph=True)[0]

        u_t = g[:, self.dim:self.dim + 1]
        lap = torch.zeros_like(u)
        for j in range(self.dim):                       # Laplace 逐方向累加
            gj = torch.autograd.grad(g[:, j:j + 1], Z, torch.ones_like(u),
                                     create_graph=create_graph, retain_graph=True)[0]
            lap = lap + gj[:, j:j + 1]

        b = prob.convection(Z).to(Z.dtype)
        adv = sum(b[:, j:j + 1] * g[:, j:j + 1] for j in range(self.dim))
        return (u_t - self.cfg.eps * lap + adv
                + prob.reaction(Z) * u - prob.forcing(Z).reshape(-1, 1))

    def residual_field(self, pts: np.ndarray, batch: int = 20000) -> np.ndarray:
        out = []
        for k in range(0, len(pts), batch):
            Z = torch.as_tensor(np.ascontiguousarray(pts[k:k + batch]),
                                dtype=torch.float32, device=self.device)
            out.append(self.pde_residual(Z, create_graph=False).detach().cpu().numpy().ravel())
        return np.concatenate(out) if out else np.zeros(0)

    def predict(self, pts: np.ndarray, batch: int = 50000) -> np.ndarray:
        self.model.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(pts), batch):
                Z = torch.as_tensor(np.ascontiguousarray(pts[k:k + batch]),
                                    dtype=torch.float32, device=self.device)
                out.append(self._u(Z).detach().cpu().numpy().ravel())
        self.model.train()
        return np.concatenate(out)

    # ---------------- 权重与损失 ----------------
    def _weights(self, P: np.ndarray, drop_axis: Optional[int], measure: float) -> np.ndarray:
        if self.cfg.weighting != "knn":
            return np.full(P.shape[0], measure / max(P.shape[0], 1))
        coords = P if drop_axis is None else np.delete(P, drop_axis, axis=1)
        return knn_quadrature_weights(coords, measure,
                                      n_neighbors=self.cfg.knn_neighbors)

    def _tensors(self) -> Dict:
        dev, cfg = self.device, self.cfg
        c = self.points

        def t_(a):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)

        w_int = self._weights(c.interior, None, cfg.T)
        w_ic = self._weights(c.ic, self.dim, 1.0)
        bc = {f: (t_(P), t_(self._weights(P, f[0], cfg.T)).reshape(-1, 1))
              for f, P in c.bc.items()}
        return {"int": t_(c.interior), "w_int": t_(w_int).reshape(-1, 1),
                "ic": t_(c.ic), "w_ic": t_(w_ic).reshape(-1, 1),
                "bc": bc}

    def _loss_parts(self, data) -> Dict[str, torch.Tensor]:
        prob, T = self.problem, self.cfg.T
        vor = self.cfg.weighting == "knn"
        _p = (lambda z: z ** 2) if self.cfg.loss_norm == "l2" else (lambda z: z.abs())
        r = self.pde_residual(data["int"], create_graph=True)
        parts = {"pde": ((data["w_int"] * _p(r)).sum() / T) if vor else _p(r).mean()}

        X_ic = data["ic"][:, :self.dim]
        e_ic = _p(self._u(data["ic"]) - prob.ic(X_ic).reshape(-1, 1))
        parts["ic"] = ((data["w_ic"] * e_ic).sum() if vor else e_ic.mean())

        bc_terms = []
        for (axis, value), (P, w) in data["bc"].items():
            X = P[:, :self.dim]
            e = _p(self._u(P) - prob.bc(P, axis, value).reshape(-1, 1))
            bc_terms.append((w * e).sum() / T if vor else e.mean())
        parts["bc"] = sum(bc_terms) / max(len(bc_terms), 1)
        return parts

    def _total(self, parts) -> torch.Tensor:
        return parts["pde"] + self.w_ic * parts["ic"] + self.w_bc * parts["bc"]

    # ---------------- 评估 ----------------
    def evaluate(self, n: Optional[int] = None) -> Dict[str, float]:
        """在 $n^d\\times n$ 的均匀网格上评估最大误差与相对 $L^2$ 误差。"""
        n = n or self.cfg.eval_n
        axes = [np.linspace(0.0, 1.0, n) for _ in range(self.dim)]
        axes.append(np.linspace(0.0, self.cfg.T, n))
        grids = np.meshgrid(*axes, indexing="ij")
        pts = np.stack([g.reshape(-1) for g in grids], axis=1)
        u_pred = self.predict(pts)
        u_true = np.asarray(self.problem.exact(pts), dtype=float).reshape(-1)
        err = np.abs(u_pred - u_true)
        denom = np.linalg.norm(u_true)
        return {"max_err": float(err.max()),
                "rel_l2": float(np.linalg.norm(err) / denom) if denom > 0 else float("nan")}

    # ---------------- 自适应增点 ----------------
    def _select(self, values: np.ndarray, m: int) -> np.ndarray:
        st = self.cfg.strategy
        if st == "residual":
            return select_topm(np.abs(values), m)
        if st == "distribution":
            return select_distribution(values, m, self.cfg.beta, self.points.rng)
        if st == "uniform":
            return self.points.rng.choice(values.size, size=int(min(m, values.size)),
                                          replace=False)
        raise KeyError(f"未知策略 {st!r}")

    def _enrich(self) -> None:
        cfg = self.cfg
        if cfg.strategy == "fixed":
            return
        pool = self.points.candidate_pool(cfg.candidate_pool)
        r = self.residual_field(pool)
        self.points.add_interior(pool[self._select(r, cfg.add_per_round)])

        # 初值面
        if cfg.add_ic_per_round > 0:
            X = self.points.rng.random((4000, self.dim))
            Zc = np.column_stack([X, np.zeros(4000)])
            e = np.abs(self.predict(Zc) - np.asarray(self.problem.ic(X), dtype=float))
            self.points.add_ic(X[self._select(e, cfg.add_ic_per_round)])

        # 各边界面
        for axis, value in self.problem.faces():
            if cfg.add_bc_per_round <= 0:
                break
            X = self.points.rng.random((4000, self.dim))
            X[:, axis] = value
            t = self.points.rng.random(4000) * cfg.T
            Zc = np.column_stack([X, t])
            e = np.abs(self.predict(Zc)
                       - np.asarray(self.problem.bc(Zc, axis, value), dtype=float))
            sel = self._select(e, cfg.add_bc_per_round)
            self.points.add_bc((axis, value), Zc[sel])

    # ---------------- 主循环 ----------------
    def run(self) -> List[Dict]:
        cfg = self.cfg
        if not self.weights_fixed and cfg.weight_mode == "init":
            parts = self._loss_parts(self._tensors())
            lp = float(parts["pde"].detach()); li = float(parts["ic"].detach())
            lb = float(parts["bc"].detach())
            if cfg.w_ic is None:
                self.w_ic = float(np.clip(lp / max(li, 1e-12), 0.1, 1e3))
            if cfg.w_bc is None:
                self.w_bc = float(np.clip(lp / max(lb, 1e-12), 0.1, 1e3))
            if cfg.verbose:
                print(f"    [权重标定] J_pde={lp:.3e} J_ic={li:.3e} J_bc={lb:.3e} "
                      f"-> w_ic={self.w_ic:.3g}, w_bc={self.w_bc:.3g}")

        for rnd in range(cfg.rounds + 1):
            t0 = time.time()
            data = self._tensors()
            n_int, n_ic, n_bc = self.points.n_interior, self.points.n_ic, self.points.n_bc

            lr = cfg.lr * (cfg.lr_gamma ** rnd)
            opt = torch.optim.Adam(self.model.parameters(), lr=lr)
            sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=max(1, cfg.epochs_per_round), eta_min=lr * 0.02)
                     if cfg.lr_schedule == "cosine" else
                     torch.optim.lr_scheduler.StepLR(
                        opt, step_size=max(1, int(cfg.epochs_per_round * 0.4)), gamma=0.5))

            last: Dict[str, float] = {}
            for _ in range(cfg.epochs_per_round):
                opt.zero_grad(set_to_none=True)
                parts = self._loss_parts(data)
                loss = self._total(parts)
                loss.backward()
                opt.step()
                sched.step()
                last = {k: float(v.detach()) for k, v in parts.items()}

            ev = self.evaluate()
            r_val = self.residual_field(self.val_pts)
            # 这些统计量描述的是**本轮训练所用的点集**（增点之前）
            frac_layer = self.points.layer_fraction(cfg.eps, problem=self.problem)
            frac_corner = self.points.corner_fraction(cfg.eps)
            N_round = n_int + n_ic + n_bc
            if rnd < cfg.rounds:
                self._enrich()

            rec = {"round": rnd, "n_int": n_int, "n_ic": n_ic, "n_bc": n_bc,
                   "N": N_round, "max_err": ev["max_err"], "rel_l2": ev["rel_l2"],
                   "loss_pde": last.get("pde", float("nan")),
                   "loss_ic": last.get("ic", float("nan")),
                   "loss_bc": last.get("bc", float("nan")),
                   "rms_res_uniform": float(np.sqrt(np.mean(r_val ** 2))),
                   "frac_layer": frac_layer, "frac_corner": frac_corner,
                   "wall": time.time() - t0}
            self.history.append(rec)
            if cfg.callback is not None:
                cfg.callback(rec)
            if cfg.verbose:
                print(f"    round {rnd}: N={rec['N']:6d} max_err={rec['max_err']:.3e} "
                      f"rel_l2={rec['rel_l2']:.3e} rms_r={rec['rms_res_uniform']:.2e} "
                      f"layer={rec['frac_layer']:.3f} corner={rec['frac_corner']:.3f} "
                      f"({rec['wall']:.1f}s)")
            if rnd == cfg.rounds:
                self.final_points = self.points.interior.copy()
                self.final_abs_res = np.abs(self.residual_field(self.final_points))
        return self.history


def run_reference_nd(cfg: NDConfig, n_interior: int, total_epochs: int,
                     verbose: bool = False) -> AdaptivePINNND:
    """参照组：固定点数、不做自适应增点。"""
    c = copy.deepcopy(cfg)
    c.n_int0 = n_interior
    c.strategy = "fixed"
    c.rounds = 0
    c.epochs_per_round = int(total_epochs)
    c.verbose = verbose
    s = AdaptivePINNND(c)
    s.run()
    return s
