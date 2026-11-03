r"""Case 4：**非矩形区域**上的奇异摄动抛物问题（花形区域 + 曲边界层）。

方程（与 1d / 2d 各算例同一族算子）

.. math::
    \mathcal L_\varepsilon u:=u_t-\varepsilon\Delta u+\mathbf b(x)\cdot\nabla u+q\,u=f ,
    \qquad (x,y)\in\Omega,\ t\in(0,1] ,

其中 $\Omega=\{p<0\}$ 是 :mod:`spp_pinn.domain` 里的三瓣花形区域，

.. math::
    \mathbf b(x,y)=(x,y)\quad(\text{径向向外的对流场}),\qquad q\equiv 1 .

在 $\partial\Omega$ 上 $\mathbf b\cdot\mathbf n=|\mathbf b|>0$，所以整条曲边界都是
**出流边界**，解沿整条边界形成一条厚度 $O(\sqrt\varepsilon)$ 的边界层。

制造解
------
记 $\psi=-p\ge0$（$\psi=0$ 恰为 $\partial\Omega$）、$s=\sqrt\varepsilon$、
$\varphi(\rho)=1-e^{-\rho}$、$C=\cos\pi t$、$S_s=\sin\pi t$，

.. math::
    u^*(x,y,t)=\varphi\!\left(\frac{\psi}{s}\right)S(x,y)\cos(\pi t),\qquad
    S(x,y)=1+0.3\sin(\pi x)\sin(\pi y).

$S$ 是光滑的"外解"，使区域内部不是平凡的常值；$\varphi(\psi/s)$ 在曲边界上
产生宽度 $O(\sqrt\varepsilon)$ 的层。

右端项
------
把 $u^*$ 代入算子并利用 $\Delta\psi\equiv-4$（因为 $x^3-3xy^2$ 是调和多项式）得

.. math::
    f=-\pi\varphi S\sin\pi t+\cos\pi t\Bigl\{
      e^{-\psi/s}\bigl[S|\nabla\psi|^2+4sS-2s\,\nabla S\!\cdot\!\nabla\psi
      +\tfrac{S}{s}\,\mathbf b\!\cdot\!\nabla\psi\bigr]
      +\varphi\,\bigl[-\varepsilon\Delta S+\mathbf b\!\cdot\!\nabla S+S\bigr]\Bigr\}.

良定性：$\mathbf b\cdot\nabla\psi<0$ 在 $\partial\Omega$ 上（约为 $-2$），
与"出流层"一致；源项因此含 $O(\varepsilon^{-1/2})$ 的项，这与 Case 1 一致
（那里的 $f$ 同样含 $1/\sqrt\varepsilon$），是这类问题的固有性质。
"""

from __future__ import annotations

import numpy as np

from . import domain
from .domain import A_WAVE, AREA, PERIMETER
from .problems import PI, _xp

S_AMP = 0.3            # 外解 S 的振幅

# 对流强度。b = B_SCALE*(x,y)。
#   B_SCALE = 1     : |b·n| ≈ 1.1–2.5（随曲率变化），层厚 √ε/|b·n|
#   B_SCALE = 0.3   : |b·n|·diam(Ω) ≈ 1，层厚与 Case 1/2（单位方域、|b·n|=1）
#                     在**相对意义**上可比
# 见 README 第 6 节：不缩放时同一个 ε 的层相对宽度比 Case 2 细约 6 倍，
# 直接拿 1e-5 / 1e-6 对比会误以为方法失效。
B_SCALE = 1.0


def _stack_cols(xp, a, b):
    """按列拼成 (N,2)，兼容 numpy(axis=) 与 torch(dim=)。"""
    try:
        return xp.stack([a, b], dim=1)
    except TypeError:
        return xp.stack([a, b], axis=1)


def _clip_lo(v, lo):
    """max(v, lo)，同时兼容 numpy 与 torch。"""
    if hasattr(v, "clamp"):
        return v.clamp(min=lo)
    return np.maximum(v, lo)


class GeneralProblem:
    """二维空间 + 时间的、区域由水平集给出的奇异摄动问题基类。"""

    dim = 2
    name = "base"

    def __init__(self, eps: float, T: float = 1.0, b_scale: float = B_SCALE):
        if eps <= 0:
            raise ValueError("eps 必须为正")
        self.eps = float(eps)
        self.b_scale = float(b_scale)
        self.T = float(T)
        lo, hi = domain.bounding_box()
        self.lo, self.hi = lo, hi
        self.area = AREA
        self.perimeter = PERIMETER

    @property
    def sqrt_eps(self) -> float:
        return float(np.sqrt(self.eps))

    # ---------------- 算子系数 ----------------
    def convection(self, Z):
        """b = (x, y)：径向向外的对流场。"""
        xp = _xp(Z)
        return _stack_cols(xp, Z[:, 0] * self.b_scale, Z[:, 1] * self.b_scale)

    def reaction(self, Z):
        """q ≡ 1。注意要在 Z 所在的设备上建张量，否则 GPU 上会报
        "found at least two devices"。"""
        xp = _xp(Z)
        if hasattr(Z, "device"):
            return xp.ones((Z.shape[0], 1), dtype=Z.dtype, device=Z.device)
        return xp.ones((Z.shape[0], 1))

    # ---------------- 问题数据 ----------------
    def exact(self, Z):
        xp = _xp(Z)
        x, y, t = Z[:, 0], Z[:, 1], Z[:, 2]
        psi = domain_psi_t(x, y, xp)
        # 不做 ρ≥0 的截断：u* 是边界邻域内的光滑解析延拓，若在 ψ=0 处截断
        # 会在边界上留下折点，使 f 在 ∂Ω 上不连续（有限差分自检立刻暴露）。
        # 只限制下界避免 exp 溢出；区域内部 ψ≥0，下界永不触发。
        rho = _clip_lo(psi / self.sqrt_eps, -50.0)
        phi = 1.0 - xp.exp(-rho)
        S = 1.0 + S_AMP * xp.sin(PI * x) * xp.sin(PI * y)
        return (phi * S * xp.cos(PI * t)).reshape(-1)

    def forcing(self, Z):
        xp = _xp(Z)
        eps, s = self.eps, self.sqrt_eps
        x, y, t = Z[:, 0], Z[:, 1], Z[:, 2]

        psi = domain_psi_t(x, y, xp)
        rho = _clip_lo(psi / s, -50.0)
        e = xp.exp(-rho)
        phi = 1.0 - e

        # ∇ψ 与 |∇ψ|²
        psi_x = A_WAVE * (3.0 * x * x - 3.0 * y * y) - 2.0 * x
        psi_y = -6.0 * A_WAVE * x * y - 2.0 * y
        g2 = psi_x * psi_x + psi_y * psi_y

        # 外解 S 及其导数
        sinx, siny = xp.sin(PI * x), xp.sin(PI * y)
        cosx, cosy = xp.cos(PI * x), xp.cos(PI * y)
        S = 1.0 + S_AMP * sinx * siny
        S_x = S_AMP * PI * cosx * siny
        S_y = S_AMP * PI * sinx * cosy
        dS = -2.0 * S_AMP * PI * PI * sinx * siny       # ΔS
        gradS_dot_gradpsi = S_x * psi_x + S_y * psi_y
        b_gradS = self.b_scale * (x * S_x + y * S_y)    # b = b_scale*(x,y)
        # 调和性给出 Δψ = -4；b=(x,y) 时
        #   b·∇ψ = x ψ_x + y ψ_y = 3a(x³-3xy²) - 2(x²+y²)
        # 恒等于 a(x³-3xy²) - 2 + 2ψ，但在界面上只有前者能直接算。
        b_gradpsi = self.b_scale * (3.0 * A_WAVE * (x ** 3 - 3.0 * x * y * y)
                                    - 2.0 * (x * x + y * y))

        C, Sn = xp.cos(PI * t), xp.sin(PI * t)
        inner = e * (S * g2 + 4.0 * s * S - 2.0 * s * gradS_dot_gradpsi
                     + S * b_gradpsi / s)
        outer = phi * (-eps * dS + b_gradS + S)
        return (-PI * phi * S * Sn + C * (inner + outer)).reshape(-1)

    def ic(self, X):
        xp = _xp(X)
        x, y = X[:, 0], X[:, 1]
        psi = domain_psi_t(x, y, xp)
        rho = _clip_lo(psi / self.sqrt_eps, -50.0)
        phi = 1.0 - xp.exp(-rho)
        S = 1.0 + S_AMP * xp.sin(PI * x) * xp.sin(PI * y)
        return (phi * S).reshape(-1)

    def bc(self, Z, axis=None, value=None):
        """整条曲边界上的 Dirichlet 数据 = 精确解的迹。"""
        return self.exact(Z)

    # ---------------- 几何接口 ----------------
    def faces(self):
        """把整条边界当作**一个**"面"；axis/value 只是占位符。"""
        return [("bdry", 0.0)]

    def boundary_sample(self, n: int, rng: np.random.Generator,
                        uniform: bool = False) -> np.ndarray:
        return domain.boundary_sample(n, rng, uniform=uniform)

    def inside_layer(self, Z, factor: float = 4.0) -> np.ndarray:
        """到边界的近似距离 ≤ factor·√ε。"""
        X = np.asarray(Z)[:, :2]
        return domain.sdf(X) <= factor * self.sqrt_eps

    def inside(self, X, margin: float = 0.0) -> np.ndarray:
        return domain.inside(X, margin=margin)


def domain_psi_t(x, y, xp):
    """ψ = -p = 1 + a(x³-3xy²) - x² - y²（对 numpy / torch 通用）。"""
    return 1.0 + A_WAVE * (x ** 3 - 3.0 * x * y * y) - x * x - y * y


class FlowerLayer2D(GeneralProblem):
    """Case 4 的算例本体。"""

    name = "flower"


_PROBLEMS = {"flower": FlowerLayer2D}


def make_problem(name: str, eps: float, T: float = 1.0,
                 b_scale: float = B_SCALE) -> GeneralProblem:
    if name not in _PROBLEMS:
        raise KeyError(f"未知算例 {name!r}, 可选: {sorted(_PROBLEMS)}")
    return _PROBLEMS[name](eps=eps, T=T, b_scale=b_scale)
