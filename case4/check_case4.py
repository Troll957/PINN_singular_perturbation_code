r"""Case 4 自检：区域几何 + 制造解 + 求积权重归一化。

与 ``空间2d/check_problems_nd.py`` 同样的目的，但检查更严：

1. **算子**：用自动微分（双精度）独立算 $\mathcal L_\varepsilon u^*$，
   与 ``case4.py`` 里手推的闭式 ``forcing`` 对比。这是"制造解是否真的满足方程"
   的唯一可信检验；容差取机器精度量级（相对 1e-12）。
2. **几何**：边界点满足 $p=0$；$r(\theta)$ 的极值；面积/周长；弧长均匀性。
3. **相容性**：$u^*(t=0)$ 与初值、边界上与边值数据一致。
4. **权重**：kNN 体积权重之和等于 $|\Omega|T$、$|\partial\Omega|T$、$|\Omega|$。
"""

from __future__ import annotations

import numpy as np
import torch

from spp_pinn import domain
from spp_pinn.case4 import FlowerLayer2D
from spp_pinn.general import GeneralCollocationSet, knn_quadrature_weights


def ad_operator(prob, Z):
    """用 autograd 算 L u*（双精度）。"""
    Z = Z.clone().requires_grad_(True)
    u = prob.exact(Z).reshape(-1, 1)
    g = torch.autograd.grad(u, Z, torch.ones_like(u), create_graph=True)[0]
    lap = 0.0
    for j in range(2):
        gj = torch.autograd.grad(g[:, j:j + 1], Z, torch.ones_like(u),
                                 create_graph=True)[0]
        lap = lap + gj[:, j:j + 1]
    b = prob.convection(Z)
    adv = b[:, 0:1] * g[:, 0:1] + b[:, 1:2] * g[:, 1:2]
    return (g[:, 2:3] - prob.eps * lap + adv + prob.reaction(Z) * u).reshape(-1)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(0)

    # ---------------- 1. 几何 ----------------
    Xb = domain.boundary_points(rng.random(4000) * 2 * np.pi)
    print(f"[几何] 边界 max|p|          = {np.abs(domain.lev(Xb[:, 0], Xb[:, 1])).max():.3e}")
    print(f"[几何] 面积                = {domain.AREA:.6f}")
    print(f"[几何] 周长                = {domain.PERIMETER:.6f}")
    print(f"[几何] r(花瓣)= {domain.radius(np.array([0.0]))[0]:.4f}   "
          f"r(腰)= {domain.radius(np.array([np.pi / 3]))[0]:.4f}")
    Xs = domain.boundary_sample(1000, rng, uniform=True)
    seg = np.hypot(np.diff(Xs[:, 0]), np.diff(Xs[:, 1]))
    print(f"[几何] 弧长均匀性 min/max  = {seg.min():.3e} / {seg.max():.3e}")

    # ---------------- 2. 算子 ----------------
    print()
    for eps, bsc in ((1e-2, 1.0), (1e-3, 1.0), (1e-5, 1.0), (1e-6, 1.0),
                     (1e-5, 0.3), (1e-6, 0.3)):
        prob = FlowerLayer2D(eps=eps, T=1.0, b_scale=bsc)
        s = prob.sqrt_eps
        n = 400
        Xbb = domain.boundary_sample(n, rng)
        nrm = domain.outward_normal(Xbb)
        P = Xbb - nrm * (rng.random(n) * 3 * s)[:, None]
        far = rng.uniform(-1.4, 1.4, (800, 2))
        far = far[domain.inside(far)][:n]
        X = np.vstack([P, far])
        Z = np.column_stack([X, rng.random(len(X))])
        Zt = torch.as_tensor(Z)
        f_ad = ad_operator(prob, Zt).detach().numpy()
        f_an = np.asarray(prob.forcing(Zt)).reshape(-1)
        rel = np.abs(f_ad - f_an).max() / np.abs(f_an).max()
        print(f"[算子] eps={eps:g}, b_scale={bsc:g}: |f|max={np.abs(f_an).max():.3e}  "
              f"max|AD-f|={np.abs(f_ad - f_an).max():.3e}  相对={rel:.2e}")

    # ---------------- 3. 相容性 ----------------
    print()
    prob = FlowerLayer2D(eps=1e-4)
    X = rng.uniform(-1.3, 1.3, (2000, 2))
    X = X[domain.inside(X)]
    Z0 = np.column_stack([X, np.zeros(len(X))])
    print(f"[相容] |u*(t=0) - ic|max   = "
          f"{np.abs(prob.exact(Z0) - prob.ic(X)).max():.3e}")
    Zb = np.column_stack([Xb, rng.random(len(Xb))])
    print(f"[相容] |bc - u*|max        = {np.abs(prob.bc(Zb) - prob.exact(Zb)).max():.3e}")
    print(f"[相容] 边界上 |ψ|max       = {np.abs(domain.psi(Xb[:, 0], Xb[:, 1])).max():.3e}")

    # ---------------- 4. 权重 ----------------
    print()
    pts = GeneralCollocationSet(prob, T=1.0, seed=0)
    pts.initialise(1500, 200, 300)
    w_int = knn_quadrature_weights(pts.interior, prob.area * prob.T, 4, exponent=3)
    w_ic = knn_quadrature_weights(pts.ic[:, :2], prob.area, 4, exponent=2)
    w_bc = knn_quadrature_weights(pts.bc, prob.perimeter * prob.T, 4, exponent=2)
    print(f"[权重] sum w_int = {w_int.sum():.6f}  (期望 {prob.area * prob.T:.6f})")
    print(f"[权重] sum w_ic  = {w_ic.sum():.6f}  (期望 {prob.area:.6f})")
    print(f"[权重] sum w_bc  = {w_bc.sum():.6f}  (期望 {prob.perimeter * prob.T:.6f})")
    print(f"[权重] 内部点全部在区域内? {np.all(domain.inside(pts.interior[:, :2]))}")
    print(f"[权重] 初值点全部在区域内? {np.all(domain.inside(pts.ic[:, :2]))}")
    print(f"[权重] 边界点 |p|max = "
          f"{np.abs(domain.lev(pts.bc[:, 0], pts.bc[:, 1])).max():.3e}")


if __name__ == "__main__":
    main()
