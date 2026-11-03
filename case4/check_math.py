"""Case 4 的数学自检（numpy 部分）。

1. 几何：边界点确实满足 p=0；面积/周长；ψ>0 在内部。
2. 算子：用 5 点中心差分独立算出 u*_t、∇u*、Δu*，与解析的 f 对比。
   这一步与 PINN 无关，是纯粹的"制造解是否正确"的检查。
3. 初值/边值的相容性。
"""

from __future__ import annotations

import numpy as np

from spp_pinn import domain
from spp_pinn.case4 import FlowerLayer2D


def stencil(f, X, axis, h):
    """5 点中心差分：返回 df/dx_axis，X 为 (N,3)。"""
    e = np.zeros(3)
    e[axis] = h
    fp2 = f(X + 2 * e)
    fp1 = f(X + e)
    fm1 = f(X - e)
    fm2 = f(X - 2 * e)
    return (-fp2 + 8 * fp1 - 8 * fm1 + fm2) / (12 * h)


def laplacian(f, X, h):
    out = np.zeros(X.shape[0])
    for j in range(2):
        out += stencil(lambda Z, j=j: stencil(f, Z, j, h), X, j, h)
    return out


def main() -> None:
    rng = np.random.default_rng(0)

    # ---------------- 1. 几何 ----------------
    th = rng.random(4000) * 2 * np.pi
    Xb = domain.boundary_points(th)
    res = np.abs(domain.lev(Xb[:, 0], Xb[:, 1]))
    print(f"[几何] 边界 |p| 的最大值 = {res.max():.3e}")
    print(f"[几何] 面积 = {domain.AREA:.6f}  周长 = {domain.PERIMETER:.6f}")
    print(f"[几何] r(花瓣 θ=0) = {domain.radius(np.array([0.0]))[0]:.4f}, "
          f"r(腰 θ=π/3) = {domain.radius(np.array([np.pi / 3]))[0]:.4f}")

    # 圆周公式校验：若 a=0 应得 π 与 2π；a≠0 时用散度定理再算一次面积
    Xord = domain.boundary_sample(2000, rng, uniform=True)
    area2 = 0.5 * np.sum(
        Xord[:, 0] * np.roll(Xord[:, 1], -1) - np.roll(Xord[:, 0], -1) * Xord[:, 1])
    print(f"[几何] 面积(边界积分, 随机点) = {abs(area2):.6f}")

    # 边界点按弧长均匀？
    Xs = domain.boundary_sample(2000, rng, uniform=True)
    seg = np.hypot(np.diff(Xs[:, 0]), np.diff(Xs[:, 1]))
    print(f"[几何] 弧长均匀性: 段长 min/max/mean = "
          f"{seg.min():.3e}/{seg.max():.3e}/{seg.mean():.3e}")

    # ---------------- 2. 算子 ----------------
    eps = 1e-3
    prob = FlowerLayer2D(eps=eps, T=1.0)
    h = 1e-4
    s = np.sqrt(eps)

    # 在区域内取点：一半在层内 (ψ<2s)，一半在外部
    n = 300
    X = domain.boundary_sample(n, rng)
    nrm = domain.outward_normal(X)
    d = rng.random(n) * 3 * s                      # 从边界往内 0..3s
    P = X - nrm * d[:, None]
    far = rng.random((n, 2)) * 2.0 - 1.0
    far = far[domain.inside(far)][:n]
    Xall = np.vstack([P, far])
    Z = np.column_stack([Xall, rng.random(Xall.shape[0])])

    u = lambda Z_: prob.exact(Z_)
    u_t = stencil(u, Z, 2, h)
    u_x = stencil(u, Z, 0, h)
    u_y = stencil(u, Z, 1, h)
    lap = laplacian(u, Z, h)
    b = prob.convection(Z)
    f_fd = u_t - eps * lap + b[:, 0] * u_x + b[:, 1] * u_y + u(Z)
    f_an = prob.forcing(Z)

    scale = np.abs(f_an).max()
    err = np.abs(f_fd - f_an)
    print(f"\n[算子] eps={eps:g}, h={h:g}, {Z.shape[0]} 点")
    print(f"[算子] |f| 的最大值 = {scale:.4e}")
    print(f"[算子] |f_FD - f_解析| 的最大值 = {err.max():.3e}  "
          f"相对 = {err.max() / scale:.3e}")

    # 分区域看误差（层内导数最大，是最苛刻的地方）
    psi = domain.psi(Xall[:, 0], Xall[:, 1])
    for lo, hi, name in [(0, 1 * s, "层内 ψ<s"), (1 * s, 3 * s, "层缘 1s..3s"),
                         (3 * s, 1e9, "外部 ψ>3s")]:
        m = (psi >= lo) & (psi < hi)
        if m.any():
            print(f"[算子]   {name:14s}: n={m.sum():4d}  相对误差 = "
                  f"{np.abs((f_fd - f_an)[m]).max() / max(scale, 1e-30):.3e}")

    # 单独验证 Δψ = -4 与 b·∇ψ
    a = domain.A_WAVE
    x, y = Xall[:, 0], Xall[:, 1]
    lap_psi = np.zeros_like(x)
    for j, v in enumerate([x, y]):
        pass
    hh = 1e-5
    lap_psi = (domain.psi(x + hh, y) - 2 * domain.psi(x, y) + domain.psi(x - hh, y)) / hh ** 2 \
        + (domain.psi(x, y + hh) - 2 * domain.psi(x, y) + domain.psi(x, y - hh)) / hh ** 2
    print(f"[几何] Δψ 的数值值 = {lap_psi.mean():.6f} (解析 -4)")

    # ---------------- 3. 初值/边值 ----------------
    print(f"\n[相容性] u*(t=0) 与 ic 的最大差 = "
          f"{np.abs(prob.exact(np.column_stack([Xall, np.zeros(len(Xall))])) - prob.ic(Xall)).max():.3e}")
    Zb = np.column_stack([Xb, rng.random(len(Xb))])
    print(f"[相容性] 边界上 |bc - exact| 的最大值 = "
          f"{np.abs(prob.bc(Zb) - prob.exact(Zb)).max():.3e}")
    print(f"[相容性] 边界上 |ψ| 的最大值 = "
          f"{np.abs(domain.psi(Xb[:, 0], Xb[:, 1])).max():.3e}")


if __name__ == "__main__":
    main()
