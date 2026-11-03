"""自检: 验证 d 维算例的制造解与右端项, 以及 kNN 求积权重的精度。

用法::
    arch -arm64 .venv/bin/python check_problems_nd.py
"""
from __future__ import annotations
import numpy as np
import torch

from spp_pinn.nd import make_problem_nd, knn_quadrature_weights
from spp_pinn.sampling import voronoi_weights_1d

torch.set_default_dtype(torch.float64)


def check_problem(name: str, eps: float, T: float = 1.0, tol: float = 1e-9) -> bool:
    """用自动微分验证 L u* - f = 0。"""
    prob = make_problem_nd(name, eps, T)
    d = prob.dim
    n = 300
    # 对数分布保证层内也有采样点
    base = np.geomspace(1e-9, 1.0, n)
    X = np.concatenate([np.random.rand(n, d), base[:, None] * np.ones((1, d)),
                        np.tile(base[:, None], (1, d)) * np.random.rand(n, d)], axis=0)
    X = np.clip(X, 1e-12, 1 - 1e-12)
    Z = np.column_stack([X, np.random.rand(X.shape[0]) * T])
    Zt = torch.tensor(Z, requires_grad=True)

    u = prob.exact(Zt).reshape(-1, 1)
    g = torch.autograd.grad(u, Zt, torch.ones_like(u), create_graph=True)[0]
    u_t, lap = g[:, d:d + 1], torch.zeros_like(u)
    for j in range(d):
        gj = torch.autograd.grad(g[:, j:j + 1], Zt, torch.ones_like(u), create_graph=True)[0]
        lap = lap + gj[:, j:j + 1]
    b = prob.convection(Zt)
    adv = sum(b[:, j:j + 1] * g[:, j:j + 1] for j in range(d))
    r = u_t - eps * lap + adv + prob.reaction(Zt) * u - prob.forcing(Zt).reshape(-1, 1)
    err = float(r.abs().max().detach())
    print(f"[{'OK ' if err < tol else 'FAIL'}] {name:9s} d={d} eps={eps:<8g} "
          f"max|L u* - f| = {err:.3e}")
    return err < tol


def check_knn() -> None:
    """kNN 权重 vs 一维精确 Voronoi 权重。"""
    rng = np.random.default_rng(0)
    for name, x in (("uniform", np.sort(rng.random(400))),
                    ("graded", np.sort(rng.random(400) ** 3)),
                    ("clustered", np.concatenate([rng.random(200) * 0.02,
                                                  rng.random(200)]))):
        x = np.sort(x)
        exact = voronoi_weights_1d(x, 0.0, 1.0)
        for k in (2, 4, 8):
            knn = knn_quadrature_weights(x.reshape(-1, 1), 1.0, n_neighbors=k)
            rel = np.abs(knn - exact).sum() / exact.sum()
            print(f"  kNN vs Voronoi(1D) [{name:9s}] k={k}: "
                  f"L1 相对误差 = {rel:6.3f}  (sum={knn.sum():.6f})")
    print()


if __name__ == "__main__":
    np.random.seed(0)
    ok = all(check_problem("corner2d", e) for e in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6))
    check_knn()
    print("全部通过" if ok else "存在失败项")
    raise SystemExit(0 if ok else 1)
