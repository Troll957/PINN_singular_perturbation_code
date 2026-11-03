"""自检脚本: 用自动微分验证三个算例的"制造解"是否严格满足方程与初边值条件。

对每个算例, 在随机点上用 float64 计算

    r = u_t - eps*u_xx + p(x)*u_x + q(x)*u - f(x, t)

若 max|r| 在机器精度量级, 说明解析解与右端项一致。

用法::

    python check_problems.py
"""

from __future__ import annotations

import numpy as np
import torch

from spp_pinn.problems import make_problem

torch.set_default_dtype(torch.float64)


def check(name: str, eps: float, T: float = 1.0, tol: float = 1e-10) -> bool:
    prob = make_problem(name, eps, T)

    # 在对数分布上取点, 保证层内也有采样
    n = 400
    x = np.concatenate([
        np.linspace(0.0, 1.0, n),
        np.geomspace(1e-9, 1.0, n),
        np.array([0.5]) + np.geomspace(1e-9, 0.5, n) * np.array([-1, 1])[np.random.randint(0, 2, n)],
    ])
    x = np.clip(x, 1e-12, 1 - 1e-12)
    t = np.random.rand(x.size)

    xt = torch.tensor(np.stack([x, t], axis=1), requires_grad=True)
    u = prob.exact(xt[:, 0:1], xt[:, 1:2])
    g = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = g[:, 0:1], g[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]

    X, Tt = xt[:, 0:1], xt[:, 1:2]
    r = (u_t - eps * u_xx + prob.p(X) * u_x + prob.q(X) * u - prob.forcing(X, Tt))
    err = float(r.abs().max())

    # 初边值一致性 (角点处 IC 与 BC 必须相容, 硬约束格式依赖这一点)
    xb = torch.tensor(np.linspace(0, 1, 200).reshape(-1, 1))
    corner = [
        float((prob.ic(torch.zeros(1, 1)) - prob.bc_left(torch.zeros(1, 1))).abs().max()),
        float((prob.ic(torch.ones(1, 1)) - prob.bc_right(torch.zeros(1, 1))).abs().max()),
    ]

    ok = err < tol and max(corner) < 1e-12
    print(f"[{'OK ' if ok else 'FAIL'}] {name:9s} eps={eps:<8g} "
          f"max|residual|={err:.3e}  corner-consistency={max(corner):.2e}  "
          f"(sqrt(eps)={np.sqrt(eps):.2e})")
    return ok


if __name__ == "__main__":
    np.random.seed(0)
    all_ok = True
    for name in ("left", "right", "interior"):
        for eps in (1e-1, 1e-3, 1e-6, 1e-8):
            all_ok &= check(name, eps)
    print("\n全部通过" if all_ok else "\n存在失败项")
    raise SystemExit(0 if all_ok else 1)
