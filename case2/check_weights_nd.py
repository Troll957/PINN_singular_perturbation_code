"""对比三种求积权重的精度：无权重 / 精确(离散) Voronoi / kNN。

评价标准不是"每个点的权重是否准"，而是**求积本身的误差**

    E(g) = | sum_i w_i g(z_i) - ∫ g | / | ∫ g | ,

因为损失函数只通过加权和用到权重；逐点的涨落会相互抵消。

用法::
    arch -arm64 .venv/bin/python check_weights_nd.py
"""
from __future__ import annotations
import numpy as np
from spp_pinn.nd import knn_quadrature_weights
from spp_pinn.sampling import voronoi_weights_1d, voronoi_weights_2d


def qerr(P, w, g, exact):
    return abs((w * g(P)).sum() - exact) / abs(exact)


def main() -> None:
    rng = np.random.default_rng(7)

    # ---------------- 1D：带边界层的点分布 ----------------
    x = np.sort(np.concatenate([rng.random(200) * 0.02, rng.random(200)]))
    P1 = x.reshape(-1, 1)
    t1 = [("x^2", lambda p: p[:, 0] ** 2, 1 / 3),
          ("x^4", lambda p: p[:, 0] ** 4, 1 / 5),
          ("exp(-x/0.02)", lambda p: np.exp(-p[:, 0] / 0.02),
           0.02 * (1 - np.exp(-50)))]

    # ---------------- 2D：角点层点分布 ----------------
    P2 = np.vstack([rng.random((300, 2)) * [0.03, 1.0], rng.random((300, 2))])
    t2 = [("x^2", lambda p: p[:, 0] ** 2, 1 / 3),
          ("exp(-x/0.02)", lambda p: np.exp(-p[:, 0] / 0.02),
           0.02 * (1 - np.exp(-50))),
          ("exp(-(x+y)/0.02)", lambda p: np.exp(-(p[:, 0] + p[:, 1]) / 0.02),
           (0.02 * (1 - np.exp(-50))) ** 2)]

    for name, P, tests, ref in (
            ("1D", P1, t1, voronoi_weights_1d(x, 0.0, 1.0)),
            ("2D", P2, t2, voronoi_weights_2d(P2, n=512))):
        print(f"\n=== {name} 求积相对误差 E(g) ===")
        rows = [("mean（无权重）", np.full(len(P), 1.0 / len(P))),
                ("Voronoi（精确/离散）", ref),
                ("kNN（k=4）", knn_quadrature_weights(P, 1.0, n_neighbors=4)),
                ("kNN（k=8）", knn_quadrature_weights(P, 1.0, n_neighbors=8))]
        print(f"{'方法':22s} " + "  ".join(f"{n:>16s}" for n, _, _ in tests))
        for lab, w in rows:
            print(f"{lab:22s} " + "  ".join(f"{qerr(P, w, g, e):16.2e}"
                                            for _, g, e in tests))


if __name__ == "__main__":
    main()
