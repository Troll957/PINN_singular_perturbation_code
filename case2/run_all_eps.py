#!/usr/bin/env python3
"""对 eps = 1e-3,1e-4,1e-5,1e-6 分别跑 Case 2 / Case 3, 并出图。

图: <tag>_sol.pdf  在 t=0.5 切片上的精确解 / PINN 预测 / log10 误差
    <tag>_pts.pdf  采样点分布 (x1-x2 边缘计数 与 x1-t 投影)
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spp_pinn.nd import AdaptivePINNND, NDConfig, make_problem_nd
from spp_pinn import pick_device, describe_device
from spp_pinn import plotting as PLT


def fig_solution(solver, eps, title, path, t_slice=0.25):
    prob = solver.problem
    n = 161
    g = np.linspace(0.0, 1.0, n)
    X1, X2 = np.meshgrid(g, g, indexing="ij")
    Z = np.column_stack([X1.ravel(), X2.ravel(), np.full(n * n, t_slice)])
    ue = np.asarray(prob.exact(Z), dtype=float).reshape(n, n)
    up = solver.predict(Z).reshape(n, n)
    err = np.abs(up - ue)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    vmin = min(ue.min(), up.min()); vmax = max(ue.max(), up.max())
    for ax, d, ttl in ((axs[0], ue, "Exact solution"), (axs[1], up, "PINN prediction")):
        im = ax.pcolormesh(X1, X2, d, cmap="viridis", shading="auto",
                           vmin=vmin, vmax=vmax)
        im.set_rasterized(True)
        ax.set_title(ttl); ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046)
    im = axs[2].pcolormesh(X1, X2, np.log10(err + 1e-16), cmap="inferno",
                           shading="auto")
    im.set_rasterized(True)
    axs[2].set_title(rf"$\log_{{10}}|u_\theta-u^*|$  (rel $L^2$={solver.history[-1]['rel_l2']:.2e})")
    axs[2].set_xlabel(r"$x_1$"); axs[2].set_ylabel(r"$x_2$"); axs[2].set_aspect("equal")
    fig.colorbar(im, ax=axs[2], fraction=0.046)
    fig.suptitle(rf"{title},  $\varepsilon={eps:g}$,  $t={t_slice}$", y=1.02)
    fig.tight_layout()
    return PLT._save(fig, path)


def fig_points(pts, eps, title, path):
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
    h = axs[0].hist2d(pts[:, 0], pts[:, 1], bins=90, cmap="magma",
                      norm=matplotlib.colors.LogNorm())
    fig.colorbar(h[3], ax=axs[0], fraction=0.046)
    axs[0].set_title(r"interior points, marginal in $(x_1,x_2)$")
    axs[0].set_xlabel(r"$x_1$"); axs[0].set_ylabel(r"$x_2$"); axs[0].set_aspect("equal")
    axs[1].scatter(pts[:, 0], pts[:, 2], s=1.5, c="k", alpha=0.35)
    axs[1].axvline(4 * np.sqrt(eps), color="r", ls="--", lw=1.0, label=r"$4\sqrt{\varepsilon}$")
    axs[1].set_xlabel(r"$x_1$"); axs[1].set_ylabel(r"$t$")
    axs[1].set_title(r"projection $(x_1,t)$"); axs[1].legend(loc="upper right")
    fig.suptitle(rf"{title},  $\varepsilon={eps:g}$", y=1.02)
    fig.tight_layout()
    return PLT._save(fig, path)


def cfg_of(problem, eps, rounds, epochs, n0, add, strategy, weighting, device, evn):
    return NDConfig(problem=problem, eps=eps, T=1.0, hidden=(64, 64, 64),
                    activation="tanh", n_int0=n0, n_ic0=128, n_bc0=64,
                    init_method="halton", rounds=rounds,
                    epochs_per_round=epochs, add_per_round=add,
                    strategy=strategy, candidate_pool=20000, beta=1.0,
                    weighting=weighting, knn_neighbors=4, lr=2e-3,
                    lr_schedule="cosine", lr_gamma=0.9, weight_mode="init",
                    seed=0, device=device, verbose=False, eval_n=evn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="case2", choices=["case2", "case3"])
    ap.add_argument("--eps", default="1e-3,1e-4,1e-5,1e-6")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--n-int0", type=int, default=1024)
    ap.add_argument("--add", type=int, default=150)
    ap.add_argument("--eval-n", type=int, default=61)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="eps_results")
    a = ap.parse_args()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    os.makedirs(a.outdir, exist_ok=True)
    prob_name = "corner2d" if a.case == "case2" else "moving"
    title = "Case 2 corner layer" if a.case == "case2" else "Case 3 moving layer"
    eps_list = [float(v) for v in a.eps.split(",")]
    n_ref = a.n_int0 + a.rounds * a.add
    print(f"device={describe_device(pick_device(a.device))}  {title}  eps={eps_list}\n", flush=True)
    rows = []
    print(f"{'eps':>8} {'variant':>9} {'N':>6} {'rel_L2':>11} {'max_err':>11} {'layer':>7} {'time':>6}")
    for eps in eps_list:
        for variant in ("adaptive", "fixed"):
            t0 = time.time()
            cfg = cfg_of(prob_name, eps, a.rounds if variant == "adaptive" else 0,
                         a.epochs, n_ref if variant == "fixed" else a.n_int0,
                         a.add, "residual" if variant == "adaptive" else "fixed",
                         "knn", a.device, a.eval_n)
            solver = AdaptivePINNND(cfg); solver.run()
            last = dict(solver.history[-1])
            tag = f"{a.case}_{variant}_eps{eps:.0e}"
            fig_solution(solver, eps, title, os.path.join(a.outdir, tag + "_sol.pdf"))
            fig_points(solver.points.interior, eps, title,
                       os.path.join(a.outdir, tag + "_pts.pdf"))
            rows.append(dict(case=a.case, eps=eps, variant=variant, N=int(last["N"]),
                             n_int=int(last["n_int"]), rel_l2=float(last["rel_l2"]),
                             max_err=float(last["max_err"]),
                             frac_layer=float(last["frac_layer"]), wall=time.time() - t0))
            r = rows[-1]
            print(f"{eps:8.0e} {variant:>9} {r['N']:6d} {r['rel_l2']:11.3e} "
                  f"{r['max_err']:11.3e} {r['frac_layer']:7.3f} {r['wall']:5.0f}s", flush=True)
    json.dump(rows, open(os.path.join(a.outdir, f"{a.case}_eps_rows.json"), "w"), indent=2)
    print("\n->", a.outdir)


if __name__ == "__main__":
    main()
