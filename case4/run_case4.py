#!/usr/bin/env python3
"""Case 4：对 eps = 1e-3,1e-4,1e-5,1e-6 逐个跑自适应 / 固定均匀对照。

输出的 json 与 ``空间2d/eps_full`` 的格式对齐，方便直接并入论文的表格。

用法::

    python run_case4.py --device cuda --outdir runs/full
    python run_case4.py --eps 1e-4 --rounds 5 --epochs 500 --outdir runs/quick
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from spp_pinn import (AdaptivePINNGeneral, GeneralConfig, describe_device,
                      pick_device, run_reference)
from spp_pinn import domain


# ----------------------------------------------------------------------
# 画图
# ----------------------------------------------------------------------
def _domain_path(pad=0.0):
    """把 ∂Ω 变成 matplotlib 的 Path，用来裁剪 pcolormesh。"""
    X = domain.boundary_sample(1200, np.random.default_rng(0), uniform=True)
    verts = np.vstack([X, X[:1]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)


def _masked_field(solver, t, n=241):
    lo, hi = solver.problem.lo, solver.problem.hi
    g = np.linspace(lo[0], hi[0], n)
    X1, X2 = np.meshgrid(g, g, indexing="ij")
    Z = np.column_stack([X1.ravel(), X2.ravel(), np.full(n * n, t)])
    ue = np.asarray(solver.problem.exact(Z), dtype=float).reshape(n, n)
    up = solver.predict(Z).reshape(n, n)
    ins = domain.inside(Z[:, :2]).reshape(n, n)
    out = np.where(ins, 1.0, np.nan)
    return X1, X2, ue * out, up * out, (np.abs(up - ue) * out)


def fig_solution(solver, eps, path, t=0.25):
    X1, X2, ue, up, err = _masked_field(solver, t)
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.2))
    vmin, vmax = np.nanmin(ue), np.nanmax(ue)
    for ax, d, ttl in ((axs[0], ue, "Exact $u^*$"), (axs[1], up, r"PINN $u_\theta$")):
        im = ax.pcolormesh(X1, X2, d, cmap="viridis", shading="auto",
                           vmin=vmin, vmax=vmax)
        im.set_rasterized(True)
        ax.add_patch(PathPatch(_domain_path(), fill=False, lw=1.0, ec="k"))
        ax.set_title(ttl)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046)
    im = axs[2].pcolormesh(X1, X2, np.log10(err + 1e-16), cmap="inferno", shading="auto")
    im.set_rasterized(True)
    axs[2].add_patch(PathPatch(_domain_path(), fill=False, lw=1.0, ec="w"))
    axs[2].set_title(rf"$\log_{{10}}|u_\theta-u^*|$  (rel $L^2$={solver.final_eval['rel_l2']:.2e})")
    axs[2].set_aspect("equal")
    fig.colorbar(im, ax=axs[2], fraction=0.046)
    for ax in axs:
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
    fig.suptitle(rf"Case 4 (non-rectangular $\Omega$), $\varepsilon={eps:g}$, $t={t}$", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def fig_points(solver, eps, path):
    P = solver.points.interior
    r = solver.final_abs_res
    if len(r) != len(P):
        r = np.zeros(len(P))
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axs[0].add_patch(PathPatch(_domain_path(), fill=False, lw=1.2, ec="k"))
    sc = axs[0].scatter(P[:, 0], P[:, 1], c=np.log10(r + 1e-16), s=3,
                        cmap="magma", linewidths=0)
    fig.colorbar(sc, ax=axs[0], fraction=0.046, label=r"$\log_{10}|r|$")
    axs[0].set_title("interior points, coloured by $|r|$")
    axs[0].set_aspect("equal")

    n = 200
    lo, hi = solver.problem.lo, solver.problem.hi
    gx = np.linspace(lo[0], hi[0], n)
    gy = np.linspace(lo[1], hi[1], n)
    X1, X2 = np.meshgrid(gx, gy, indexing="ij")
    Z = np.column_stack([X1.ravel(), X2.ravel(), np.full(n * n, 0.25)])
    res = np.abs(solver.residual_field(Z)).reshape(n, n)
    res = np.where(domain.inside(Z[:, :2]).reshape(n, n), res, np.nan)
    im = axs[1].pcolormesh(X1, X2, np.log10(res + 1e-16), cmap="magma", shading="auto")
    im.set_rasterized(True)
    axs[1].add_patch(PathPatch(_domain_path(), fill=False, lw=1.0, ec="w"))
    fig.colorbar(im, ax=axs[1], fraction=0.046, label=r"$\log_{10}|r|$")
    axs[1].set_title(r"PDE residual $|r|$ at $t=1/4$")
    axs[1].set_aspect("equal")
    for ax in axs:
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
    fig.suptitle(rf"Case 4, $\varepsilon={eps:g}$: {len(P)} interior points", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def fig_summary(rows, path):
    eps = sorted({r["eps"] for r in rows})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, lab in (("rel_l2", r"relative $L^2$"), ("max_err", r"max $|u-u^*|$")):
        for var, col, mk in (("adaptive", "tab:blue", "o"), ("fixed", "tab:red", "s")):
            if not any(r["variant"] == var for r in rows):
                continue
            ys = [next(r[key] for r in rows if r["eps"] == e and r["variant"] == var)
                  for e in eps]
            axs[0 if key == "rel_l2" else 1].loglog(eps, ys, mk + "-", color=col, label=var)
        axs[0 if key == "rel_l2" else 1].set_xlabel(r"$\varepsilon$")
        axs[0 if key == "rel_l2" else 1].set_title(lab)
        axs[0 if key == "rel_l2" else 1].grid(True, which="both", alpha=0.3)
        axs[0 if key == "rel_l2" else 1].legend()
    fig.suptitle("Case 4: adaptive vs fixed uniform collocation", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------
def cfg_of(eps, a, strategy="residual"):
    return GeneralConfig(problem="flower", eps=eps, T=1.0, hidden=(64, 64, 64),
                         activation="tanh", b_scale=a.b_scale, n_int0=a.n_int0, n_ic0=a.n_ic0,
                         n_bc0=a.n_bc0, init_method="halton", rounds=a.rounds,
                         epochs_per_round=a.epochs, add_per_round=a.add,
                         strategy=strategy, candidate_pool=a.candidate_pool,
                         beta=1.0, pool_mode=a.pool_mode,
                         pool_band_frac=a.pool_band_frac, pool_band=a.pool_band,
                         add_ic_per_round=a.add_ic,
                         add_bc_per_round=a.add_bc, weighting="knn",
                         knn_neighbors=4, lr=2e-3, lr_schedule="cosine",
                         lr_gamma=0.9, weight_mode="init", seed=a.seed,
                         device=a.device, verbose=False,
                         eval_n=a.eval_n, eval_nt=a.eval_nt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="1e-3,1e-4,1e-5,1e-6")
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--n-int0", type=int, default=2048)
    ap.add_argument("--add", type=int, default=300)
    ap.add_argument("--n-ic0", type=int, default=256)
    ap.add_argument("--n-bc0", type=int, default=384)
    ap.add_argument("--add-ic", type=int, default=32)
    ap.add_argument("--add-bc", type=int, default=48)
    ap.add_argument("--candidate-pool", type=int, default=20000)
    ap.add_argument("--pool-mode", default="uniform", choices=["uniform", "graded"])
    ap.add_argument("--pool-band-frac", type=float, default=0.5)
    ap.add_argument("--pool-band", type=float, default=4.0)
    ap.add_argument("--eval-n", type=int, default=97)
    ap.add_argument("--eval-nt", type=int, default=33)
    ap.add_argument("--b-scale", type=float, default=1.0,
                    help="对流强度 b=b_scale*(x,y)；0.3 时 |b·n|·diam≈1，"
                         "层相对宽度与 Case 1/2 可比")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--only", default="both", choices=["both", "adaptive", "fixed"])
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    print(f"device = {describe_device(pick_device(a.device))}")
    print(f"域: 面积={domain.AREA:.4f} 周长={domain.PERIMETER:.4f} "
          f"r∈[{domain.radius(np.array([np.pi/3]))[0]:.3f},"
          f"{domain.radius(np.array([0.0]))[0]:.3f}]")
    n_ref = a.n_int0 + a.rounds * a.add
    steps = (a.rounds + 1) * a.epochs
    print(f"预算: 自适应 n_int0={a.n_int0} + {a.rounds}×{a.add} = {n_ref} 内点, "
          f"{a.rounds + 1}×{a.epochs} = {steps} 步\n")

    eps_list = [float(v) for v in a.eps.split(",")]
    rows = []
    hdr = (f"{'eps':>8} {'variant':>9} {'N':>7} {'n_int':>7} "
           f"{'rel_L2':>11} {'max_grid':>11} {'max_layer':>11} {'layer':>7} {'time':>7}")
    print(hdr)
    print("-" * len(hdr), flush=True)

    for eps in eps_list:
        for variant in (["adaptive", "fixed"] if a.only == "both" else [a.only]):
            t0 = time.time()
            if variant == "adaptive":
                s = AdaptivePINNGeneral(cfg_of(eps, a))
                s.run()
            else:
                s = run_reference(cfg_of(eps, a), n_interior=n_ref,
                                  total_epochs=steps)
            fe = s.final_eval
            frac = float(np.mean(s.problem.inside_layer(s.points.interior)))
            tag = f"case4_{variant}_eps{eps:.0e}"
            np.savez_compressed(
                os.path.join(a.outdir, tag + ".npz"),
                points=s.points.interior, bc=s.points.bc,
                hist=np.array([(h["N"], h["rel_l2"], h["max_grid"], h["frac_layer"])
                               for h in s.history]))
            fig_solution(s, eps, os.path.join(a.outdir, tag + "_sol.pdf"))
            fig_points(s, eps, os.path.join(a.outdir, tag + "_pts.pdf"))
            row = dict(case="case4", eps=eps, variant=variant,
                       N=s.points.total, n_int=s.points.n_interior,
                       rel_l2=fe["rel_l2"], max_grid=fe["max_grid"],
                       max_layer=fe["max_layer"], max_err=fe["max_err"],
                       frac_layer=frac, wall=time.time() - t0,
                       budget=dict(rounds=a.rounds, epochs=a.epochs,
                                   n_int0=a.n_int0, add=a.add,
                                   n_ic0=a.n_ic0, n_bc0=a.n_bc0,
                                   add_ic=a.add_ic, add_bc=a.add_bc,
                                   pool_mode=a.pool_mode, b_scale=a.b_scale,
                                   knn_neighbors=4, seed=a.seed))
            rows.append(row)
            print(f"{eps:8.0e} {variant:>9} {row['N']:7d} {row['n_int']:7d} "
                  f"{row['rel_l2']:11.3e} {row['max_grid']:11.3e} "
                  f"{row['max_layer']:11.3e} {frac:7.3f} {row['wall']:6.0f}s",
                  flush=True)
            json.dump(rows, open(os.path.join(a.outdir, "case4_eps_rows.json"), "w"),
                      indent=2, ensure_ascii=False)

    fig_summary(rows, os.path.join(a.outdir, "case4_summary.pdf"))
    print(f"\n-> {a.outdir}")


if __name__ == "__main__":
    main()
