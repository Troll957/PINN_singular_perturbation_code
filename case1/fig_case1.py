#!/usr/bin/env python3
"""Case 1 论文用图 (6 面板): 解, 预测, 误差 (时空图), 点分布, 层内曲线, 层内点。"""
from __future__ import annotations
import argparse, math, os, time
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from spp_pinn import AdaptivePINNSolver, TrainConfig, pick_device, describe_device
from spp_pinn import plotting as PLT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--problem", default="left")
    ap.add_argument("--t-show", type=float, default=0.25)
    ap.add_argument("--nx", type=int, default=401)
    ap.add_argument("--nt", type=int, default=201)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--n-int0", type=int, default=4096)
    ap.add_argument("--add", type=int, default=500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="case_figs")
    a = ap.parse_args()
    torch.set_num_threads(6)
    os.makedirs(a.outdir, exist_ok=True)
    eps, thr = a.eps, 20.0 * math.sqrt(a.eps)
    print(f"device={describe_device(pick_device(a.device))}  {a.problem}  eps={eps:g}  "
          f"层内区 [0,{thr:.2e}]", flush=True)
    cfg = TrainConfig(problem=a.problem, eps=eps, T=1.0, hidden=(64, 64, 64),
                      activation="tanh", n_int0=a.n_int0, n_ic0=64, n_bc0=64,
                      init_method="halton", rounds=a.rounds,
                      epochs_per_round=a.epochs, add_per_round=a.add,
                      strategy="residual", candidate_pool=20000, beta=1.0,
                      min_dist=0.0, lr=2e-3, lr_schedule="cosine", lr_gamma=0.9,
                      weight_mode="init", seed=0, device=a.device, verbose=False)
    t0 = time.time()
    s = AdaptivePINNSolver(cfg); s.run()
    last = dict(s.history[-1])
    print(f"[训练完成] N={last['N']} rel_L2={last['rel_l2']:.3e} "
          f"max={last['max_err']:.3e} layer={last['frac_layer']:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    torch.save({"model": s.model.state_dict(), "cfg": cfg.to_dict(), "hist": s.history,
                "points": s.final_points},
               os.path.join(a.outdir, f"case1_eps{eps:g}.pt"))

    ev = s.evaluate(nx=a.nx, nt=a.nt)
    X, T, ue, up, err = ev["X"], ev["T"], ev["u_exact"], ev["u_pred"], ev["err"]
    pts = s.final_points
    fig, axs = plt.subplots(2, 3, figsize=(14.0, 7.4))
    vmin, vmax = float(min(ue.min(), up.min())), float(max(ue.max(), up.max()))
    lv = np.linspace(vmin, vmax, 25)
    for ax, d, ttl in ((axs[0, 0], ue, "Exact solution"), (axs[0, 1], up, "PINN prediction")):
        cf = ax.contourf(X, T, d, levels=lv, cmap="viridis")
        ax.set_title(ttl); ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$t$")
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    epos = np.maximum(err, 1e-14)
    lo, hi = math.floor(math.log10(epos.min())), math.ceil(math.log10(epos.max()))
    lev = np.logspace(lo, hi, max(9, (hi - lo) * 3 + 1))
    cf = axs[0, 2].contourf(X, T, epos, levels=lev, cmap="RdPu",
                            norm=LogNorm(vmin=lev[0], vmax=lev[-1]), extend="both")
    axs[0, 2].contour(X, T, epos, levels=lev[::3], colors="k", linewidths=0.3, alpha=0.45)
    axs[0, 2].set_title("|u_pred - u_exact|  (rel L2 = %.2e)" % last["rel_l2"])
    axs[0, 2].set_xlabel(r"$x$"); axs[0, 2].set_ylabel(r"$t$")
    cb = fig.colorbar(cf, ax=axs[0, 2], fraction=0.046, pad=0.03)
    cb.set_label("abs error", fontsize=8)

    axs[1, 0].scatter(pts[:, 0], pts[:, 1], s=2, c="k", alpha=0.35)
    axs[1, 0].set_title("collocation points $(x,t)$")
    axs[1, 0].set_xlabel(r"$x$"); axs[1, 0].set_ylabel(r"$t$")

    xs = np.linspace(0.0, thr, 400)
    Z = np.column_stack([xs, np.full_like(xs, a.t_show)])
    ts = np.full_like(xs, a.t_show)
    axs[1, 1].plot(xs, np.asarray(s.problem.exact(xs, ts), float),
                   "k-", lw=2, label="exact")
    axs[1, 1].plot(xs, s.predict(Z), "r--", lw=1.4, label="PINN")
    axs[1, 1].set_xlim(0, thr)
    zs = abs(math.cos(math.pi * a.t_show)) < 1e-8
    axs[1, 1].set_title("layer zoom, t=%.2f%s"
                        % (a.t_show, "  [cos(pi t)=0: exact vanishes]" if zs else ""))
    if zs:
        print("  [警告] t=%g 处 cos(pi t)=0, 精确解恒为零" % a.t_show, flush=True)
    axs[1, 1].set_xlabel(r"$x$"); axs[1, 1].legend(fontsize=8)
    axs[1, 1].grid(alpha=0.3)

    m = pts[:, 0] <= thr
    axs[1, 2].scatter(pts[m, 0], pts[m, 1], s=4, c="crimson", alpha=0.6)
    axs[1, 2].set_xlim(0, thr)
    axs[1, 2].set_title("points with x <= 20 sqrt(eps)  (%d of %d)" % (m.sum(), len(pts)))
    axs[1, 2].set_xlabel(r"$x$"); axs[1, 2].set_ylabel(r"$t$")
    fig.suptitle("Case 1 (layer at x=0),  eps=%g" % eps, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    print("  ->", PLT._save(fig, os.path.join(a.outdir, f"case1_eps{eps:g}.pdf")), flush=True)


if __name__ == "__main__":
    main()
