#!/usr/bin/env python3
"""Case 1 in the 3D-surface style of the original plotting module."""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style as st


def surf(ax, X, T, U, vmin, vmax, title):
    s = max(1, X.shape[0] // 90), max(1, X.shape[1] // 50)
    ax.plot_surface(X[::s[0], ::s[1]], T[::s[0], ::s[1]], U[::s[0], ::s[1]],
                    cmap="viridis", linewidth=0, antialiased=True,
                    vmin=vmin, vmax=vmax, rcount=60, ccount=60)
    ax.set_title(title, fontsize=8.5, pad=4)
    ax.set_xlabel(r"$x$", labelpad=-4); ax.set_ylabel(r"$t$", labelpad=-4)
    ax.set_zlabel(r"$u$", labelpad=1)
    ax.tick_params(labelsize=5.5, pad=-1)
    ax.view_init(elev=22, azim=-62)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/case1_eps1e-6.npz")
    ap.add_argument("--out", default="figs/case1.pdf")
    a = ap.parse_args()

    d = np.load(a.npz)
    X, T, ue, up, err = d["X"], d["T"], d["u_exact"], d["u_pred"], d["err"]
    vmin, vmax = float(ue.min()), float(ue.max())

    fig = plt.figure(figsize=(5.0, 4.4))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    surf(ax, X, T, ue, vmin, vmax, "Exact solution")

    ax = fig.add_subplot(2, 2, 2, projection="3d")
    surf(ax, X, T, up, vmin, vmax, "PINN prediction")

    ax = fig.add_subplot(2, 2, 3)
    im = ax.pcolormesh(X, T, np.log10(err + 1e-16), cmap="inferno", shading="auto",
                       rasterized=True)
    ax.set_title(rf"$\log_{{10}}|u_\theta-u^*|$  (max $={err.max():.2e}$)",
                 fontsize=8.5, pad=4)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$t$"); ax.set_aspect("auto")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6, length=2, width=0.5)

    ax = fig.add_subplot(2, 2, 4)
    pts = d["points"]
    st.density_scatter(ax, pts[:, 0], pts[:, 1], fig=fig,
                       label=r"points per bin")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$t$")
    ax.set_title(rf"collocation points, $N={pts.shape[0]}$", fontsize=8.5, pad=4)

    fig.subplots_adjust(left=0.03, right=0.96, bottom=0.08, top=0.93, wspace=0.26, hspace=0.18)
    st.save(fig, a.out)


if __name__ == "__main__":
    main()
