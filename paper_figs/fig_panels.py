#!/usr/bin/env python3
"""Single-panel PDFs for Case 1; the layout is assembled in LaTeX.

Usage: python fig_panels.py --npz data/case1_eps1e-5.npz --prefix case1
Writes <prefix>_exact.pdf, <prefix>_pred.pdf, <prefix>_err.pdf, <prefix>_pts.pdf.
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style as st


def surf(ax, X, T, U, vmin, vmax):
    s0 = max(1, X.shape[0] // 90)
    s1 = max(1, X.shape[1] // 50)
    ax.plot_surface(X[::s0, ::s1], T[::s0, ::s1], U[::s0, ::s1],
                    cmap="viridis", linewidth=0, antialiased=True,
                    vmin=vmin, vmax=vmax, rcount=60, ccount=60)
    ax.set_xlabel(r"$x$", labelpad=-2); ax.set_ylabel(r"$t$", labelpad=-2)
    ax.set_zlabel(r"$u$", labelpad=0)
    ax.tick_params(labelsize=6, pad=-1)
    ax.view_init(elev=22, azim=-62)
    try:
        ax.set_box_aspect(None, zoom=1.12)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/case1_eps1e-5.npz")
    ap.add_argument("--prefix", default="case1")
    ap.add_argument("--outdir", default="figs")
    a = ap.parse_args()

    d = np.load(a.npz)
    X, T, ue, up, err = d["X"], d["T"], d["u_exact"], d["u_pred"], d["err"]
    vmin, vmax = float(ue.min()), float(ue.max())
    P = a.prefix

    fig = plt.figure(figsize=(3.1, 2.8)); ax = fig.add_subplot(111, projection="3d")
    surf(ax, X, T, ue, vmin, vmax)
    ax.set_title("Exact solution", fontsize=8.5, pad=-2)
    fig.subplots_adjust(left=0.05, right=0.88, bottom=0.15, top=0.90)
    st.save(fig, f"{a.outdir}/{P}_exact.pdf", bbox=None)

    fig = plt.figure(figsize=(3.1, 2.8)); ax = fig.add_subplot(111, projection="3d")
    surf(ax, X, T, up, vmin, vmax)
    ax.set_title("PINN prediction", fontsize=8.5, pad=-2)
    fig.subplots_adjust(left=0.05, right=0.88, bottom=0.15, top=0.90)
    st.save(fig, f"{a.outdir}/{P}_pred.pdf", bbox=None)

    fig, ax = plt.subplots(figsize=(2.7, 2.4))
    im = ax.pcolormesh(X, T, np.log10(err + 1e-16), cmap=st.CMAP_E,
                       shading="auto", rasterized=True)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$t$")
    ax.set_title(rf"$\log_{{10}}|u_\theta-u^*|$  (max $={err.max():.2e}$)",
                 fontsize=8, pad=4)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=6, length=2, width=0.5)
    fig.tight_layout(pad=0.2)
    st.save(fig, f"{a.outdir}/{P}_err.pdf")

    fig, ax = plt.subplots(figsize=(2.7, 2.4))
    pts = d["points"]
    st.density_scatter(ax, pts[:, 0], pts[:, 1], fig=fig, label=r"points per bin")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$t$")
    ax.set_title(rf"Collocation points, $N={pts.shape[0]}$", fontsize=8, pad=4)
    fig.tight_layout(pad=0.2)
    st.save(fig, f"{a.outdir}/{P}_pts.pdf")


if __name__ == "__main__":
    main()
