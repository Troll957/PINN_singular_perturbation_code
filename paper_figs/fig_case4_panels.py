#!/usr/bin/env python3
"""Case 4: single-panel PDFs on the non-rectangular domain.

The field is *clipped* by the level-set curve p(x,y)=0 (as in the original
plotting code), not masked cell by cell, so the boundary is smooth.
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import paper_style as st

A_WAVE = 0.30


def radius(theta: np.ndarray) -> np.ndarray:
    """Outer boundary p = r^2 - 1 - 0.3 r^3 cos(3 theta) = 0, solved by bisection."""
    out = np.empty_like(theta)
    c = np.cos(3.0 * theta)
    for i, cc in enumerate(c):
        lo, hi = 0.5, 1.6
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f = mid * mid - 1.0 - A_WAVE * mid ** 3 * cc
            if f < 0:
                lo = mid
            else:
                hi = mid
        out[i] = 0.5 * (lo + hi)
    return out


def domain_path(n=1200):
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = radius(th)
    verts = np.column_stack([r * np.cos(th), r * np.sin(th)])
    verts = np.vstack([verts, verts[:1]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)


def field_panel(X, Y, U, ttl, path, cpath, cmap="viridis", log=False,
                vmin=None, vmax=None, label=""):
    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    Z = np.log10(U + 1e-16) if log else U
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto", rasterized=True,
                       vmin=vmin, vmax=vmax)
    clip = PathPatch(cpath, transform=ax.transData, fc="none", ec="none")
    ax.add_patch(clip)
    im.set_clip_path(clip)
    ax.add_patch(PathPatch(cpath, transform=ax.transData, fc="none",
                           ec="k", lw=0.8))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_title(ttl, fontsize=8, pad=4)
    ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max())
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=6, length=2, width=0.5)
    if label:
        cb.set_label(label, fontsize=7)
    fig.tight_layout(pad=0.2)
    st.save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/case4_eps1e-4_paper.npz")
    ap.add_argument("--prefix", default="case4")
    ap.add_argument("--outdir", default="figs")
    a = ap.parse_args()

    d = np.load(a.npz)
    X, Y = d["X1"], d["X2"]
    ue, up, err = d["ue"], d["up"], d["err"]
    cpath = domain_path()
    vmin, vmax = float(np.nanmin(ue)), float(np.nanmax(ue))
    P = a.prefix

    field_panel(X, Y, ue, r"Exact solution", f"{a.outdir}/{P}_exact.pdf",
                cpath, vmin=vmin, vmax=vmax)
    field_panel(X, Y, up, r"PINN prediction", f"{a.outdir}/{P}_pred.pdf",
                cpath, vmin=vmin, vmax=vmax)
    field_panel(X, Y, err, r"$\log_{10}|u_\theta-u^*|$",
                f"{a.outdir}/{P}_err.pdf", cpath, cmap=st.CMAP_E, log=True)

    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    pts = d["points"]
    st.density_scatter(ax, pts[:, 0], pts[:, 1], fig=fig, label=r"points per bin")
    ax.add_patch(PathPatch(cpath, transform=ax.transData, fc="none",
                           ec="k", lw=0.8))
    ax.set_aspect("equal")
    ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max())
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_title(rf"Collocation points, $N={pts.shape[0]}$", fontsize=8, pad=4)
    fig.tight_layout(pad=0.2)
    st.save(fig, f"{a.outdir}/{P}_pts.pdf")


if __name__ == "__main__":
    main()
