#!/usr/bin/env python3
"""Case 2: single-panel PDFs at a fixed time (2D fields, no 3D surfaces)."""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style as st


def field_panel(X, Y, Z, ttl, path, cmap="viridis", vmin=None, vmax=None, label=""):
    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto", rasterized=True,
                       vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_title(ttl, fontsize=8, pad=4)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=6, length=2, width=0.5)
    if label:
        cb.set_label(label, fontsize=7)
    fig.tight_layout(pad=0.2)
    st.save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/case2_eps1e-5.npz")
    ap.add_argument("--prefix", default="case2")
    ap.add_argument("--t", type=float, default=0.25)
    ap.add_argument("--outdir", default="figs")
    a = ap.parse_args()

    d = np.load(a.npz)
    g, times = d["grid"], d["times"]
    k = int(np.argmin(np.abs(times - a.t)))
    ue, up = d["ue"][k], d["up"][k]
    X, Y = np.meshgrid(g, g, indexing="ij")
    vmin, vmax = float(ue.min()), float(ue.max())
    P = a.prefix

    field_panel(X, Y, ue, r"Exact solution", f"{a.outdir}/{P}_exact.pdf",
                vmin=vmin, vmax=vmax)
    field_panel(X, Y, up, r"PINN prediction", f"{a.outdir}/{P}_pred.pdf",
                vmin=vmin, vmax=vmax)
    field_panel(X, Y, np.log10(np.abs(up - ue) + 1e-16),
                r"$\log_{10}|u_\theta-u^*|$", f"{a.outdir}/{P}_err.pdf",
                cmap=st.CMAP_E)

    # zoom on the corner where the two layers meet
    for tag, U in (("exact", ue), ("pred", up)):
        fig, ax = plt.subplots(figsize=(2.9, 2.6))
        im = ax.pcolormesh(X, Y, U, cmap="viridis", shading="auto",
                           rasterized=True, vmin=vmin, vmax=vmax)
        ax.set_xlim(0.0, 0.06); ax.set_ylim(0.0, 0.06)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
        ax.set_title("zoom on the corner", fontsize=8, pad=4)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.ax.tick_params(labelsize=6, length=2, width=0.5)
        fig.tight_layout(pad=0.2)
        st.save(fig, f"{a.outdir}/{P}_{tag}_zoom.pdf")

    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    pts = d["pts_int"]
    m = np.abs(pts[:, 2] - a.t) < 0.02
    st.density_scatter(ax, pts[m, 0], pts[m, 1], fig=fig, label=r"points per bin")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_title(rf"Collocation points, $|t-{a.t}|<0.02$", fontsize=8, pad=4)
    fig.tight_layout(pad=0.2)
    st.save(fig, f"{a.outdir}/{P}_pts.pdf")


if __name__ == "__main__":
    main()
