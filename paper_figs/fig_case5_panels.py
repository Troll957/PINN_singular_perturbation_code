#!/usr/bin/env python3
"""Case 5: source and solution panels for the inverse problem."""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style as st


def panel(X, Y, Z, ttl, path, cmap="viridis", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(2.9, 2.5))
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto", rasterized=True,
                       vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_title(ttl, fontsize=8, pad=4)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=6, length=2, width=0.5)
    fig.tight_layout(pad=0.2)
    st.save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/case5_fields.npz")
    ap.add_argument("--prefix", default="case5")
    ap.add_argument("--outdir", default="figs")
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=True)
    g = d["grid"]
    X, Y = np.meshgrid(g, g, indexing="ij")
    ft, fp = d["f_true"], d["f_pred"]
    ut, up = d["u_true"], d["u_pred"]
    P = a.prefix

    vf = float(max(abs(ft).max(), abs(fp).max()))
    panel(X, Y, ft, "Exact source", f"{a.outdir}/{P}_f_true.pdf",
          vmin=-vf, vmax=vf) if False else panel(X, Y, ft, "Exact source",
          f"{a.outdir}/{P}_f_true.pdf", cmap="RdBu_r", vmin=-vf, vmax=vf)
    panel(X, Y, fp, "Recovered source", f"{a.outdir}/{P}_f_pred.pdf",
          cmap="RdBu_r", vmin=-vf, vmax=vf)
    panel(X, Y, np.log10(np.abs(fp - ft) + 1e-16), r"$\log_{10}|f-f_\theta|$",
          f"{a.outdir}/{P}_f_err.pdf", cmap=st.CMAP_E)

    vu = float(max(abs(ut).max(), abs(up).max()))
    panel(X, Y, ut, "Exact solution", f"{a.outdir}/{P}_u_true.pdf",
          cmap="RdBu_r", vmin=-vu, vmax=vu)
    panel(X, Y, up, "PINN solution", f"{a.outdir}/{P}_u_pred.pdf",
          cmap="RdBu_r", vmin=-vu, vmax=vu)
    panel(X, Y, np.log10(np.abs(up - ut) + 1e-16), r"$\log_{10}|u-u_\theta|$",
          f"{a.outdir}/{P}_u_err.pdf", cmap=st.CMAP_E)

    fig, ax = plt.subplots(figsize=(2.9, 2.5))
    pts = d["pts"]
    st.density_scatter(ax, pts[:, 0], pts[:, 1], fig=fig, label=r"points per bin")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_title(rf"Collocation points, $N={pts.shape[0]}$", fontsize=8, pad=4)
    fig.tight_layout(pad=0.2)
    st.save(fig, f"{a.outdir}/{P}_pts.pdf")


if __name__ == "__main__":
    main()
