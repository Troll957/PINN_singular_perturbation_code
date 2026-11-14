#!/usr/bin/env python3
"""Case 3: exact and predicted solution at t=0, 0.5, 1, and the points there."""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style as st

TIMES = [0.25, 0.5, 0.75]
TAGS = ["t025", "t050", "t075"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/case3_eps1e-6.npz")
    ap.add_argument("--prefix", default="case3")
    ap.add_argument("--outdir", default="figs")
    a = ap.parse_args()

    d = np.load(a.npz)
    g, times = d["grid"], d["times"]
    ue, up, pts = d["ue"], d["up"], d["pts_int"]
    X, Y = np.meshgrid(g, g, indexing="ij")
    amp = float(max(abs(ue).max(), abs(up).max()))
    vmax, vmin = amp, -amp

    for t, tag in zip(TIMES, TAGS):
        k = int(np.argmin(np.abs(times - t)))
        for name, U in (("exact", ue[k]), ("pred", up[k])):
            if name == "exact":
                ttl = rf"Exact solution, $t={t}$"
            else:
                ttl = rf"PINN, $t={t}$"
            fig, ax = plt.subplots(figsize=(2.6, 2.4))
            im = ax.pcolormesh(X, Y, U, cmap="RdBu_r", shading="auto",
                               rasterized=True, vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
            ax.set_title(ttl, fontsize=8, pad=4)
            cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
            cb.ax.tick_params(labelsize=6, length=2, width=0.5)
            fig.tight_layout(pad=0.2)
            st.save(fig, f"{a.outdir}/{a.prefix}_{name}_{tag}.pdf")

        m = np.abs(pts[:, 2] - t) < 0.02
        fig, ax = plt.subplots(figsize=(2.6, 2.4))
        st.density_scatter(ax, pts[m, 0], pts[m, 1], fig=fig, label=r"points per bin")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
        ax.set_title(rf"Points, $t={t}$", fontsize=8, pad=4)
        fig.tight_layout(pad=0.2)
        st.save(fig, f"{a.outdir}/{a.prefix}_pts_{tag}.pdf")


if __name__ == "__main__":
    main()
