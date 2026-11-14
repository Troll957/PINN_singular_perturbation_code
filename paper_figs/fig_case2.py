#!/usr/bin/env python3
"""Case 2: exact solution, prediction, error and point distribution at t=0.25."""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/case2_eps1e-5.npz")
    ap.add_argument("--out", default="figs/case2.pdf")
    ap.add_argument("--t", type=float, default=0.25)
    a = ap.parse_args()

    d = np.load(a.npz)
    g, times = d["grid"], d["times"]
    k = int(np.argmin(np.abs(times - a.t)))
    ue, up = d["ue"][k], d["up"][k]
    pts = d["pts_int"]
    X, Y = np.meshgrid(g, g, indexing="ij")
    rel = float(d["rel_l2"][-1])

    fig, axs = plt.subplots(2, 2, figsize=(5.0, 4.6))
    vmin, vmax = float(ue.min()), float(ue.max())

    ax = axs[0, 0]
    st.field(ax, X, Y, ue, vmin=vmin, vmax=vmax, fig=fig, label=r"$u^*$")
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); st.panel(ax, "a")

    ax = axs[0, 1]
    st.field(ax, X, Y, up, vmin=vmin, vmax=vmax, fig=fig, label=r"$u_\theta$")
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); st.panel(ax, "b")

    ax = axs[1, 0]
    st.field(ax, X, Y, np.log10(np.abs(up - ue) + 1e-16), cmap=st.CMAP_E, fig=fig,
             label=r"$\log_{10}|u_\theta-u^*|$")
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); st.panel(ax, "c")

    ax = axs[1, 1]
    m = np.abs(pts[:, 2] - a.t) < 0.02
    ax.scatter(pts[m, 0], pts[m, 1], s=0.6, c="k", alpha=0.5, linewidths=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_title(rf"$N={pts.shape[0]}$, $\varepsilon=10^{{-5}}$, rel. $L^2={rel:.1e}$",
                 fontsize=6.5, pad=2)
    st.panel(ax, "d")

    fig.tight_layout(pad=0.25, w_pad=0.5, h_pad=0.5)
    st.save(fig, a.out)


if __name__ == "__main__":
    main()
