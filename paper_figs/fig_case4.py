#!/usr/bin/env python3
"""Case 4: exact solution, prediction, error and points on the curved domain."""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/case4_eps1e-4.npz")
    ap.add_argument("--out", default="figs/case4.pdf")
    a = ap.parse_args()

    d = np.load(a.npz)
    X1, X2, ue, up, err = d["X1"], d["X2"], d["ue"], d["up"], d["err"]
    pts = d["points"]

    fig, axs = plt.subplots(2, 2, figsize=(5.0, 4.8))
    vmin, vmax = float(np.nanmin(ue)), float(np.nanmax(ue))

    ax = axs[0, 0]
    st.field(ax, X1, X2, ue, vmin=vmin, vmax=vmax, fig=fig, label=r"$u^*$")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$"); st.panel(ax, "a")

    ax = axs[0, 1]
    st.field(ax, X1, X2, up, vmin=vmin, vmax=vmax, fig=fig, label=r"$u_\theta$")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$"); st.panel(ax, "b")

    ax = axs[1, 0]
    st.field(ax, X1, X2, np.log10(err + 1e-16), cmap=st.CMAP_E, fig=fig,
             label=r"$\log_{10}|u_\theta-u^*|$")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$"); st.panel(ax, "c")

    ax = axs[1, 1]
    ax.scatter(pts[:, 0], pts[:, 1], s=0.35, c="k", alpha=0.5, linewidths=0)
    ax.set_aspect("equal"); ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
    ax.set_title(rf"$N={pts.shape[0]}$", fontsize=7, pad=2)
    st.panel(ax, "d")

    fig.tight_layout(pad=0.25, w_pad=0.5, h_pad=0.5)
    st.save(fig, a.out)


if __name__ == "__main__":
    main()
