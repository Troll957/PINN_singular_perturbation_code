#!/usr/bin/env python3
"""Case 1 at eps=1e-6: exact, prediction, error, layer zoom, points."""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/case1_eps1e-6.npz")
    ap.add_argument("--out", default="figs/case1.pdf")
    ap.add_argument("--t", type=float, default=0.5)
    a = ap.parse_args()

    d = np.load(a.npz)
    X, T = d["X"], d["T"]
    ue, up, err = d["u_exact"], d["u_pred"], d["err"]
    pts = d["points"]
    j = int(np.argmin(np.abs(T[0, :] - a.t)))

    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.3))
    vmin, vmax = float(ue.min()), float(ue.max())

    ax = axs[0, 0]
    st.field(ax, X, T, ue, vmin=vmin, vmax=vmax, fig=fig, label=r"$u^*$")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$t$"); ax.set_aspect("auto")
    st.panel(ax, "a")

    ax = axs[0, 1]
    st.field(ax, X, T, up, vmin=vmin, vmax=vmax, fig=fig, label=r"$u_\theta$")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$t$"); ax.set_aspect("auto")
    st.panel(ax, "b")

    ax = axs[0, 2]
    st.field(ax, X, T, np.log10(err + 1e-16), cmap=st.CMAP_E, fig=fig,
             label=r"$\log_{10}|u_\theta-u^*|$")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$t$"); ax.set_aspect("auto")
    st.panel(ax, "c")

    # zoom on the layer: profiles at t = a.t
    x = X[:, 0]
    ax = axs[1, 0]
    ax.semilogx(x, ue[:, j], "-", lw=1.2, color="C0", label=r"$u^*$")
    ax.semilogx(x, up[:, j], "--", lw=1.0, color="C3", label=r"$u_\theta$")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(rf"$u(x,{a.t})$")
    ax.set_xlim(1e-4, 5e-2); ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, loc="lower right")
    st.panel(ax, "d")

    ax = axs[1, 1]
    m = x <= 5e-2
    ax.semilogy(x[m], err[m, j] + 1e-16, "-", lw=1.0, color="C3")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(rf"$|u_\theta-u^*|$ at $t={a.t}$")
    ax.set_xlim(0, 5e-2)
    st.panel(ax, "e")

    ax = axs[1, 2]
    ax.scatter(pts[:, 0], pts[:, 1], s=0.30, c="k", alpha=0.55, linewidths=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$t$")
    ax.set_title(rf"$N={pts.shape[0]}$", fontsize=7, pad=2)
    st.panel(ax, "f")

    fig.tight_layout(pad=0.25, w_pad=0.5, h_pad=0.5)
    st.save(fig, a.out)


if __name__ == "__main__":
    main()
