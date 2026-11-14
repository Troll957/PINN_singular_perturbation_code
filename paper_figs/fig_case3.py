#!/usr/bin/env python3
"""Case 3: solution at five times, absolute error, point distribution per time."""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style as st

TIMES = [0.0, 0.4, 0.6, 0.8, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/case3_eps1e-6.npz")
    ap.add_argument("--out", default="figs/case3.pdf")
    ap.add_argument("--out-err", default="figs/case3_err.pdf")
    ap.add_argument("--out-pts", default="figs/case3_pts.pdf")
    a = ap.parse_args()

    d = np.load(a.npz)
    g, times = d["grid"], d["times"]
    ue, up = d["ue"], d["up"]
    pts = d["pts_int"]
    X, Y = np.meshgrid(g, g, indexing="ij")

    # --- solutions at the requested times -----------------------------------
    fig, axs = plt.subplots(1, 5, figsize=(7.2, 1.85))
    vmin = float(min(ue.min(), up.min())); vmax = float(max(ue.max(), up.max()))
    for ax, t in zip(axs, TIMES):
        k = int(np.argmin(np.abs(times - t)))
        im = ax.pcolormesh(X, Y, up[k], cmap=st.CMAP_U, shading="auto",
                           vmin=vmin, vmax=vmax, rasterized=True)
        ax.contour(X, Y, ue[k], levels=6, colors="w", linewidths=0.35, alpha=0.8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(rf"$t={t}$", fontsize=7, pad=2)
    cb = fig.colorbar(im, ax=axs, fraction=0.022, pad=0.01)
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.5)
    cb.set_label(r"$u_\theta$", fontsize=7)
    st.save(fig, a.out)

    # --- absolute error over time -------------------------------------------
    err = np.abs(up - ue).reshape(len(times), -1)
    mean_abs = err.mean(axis=1)
    max_abs = err.max(axis=1)
    fig, ax = plt.subplots(figsize=(2.6, 2.0))
    ax.semilogy(times, max_abs, "o-", ms=3, lw=1.0, label="max")
    ax.semilogy(times, mean_abs, "s--", ms=3, lw=1.0, label="mean")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$|u_\theta-u^*|$")
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.25)
    st.save(fig, a.out_err)

    # --- points at the same times -------------------------------------------
    fig, axs = plt.subplots(1, 5, figsize=(7.2, 1.85))
    for ax, t in zip(axs, TIMES):
        m = np.abs(pts[:, 2] - t) < 0.02
        ax.scatter(pts[m, 0], pts[m, 1], s=0.25, c="k", alpha=0.5, linewidths=0)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(rf"$t={t}$", fontsize=7, pad=2)
    st.save(fig, a.out_pts)


if __name__ == "__main__":
    main()
