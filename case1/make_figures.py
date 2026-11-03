"""汇总自适应配点实验并生成全部图表。

用法::

    python make_figures.py --outdir sampling_results --figdir figures_sampling
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spp_pinn import plotting as P

LABELS = {
    "adapt-vor": "residual + Voronoi weights",
    "adapt-mean": "residual + plain mean",
    "uniE-vor": "uniform enrichment",
    "fixed": "fixed uniform",
    "hard-adapt": "hard constraint, adaptive",
    "hard-fixed": "hard constraint, fixed",
}


def load_results(outdir: str) -> Dict[str, Tuple[dict, dict]]:
    out = {}
    for f in sorted(glob.glob(os.path.join(outdir, "*.json"))):
        if os.path.basename(f) == "summary.json":
            continue
        with open(f) as fh:
            meta = json.load(fh)
        tag = meta["tag"]
        p = os.path.join(outdir, f"{tag}.npz")
        out[tag] = (meta, dict(np.load(p)) if os.path.exists(p) else {})
    return out


def _row(meta, label):
    f = meta["final"]
    return {"label": label, "N": f["N"], "max_err": f["max_err"],
            "rel_l2": f["rel_l2"], "frac_layer": f["frac_layer"]}


# ----------------------------------------------------------------------
def fig_strategy(res, figdir, eps_tag, eps):
    keys = [("S1-adapt-vor", "adapt-vor"), ("S1-adapt-mean", "adapt-mean"),
            ("S1-uniE-vor", "uniE-vor"), ("S1-fixed", "fixed")]
    rows = []
    for pre, lab in keys:
        t = f"{pre}-eps{eps_tag}"
        if t in res:
            rows.append(_row(res[t][0], LABELS[lab]))
    if not rows:
        return
    P.plot_strategy_bars(rows, eps, os.path.join(figdir, f"S1_strategy_eps{eps_tag}.png"))


def fig_convergence(res, figdir, eps_tag, eps):
    curves = {}
    for pre, lab in (("S1-adapt-vor", "residual + Voronoi"),
                     ("S1-adapt-mean", "residual + plain mean"),
                     ("S1-uniE-vor", "uniform enrichment")):
        t = f"{pre}-eps{eps_tag}"
        if t in res:
            h = res[t][0]["history"]
            curves[lab] = (h, "o-" if "Voronoi" in lab else "s--")

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for lab, (h, st) in curves.items():
        axs[0].semilogy([q["N"] for q in h], [q["max_err"] for q in h], st,
                        lw=1.9, ms=6, label=lab)
        axs[1].semilogy([q["N"] for q in h], [q["rms_res_uniform"] for q in h], st,
                        lw=1.9, ms=6, label=lab)
    ftag = f"S1-fixed-eps{eps_tag}"
    if ftag in res:
        f = res[ftag][0]["final"]
        for ax in axs:
            ax.axhline(f["max_err"] if ax is axs[0] else f["rms_res_uniform"],
                       color="gray", ls=":", lw=2, label="fixed uniform")
    axs[0].set_xlabel("collocation points $N$"); axs[0].set_ylabel(r"$\|u-u^*\|_\infty$")
    axs[0].set_title("Solution error vs. point budget")
    axs[1].set_xlabel("collocation points $N$")
    axs[1].set_ylabel(r"RMS residual on a uniform test set")
    axs[1].set_title("Objective residual vs. point budget")
    for ax in axs:
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.suptitle(rf"Adaptive sampling strategies ($\varepsilon={eps:g}$)", y=1.02)
    P._save(fig, os.path.join(figdir, f"S1_convergence_eps{eps_tag}.png"))


def fig_layer(res, figdir, eps_tag, eps):
    curves = {}
    for pre, lab in (("S1-adapt-vor", "residual + Voronoi"),
                     ("S1-uniE-vor", "uniform enrichment")):
        t = f"{pre}-eps{eps_tag}"
        if t in res:
            curves[lab] = res[t][0]["history"]
    if curves:
        P.plot_layer_fraction(curves, eps,
                              os.path.join(figdir, f"S1_layer_eps{eps_tag}.png"))


def fig_points(res, figdir, tag, name, eps, T=1.0):
    if tag not in res:
        return
    meta, npz = res[tag]
    if "points" not in npz:
        return
    P.plot_point_cloud(npz["points"], npz["abs_res"], npz.get("ic", np.zeros((0, 2))),
                       npz.get("bc", np.zeros((0, 2))), eps, T,
                       os.path.join(figdir, f"{name}_points.png"),
                       title=tag)


def fig_fields(res, figdir, tag, name, eps, layer):
    if tag not in res:
        return
    meta, npz = res[tag]
    ev = {"X": npz["X"], "T": npz["T"], "u_pred": npz["u_pred"],
          "u_exact": npz["u_exact"], "err": npz["err"],
          "x": npz["X"][:, 0], "t": npz["T"][0, :], "max_err": meta["final"]["max_err"]}
    P.plot_solution_fields(ev, eps, layer, os.path.join(figdir, f"{name}_fields.png"))
    P.plot_cross_sections(ev, eps, os.path.join(figdir, f"{name}_sections.png"))


def fig_eps_sweep(res, figdir):
    rows = []
    for tag, (meta, _) in res.items():
        if tag.startswith("S1-") or tag.startswith("S2-"):
            f = meta["final"]
            rows.append({"kind": meta["kind"], "eps": meta["cfg"]["eps"],
                         "max_err": f["max_err"], "rel_l2": f["rel_l2"]})
    # 每个 eps 只保留一个自适应/固定代表
    keep = {}
    for r in rows:
        key = (r["kind"], r["eps"])
        if key not in keep or r["max_err"] < keep[key]["max_err"]:
            keep[key] = r
    rows = list(keep.values())
    if rows:
        P.plot_eps_sweep(rows, os.path.join(figdir, "S2_eps_sweep.png"))


def fig_hard(res, figdir, tag_a="S5-hard-adapt-eps1e-05",
             tag_f="S5-hard-fixed-eps1e-05", eps=1e-5):
    rows = []
    for t, lab in ((tag_a, LABELS["hard-adapt"]), (tag_f, LABELS["hard-fixed"]),
                   ("S1-adapt-vor-eps1e-05", "soft constraint, adaptive"),
                   ("S1-fixed-eps1e-05", "soft constraint, fixed")):
        if t in res:
            rows.append(_row(res[t][0], lab))
    if rows:
        P.plot_strategy_bars(rows, eps, os.path.join(figdir, "S5_hard_vs_soft.png"))


def fig_round_history(res, figdir, tag, name, eps):
    if tag not in res:
        return
    P.plot_round_history(res[tag][0]["history"],
                         os.path.join(figdir, f"{name}_history.png"), eps=eps)


def write_table(res, path):
    lines = ["| tag | problem | eps | kind | strategy | weighting | N | max err | rel L2 |"
             " layer frac | wall [s] |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, (meta, _) in sorted(res.items()):
        f, c = meta["final"], meta["cfg"]
        lines.append(f"| {tag} | {c['problem']} | {c['eps']:g} | {meta['kind']} | "
                     f"{c['strategy']} | {c['weighting']} | {f['N']} | {f['max_err']:.3e} | "
                     f"{f['rel_l2']:.3e} | {f['frac_layer']:.3f} | "
                     f"{f.get('wall_total', float('nan')):.1f} |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="sampling_results")
    ap.add_argument("--figdir", default="figures_sampling")
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)

    res = load_results(args.outdir)
    print(f"载入 {len(res)} 个算例")
    if not res:
        return

    for etag, eps in (("1e-05", 1e-5), ("1e-04", 1e-4)):
        fig_strategy(res, args.figdir, etag, eps)
        fig_convergence(res, args.figdir, etag, eps)
        fig_layer(res, args.figdir, etag, eps)

    fig_points(res, args.figdir, "S1-adapt-vor-eps1e-05", "S1_adapt", 1e-5)
    fig_points(res, args.figdir, "S1-uniE-vor-eps1e-05", "S1_uniformE", 1e-5)
    fig_fields(res, args.figdir, "S1-adapt-vor-eps1e-05", "S1_adapt", 1e-5, "x=0")
    fig_round_history(res, args.figdir, "S1-adapt-vor-eps1e-05", "S1_adapt", 1e-5)

    fig_eps_sweep(res, args.figdir)

    for tag, name, eps, prob in (
            ("S3-adapt-interior-eps1e-04", "S3_interior", 1e-4, "interior"),
            ("S3-adapt-right-eps1e-05", "S3_right", 1e-5, "right")):
        fig_points(res, args.figdir, tag, name, eps)
        fig_fields(res, args.figdir, tag, name, eps, prob)

    fig_hard(res, args.figdir)
    print("表格:", write_table(res, os.path.join(args.figdir, "results_table.md")))
    print("图目录:", args.figdir)


if __name__ == "__main__":
    main()
