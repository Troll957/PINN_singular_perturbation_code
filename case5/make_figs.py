#!/usr/bin/env python
"""从已保存的 run 目录重新生成插图（不重新训练）。

用法::

    python make_figs.py --run runs/e4_big_adapt
    python make_figs.py --run runs/e4_big_adapt --fixed runs/e4_big_fixed
    CASE5_FIG_EXT=png python make_figs.py --run runs/e4_big_adapt   # 要位图时

图默认存成 PDF（论文插图用），文件名：``params / history / solution /
source / points``；给了 ``--fixed`` 时额外产出 ``points_compare``。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from case5 import Case5Config, InversePINN, plotting


def load_run(run_dir: str) -> InversePINN:
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        cfg_d = json.load(fh)
    keys = {k: v for k, v in cfg_d.items() if k in Case5Config.__dataclass_fields__}
    if isinstance(keys.get("hidden"), list):
        keys["hidden"] = tuple(keys["hidden"])
    for k in ("slab", "x_span", "y_span"):
        if isinstance(keys.get(k), list):
            keys[k] = tuple(keys[k])
    cfg = Case5Config(**keys)
    cfg.rounds = 0
    cfg.epochs_per_round = 1
    solver = InversePINN(cfg)
    state = torch.load(os.path.join(run_dir, "state.pt"), map_location=solver.device)
    solver.model.load_state_dict(state["model"])
    solver.params.load_state_dict(state["params"])
    solver.history = json.load(open(os.path.join(run_dir, "history.json"),
                                    encoding="utf-8"))
    pts = np.load(os.path.join(run_dir, "data.npz"))["points"]
    solver.final_points = pts
    return solver


def draw(solver: InversePINN, outdir: str, tag: str = "") -> None:
    truth = solver.problem.truth_dict
    names = solver.param_names
    fp = plotting.fig_path
    plotting.plot_params(solver.history, truth, fp(outdir, f"params{tag}"),
                         names=names)
    plotting.plot_history(solver.history, fp(outdir, f"history{tag}"))
    cut_y = 0.5 if solver.cfg.case == "moving" else float(truth.get("ys", 0.5))
    plotting.plot_solution(solver.problem, solver.predict, truth,
                           fp(outdir, f"solution{tag}"), cut_y=cut_y)
    plotting.plot_source(solver.problem, truth, solver.params.numpy_values(),
                         fp(outdir, f"source{tag}"))
    obs = np.load(os.path.join(outdir, "data.npz"))
    plotting.plot_data(solver.problem, obs["Z_data"], obs["u_obs"],
                       solver.sigma, fp(outdir, f"data{tag}"))
    res = np.abs(solver.residual_field(solver.final_points))
    plotting.plot_points(solver.problem, solver.final_points, truth,
                         fp(outdir, f"points{tag}"), abs_res=res)
    print(f"[figs] {outdir}: data/params/history/solution/source/points{tag}"
          f".{plotting.FIG_EXT}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="自适应跑的目录")
    ap.add_argument("--fixed", default=None, help="同预算均匀跑的目录（可选）")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args(argv)

    solver = load_run(a.run)
    outdir = a.outdir or a.run
    draw(solver, outdir)

    if a.fixed:
        fixed = load_run(a.fixed)
        draw(fixed, a.fixed)
        plotting.plot_points_compare(
            solver.problem,
            [solver.final_points, fixed.final_points],
            ["adaptive (residual-driven)", "uniform (same budget)"],
            solver.problem.truth_dict,
            plotting.fig_path(outdir, "points_compare"),
            abs_res_list=[np.abs(solver.residual_field(solver.final_points)),
                          np.abs(fixed.residual_field(fixed.final_points))])
        print(f"[figs] {outdir}: points_compare.{plotting.FIG_EXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
