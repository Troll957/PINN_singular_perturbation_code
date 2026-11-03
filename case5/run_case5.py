#!/usr/bin/env python
"""Case 5 主程序：二维奇异摄动方程的反源问题（默认：移动的层）。

示例
----
先跑自检（不训练）::

    python check_case5.py

快速冒烟（一分钟，只验证能跑通）::

    python run_case5.py --smoke

默认算例（moving：反演源的轨迹 A, s0, v）::

    python run_case5.py --outdir runs/moving

与同预算均匀配点对照::

    python run_case5.py --compare-fixed --outdir runs/cmp_moving

不知道源在哪时，先做多起点粗搜索::

    python run_case5.py --multistart 8 --outdir runs/ms

静止源（上一版算例，作对照）::

    python run_case5.py --case static --outdir runs/static
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from case5 import Case5Config, InversePINN, describe_device, pick_device, plotting
from case5 import run_fixed_baseline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="2D singularly perturbed inverse source problem")
    g = p.add_argument_group("case")
    g.add_argument("--case", choices=["moving", "static"], default="moving")
    g.add_argument("--eps", type=float, default=1e-3)
    g.add_argument("--b1", type=float, default=1.0)
    g.add_argument("--b2", type=float, default=0.0)
    g.add_argument("--q", type=float, default=1.0)
    g.add_argument("--T", type=float, default=1.0)

    g = p.add_argument_group("truth")
    g.add_argument("--true-A", type=float, default=None)
    g.add_argument("--true-s0", type=float, default=0.15)
    g.add_argument("--true-v", type=float, default=0.70)
    g.add_argument("--true-xs", type=float, default=0.30)
    g.add_argument("--true-ys", type=float, default=0.35)
    g.add_argument("--true-rho", type=float, default=0.08)

    g = p.add_argument_group("initial guess")
    g.add_argument("--init-A", type=float, default=1.5)
    g.add_argument("--init-s0", type=float, default=0.35)
    g.add_argument("--init-v", type=float, default=0.40)
    g.add_argument("--init-xs", type=float, default=0.35)
    g.add_argument("--init-ys", type=float, default=0.40)
    g.add_argument("--init-rho", type=float, default=0.10)
    g.add_argument("--freeze", type=str, default="",
                   help="逗号分隔的冻结参数，如 v 或 xs,ys")

    g = p.add_argument_group("collocation")
    g.add_argument("--n-int0", type=int, default=2048)
    g.add_argument("--n-ic0", type=int, default=128)
    g.add_argument("--n-bc0", type=int, default=64)
    g.add_argument("--rounds", type=int, default=8)
    g.add_argument("--epochs", type=int, default=3000)
    g.add_argument("--add", type=int, default=250)
    g.add_argument("--strategy", choices=["residual", "uniform", "fixed"],
                   default="residual")
    g.add_argument("--pool", type=int, default=20000)
    g.add_argument("--weighting", choices=["knn", "mean"], default="knn")
    g.add_argument("--hidden", type=str, default="64,64,64")
    g.add_argument("--activation", type=str, default="tanh")

    g = p.add_argument_group("data")
    g.add_argument("--sensors", type=int, default=None)
    g.add_argument("--times", type=int, default=None)
    g.add_argument("--noise", type=float, default=0.01)
    g.add_argument("--sensor-mode", choices=["grid", "random", "ridge", "slab"],
                   default=None)
    g.add_argument("--x-span", type=str, default="0.05,0.95")
    g.add_argument("--y-span", type=str, default="0.20,0.80")
    g.add_argument("--slab", type=str, default="0.10,0.50")
    g.add_argument("--wall-layer", action="store_true",
                   help="moving 算例在 y=0 处再加一条 sqrt(eps) 壁面层（正文 Case 3 的设定）")
    g.add_argument("--yc", type=float, default=0.5)
    g.add_argument("--rho-y", type=float, default=0.20)

    g = p.add_argument_group("loss / optim")
    g.add_argument("--w-data", type=float, default=1.0)
    g.add_argument("--no-pde", action="store_true",
                   help="去掉 PDE 残差项（只拟合数据），做消融")
    g.add_argument("--lr", type=float, default=2e-3)
    g.add_argument("--lr-param", type=float, default=5e-3)

    g = p.add_argument_group("run")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", type=str, default="auto")
    g.add_argument("--outdir", type=str, default=None)
    g.add_argument("--tag", type=str, default="case5")
    g.add_argument("--eval-n", type=int, default=81)
    g.add_argument("--eval-nt", type=int, default=41)
    g.add_argument("--multistart", type=int, default=0,
                   help="粗搜索的候选数（0=关闭）")
    g.add_argument("--probe-epochs", type=int, default=1000,
                   help="multistart 每个候选的训练步数")
    g.add_argument("--compare-fixed", action="store_true",
                   help="同时跑同预算的均匀配点对照")
    g.add_argument("--smoke", action="store_true", help="极小规模冒烟测试")
    g.add_argument("--quiet", action="store_true")
    return p


def cfg_from_args(a) -> Case5Config:
    hidden = tuple(int(s) for s in a.hidden.split(",") if s.strip())
    slab = tuple(float(s) for s in a.slab.split(","))
    x_span = tuple(float(s) for s in a.x_span.split(","))
    y_span = tuple(float(s) for s in a.y_span.split(","))

    # 两种算例的默认数据量不同：moving 用"少量传感器 × 密集时间"
    if a.sensors is None:
        n_sensors = 12 if a.case == "moving" else 100
    else:
        n_sensors = a.sensors
    if a.times is None:
        n_times = 16 if a.case == "moving" else 5
    else:
        n_times = a.times
    if a.sensor_mode is None:
        sensor_mode = "grid" if a.case == "moving" else "slab"
    else:
        sensor_mode = a.sensor_mode
    true_A = 2.0 if a.true_A is None else a.true_A

    cfg = Case5Config(
        case=a.case,
        eps=a.eps, b1=a.b1, b2=a.b2, q=a.q, T=a.T,
        true_A=true_A, true_s0=a.true_s0, true_v=a.true_v,
        true_xs=a.true_xs, true_ys=a.true_ys, true_rho=a.true_rho,
        init_A=a.init_A, init_s0=a.init_s0, init_v=a.init_v,
        init_xs=a.init_xs, init_ys=a.init_ys, init_rho=a.init_rho,
        freeze=a.freeze,
        hidden=hidden, activation=a.activation,
        n_int0=a.n_int0, n_ic0=a.n_ic0, n_bc0=a.n_bc0,
        rounds=a.rounds, epochs_per_round=a.epochs, add_per_round=a.add,
        strategy=a.strategy, candidate_pool=a.pool, weighting=a.weighting,
        n_sensors=n_sensors, n_times=n_times, noise=a.noise,
        sensor_mode=sensor_mode, slab=slab, x_span=x_span, y_span=y_span,
        wall_layer=a.wall_layer, yc=a.yc, rho_y=a.rho_y,
        w_data=a.w_data, use_pde=not a.no_pde,
        lr=a.lr, lr_param=a.lr_param,
        seed=a.seed, device=a.device, verbose=not a.quiet,
        eval_n=a.eval_n, eval_nt=a.eval_nt, tag=a.tag,
    )
    if a.smoke:
        cfg.n_int0 = 512
        cfg.n_ic0 = 32
        cfg.n_bc0 = 16
        cfg.rounds = 2
        cfg.epochs_per_round = 40
        cfg.add_per_round = 64
        cfg.candidate_pool = 4000
        cfg.add_ic_per_round = 4
        cfg.add_bc_per_round = 4
        cfg.eval_n = 31
        cfg.eval_nt = 11
    return cfg


def multistart_init(cfg: Case5Config, n_cand: int, probe_epochs: int):
    """在低维初值网格上做一次便宜的多起点搜索，返回残差最小的初值。

    动机：源参数与网络是**联合**优化的，残差对源的位置极其敏感
    （薄层宽度只有 :math:`O(\\sqrt\\varepsilon)`），从很远的初值出发容易
    落进"源的位置错了、但网络把数据拟合上了"的伪极小。
    """
    if cfg.case == "moving":
        g1 = np.linspace(0.05, 0.45, max(2, n_cand // 2))      # s0
        g2 = np.linspace(0.40, 1.00, 2)                        # v
        keys = ("s0", "v")
    else:
        g1 = np.linspace(0.15, 0.55, max(2, n_cand // 2))      # xs
        g2 = np.linspace(0.25, 0.65, 2)                        # ys
        keys = ("xs", "ys")

    best = (float("inf"), None)
    i, total = 0, len(g1) * len(g2)
    for a1 in g1:
        for a2 in g2:
            i += 1
            c = copy.deepcopy(cfg)
            init = c.init_dict()
            init[keys[0]] = float(a1)
            init[keys[1]] = float(a2)
            c.init_A = init["A"]
            c.init_s0 = init.get("s0", c.init_s0)
            c.init_v = init.get("v", c.init_v)
            c.init_xs = init.get("xs", c.init_xs)
            c.init_ys = init.get("ys", c.init_ys)
            c.init_rho = init.get("rho", c.init_rho)
            c.rounds = 0
            c.epochs_per_round = probe_epochs
            c.strategy = "fixed"
            c.verbose = False
            s = InversePINN(c)
            s.run()
            # 用 **PDE 项** 排序：数据项在各候选之间几乎一样，
            # 真正区分"源的位置对不对"的是残差（实测好/坏相差几十倍）。
            val = float(s.history[-1].get("loss_pde", float("inf")))
            print(f"    [multistart {i}/{total}] {keys[0]}={a1:.3f} "
                  f"{keys[1]}={a2:.3f} -> J_pde={val:.4e}")
            if np.isfinite(val) and val < best[0]:
                best = (val, dict(init))
    print(f"    [multistart] 选中 {best[1]} (J_pde={best[0]:.4e})")
    return best[1]


def report(solver: InversePINN, name: str) -> dict:
    h = solver.history[-1]
    names = solver.param_names
    p_hat = {k: h[f"param_{k}"] for k in names}
    print(f"\n== {name} ==")
    print(f"  N = {h['N']}   rounds = {solver.cfg.rounds + 1}   "
          f"epochs/round = {solver.cfg.epochs_per_round}")
    print("  params: " + "  ".join(f"{k}={p_hat[k]:.4f}" for k in names))
    print("  truth : " + "  ".join(f"{k}={solver.problem.truth_dict[k]:.4f}"
                                   for k in names))
    print("  param rel err: " + ", ".join(
        f"{k}={h['param_rel_' + k]:.2%}" for k in names))
    print(f"  max rel err = {h['param_err_max']:.2%}   "
          f"source rel L2 = {h['src_rel_l2']:.2%}")
    print(f"  solution: max_err={h['max_err']:.3e}  rel_l2={h['rel_l2']:.3e}")
    print(f"  layer fraction of collocation points = {h['frac_layer']:.3f}")
    if h.get("grad_share_layer"):
        print("  gradient share from layer: " + ", ".join(
            f"{k}={v:.2%}" for k, v in h["grad_share_layer"].items()))
    return {"name": name, **{f"final_{k}": v for k, v in h.items()
                             if isinstance(v, (int, float))},
            "estimate": p_hat, "truth": solver.problem.truth_dict}


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    cfg = cfg_from_args(a)
    outdir = a.outdir or os.path.join(
        "runs", cfg.tag,
        f"{cfg.case}_eps{cfg.eps:g}_{cfg.strategy}_noise{cfg.noise:g}_seed{cfg.seed}")
    os.makedirs(outdir, exist_ok=True)

    dev = pick_device(cfg.device)
    print(f"[case5] device = {describe_device(dev)}")
    print(f"[case5] case={cfg.case}  eps={cfg.eps:g}  noise={cfg.noise:g}  "
          f"strategy={cfg.strategy}  outdir={outdir}")

    if a.multistart > 0:
        init = multistart_init(cfg, a.multistart, a.probe_epochs)
        cfg.init_A = init.get("A", cfg.init_A)
        cfg.init_s0 = init.get("s0", cfg.init_s0)
        cfg.init_v = init.get("v", cfg.init_v)
        cfg.init_xs = init.get("xs", cfg.init_xs)
        cfg.init_ys = init.get("ys", cfg.init_ys)
        cfg.init_rho = init.get("rho", cfg.init_rho)

    solver = InversePINN(cfg)
    print(f"[case5] interior={solver.points.n_interior}  ic={solver.points.n_ic}  "
          f"bc={solver.points.n_bc}  data={solver.Z_data.shape[0]}  "
          f"sigma={solver.sigma:.3e}")
    print(f"[case5] 参数 = {solver.param_names}  初值 = "
          + str({k: round(v, 4) for k, v in solver.params.numpy_values().items()}))
    solver.run()
    solver.save(outdir)
    rows = [report(solver, cfg.strategy)]

    truth = solver.problem.truth_dict
    names = solver.param_names
    p_hat = solver.params.numpy_values()
    fp = plotting.fig_path
    plotting.plot_params(solver.history, truth, fp(outdir, "params"), names=names)
    plotting.plot_history(solver.history, fp(outdir, "history"))
    cut_y = 0.5 if cfg.case == "moving" else float(truth.get("ys", 0.5))
    plotting.plot_solution(solver.problem, solver.predict, truth,
                           fp(outdir, "solution"), cut_y=cut_y)
    plotting.plot_source(solver.problem, truth, p_hat, fp(outdir, "source"))
    plotting.plot_data(solver.problem, solver.Z_data, solver.u_obs, solver.sigma,
                       fp(outdir, "data"))
    abs_res = np.abs(solver.residual_field(solver.final_points))
    plotting.plot_points(solver.problem, solver.final_points, truth,
                         fp(outdir, "points"), abs_res=abs_res)

    if a.compare_fixed:
        base = run_fixed_baseline(cfg)
        bdir = os.path.join(outdir, "fixed_baseline")
        base.save(bdir)
        rows.append(report(base, "fixed (uniform, same budget)"))
        plotting.plot_solution(base.problem, base.predict, truth,
                               fp(bdir, "solution"), cut_y=cut_y)
        plotting.plot_points(base.problem, base.final_points, truth,
                             fp(bdir, "points"))
        plotting.plot_points_compare(
            solver.problem,
            [solver.final_points, base.final_points],
            ["adaptive (residual-driven)", "uniform (same budget)"],
            truth, fp(outdir, "points_compare"),
            abs_res_list=[np.abs(solver.residual_field(solver.final_points)),
                          np.abs(base.residual_field(base.final_points))])
        with open(os.path.join(outdir, "compare.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)

        print("\n== comparison ==")
        print(f"  {'run':28s} {'param_err_max':>14s} {'src_rel_l2':>12s} "
              f"{'u_max_err':>12s}")
        for r in rows:
            print(f"  {r['name']:28s} {r['final_param_err_max']:14.2%} "
                  f"{r['final_src_rel_l2']:12.2%} {r['final_max_err']:12.3e}")

    print(f"\n[case5] 输出目录: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
