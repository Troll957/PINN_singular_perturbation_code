"""单算例演示：残差驱动的自适应配点 PINN 求解抛物型奇异摄动问题。

用法::

    arch -arm64 .venv/bin/python demo.py --problem left --eps 1e-5
    arch -arm64 .venv/bin/python demo.py --problem interior --eps 1e-4 --rounds 8

输出：终端逐轮进度表 + ``demo_out/`` 下的解、误差、配点分布与收敛图。
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dsh_mpl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from spp_pinn import (AdaptivePINNSolver, TrainConfig,  # noqa: E402
                      describe_device, pick_device)
from spp_pinn import plotting as P  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="残差驱动自适应配点 PINN 演示")
    ap.add_argument("--problem", default="left", choices=["left", "right", "interior"])
    ap.add_argument("--eps", type=float, default=1e-6, help="奇异摄动参数")
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=20, help="自适应增点轮数")
    ap.add_argument("--epochs", type=int, default=1000, help="每轮训练步数")
    ap.add_argument("--n-int0", type=int, default=1024, help="初始内部配点数")
    ap.add_argument("--add", type=int, default=150, help="每轮新增内部配点数")
    ap.add_argument("--strategy", default="residual",
                    choices=["residual", "distribution", "uniform", "fixed"])
    ap.add_argument("--weighting", default="voronoi", choices=["voronoi", "mean"])
    ap.add_argument("--candidate-pool", type=int, default=20000)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--min-dist", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hard-constraint", action="store_true")
    ap.add_argument("--hidden", default="64,64,64")
    ap.add_argument("--device", default="auto",
                    help="auto | cuda | cuda:0 | mps | cpu (默认 auto: CUDA>MPS>CPU)")
    ap.add_argument("--outdir", default="demo_out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    os.makedirs(args.outdir, exist_ok=True)

    cfg = TrainConfig(
        problem=args.problem, eps=args.eps, T=args.T,
        hidden=tuple(int(h) for h in args.hidden.split(",")),
        n_int0=args.n_int0, n_ic0=64, n_bc0=64, init_method="halton",
        rounds=args.rounds, epochs_per_round=args.epochs,
        add_per_round=args.add, strategy=args.strategy, weighting=args.weighting,
        candidate_pool=args.candidate_pool, beta=args.beta, min_dist=args.min_dist,
        lr=args.lr, lr_schedule="cosine", lr_gamma=0.9,
        hard_constraint=args.hard_constraint,
        seed=args.seed, device=args.device, verbose=False,
    )

    print(f"设备 {describe_device(pick_device(args.device))} | 算例 {args.problem} | eps={args.eps:g} | "
          f"sqrt(eps)={np.sqrt(args.eps):.3e} | 策略={args.strategy}/{args.weighting}")
    print("-" * 100)
    print(f"{'轮次':>4} {'N':>6} {'内部点':>7} {'新增':>5} {'max误差':>11} {'rel L2':>11} "
          f"{'均匀残差':>11} {'层内占比':>9} {'耗时':>7}")
    print("-" * 100)

    def on_round(rec):
        print(f"{rec['round']:>4d} {rec['N']:>6d} {rec['n_int']:>7d} {rec['added']:>5d} "
              f"{rec['max_err']:>11.3e} {rec['rel_l2']:>11.3e} "
              f"{rec['rms_res_uniform']:>11.3e} {rec['frac_layer']:>9.3f} "
              f"{rec['wall']:>6.1f}s", flush=True)

    cfg.callback = on_round
    t0 = time.time()
    solver = AdaptivePINNSolver(cfg)
    solver.run()
    print("-" * 100)
    print(f"总耗时 {time.time()-t0:.1f}s")

    ev = solver.evaluate()
    tag = f"{args.problem}_eps{args.eps:g}"
    P.plot_solution_fields(ev, args.eps, solver.problem.layer,
                           os.path.join(args.outdir, f"{tag}_fields.png"))
    P.plot_cross_sections(ev, args.eps, os.path.join(args.outdir, f"{tag}_sections.png"))
    P.plot_round_history(solver.history, os.path.join(args.outdir, f"{tag}_history.png"),
                         eps=args.eps)
    P.plot_point_cloud(solver.final_points, solver.final_abs_res,
                       solver.final_ic, solver.final_bc, args.eps, args.T,
                       os.path.join(args.outdir, f"{tag}_points.png"),
                       title=f"{args.problem}, eps={args.eps:g}, N={solver.points.total}")
    print(f"max err = {ev['max_err']:.4e}, relative L2 = {ev['rel_l2']:.4e}")
    print(f"配点: 内部 {solver.points.n_interior}, 初值 {solver.points.n_ic}, "
          f"边值 {solver.points.n_bc}")
    print(f"图像已保存到 {args.outdir}/")


if __name__ == "__main__":
    main()
