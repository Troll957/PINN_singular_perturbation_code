"""二维空间 + 时间的自适应配点 PINN 演示（角点层算例）。

这是把 `demo.py`（1D+时间）推广到高维的对照脚本：求解器的实现全部在
`spp_pinn/nd.py` 里，本脚本只负责命令行参数、进度显示与出图。

用法::

    arch -arm64 .venv/bin/python demo_2d.py --eps 1e-3
    arch -arm64 .venv/bin/python demo_2d.py --eps 1e-3 --rounds 8 --epochs 800 \
        --weighting knn --strategy residual
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dsh_mpl")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spp_pinn import plotting as PLT  # noqa: E402
from spp_pinn.nd import (AdaptivePINNND, NDConfig,  # noqa: E402
                         run_reference_nd, pick_device, describe_device)


def plot_2d_result(solver, outdir: str, tag: str) -> None:
    """角点层结果：x-y 平面上的配点、解与误差切片（t = T/2）。"""
    n = 121
    xs = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    t_slice = solver.cfg.T / 4.0          # 注意不能取 T/2: cos(pi*T/2)=0 会得到平凡零场
    Z = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, t_slice)])
    up = solver.predict(Z).reshape(n, n)
    ue = np.asarray(solver.problem.exact(Z), dtype=float).reshape(n, n)
    err = np.abs(up - ue)

    fig, axs = plt.subplots(1, 4, figsize=(20, 4.4))
    P = solver.final_points
    sc = axs[0].scatter(P[:, 0], P[:, 1], s=5, c=np.log10(solver.final_abs_res + 1e-16),
                        cmap="inferno", linewidths=0)
    axs[0].set_title(f"Collocation points ($N={len(P)}$)")
    axs[0].set_xlabel("x"); axs[0].set_ylabel("y")
    fig.colorbar(sc, ax=axs[0], label=r"$\log_{10}|r|$")

    for ax, F, ttl, cm in ((axs[1], ue, "Exact $u^*(x,y,T/4)$", "viridis"),
                           (axs[2], up, "PINN prediction", "viridis"),
                           (axs[3], np.log10(err + 1e-16), "log10 |error|", "inferno")):
        im = ax.pcolormesh(X, Y, F, cmap=cm, shading="auto")
        im.set_rasterized(True)      # 1.5 万个四边形矢量化会让 PDF 变大
        ax.set_title(ttl); ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"2D corner layer, eps={solver.cfg.eps:g}", y=1.02)
    fig.tight_layout()
    PLT.save_figure(fig, os.path.join(outdir, f"{tag}_fields"))
    plt.close(fig)

    # 收敛曲线
    h = solver.history
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.1))
    axs[0].semilogy([q["N"] for q in h], [q["max_err"] for q in h], "o-")
    axs[0].set_xlabel("points $N$"); axs[0].set_ylabel(r"$\|u-u^*\|_\infty$")
    axs[0].set_title("Error vs. budget"); axs[0].grid(alpha=0.3, which="both")
    axs[1].plot([q["round"] for q in h], [q["frac_layer"] for q in h], "o-", label="x<=4$\\sqrt{\\varepsilon}$")
    axs[1].plot([q["round"] for q in h], [q["frac_corner"] for q in h], "s-", label="all coords")
    axs[1].set_xlabel("round"); axs[1].set_ylabel("fraction"); axs[1].legend()
    axs[1].set_title("Points captured by the layer"); axs[1].grid(alpha=0.3)
    axs[2].semilogy([q["round"] for q in h], [q["rms_res_uniform"] for q in h], "o-")
    axs[2].set_xlabel("round"); axs[2].set_ylabel("RMS residual (uniform test set)")
    axs[2].set_title("Residual"); axs[2].grid(alpha=0.3, which="both")
    fig.tight_layout()
    PLT.save_figure(fig, os.path.join(outdir, f"{tag}_history"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="2D+时间 自适应配点 PINN")
    ap.add_argument("--eps", type=float, default=1e-6) #epsilon的取值，这里默认是1e-3
    ap.add_argument("--rounds", type=int, default=20) #迭代轮数，此处默认是8
    ap.add_argument("--epochs", type=int, default=800) #每轮训练的epoch数，此处默认是800
    ap.add_argument("--n-int0", type=int, default=1024) #初始采样点选取数量，此处是1024
    ap.add_argument("--add", type=int, default=150)
    ap.add_argument("--strategy", default="residual",
                    choices=["residual", "distribution", "uniform", "fixed"])
    ap.add_argument("--weighting", default="knn", choices=["knn", "mean"])
    ap.add_argument("--knn-neighbors", type=int, default=4)
    ap.add_argument("--hidden", default="64,64,64")
    ap.add_argument("--eval-n", type=int, default=61)
    ap.add_argument("--device", default="auto",
                    help="auto | cuda | cuda:0 | mps | cpu (默认 auto)")
    ap.add_argument("--outdir", default="demo_2d_out")
    ap.add_argument("--reference", action="store_true",
                    help="同时跑一个同点数的固定均匀参照组")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    os.makedirs(args.outdir, exist_ok=True)
    cfg = NDConfig(eps=args.eps, rounds=args.rounds, epochs_per_round=args.epochs,
                   n_int0=args.n_int0, add_per_round=args.add,
                   strategy=args.strategy, weighting=args.weighting,
                   knn_neighbors=args.knn_neighbors,
                   hidden=tuple(int(h) for h in args.hidden.split(",")),
                   eval_n=args.eval_n, device=args.device, verbose=False)

    print(f"算例 {cfg.problem} (d=2) | eps={args.eps:g} | sqrt(eps)={np.sqrt(args.eps):.3e} | "
          f"策略={cfg.strategy}/{cfg.weighting} | "
          f"设备={describe_device(pick_device(args.device))}")
    print("-" * 104)
    print(f"{'轮次':>4} {'内部点':>7} {'总点数':>7} {'max误差':>11} {'rel L2':>11} "
          f"{'均匀残差':>11} {'x层占比':>9} {'角点占比':>9} {'耗时':>7}")
    print("-" * 104)

    def on_round(r):
        print(f"{r['round']:>4d} {r['n_int']:>7d} {r['N']:>7d} {r['max_err']:>11.3e} "
              f"{r['rel_l2']:>11.3e} {r['rms_res_uniform']:>11.3e} "
              f"{r['frac_layer']:>9.3f} {r['frac_corner']:>9.3f} {r['wall']:>6.1f}s",
              flush=True)

    cfg.callback = on_round
    t0 = time.time()
    solver = AdaptivePINNND(cfg)
    solver.run()
    tag = f"corner2d_eps{args.eps:g}_{args.strategy}_{args.weighting}"
    print("-" * 104)
    print(f"总耗时 {time.time()-t0:.1f}s")
    plot_2d_result(solver, args.outdir, tag)

    if args.reference:
        c = NDConfig(**{**cfg.to_dict(), "n_int0": args.n_int0 + args.rounds * args.add,
                        "strategy": "fixed", "rounds": 0,
                        "epochs_per_round": (args.rounds + 1) * args.epochs,
                        "callback": None, "verbose": False})
        ref = run_reference_nd(c, n_interior=args.n_int0 + args.rounds * args.add,
                               total_epochs=(args.rounds + 1) * args.epochs)
        ev = ref.evaluate()
        print(f"参照组(固定均匀, 同点数): max_err={ev['max_err']:.3e} rel_l2={ev['rel_l2']:.3e}")
    print(f"图像已保存到 {args.outdir}/")


if __name__ == "__main__":
    main()
