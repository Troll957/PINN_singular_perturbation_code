#!/usr/bin/env python3
"""为 Case 2 / Case 3 生成论文用图: 5 面板 (解, 预测, 误差, 点分布, 层内点)。

用法:
  fig_cases.py --case case2 --eps 1e-5 --times 0.25
  fig_cases.py --case case3 --eps 1e-6 --times 0.0,0.4,0.6,1.0
"""
from __future__ import annotations
import argparse, math, os, time
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from spp_pinn.nd import AdaptivePINNND, NDConfig
from spp_pinn import pick_device, describe_device
from spp_pinn import plotting as PLT


def _decade_ticks(cb, vmin, vmax):
    """把误差色条的刻度设为 vmin..vmax 内的十进位, 避免 contourf+LogNorm 下的乱码刻度。"""
    lo = math.ceil(math.log10(vmin))
    hi = math.floor(math.log10(vmax))
    tks = [10.0 ** k for k in range(lo, hi + 1)]
    if tks:
        cb.set_ticks(tks)
        cb.ax.set_yticklabels([r"$10^{%d}$" % k for k in range(lo, hi + 1)], fontsize=8)
        cb.ax.minorticks_off()


def grid(n, t):
    g = np.linspace(0.0, 1.0, n)
    X1, X2 = np.meshgrid(g, g, indexing="ij")
    Z = np.column_stack([X1.ravel(), X2.ravel(), np.full(n * n, t)])
    return X1, X2, Z


def make_figure(solver, eps, t, n, outpath, tag):
    prob, thr = solver.problem, 4.0 * math.sqrt(eps)
    # 本算例的精确解含因子 cos(pi t)。若 t 使该因子为零, 精确解恒为 0,
    # 此时解对比面板没有意义 —— 给出警告并在标题中标注。
    zero_slice = abs(math.cos(math.pi * t)) < 1e-8
    if zero_slice:
        print("  [警告] t=%g 处 cos(pi t)=0, 精确解恒为零, 该切片不携带信息" % t,
              flush=True)
    X1, X2, Z = grid(n, t)
    ue = np.asarray(prob.exact(Z), float).reshape(n, n)
    up = solver.predict(Z).reshape(n, n)
    err = np.abs(up - ue)
    pts = solver.points.interior

    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.6))
    vmin, vmax = float(min(ue.min(), up.min())), float(max(ue.max(), up.max()))
    lv = np.linspace(vmin, vmax, 25)

    for ax, d, ttl in ((axs[0, 0], ue, "Exact solution"),
                       (axs[0, 1], up, "PINN prediction")):
        cf = ax.contourf(X1, X2, d, levels=lv, cmap="viridis")
        ax.set_title(ttl, fontsize=11)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)

    # 误差: 对数分级的填充等值线 + 等值线描边
    # 色条范围严格取数据的实际范围: 不向上取整到 10 的整数幂,
    # 也不加 extend 三角 —— 否则会出现"有数据落在更高量级"的假象。
    emax = float(err.max())
    emin = max(float(err.min()), emax * 1e-6)
    lev = np.logspace(math.log10(emin), math.log10(emax), 24)
    cf = axs[0, 2].contourf(X1, X2, np.maximum(err, emin), levels=lev,
                            cmap="RdPu", norm=LogNorm(vmin=emin, vmax=emax),
                            extend="neither")
    axs[0, 2].contour(X1, X2, np.maximum(err, emin), levels=lev[::4],
                      colors="k", linewidths=0.3, alpha=0.45)
    axs[0, 2].set_title("|u_pred - u_exact|  (rel L2 = %.2e,  max = %.2e)"
                        % (solver.history[-1]["rel_l2"], emax), fontsize=9)
    cb = fig.colorbar(cf, ax=axs[0, 2], fraction=0.046, pad=0.03)
    cb.set_label("abs error", fontsize=9)
    _decade_ticks(cb, emin, emax)

    # 点分布 (x1,x2), 对 t 边缘化
    h = axs[1, 0].hist2d(pts[:, 0], pts[:, 1], bins=100, cmap="cividis",
                         norm=LogNorm())
    fig.colorbar(h[3], ax=axs[1, 0], fraction=0.046, pad=0.03)
    axs[1, 0].set_title("collocation points, marginal in (x1, x2)", fontsize=11)

    # 层内点: 用问题自带的判据 (静止层 x1<=c*sqrt(eps); 移动层 |x1-t|<=c*sqrt(eps))
    w = 20.0 * math.sqrt(eps)
    m = np.asarray(prob.inside_layer(pts, 20.0), dtype=bool)
    moving = hasattr(prob, "layer_center")
    ttl = ("layer points (%d of %d), projection (x1, t)"
           if moving else "points with x1 <= 20 sqrt(eps)  (%d of %d)")
    axs[1, 1].scatter(pts[m, 0], pts[m, 2] if moving else pts[m, 1],
                      s=2.0, c="crimson", alpha=0.55)
    axs[1, 1].set_title(ttl % (m.sum(), len(pts)), fontsize=11)
    axs[1, 1].set_ylabel(r"$t$" if moving else r"$x_2$", fontsize=9)

    axs[1, 2].scatter(pts[m, 0], pts[m, 1] if moving else pts[m, 2],
                      s=2.0, c="steelblue", alpha=0.55)
    axs[1, 2].set_title("same points, projection (x1, %s)" % ("x2" if moving else "t"),
                        fontsize=11)
    axs[1, 2].set_ylabel(r"$x_2$" if moving else r"$t$", fontsize=9)
    if not moving:
        axs[1, 1].set_xlim(0, w); axs[1, 2].set_xlim(0, w)

    for ax in axs.ravel():
        ax.set_xlabel(r"$x_1$", fontsize=9)
        ax.set_ylabel(r"$x_2$" if ax is not axs[1, 2] else r"$t$", fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in (axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0]):
        ax.set_aspect("equal")
    supt = "%s,  eps=%g,  t=%g" % (tag, eps, t)
    if zero_slice:
        supt += "   [cos(pi t)=0: exact solution vanishes]"
    fig.suptitle(supt, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = PLT._save(fig, outpath)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="case2", choices=["case2", "case3"])
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--times", default="0.25")
    ap.add_argument("--n", type=int, default=241)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--n-int0", type=int, default=4096)
    ap.add_argument("--add", type=int, default=500)
    ap.add_argument("--eval-n", type=int, default=61)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="case_figs")
    ap.add_argument("--replot", action="store_true",
                    help="不训练, 从 <outdir>/<case>_eps<eps>.pt 直接重画")
    a = ap.parse_args()
    torch.set_num_threads(4)
    os.makedirs(a.outdir, exist_ok=True)
    prob = "corner2d" if a.case == "case2" else "moving"
    tag = "Case 2 corner layer" if a.case == "case2" else "Case 3 moving layer"
    times = [float(v) for v in a.times.split(",")]
    print(f"device={describe_device(pick_device(a.device))}  {prob}  eps={a.eps:g}  "
          f"times={times}", flush=True)
    cfg = NDConfig(problem=prob, eps=a.eps, T=1.0, hidden=(64, 64, 64),
                   activation="tanh", n_int0=a.n_int0, n_ic0=128, n_bc0=64,
                   init_method="halton", rounds=a.rounds,
                   epochs_per_round=a.epochs, add_per_round=a.add,
                   strategy="residual", candidate_pool=20000, beta=1.0,
                   weighting="knn", knn_neighbors=4, lr=2e-3,
                   lr_schedule="cosine", lr_gamma=0.9, weight_mode="init",
                   seed=0, device=a.device, verbose=False, eval_n=a.eval_n)
    ckpt = os.path.join(a.outdir, f"{a.case}_eps{a.eps:g}.pt")
    if a.replot:
        if not os.path.exists(ckpt):
            raise SystemExit(f"找不到检查点 {ckpt}; 先跑一次训练")
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        s = AdaptivePINNND(NDConfig(**d["cfg"]))
        s.model.load_state_dict(d["model"])
        s.points.interior = np.asarray(d["points"])
        s.history = d["hist"]
        print(f"[replot] 从 {ckpt} 载入模型与点集, 不重新训练", flush=True)
        last = dict(s.history[-1])
    else:
        t0 = time.time()
        s = AdaptivePINNND(cfg); s.run()
        last = dict(s.history[-1])
    if not a.replot:
        print(f"[训练完成] N={last['N']} rel_L2={last['rel_l2']:.3e} "
              f"max={last['max_err']:.3e} layer={last['frac_layer']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    if not a.replot:
        torch.save({"model": s.model.state_dict(), "cfg": cfg.to_dict(),
                    "hist": s.history, "points": s.points.interior,
                    "ic": s.points.ic, "bc": s.points.bc}, ckpt)
    for t in times:
        p = make_figure(s, a.eps, t, a.n,
                        os.path.join(a.outdir, f"{a.case}_eps{a.eps:g}_t{t:g}.pdf"), tag)
        print("  ->", p, flush=True)


if __name__ == "__main__":
    main()
