"""作图（论文用）。坐标轴标签用英文，避免服务器缺中文字体。"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .problem import PARAM_LABELS

# 论文里插图统一用 PDF（矢量、可缩放）；需要位图时设 CASE5_FIG_EXT=png
FIG_EXT = os.environ.get("CASE5_FIG_EXT", "pdf")


def fig_path(outdir: str, name: str) -> str:
    """``<outdir>/<name>.<FIG_EXT>``。"""
    return os.path.join(outdir, f"{name}.{FIG_EXT}")


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# 1. 参数收敛
# ----------------------------------------------------------------------
def plot_params(history: List[Dict], truth: Dict[str, float], path: str,
                names=None) -> None:
    names = tuple(names or [k for k in truth])
    rounds = [h["round"] for h in history]
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.0),
                             squeeze=False)
    for ax, k in zip(axes[0], names):
        vals = [h[f"param_{k}"] for h in history]
        ax.plot(rounds, vals, "o-", ms=4, lw=1.4, label="estimate")
        ax.axhline(truth[k], color="k", ls="--", lw=1.2, label="truth")
        ax.set_xlabel("round")
        ax.set_ylabel(PARAM_LABELS.get(k, k))
        ax.set_title(k)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, path)


# ----------------------------------------------------------------------
# 2. 解的对比
# ----------------------------------------------------------------------
def plot_solution(problem, predict, truth: Dict[str, float], path: str,
                  t_slice: Optional[float] = None, cut_y: Optional[float] = None,
                  n: int = 241) -> None:
    """固定时刻的 (x,y) 平面：真解 / 预测 / 误差，外加一条 y=cut_y 的切线。"""
    t0 = 0.25 * problem.T if t_slice is None else float(t_slice)
    y0 = 0.5 if cut_y is None else float(cut_y)
    ax1 = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(ax1, ax1, indexing="ij")
    Z = np.stack([X.ravel(), Y.ravel(), np.full(X.size, t0)], axis=1)
    u_true = np.asarray(problem.exact(Z), dtype=float).reshape(n, n)
    u_pred = predict(Z).reshape(n, n)
    err = np.abs(u_pred - u_true)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, field, title in zip(axes[:3],
                                [u_true, u_pred, err],
                                ["exact $u^*$", "prediction $u_\\theta$",
                                 r"$|u_\theta-u^*|$"]):
        im = ax.imshow(field.T, origin="lower", extent=(0, 1, 0, 1),
                       aspect="auto",
                       cmap="viridis" if title != r"$|u_\theta-u^*|$" else "magma")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046)

    xs = np.linspace(0.0, 1.0, 601)
    Zs = np.stack([xs, np.full_like(xs, y0), np.full_like(xs, t0)], axis=1)
    us = np.asarray(problem.exact(Zs), dtype=float).ravel()
    up = predict(Zs)
    axes[3].plot(xs, us, "k-", lw=2, label="exact")
    axes[3].plot(xs, up, "r--", lw=1.5, label="PINN")
    if hasattr(problem, "trajectory"):
        xc = float(problem.trajectory(np.array([t0]))[0])
    else:
        xc = float(truth.get("xs", np.nan))
    axes[3].axvline(xc, color="0.6", ls=":", lw=1)
    axes[3].set_title(f"cut at $y={y0:g}$, t={t0:g}")
    axes[3].set_xlabel("x")
    axes[3].legend(frameon=False, fontsize=8)
    fig.suptitle(f"solution at t = {t0:g}", y=1.02)
    fig.tight_layout()
    _save(fig, path)


# ----------------------------------------------------------------------
# 3. 源项的对比
# ----------------------------------------------------------------------
def plot_source(problem, truth: Dict[str, float], p_hat: Dict[str, float],
                path: str, t_slice: Optional[float] = None, n: int = 241) -> None:
    t0 = 0.0 if t_slice is None else float(t_slice)
    ax1 = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(ax1, ax1, indexing="ij")
    Z = np.stack([X.ravel(), Y.ravel(), np.full(X.size, t0)], axis=1)
    f_true = np.asarray(problem.forcing(Z, truth), dtype=float).reshape(n, n)
    f_hat = np.asarray(problem.forcing(Z, p_hat), dtype=float).reshape(n, n)
    err = np.abs(f_hat - f_true)
    vmax = float(np.abs(f_true).max())

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, field, title, cmap in zip(
            axes,
            [f_true, f_hat, err],
            ["true source $f^*$", "recovered $f_{\\hat\\phi}$",
             r"$|f_{\hat\phi}-f^*|$"],
            ["RdBu_r", "RdBu_r", "magma"]):
        v = vmax if cmap == "RdBu_r" else None
        im = ax.imshow(field.T, origin="lower", extent=(0, 1, 0, 1),
                       aspect="auto", cmap=cmap, vmin=-v if v else None,
                       vmax=v if v else None)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"source at t = {t0:g}", y=1.02)
    fig.tight_layout()
    _save(fig, path)


# ----------------------------------------------------------------------
# 4. 配点分布（叠加真值轨迹）
# ----------------------------------------------------------------------
def plot_points(problem, points: np.ndarray, truth: Dict[str, float],
                path: str, abs_res: Optional[np.ndarray] = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    s = 4
    c = None if abs_res is None else np.log10(np.abs(abs_res) + 1e-12)
    tt = np.linspace(0.0, problem.T, 200)
    curve = problem.layer_curve(tt)
    half = 4.0 * problem.sqrt_eps

    sc = axes[0].scatter(points[:, 2], points[:, 0], c=c, s=s, cmap="viridis")
    axes[0].plot(tt, curve, "r--", lw=1.4, label="true $s(t)$")
    axes[0].fill_between(tt, curve - half, curve + half, color="r", alpha=0.10)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    axes[0].set_title("projection $(t,x)$")
    axes[0].legend(frameon=False, fontsize=8)
    if c is not None:
        fig.colorbar(sc, ax=axes[0], fraction=0.046, label=r"$\log_{10}|r|$")

    axes[1].scatter(points[:, 0], points[:, 1], c=c, s=s, cmap="viridis")
    axes[1].plot(curve, np.full_like(curve, 0.5), "r--", lw=1.0, alpha=0.0)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("projection $(x,y)$")
    fig.tight_layout()
    _save(fig, path)


# ----------------------------------------------------------------------
# 5. 训练曲线
# ----------------------------------------------------------------------
def plot_history(history: List[Dict], path: str) -> None:
    rounds = [h["round"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    axes[0].semilogy(rounds, [h["param_err_max"] for h in history], "o-", ms=4)
    axes[0].set_title("max relative parameter error")
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("rel. err")

    axes[1].semilogy(rounds, [h["src_rel_l2"] for h in history], "s-", ms=4)
    axes[1].set_title(r"source error $\|f_{\hat\phi}-f^*\|/\|f^*\|$")
    axes[1].set_xlabel("round")

    axes[2].semilogy(rounds, [h["loss_pde"] for h in history], "^-", ms=4,
                     label=r"$J_{pde}$")
    axes[2].semilogy(rounds, [h["loss_data"] for h in history], "v-", ms=4,
                     label=r"$J_{data}$")
    axes[2].set_title("losses")
    axes[2].set_xlabel("round")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, path)


# ----------------------------------------------------------------------
# 6. 扫描汇总图
# ----------------------------------------------------------------------
def plot_sweep(rows: List[Dict], xkey: str, path: str,
               xlabel: Optional[str] = None) -> None:
    groups = sorted({r["strategy"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for g in groups:
        rs = sorted([r for r in rows if r["strategy"] == g], key=lambda r: r[xkey])
        xs = [r[xkey] for r in rs]
        axes[0].plot(xs, [r["param_err_max"] for r in rs], "o-", ms=5, label=g)
        axes[1].plot(xs, [r["src_rel_l2"] for r in rs], "s-", ms=5, label=g)
    axes[0].set_ylabel("max relative parameter error")
    axes[1].set_ylabel(r"source error")
    for ax in axes:
        ax.set_xlabel(xlabel or xkey)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, path)


# ----------------------------------------------------------------------
# 7. 自适应 vs 均匀：配点分布对比（论文用）
# ----------------------------------------------------------------------
def plot_points_compare(problem, pts_list, labels, truth: Dict[str, float],
                        path: str, abs_res_list=None) -> None:
    """把若干套配点方案的 ``(t,x)`` 投影并排画出来，叠加真值轨迹与层带。"""
    n = len(pts_list)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n + 0.6, 4.2),
                             squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(right=0.88 if n > 1 else 0.86, wspace=0.28)
    tt = np.linspace(0.0, problem.T, 400)
    curve = problem.layer_curve(tt)
    half = 4.0 * problem.sqrt_eps
    sc = None
    for ax, pts, lab, res in zip(axes, pts_list, labels,
                                 abs_res_list or [None] * n):
        if res is not None:
            sc = ax.scatter(pts[:, 2], pts[:, 0],
                            c=np.log10(np.abs(res) + 1e-12), s=3,
                            cmap="viridis", vmin=-4, vmax=0)
        else:
            ax.scatter(pts[:, 2], pts[:, 0], s=3, color="0.4")
        ax.plot(tt, curve, "r--", lw=1.6)
        ax.fill_between(tt, curve - half, curve + half, color="r", alpha=0.12)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        ax.set_xlim(0, problem.T)
        ax.set_ylim(0, 1)
        ax.set_title(lab, fontsize=10)
    if sc is not None:
        cax = fig.add_axes([0.905, 0.16, 0.012, 0.68])
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label(r"$\log_{10}|r|$")
    _save(fig, path)


# ----------------------------------------------------------------------
# 8. 观测数据长什么样（论文/说明用）
# ----------------------------------------------------------------------
def plot_data(problem, Z: np.ndarray, u_obs: np.ndarray, sigma: float,
              path: str, n_show: int = 3) -> None:
    """挑几个传感器，把观测 ``u^delta(t)`` 与真解曲线画在一起。

    典型形态：层扫过的那一两个时刻出现一个脉冲，其余时刻只有噪声。
    """
    xs = np.unique(np.round(Z[:, 0], 6))
    # 按每个传感器的峰值 |u^delta| 排序，挑"信号最强 / 中等 / 最弱"三个代表，
    # 这样一眼能看出"有的传感器能看到脉冲，有的只看到噪声"
    peak = np.array([np.abs(u_obs[np.isclose(Z[:, 0], x)]).max() for x in xs])
    order = np.argsort(-peak)
    pick = np.unique(np.array([order[0], order[len(order) // 2], order[-1]])
                     [:n_show])
    xs = xs[np.sort(pick)]
    fig, axes = plt.subplots(1, len(xs), figsize=(4.0 * len(xs), 3.2),
                             squeeze=False)
    axes = axes[0]
    tt = np.linspace(0.0, problem.T, 400)
    for ax, xk in zip(axes, xs):
        m = np.isclose(Z[:, 0], xk)
        yk = float(Z[m, 1][0])
        ax.plot(tt, np.asarray(problem.exact(np.stack(
            [np.full_like(tt, xk), np.full_like(tt, yk), tt], 1)),
            dtype=float).ravel(), "k-", lw=1.4, label=r"exact $u^*$")
        ax.errorbar(Z[m, 2], u_obs[m], yerr=sigma, fmt="o", ms=5, mfc="tab:red",
                    mec="tab:red", elinewidth=0.8, capsize=2, label="observations")
        for yy in (0.0, sigma, -sigma):
            ax.axhline(yy, color="0.85", lw=0.6,
                       ls="-" if yy == 0 else ":")
        ax.set_title(rf"$x={xk:.3f}$,  $y={yk:.2f}$", fontsize=9)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$u$")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, path)
