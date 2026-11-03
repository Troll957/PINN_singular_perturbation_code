"""绘图工具 (统一输出 PNG)。"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

#纯画图程序
def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# 解与误差
# ----------------------------------------------------------------------
def plot_solution_fields(ev: Dict, eps: float, layer: str, path: str) -> str:
    X, Tt = ev["X"], ev["T"]
    fig = plt.figure(figsize=(15, 4.2))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot_surface(X, Tt, ev["u_exact"], cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title("Exact solution")
    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_zlabel(r"$u$")

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.plot_surface(X, Tt, ev["u_pred"], cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title("PINN prediction")
    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_zlabel(r"$u$")

    ax = fig.add_subplot(1, 3, 3)
    im = ax.pcolormesh(X, Tt, np.log10(ev["err"] + 1e-16), cmap="inferno", shading="auto")
    ax.set_title(f"log10 |error|   (max={ev['max_err']:.2e})")
    ax.set_xlabel("x"); ax.set_ylabel("t")
    fig.colorbar(im, ax=ax)
    fig.suptitle(rf"$\varepsilon={eps:g}$, layer: {layer}", y=1.03)
    return _save(fig, path)


def plot_cross_sections(ev: Dict, eps: float, path: str) -> str:
    x, t = ev["x"], ev["t"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))

    j = len(t) // 2
    axs[0].plot(x, ev["u_exact"][:, j], "k-", lw=2, label="exact")
    axs[0].plot(x, ev["u_pred"][:, j], "r--", lw=1.6, label="PINN")
    axs[0].set_xlabel("x"); axs[0].set_ylabel("u(x, t=0.5)")
    axs[0].legend(); axs[0].grid(alpha=0.3)
    axs[0].set_title("Cross section at t = 0.5")

    i = 0
    axs[1].semilogy(x, np.abs(ev["err"][:, j]) + 1e-16, "b-")
    axs[1].axvline(np.sqrt(eps), color="r", ls="--", lw=1,
                   label=rf"$\sqrt{{\varepsilon}}={np.sqrt(eps):.1e}$")
    axs[1].set_xlim(0, min(1.0, 10 * np.sqrt(eps) + 0.02))
    axs[1].set_xlabel("x"); axs[1].set_ylabel("|error|")
    axs[1].set_title("Error near the layer (t = 0.5)")
    axs[1].legend(); axs[1].grid(alpha=0.3, which="both")
    return _save(fig, path)


# ----------------------------------------------------------------------
# 网格
# ----------------------------------------------------------------------
def plot_mesh(cells: np.ndarray, pts: np.ndarray, abs_res: np.ndarray, T: float,
              path: str, title: str = "Adaptive mesh") -> str:
    res = abs_res / max(float(abs_res.max()), 1e-30)

    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.4))

    patches, vals = [], []
    for xa, xb, ta, tb, lev in cells:
        patches.append(Rectangle((xa, ta), xb - xa, tb - ta))
        vals.append(lev)
    pc = PatchCollection(patches, cmap="viridis", edgecolor="none")
    pc.set_array(np.array(vals))
    axs[0].add_collection(pc)
    axs[0].set_xlim(0, 1); axs[0].set_ylim(0, T)
    axs[0].set_title(f"Refinement level (cells={len(cells)}, max={int(max(vals))})")
    axs[0].set_xlabel("x"); axs[0].set_ylabel("t")
    fig.colorbar(pc, ax=axs[0], label="level")

    sc = axs[1].scatter(pts[:, 0], pts[:, 1], c=np.log10(res + 1e-16), s=6,
                        cmap="inferno", linewidths=0)
    axs[1].set_xlim(0, 1); axs[1].set_ylim(0, T)
    axs[1].set_title(rf"Sampling points ($16$/cell, $N={len(pts)}$) coloured by $\log_{{10}}|r|$")
    axs[1].set_xlabel("x"); axs[1].set_ylabel("t")
    fig.colorbar(sc, ax=axs[1], label=r"$\log_{10}|r|$")
    fig.suptitle(title, y=1.0)
    return _save(fig, path)


def plot_point_distribution(pts: np.ndarray, eps: float, path: str) -> str:
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))

    axs[0].hist(pts[:, 0], bins=120, color="steelblue", edgecolor="none")
    axs[0].set_yscale("log")
    axs[0].axvline(np.sqrt(eps), color="r", ls="--",
                   label=rf"$\sqrt{{\varepsilon}}={np.sqrt(eps):.1e}$")
    axs[0].set_xlabel("x"); axs[0].set_ylabel("count (log)")
    axs[0].set_title("Distribution of mesh sampling points in $x$")
    axs[0].legend()

    axs[1].scatter(pts[:, 0], pts[:, 1], s=3, c="k", alpha=0.4)
    axs[1].set_xlim(0, min(1.0, 40 * np.sqrt(eps) + 0.01))
    axs[1].set_xlabel("x"); axs[1].set_ylabel("t")
    axs[1].set_title("Zoom on the layer region")
    axs[1].grid(alpha=0.3)
    return _save(fig, path)


# ----------------------------------------------------------------------
# 收敛 / 历史
# ----------------------------------------------------------------------
def plot_convergence(curves: Dict[str, Sequence[Dict]], path: str,
                     xkey: str = "n_points", ykey: str = "max_err") -> str:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, (label, hist) in enumerate(curves.items()):
        xs = [h[xkey] for h in hist]
        ys = [h[ykey] for h in hist]
        ax.loglog(xs, ys, marker=markers[i % len(markers)], lw=1.8, ms=6, label=label)
    ax.set_xlabel(xkey); ax.set_ylabel(ykey)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    ax.set_title("Accuracy vs. number of collocation points")
    return _save(fig, path)


def plot_round_history(hist: List[Dict], path: str, eps: float = None) -> str:
    """逐轮历史: 误差 / 配点数与层内占比 / 均匀测试集上的残差。"""
    r = [h["round"] for h in hist]
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.3))

    axs[0].semilogy(r, [h["max_err"] for h in hist], "o-", label=r"$\|e\|_\infty$")
    axs[0].semilogy(r, [h["rel_l2"] for h in hist], "s-", label="rel. $L^2$")
    axs[0].set_xlabel("round $k$"); axs[0].set_ylabel("error")
    axs[0].grid(alpha=0.3, which="both"); axs[0].legend()
    axs[0].set_title("Solution error vs. round")

    ax1 = axs[1]
    ax1.plot(r, [h["N"] for h in hist], "o-", color="darkorange", label="total $N$")
    ax1.plot(r, [h["n_int"] for h in hist], "s--", color="chocolate",
             label="interior")
    ax1.set_xlabel("round $k$"); ax1.set_ylabel("number of points")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title("Growth of the collocation set")
    ax2 = ax1.twinx()
    ax2.plot(r, [h["frac_layer"] for h in hist], "^-", color="tab:green",
             label=r"fraction with $x\leq 4\sqrt{\varepsilon}$")
    ax2.set_ylabel(r"fraction inside $4\sqrt{\varepsilon}$", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.legend(fontsize=8, loc="lower right")

    axs[2].semilogy(r, [h["rms_res_uniform"] for h in hist], "o-", color="tab:red",
                    label="RMS residual")
    axs[2].semilogy(r, [h["loss_pde"] for h in hist], "s--", color="tab:blue",
                    label="training loss $J_{pde}$")
    axs[2].set_xlabel("round $k$"); axs[2].set_ylabel("residual")
    axs[2].grid(alpha=0.3, which="both"); axs[2].legend(fontsize=8)
    axs[2].set_title("Residual on a uniform test set")
    return _save(fig, path)


def plot_training_curves(hist: List[Dict], path: str) -> str:
    r = [h["round"] for h in hist]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(r, [h["loss_pde"] for h in hist], "o-", label=r"$L_{pde}$")
    if np.isfinite(hist[0].get("loss_ic", np.nan)):
        ax.semilogy(r, [h["loss_ic"] for h in hist], "s-", label=r"$L_{ic}$")
        ax.semilogy(r, [h["loss_bc"] for h in hist], "^-", label=r"$L_{bc}$")
    ax.set_xlabel("refinement round"); ax.set_ylabel("loss")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    ax.set_title("Loss components at the end of each round")
    return _save(fig, path)


def plot_indicator_correlation(eta: np.ndarray, eta_true: np.ndarray, cells: np.ndarray,
                               path: str) -> str:
    """指示子与真实单元误差的相关性 (左: 单元 L2 范数; 右: 单元密度)。

    在自适应网格上单元尺寸相差若干个数量级, 而 ``||e||_{L2(K)}`` 与 ``||r||_{L2(K)}``
    都随 |K| 减小而减小, 直接比较会被"单元尺寸"主导, 得到虚假的负相关。
    除以 sqrt(|K|) 之后 (即比较"密度"), 才能真正看出指示子的质量。
    """
    area = (cells[:, 1] - cells[:, 0]) * (cells[:, 3] - cells[:, 2])
    s = np.sqrt(np.maximum(area, 1e-300))

    fig, axs = plt.subplots(1, 2, figsize=(12.6, 5.2))
    panels = [
        (eta, eta_true, r"$\|r\|_{L^2(K)}$", r"$\|u-u^*\|_{L^2(K)}$", "cell $L^2$ norms"),
        (eta / s, eta_true / s, r"$\|r\|_{L^2(K)}/\sqrt{|K|}$",
         r"$\|u-u^*\|_{L^2(K)}/\sqrt{|K|}$", "cell densities"),
    ]
    for ax, (a, b, xl, yl, ttl) in zip(axs, panels):
        m = (a > 0) & (b > 0)
        c = float(np.corrcoef(np.log10(a[m]), np.log10(b[m]))[0, 1]) if m.sum() > 2 else float("nan")
        ax.loglog(a[m], b[m], "o", ms=4, alpha=0.55, color="tab:blue")
        lo = min(a[m].min(), b[m].min()); hi = max(a[m].max(), b[m].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"{ttl}   (corr={c:.3f})")
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("Residual indicator vs. true error per cell", y=1.02)
    return _save(fig, path)


# ----------------------------------------------------------------------
# 自适应配点专用图
# ----------------------------------------------------------------------
def plot_point_cloud(points: np.ndarray, abs_res: np.ndarray, ic: np.ndarray,
                     bc: np.ndarray, eps: float, T: float, path: str,
                     title: str = "") -> str:
    """最终配点分布: 全区域 / 层附近放大 / x 方向直方图。"""
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.6))
    sc = axs[0].scatter(points[:, 0], points[:, 1], s=7,
                        c=np.log10(abs_res + 1e-16), cmap="inferno", linewidths=0)
    axs[0].set_xlim(0, 1); axs[0].set_ylim(0, T)
    axs[0].set_title(f"Collocation points ($N={len(points)}$)")
    axs[0].set_xlabel("x"); axs[0].set_ylabel("t")
    fig.colorbar(sc, ax=axs[0], label=r"$\log_{10}|r|$")

    xmax = min(1.0, 20.0 * np.sqrt(eps) + 0.005)
    m = points[:, 0] <= xmax
    axs[1].scatter(points[m, 0], points[m, 1], s=8, color="tab:blue", alpha=0.6,
                   linewidths=0, label="interior")
    if ic.size:
        axs[1].scatter(ic[:, 0], ic[:, 1], s=8, color="tab:green", label="initial $t=0$")
    axs[1].axvline(np.sqrt(eps), color="r", ls="--", lw=1.2,
                   label=rf"$\sqrt{{\varepsilon}}={np.sqrt(eps):.1e}$")
    axs[1].set_xlim(0, xmax); axs[1].set_ylim(0, T)
    axs[1].set_title("Zoom near the layer")
    axs[1].set_xlabel("x"); axs[1].legend(fontsize=7.5, loc="upper right")

    axs[2].hist(points[:, 0], bins=120, color="steelblue", edgecolor="none")
    axs[2].set_yscale("log")
    axs[2].axvline(np.sqrt(eps), color="r", ls="--",
                   label=rf"$\sqrt{{\varepsilon}}$")
    axs[2].set_xlabel("x"); axs[2].set_ylabel("count (log)")
    axs[2].set_title("Distribution in $x$")
    axs[2].legend(fontsize=8)
    if title:
        fig.suptitle(title, y=1.03)
    return _save(fig, path)


def plot_layer_fraction(curves: Dict[str, List[Dict]], eps: float, path: str,
                        factor: float = 4.0) -> str:
    """各策略"落在层内的配点比例"随轮次的变化。"""
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.5))
    mk = ["o", "s", "^", "D"]
    for i, (lab, h) in enumerate(curves.items()):
        r = [q["round"] for q in h]
        axs[0].plot(r, [q["frac_layer"] for q in h], mk[i % 4] + "-", lw=1.8,
                    label=lab)
        axs[1].semilogy(r, [q["rms_res_uniform"] for q in h], mk[i % 4] + "-",
                        lw=1.8, label=lab)
    axs[0].axhline(factor * np.sqrt(eps), color="gray", ls=":",
                   label="uniform reference")
    axs[0].set_xlabel("round $k$")
    axs[0].set_ylabel(rf"fraction with $x\leq {factor:g}\sqrt{{\varepsilon}}$")
    axs[0].set_title("Concentration of points inside the layer")
    axs[0].grid(alpha=0.3); axs[0].legend(fontsize=8)
    axs[1].set_xlabel("round $k$")
    axs[1].set_ylabel(r"RMS residual on a uniform test set")
    axs[1].set_title("Objective residual measured on a uniform grid")
    axs[1].grid(alpha=0.3, which="both"); axs[1].legend(fontsize=8)
    return _save(fig, path)


def plot_strategy_bars(rows: List[Dict], eps: float, path: str) -> str:
    """策略对比柱状图 (max 误差与相对 L2)。"""
    labels = [r["label"] for r in rows]
    mx = [r["max_err"] for r in rows]
    l2 = [r["rel_l2"] for r in rows]
    N = [r["N"] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - w / 2, mx, w, label=r"$\|u-u^*\|_\infty$", color="tab:red")
    ax.bar(x + w / 2, l2, w, label="relative $L^2$", color="tab:blue")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n$N={n}$" for l, n in zip(labels, N)], fontsize=8.5)
    ax.set_ylabel("error"); ax.grid(alpha=0.3, axis="y"); ax.legend()
    ax.set_title(rf"Sampling strategy comparison ($\varepsilon={eps:g}$)")
    return _save(fig, path)


def plot_eps_sweep(rows: List[Dict], path: str) -> str:
    """自适应 vs 固定均匀: 误差随 eps 的变化。"""
    ad = sorted([r for r in rows if r["kind"] == "adaptive"], key=lambda r: -r["eps"])
    fi = sorted([r for r in rows if r["kind"] == "fixed"], key=lambda r: -r["eps"])
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, key, lab in ((axs[0], "max_err", r"$\|u-u^*\|_\infty$"),
                         (axs[1], "rel_l2", "relative $L^2$")):
        if ad:
            ax.loglog([r["eps"] for r in ad], [r[key] for r in ad], "o-",
                      color="tab:red", lw=2, ms=7,
                      label="residual-driven adaptive sampling")
        if fi:
            ax.loglog([r["eps"] for r in fi], [r[key] for r in fi], "s--",
                      color="tab:blue", lw=2, ms=7, label="fixed uniform points")
        ax.set_xlabel(r"$\varepsilon$"); ax.set_ylabel(lab)
        ax.invert_xaxis(); ax.grid(alpha=0.3, which="both"); ax.legend()
        ax.set_title(lab)
    fig.suptitle("Effect of the perturbation parameter", y=1.02)
    return _save(fig, path)


def plot_training_trace(trace: List[Dict], path: str) -> str:
    """逐 epoch 的误差 (若记录了 trace)。"""
    if not trace:
        return ""
    ep = [t["epoch"] for t in trace]
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.3))
    axs[0].semilogy(ep, [t["max_err"] for t in trace], "-", lw=1.6, label=r"$\|e\|_\infty$")
    axs[0].semilogy(ep, [t["rel_l2"] for t in trace], "-", lw=1.6, label="rel. $L^2$")
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("error")
    axs[0].grid(alpha=0.3, which="both"); axs[0].legend()
    axs[0].set_title("Error during training")
    axs[1].semilogy(ep, [t["loss"] for t in trace], "-", lw=1.6, color="darkorange")
    axs[1].set_xlabel("epoch"); axs[1].set_ylabel("training loss")
    axs[1].grid(alpha=0.3, which="both")
    axs[1].set_title("Training loss")
    return _save(fig, path)
