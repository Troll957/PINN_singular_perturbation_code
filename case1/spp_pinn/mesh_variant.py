"""[对照变体] 基于 quadtree 单元各向异性二分细化的 h-自适应 PINN。

本模块保留作为与"自适应配点"(主方法, 见 :mod:`spp_pinn.solver`)的对照实现。

原文档:

残差驱动 h-自适应 PINN 求解器。

算法 (每一轮 r = 0, 1, ..., R):

1. 在当前网格上取 Gauss 配点, 在边界单元上取初/边值配点;
2. 用 Adam 训练网络 ``epochs_per_round`` 步, 目标泛函

   .. math::
       J(\\theta) = \\sum_q w_q r(x_q, t_q; \\theta)^2
                  + \\lambda_{ic} \\sum_{q\\in\\Gamma_0} w_q (u-u_0)^2
                  + \\lambda_{bc} \\sum_{q\\in\\Gamma_D} w_q (u-g)^2

   其中 r 为 PDE 残差, 权重为 Gauss 求积权重;
3. 在配点上计算残差并聚合成单元指示子
   ``eta_K = ||r||_{L^2(K)}``;
4. Doerfler 标记 + 各向异性二分细化网格, 回到第 1 步。

也支持硬约束格式 (把初边值条件嵌入网络输出), 此时只剩 PDE 损失。
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import torch

from .mesh import AdaptiveMesh2D
from .model import MLP
from .problems import PI, make_problem


def pick_device(name: str = "auto") -> torch.device:
    """选择计算设备。

    Parameters
    ----------
    name : str
        ``"auto"``（默认）按 **CUDA > CPU** 自动选（理由见下）；
        也可以显式指定 ``"cuda"`` / ``"cuda:0"`` / ``"mps"`` / ``"cpu"``。

    Notes
    -----
    显式要求 ``"cuda"`` 但机器上没有可用 CUDA 时会**直接报错**，而不是悄悄退回
    CPU——否则你以为在用显卡，其实一直在用 CPU，白白浪费机时。
    """
    if name != "auto":
        dev = torch.device(name)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "指定了 cuda，但 torch.cuda.is_available() 为 False。逐条排查：\n"
                "  1) 装的是不是 CUDA 版 torch：\n"
                "     python -c \"import torch; print(torch.__version__, torch.version.cuda)\"\n"
                "     若 version.cuda 是 None，说明装成了 CPU 版，要重装 CUDA wheel。\n"
                "  2) 驱动是否正常：nvidia-smi\n"
                "  3) 是否被环境变量屏蔽：echo $CUDA_VISIBLE_DEVICES\n"
                "  4) 驱动版本是否低于 torch 要求的 CUDA runtime 版本。"
            )
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    # 注意: 这里**不自动选 MPS**。本项目的网络很小 (3x64) 且要用二阶自动微分,
    # 在 Apple Silicon 上实测 MPS 比 CPU 慢约 5 倍 (kernel 启动开销占主导)。
    # 静默选一个更慢的设备不如不选; 需要时显式传 --device mps。
    return torch.device("cpu")


def describe_device(dev: torch.device) -> str:
    """一行设备描述，打印在日志开头，方便确认到底跑在哪。"""
    if dev.type == "cuda":
        i = 0 if dev.index is None else dev.index
        try:
            return f"cuda:{i} ({torch.cuda.get_device_name(i)})"
        except Exception:                                   # noqa: BLE001
            return f"cuda:{i}"
    if dev.type == "mps":
        return "mps (Apple Silicon GPU)"
    return "cpu"


def device_benchmark(device: str = "auto", n: int = 2000, epochs: int = 200,
                     verbose: bool = True) -> float:
    """粗略测一下当前设备跑一步训练要多久（秒/步）。

    小网络 + 少量点时，显卡往往**不比 CPU 快**：每个 epoch 的算子很小，
    kernel 启动开销占主导。建议正式跑之前先用它对比 ``cpu`` 与 ``cuda``。
    """
    import time as _time
    from .model import MLP
    dev = pick_device(device)
    torch.manual_seed(0)
    net = MLP(hidden=(64, 64, 64), in_dim=2, T=1.0).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.rand(n, 1, device=dev, requires_grad=True)
    t = torch.rand(n, 1, device=dev, requires_grad=True)
    Z = torch.cat([x, t], dim=1)
    for _ in range(10):                                     # 预热
        opt.zero_grad(set_to_none=True)
        u = net(Z)
        g = torch.autograd.grad(u, Z, torch.ones_like(u), create_graph=True,
                                retain_graph=True)[0]
        uxx = torch.autograd.grad(g[:, 0:1], Z, torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0][:, 0:1]
        loss = (uxx ** 2).mean()
        loss.backward()
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = _time.time()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        u = net(Z)
        g = torch.autograd.grad(u, Z, torch.ones_like(u), create_graph=True,
                                retain_graph=True)[0]
        uxx = torch.autograd.grad(g[:, 0:1], Z, torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0][:, 0:1]
        loss = (uxx ** 2).mean()
        loss.backward()
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (_time.time() - t0) / epochs
    if verbose:
        print(f"[benchmark] {describe_device(dev)}  {n} 点, {dt*1000:.2f} ms/步")
    return dt



@dataclass
class MeshConfig:
    # --- 问题 ---
    problem: str = "left"
    eps: float = 1e-3
    T: float = 1.0
    # --- 网络 ---
    hidden: tuple = (64, 64, 64)
    activation: str = "tanh"
    # --- 网格 ---
    quad_order: int = 2
    nx0: int = 8
    nt0: int = 8
    min_size: float = 1e-6
    max_level: int = 24
    # --- 自适应 ---
    rounds: int = 7
    epochs_per_round: int = 3000
    theta: float = 0.3
    marking: str = "mix"              # 'mix' | 'max' | 'dorfler'
    beta: float = 0.25                # 最大值策略的相对阈值
    max_growth: float = 0.2           # 每轮最多细化的单元比例 (限制网格增长)
    max_points: int = 20000           # 配点数硬上限
    direction: str = "auto"
    indicator_samples: int = 4        # 每个单元用于估计指示子的均匀子样本数 (n x n)
    indicator_normalize: bool = False # True: 用残差密度 ||r||/sqrt(|K|) 作指示子
    # --- 优化 ---
    lr: float = 2e-3
    lr_gamma: float = 0.6
    lr_schedule: str = "step"          # 'step' | 'cosine'
    eval_every: int = 0                # >0 时每隔若干步记录一次误差
    weight_mode: str = "init"          # 'init' | 'fixed'
    w_ic: Optional[float] = None
    w_bc: Optional[float] = None
    hard_constraint: bool = False
    # --- 其他 ---
    seed: int = 0
    device: str = "auto"
    verbose: bool = True
    callback: object = None           # 每轮结束后的回调 callback(record)

    def to_dict(self):
        d = asdict(self)
        d["hidden"] = tuple(self.hidden)
        return d


class MeshAdaptivePINNSolver:
    """残差驱动 h-自适应 PINN。"""

    def __init__(self, cfg: MeshConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = pick_device(cfg.device)
        self.problem = make_problem(cfg.problem, cfg.eps, cfg.T)
        self.mesh = AdaptiveMesh2D(
            order=cfg.quad_order, nx=cfg.nx0, nt=cfg.nt0,
            extent=(0.0, 1.0, 0.0, cfg.T),
            min_size=cfg.min_size, max_level=cfg.max_level,
        )
        self.model = MLP(
            hidden=tuple(cfg.hidden), T=cfg.T, activation=cfg.activation,
        ).to(self.device)
        self.history: List[Dict] = []
        self.mesh_snapshots: List[np.ndarray] = []
        self.trace: List[Dict] = []
        self.global_epoch = 0
        self.w_ic = 1.0 if cfg.w_ic is None else float(cfg.w_ic)
        self.w_bc = 1.0 if cfg.w_bc is None else float(cfg.w_bc)
        self.weights_fixed = cfg.weight_mode == "fixed" or cfg.hard_constraint

    # ==================================================================
    # 网络输出 (含可选硬约束)
    # ==================================================================
    def _u(self, xt: torch.Tensor) -> torch.Tensor:
        if not self.cfg.hard_constraint:
            return self.model(xt)
        prob = self.problem
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        ic1 = torch.as_tensor(
            float(np.asarray(prob.ic(np.array([1.0])))[0]),
            dtype=xt.dtype, device=xt.device,
        )
        g = ((1.0 - x) * prob.bc_left(t)
             + x * prob.bc_right(t)
             + (1.0 - t / self.cfg.T) * (prob.ic(x) - x * ic1))
        return g + x * (1.0 - x) * t * self.model(xt)

    # ==================================================================
    # PDE 残差
    # ==================================================================
    def pde_residual(self, pts: torch.Tensor, create_graph: bool = True,
                     with_utt: bool = False):
        """计算 r = u_t - eps u_xx + p u_x + q u - f, 可选返回二阶导诊断量。"""
        prob = self.problem
        pts = pts.clone().detach().requires_grad_(True)
        u = self._u(pts)
        ones = torch.ones_like(u)
        g1 = torch.autograd.grad(u, pts, ones, create_graph=True)[0]
        u_x, u_t = g1[:, 0:1], g1[:, 1:2]

        gxx = torch.autograd.grad(u_x, pts, torch.ones_like(u_x),
                                  create_graph=create_graph, retain_graph=True)[0]
        u_xx = gxx[:, 0:1]
        u_tt = None
        if with_utt:
            gtt = torch.autograd.grad(u_t, pts, torch.ones_like(u_t),
                                      create_graph=create_graph)[0]
            u_tt = gtt[:, 1:2]

        x, t = pts[:, 0:1], pts[:, 1:2]
        r = (u_t - self.cfg.eps * u_xx
             + prob.p(x) * u_x + prob.q(x) * u - prob.forcing(x, t))
        return r, u_xx, u_tt

    def residual_field(self, pts: np.ndarray, with_utt: bool = False, batch: int = 20000):
        """在给定 (numpy) 点集上批量计算残差及二阶导诊断量, 不构建计算图。"""
        rs, axx, att = [], [], []
        for k in range(0, len(pts), batch):
            xt = torch.as_tensor(np.ascontiguousarray(pts[k:k + batch]),
                                 dtype=torch.float32, device=self.device)
            r, u_xx, u_tt = self.pde_residual(xt, create_graph=False, with_utt=with_utt)
            rs.append(r.detach().cpu().numpy().reshape(-1))
            axx.append(np.abs(u_xx.detach().cpu().numpy().reshape(-1)))
            if with_utt and u_tt is not None:
                att.append(np.abs(u_tt.detach().cpu().numpy().reshape(-1)))
        return (np.concatenate(rs),
                np.concatenate(axx) if axx else None,
                np.concatenate(att) if att else None)

    # ==================================================================
    # 损失
    # ==================================================================
    def _quad_tensors(self):
        pts, w, cell_id = self.mesh.quadrature()
        bnd = self.mesh.boundary_quadrature()
        dev = self.device

        def t_(a):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)

        q_pts = t_(pts)
        q_w = t_(w).reshape(-1, 1)
        ic_pts, ic_w = bnd["ic"]
        lf_pts, lf_w = bnd["left"]
        rt_pts, rt_w = bnd["right"]
        return {
            "pts": q_pts, "w": q_w, "cell_id": cell_id,
            "ic_pts": t_(ic_pts), "ic_w": t_(ic_w).reshape(-1, 1),
            "lf_pts": t_(lf_pts), "lf_w": t_(lf_w).reshape(-1, 1),
            "rt_pts": t_(rt_pts), "rt_w": t_(rt_w).reshape(-1, 1),
        }

    def _loss_parts(self, data) -> Dict[str, torch.Tensor]:
        prob = self.problem
        r, _, _ = self.pde_residual(data["pts"], create_graph=True)
        parts = {"pde": (data["w"] * r ** 2).sum()}
        if not self.cfg.hard_constraint:
            x_ic = data["ic_pts"][:, 0:1]
            t_ic = data["ic_pts"][:, 1:2]
            u_ic = self._u(data["ic_pts"])
            parts["ic"] = (data["ic_w"] * (u_ic - prob.ic(x_ic)) ** 2).sum()

            u_lf = self._u(data["lf_pts"])
            u_rt = self._u(data["rt_pts"])
            g_lf = prob.bc_left(data["lf_pts"][:, 1:2])
            g_rt = prob.bc_right(data["rt_pts"][:, 1:2])
            parts["bc"] = ((data["lf_w"] * (u_lf - g_lf) ** 2).sum()
                           + (data["rt_w"] * (u_rt - g_rt) ** 2).sum())
        return parts

    def _total(self, parts) -> torch.Tensor:
        total = parts["pde"]
        if "ic" in parts:
            total = total + self.w_ic * parts["ic"] + self.w_bc * parts["bc"]
        return total

    # ==================================================================
    # 评估
    # ==================================================================
    def predict(self, pts: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            xt = torch.as_tensor(np.ascontiguousarray(pts), dtype=torch.float32,
                                 device=self.device)
            u = self._u(xt).detach().cpu().numpy().reshape(-1)
        self.model.train()
        return u

    def evaluate(self, nx: int = 401, nt: int = 201) -> Dict[str, np.ndarray]:
        xs = np.linspace(0.0, 1.0, nx)
        ts = np.linspace(0.0, self.cfg.T, nt)
        X, Tt = np.meshgrid(xs, ts, indexing="ij")
        pts = np.stack([X.reshape(-1), Tt.reshape(-1)], axis=1)
        u_pred = self.predict(pts).reshape(nx, nt)
        u_true = np.asarray(self.problem.exact(X, Tt), dtype=float)
        err = np.abs(u_pred - u_true)
        denom = np.linalg.norm(u_true)
        return {
            "x": xs, "t": ts, "X": X, "T": Tt,
            "u_pred": u_pred, "u_exact": u_true, "err": err,
            "max_err": float(err.max()),
            "rel_l2": float(np.linalg.norm(err) / denom) if denom > 0 else float("nan"),
        }

    def cell_error(self, n: int = 4):
        """在每个单元上估计真实 L2 误差, 用于检验指示子的可靠性。"""
        pts, cell_id, w = self.mesh.cell_samples(n)
        u_pred = self.predict(pts)
        u_true = np.asarray(self.problem.exact(pts[:, 0], pts[:, 1]), dtype=float)
        e2 = (u_pred - u_true) ** 2
        err2 = np.bincount(cell_id, weights=e2 * w, minlength=self.mesh.n_cells)
        return np.sqrt(np.maximum(err2, 0.0))

    # ==================================================================
    # 主循环
    # ==================================================================
    def run(self) -> List[Dict]:
        cfg = self.cfg
        data = self._quad_tensors()

        # 损失权重标定 (仅软约束)
        if not self.weights_fixed and cfg.weight_mode == "init":
            with torch.enable_grad():
                parts = self._loss_parts(data)
            lp = float(parts["pde"].detach())
            li = float(parts["ic"].detach())
            lb = float(parts["bc"].detach())
            w_ic = np.clip(lp / max(li, 1e-12), 0.1, 1e3)
            w_bc = np.clip(lp / max(lb, 1e-12), 0.1, 1e3)
            if cfg.w_ic is None:
                self.w_ic = float(w_ic)
            if cfg.w_bc is None:
                self.w_bc = float(w_bc)
            if cfg.verbose:
                print(f"    [权重标定] L_pde={lp:.3e} L_ic={li:.3e} L_bc={lb:.3e} "
                      f"-> w_ic={self.w_ic:.3g}, w_bc={self.w_bc:.3g}")

        for rnd in range(cfg.rounds + 1):
            t0 = time.time()
            data = self._quad_tensors()
            stats = self.mesh.stats()
            self.mesh_snapshots.append(self.mesh.cells.copy())

            lr = cfg.lr * (cfg.lr_gamma ** rnd)
            opt = torch.optim.Adam(self.model.parameters(), lr=lr)
            if cfg.lr_schedule == "cosine":
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, cfg.epochs_per_round), eta_min=lr * 0.02)
            else:
                sched = torch.optim.lr_scheduler.StepLR(
                    opt, step_size=max(1, int(cfg.epochs_per_round * 0.4)), gamma=0.5)

            if rnd == 0 and cfg.verbose:
                print(f"    round {rnd}: cells={stats.n_cells}, pts={stats.n_points}, "
                      f"ic={data['ic_pts'].shape[0]}, lr={lr:.2e}")

            last = {}
            for ep in range(cfg.epochs_per_round):
                opt.zero_grad(set_to_none=True)
                parts = self._loss_parts(data)
                loss = self._total(parts)
                loss.backward()
                opt.step()
                sched.step()
                last = {k: float(v.detach()) for k, v in parts.items()}
                last["total"] = float(loss.detach())
                if cfg.eval_every and (ep + 1) % cfg.eval_every == 0:
                    ev_t = self.evaluate(nx=201, nt=101)
                    self.trace.append({
                        "epoch": self.global_epoch + ep + 1, "round": rnd,
                        "max_err": ev_t["max_err"], "rel_l2": ev_t["rel_l2"],
                        "loss": last["total"], "n_points": self.mesh.n_points})
                if cfg.verbose and (ep + 1) % max(1, cfg.epochs_per_round // 3) == 0:
                    print(f"      ep {ep+1:5d}  total={last['total']:.3e}  "
                          f"pde={last['pde']:.3e}"
                          + (f"  ic={last.get('ic', float('nan')):.2e}"
                             f"  bc={last.get('bc', float('nan')):.2e}"
                             if "ic" in last else ""))

            ev = self.evaluate()
            stats = self.mesh.stats()

            # --- 后验误差指示子 ---
            # 在每个单元内部用 n x n 个均匀子样本估计 eta_K = ||r||_{L2(K)}。
            # 注意: 直接用损失用的 Gauss 点会严重低估层内残差 (粗单元的 Gauss
            # 点可能整体落在层外), 因此这里用更密的子样本。
            need_utt = (cfg.direction == "auto")
            sub_pts, sub_cid, sub_w = self.mesh.cell_samples(max(2, cfg.indicator_samples))
            r_sub, axx, att = self.residual_field(sub_pts, with_utt=need_utt)
            eta = self.mesh.cell_indicators(r_sub ** 2, sub_w, sub_cid,
                                            normalize=cfg.indicator_normalize)
            n_refined = 0
            n_marked = 0
            dirs = None
            if rnd < cfg.rounds:
                budget = self.mesh.n_cells
                allowed = max(4, int(cfg.max_growth * budget))
                if self.mesh.n_points >= cfg.max_points:
                    allowed = 0
                marked = self.mesh.mark(
                    eta, cfg.marking,
                    cfg.beta if cfg.marking == "max" else cfg.theta,
                    max_marked=allowed)
                n_marked = int(marked.sum())
                if need_utt:
                    ex, et = self.mesh.directional_estimates(axx, att, sub_w, sub_cid)
                    dirs = self.mesh.choose_directions(ex, et, "auto")
                else:
                    dirs = self.mesh.choose_directions(None, None, cfg.direction)
                n_refined = self.mesh.refine(marked, dirs)

            self.global_epoch += cfg.epochs_per_round
            rec = {
                "round": rnd,
                "n_cells": stats.n_cells,
                "n_points": stats.n_points,
                "n_ic": int(data["ic_pts"].shape[0]),
                "n_bc": int(data["lf_pts"].shape[0] + data["rt_pts"].shape[0]),
                "max_err": ev["max_err"],
                "rel_l2": ev["rel_l2"],
                "loss_total": last.get("total", float("nan")),
                "loss_pde": last.get("pde", float("nan")),
                "loss_ic": last.get("ic", float("nan")),
                "loss_bc": last.get("bc", float("nan")),
                "eta_max": float(eta.max()) if eta.size else 0.0,
                "eta_sum": float(np.sqrt((eta ** 2).sum())),
                "min_hx": stats.min_hx,
                "min_ht": stats.min_ht,
                "max_level": stats.max_level,
                "n_marked": n_marked,
                "n_refined": n_refined,
                "lr": lr,
                "wall": time.time() - t0,
            }
            self.history.append(rec)
            if cfg.callback is not None:
                cfg.callback(rec)
            if cfg.verbose:
                print(f"    round {rnd}: cells={rec['n_cells']:5d} pts={rec['n_points']:6d} "
                      f"max_err={rec['max_err']:.3e} rel_l2={rec['rel_l2']:.3e} "
                      f"eta_max={rec['eta_max']:.2e} marked={n_marked} refined={n_refined} "
                      f"min_hx={rec['min_hx']:.2e} ({rec['wall']:.1f}s)")
            if rnd == cfg.rounds:
                # 最后一轮: 记录指示子与真实误差的相关性
                eta_true = self.cell_error()
                if eta.size > 1:
                    rec["indicator_corr"] = float(np.corrcoef(
                        np.log10(eta + 1e-30), np.log10(eta_true + 1e-30))[0, 1])
                    rec["effectivity"] = float(
                        np.sqrt((eta_true ** 2).sum()) / max(np.sqrt((eta ** 2).sum()), 1e-30))
                # 保存末端网格与诊断量 (供绘图 / 复现)
                self.final_cells = self.mesh.cells.copy()
                self.final_quad_pts = sub_pts
                self.final_abs_res = np.abs(r_sub)
                self.final_eta = eta
                self.final_eta_true = eta_true
        return self.history


def run_uniform(cfg: MeshConfig, n: int, total_epochs: int, verbose: bool = True) -> AdaptivePINNSolver:
    """均匀网格基线: 与自适应算例使用完全相同的求解器与训练预算。"""
    cfg_u = copy.deepcopy(cfg)
    cfg_u.nx0 = n
    cfg_u.nt0 = n
    cfg_u.rounds = 0
    cfg_u.epochs_per_round = int(total_epochs)
    cfg_u.verbose = verbose
    solver = MeshAdaptivePINNSolver(cfg_u)
    solver.run()
    return solver
