"""残差驱动的自适应配点 PINN 求解器。

算法
----
设 $\\Omega=(0,1)\\times(0,T)$。取初始配点集 $X_0\\subset\\Omega$（Halton 低差异序列），
在初值线 $\\Gamma_0$ 与边界 $\\Gamma_D$ 上取 $X_0^{ic}$、$X_0^{bc}$。对 $k=0,1,\\dots,K$：

1. **求解**：用 Adam 极小化

   $$
   J(\\theta;X_k)=\\frac1{|X_k|}\\sum_{z\\in X_k} r(z;\\theta)^2
   +\\lambda_{ic}\\sum_{\\xi\\in X_k^{ic}}\\frac{\\bigl(u_\\theta(\\xi)-u_0(\\xi)\\bigr)^2}{|X_k^{ic}|}
   +\\lambda_{bc}\\sum_{\\eta\\in X_k^{bc}}\\frac{\\bigl(u_\\theta(\\eta)-g(\\eta)\\bigr)^2}{|X_k^{bc}|},
   $$

   其中 $r=\\mathcal L[u_\\theta]-f$ 为 PDE 残差。
2. **评估**：在候选池 $C_k\\subset\\Omega$（均匀采样，$|C_k|\\gg|X_k|$）上计算
   $|r(z;\\theta)|$ 作为后验指标。
3. **增点**：$X_{k+1}=X_k\\cup\\{\\text{由 }|r|\\text{ 选出的 }m\\text{ 个候选点}\\}$，
   策略见 :mod:`spp_pinn.sampling`；初值/边值点按约束违反量同样增点。

网络参数在轮与轮之间热启动，因此整个过程等价于在逐步加密的点集上做一次连续优化。
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from .model import MLP
from .problems import make_problem
from .sampling import (CollocationSet, halton, select_distribution,
                       select_separated, select_topm, voronoi_weights_1d,
                       voronoi_weights_2d)


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
class TrainConfig:
    # ---------------- 问题 ----------------
    problem: str = "left"
    eps: float = 1e-4
    T: float = 1.0
    # ---------------- 网络 ----------------
    hidden: tuple = (64, 64, 64)
    activation: str = "tanh"
    # ---------------- 初始采样 ----------------
    n_int0: int = 1024
    n_ic0: int = 64
    n_bc0: int = 64
    init_method: str = "halton"          # 'halton' | 'uniform'
    # ---------------- 自适应增点 ----------------
    rounds: int = 8
    epochs_per_round: int = 1000
    add_per_round: int = 150             # 每轮新增内部配点数 m
    strategy: str = "residual"           # residual | distribution | uniform | fixed
    candidate_pool: int = 20000          # 候选池大小 |C_k|
    beta: float = 1.0                    # distribution 策略的指数
    min_dist: float = 0.0                # 贪心选点的最小间距 (0 = 不限制)
    adapt_ic: bool = True
    adapt_bc: bool = True
    add_ic_per_round: int = 16
    add_bc_per_round: int = 16
    # ---------------- 优化 ----------------
    lr: float = 2e-3
    lr_schedule: str = "cosine"          # 'cosine' | 'step'
    lr_gamma: float = 0.9                # 轮间学习率衰减
    weighting: str = "voronoi"           # 'voronoi' (保范数) | 'mean' (朴素平均)
    voronoi_n: int = 256                 # 离散 Voronoi 的网格分辨率
    weight_mode: str = "init"            # 'init' | 'fixed'
    w_ic: Optional[float] = None
    w_bc: Optional[float] = None
    hard_constraint: bool = False
    loss_norm: str = 'l2'            # 'l2' | 'l1'
    scaled_input: str = None        # None | 'raw' | 'tanh'
    feature: str = None             # None | 'fourier'
    n_freq: int = 64
    # ---------------- 其他 ----------------
    seed: int = 0
    device: str = "auto"
    verbose: bool = True
    eval_every: int = 0
    callback: object = None

    def to_dict(self):
        d = asdict(self)
        d["hidden"] = tuple(self.hidden)
        d["callback"] = None
        return d


class AdaptivePINNSolver:
    """残差驱动自适应配点的 PINN。"""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = pick_device(cfg.device)
        self.problem = make_problem(cfg.problem, cfg.eps, cfg.T)

        self.points = CollocationSet(T=cfg.T, seed=cfg.seed)
        self.points.initialise(cfg.n_int0, cfg.n_ic0, cfg.n_bc0, cfg.init_method)

        self.model = MLP(hidden=tuple(cfg.hidden), T=cfg.T,
                         activation=cfg.activation,
                         sqrt_eps=(float(np.sqrt(cfg.eps))
                                   if (cfg.scaled_input or cfg.feature) else None),
                         scaled=cfg.scaled_input,
                         feature=cfg.feature, n_freq=cfg.n_freq).to(self.device)

        # 固定的"均匀验证点集": 与训练点集无关, 用来客观度量残差是否真的下降
        u = halton(20000, 2, offset=777)
        self.val_pts = np.column_stack([u[:, 0], u[:, 1] * cfg.T])

        self.history: List[Dict] = []
        self.trace: List[Dict] = []
        self.global_epoch = 0
        self.w_ic = 1.0 if cfg.w_ic is None else float(cfg.w_ic)
        self.w_bc = 1.0 if cfg.w_bc is None else float(cfg.w_bc)
        self.weights_fixed = cfg.weight_mode == "fixed" or cfg.hard_constraint
        self._pool = np.zeros((0, 2))
        self.final_points = self.points.interior
        self.final_abs_res = np.zeros(0)
        self.final_ic = self.points.ic
        self.final_bc = self.points.bc

    # ==================================================================
    # 网络输出 (含可选硬约束)
    # ==================================================================
    def _u(self, xt: torch.Tensor) -> torch.Tensor:
        if not self.cfg.hard_constraint:
            return self.model(xt)
        prob = self.problem
        x, t = xt[:, 0:1], xt[:, 1:2]
        ic1 = torch.as_tensor(float(np.asarray(prob.ic(np.array([1.0])))[0]),
                              dtype=xt.dtype, device=xt.device)
        g = ((1.0 - x) * prob.bc_left(t) + x * prob.bc_right(t)
             + (1.0 - t / self.cfg.T) * (prob.ic(x) - x * ic1))
        return g + x * (1.0 - x) * t * self.model(xt)

    # ==================================================================
    # PDE 残差
    # ==================================================================
    def pde_residual(self, pts: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        prob = self.problem
        pts = pts.clone().detach().requires_grad_(True)
        u = self._u(pts)
        g1 = torch.autograd.grad(u, pts, torch.ones_like(u), create_graph=True)[0]
        u_x, u_t = g1[:, 0:1], g1[:, 1:2]
        u_xx = torch.autograd.grad(u_x, pts, torch.ones_like(u_x),
                                   create_graph=create_graph)[0][:, 0:1]
        x, t = pts[:, 0:1], pts[:, 1:2]
        return (u_t - self.cfg.eps * u_xx + prob.p(x) * u_x
                + prob.q(x) * u - prob.forcing(x, t))

    def residual_field(self, pts: np.ndarray, batch: int = 20000) -> np.ndarray:
        """在 numpy 点集上批量计算残差 (不建计算图)。"""
        out = []
        for k in range(0, len(pts), batch):
            xt = torch.as_tensor(np.ascontiguousarray(pts[k:k + batch]),
                                 dtype=torch.float32, device=self.device)
            r = self.pde_residual(xt, create_graph=False)
            out.append(r.detach().cpu().numpy().reshape(-1))
        return np.concatenate(out) if out else np.zeros(0)

    # ==================================================================
    # 损失
    # ==================================================================
    def _tensors(self) -> Dict[str, torch.Tensor]:
        """把当前点集与（可选）求积权重转成张量。"""
        dev, c, cfg = self.device, self.points, self.cfg

        def t_(a):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)

        left = c.bc[c.bc[:, 0] == 0.0]
        right = c.bc[c.bc[:, 0] == 1.0]
        d = {"int": t_(c.interior), "ic": t_(c.ic),
             "left": t_(left), "right": t_(right)}
        if cfg.weighting == "voronoi":
            wi = voronoi_weights_2d(c.interior, extent=(0.0, 1.0, 0.0, cfg.T),
                                    n=cfg.voronoi_n)
            d["w_int"] = t_(wi).reshape(-1, 1)
            d["w_ic"] = t_(voronoi_weights_1d(c.ic[:, 0], 0.0, 1.0)).reshape(-1, 1)
            d["w_left"] = t_(voronoi_weights_1d(left[:, 1], 0.0, cfg.T)).reshape(-1, 1)
            d["w_right"] = t_(voronoi_weights_1d(right[:, 1], 0.0, cfg.T)).reshape(-1, 1)
        return d

    def _loss_parts(self, data) -> Dict[str, torch.Tensor]:
        """损失各项。

        ``weighting="voronoi"``: $J=\\sum_i\\omega_i r_i^2/|\\Omega|$, 是
        $\\|r\\|_{L^2}^2$ 的相容求积 (点变密处 $\\omega_i$ 自动变小),
        因此自适应加点**不改变**目标泛函。

        ``weighting="mean"``: $J=\\frac1N\\sum_i r_i^2$。若点集是自适应的,
        这等价于把目标换成 $\\int r^2\\rho$ (被隐式重新加权)。
        """
        prob, T = self.problem, self.cfg.T
        vor = self.cfg.weighting == "voronoi"
        _p = (lambda z: z ** 2) if self.cfg.loss_norm == "l2" else (lambda z: z.abs())
        r = self.pde_residual(data["int"], create_graph=True)
        parts = {"pde": ((data["w_int"] * _p(r)).sum() / T) if vor else _p(r).mean()}
        if not self.cfg.hard_constraint:
            x_ic = data["ic"][:, 0:1]
            e_ic = _p(self._u(data["ic"]) - prob.ic(x_ic))
            e_l = _p(self._u(data["left"]) - prob.bc_left(data["left"][:, 1:2]))
            e_r = _p(self._u(data["right"]) - prob.bc_right(data["right"][:, 1:2]))
            if vor:
                parts["ic"] = (data["w_ic"] * e_ic).sum()
                parts["bc"] = 0.5 * ((data["w_left"] * e_l).sum()
                                     + (data["w_right"] * e_r).sum()) / T
            else:
                parts["ic"] = e_ic.mean()
                parts["bc"] = 0.5 * (e_l.mean() + e_r.mean())
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
        return {"x": xs, "t": ts, "X": X, "T": Tt, "u_pred": u_pred,
                "u_exact": u_true, "err": err, "max_err": float(err.max()),
                "rel_l2": float(np.linalg.norm(err) / denom) if denom > 0 else float("nan")}

    def layer_fraction(self, factor: float = 4.0) -> float:
        """内部配点中落在 $x\\le \\text{factor}\\cdot\\sqrt{\\varepsilon}$ 内的比例。"""
        thr = factor * float(np.sqrt(self.cfg.eps))
        return float(np.mean(self.points.interior[:, 0] <= thr))

    # ==================================================================
    # 自适应增点
    # ==================================================================
    def _choose(self, values: np.ndarray, m: int, existing: Optional[np.ndarray] = None):
        """按当前策略从候选池里挑 m 个点。"""
        st = self.cfg.strategy
        if st == "residual":
            if self.cfg.min_dist > 0:
                return select_separated(self._pool, np.abs(values), m,
                                        existing=existing, min_dist=self.cfg.min_dist)
            return select_topm(np.abs(values), m)
        if st == "distribution":
            return select_distribution(values, m, self.cfg.beta, self.points.rng)
        if st == "uniform":
            return self.points.rng.choice(values.size, size=int(min(m, values.size)),
                                          replace=False)
        raise KeyError(f"未知增点策略 {st!r}")

    def _enrich(self) -> int:
        cfg = self.cfg
        if cfg.strategy == "fixed":
            return 0

        # ---- 内部配点 ----
        self._pool = self.points.candidate_pool(cfg.candidate_pool)
        r_pool = self.residual_field(self._pool)
        idx = self._choose(r_pool, cfg.add_per_round, existing=self.points.interior)
        self.points.add_interior(self._pool[idx])

        # ---- 初值点: 用初值条件违反量选点 ----
        if cfg.adapt_ic and cfg.add_ic_per_round > 0:
            cand = self.points.rng.random(4000)
            pts = np.column_stack([cand, np.zeros_like(cand)])
            err = np.abs(self.predict(pts) - np.asarray(self.problem.ic(cand), dtype=float))
            j = self._choose(err, cfg.add_ic_per_round)
            self.points.add_ic(cand[j])

        # ---- 边值点 ----
        if cfg.adapt_bc and cfg.add_bc_per_round > 0:
            cand_t = self.points.rng.random(4000) * cfg.T
            left = np.column_stack([np.zeros_like(cand_t), cand_t])
            right = np.column_stack([np.ones_like(cand_t), cand_t])
            el = np.abs(self.predict(left) - np.asarray(self.problem.bc_left(cand_t), dtype=float))
            er = np.abs(self.predict(right) - np.asarray(self.problem.bc_right(cand_t), dtype=float))
            jl = self._choose(el, cfg.add_bc_per_round)
            jr = self._choose(er, cfg.add_bc_per_round)
            self.points.add_bc(np.concatenate([left[jl], right[jr]], axis=0))
        return int(cfg.add_per_round)

    # ==================================================================
    # 主循环
    # ==================================================================
    def run(self) -> List[Dict]:
        cfg = self.cfg

        if not self.weights_fixed and cfg.weight_mode == "init":
            parts = self._loss_parts(self._tensors())
            lp = float(parts["pde"].detach())
            li = float(parts["ic"].detach())
            lb = float(parts["bc"].detach())
            if cfg.w_ic is None:
                self.w_ic = float(np.clip(lp / max(li, 1e-12), 0.1, 1e3))
            if cfg.w_bc is None:
                self.w_bc = float(np.clip(lp / max(lb, 1e-12), 0.1, 1e3))
            if cfg.verbose:
                print(f"    [权重标定] L_pde={lp:.3e} L_ic={li:.3e} L_bc={lb:.3e} "
                      f"-> w_ic={self.w_ic:.3g}, w_bc={self.w_bc:.3g}")

        for rnd in range(cfg.rounds + 1):
            t0 = time.time()
            data = self._tensors()
            stats = self.points.stats()

            lr = cfg.lr * (cfg.lr_gamma ** rnd)
            opt = torch.optim.Adam(self.model.parameters(), lr=lr)
            if cfg.lr_schedule == "cosine":
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, cfg.epochs_per_round), eta_min=lr * 0.02)
            else:
                sched = torch.optim.lr_scheduler.StepLR(
                    opt, step_size=max(1, int(cfg.epochs_per_round * 0.4)), gamma=0.5)

            last: Dict[str, float] = {}
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
                    self.trace.append({"epoch": self.global_epoch + ep + 1, "round": rnd,
                                       "max_err": ev_t["max_err"], "rel_l2": ev_t["rel_l2"],
                                       "loss": last["total"], "N": self.points.total})

            ev = self.evaluate()
            # 客观残差: 在固定的均匀验证点集上测量, 与训练点集无关
            r_val = self.residual_field(self.val_pts)
            rms_val = float(np.sqrt(np.mean(r_val ** 2)))

            added = 0
            if rnd < cfg.rounds:
                added = self._enrich()

            rec = {
                "round": rnd, "n_int": stats.n_interior, "n_ic": stats.n_ic,
                "n_bc": stats.n_bc, "N": stats.total,
                "max_err": ev["max_err"], "rel_l2": ev["rel_l2"],
                "loss_pde": last.get("pde", float("nan")),
                "loss_ic": last.get("ic", float("nan")),
                "loss_bc": last.get("bc", float("nan")),
                "rms_res_uniform": rms_val,
                "max_res_uniform": float(np.abs(r_val).max()),
                "frac_layer": self.layer_fraction(),
                "added": added, "lr": lr, "wall": time.time() - t0,
            }
            self.global_epoch += cfg.epochs_per_round
            self.history.append(rec)
            if cfg.callback is not None:
                cfg.callback(rec)
            if cfg.verbose:
                print(f"    round {rnd}: N={rec['N']:6d} (int {rec['n_int']:5d}) "
                      f"max_err={rec['max_err']:.3e} rel_l2={rec['rel_l2']:.3e} "
                      f"rms_r={rms_val:.3e} layer_frac={rec['frac_layer']:.3f} "
                      f"({rec['wall']:.1f}s)")

            if rnd == cfg.rounds:
                self.final_points = self.points.interior.copy()
                self.final_abs_res = np.abs(self.residual_field(self.final_points))
                self.final_ic = self.points.ic.copy()
                self.final_bc = self.points.bc.copy()
        return self.history


def run_reference(cfg: TrainConfig, n_interior: int, total_epochs: int,
                  verbose: bool = True) -> AdaptivePINNSolver:
    """参照组: 固定点数、不做自适应增点 (点数与自适应组最终点数相同)。"""
    c = copy.deepcopy(cfg)
    c.n_int0 = n_interior
    c.strategy = "fixed"
    c.rounds = 0
    c.epochs_per_round = int(total_epochs)
    c.verbose = verbose
    s = AdaptivePINNSolver(c)
    s.run()
    return s
