"""反源问题的求解器：残差驱动自适应配点 + 联合反演源参数。

与正文正问题算例的区别只有两处：

1. 残差里的源项是 :math:`f(z;\\phi)`，:math:`\\phi` 是**可训练参数**，
   与网络权重一起优化；
2. 损失里多了一项数据拟合 ``J_data``（观测点上的均方误差 / 噪声方差）。

其余（点集、kNN 体积权重、增点规则、学习率调度）与正问题完全一致，
这样可以保证正问题与反问题的对比是干净的。

两种算例（见 :mod:`case5.problem`）：

* ``moving``（默认）：源是一条以速度 :math:`v` 扫过区域的薄脊，
  未知量 :math:`(A,s_0,v)`；
* ``static``：源固定在 :math:`x=x_s`，未知量 :math:`(A,x_s,y_s,\\rho)`。
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .model import MLP
from .problem import (MovingTruth, SourceParams, StaticTruth, crlb,
                      make_problem)
from .quadrature import Points2D, knn_quadrature_weights, select_topm


# ======================================================================
# 设备
# ======================================================================
def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        dev = torch.device(name)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定了 cuda，但 torch.cuda.is_available() 为 False")
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(dev: torch.device) -> str:
    if dev.type == "cuda":
        i = 0 if dev.index is None else dev.index
        try:
            return f"cuda:{i} ({torch.cuda.get_device_name(i)})"
        except Exception:                                   # noqa: BLE001
            return f"cuda:{i}"
    return str(dev)


# ======================================================================
# 配置
# ======================================================================
@dataclass
class Case5Config:
    # ---- 算例 ----
    case: str = "moving"                # moving | static
    # ---- 方程 ----
    eps: float = 1e-3
    b1: float = 1.0
    b2: float = 0.0
    q: float = 1.0
    T: float = 1.0
    # ---- 真值（按算例取用）----
    true_A: float = 2.0
    true_s0: float = 0.15               # moving
    true_v: float = 0.70                # moving
    true_xs: float = 0.30               # static
    true_ys: float = 0.35               # static
    true_rho: float = 0.08              # static
    # ---- 初值（故意偏离真值）----
    init_A: float = 1.5
    init_s0: float = 0.35               # moving
    init_v: float = 0.40                # moving
    init_xs: float = 0.35               # static
    init_ys: float = 0.40               # static
    init_rho: float = 0.10              # static
    freeze: str = ""                    # 例如 "v" 或 "xs,ys"
    # ---- 网络 ----
    hidden: tuple = (64, 64, 64)
    activation: str = "tanh"
    # ---- 配点 ----
    n_int0: int = 2048
    n_ic0: int = 128
    n_bc0: int = 64
    init_method: str = "halton"
    rounds: int = 8
    epochs_per_round: int = 3000
    add_per_round: int = 250
    strategy: str = "residual"          # residual | uniform | fixed
    candidate_pool: int = 20000
    add_ic_per_round: int = 16
    add_bc_per_round: int = 8
    weighting: str = "knn"              # knn | mean
    knn_neighbors: int = 4
    # ---- 观测数据 ----
    n_sensors: int = 12
    n_times: int = 16
    noise: float = 0.01
    sensor_mode: str = "grid"           # grid | random | ridge | slab
    x_span: tuple = (0.05, 0.95)        # grid 模式下传感器的 x 范围
    y_span: tuple = (0.20, 0.80)
    # ---- moving 算例的 y 剖面 ----
    wall_layer: bool = False            # True: y=0 处再加一条 sqrt(eps) 壁面层
    yc: float = 0.50                    # y 剖面中心
    rho_y: float = 0.20                 # y 剖面宽度
    slab: tuple = (0.10, 0.50)          # static + slab 模式用
    # ---- 损失 ----
    w_data: float = 1.0
    w_ic: Optional[float] = None
    w_bc: Optional[float] = None
    use_pde: bool = True
    # ---- 优化 ----
    lr: float = 2e-3
    lr_param: float = 5e-3
    lr_gamma: float = 0.9
    # ---- 其他 ----
    seed: int = 0
    device: str = "auto"
    verbose: bool = True
    eval_n: int = 81
    eval_nt: int = 41
    tag: str = "case5"

    # ---------- 派生 ----------
    def truth(self):
        if self.case == "moving":
            return MovingTruth(A=self.true_A, s0=self.true_s0, v=self.true_v)
        if self.case == "static":
            return StaticTruth(A=self.true_A, xs=self.true_xs,
                               ys=self.true_ys, rho=self.true_rho)
        raise KeyError(f"未知算例 {self.case!r}")

    @property
    def param_names(self) -> Tuple[str, ...]:
        return ("A", "s0", "v") if self.case == "moving" else ("A", "xs", "ys", "rho")

    def init_dict(self) -> Dict[str, float]:
        if self.case == "moving":
            return {"A": self.init_A, "s0": self.init_s0, "v": self.init_v}
        return {"A": self.init_A, "xs": self.init_xs,
                "ys": self.init_ys, "rho": self.init_rho}

    def trainable(self) -> Tuple[str, ...]:
        drop = {s.strip() for s in self.freeze.split(",") if s.strip()}
        return tuple(k for k in self.param_names if k not in drop)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        d["slab"] = list(self.slab)
        d["x_span"] = list(self.x_span)
        d["y_span"] = list(self.y_span)
        d["trainable"] = list(self.trainable())
        return d


# ======================================================================
# 观测数据
# ======================================================================
def make_sensors(problem, cfg: Case5Config) -> Tuple[np.ndarray, np.ndarray, float]:
    """生成观测点 ``(K,3)``、观测值 ``(K,)`` 与噪声标准差。

    ``sensor_mode``：

    * ``grid``（moving 的默认）：在 :math:`x` 方向均匀摆 ``n_sensors`` 个固定位置，
      每个位置记 ``n_times`` 个时刻。层扫过某个传感器时它会看到一段"脉冲"，
      脉冲的到达时刻就是 :math:`(x_k-s_0)/v`，这就是速度可辨识的来源。
    * ``random``：在区域里均匀撒 ``n_sensors*n_times`` 个时空点。
    * ``ridge``：围绕真值轨迹撒点（上限对照）。
    * ``slab``：static 算例的竖直测量窗。
    """
    rng = np.random.default_rng(cfg.seed + 1000)
    tr = problem.truth
    names = problem.param_names

    if cfg.sensor_mode == "grid":
        K = int(cfg.n_sensors)
        x = np.linspace(cfg.x_span[0], cfg.x_span[1], K)
        y = rng.uniform(cfg.y_span[0], cfg.y_span[1], K)
        t = np.linspace(0.02, 0.98, int(cfg.n_times)) * cfg.T
        Z = np.stack([np.repeat(x, t.size),
                      np.repeat(y, t.size),
                      np.tile(t, K)], axis=1)
    elif cfg.sensor_mode == "random":
        K = int(cfg.n_sensors) * int(cfg.n_times)
        Z = np.column_stack([rng.uniform(0, 1, K), rng.uniform(0, 1, K),
                             rng.uniform(0, cfg.T, K)])
    elif cfg.sensor_mode == "ridge":
        K = int(cfg.n_sensors) * int(cfg.n_times)
        t = rng.uniform(0, cfg.T, K)
        s = float(tr.s0) + float(tr.v) * t
        Z = np.column_stack([
            np.clip(s + rng.uniform(-4, 4, K) * problem.sqrt_eps, 1e-4, 1 - 1e-4),
            np.clip(rng.uniform(0, 1, K), 1e-4, 1 - 1e-4), t])
    elif cfg.sensor_mode == "slab":
        K = int(cfg.n_sensors)
        x = rng.uniform(cfg.slab[0], cfg.slab[1], K)
        y = rng.uniform(0.0, 1.0, K)
        t = (np.arange(int(cfg.n_times)) + 0.5) / int(cfg.n_times) * cfg.T
        Z = np.stack([np.repeat(x, t.size), np.repeat(y, t.size),
                      np.tile(t, K)], axis=1)
    else:
        raise KeyError(f"未知 sensor_mode {cfg.sensor_mode!r}")

    u_true = np.asarray(problem.exact(Z), dtype=float).ravel()
    scale = float(np.abs(u_true).max())
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(tr.A)
    sigma = float(cfg.noise) * scale
    u_obs = u_true + sigma * rng.standard_normal(u_true.size)
    return Z, u_obs, sigma


# ======================================================================
# 求解器
# ======================================================================
class InversePINN:
    """联合反演 :math:`(\\phi, \\theta)` 的自适应配点 PINN。"""

    def __init__(self, cfg: Case5Config):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = pick_device(cfg.device)

        self.problem = make_problem(cfg.case, eps=cfg.eps,
                                    b=(cfg.b1, cfg.b2), q=cfg.q,
                                    T=cfg.T, truth=cfg.truth(),
                                    wall=cfg.wall_layer, yc=cfg.yc,
                                    rho_y=cfg.rho_y)
        self.param_names = self.problem.param_names
        self.dim, self.D = 2, 3

        # ---- 网络与参数 ----
        self.model = MLP(hidden=tuple(cfg.hidden), in_dim=3, T=cfg.T,
                         activation=cfg.activation).to(self.device)
        self.params = SourceParams(names=self.param_names,
                                   init=cfg.init_dict(),
                                   trainable=cfg.trainable()).to(self.device)

        # ---- 点集 ----
        self.points = Points2D(T=cfg.T, seed=cfg.seed)
        self.points.initialise(cfg.n_int0, cfg.n_ic0, cfg.n_bc0,
                               self.problem.faces(), cfg.init_method)

        # ---- 数据 ----
        self.Z_data, self.u_obs, self.sigma = make_sensors(self.problem, cfg)

        # ---- 损失权重 ----
        self.w_ic = 1.0 if cfg.w_ic is None else float(cfg.w_ic)
        self.w_bc = 1.0 if cfg.w_bc is None else float(cfg.w_bc)
        self.w_data = float(cfg.w_data)

        self.history: List[Dict] = []
        self.final_points = self.points.interior.copy()
        self.val_pts = self.points.candidate_pool(20000)
        self.crlb: Dict[str, float] = {}
        self.cond_fisher = float("nan")

    # ---------------- 工具 ----------------
    def _t(self, a) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32,
                               device=self.device)

    # ---------------- 残差 ----------------
    def pde_residual(self, Z: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        r"""``r = u_t - eps*lap(u) + b.grad(u) + q u - f(z; phi)``。"""
        prob = self.problem
        Z = Z.clone().detach().requires_grad_(True)
        u = self.model(Z)
        ones = torch.ones_like(u)
        g = torch.autograd.grad(u, Z, ones, create_graph=True, retain_graph=True)[0]
        u_t = g[:, 2:3]

        gxx = torch.autograd.grad(g[:, 0:1], Z, ones, create_graph=create_graph,
                                  retain_graph=True)[0]
        gyy = torch.autograd.grad(g[:, 1:2], Z, ones, create_graph=create_graph,
                                  retain_graph=True)[0]
        lap = gxx[:, 0:1] + gyy[:, 1:2]

        f = prob.forcing(Z, self.params.values())
        return (u_t - prob.eps * lap
                + prob.b1 * g[:, 0:1] + prob.b2 * g[:, 1:2]
                + prob.q * u - f)

    def residual_field(self, pts: np.ndarray, batch: int = 20000) -> np.ndarray:
        out = []
        for k in range(0, len(pts), batch):
            Z = self._t(pts[k:k + batch])
            out.append(self.pde_residual(Z, create_graph=False)
                       .detach().cpu().numpy().ravel())
        return np.concatenate(out) if out else np.zeros(0)

    def predict(self, pts: np.ndarray, batch: int = 50000) -> np.ndarray:
        self.model.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(pts), batch):
                out.append(self.model(self._t(pts[k:k + batch]))
                           .detach().cpu().numpy().ravel())
        self.model.train()
        return np.concatenate(out)

    # ---------------- 权重 ----------------
    def _weights(self, P: np.ndarray, drop_axis: Optional[int],
                 measure: float) -> np.ndarray:
        if self.cfg.weighting != "knn":
            return np.full(P.shape[0], measure / max(P.shape[0], 1))
        coords = P if drop_axis is None else np.delete(P, drop_axis, axis=1)
        return knn_quadrature_weights(coords, measure,
                                      n_neighbors=self.cfg.knn_neighbors)

    def _tensors(self) -> Dict:
        c, cfg = self.points, self.cfg
        w_int = self._weights(c.interior, None, cfg.T)
        w_ic = self._weights(c.ic, 2, 1.0)
        bc = {f: (self._t(P), self._t(self._weights(P, f[0], cfg.T)).reshape(-1, 1))
              for f, P in c.bc.items()}
        return {"int": self._t(c.interior),
                "w_int": self._t(w_int).reshape(-1, 1),
                "ic": self._t(c.ic),
                "w_ic": self._t(w_ic).reshape(-1, 1),
                "bc": bc}

    # ---------------- 损失 ----------------
    def _loss_parts(self, data: Dict) -> Dict[str, torch.Tensor]:
        prob, T = self.problem, self.cfg.T
        vor = self.cfg.weighting == "knn"

        if self.cfg.use_pde:
            r = self.pde_residual(data["int"], create_graph=True)
            pde = (data["w_int"] * r * r).sum() / T if vor else (r * r).mean()
        else:
            pde = torch.zeros((), device=self.device)

        e_ic = (self.model(data["ic"]) - prob.ic(data["ic"][:, :2])).pow(2)
        parts_ic = ((data["w_ic"] * e_ic).sum() if vor else e_ic.mean())

        bc_terms = []
        for (axis, value), (P, w) in data["bc"].items():
            e = (self.model(P) - prob.bc(P)).pow(2)
            bc_terms.append((w * e).sum() / T if vor else e.mean())
        parts_bc = sum(bc_terms) / max(len(bc_terms), 1)

        u_d = self.model(self.Z_data_t).reshape(-1)
        data_misfit = ((u_d - self.u_obs_t) ** 2).mean() / max(self.sigma ** 2, 1e-30)

        return {"pde": pde, "ic": parts_ic, "bc": parts_bc, "data": data_misfit}

    def _total(self, parts: Dict[str, torch.Tensor]) -> torch.Tensor:
        return (parts["pde"] + self.w_ic * parts["ic"]
                + self.w_bc * parts["bc"] + self.w_data * parts["data"])

    # ---------------- 梯度归因 ----------------
    def _grad_share(self) -> Dict[str, float]:
        """参数梯度中来自"层内配点"的比例（诊断量，不参与训练）。"""
        names = self.params.trainable_names()
        if not names:
            return {}
        c = self.points
        mask = self.problem.inside_layer(c.interior).astype(np.float32)
        Z = self._t(c.interior)
        w = self._t(self._weights(c.interior, None, self.cfg.T)).reshape(-1, 1)
        m = torch.as_tensor(mask, dtype=torch.float32,
                            device=self.device).reshape(-1, 1)
        r = self.pde_residual(Z, create_graph=True)
        term = w * r * r
        ps = [self.params.raw[k] for k in names]
        g_in = torch.autograd.grad((term * m).sum(), ps, retain_graph=True,
                                   allow_unused=True)
        g_out = torch.autograd.grad((term * (1.0 - m)).sum(), ps, allow_unused=True)
        out = {}
        for k, a, b in zip(names, g_in, g_out):
            a = 0.0 if a is None else abs(float(a))
            b = 0.0 if b is None else abs(float(b))
            out[k] = a / max(a + b, 1e-30)
        return out

    # ---------------- 评估 ----------------
    def evaluate(self) -> Dict[str, float]:
        cfg = self.cfg
        n, nt = cfg.eval_n, cfg.eval_nt
        ax = np.linspace(0.0, 1.0, n)
        at = np.linspace(0.0, cfg.T, nt)
        X, Y, Tt = np.meshgrid(ax, ax, at, indexing="ij")
        Z = np.stack([X.ravel(), Y.ravel(), Tt.ravel()], axis=1)
        u_pred = self.predict(Z)
        u_true = np.asarray(self.problem.exact(Z), dtype=float).ravel()
        err = np.abs(u_pred - u_true)
        den = np.linalg.norm(u_true)
        # 全区域误差 + 只统计移动层内的误差（层内才是反演真正关心的区域）
        s_curve = np.asarray(self.problem.layer_curve(Z[:, 2]), dtype=float)
        in_layer = np.abs(Z[:, 0] - s_curve) <= 4.0 * self.problem.sqrt_eps
        e_l = err[in_layer]
        out = {
            "max_err": float(err.max()),
            "rel_l2": float(np.linalg.norm(err) / den) if den > 0 else float("nan"),
            "max_err_layer": float(e_l.max()) if e_l.size else float("nan"),
            "rel_l2_layer": (float(np.linalg.norm(e_l)
                                   / max(np.linalg.norm(u_true[in_layer]), 1e-30))
                             if e_l.size else float("nan")),
            "frac_eval_layer": float(in_layer.mean()),
        }
        out.update(self.problem.source_error(self.params.numpy_values()))

        p_hat = self.params.numpy_values()
        p_true = self.problem.truth_dict
        for k in self.param_names:
            out[f"param_{k}"] = float(p_hat[k])
            out[f"param_rel_{k}"] = float(abs(p_hat[k] - p_true[k])
                                          / max(abs(p_true[k]), 1e-12))
        out["param_err_max"] = float(max(out[f"param_rel_{k}"]
                                         for k in self.param_names))

        u_d = self.predict(self.Z_data)
        out["data_rmse"] = float(np.sqrt(np.mean((u_d - self.u_obs) ** 2)))
        return out

    # ---------------- 增点 ----------------
    def _enrich(self) -> None:
        cfg = self.cfg
        if cfg.strategy == "fixed":
            return
        pool = self.points.candidate_pool(cfg.candidate_pool)
        r = self.residual_field(pool)
        if cfg.strategy == "residual":
            sel = select_topm(np.abs(r), cfg.add_per_round)
        elif cfg.strategy == "uniform":
            sel = self.points.rng.choice(pool.shape[0],
                                         size=min(cfg.add_per_round, pool.shape[0]),
                                         replace=False)
        else:
            raise KeyError(f"未知策略 {cfg.strategy!r}")
        self.points.add_interior(pool[sel])

        if cfg.add_ic_per_round > 0:
            X = self.points.rng.random((4000, 2))
            Zc = np.column_stack([X, np.zeros(4000)])
            e = np.abs(self.predict(Zc)
                       - np.asarray(self.problem.ic(X), dtype=float).ravel())
            self.points.add_ic(X[select_topm(e, cfg.add_ic_per_round)])

        for axis, value in self.problem.faces():
            if cfg.add_bc_per_round <= 0:
                break
            X = self.points.rng.random((4000, 2))
            X[:, axis] = value
            Zc = np.column_stack([X, self.points.rng.random(4000) * cfg.T])
            e = np.abs(self.predict(Zc)
                       - np.asarray(self.problem.bc(Zc), dtype=float).ravel())
            sel = select_topm(e, cfg.add_bc_per_round)
            self.points.add_bc((axis, value), Zc[sel])

    # ---------------- 主循环 ----------------
    def run(self) -> List[Dict]:
        cfg = self.cfg
        self.Z_data_t = self._t(self.Z_data)
        self.u_obs_t = self._t(self.u_obs).reshape(-1)

        if cfg.use_pde and (cfg.w_ic is None or cfg.w_bc is None):
            parts = self._loss_parts(self._tensors())
            lp = float(parts["pde"].detach())
            if cfg.w_ic is None:
                self.w_ic = float(np.clip(lp / max(float(parts["ic"]), 1e-12), 0.1, 1e3))
            if cfg.w_bc is None:
                self.w_bc = float(np.clip(lp / max(float(parts["bc"]), 1e-12), 0.1, 1e3))
            if cfg.verbose:
                print(f"    [权重标定] J_pde={lp:.3e}  J_ic={float(parts['ic']):.3e}  "
                      f"J_bc={float(parts['bc']):.3e}  J_data={float(parts['data']):.3e}"
                      f"  ->  w_ic={self.w_ic:.3g}  w_bc={self.w_bc:.3g}")

        info = crlb(self.problem, self.Z_data, self.sigma)
        self.cond_fisher = info.pop("cond_fisher")
        self.crlb = info
        if cfg.verbose:
            print(f"    [可辨识性] cond(F)={self.cond_fisher:.2e}, CRLB(rel): "
                  + ", ".join(f"{k}={self.crlb['crlb_rel_' + k]:.2%}"
                              for k in self.param_names))

        src_params = [p for p in self.params.parameters() if p.requires_grad]

        for rnd in range(cfg.rounds + 1):
            t0 = time.time()
            data = self._tensors()
            n_int, n_ic, n_bc = (self.points.n_interior, self.points.n_ic,
                                 self.points.n_bc)

            lr = cfg.lr * (cfg.lr_gamma ** rnd)
            groups = [{"params": self.model.parameters(), "lr": lr}]
            if src_params:
                groups.append({"params": src_params, "lr": cfg.lr_param})
            opt = torch.optim.Adam(groups)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, cfg.epochs_per_round), eta_min=lr * 0.02)

            last: Dict[str, float] = {}
            for _ in range(cfg.epochs_per_round):
                opt.zero_grad(set_to_none=True)
                parts = self._loss_parts(data)
                loss = self._total(parts)
                loss.backward()
                opt.step()
                sched.step()
                last = {k: float(v.detach()) for k, v in parts.items()}

            ev = self.evaluate()
            r_val = self.residual_field(self.val_pts)
            share = self._grad_share() if self.params.trainable_names() else {}
            frac_layer = self.points.layer_fraction(self.problem)
            frac_ridge = self.points.ridge_fraction(self.problem)

            rec = {"round": rnd, "n_int": n_int, "n_ic": n_ic, "n_bc": n_bc,
                   "N": n_int + n_ic + n_bc,
                   "loss_pde": last.get("pde", float("nan")),
                   "loss_ic": last.get("ic", float("nan")),
                   "loss_bc": last.get("bc", float("nan")),
                   "loss_data": last.get("data", float("nan")),
                   "loss_total": (last.get("pde", float("nan"))
                                  + self.w_ic * last.get("ic", float("nan"))
                                  + self.w_bc * last.get("bc", float("nan"))
                                  + self.w_data * last.get("data", float("nan"))),
                   "rms_res_uniform": float(np.sqrt(np.mean(r_val ** 2))),
                   "frac_layer": frac_layer, "frac_ridge": frac_ridge,
                   "grad_share_layer": share,
                   "wall": time.time() - t0}
            rec.update(ev)
            self.history.append(rec)

            if cfg.verbose:
                ph = self.params.numpy_values()
                desc = " ".join(f"{k}={ph[k]:.4f}" for k in self.param_names)
                print(f"    round {rnd:2d}: N={rec['N']:6d} {desc} | "
                      f"err={rec['param_err_max']:.2%} src={rec['src_rel_l2']:.2%} "
                      f"u_max={rec['max_err']:.2e} layer={frac_layer:.3f} "
                      f"({rec['wall']:.1f}s)")

            if rnd == cfg.rounds:
                self.final_points = self.points.interior.copy()
            else:
                self._enrich()

        return self.history

    # ---------------- 保存 ----------------
    def save(self, outdir: str, extra: Optional[Dict] = None) -> None:
        os.makedirs(outdir, exist_ok=True)
        cfg_d = self.cfg.to_dict()
        cfg_d["w_ic_used"] = self.w_ic
        cfg_d["w_bc_used"] = self.w_bc
        cfg_d["sigma"] = self.sigma
        cfg_d["n_data"] = int(self.Z_data.shape[0])
        with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg_d, fh, ensure_ascii=False, indent=2)
        with open(os.path.join(outdir, "history.json"), "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, ensure_ascii=False, indent=2)

        p_hat = self.params.numpy_values()
        p_true = self.problem.truth_dict
        summary = {
            "truth": p_true,
            "estimate": p_hat,
            "param_rel_err": {k: abs(p_hat[k] - p_true[k]) / max(abs(p_true[k]), 1e-12)
                              for k in self.param_names},
            "final": self.history[-1],
            "crlb": self.crlb,
            "cond_fisher": self.cond_fisher,
            "sigma": self.sigma,
        }
        if extra:
            summary.update(extra)
        with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

        np.savez(os.path.join(outdir, "data.npz"),
                 Z_data=self.Z_data, u_obs=self.u_obs, points=self.final_points)
        torch.save({"model": self.model.state_dict(),
                    "params": self.params.state_dict(),
                    "cfg": self.cfg.to_dict()},
                   os.path.join(outdir, "state.pt"))


def run_fixed_baseline(cfg: Case5Config) -> InversePINN:
    """均匀配点对照：总点数与自适应跑结束时相同。"""
    c = copy.deepcopy(cfg)
    c.n_int0 = cfg.n_int0 + cfg.rounds * cfg.add_per_round
    c.n_ic0 = cfg.n_ic0 + cfg.rounds * cfg.add_ic_per_round
    c.n_bc0 = cfg.n_bc0 + cfg.rounds * cfg.add_bc_per_round
    c.strategy = "fixed"
    c.rounds = 0
    c.epochs_per_round = cfg.epochs_per_round * (cfg.rounds + 1)
    c.tag = cfg.tag + "_fixed"
    s = InversePINN(c)
    s.run()
    return s
