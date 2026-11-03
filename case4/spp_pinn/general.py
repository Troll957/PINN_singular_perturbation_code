r"""非矩形区域上的自适应配点 PINN（与 ``空间2d/spp_pinn/nd.py`` 的算法逐项一致）。

与矩形版本的区别只有"几何"这一层，算法本身不动：

=====================  =================================  ================================
项目                   矩形版 (Case 2/3)                  非矩形版 (Case 4)
=====================  =================================  ================================
内部点                 单位正方形内 Halton                  外接盒内 Halton + **拒绝采样**
初值点                 单位正方形内均匀                     外接盒内均匀 + 拒绝采样
边界点                 $2d$ 个面，各面均匀采样              整条曲边界，按**弧长**均匀采样
边界面测度             $T$（面长为 1）                      $\lvert\partial\Omega\rvert T$
区域测度               $T$（面积为 1）                      $\lvert\Omega\rvert T$
求积权重               kNN, 指数 = 3（时空）                内部同左；边界面/初值面指数 = 2
误差评估               $n^d\times n$ 张量网格              张量网格 **掩掉区域外的点**
层内判定               $x_1\le4\sqrt\varepsilon$             $\psi/\lvert\nabla\psi\rvert\le4\sqrt\varepsilon$
=====================  =================================  ================================

另外评估多了一层"边界法向"采样：沿外法向以 $4\sqrt\varepsilon$ 为尺度取若干层点，
因为 $n=161$ 的张量网格在 $\varepsilon=10^{-6}$ 时（$h\gg\sqrt\varepsilon$）根本看不见层。
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from .case4 import GeneralProblem, make_problem
from .model import MLP
from .sampling import halton, select_distribution, select_topm


# ======================================================================
# 1. kNN 求积权重（多一个 exponent 参数）
# ======================================================================
def knn_quadrature_weights(coords: np.ndarray, measure: float,
                           n_neighbors: int = 4, exponent: Optional[int] = None,
                           chunk: int = 2048) -> np.ndarray:
    r"""$\omega_i\propto\bar r_i^{D}$，$\sum_i\omega_i=\text{measure}$。

    ``exponent`` 是**流形维数** $D$，不是坐标维数。内部点（时空三维体）取
    $D=3$；边界面/初值面是嵌入三维空间的二维曲面，必须取 $D=2$，否则权重
    会系统性偏大。矩形版靠"丢掉法向那一列"隐式地做到这一点；曲边界没有
    可丢的列，所以这里显式传。
    """
    P = np.ascontiguousarray(coords, dtype=float)
    N, Dc = P.shape
    D = Dc if exponent is None else int(exponent)
    if N <= 1:
        return np.full(N, measure)
    k = int(min(max(1, n_neighbors), N - 1))
    rbar = np.empty(N)
    for s0 in range(0, N, chunk):
        c = P[s0:s0 + chunk]
        d2 = ((c[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        part = np.partition(d2, k, axis=1)[:, 1:k + 1]
        rbar[s0:s0 + chunk] = np.sqrt(part).mean(axis=1)
    w = rbar ** D
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0:
        return np.full(N, measure / N)
    return w / tot * measure


# ======================================================================
# 2. 点集（非矩形区域）
# ======================================================================
class GeneralCollocationSet:
    """内部点 / 初值面 / 整条曲边界，三组点。"""

    def __init__(self, problem: GeneralProblem, T: float = 1.0, seed: int = 0):
        self.problem = problem
        self.T = float(T)
        self.rng = np.random.default_rng(seed)
        self.lo, self.hi = problem.lo, problem.hi
        self.interior = np.zeros((0, 3))
        self.ic = np.zeros((0, 3))
        self.bc = np.zeros((0, 3))          # 整条边界，单一一组点

    # ---------------- 采样工具 ----------------
    def _box_pts(self, n: int, offset: int = 0) -> np.ndarray:
        """外接盒内的 Halton 点 (n,2)。"""
        u = halton(n, 2, offset=offset)
        return self.lo + u * (self.hi - self.lo)

    def sample_interior(self, n: int, t_random: bool = True) -> np.ndarray:
        """区域内 n 个点（Halton + 拒绝采样）。返回 (n,3)。"""
        out, offset, need = [], 0, int(n)
        while need > 0:
            m = max(4 * need, 512)
            X = self._box_pts(m, offset)
            offset += m
            X = X[self.problem.inside(X)]
            if len(X) == 0:
                continue
            t = (self.rng.random(len(X)) * self.T if t_random
                 else np.zeros(len(X)))
            out.append(np.column_stack([X, t])[:need])
            need -= min(need, len(X))
        return np.vstack(out)

    def sample_uniform_interior(self, n: int, t_random: bool = True) -> np.ndarray:
        """区域内 n 个均匀随机点（用于候选池）。"""
        out, need = [], int(n)
        while need > 0:
            X = self.rng.uniform(self.lo, self.hi, size=(4 * need, 2))
            X = X[self.problem.inside(X)]
            if len(X) == 0:
                continue
            t = (self.rng.random(len(X)) * self.T if t_random
                 else np.zeros(len(X)))
            out.append(np.column_stack([X, t])[:need])
            need -= min(need, len(X))
        return np.vstack(out)

    def sample_boundary(self, n: int, uniform: bool = False) -> np.ndarray:
        """边界上 n 个点，配随机时刻。"""
        X = self.problem.boundary_sample(n, self.rng, uniform=uniform)
        return np.column_stack([X, self.rng.random(n) * self.T])

    # ---------------- 初始化 ----------------
    def initialise(self, n_interior: int, n_ic: int, n_bc: int,
                   method: str = "halton") -> None:
        if method == "halton":
            self.interior = self.sample_interior(n_interior)
        else:
            self.interior = self.sample_uniform_interior(n_interior)
        self.ic = self.sample_uniform_interior(n_ic, t_random=False)
        self.ic[:, 2] = 0.0
        self.bc = self.sample_boundary(n_bc)

    # ---------------- 增点 ----------------
    def add_interior(self, pts: np.ndarray) -> int:
        self.interior = np.concatenate([self.interior, np.atleast_2d(pts)], 0)
        return len(pts)

    def add_ic(self, pts: np.ndarray) -> int:
        pts = np.atleast_2d(pts).copy()
        pts[:, 2] = 0.0
        self.ic = np.concatenate([self.ic, pts], 0)
        return len(pts)

    def add_bc(self, pts: np.ndarray) -> int:
        self.bc = np.concatenate([self.bc, np.atleast_2d(pts)], 0)
        return len(pts)

    # ---------------- 查询 ----------------
    @property
    def n_interior(self) -> int:
        return self.interior.shape[0]

    @property
    def n_ic(self) -> int:
        return self.ic.shape[0]

    @property
    def n_bc(self) -> int:
        return self.bc.shape[0]

    @property
    def total(self) -> int:
        return self.n_interior + self.n_ic + self.n_bc

    def candidate_pool(self, m: int, eps=None,
                       band: float = 4.0, band_frac: float = 0.5) -> np.ndarray:
        r"""候选池。``eps`` 给定时按 sqrt(eps) 分级（论文 §failure mode 的建议）。

        均匀候选池在 eps=1e-6 时几乎取不到层内的点：层带面积占比只有
        4*sqrt(eps)*|dOmega|/|Omega| ≈ 2.6e-4，2e4 个候选点里只有约 5 个落在层里
        ——**指示量本身先失效**，而不是采样不足。分级池把 ``band_frac`` 比例的点
        直接撒在 0 <= d <= band*sqrt(eps) 的边界带内。
        """
        if eps is None:
            return self.sample_uniform_interior(m)
        from . import domain as _dom
        n_band = int(round(band_frac * m))
        Xb = self.problem.boundary_sample(n_band, self.rng)
        nrm = _dom.outward_normal(Xb)
        d = (1.0 - self.rng.random(n_band)) * band * np.sqrt(eps)
        X = Xb - nrm * d[:, None]
        X = X[self.problem.inside(X)]                    # 非凸区域的保险
        t = self.rng.random(len(X)) * self.T
        band_pts = np.column_stack([X, t])
        rest = self.sample_uniform_interior(max(0, m - len(band_pts)))
        return np.vstack([band_pts, rest])

    def boundary_candidates(self, m: int) -> np.ndarray:
        return self.sample_boundary(m)

    def layer_fraction(self, eps: float, factor: float = 4.0) -> float:
        return float(np.mean(self.problem.inside_layer(self.interior, factor)))


# ======================================================================
# 3. 求解器
# ======================================================================
def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        dev = torch.device(name)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定了 cuda，但 torch.cuda.is_available() 为 False")
        return dev
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def describe_device(dev: torch.device) -> str:
    if dev.type == "cuda":
        i = 0 if dev.index is None else dev.index
        try:
            return f"cuda:{i} ({torch.cuda.get_device_name(i)})"
        except Exception:                                   # noqa: BLE001
            return f"cuda:{i}"
    return str(dev)


@dataclass
class GeneralConfig:
    problem: str = "flower"
    eps: float = 1e-3
    T: float = 1.0
    b_scale: float = 1.0     # b = b_scale*(x,y)；见 case4.py 与 README 第 6 节
    hidden: tuple = (64, 64, 64)
    activation: str = "tanh"
    # 采样
    n_int0: int = 2048
    n_ic0: int = 256
    n_bc0: int = 384
    init_method: str = "halton"
    # 自适应
    rounds: int = 24
    epochs_per_round: int = 4000
    add_per_round: int = 300
    strategy: str = "residual"
    candidate_pool: int = 20000
    pool_mode: str = "uniform"     # uniform | graded
    pool_band_frac: float = 0.5    # graded 时落在边界带内的比例
    pool_band: float = 4.0         # 边界带宽度 = pool_band*sqrt(eps)
    beta: float = 1.0
    add_ic_per_round: int = 32
    add_bc_per_round: int = 48
    # 权重
    weighting: str = "knn"
    knn_neighbors: int = 4
    # 优化
    lr: float = 2e-3
    lr_schedule: str = "cosine"
    lr_gamma: float = 0.9
    weight_mode: str = "init"
    loss_norm: str = "l2"
    w_ic: Optional[float] = None
    w_bc: Optional[float] = None
    # 其他
    seed: int = 0
    device: str = "auto"
    verbose: bool = True
    eval_n: int = 97
    eval_nt: int = 33
    callback: object = None

    def to_dict(self):
        d = asdict(self)
        d["hidden"] = tuple(self.hidden)
        d["callback"] = None
        return d


class AdaptivePINNGeneral:
    """非矩形区域上的残差驱动自适应配点 PINN。"""

    def __init__(self, cfg: GeneralConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = pick_device(cfg.device)
        self.problem = make_problem(cfg.problem, cfg.eps, cfg.T,
                                    b_scale=cfg.b_scale)
        self.dim, self.D = 2, 3
        self.area, self.perimeter = self.problem.area, self.problem.perimeter

        self.points = GeneralCollocationSet(self.problem, T=cfg.T, seed=cfg.seed)
        self.points.initialise(cfg.n_int0, cfg.n_ic0, cfg.n_bc0, cfg.init_method)
        self.model = MLP(hidden=tuple(cfg.hidden), in_dim=self.D, T=cfg.T,
                         activation=cfg.activation,
                         lo=self.problem.lo, hi=self.problem.hi).to(self.device)
        self.val_pts = self.points.candidate_pool(20000)
        self.history: List[Dict] = []
        self.w_ic = 1.0 if cfg.w_ic is None else float(cfg.w_ic)
        self.w_bc = 1.0 if cfg.w_bc is None else float(cfg.w_bc)
        self.weights_fixed = cfg.weight_mode == "fixed"
        self.final_points = self.points.interior
        self.final_abs_res = np.zeros(0)

    # ---------------- 网络与残差 ----------------
    def _u(self, Z: torch.Tensor) -> torch.Tensor:
        return self.model(Z)

    def pde_residual(self, Z: torch.Tensor, create_graph: bool = True,
                     aux=None) -> torch.Tensor:
        """r = u_t - εΔu + b·∇u + q u - f。

        ``aux`` 是可选的三元组 ``(b, q, f)``，由 ``_aux`` 在每轮开始时算一次。
        这三个量只依赖坐标、不依赖网络参数，逐 epoch 重算纯属浪费——
        实测可省掉约三分之二的单步时间。
        """
        prob = self.problem
        Z = Z.clone().detach().requires_grad_(True)
        u = self._u(Z)
        g = torch.autograd.grad(u, Z, torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
        u_t = g[:, 2:3]
        lap = torch.zeros_like(u)
        for j in range(2):
            gj = torch.autograd.grad(g[:, j:j + 1], Z, torch.ones_like(u),
                                     create_graph=create_graph, retain_graph=True)[0]
            lap = lap + gj[:, j:j + 1]
        if aux is None:
            b = prob.convection(Z).to(Z.dtype)
            q = prob.reaction(Z)
            f = prob.forcing(Z).reshape(-1, 1)
        else:
            b, q, f = aux
        adv = b[:, 0:1] * g[:, 0:1] + b[:, 1:2] * g[:, 1:2]
        return u_t - self.cfg.eps * lap + adv + q * u - f

    def _aux(self, P: np.ndarray):
        """在给定点集上预计算 (b, q, f)，三者都与网络参数无关。"""
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a),
                                      dtype=torch.float32, device=self.device)
        prob = self.problem
        Z = t(P)
        b = prob.convection(Z).detach()
        q = prob.reaction(Z).detach()
        f = prob.forcing(Z).reshape(-1, 1).detach()
        return b, q, f

    def residual_field(self, pts: np.ndarray, batch: int = 20000) -> np.ndarray:
        out = []
        for k in range(0, len(pts), batch):
            Z = torch.as_tensor(np.ascontiguousarray(pts[k:k + batch]),
                                dtype=torch.float32, device=self.device)
            out.append(self.pde_residual(Z, create_graph=False)
                       .detach().cpu().numpy().ravel())
        return np.concatenate(out) if out else np.zeros(0)

    def predict(self, pts: np.ndarray, batch: int = 50000) -> np.ndarray:
        self.model.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(pts), batch):
                Z = torch.as_tensor(np.ascontiguousarray(pts[k:k + batch]),
                                    dtype=torch.float32, device=self.device)
                out.append(self._u(Z).detach().cpu().numpy().ravel())
        self.model.train()
        return np.concatenate(out)

    # ---------------- 权重与损失 ----------------
    def _tensors(self) -> Dict:
        dev, cfg = self.device, self.cfg
        c = self.points
        knn = cfg.weighting == "knn"

        def t_(a):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)

        w_int = (knn_quadrature_weights(c.interior, self.area * cfg.T,
                                        cfg.knn_neighbors, exponent=3)
                 if knn else np.full(c.n_interior, self.area * cfg.T / c.n_interior))
        w_ic = (knn_quadrature_weights(c.ic[:, :2], self.area,
                                       cfg.knn_neighbors, exponent=2)
                if knn else np.full(c.n_ic, self.area / c.n_ic))
        w_bc = (knn_quadrature_weights(c.bc, self.perimeter * cfg.T,
                                       cfg.knn_neighbors, exponent=2)
                if knn else np.full(c.n_bc, self.perimeter * cfg.T / c.n_bc))
        return {"int": t_(c.interior), "w_int": t_(w_int).reshape(-1, 1),
                "aux": self._aux(c.interior),
                "ic": t_(c.ic), "w_ic": t_(w_ic).reshape(-1, 1),
                "bc": t_(c.bc), "w_bc": t_(w_bc).reshape(-1, 1)}

    def _loss_parts(self, data) -> Dict[str, torch.Tensor]:
        prob, T = self.problem, self.cfg.T
        knn = self.cfg.weighting == "knn"
        _p = (lambda z: z ** 2) if self.cfg.loss_norm == "l2" else (lambda z: z.abs())
        m_int, m_ic, m_bc = self.area * T, self.area, self.perimeter * T

        r = self.pde_residual(data["int"], create_graph=True, aux=data["aux"])
        parts = {"pde": ((data["w_int"] * _p(r)).sum() / m_int) if knn else _p(r).mean()}

        e_ic = _p(self._u(data["ic"]) - prob.ic(data["ic"][:, :2]).reshape(-1, 1))
        parts["ic"] = ((data["w_ic"] * e_ic).sum() / m_ic) if knn else e_ic.mean()

        e_bc = _p(self._u(data["bc"]) - prob.bc(data["bc"]).reshape(-1, 1))
        parts["bc"] = ((data["w_bc"] * e_bc).sum() / m_bc) if knn else e_bc.mean()
        return parts

    def _total(self, parts) -> torch.Tensor:
        return parts["pde"] + self.w_ic * parts["ic"] + self.w_bc * parts["bc"]

    # ---------------- 评估 ----------------
    def eval_points(self, n: Optional[int] = None,
                    nt: Optional[int] = None, layer: bool = False):
        """区域内的评估点：(张量网格掩掉外部) ∪ (可选的边界法向层点)。"""
        n = int(n or self.cfg.eval_n)
        nt = int(nt or self.cfg.eval_nt)
        lo, hi = self.problem.lo, self.problem.hi
        gx = np.linspace(lo[0], hi[0], n)
        gy = np.linspace(lo[1], hi[1], n)
        gt = np.linspace(0.0, self.cfg.T, nt)
        X1, X2, TT = np.meshgrid(gx, gy, gt, indexing="ij")
        Zg = np.column_stack([X1.ravel(), X2.ravel(), TT.ravel()])
        Zg = Zg[self.problem.inside(Zg[:, :2])]
        if not layer:
            return Zg

        # 边界法向：M 个边界点 × J 层 offset × nt 个时刻
        M, J = 600, 10
        Xb = self.problem.boundary_sample(M, np.random.default_rng(12345),
                                          uniform=True)
        nrm = self._outward_normal(Xb)
        d = (np.arange(1, J + 1) / J) * 4.0 * self.problem.sqrt_eps
        pts = []
        for dj in d:
            Xl = Xb - nrm * dj
            for tv in gt[::max(1, nt // 21)]:
                pts.append(np.column_stack([Xl, np.full(M, tv)]))
        return np.vstack([Zg, np.vstack(pts)])

    @staticmethod
    def _outward_normal(X: np.ndarray) -> np.ndarray:
        from . import domain
        return domain.outward_normal(X)

    def evaluate(self, n: Optional[int] = None, nt: Optional[int] = None,
                 layer_points: bool = False) -> Dict[str, float]:
        n = int(n or self.cfg.eval_n)
        nt = int(nt or self.cfg.eval_nt)
        Zg = self.eval_points(n, nt, layer=False)
        err_g, u_g = self._errors(Zg)
        out = {"max_grid": float(err_g.max()),
               "rel_l2": float(np.linalg.norm(err_g) / max(np.linalg.norm(u_g), 1e-30)),
               "n_grid": int(len(Zg))}
        if layer_points:
            Zl = self.eval_points(n, nt, layer=True)
            err_l, _ = self._errors(Zl)
            out["max_layer"] = float(err_l.max())
            out["max_err"] = float(max(out["max_grid"], out["max_layer"]))
        else:
            out["max_layer"] = out["max_grid"]
            out["max_err"] = out["max_grid"]
        return out

    def _errors(self, Z: np.ndarray):
        u_pred = self.predict(Z)
        u_true = np.asarray(self.problem.exact(Z), dtype=float).reshape(-1)
        return np.abs(u_pred - u_true), u_true

    # ---------------- 自适应增点 ----------------
    def _select(self, values: np.ndarray, m: int) -> np.ndarray:
        st = self.cfg.strategy
        if st == "residual":
            return select_topm(np.abs(values), m)
        if st == "distribution":
            return select_distribution(values, m, self.cfg.beta, self.points.rng)
        if st == "uniform":
            return self.points.rng.choice(values.size,
                                          size=int(min(m, values.size)), replace=False)
        raise KeyError(f"未知策略 {st!r}")

    def _enrich(self) -> None:
        cfg = self.cfg
        if cfg.strategy == "fixed":
            return
        eps_pool = cfg.eps if cfg.pool_mode == "graded" else None
        pool = self.points.candidate_pool(cfg.candidate_pool, eps=eps_pool,
                                          band=cfg.pool_band,
                                          band_frac=cfg.pool_band_frac)
        r = self.residual_field(pool)
        self.points.add_interior(pool[self._select(r, cfg.add_per_round)])

        if cfg.add_ic_per_round > 0:
            Zc = self.points.sample_uniform_interior(4000, t_random=False)
            Zc[:, 2] = 0.0
            e = np.abs(self.predict(Zc)
                       - np.asarray(self.problem.ic(Zc[:, :2]), dtype=float))
            self.points.add_ic(Zc[self._select(e, cfg.add_ic_per_round)])

        if cfg.add_bc_per_round > 0:
            Zc = self.points.boundary_candidates(4000)
            e = np.abs(self.predict(Zc)
                       - np.asarray(self.problem.bc(Zc), dtype=float))
            self.points.add_bc(Zc[self._select(e, cfg.add_bc_per_round)])

    # ---------------- 主循环 ----------------
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
                print(f"    [权重标定] J_pde={lp:.3e} J_ic={li:.3e} J_bc={lb:.3e}"
                      f" -> w_ic={self.w_ic:.3g}, w_bc={self.w_bc:.3g}")

        for rnd in range(cfg.rounds + 1):
            t0 = time.time()
            data = self._tensors()
            n_int, n_ic, n_bc = (self.points.n_interior, self.points.n_ic,
                                 self.points.n_bc)
            lr = cfg.lr * (cfg.lr_gamma ** rnd)
            opt = torch.optim.Adam(self.model.parameters(), lr=lr)
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
            frac_layer = self.points.layer_fraction(cfg.eps)
            N_round = n_int + n_ic + n_bc
            if rnd < cfg.rounds:
                self._enrich()

            rec = {"round": rnd, "n_int": n_int, "n_ic": n_ic, "n_bc": n_bc,
                   "N": N_round, "max_err": ev["max_err"],
                   "max_grid": ev["max_grid"], "rel_l2": ev["rel_l2"],
                   "loss_pde": last.get("pde", float("nan")),
                   "loss_ic": last.get("ic", float("nan")),
                   "loss_bc": last.get("bc", float("nan")),
                   "rms_res_uniform": float(np.sqrt(np.mean(r_val ** 2))),
                   "frac_layer": frac_layer, "wall": time.time() - t0}
            self.history.append(rec)
            if cfg.callback is not None:
                cfg.callback(rec)
            if cfg.verbose:
                print(f"    round {rnd:3d}: N={rec['N']:6d} max={rec['max_grid']:.3e} "
                      f"rel={rec['rel_l2']:.3e} rms_r={rec['rms_res_uniform']:.2e} "
                      f"layer={rec['frac_layer']:.3f} ({rec['wall']:.1f}s)")
            if rnd == cfg.rounds:
                self.final_points = self.points.interior.copy()
                self.final_abs_res = np.abs(self.residual_field(self.final_points))
        self.final_eval = self.evaluate(n=201, nt=81, layer_points=True)
        return self.history


def run_reference(cfg: GeneralConfig, n_interior: int, total_epochs: int,
                  verbose: bool = False) -> AdaptivePINNGeneral:
    """固定均匀参照组：点数相同、训练步数相同，不做自适应增点。"""
    c = copy.deepcopy(cfg)
    c.n_int0 = int(n_interior)
    c.strategy = "fixed"
    c.rounds = 0
    c.epochs_per_round = int(total_epochs)
    c.verbose = verbose
    s = AdaptivePINNGeneral(c)
    s.run()
    return s
