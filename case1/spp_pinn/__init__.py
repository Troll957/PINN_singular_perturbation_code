"""残差驱动的自适应配点物理信息神经网络 (PINN)。

求解抛物型奇异摄动问题

    u_t - eps*u_xx + p(x)*u_x + q(x)*u = f(x,t),   (x,t) in (0,1) x (0,T]

主方法（见 :mod:`spp_pinn.solver`）：先用一组初始配点求解，再根据 PDE 残差
|r| **在残差大的地方追加采样点**，如此迭代。全程只维护点集，不划分单元。

模块结构::

    problems.py      算例 (解析解 / 右端项 / 初边值条件)
    sampling.py      自适应点集 (Halton 初始化 / 残差选点 / 增点策略)
    model.py         全连接网络
    solver.py        训练 + 残差评估 + 增点 主循环
    plotting.py      绘图
    mesh.py          [对照] quadtree 各向异性 h-自适应网格
    mesh_variant.py  [对照] 基于上述网格的 h-自适应求解器

高维（空间 2d）版本见同级目录 空间2d/。
"""

from .problems import (
    ParabolicSPP,
    LeftBoundaryLayer,
    RightBoundaryLayer,
    InteriorLayer,
    make_problem,
)
from .sampling import (
    CollocationSet,
    SampleStats,
    halton,
    select_topm,
    select_distribution,
    select_separated,
)
from .model import MLP
from .solver import (AdaptivePINNSolver, TrainConfig, run_reference,
                     pick_device, describe_device, device_benchmark)

__all__ = [
    "ParabolicSPP", "LeftBoundaryLayer", "RightBoundaryLayer", "InteriorLayer",
    "make_problem",
    "CollocationSet", "SampleStats", "halton", "select_topm",
    "select_distribution", "select_separated",
    "MLP", "AdaptivePINNSolver", "TrainConfig", "run_reference",
    "pick_device", "describe_device", "device_benchmark",
]
