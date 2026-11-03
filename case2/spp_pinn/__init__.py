"""二维空间 + 时间的残差驱动自适应配点 PINN。

主方法见 :mod:`spp_pinn.nd`：先用一组初始配点求解，再根据 PDE 残差 |r|
**在残差大的地方追加采样点**，如此迭代。全程只维护点集，不划分单元。

二维相比一维多出的三件事：

1. 区域是 (0,1)^2 x (0,T]，边界是 4 个面（而不是左右两条边）；
2. 求积权重不能再用 n^D 的离散 Voronoi 网格，改用 k 近邻体积估计
   :func:`knn_quadrature_weights`；
3. Laplace 项要按方向循环求和 d 次二阶导。

模块结构::

    problems.py    一维制造解算例（nd.py 复用其中的 _xp / PI 工具）
    sampling.py    低差异序列、选点规则、一维/二维 Voronoi 权重（供对比）
    model.py       全连接网络（in_dim = d+1，自动适配维数）
    nd.py          高维问题基类、2D 角点层算例、高维点集、kNN 权重、求解器
    plotting.py    绘图
"""

from .problems import PI, _xp

from .sampling import (
    halton,
    select_topm,
    select_distribution,
    select_separated,
    voronoi_weights_1d,
    voronoi_weights_2d,
)

from .model import MLP

from .nd import (
    NDProblem,
    CornerLayer2D,
    make_problem_nd,
    knn_quadrature_weights,
    CollocationSetND,
    NDConfig,
    AdaptivePINNND,
    run_reference_nd,
    pick_device,
    describe_device,
    device_benchmark,
)

__all__ = [
    "PI", "_xp",
    "halton", "select_topm", "select_distribution", "select_separated",
    "voronoi_weights_1d", "voronoi_weights_2d",
    "MLP",
    "NDProblem", "CornerLayer2D", "make_problem_nd", "knn_quadrature_weights",
    "CollocationSetND", "NDConfig", "AdaptivePINNND", "run_reference_nd",
    "pick_device", "describe_device", "device_benchmark",
]
