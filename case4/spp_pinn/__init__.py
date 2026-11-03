"""Case 4：**非矩形区域**上的残差驱动自适应配点 PINN。

与 ``空间1d`` / ``空间2d`` 两个包同一套算法，只把"区域"从矩形换成了
一条光滑闭曲线围成的三瓣花形区域：

    domain.py   区域定义、弧形边界采样、近似有符号距离、面积/周长
    case4.py    Case 4 的制造解 u* 与闭式右端项 f
    general.py  非矩形点集 + 求解器（内部点/初值面/整条曲边界）
    model.py    全连接网络（输入归一化支持任意包围盒）
    sampling.py Halton 与选点规则
"""

from .problems import PI, _xp
from . import domain
from .case4 import FlowerLayer2D, GeneralProblem, make_problem
from .model import MLP
from .sampling import halton, select_distribution, select_topm
from .general import (
    AdaptivePINNGeneral,
    GeneralCollocationSet,
    GeneralConfig,
    describe_device,
    knn_quadrature_weights,
    pick_device,
    run_reference,
)

__all__ = [
    "PI", "_xp", "domain",
    "GeneralProblem", "FlowerLayer2D", "make_problem",
    "MLP", "halton", "select_topm", "select_distribution",
    "knn_quadrature_weights", "GeneralCollocationSet", "GeneralConfig",
    "AdaptivePINNGeneral", "run_reference", "pick_device", "describe_device",
]
