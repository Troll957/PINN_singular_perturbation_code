"""case5：二维（空间）奇异摄动方程的反源问题。

两种算例（``--case``）：

* ``moving``（默认）：源是一条以速度 :math:`v` 扫过区域的薄脊，
  反演 :math:`(A,s_0,v)`；
* ``static``：源固定在 :math:`x=x_s`，反演 :math:`(A,x_s,y_s,\\rho)`。

与 ``spp_pinn`` 主包同一套残差驱动自适应配点算法，唯一的变化是
**源项含未知参数**，需要与网络联合反演。
"""

from .problem import (DEFAULT_BOUNDS, PARAM_LABELS, PARAM_NAMES_ALL,
                      BaseInverse2D, MovingSource2D, MovingTruth,
                      StaticSource2D, StaticTruth, SourceParams, TruthParams,
                      InverseSource2D, crlb, make_problem, sensitivity_matrix)
from .model import MLP
from .quadrature import (Points2D, halton, knn_quadrature_weights, select_topm)
from .solver import (Case5Config, InversePINN, describe_device, make_sensors,
                     pick_device, run_fixed_baseline)

__all__ = [
    "DEFAULT_BOUNDS", "PARAM_LABELS", "PARAM_NAMES_ALL",
    "BaseInverse2D", "MovingSource2D", "MovingTruth",
    "StaticSource2D", "StaticTruth", "SourceParams", "TruthParams",
    "InverseSource2D", "crlb", "make_problem", "sensitivity_matrix",
    "MLP",
    "Points2D", "halton", "knn_quadrature_weights", "select_topm",
    "Case5Config", "InversePINN", "describe_device", "make_sensors",
    "pick_device", "run_fixed_baseline",
]
