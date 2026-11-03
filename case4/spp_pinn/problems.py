"""共用的数组后端工具（numpy / torch 双后端）。

只有两个东西：常数 `PI` 和后端选择器 `_xp`。整个算例的解析解与右端项都写成
"先选后端、再算"的形式，这样同一份代码既能给 numpy 评估、也能给 torch 自动微分。
"""

from __future__ import annotations

import numpy as np
import torch

PI = 3.14159265358979323846


def _xp(*arrays):
    """根据输入类型返回数组后端 (torch 或 numpy)。"""
    for a in arrays:
        if isinstance(a, torch.Tensor):
            return torch
    return np
