"""用于求解奇异摄动问题的全连接网络。

与 ``空间2d/spp_pinn/model.py`` 的唯一区别：输入归一化用的区间从硬编码的
``[0,1]`` 改成显式的 ``lo`` / ``hi``，因为 Case 4 的区域不在单位正方形里。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Sine(nn.Module):
    """正弦激活 (SIREN 风格)。"""

    def forward(self, x):
        return torch.sin(x)


_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "sin": Sine,
    "softplus": nn.Softplus,
    "gelu": nn.GELU,
}


class MLP(nn.Module):
    """全连接网络 u_theta(x, t)。

    输入先仿射映射到 ``[-1,1]^{d+1}``（有利于 tanh 网络的训练），最后一层线性。
    ``lo`` / ``hi`` 是空间坐标的逐维下界与上界（默认 ``[0,1]^d``）；时间始终是
    ``[0, T]``。
    """

    def __init__(
        self,
        hidden=(64, 64, 64),
        in_dim: int = 2,
        out_dim: int = 1,
        T: float = 1.0,
        activation: str = "tanh",
        lo=None,
        hi=None,
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise KeyError(f"未知激活函数 {activation!r}")
        self.T = float(T)
        d = in_dim - 1
        lo = [0.0] * d if lo is None else list(lo)
        hi = [1.0] * d if hi is None else list(hi)
        if len(lo) != d or len(hi) != d:
            raise ValueError("lo/hi 的长度必须等于空间维数 d")
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(hi, dtype=torch.float32))

        layers = []
        cur = in_dim
        for h in hidden:
            layers.append(nn.Linear(cur, h))
            layers.append(_ACTIVATIONS[activation]())
            cur = h
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        d = Z.shape[1] - 1
        xs = Z[:, :d]
        t = Z[:, d:d + 1]
        zs = 2.0 * (xs - self.lo) / (self.hi - self.lo) - 1.0
        z = torch.cat([zs, 2.0 * t / self.T - 1.0], dim=1)
        return self.net(z)
