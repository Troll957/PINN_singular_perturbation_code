"""全连接网络 u_theta(x, y, t)。"""

from __future__ import annotations

import torch
import torch.nn as nn


class Sine(nn.Module):
    """正弦激活。"""

    def forward(self, x):
        return torch.sin(x)


_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "sin": Sine,
    "gelu": nn.GELU,
    "softplus": nn.Softplus,
}


class MLP(nn.Module):
    """全连接网络 ``u_theta(x,y,t)``。

    输入先归一化到 :math:`[-1,1]^3`（对 tanh 网络的训练有利），
    输出层为线性。最后一维的时间按 ``T`` 缩放。
    """

    def __init__(self, hidden=(64, 64, 64), in_dim: int = 3, out_dim: int = 1,
                 T: float = 1.0, activation: str = "tanh"):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise KeyError(f"未知激活函数 {activation!r}，可选 {sorted(_ACTIVATIONS)}")
        self.T = float(T)
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(_ACTIVATIONS[activation]())
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """``Z``: ``(N,3)``，列为 ``(x, y, t)``。"""
        d = Z.shape[1] - 1
        xs = Z[:, :d]
        t = Z[:, d:d + 1]
        z = torch.cat([2.0 * xs - 1.0, 2.0 * t / self.T - 1.0], dim=1)
        return self.net(z)
