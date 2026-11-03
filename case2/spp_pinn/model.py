"""用于求解奇异摄动问题的全连接网络。"""

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

    输入 (x, t) 在网络内部先归一化到 [-1, 1] (有利于 tanh 型网络的训练),
    最后一层为线性输出。
    """

    def __init__(
        self,
        hidden=(64, 64, 64),
        in_dim: int = 2,
        out_dim: int = 1,
        T: float = 1.0,
        activation: str = "tanh",
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise KeyError(f"未知激活函数 {activation!r}")
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
        """Glorot (Xavier) 初始化 - tanh 网络的推荐初始化方式。"""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """``Z``: (N, d+1)。前 ``d = in_dim - 1`` 列是空间坐标 $x_1..x_d$，
        最后一列是时间 $t$。全部线性映射到 $[-1,1]$ 后再送入网络。

        一维时 ``Z`` 就是 ``(N, 2)`` 的 ``(x, t)``，与旧版行为完全一致。
        """
        d = Z.shape[1] - 1
        xs = Z[:, :d]
        t = Z[:, d:d + 1]
        z = torch.cat([2.0 * xs - 1.0, 2.0 * t / self.T - 1.0], dim=1)
        return self.net(z)
