"""用于求解奇异摄动问题的全连接网络。"""

from __future__ import annotations

import torch
import math

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


class FourierFeatures(nn.Module):
    r"""随机傅里叶特征, 频率谱对数均匀覆盖 $[1,\lambda_{\max}]$。

    只指定**尺度**、不指定位置, 因此与几何无关: 层的存在不体现在特征里,
    只体现在"网络有能力表示 $O(1/\lambda_{\max})$ 的过渡"上。
    """

    def __init__(self, in_dim: int, n_freq: int = 64,
                 lam_max: float = 1.0, T: float = 1.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        U = torch.randn(in_dim, n_freq, generator=g)
        U = U / U.norm(dim=0, keepdim=True)
        lam = torch.exp(torch.rand(n_freq, generator=g) * math.log(max(lam_max, 1.0)))
        self.register_buffer("B", U * lam.unsqueeze(0))
        self.T = float(T)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        d = Z.shape[1] - 1
        z = torch.cat([Z[:, :d], Z[:, d:] / self.T], dim=1)
        p = 2.0 * math.pi * z @ self.B
        return torch.cat([torch.cos(p), torch.sin(p)], dim=1)


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
        sqrt_eps: float = None,
        scaled: str = None,
        feature: str = None,
        n_freq: int = 64,
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise KeyError(f"未知激活函数 {activation!r}")
        self.T = float(T)
        # 可选的"层尺度通道": 额外输入 (2x-1)/sqrt(eps), 使层在输入空间中
        # 宽度变成 O(1), 网络无需学出 ~1/sqrt(eps) 量级的权重。
        self.sqrt_eps = float(sqrt_eps) if sqrt_eps else None
        self.scaled = scaled if self.sqrt_eps is not None else None
        self.ff = None
        layers = []
        if feature == "fourier":
            lam_max = (1.0 / self.sqrt_eps) if self.sqrt_eps else 1.0
            self.ff = FourierFeatures(in_dim, n_freq, lam_max, T, seed=0)
            d = 2 * n_freq
        else:
            d = in_dim + (in_dim - 1) if self.scaled is not None else in_dim
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
        if self.ff is not None:
            return self.net(self.ff(Z))
        d = Z.shape[1] - 1
        xs = Z[:, :d]
        t = Z[:, d:d + 1]
        z = torch.cat([2.0 * xs - 1.0, 2.0 * t / self.T - 1.0], dim=1)
        if self.scaled == "raw":
            z = torch.cat([z, (2.0 * xs - 1.0) / self.sqrt_eps], dim=1)
        elif self.scaled == "tanh":
            z = torch.cat([z, torch.tanh(xs / self.sqrt_eps) - 0.5], dim=1)
        return self.net(z)
