"""对比 CPU 与 GPU 的实际训练速度（毫秒/步）——二维空间版本。

二维每个 epoch 要算 d 次二阶导（Laplace），算子比一维多，
所以显卡的收益通常比一维明显，但仍然要用数据说话。

用法::

    arch -arm64 ../../.venv/bin/python benchmark_device.py
    arch -arm64 ../../.venv/bin/python benchmark_device.py --points 8192 --width 128
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dsh_mpl")

import torch  # noqa: E402

from spp_pinn.nd import (AdaptivePINNND, NDConfig,  # noqa: E402
                         pick_device, describe_device)


def time_solver(dev: torch.device, points: int, width: int, epochs: int) -> float:
    """用真实的损失 + Laplace 二阶导，测「毫秒/训练步」。"""
    cfg = NDConfig(problem="corner2d", eps=1e-3, n_int0=points, rounds=0,
                   epochs_per_round=epochs, hidden=(width,) * 3,
                   device=str(dev), verbose=False, eval_n=11)
    s = AdaptivePINNND(cfg)
    data = s._tensors()
    opt = torch.optim.Adam(s.model.parameters(), lr=2e-3)

    def step():
        opt.zero_grad(set_to_none=True)
        s._total(s._loss_parts(data)).backward()
        opt.step()

    for _ in range(10):
        step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(epochs):
        step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / epochs * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU vs GPU 训练速度对比 (2D)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--points", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--devices", default=None,
                    help="逗号分隔, 例如 cpu,cuda; 默认自动列出 cpu 与可用的 cuda")
    ap.add_argument("--torch-threads", type=int, default=0)
    args = ap.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    if args.devices:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    else:
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    print(f"torch {torch.__version__} | CUDA 可用: {torch.cuda.is_available()} | "
          f"CPU 线程: {torch.get_num_threads()}")
    for d in devices:
        print(f"  {d:6s} -> {describe_device(pick_device(d))}")

    grid = ([(args.points, args.width)] if args.points and args.width
            else [(1024, 64), (2048, 64), (4096, 128), (8192, 128)])
    print(f"\n{'配点数':>7} {'宽度':>5} " + "".join(f"{d+' [ms/步]':>16}" for d in devices))
    for pts, w in grid:
        row = [time_solver(pick_device(d), pts, w, args.epochs) for d in devices]
        best = min(row)
        print(f"{pts:>7d} {w:>5d} " + "".join(f"{v:16.3f}" for v in row))
        if len(row) > 1:
            print(f"{'':>13} 相对最快: " + " ".join(f"{v/best:.2f}x" for v in row))
    print("\n读法: 毫秒/步, 越小越快。cuda 若比 cpu 慢, 说明该规模下显卡不划算——"
          "\n增大 --points / --width 或加长训练再看。")


if __name__ == "__main__":
    main()
