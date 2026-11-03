"""对比 CPU 与 GPU 的实际训练速度（毫秒/步）。

小网络（3x64）+ 两三千个点时，显卡常常**不比 CPU 快**：每个 epoch 的算子太小，
kernel 启动开销占主导。正式跑之前先用本脚本量一下，别凭感觉选设备。

用法::

    arch -arm64 ../../.venv/bin/python benchmark_device.py
    arch -arm64 ../../.venv/bin/python benchmark_device.py --points 4096 --width 128
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dsh_mpl")

import torch  # noqa: E402

from spp_pinn import (AdaptivePINNSolver, TrainConfig,  # noqa: E402
                      describe_device, pick_device)
from spp_pinn.model import MLP  # noqa: E402


def time_solver(dev: torch.device, points: int, width: int, epochs: int) -> float:
    """用真实的损失与二阶自动微分，测「毫秒/训练步」。"""
    cfg = TrainConfig(problem="left", eps=1e-4, n_int0=points, rounds=0,
                      epochs_per_round=epochs, hidden=(width,) * 3,
                      device=str(dev), verbose=False)
    s = AdaptivePINNSolver(cfg)
    data = s._tensors()
    opt = torch.optim.Adam(s.model.parameters(), lr=2e-3)

    def step():
        opt.zero_grad(set_to_none=True)
        s._total(s._loss_parts(data)).backward()
        opt.step()

    for _ in range(10):                      # 预热
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
    ap = argparse.ArgumentParser(description="CPU vs GPU 训练速度对比")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--points", type=int, default=None, help="单次测的配点数")
    ap.add_argument("--width", type=int, default=None, help="单次测的网络宽度")
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
            else [(1024, 64), (4096, 64), (4096, 128), (16384, 128)])
    print(f"\n{'配点数':>7} {'宽度':>5} " + "".join(f"{d+' [ms/步]':>16}" for d in devices))
    for pts, w in grid:
        row = []
        for d in devices:
            row.append(time_solver(pick_device(d), pts, w, args.epochs))
        best = min(row)
        cells = "".join(f"{v:16.3f}" for v in row)
        print(f"{pts:>7d} {w:>5d} {cells}")
        if len(row) > 1:
            print(f"{'':>13} 相对最快: " +
                  " ".join(f"{v/best:.2f}x" for v in row))
    print("\n读法: 单位是毫秒/步, 越小越快。若 cuda 一列比 cpu 慢, 说明这个规模下"
          "\n显卡不划算——增大 --points / --width 或加长训练再看。")


if __name__ == "__main__":
    main()
