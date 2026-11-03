#!/usr/bin/env python3
"""数值实验: Case 1 (一维奇异摄动问题)。

三组实验:

* ``sweep``    —— 自适应配点 vs 同点数、同步数的固定均匀配点, 跨多个 epsilon;
* ``ablation`` —— 选点策略 (RAR / RAD / uniform) 与求积权重 (体积 / 平均) 的消融;
* ``layers``   —— 三种层位置 (左边界 / 右边界 / 内部) 的自动定位;
* ``rounds``   —— 逐轮增点过程。

结果写入 ``<outdir>/case1_results.json``。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from spp_pinn import (AdaptivePINNSolver, TrainConfig, describe_device,
                      pick_device, run_reference)


def make_cfg(eps: float, problem: str, rounds: int, epochs: int, n_int0: int,
             add: int, strategy: str, weighting: str, device: str,
             seed: int = 0, hard: bool = False) -> TrainConfig:
    return TrainConfig(problem=problem, eps=eps, T=1.0, hidden=(64, 64, 64),
                       activation="tanh", n_int0=n_int0, n_ic0=64, n_bc0=64,
                       init_method="halton", rounds=rounds,
                       epochs_per_round=epochs, add_per_round=add,
                       strategy=strategy, weighting=weighting,
                       candidate_pool=20000, beta=1.0, min_dist=0.0,
                       adapt_ic=True, adapt_bc=True,
                       add_ic_per_round=16, add_bc_per_round=16,
                       lr=2e-3, lr_schedule="cosine", lr_gamma=0.9,
                       weight_mode="init", hard_constraint=hard,
                       seed=seed, device=device, verbose=False)


def run_one(cfg: TrainConfig, fixed: bool = False, n_ref: int = 0,
            total_epochs: int = 0) -> dict:
    t0 = time.time()
    if fixed:
        solver = run_reference(cfg, n_interior=n_ref,
                               total_epochs=total_epochs, verbose=False)
    else:
        solver = AdaptivePINNSolver(cfg)
        solver.run()
    ev = solver.evaluate()
    last = dict(solver.history[-1])
    return {"max_err": float(last["max_err"]), "rel_l2": float(last["rel_l2"]),
            "frac_layer": float(last["frac_layer"]), "N": int(last["N"]),
            "n_int": int(last["n_int"]), "wall": time.time() - t0,
            "err": float(np.max(ev["err"])),
            "cfg": {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in asdict(cfg).items() if k != "callback"}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--n-int0", type=int, default=1024)
    ap.add_argument("--add", type=int, default=150)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="case_results")
    ap.add_argument("--quick", action="store_true", help="小预算自检")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    os.makedirs(args.outdir, exist_ok=True)
    print(f"device = {describe_device(pick_device(args.device))}\n", flush=True)

    eps_list = (1e-4, 1e-5, 1e-6)
    n_ref = args.n_int0 + args.rounds * args.add
    steps = (args.rounds + 1) * args.epochs
    out = {"case": "case1", "rounds": args.rounds, "epochs": args.epochs,
           "n_int0": args.n_int0, "add": args.add, "n_ref": n_ref,
           "total_steps": steps, "sweep": [], "ablation": [], "layers": []}

    # ---- (1) epsilon 扫描 ----
    print("=" * 92)
    print("Case 1  epsilon sweep: adaptive vs fixed uniform")
    print("=" * 92)
    print(f"{'eps':>8} {'variant':>9} {'N':>6} {'rel_L2':>11} {'max_err':>11} "
          f"{'layer':>7} {'time':>7}")
    for eps in eps_list:
        for variant in ("adaptive", "fixed"):
            cfg = make_cfg(eps, "left", args.rounds, args.epochs, args.n_int0,
                           args.add, "residual", "voronoi", args.device)
            r = (run_one(cfg, fixed=True, n_ref=n_ref, total_epochs=steps)
                 if variant == "fixed" else run_one(cfg))
            r.update(eps=eps, variant=variant)
            out["sweep"].append(r)
            print(f"{eps:8.0e} {variant:>9} {r['N']:6d} {r['rel_l2']:11.3e} "
                  f"{r['max_err']:11.3e} {r['frac_layer']:7.3f} {r['wall']:6.0f}s",
                  flush=True)

    # ---- (2) 选点策略 x 求积权重 ----
    print("\n" + "=" * 92)
    print("Case 1  ablation: placement x weighting")
    print("=" * 92)
    print(f"{'eps':>8} {'strategy':>10} {'weight':>8} {'rel_L2':>11} "
          f"{'max_err':>11} {'layer':>7}")
    for eps in (1e-5,):
        for strat in ("residual", "distribution", "uniform"):
            for wname in ("voronoi", "mean"):
                cfg = make_cfg(eps, "left", args.rounds, args.epochs,
                               args.n_int0, args.add, strat, wname, args.device)
                r = run_one(cfg)
                r.update(eps=eps, strategy=strat, weighting=wname)
                out["ablation"].append(r)
                print(f"{eps:8.0e} {strat:>10} {wname:>8} "
                      f"{r['rel_l2']:11.3e} {r['max_err']:11.3e} "
                      f"{r['frac_layer']:7.3f}", flush=True)

    # ---- (3) 三种层位置 ----
    print("\n" + "=" * 92)
    print("Case 1  layer localization: left / right / interior")
    print("=" * 92)
    print(f"{'problem':>9} {'eps':>8} {'variant':>9} {'N':>6} {'rel_L2':>11} "
          f"{'max_err':>11} {'layer':>7}")
    for prob, eps in (("left", 1e-6), ("right", 1e-5), ("interior", 1e-5)):
        for variant in ("adaptive", "fixed"):
            cfg = make_cfg(eps, prob, args.rounds, args.epochs, args.n_int0,
                           args.add, "residual", "voronoi", args.device)
            r = (run_one(cfg, fixed=True, n_ref=n_ref, total_epochs=steps)
                 if variant == "fixed" else run_one(cfg))
            r.update(eps=eps, problem=prob, variant=variant)
            out["layers"].append(r)
            print(f"{prob:>9} {eps:8.0e} {variant:>9} {r['N']:6d} "
                  f"{r['rel_l2']:11.3e} {r['max_err']:11.3e} "
                  f"{r['frac_layer']:7.3f}", flush=True)

    path = os.path.join(args.outdir, "case1_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=float)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
