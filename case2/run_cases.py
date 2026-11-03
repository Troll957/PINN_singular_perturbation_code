#!/usr/bin/env python3
"""数值实验: Case 2 (二维角点层) 与 Case 3 (二维移动层)。

每个 Case 跑三组实验:

* ``sweep``   —— 自适应配点 vs 同点数、同步数的固定均匀配点, 跨多个 epsilon;
* ``weight``  —— 求积权重消融 (体积权重 vs 朴素平均);
* ``rounds``  —— 逐轮增点过程 (用于表格与收敛图)。

结果写入 ``<outdir>/<case>_results.json``。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from spp_pinn.nd import AdaptivePINNND, NDConfig, make_problem_nd
from spp_pinn import describe_device, pick_device


CASES = {
    "case2": dict(problem="corner2d", d=2,
                  eps_list=(1e-3, 1e-4, 1e-5),
                  weight="knn",
                  title="Case 2  corner layer in two space dimensions"),
    "case3": dict(problem="moving", d=2,
                  eps_list=(1e-2, 1e-3, 1e-4),
                  weight="knn",
                  title="Case 3  layer moving in time in two space dimensions"),
}


def make_cfg(case: str, eps: float, rounds: int, epochs: int, n_int0: int,
             add: int, strategy: str, weighting: str, neighbours: int,
             eval_n: int, device: str, seed: int = 0) -> NDConfig:
    return NDConfig(problem=CASES[case]["problem"], eps=eps, T=1.0,
                    hidden=(64, 64, 64), activation="tanh",
                    n_int0=n_int0, n_ic0=128, n_bc0=64, init_method="halton",
                    rounds=rounds, epochs_per_round=epochs, add_per_round=add,
                    strategy=strategy, candidate_pool=20000, beta=1.0,
                    weighting=weighting, knn_neighbors=neighbours,
                    lr=2e-3, lr_schedule="cosine", lr_gamma=0.9,
                    weight_mode="init", seed=seed, device=device,
                    verbose=False, eval_n=eval_n)


def run_one(cfg: NDConfig) -> dict:
    t0 = time.time()
    solver = AdaptivePINNND(cfg)
    hist = solver.run()
    last = hist[-1]
    return {"max_err": float(last["max_err"]), "rel_l2": float(last["rel_l2"]),
            "frac_layer": float(last["frac_layer"]),
            "rms_res": float(last["rms_res_uniform"]),
            "N": int(last["N"]), "n_int": int(last["n_int"]),
            "wall": time.time() - t0, "history": hist,
            "cfg": {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in asdict(cfg).items() if k != "callback"}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="all", choices=["case2", "case3", "all"])
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--n-int0", type=int, default=1024)
    ap.add_argument("--add", type=int, default=150)
    ap.add_argument("--eval-n", type=int, default=41)
    ap.add_argument("--neighbours", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="case_results")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    os.makedirs(args.outdir, exist_ok=True)
    dev = pick_device(args.device)
    print(f"device = {describe_device(dev)}\n", flush=True)

    keys = ["case2", "case3"] if args.case == "all" else [args.case]
    for case in keys:
        meta = CASES[case]
        out = {"case": case, "title": meta["title"], "sweep": [],
               "weight": [], "rounds": []}
        print("=" * 96)
        print(meta["title"])
        print("=" * 96, flush=True)

        # ---- (1) epsilon 扫描: 自适应 vs 固定均匀 ----
        print(f"\n-- epsilon sweep (adaptive vs fixed) --")
        print(f"{'eps':>8} {'variant':>9} {'N':>6} {'rel_L2':>11} "
              f"{'max_err':>11} {'layer':>7} {'time':>6}")
        for eps in meta["eps_list"]:
            n_ref = args.n_int0 + args.rounds * args.add
            for variant, strat, rounds in (("adaptive", "residual", args.rounds),
                                           ("fixed", "fixed", 0)):
                cfg = make_cfg(case, eps, rounds, args.epochs, args.n_int0,
                               args.add, strat, meta["weight"], args.neighbours,
                               args.eval_n, args.device)
                if variant == "fixed":
                    cfg.n_int0 = n_ref
                r = run_one(cfg)
                r.update(eps=eps, variant=variant)
                out["sweep"].append(r)
                print(f"{eps:8.0e} {variant:>9} {r['N']:6d} {r['rel_l2']:11.3e} "
                      f"{r['max_err']:11.3e} {r['frac_layer']:7.3f} "
                      f"{r['wall']:5.0f}s", flush=True)

        # ---- (2) 权重消融 ----
        eps_w = meta["eps_list"][1]
        print(f"\n-- weighting ablation at eps={eps_w:g} --")
        for wname in ("mean", meta["weight"]):
            cfg = make_cfg(case, eps_w, args.rounds, args.epochs, args.n_int0,
                           args.add, "residual", wname, args.neighbours,
                           args.eval_n, args.device)
            r = run_one(cfg)
            r.update(eps=eps_w, weighting=wname)
            out["weight"].append(r)
            print(f"{wname:>8} N={r['N']:6d} rel_L2={r['rel_l2']:.3e} "
                  f"max_err={r['max_err']:.3e}", flush=True)

        # ---- (3) 逐轮过程 ----
        eps_r = meta["eps_list"][1]
        cfg = make_cfg(case, eps_r, args.rounds, args.epochs, args.n_int0,
                       args.add, "residual", meta["weight"], args.neighbours,
                       args.eval_n, args.device)
        r = run_one(cfg)
        out["rounds"] = r["history"]
        print(f"\n-- per-round enrichment at eps={eps_r:g} --")
        print(f"{'round':>5} {'n_int':>6} {'N':>6} {'rel_L2':>11} "
              f"{'max_err':>11} {'layer':>7}")
        for h in r["history"]:
            print(f"{h['round']:>5d} {h['n_int']:>6d} {h['N']:>6d} "
                  f"{h['rel_l2']:>11.3e} {h['max_err']:>11.3e} "
                  f"{h['frac_layer']:>7.3f}", flush=True)

        path = os.path.join(args.outdir, f"{case}_results.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False, default=float)
        print(f"\n-> {path}\n", flush=True)


if __name__ == "__main__":
    main()
