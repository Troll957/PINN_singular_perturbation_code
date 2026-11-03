#!/usr/bin/env python3
"""L2 vs L1 损失范数对照 (二维)。"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch
from spp_pinn.nd import AdaptivePINNND, NDConfig
from spp_pinn import pick_device, describe_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="case2", choices=["case2", "case3"])
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--n-int0", type=int, default=1024)
    ap.add_argument("--add", type=int, default=150)
    ap.add_argument("--eval-n", type=int, default=61)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="norm_results")
    a = ap.parse_args()
    torch.set_num_threads(4)
    os.makedirs(a.outdir, exist_ok=True)
    prob = "corner2d" if a.case == "case2" else "moving"
    thr = 4.0 * math.sqrt(a.eps)
    print(f"device={describe_device(pick_device(a.device))}  {prob}  eps={a.eps:g}\n", flush=True)
    print(f"{'loss':>6} {'rel_L2':>11} {'max':>11} {'layer frac':>11} {'time':>6}", flush=True)
    rows = []
    for name, norm in (("L2", "l2"), ("L1", "l1")):
        t0 = time.time()
        cfg = NDConfig(problem=prob, eps=a.eps, T=1.0, hidden=(64, 64, 64),
                       activation="tanh", n_int0=a.n_int0, n_ic0=128, n_bc0=64,
                       init_method="halton", rounds=a.rounds,
                       epochs_per_round=a.epochs, add_per_round=a.add,
                       strategy="residual", candidate_pool=20000, beta=1.0,
                       weighting="knn", knn_neighbors=4, lr=2e-3,
                       lr_schedule="cosine", lr_gamma=0.9, weight_mode="init",
                       loss_norm=norm, seed=0, device=a.device, verbose=False,
                       eval_n=a.eval_n)
        s = AdaptivePINNND(cfg); s.run()
        last = dict(s.history[-1])
        r = dict(loss=name, rel_l2=float(last["rel_l2"]),
                 max_err=float(last["max_err"]),
                 frac_layer=float(last["frac_layer"]), wall=time.time() - t0)
        rows.append(r)
        print(f"{name:>6} {r['rel_l2']:11.3e} {r['max_err']:11.3e} "
              f"{r['frac_layer']:11.3f} {r['wall']:5.0f}s", flush=True)
    json.dump(rows, open(os.path.join(a.outdir, f"norm_{a.case}_eps{a.eps:g}.json"), "w"), indent=2)
    print("\n->", a.outdir)


if __name__ == "__main__":
    main()
