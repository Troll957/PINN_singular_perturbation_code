#!/usr/bin/env python3
"""精确 Voronoi 体积权重 vs 算术平均权重 (二维)。"""
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
    ap.add_argument("--add", type=int, default=200)
    ap.add_argument("--eval-n", type=int, default=61)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="cmp_weight")
    a = ap.parse_args()
    torch.set_num_threads(4)
    os.makedirs(a.outdir, exist_ok=True)
    prob = "corner2d" if a.case == "case2" else "moving"
    print(f"device={describe_device(pick_device(a.device))}  {prob}  eps={a.eps:g}\n")
    seeds = [int(v) for v in a.seeds.split(",")]
    print(f"{'weighting':>10} {'volume_method':>14} {'seed':>4} {'rel_L2':>11} {'max':>11} {'layer':>7} {'time':>6}")
    rows = []
    all_arms = (("volume", "knn"), ("mean", "knn"), ("volume", "voronoi"))
    arms = all_arms if a.arms == "all" else tuple(
        x for x in all_arms if x[1] == a.arms or x[0] == a.arms)
    for w, vm in arms:
      for sd in seeds:
        t0 = time.time()
        cfg = NDConfig(problem=prob, eps=a.eps, T=1.0, hidden=(64, 64, 64),
                       activation="tanh", n_int0=a.n_int0, n_ic0=128, n_bc0=64,
                       init_method="halton", rounds=a.rounds,
                       epochs_per_round=a.epochs, add_per_round=a.add,
                       strategy="residual", candidate_pool=20000, beta=1.0,
                       weighting=w, knn_neighbors=4, volume_method=vm,
                       lr=2e-3, lr_schedule="cosine", lr_gamma=0.9,
                       weight_mode="init", seed=sd, device=a.device,
                       verbose=False, eval_n=a.eval_n)
        s = AdaptivePINNND(cfg); s.run()
        h = dict(s.history[-1])
        r = dict(weighting=w, volume_method=vm, seed=sd,
                 rel_l2=float(h["rel_l2"]), max_err=float(h["max_err"]),
                 frac_layer=float(h["frac_layer"]), wall=time.time() - t0)
        rows.append(r)
        print(f"{w:>10} {vm:>14} {sd:>4} {r['rel_l2']:11.3e} {r['max_err']:11.3e} "
              f"{r['frac_layer']:7.3f} {r['wall']:5.0f}s", flush=True)
    json.dump(rows, open(os.path.join(a.outdir, f"cmp_{a.case}_eps{a.eps:g}.json"), "w"), indent=2)
    print("\n=== 汇总 (mean +/- std) ===")
    for w, vm in arms:
        sub = [r for r in rows if r["weighting"] == w and r["volume_method"] == vm]
        e = np.array([r["rel_l2"] for r in sub]); m = np.array([r["max_err"] for r in sub])
        print(f"{w:>7}/{vm:<8}  rel_L2 {e.mean():.3e} +/- {e.std():.3e}   "
              f"max {m.mean():.3e} +/- {m.std():.3e}   (n={len(sub)})")


if __name__ == "__main__":
    main()
