#!/usr/bin/env python3
"""诊断: 二维结果差是点数不够、训练不够、还是容量不够?

对同一个 eps 逐项放大一个因素, 比较相对 L2 误差。
"""
from __future__ import annotations
import argparse, json, os, time
import torch
from spp_pinn.nd import AdaptivePINNND, NDConfig
from spp_pinn import pick_device, describe_device

VARIANTS = [
    ("baseline      ", dict(rounds=20, epochs=1000, add=150, hidden=(64, 64, 64))),
    ("points x2.7   ", dict(rounds=20, epochs=1000, add=400, hidden=(64, 64, 64))),
    ("points x5.3   ", dict(rounds=20, epochs=1000, add=800, hidden=(64, 64, 64))),
    ("train x3      ", dict(rounds=20, epochs=3000, add=150, hidden=(64, 64, 64))),
    ("capacity x4   ", dict(rounds=20, epochs=1000, add=150, hidden=(128, 128, 128))),
    ("deeper        ", dict(rounds=20, epochs=1000, add=150, hidden=(64, 64, 64, 64))),
    ("points+train  ", dict(rounds=20, epochs=3000, add=400, hidden=(64, 64, 64))),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="case2", choices=["case2", "case3"])
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--eval-n", type=int, default=81)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="diag_results")
    a = ap.parse_args()
    torch.set_num_threads(4)
    os.makedirs(a.outdir, exist_ok=True)
    prob = "corner2d" if a.case == "case2" else "moving"
    print(f"device={describe_device(pick_device(a.device))}  {prob}  eps={a.eps:g}  "
          f"eval_n={a.eval_n}\n", flush=True)
    print(f"{'variant':16s} {'rounds':>6} {'epochs':>7} {'add':>5} {'hidden':>16} "
          f"{'n_int':>6} {'N':>6} {'rel_L2':>11} {'max_err':>11} {'time':>6}")
    rows = []
    for name, kw in VARIANTS:
        t0 = time.time()
        cfg = NDConfig(problem=prob, eps=a.eps, T=1.0, hidden=kw["hidden"],
                       activation="tanh", n_int0=1024, n_ic0=128, n_bc0=64,
                       init_method="halton", rounds=kw["rounds"],
                       epochs_per_round=kw["epochs"], add_per_round=kw["add"],
                       strategy="residual", candidate_pool=20000, beta=1.0,
                       weighting="knn", knn_neighbors=4, lr=2e-3,
                       lr_schedule="cosine", lr_gamma=0.9, weight_mode="init",
                       seed=0, device=a.device, verbose=False, eval_n=a.eval_n)
        s = AdaptivePINNND(cfg); s.run()
        last = dict(s.history[-1])
        r = dict(variant=name.strip(), rel_l2=float(last["rel_l2"]),
                 max_err=float(last["max_err"]), N=int(last["N"]),
                 n_int=int(last["n_int"]), frac_layer=float(last["frac_layer"]),
                 wall=time.time() - t0)
        rows.append(r)
        print(f"{name:16s} {kw['rounds']:>6d} {kw['epochs']:>7d} {kw['add']:>5d} "
              f"{str(kw['hidden']):>16s} {r['n_int']:>6d} {r['N']:>6d} "
              f"{r['rel_l2']:>11.3e} {r['max_err']:>11.3e} {r['wall']:>5.0f}s",
              flush=True)
    json.dump(rows, open(os.path.join(a.outdir, f"diag_{a.case}_eps{a.eps:g}.json"), "w"),
              indent=2)
    print("\n->", a.outdir)


if __name__ == "__main__":
    main()
