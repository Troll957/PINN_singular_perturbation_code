#!/usr/bin/env python3
"""Case 2 上的 2x3 消融：三种选点规则 x 两种求积权重。

预算与 Table 4 一致（29096 内点 / 408000 步），因此六行可直接互相比较。
同一 strategy 下的两个 weighting 共享同一条随机流（seed=0），点集相同。
可断点续跑：已存在的 ab_<策略>_<权重>.json 会被跳过。
"""
from __future__ import annotations
import argparse, json, os, time
import torch

from spp_pinn.nd import AdaptivePINNND, NDConfig
from spp_pinn import pick_device, describe_device

CONFIGS = [("residual", "knn"), ("residual", "mean"),
           ("distribution", "knn"), ("distribution", "mean"),
           ("uniform", "knn"), ("uniform", "mean")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=1e-05)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--n-int0", type=int, default=4096)
    ap.add_argument("--add", type=int, default=500)
    ap.add_argument("--eval-n", type=int, default=61)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="ablation")
    a = ap.parse_args()
    torch.set_num_threads(4)
    os.makedirs(a.outdir, exist_ok=True)
    n_ref = a.n_int0 + a.rounds * a.add
    steps = (a.rounds + 1) * a.epochs
    print(f"device={describe_device(pick_device(a.device))}", flush=True)
    print(f"Case 2, eps={a.eps:g}: {n_ref} 内点 / {steps} 步, 六种配置", flush=True)

    for strat, wname in CONFIGS:
        tag = f"ab_{strat}_{wname}"
        path = os.path.join(a.outdir, tag + ".json")
        if os.path.exists(path):
            print(f"[跳过] {tag} 已完成", flush=True)
            continue
        t0 = time.time()

        def cb(rec, _t=tag, _t0=t0):
            if rec["round"] % 5 == 0:
                print(f"  [{_t}] round {rec['round']:3d} rel_L2={rec['rel_l2']:.3e} "
                      f"max={rec['max_err']:.3e} ({(time.time()-_t0)/60:.0f} min)", flush=True)

        cfg = NDConfig(problem="corner2d", eps=a.eps, T=1.0, hidden=(64, 64, 64),
                       activation="tanh", n_int0=a.n_int0, n_ic0=128, n_bc0=64,
                       init_method="halton", rounds=a.rounds,
                       epochs_per_round=a.epochs, add_per_round=a.add,
                       strategy=strat, candidate_pool=20000, beta=1.0,
                       weighting=wname, knn_neighbors=4, lr=2e-3,
                       lr_schedule="cosine", lr_gamma=0.9, weight_mode="init",
                       seed=0, device=a.device, verbose=False, eval_n=a.eval_n,
                       callback=cb)
        s = AdaptivePINNND(cfg); s.run()
        last = dict(s.history[-1])
        r = dict(case="case2", eps=a.eps, strategy=strat, weighting=wname,
                 N=int(last["N"]), n_int=int(last["n_int"]),
                 rel_l2=float(last["rel_l2"]), max_err=float(last["max_err"]),
                 frac_layer=float(last["frac_layer"]),
                 wall=time.time() - t0, rounds=a.rounds,
                 epochs_per_round=a.epochs, total_steps=steps, seed=0)
        json.dump(r, open(path, "w"), indent=2)
        print(f"[完成] {tag}: rel_L2={r['rel_l2']:.4e} max={r['max_err']:.4e} "
              f"{r['wall']/60:.0f} min", flush=True)
    print("全部完成", flush=True)


if __name__ == "__main__":
    main()
