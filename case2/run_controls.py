#!/usr/bin/env python3
"""只重跑 Case 2/3 的对照组（固定均匀点集）。

修正要点：对照组也走 K+1 个训练阶段（rounds=K, epochs_per_round=E），
因此总步数与自适应组相同 (K+1)*E，余弦重启与几何衰减调度也相同；
strategy="fixed" 使加点为空操作，点集始终固定。

可断点续跑：已存在的 <outdir>/ctrl_<case>_eps<e>.json 会被跳过。
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, torch

from spp_pinn.nd import AdaptivePINNND, NDConfig
from spp_pinn import pick_device, describe_device


def cfg_of(problem, eps, rounds, epochs, n0, add, device, evn, cb):
    return NDConfig(problem=problem, eps=eps, T=1.0, hidden=(64, 64, 64),
                    activation="tanh", n_int0=n0, n_ic0=128, n_bc0=64,
                    init_method="halton", rounds=rounds,
                    epochs_per_round=epochs, add_per_round=add,
                    strategy="fixed", candidate_pool=20000, beta=1.0,
                    weighting="knn", knn_neighbors=4, lr=2e-3,
                    lr_schedule="cosine", lr_gamma=0.9, weight_mode="init",
                    seed=0, device=device, verbose=False, eval_n=evn,
                    callback=cb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="1e-05,1e-06,1e-04,0.001",
                    help="按重要性排序, 先跑最关键的")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--n-int0", type=int, default=4096)
    ap.add_argument("--add", type=int, default=500)
    ap.add_argument("--eval-n", type=int, default=61)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--outdir", default="controls")
    a = ap.parse_args()
    torch.set_num_threads(4)
    os.makedirs(a.outdir, exist_ok=True)
    n_ref = a.n_int0 + a.rounds * a.add
    steps = (a.rounds + 1) * a.epochs
    eps_list = [float(v) for v in a.eps.split(",")]
    print(f"device={describe_device(pick_device(a.device))}", flush=True)
    print(f"每轮: rounds={a.rounds} epochs={a.epochs} -> 总步数 {steps}, "
          f"内部点 {n_ref} 固定", flush=True)

    for case in ("case2", "case3"):
        prob = "corner2d" if case == "case2" else "moving"
        for eps in eps_list:
            tag = f"{case}_eps{eps:g}"
            path = os.path.join(a.outdir, f"ctrl_{tag}.json")
            if os.path.exists(path):
                print(f"[跳过] {tag} 已完成", flush=True)
                continue
            t0 = time.time()

            def cb(rec, _tag=tag, _t0=t0):
                print(f"  [{_tag}] round {rec['round']:3d}  N={rec['N']:6d} "
                      f"rel_L2={rec['rel_l2']:.3e} max={rec['max_err']:.3e} "
                      f"lr={rec.get('lr', float('nan')):.2e} "
                      f"({(time.time()-_t0)/60:.1f} min)", flush=True)

            cfg = cfg_of(prob, eps, a.rounds, a.epochs, n_ref, a.add,
                         a.device, a.eval_n, cb)
            s = AdaptivePINNND(cfg); s.run()
            last = dict(s.history[-1])
            r = dict(case=case, eps=eps, variant="fixed", N=int(last["N"]),
                     n_int=int(last["n_int"]), rel_l2=float(last["rel_l2"]),
                     max_err=float(last["max_err"]),
                     frac_layer=float(last["frac_layer"]),
                     wall=time.time() - t0, rounds=a.rounds,
                     epochs_per_round=a.epochs, total_steps=steps,
                     strategy="fixed", seed=0)
            json.dump(r, open(path, "w"), indent=2)
            print(f"[完成] {tag}: rel_L2={r['rel_l2']:.4e} "
                  f"max={r['max_err']:.4e}  {r['wall']/3600:.2f} h", flush=True)
    print("全部完成", flush=True)


if __name__ == "__main__":
    main()
