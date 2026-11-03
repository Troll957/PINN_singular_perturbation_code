#!/usr/bin/env python3
"""Case 1 的 eps 扫描: 每个 (eps, 变体) 跑多个种子, 报告 mean +- std。"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch
from spp_pinn import AdaptivePINNSolver, TrainConfig, pick_device, describe_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="1e-3,1e-4,1e-5,1e-6")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--n-int0", type=int, default=1024)
    ap.add_argument("--add", type=int, default=150)
    ap.add_argument("--fine-nx", type=int, default=2001)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="seeds_results")
    a = ap.parse_args()
    torch.set_num_threads(6)
    os.makedirs(a.outdir, exist_ok=True)
    eps_list = [float(v) for v in a.eps.split(",")]
    n_ref = a.n_int0 + a.rounds * a.add
    steps = (a.rounds + 1) * a.epochs
    print(f"device={describe_device(pick_device(a.device))}  eps={eps_list}  "
          f"seeds=0..{a.seeds-1}  预算: {n_ref} 点 / {steps} 步\n", flush=True)
    rows = []
    hdr = (f"{'eps':>7} {'variant':>9} {'run':>4} {'rel_L2':>10} "
           f"{'max':>10} {'layer max':>10} {'time':>6}")
    print(hdr); print("-" * len(hdr), flush=True)
    for eps in eps_list:
        thr = 4.0 * math.sqrt(eps)
        for variant in ("adaptive", "fixed"):
            for sd in range(a.seeds):
                t0 = time.time()
                if variant == "adaptive":
                    cfg = TrainConfig(problem="left", eps=eps, T=1.0,
                                      hidden=(64, 64, 64), activation="tanh",
                                      n_int0=a.n_int0, n_ic0=64, n_bc0=64,
                                      init_method="halton", rounds=a.rounds,
                                      epochs_per_round=a.epochs,
                                      add_per_round=a.add, strategy="residual",
                                      candidate_pool=20000, beta=1.0, min_dist=0.0,
                                      lr=2e-3, lr_schedule="cosine", lr_gamma=0.9,
                                      weight_mode="init", seed=sd,
                                      device=a.device, verbose=False)
                else:
                    cfg = TrainConfig(problem="left", eps=eps, T=1.0,
                                      hidden=(64, 64, 64), activation="tanh",
                                      n_int0=n_ref, n_ic0=64, n_bc0=64,
                                      init_method="halton", rounds=0,
                                      epochs_per_round=steps, add_per_round=0,
                                      strategy="fixed", candidate_pool=20000,
                                      beta=1.0, min_dist=0.0, lr=2e-3,
                                      lr_schedule="cosine", lr_gamma=0.9,
                                      weight_mode="init", seed=sd,
                                      device=a.device, verbose=False)
                s = AdaptivePINNSolver(cfg); s.run()
                fine = s.evaluate(nx=a.fine_nx, nt=201)
                X, err = np.asarray(fine["X"]), np.asarray(fine["err"])
                lm = float(err[X <= thr].max()) if np.any(X <= thr) else float("nan")
                last = dict(s.history[-1])
                r = dict(eps=eps, variant=variant, seed=sd,
                         rel_l2=float(last["rel_l2"]),
                         max_fine=float(fine["max_err"]), layer_max=lm,
                         frac_layer=float(last["frac_layer"]),
                         wall=time.time() - t0)
                rows.append(r)
                print(f"{eps:7.0e} {variant:>9} {sd:>4d} {r['rel_l2']:10.3e} "
                      f"{r['max_fine']:10.3e} {r['layer_max']:10.3e} "
                      f"{r['wall']:5.0f}s", flush=True)
    json.dump(rows, open(os.path.join(a.outdir, "seeds_eps_rows.json"), "w"), indent=2)

    print("\n=== 汇总 (mean +- std) ===", flush=True)
    print(f"{'eps':>7} {'variant':>9} {'rel_L2':>20} {'max':>20}", flush=True)
    for eps in eps_list:
        for variant in ("adaptive", "fixed"):
            sel = [r for r in rows if r["eps"] == eps and r["variant"] == variant]
            rl = np.array([r["rel_l2"] for r in sel]); mx = np.array([r["max_fine"] for r in sel])
            print(f"{eps:7.0e} {variant:>9} "
                  f"{rl.mean():.3e} +- {rl.std():.1e}   "
                  f"{mx.mean():.3e} +- {mx.std():.1e}", flush=True)
    print("\n->", a.outdir)


if __name__ == "__main__":
    main()
