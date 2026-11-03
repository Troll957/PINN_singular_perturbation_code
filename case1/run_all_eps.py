#!/usr/bin/env python3
"""对 eps = 1e-3,1e-4,1e-5,1e-6 逐个跑自适应/固定对照, 并直接出图。

输出: <outdir>/<tag>.npz, <tag>_sol.pdf (解析解/预测/误差), <tag>_pts.pdf (点分布)
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, torch
from spp_pinn import AdaptivePINNSolver, TrainConfig, pick_device, describe_device, run_reference
from spp_pinn import plotting as PLT


def cfg_of(eps, rounds, epochs, n0, add, strategy, weighting, device, hard=False):
    return TrainConfig(problem="left", eps=eps, T=1.0, hidden=(64, 64, 64),
                       activation="tanh", n_int0=n0, n_ic0=64, n_bc0=64,
                       init_method="halton", rounds=rounds,
                       epochs_per_round=epochs, add_per_round=add,
                       strategy=strategy, weighting=weighting,
                       candidate_pool=20000, beta=1.0, min_dist=0.0,
                       adapt_ic=True, adapt_bc=True, add_ic_per_round=16,
                       add_bc_per_round=16, lr=2e-3, lr_schedule="cosine",
                       lr_gamma=0.9, weight_mode="init",
                       hard_constraint=hard, seed=0, device=device, verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="1e-3,1e-4,1e-5,1e-6")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--n-int0", type=int, default=1024)
    ap.add_argument("--add", type=int, default=150)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="eps_results")
    a = ap.parse_args()
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    os.makedirs(a.outdir, exist_ok=True)
    eps_list = [float(v) for v in a.eps.split(",")]
    n_ref = a.n_int0 + a.rounds * a.add
    steps = (a.rounds + 1) * a.epochs
    print(f"device={describe_device(pick_device(a.device))}  eps={eps_list}\n", flush=True)
    rows = []
    print(f"{'eps':>8} {'variant':>9} {'N':>6} {'rel_L2':>11} {'max_err':>11} {'layer':>7} {'time':>6}")
    for eps in eps_list:
        for variant in ("adaptive", "fixed"):
            t0 = time.time()
            cfg = cfg_of(eps, a.rounds, a.epochs, a.n_int0, a.add,
                         "residual", "voronoi", a.device)
            if variant == "fixed":
                solver = run_reference(cfg, n_interior=n_ref, total_epochs=steps, verbose=False)
            else:
                solver = AdaptivePINNSolver(cfg); solver.run()
            ev = solver.evaluate()
            last = dict(solver.history[-1])
            tag = f"case1_{variant}_eps{eps:.0e}"
            np.savez_compressed(os.path.join(a.outdir, tag + ".npz"),
                                X=ev["X"], T=ev["T"], u_pred=ev["u_pred"],
                                u_exact=ev["u_exact"], err=ev["err"],
                                points=solver.final_points)
            PLT.plot_solution_fields(ev, eps, "left (adaptive)" if variant == "adaptive"
                                     else "left (uniform)", os.path.join(a.outdir, tag + "_sol.pdf"))
            PLT.plot_point_distribution(solver.final_points, eps,
                                        os.path.join(a.outdir, tag + "_pts.pdf"))
            rows.append(dict(case="case1", eps=eps, variant=variant, N=int(last["N"]),
                             n_int=int(last["n_int"]), rel_l2=float(last["rel_l2"]),
                             max_err=float(last["max_err"]),
                             frac_layer=float(last["frac_layer"]), wall=time.time() - t0))
            r = rows[-1]
            print(f"{eps:8.0e} {variant:>9} {r['N']:6d} {r['rel_l2']:11.3e} "
                  f"{r['max_err']:11.3e} {r['frac_layer']:7.3f} {r['wall']:5.0f}s", flush=True)
    json.dump(rows, open(os.path.join(a.outdir, "case1_eps_rows.json"), "w"), indent=2)
    print("\n->", a.outdir)


if __name__ == "__main__":
    main()
