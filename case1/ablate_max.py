#!/usr/bin/env python3
"""2x2 因子对照: 权重方案 (voronoi/mean) x 输入是否加层尺度通道。

关键点: 误差在**细网格**上评估, 并单独报告层内 (x <= 4 sqrt(eps)) 的最大误差,
因为 401 点网格在 eps=1e-6 时分不清层。
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch
from spp_pinn import AdaptivePINNSolver, TrainConfig, pick_device, describe_device

VARIANTS = [
    ("L2 损失", dict(weighting="voronoi", loss_norm="l2", n_int0=1024, rounds=20, epochs=1000)),
    ("L1 损失", dict(weighting="voronoi", loss_norm="l1", n_int0=1024, rounds=20, epochs=1000)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--n-int0", type=int, default=1024)
    ap.add_argument("--add", type=int, default=150)
    ap.add_argument("--fine-nx", type=int, default=2001)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="ablate_results")
    a = ap.parse_args()
    torch.set_num_threads(6)
    os.makedirs(a.outdir, exist_ok=True)
    thr = 4.0 * math.sqrt(a.eps)
    print(f"device={describe_device(pick_device(a.device))}  eps={a.eps:g}  "
          f"层宽 4sqrt(eps)={thr:.3e}  细网格 dx={1/(a.fine_nx-1):.2e}\n", flush=True)
    hdr = (f"{'variant':26s} {'rel_L2':>10s} {'max(401)':>10s} "
           f"{'max(细网格)':>12s} {'层内max':>10s} {'layer frac':>10s} {'time':>6s}")
    print(hdr); print("-" * len(hdr), flush=True)
    rows = []
    for name, kw in VARIANTS:
        t0 = time.time()
        kw = dict(kw)
        n0 = kw.pop("n_int0", a.n_int0); rnd = kw.pop("rounds", a.rounds)
        ep = kw.pop("epochs", a.epochs)
        cfg = TrainConfig(problem="left", eps=a.eps, T=1.0, hidden=(64, 64, 64),
                          activation="tanh", n_int0=n0, n_ic0=64, n_bc0=64,
                          init_method="halton", rounds=rnd,
                          epochs_per_round=ep, add_per_round=a.add,
                          strategy="residual", candidate_pool=20000, beta=1.0,
                          min_dist=0.0, lr=2e-3, lr_schedule="cosine",
                          lr_gamma=0.9, weight_mode="init", seed=0,
                          device=a.device, verbose=False, **kw)
        s = AdaptivePINNSSolver(cfg) if False else AdaptivePINNSolver(cfg)
        s.run()
        coarse = s.evaluate()                       # 401 x 201, 与论文一致
        fine = s.evaluate(nx=a.fine_nx, nt=201)     # 层可分辨
        X, err = np.asarray(fine["X"]), np.asarray(fine["err"])
        layer_max = float(err[X <= thr].max()) if np.any(X <= thr) else float("nan")
        last = dict(s.history[-1])
        r = dict(variant=name, rel_l2=float(last["rel_l2"]),
                 max_coarse=float(coarse["max_err"]),
                 max_fine=float(fine["max_err"]), max_layer=layer_max,
                 frac_layer=float(last["frac_layer"]), wall=time.time() - t0,
                 **{k: v for k, v in kw.items()})
        rows.append(r)
        print(f"{name:26s} {r['rel_l2']:10.3e} {r['max_coarse']:10.3e} "
              f"{r['max_fine']:12.3e} {r['max_layer']:10.3e} "
              f"{r['frac_layer']:10.3f} {r['wall']:5.0f}s", flush=True)
    json.dump(rows, open(os.path.join(a.outdir, f"ablate_max_eps{a.eps:g}.json"), "w"),
              indent=2, ensure_ascii=False)
    print("\n->", a.outdir)


if __name__ == "__main__":
    main()
