r"""实验驱动：残差驱动的自适应配点 PINN。

用法::

    python run_experiments.py --scale quick --workers 3
    python run_experiments.py --scale full  --workers 4 --outdir sampling_results

实验内容
--------
* ``S1`` 策略对比（同一预算、同一初始点集）：

  - ``adapt-vor``  : 残差驱动增点 + Voronoi 保范数权重（本文方法）
  - ``adapt-mean`` : 残差驱动增点 + 朴素算术平均（常见的 RAR 写法）
  - ``uniE-vor``   : **均匀**随机增点 + Voronoi 权重（对照：点数增加 vs 点放哪里）
  - ``fixed``      : 一次性给足点数、不做自适应（同最终内部点数）

* ``S2`` 摄动参数扫描：$\varepsilon=10^{-3}\sim10^{-6}$，自适应 vs 固定均匀
* ``S3`` 内层问题（层在 $x=1/2$）与右边界层（层在 $x=1$，检验自动定位）
* ``S5`` 硬约束格式

每个算例保存 ``<outdir>/<tag>.json``（配置 + 逐轮历史）与 ``<outdir>/<tag>.npz``
（末端点集、残差、误差场）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dsh_mpl")

SCALES = {
    "quick": dict(rounds=4, epochs=350, n_int0=512, add=128,
                  fixed_int=1024, fixed_epochs=1750),
    "full": dict(rounds=8, epochs=1000, n_int0=1024, add=150,
                 fixed_int=2224, fixed_epochs=9000),
    # 加大训练预算: 21 个训练阶段 (初始 + 20 次增点) x 2000 步 = 42000 步,
    # 终态内部点数 1024 + 20*150 = 4024, 参照组同点数同步数.
    "long": dict(rounds=20, epochs=2000, n_int0=1024, add=150,
                 fixed_int=4024, fixed_epochs=42000),
}


# ----------------------------------------------------------------------
def run_job(job: dict) -> dict:
    import numpy as np
    import torch
    torch.set_num_threads(job.get("threads", 2))
    from spp_pinn import AdaptivePINNSolver, TrainConfig, run_reference

    outdir = job["outdir"]
    os.makedirs(outdir, exist_ok=True)
    cfg = TrainConfig(**job["cfg"])

    t0 = time.time()
    if job["kind"] == "adaptive":
        solver = AdaptivePINNSolver(cfg)
        solver.run()
    else:                                     # 固定点集参照组
        solver = run_reference(cfg, n_interior=job["n_interior"],
                               total_epochs=job["epochs"], verbose=cfg.verbose)
    wall = time.time() - t0

    ev = solver.evaluate()
    final = dict(solver.history[-1])
    final.update(wall_total=wall, kind=job["kind"], tag=job["tag"])

    with open(os.path.join(outdir, f"{job['tag']}.json"), "w") as fh:
        json.dump({"tag": job["tag"], "kind": job["kind"], "cfg": cfg.to_dict(),
                   "n_interior": job.get("n_interior"), "epochs": job.get("epochs"),
                   "history": solver.history, "final": final},
                  fh, indent=2, ensure_ascii=False, default=float)

    np.savez_compressed(
        os.path.join(outdir, f"{job['tag']}.npz"),
        X=ev["X"], T=ev["T"], u_pred=ev["u_pred"], u_exact=ev["u_exact"], err=ev["err"],
        points=solver.final_points, abs_res=solver.final_abs_res,
        ic=solver.final_ic, bc=solver.final_bc,
    )
    print(f"[done] {job['tag']:30s} N={final['N']:5d} max_err={final['max_err']:.3e} "
          f"rel_l2={final['rel_l2']:.3e} layer_frac={final['frac_layer']:.3f} "
          f"({wall:.0f}s)", flush=True)
    return {"tag": job["tag"], "kind": job["kind"], "eps": cfg.eps,
            "problem": cfg.problem, "strategy": cfg.strategy,
            "weighting": cfg.weighting, "N": final["N"], "n_int": final["n_int"],
            "max_err": final["max_err"], "rel_l2": final["rel_l2"],
            "frac_layer": final["frac_layer"], "wall": wall}


# ----------------------------------------------------------------------
def build_jobs(scale: str, outdir: str, verbose: bool = False,
               device: str = "auto") -> list:
    sc = SCALES[scale]
    R, E, N0, ADD = sc["rounds"], sc["epochs"], sc["n_int0"], sc["add"]
    FI, FE = sc["fixed_int"], sc["fixed_epochs"]

    def base(**kw):
        cfg = dict(problem="left", eps=1e-5, hidden=(64, 64, 64),
                   n_int0=N0, n_ic0=64, n_bc0=64, init_method="halton",
                   rounds=R, epochs_per_round=E, add_per_round=ADD,
                   strategy="residual", weighting="voronoi", candidate_pool=20000,
                   beta=1.0, min_dist=0.0, lr=2e-3, lr_schedule="cosine",
                   lr_gamma=0.9, seed=0, device=device, verbose=verbose)
        cfg.update(kw)
        return cfg

    def adaptive(tag, **kw):
        return dict(tag=tag, kind="adaptive", outdir=outdir, cfg=base(**kw))

    def fixed(tag, **kw):
        return dict(tag=tag, kind="fixed", outdir=outdir, n_interior=FI,
                    epochs=FE, cfg=base(strategy="fixed", rounds=0,
                                        epochs_per_round=FE, **kw))

    jobs = []

    # ---- S1: 策略对比 (eps = 1e-4 与 1e-5) ----
    for eps, es in ((1e-4, "1e-04"), (1e-5, "1e-05")):
        jobs.append(adaptive(f"S1-adapt-vor-eps{es}", eps=eps))
        jobs.append(adaptive(f"S1-adapt-mean-eps{es}", eps=eps, weighting="mean"))
        jobs.append(adaptive(f"S1-uniE-vor-eps{es}", eps=eps, strategy="uniform"))
        jobs.append(fixed(f"S1-fixed-eps{es}", eps=eps))

    # ---- S2: 摄动参数扫描 (1e-4/1e-5 已在 S1 中) ----
    for eps, es in ((1e-3, "1e-03"), (1e-6, "1e-06")):
        jobs.append(adaptive(f"S2-adapt-eps{es}", eps=eps))
        jobs.append(fixed(f"S2-fixed-eps{es}", eps=eps))

    # ---- S3/S4: 内层与右边界层 ----
    for prob, es in (("interior", "1e-04"), ("right", "1e-05")):
        jobs.append(adaptive(f"S3-adapt-{prob}-eps{es}", problem=prob, eps=float(es)))
        jobs.append(fixed(f"S3-fixed-{prob}-eps{es}", problem=prob, eps=float(es)))

    # ---- S5: 硬约束 ----
    jobs.append(adaptive("S5-hard-adapt-eps1e-05", hard_constraint=True))
    jobs.append(fixed("S5-hard-fixed-eps1e-05", hard_constraint=True))

    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description="残差驱动自适应配点 PINN 实验")
    ap.add_argument("--scale", choices=sorted(SCALES), default="quick")
    ap.add_argument("--outdir", default="sampling_results")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--only", default=None)
    ap.add_argument("--device", default="auto",
                    help="auto | cuda | cuda:0 | mps | cpu")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs(args.scale, args.outdir, verbose=args.verbose,
                      device=args.device)
    if args.only:
        jobs = [j for j in jobs if args.only in j["tag"]]
    for j in jobs:
        j["threads"] = args.threads
    from spp_pinn import describe_device, pick_device
    dev = pick_device(args.device)
    print(f"共 {len(jobs)} 个算例, scale={args.scale}, workers={args.workers}, "
          f"device={describe_device(dev)}\n", flush=True)
    if dev.type == "cuda" and args.workers > 1:
        print("提示: 多个 worker 会共用同一块显卡, 显存够但速度可能不如串行; "
              "建议 --workers 1, 或用 CUDA_VISIBLE_DEVICES 为每个进程指定不同卡。\n",
              flush=True)

    t0 = time.time()
    results = []
    # CUDA 不能在 fork 出的子进程中重新初始化, 因此 GPU 上改用 spawn 启动方式
    ctx = mp.get_context("spawn") if dev.type == "cuda" else None
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = {ex.submit(run_job, j): j["tag"] for j in jobs}
        for fu in as_completed(futs):
            try:
                results.append(fu.result())
            except Exception as exc:            # noqa: BLE001
                print(f"[FAIL] {futs[fu]}: {exc!r}", flush=True)

    results.sort(key=lambda r: (r["problem"], -r["eps"], r["tag"]))
    with open(os.path.join(args.outdir, "summary.json"), "w") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=float)

    print(f"\n全部完成, 用时 {time.time()-t0:.1f}s")
    print(f"{'tag':30s} {'prob':9s} {'eps':>8s} {'strat':9s} {'weight':8s} "
          f"{'N':>5s} {'max_err':>10s} {'rel_l2':>10s} {'layer':>6s}")
    for r in results:
        print(f"{r['tag']:30s} {r['problem']:9s} {r['eps']:8.1e} {r['strategy']:9s} "
              f"{r['weighting']:8s} {r['N']:5d} {r['max_err']:10.3e} {r['rel_l2']:10.3e} "
              f"{r['frac_layer']:6.3f}")


if __name__ == "__main__":
    main()
