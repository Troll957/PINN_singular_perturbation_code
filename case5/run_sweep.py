#!/usr/bin/env python
"""按参数扫描跑多组实验，汇总成 CSV / Markdown 表格 / 图。

示例::

    # 扫 epsilon，自适应 vs 均匀配点
    python run_sweep.py --sweep eps --eps-list 1e-2,1e-3,1e-4 \
        --noise 0.01 --outdir runs/sweep_eps

    # 扫噪声水平
    python run_sweep.py --sweep noise --noise-list 0.001,0.01,0.05 \
        --eps 1e-3 --outdir runs/sweep_noise

每一组都保存到 ``<outdir>/<tag>/``，汇总写到 ``<outdir>/sweep.csv``、
``<outdir>/sweep.md`` 与 ``<outdir>/sweep_<key>.png``。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from case5 import Case5Config, InversePINN, InverseSource2D, plotting


def parse_list(s: str, cast=float):
    return [cast(x) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", choices=["eps", "noise", "sensors"], default="eps")
    p.add_argument("--eps-list", type=str, default="1e-2,1e-3,1e-4")
    p.add_argument("--noise-list", type=str, default="0.001,0.01,0.05")
    p.add_argument("--sensors-list", type=str, default="20,40,80")
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument("--strategies", type=str, default="residual,fixed")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--add", type=int, default=200)
    p.add_argument("--n-int0", type=int, default=2048)
    p.add_argument("--sensors", type=int, default=60)
    p.add_argument("--case", type=str, default="moving")
    p.add_argument("--sensor-mode", type=str, default="grid")
    p.add_argument("--w-data", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=1,
                   help="每个配置重复几个种子（>1 时报告均值±标准差）")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--outdir", type=str, default="runs/sweep")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    if a.sweep == "eps":
        values = parse_list(a.eps_list)
        key = "eps"
    elif a.sweep == "noise":
        values = parse_list(a.noise_list)
        key = "noise"
    else:
        values = parse_list(a.sensors_list, int)
        key = "n_sensors"

    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
    rows = []
    t_start = time.time()

    total = len(values) * len(strategies) * a.seeds
    i = 0
    for v in values:
        for strat in strategies:
            for s in range(a.seeds):
                i += 1
                eps = v if key == "eps" else a.eps
                noise = v if key == "noise" else a.noise
                sensors = v if key == "n_sensors" else a.sensors
                tag = (f"{key}={v:g}_{strat}_seed{a.seed + s}")
                print(f"\n[{i}/{total}] {tag}  "
                      f"(已经过 {(time.time() - t_start) / 60:.1f} min)")
                cfg = Case5Config(
                    case=a.case, eps=eps, noise=noise, n_sensors=int(sensors),
                    sensor_mode=a.sensor_mode, strategy=strat,
                    rounds=a.rounds if strat != "fixed" else 0,
                    epochs_per_round=(a.epochs if strat != "fixed"
                                      else a.epochs * (a.rounds + 1)),
                    add_per_round=a.add, n_int0=a.n_int0,
                    w_data=a.w_data, seed=a.seed + s, device=a.device,
                    verbose=not a.quiet, tag=strat)
                if strat == "fixed":
                    cfg.n_int0 = a.n_int0 + a.rounds * a.add
                solver = InversePINN(cfg)
                solver.run()
                sdir = os.path.join(a.outdir, tag.replace("=", "").replace(".", "p"))
                solver.save(sdir)
                h = solver.history[-1]
                row = {
                    key: v, "strategy": strat, "seed": a.seed + s,
                    "eps": eps, "noise": noise, "n_sensors": int(sensors),
                    "param_err_max": h["param_err_max"],
                    "src_rel_l2": h["src_rel_l2"],
                    "max_err": h["max_err"],
                    "rel_l2": h["rel_l2"],
                    "frac_layer": h["frac_layer"],
                    "N": h["N"],
                    "est": " ".join(f"{k}={h['param_' + k]:.4f}"
                                    for k in solver.param_names),
                    "crlb": " ".join(f"{k}={solver.crlb.get('crlb_rel_' + k, float('nan')):.4f}"
                                     for k in solver.param_names),
                    **{f"est_{k}": h[f"param_{k}"] for k in solver.param_names},
                    **{f"crlb_{k}": solver.crlb.get(f"crlb_rel_{k}", float("nan"))
                       for k in solver.param_names},
                }
                rows.append(row)
                with open(os.path.join(a.outdir, "sweep.csv"), "w",
                          newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

    # ---- 汇总（按配置求均值）----
    agg = {}
    for r in rows:
        k = (r[key], r["strategy"])
        agg.setdefault(k, []).append(r)
    table = []
    for (v, strat), rs in sorted(agg.items()):
        def m(f):
            arr = np.array([r[f] for r in rs], dtype=float)
            return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        pm, ps = m("param_err_max")
        sm, ss = m("src_rel_l2")
        table.append({key: v, "strategy": strat, "n": len(rs),
                      "param_err_max": pm, "param_err_max_std": ps,
                      "src_rel_l2": sm, "src_rel_l2_std": ss})

    with open(os.path.join(a.outdir, "sweep_agg.json"), "w", encoding="utf-8") as fh:
        json.dump(table, fh, ensure_ascii=False, indent=2)

    md = ["| %s | strategy | param rel err | source rel L2 |" % key,
          "|---|---|---|---|"]
    for t in table:
        md.append(f"| {t[key]:g} | {t['strategy']} | "
                  f"{t['param_err_max']:.2%} | {t['src_rel_l2']:.2%} |")
    with open(os.path.join(a.outdir, "sweep.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    plotting.plot_sweep(table, key, os.path.join(a.outdir, f"sweep_{key}.png"),
                        xlabel=key)

    print("\n".join(md))
    print(f"\n[done] {len(rows)} runs, {(time.time() - t_start) / 60:.1f} min, "
          f"见 {a.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
