#!/usr/bin/env python
"""Case 5 的自检（不需要训练，几秒钟跑完）。

检查五件事：

1. **闭式右端项**。用双精度自动微分独立算 :math:`\\mathcal L_\\varepsilon u^*`，
   与 ``problem.forcing`` 的闭式对比，容差取机器精度量级
   （moving / static 两个算例都查）。
2. **定解条件**。``ic``/``bc`` 与 :math:`u^*` 的迹一致。
3. **求积权重**。kNN 权重之和等于所在流形的测度。
4. **参数化**。``SourceParams`` 的初值可复原，且取值始终落在给定区间内。
5. **可辨识性**。Fisher 矩阵条件数有限、CRLB 为正。

用法::

    python check_case5.py
    python check_case5.py --smoke     # 额外跑一次极小规模的端到端训练
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from case5 import (Case5Config, InversePINN, Points2D, SourceParams, crlb,
                   make_problem, make_sensors, knn_quadrature_weights)


def _operator_residual(prob, Z):
    """用自动微分算 ``L u*``（双精度）。"""
    Z = Z.clone().detach().requires_grad_(True)
    u = prob.exact(Z)
    ones = torch.ones_like(u)
    g = torch.autograd.grad(u, Z, ones, create_graph=True, retain_graph=True)[0]
    u_t = g[:, 2:3]
    gxx = torch.autograd.grad(g[:, 0:1], Z, ones, create_graph=True,
                              retain_graph=True)[0]
    gyy = torch.autograd.grad(g[:, 1:2], Z, ones, create_graph=True,
                              retain_graph=True)[0]
    lap = gxx[:, 0:1] + gyy[:, 1:2]
    return u_t - prob.eps * lap + prob.b1 * g[:, 0:1] + prob.b2 * g[:, 1:2] \
        + prob.q * u


def _layer_points(prob, n, rng):
    """一半的点放到层内，避免"层外 f≈0"把误差掩盖掉。"""
    t = rng.random(n)
    if prob.name == "moving":
        s = prob.layer_curve(t)
    else:
        s = np.full(n, prob.truth.xs)
    x = s + rng.uniform(-4, 4, n) * prob.sqrt_eps
    y = rng.random(n)
    return np.column_stack([np.clip(x, 0, 1), y, t])


def check_operator(eps_list=(1e-2, 1e-3, 1e-4)) -> bool:
    ok = True
    print("[1] 闭式右端项 vs 自动微分（float64）")
    for case in ("moving", "static"):
        for eps in eps_list:
            prob = make_problem(case, eps=eps, b=(1.0, 0.0), q=1.0, T=1.0)
            rng = np.random.default_rng(0)
            half = _layer_points(prob, 1500, rng)
            rest = rng.random((1500, 3))
            Z = torch.tensor(np.vstack([half, rest]), dtype=torch.float64)
            Lu = _operator_residual(prob, Z)
            f = prob.forcing(Z, prob.truth_dict)
            num = float((Lu - f).abs().max())
            den = float(f.abs().max())
            rel = num / max(den, 1e-300)
            flag = rel < 1e-10
            ok &= flag
            print(f"    {case:6s} eps={eps:.0e}: max|f|={den:.4e}  "
                  f"max|AD-f|={num:.3e}  相对={rel:.2e}  "
                  f"{'OK' if flag else 'FAIL'}")
    return ok


def check_data_conditions() -> bool:
    print("[2] 定解条件（初值 / 边界）")
    ok = True
    for case in ("moving", "static"):
        for eps in (1e-3, 1e-4):
            prob = make_problem(case, eps=eps)
            rng = np.random.default_rng(1)
            X = rng.random((500, 2))
            Z0 = np.column_stack([X, np.zeros(500)])
            d0 = float(np.abs(np.asarray(prob.ic(X))
                              - np.asarray(prob.exact(Z0))).max())
            Zb = []
            for axis, value in prob.faces():
                Xb = rng.random((100, 2))
                Xb[:, axis] = value
                Zb.append(np.column_stack([Xb, rng.random(100)]))
            Zb = np.vstack(Zb)
            db = float(np.abs(np.asarray(prob.bc(Zb))
                              - np.asarray(prob.exact(Zb))).max())
            flag = (d0 < 1e-14) and (db < 1e-14)
            ok &= flag
            print(f"    {case:6s} eps={eps:.0e}: |ic-u*(.,0)|={d0:.1e}  "
                  f"|bc-u*|={db:.1e}  {'OK' if flag else 'FAIL'}")
    return ok


def check_weights() -> bool:
    print("[3] kNN 求积权重归一化")
    ok = True
    rng = np.random.default_rng(2)
    for dim, measure, name in ((3, 1.0, "interior (D=3)"), (2, 1.0, "ic (D=2)")):
        P = rng.random((3000, dim))
        w = knn_quadrature_weights(P, measure, n_neighbors=4)
        tot = float(w.sum())
        flag = abs(tot - measure) < 1e-12 and w.min() > 0
        ok &= flag
        print(f"    {name}: sum(w)={tot:.12f} (target {measure:g})  "
              f"w_min={w.min():.3e}  {'OK' if flag else 'FAIL'}")
    return ok


def check_params() -> bool:
    print("[4] 源项参数化")
    ok = True
    cases = {
        "moving": ({"A": 2.0, "s0": 0.15, "v": 0.70}, ("A", "s0", "v")),
        "static": ({"A": 2.0, "xs": 0.30, "ys": 0.35, "rho": 0.08},
                   ("A", "xs", "ys", "rho")),
    }
    for case, (init, names) in cases.items():
        sp = SourceParams(names=names, init=init)
        v = sp.numpy_values()
        for k in names:
            d = abs(v[k] - init[k])
            ok &= d < 1e-5
            print(f"    {case:6s} {k}: 复原={v[k]:.6f} (target {init[k]})  "
                  f"diff={d:.1e}")
        with torch.no_grad():
            for k in names:
                if k != "A":
                    sp.raw[k].fill_(-50.0)
        lo_vals = sp.numpy_values()
        with torch.no_grad():
            for k in names:
                if k != "A":
                    sp.raw[k].fill_(50.0)
        hi_vals = sp.numpy_values()
        from case5 import DEFAULT_BOUNDS
        for k in names:
            if k == "A":
                continue
            lo, hi = DEFAULT_BOUNDS[k]
            inside = (lo_vals[k] >= lo - 1e-6) and (hi_vals[k] <= hi + 1e-6)
            ok &= inside
            print(f"    {case:6s} {k}: raw=-50 -> {lo_vals[k]:.4f}, "
                  f"raw=+50 -> {hi_vals[k]:.4f} (区间 [{lo},{hi}])  "
                  f"{'OK' if inside else 'FAIL'}")
    return ok


def check_identifiability() -> bool:
    print("[5] 可辨识性（Fisher 信息 / CRLB）")
    ok = True
    for case in ("moving", "static"):
        for eps in (1e-3, 1e-4):
            prob = make_problem(case, eps=eps)
            cfg = Case5Config(case=case, eps=eps, seed=0)
            Z, u_obs, sigma = make_sensors(prob, cfg)
            c = crlb(prob, Z, sigma)
            names = prob.param_names
            finite = np.isfinite(c["cond_fisher"]) and c["cond_fisher"] > 0
            rel = [c[f"crlb_rel_{k}"] for k in names]
            good = all(np.isfinite(rel)) and all(r > 0 for r in rel)
            ok &= finite and good
            print(f"    {case:6s} eps={eps:.0e}: n_data={Z.shape[0]} "
                  f"sigma={sigma:.2e} cond(F)={c['cond_fisher']:.2e}")
            print("      CRLB(rel): " + ", ".join(
                f"{k}={c['crlb_rel_' + k]:.2%}" for k in names))
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    print("=" * 66)
    print("Case 5 自检：二维奇异摄动反源问题（moving / static）")
    print("=" * 66)
    ok = True
    ok &= check_operator()
    ok &= check_data_conditions()
    ok &= check_weights()
    ok &= check_params()
    ok &= check_identifiability()

    if a.smoke:
        print("[6] 端到端冒烟训练（极小规模）")
        cfg = Case5Config(eps=1e-3, n_int0=512, n_ic0=32, n_bc0=16,
                          rounds=2, epochs_per_round=30, add_per_round=64,
                          candidate_pool=4000, eval_n=31, eval_nt=11,
                          add_ic_per_round=4, add_bc_per_round=4,
                          device="auto", verbose=True)
        solver = InversePINN(cfg)
        solver.run()
        last = solver.history[-1]
        finite = all(np.isfinite(last[k]) for k in
                     ("param_err_max", "src_rel_l2", "max_err"))
        ok &= bool(finite)
        print(f"    冒烟结果: param_err={last['param_err_max']:.2%} "
              f"src={last['src_rel_l2']:.2%} u_max={last['max_err']:.2e}  "
              f"{'OK' if finite else 'FAIL'}")

    print("=" * 66)
    print("全部通过" if ok else "有检查未通过，见上面 FAIL 行")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
