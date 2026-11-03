# Code for *Residual-driven adaptive collocation for physics-informed neural networks*

This repository contains the code that produced the numerical results of the
paper. It solves singularly perturbed parabolic problems

$$
\partial_t u - \varepsilon \Delta u + \mathbf{b}\cdot\nabla u + q\,u = f
\quad\text{in } \Omega\times(0,T],
$$

with a physics-informed neural network whose collocation points are enriched
where the residual is largest, and whose loss is weighted by the volume that each
collocation point occupies.

The code is organised **by case**: every case is a self-contained folder with its
own copy of the solver package, so each folder can be copied and run on its own.

---

## Layout

```
PINN_singular_perturbation_code/
├── README.md                 this file
├── requirements.txt          Python dependencies
├── case1/                    boundary layer, one space dimension
├── case2/                    corner layer, two space dimensions
├── case3/                    layer moving in time
├── case4/                    non-rectangular spatial domain
└── case5/                    inverse source problem
```

| Folder | Problem | Space–time dimension | Collocation sets | Quadrature weights |
|---|---|---|---|---|
| `case1` | boundary layer at the inflow edge, at the outflow edge, and an interior layer at a turning point | $1+1$ | interior, initial line, two lateral lines | exact discrete Voronoi (auxiliary grid, $n=256$) |
| `case2` | corner layer in the square | $2+1$ | interior, initial plane, four lateral planes | $k$-nearest-neighbour volume estimate, $k=4$ |
| `case3` | layer whose position moves with time | $2+1$ | same as `case2` | same as `case2` |
| `case4` | layer near the curved boundary of a non-rectangular domain | $2+1$ | same as `case2` | same as `case2` |
| `case5` | unknown source recovered from sparse interior sensors | $2+1$ | same as `case2` plus sensor set | same as `case2` |

`case2` and `case3` share the same solver; they differ only by the command-line
flag `--case case2` or `--case case3`, and each folder carries its own copy so
that it can be run independently.

---

## Requirements

* Python ≥ 3.9
* `torch` ≥ 2.0, `numpy` ≥ 1.24, `matplotlib` ≥ 3.7 (nothing else; no SciPy, no
  FEniCS)

```bash
pip install -r requirements.txt
```

The results in the paper were obtained with Python 3.10 and PyTorch 2.5.1
(CUDA 12.1) on a single GPU. Every script also runs on the CPU: pass `--device cpu`, or simply omit
`--device cuda`. A CPU run of a single case is roughly one order of magnitude
slower.

---

## The method in one paragraph

The loss is

$$
J_X(\theta) = \sum_i \omega_i\, r(z_i;\theta)^2
+ \lambda_{ic}\sum_j \omega_j^{\mathrm{ic}}\,(u_\theta-u_0)^2(z_j)
+ \lambda_{bc}\sum_k \omega_k^{\mathrm{bc}}\,(u_\theta-g)^2(z_k),
$$

where $r=\mathcal{L}_\varepsilon u_\theta - f$ is the PDE residual and
$\omega_i = |V_i|$ is the volume of the Voronoi cell of $z_i$ — that is, the
volume of the region that the point $z_i$ represents. With these weights the
discrete sum is a consistent quadrature of the continuous residual norm,
independently of how the points are distributed; with any other weight the
functional depends on the sampling density.

Every round consists of $E$ Adam steps followed by one enrichment step:
the residual $|r(\cdot;\theta_k)|$ is evaluated on a candidate pool of $N_c$
points drawn uniformly from $Q$, and the $m$ candidates with the largest residual
are appended to the collocation set (residual-based adaptive refinement, RAR).
Initial and boundary points are enriched by the same rule, using the constraint
violation as the indicator. The parameters are warm started across rounds.

---

## Case 1 — boundary layer in one space dimension

`case1/` solves the $1+1$-dimensional problem on $(0,1)\times(0,T]$ with three
manufactured solutions: a boundary layer at $x=0$ (`left`), one at $x=1`
(`right`), and an interior layer at the turning point of a coefficient that
changes sign (`interior`).

| File | Purpose |
|---|---|
| `spp_pinn/problems.py` | problem base class and the three manufactured solutions (exact solution, source term, data) |
| `spp_pinn/solver.py` | `TrainConfig`, the adaptive solver, and the fixed-set reference run |
| `spp_pinn/sampling.py` | low-discrepancy initial points, RAR selection, collocation sets, Voronoi weights |
| `spp_pinn/model.py` | fully connected tanh network |
| `spp_pinn/plotting.py` | figures |
| `check_problems.py` | **self-check**: verifies $\mathcal{L}_\varepsilon u^\*-f=0$ by automatic differentiation |
| `demo.py` | single run with a per-round progress table and figures |
| `run_all_eps.py` | $\varepsilon$-sweep, adaptive against fixed, writes figures |
| `run_experiments.py` | batch study: strategy comparison, $\varepsilon$-sweep, boundary cases |
| `make_figures.py` | re-plots stored results without retraining |
| `fig_case1.py`, `seeds_eps.py`, `ablate_max.py` | auxiliary studies (figures, seed variability, threshold ablation) |

Typical use:

```bash
cd case1

# 0) verify the manufactured solutions (no training, a few seconds)
python check_problems.py

# 1) one run, with figures
python demo.py --eps 1e-5 --device cuda

# 2) the paper configuration for the epsilon-sweep
python run_all_eps.py --rounds 20 --epochs 1000 --n-int0 1024 --add 150 \
                      --device cuda --outdir eps_full
```

Ablations use the same entry points:

```bash
# uniform placement instead of residual-driven placement
python demo.py --eps 1e-5 --strategy uniform
# arithmetic mean weights instead of volumes
python demo.py --eps 1e-5 --weighting mean
# no enrichment at all
python demo.py --eps 1e-5 --strategy fixed
```

`demo.py` accepts `--strategy {residual, distribution, uniform, fixed}` and
`--weighting {voronoi, mean}`.
The additional `distribution` rule draws the new points with probability
proportional to a power of the residual; it is implemented for completeness but
is **not** used anywhere in the paper, whose results use `residual` throughout. The uniform rule draws from the *same* candidate
pool with the same cardinality, so the comparison isolates the effect of
placement from the effect of adding points.

> **Legacy files.** `spp_pinn/mesh.py` and `spp_pinn/mesh_variant.py` are an
> early quadtree $h$-adaptive prototype. They are not used anywhere in the
> method or in the results, and can be deleted.

---

## Case 2 — corner layer in two space dimensions

`case2/` solves the problem in $(0,1)^2\times(0,T]$, where the solution has a
corner layer at the origin. The solver is the two-dimensional counterpart of
`case1`: the four lateral planes and the initial plane are carried as separate
collocation sets, and the quadrature weights are estimated by the
$k$-nearest-neighbour rule of Section 2.3 rather than by exact Voronoi cells.

| File | Purpose |
|---|---|
| `spp_pinn/nd.py` | **core**: problem base class, corner-layer problem, kNN weights, collocation sets, adaptive solver |
| `check_problems_nd.py` | self-check of the manufactured solution, and kNN weights against exact Voronoi |
| `check_weights_nd.py` | quadrature accuracy of the three weighting schemes |
| `demo_2d.py` | single run with figures |
| `run_all_eps.py` | $\varepsilon$-sweep (`--case case2`) |
| `fig_cases.py`, `norm_2d.py`, `diag_2d.py` | figure and diagnostics scripts |

```bash
cd case2
python check_problems_nd.py
python check_weights_nd.py
python run_all_eps.py --case case2 --rounds 20 --epochs 1000 \
                      --n-int0 1024 --add 150 --device cuda --outdir eps_full
```

---

## Case 3 — a layer that moves in time

The code is the same as for case 2; only the problem selected on the command line
changes. The exact solution contains a layer whose position $s(t)$ is not
supplied to the solver — the enrichment rule has to find it, round after round.

```bash
cd case3
python run_all_eps.py --case case3 --rounds 20 --epochs 1000 \
                      --n-int0 1024 --add 150 --device cuda --outdir eps_full
```

---

## Case 4 — non-rectangular spatial domain

`case4/` solves the problem on a domain whose boundary is a smooth closed curve,
so that the layer follows a curved boundary rather than a straight edge. The
initial and boundary points live on manifolds of dimension $d=2$, while the
interior points live in dimension $D=3$; the kNN estimator uses the intrinsic
dimension of each set.

| File | Purpose |
|---|---|
| `run_case4.py` | $\varepsilon$-sweep, adaptive against fixed uniform |
| `check_case4.py` | consistency of the data with the exact solution |
| `check_math.py` | symbolic/numerical checks of the domain and the manufactured solution |
| `spp_pinn/` | domain, problem, collocation sets and solver for this geometry |

```bash
cd case4
python check_math.py
python check_case4.py
python run_case4.py --device cuda --outdir runs/full
python run_case4.py --eps 1e-4 --rounds 5 --epochs 500 --outdir runs/quick
```

---

## Case 5 — inverse source problem

`case5/` recovers an unknown source (its amplitude, position and width) from
noisy values of $u$ on a small set of interior sensors, with the same
residual-driven enrichment in the forward solve. The forward and inverse
problems are solved in one optimisation.

| File | Purpose |
|---|---|
| `run_case5.py` | single inverse run; `--smoke` finishes in about a minute |
| `run_sweep.py` | sweeps over noise level, sensor count and $\varepsilon$ |
| `check_case5.py` | self-check of the data, the conditioning and the Cramér–Rao bounds |
| `make_figs.py` | figures from stored runs |
| `case5/` | problem, quadrature, model, solver, plotting |

```bash
cd case5
python check_case5.py
python run_case5.py --smoke
python run_case5.py --eps 1e-4 --noise 0.01 --sensors 24 --device cuda
python run_sweep.py --sweep noise --eps-list 1e-4 --seeds 3
```

---

## Configuration reported in the paper

| Item | Value |
|---|---|
| Network | fully connected, $3\times64$ units, $\tanh$, Glorot normal initialisation, inputs mapped affinely to $[-1,1]^{d+1}$ |
| Optimiser | Adam, learning rate $2\times10^{-3}$, cosine annealing to $2\%$ within a round, $\times0.9$ between rounds, full batch |
| Rounds / epochs | $K=20$ rounds of $E=1000$ Adam steps ($21\,000$ steps in total) |
| Points | $n_0=1024$ interior points, $m=150$ appended per round, $N_c=2\times10^{4}$ candidates |
| Initial / boundary points | $64$ / $64$ per set ($d=1$), $128$ / $64$ per face ($d=2$), enriched by $16$ per round |
| Loss weights | $\lambda_{ic}$, $\lambda_{bc}$ calibrated once at initialisation, then fixed |
| Quadrature | exact discrete Voronoi weights ($d=1$), kNN with $k=4$ ($d=2$) |

---

## Outputs

Every runner writes into the directory given by `--outdir`:

* `*.npz` — arrays (prediction, exact solution, collocation sets, per-round errors);
* `*.json` — one record per run (configuration, errors, layer statistics);
* `*.pdf` — figures.

Nothing is overwritten silently: each run uses a tag built from the case, the
strategy and $\varepsilon$, so several runs can share one output directory.

---

## Notes on the implementation

* The residual is formed by automatic differentiation; the Laplacian costs a
  second reverse pass, so one training step costs one forward and two reverse
  passes.
* The weights are recomputed once per round, before training, at a cost of
  $O(N^2)$ per set with the kNN estimator; a KD-tree would reduce this to
  $O(N\log N)$.
* The point sets are reproducible: the scripts that expose `--seed` default to
  seed 0, and the initial points come from a Halton sequence rather than a
  pseudorandom draw.
