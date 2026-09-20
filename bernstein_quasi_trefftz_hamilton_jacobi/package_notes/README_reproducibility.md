# ANM stationary Hamilton--Jacobi manuscript package

**Manuscript:** `ANM_Bernstein_QuasiTrefftz_Hamilton_Jacobi_Marmousi.pdf`

**Source:** `main.tex`

The paper develops a global C0 Bernstein degree-(p-1) residual-moment method for stationary Hamilton--Jacobi equations. The eikonal equation is the principal numerical application, with non-eikonal anisotropic-quadratic and quartic-convex tests establishing that the construction is genuinely Hamilton--Jacobi rather than eikonal-specific.

## Build

Run twice:

```bash
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The source uses `elsarticle` in preprint format for Applied Numerical Mathematics.

## Numerical convention

All reported degree sweeps use the single odd sequence

```text
p = 3, 5, 7, 9, 11, 13.
```

Even degrees remain admissible; the manuscript reports one parity consistently. Every convergence graph has at least six computed points. The carrier-frequency graph has nine.

## Optimized solver

`code/optimized_hj.py` contains the production path used for the revised experiments:

- batched element values, gradients, Hamiltonian evaluations, and moment contractions;
- analytic Jacobians assembled directly into a fixed CSR sparsity pattern;
- column-scaled sparse Gauss--Newton steps solved with LSMR;
- backtracking on the true nonlinear residual norm;
- exact triangular Bernstein degree elevation for p-to-p+2 continuation;
- no dense global Jacobian or dense QR factorization.

The optimized code contains factored eikonal, general HJ, and multiplicatively factored point-source solvers.

## Numerical provenance

- `data/general_hj_benchmarks.csv`: anisotropic-quadratic and quartic-convex HJ sweeps, p=3,...,13 odd.
- `data/polynomial_recovery_odd.csv`: eikonal exact polynomial recovery.
- `data/smooth_p_odd.csv`: smooth eikonal odd-degree sweep.
- `data/smooth_h_p5.csv`: six-level smooth mesh sweep at p=5.
- `data/smooth_inflow_p5.csv`: inflow-only smooth validation at p=5.
- `data/high_frequency_benchmark_results.csv`: Potter--Cameron and linear-speed point-source data.
- `data/pointsource_p_sweep.csv`: external point-source odd-degree sweep.
- `data/stratified_p5_h_sweep.csv`: six-level heterogeneous stratified sweep at p=5.
- `data/focal_caustic_p_sweep.csv`: focal-caustic odd-degree sweep.
- `data/carrier_frequency_sweep.csv`: reconstructed field errors through carrier frequency 20000.
- `data/treister_haber_case1_odd_h1.csv`: fixed-h=1 Treister--Haber case-1 odd-degree benchmark.

Main drivers are `run_examples.py`, `hj_general_benchmarks.py`, `run_stratified_heterogeneous.py`, `run_high_frequency_benchmarks.py`, `run_focal_caustic_sweep.py`, `run_treister_haber_odd.py`, `run_carrier_frequency_sweep.py`, and `regenerate_odd_figures.py`.

## Scope note

The theory concerns smooth classical branches. The global least-squares estimate is conditional on stability of the assembled linearized characteristic map. The formulation is not claimed to be a monotone viscosity-solution scheme. The focal experiment uses a known singular factor and the two-source experiment uses explicit branchwise minimization; the paper does not claim automatic generation of multivalued post-caustic phases.


## Marmousi benchmark

The manuscript now includes a causal-plus-Bernstein Marmousi first-arrival experiment.
The numerical results used in the paper are in `data/marmousi_smooth_h_sweep_p7.csv` and
`data/marmousi_raw_h_sweep_p7.csv`.  The figures are in `figures/marmousi_*.pdf`.
The drivers in `code/marmousi_*` expect external Marmousi velocity arrays; the velocity
files themselves are intentionally not redistributed in this archive.  The benchmark uses
a surface source at x=4.6 km, a native 8 m causal reference grid, a 73 x 25 coarse causal
branch selector, and p=7 Bernstein h-continuation.

To reproduce the smooth-model benchmark from a downloaded 376 x 1151 velocity file:

```bash
python code/run_marmousi_smooth.py \
  --velocity /path/to/vel_marmousi_smooth400_376x1151.csv.txt \
  --output-dir reproduced_marmousi
```

Optionally add `--raw-velocity /path/to/vel_marmousi_376x1151.csv.txt` to repeat the
unsmoothed robustness sweep.  The script recomputes the native causal reference, the
coarse branch selector, all six p=7 h-continuation levels, the CSV table, and the three
Marmousi figures.
