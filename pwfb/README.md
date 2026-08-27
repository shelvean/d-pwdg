# Reproduction package

**Adaptive Local Representations for Helmholtz Trefftz Discontinuous Galerkin Methods**  
Shelvean Kapita, Department of Mathematics, Texas A&M University

This archive contains the executable Python drivers, reference CSV data, and generated figures accompanying the JCP submission. The numerical algorithms are unchanged from the submitted package; a few output paths were made portable so the scripts write inside this archive rather than to an environment-specific `/mnt/data` directory.

## 1. Environment

Recommended:

- Python 3.13
- NumPy, SciPy, pandas, mpmath, Matplotlib
- A LaTeX installation with `pdflatex` only for scripts that regenerate PGF/PDF figures

Create a clean environment from the package root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The high-precision Hankel audit has its own historically audited pins in
`code/reproduction_highp_hankel/requirements.txt`.

## 2. Package check

From the package root:

```bash
python verify_package.py
```

This checks Python syntax/imports, required reference files, and the shipped high-precision audit results. It is intentionally fast and does not rerun every expensive experiment. To recompute the independent high-precision audit from scratch, run:

```bash
python verify_package.py --full
```

## 3. Principal manuscript reproductions

All commands below are run from the package root.

### High-order centered-Hankel PW sweep

```bash
python code/run_highp_global_dtn.py
```

Writes `data/highp_global_dtn_pwdg_reproduced.csv`. Compare with the reference data used for the manuscript in `data/highp_global_dtn_pwdg.csv`.

### High-precision independent Hankel audit

```bash
cd code/reproduction_highp_hankel
python reproduce_highp_hankel.py --out results
python verify.py
cd ../..
```

The audit uses arbitrary precision for the independent boundary-quadrature validation. See the README in that directory for details.

### Rank-threshold sensitivity

Run one threshold at a time, for example:

```bash
TAU=1e-12 python code/run_highp_global_dtn_tau.py
```

The generated row is written to `data/highp_global_dtn_results.csv`. The manuscript reference sweep is `data/highp_rank_threshold_sensitivity.csv`.

### Stability-aware PW-FB continuation

Reference implementation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python code/run_hybrid_global_dtn_esprit_stable.py
```

Faster consecutive-order Bessel implementation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python code/run_hybrid_global_dtn_esprit_stable_fast.py
```

The reference manuscript tables are in `data/hybrid_selector_global_results.csv` and `data/hybrid_selector_candidates.csv`. Generated results are written beside them with distinct filenames. See `code/FAST_BESSEL_README.md` for the two Bessel kernels.

### Selection from estimated traces

```bash
python code/run_coarse_trace_selector.py
```

Writes `data/coarse_trace_selector.csv`.

### Global-search versus local-trace timing

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python code/run_timing_levels.py
```

Writes `data/timing_levels.csv`. Wall-clock timings are machine-dependent; representation choices and errors are the scientifically relevant reproducibility checks.

### ESPRIT capacity experiments

For the original `N=81` capacity test:

```bash
python code/esprit_capacity_experiment.py
```

For the extended `N=401` test used in the manuscript:

```bash
python code/esprit_capacity_extended.py
```

These write reproduced CSV files into `data/` and reproduced PDF figures into `figures/`.

Conditioning extensions:

```bash
python code/esprit_conditioning_selected.py
python code/esprit_conditioning_long_windows.py
```

The longer continuation driver `code/esprit_conditioning_continuation.py` is also included for the extended study.

### Parallel elementwise ESPRIT scaling

```bash
python code/run_esprit_parallel.py
python code/make_parallel_table.py
```

The first command measures process-level parallel scaling and writes `data/esprit_parallel.csv`; the second prints manuscript-ready table rows. Because timings depend strongly on the machine, this experiment should be rerun on the hardware being reported.

## 4. Additional local plane-wave experiments

```bash
python code/hankel_pw_local_experiment.py
python code/hankel_pw_fan_experiment.py
python code/make_hankel_pw_figures.py
```

These produce the centered/off-center Hankel local-representation data and figures.

## 5. Figure regeneration

Core JCP figures from closed-form expressions:

```bash
python code/make_figures_jcp.py
```

High-p summary panels from the shipped CSV data:

```bash
python code/make_highp_figures.py
```

Some figure scripts use the Matplotlib PGF backend and therefore require a working LaTeX installation.

## 6. Directory layout

- `code/` - executable drivers and numerical kernels
- `data/` - CSV files used in the manuscript and generated comparison files
- `figures/` - manuscript figures and regenerated figures
- `results/` - empty convenience directory for user-generated output
- `requirements.txt` - tested top-level Python environment
- `verify_package.py` - fast package validation
- `MANIFEST.sha256` - SHA-256 checksums for package files

## 7. Numerical reproducibility notes

The computations use IEEE binary64 unless a script explicitly states otherwise. Roundoff-level entries can vary slightly across BLAS/LAPACK implementations, processor architectures, and library versions. The high-precision Hankel subpackage is included specifically to audit the most delicate high-order values independently. Timing tables are expected to vary by machine.

No external datasets are required by the supplied drivers.
