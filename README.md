# Direction-Adaptive PWDG for the Helmholtz Equation

**Shelvean Kapita**  
Department of Mathematics, Texas A&M University  
**August 2026**

This archive contains the cleaned research implementation accompanying

> **Direction-Adaptive Plane-Wave Discontinuous Galerkin Methods for the Helmholtz Equation**.

The code is written as a transparent numerical-analysis implementation rather than a general finite-element package.  The aim is that the main mathematical objects in the paper can be found directly in the source: local Trefftz functions, PWDG fluxes, the weighted skeleton residual, graph-Riesz normalization, local Cauchy-trace compression, complex directions, variable projection, and the MOVE/ENRICH/REFINE adaptive loop.

The cleanup in this archive is organizational only.  Dense numerical kernels remain vectorized with NumPy/SciPy, sparse assembly is retained, and the quadrature orders, nonlinear tolerances, continuation strategy, and audited precision settings are unchanged.  Long algebraic expressions have been broken into named intermediate quantities and comments have been added around the mathematical steps.

## Layout

```text
core/
    pwdg.py                  basic triangular-mesh utilities
    quadrature.py            edge and triangle quadrature
    direction_adaptive.py    PWDG state solve and direction adaptation
    variable_projection.py   Part B skeleton least squares
    exact_fields.py          exact/manufactured fields
    dtn_pwdg.py              exact circular DtN-PWDG solver
    compressed_dtn.py        local trace compression + graph-Riesz solve

experiments/
    experiment_transmission.py
    experiment_dtn_direction_recovery.py
    experiment_dtn_boundary_direction_oracle.py
    experiment_finite_direction_capacity.py
    experiment_finite_direction_enrichment.py
    polish_finite_direction_enrichment.py
    experiment_finite_direction_enrichment_11_20.py
    experiment_adaptive_threshold.py
    experiment_variable_projection_hankel.py
    precision_audit/
        experiment_dtn_tau_sweep.py

    # Additional development/reproducibility drivers are also retained.

data/
    reference data from earlier runs
    audited/                 audited DtN and finite-direction tables/figures
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run commands from the top level of the archive.

## Main experiments

### Transmission: propagating and evanescent directions

```bash
python -m experiments.experiment_transmission
```

The same complex-angle parameterization is used on both sides of the interface.  Frequency continuation recovers a real Snell direction in the propagating case and a complex/evanescent direction in the total-internal-reflection case.

### Exact circular DtN direction recovery

```bash
python -m experiments.experiment_dtn_direction_recovery --case centered --k 8 --nr 1 --ntheta 8 --p 3
python -m experiments.experiment_dtn_direction_recovery --case offcenter --k 8 --nr 1 --ntheta 8 --p 3
python -m experiments.experiment_dtn_direction_recovery --case scattering --k 8 --nr 1 --ntheta 8 --p 3
```

`core/dtn_pwdg.py` represents both circles exactly with polar-sector elements.  The circular DtN map is diagonal in Fourier modes and is assembled as a global outer-boundary block.

### Finite-direction capacity / ENRICH--MOVE continuation

The exact field is a nested sum of up to twenty plane waves.  The true directions are not used by the nonlinear search; they are used only to prescribe the exact data and to measure the recovered angles after the solve.

```bash
python -m experiments.experiment_finite_direction_capacity
python -m experiments.experiment_finite_direction_enrichment
python -m experiments.polish_finite_direction_enrichment
python -m experiments.experiment_finite_direction_enrichment_11_20
```

The continuation experiment is the one used to distinguish finite directional complexity from a broad angular spectrum.  The birth dictionary is a search device only; it does not prescribe the number of active directions.

### Hybrid h-direction adaptivity

```bash
python -m experiments.experiment_adaptive_threshold
```

This is the expensive benchmark.  The algorithm is written explicitly as

```text
SOLVE -> ESTIMATE -> MARK -> TEST -> ACT -> SOLVE.
```

During the direction-learning stage a marked element may MOVE, ENRICH, or REFINE.  After cycle 14 the learned direction sets are frozen and the calculation continues with residual-driven h-refinement.  The nominal local cap is `p_K <= 20`; it is inactive in the reported run, whose maximum local dimension is 8.

### DtN trace-cutoff audit

```bash
python -m experiments.precision_audit.experiment_dtn_tau_sweep
```

This reproduces the double-precision trace-cutoff audit on the exact 8-sector annulus with `k=8`, DtN truncation `N=40`, and 44-point edge/trace quadrature.  The exact field is plane-wave scattering by the sound-soft disk.

`core/compressed_dtn.py` makes the two numerical operations separate and visible:

1. diagonalize the local Cauchy-trace Gramian and retain modes according to `tau_rank`;
2. solve the compressed global system using the graph matrix as a Riesz-map preconditioner.

The audited CSV/figure outputs used in the manuscript are supplied under `data/audited/`.  They should be used as the record of the reported computations; the code is supplied so that the experiments can be rerun.

## Reproducibility conventions

- The nonlinear direction problems are nonconvex.  Deterministic initial states and continuation are therefore part of the experiment definition.
- Exact fields prescribe boundary data and measure errors.  They are not passed to the direction search unless a script is explicitly labeled as an oracle diagnostic.
- Raw Euclidean conditioning and graph-Riesz conditioning are reported separately.
- The all-Dirichlet skeleton residual equals the graph error for the manufactured exact solution used there.
- The DtN geometry is exact: there is no polygonal approximation of the circles `r=0.5` and `r=1`.
- Expensive audit results are written incrementally so completed parameter points are not lost if a long run is interrupted.

## Code style

Every principal source file identifies **Shelvean Kapita** and **August 2026** near the top.  Comments are concentrated around the numerical method rather than narrating ordinary Python syntax.  Vectorized `einsum`, sparse matrix assembly, and SciPy linear algebra are retained where they materially affect runtime.
