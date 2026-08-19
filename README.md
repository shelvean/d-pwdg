# Direction-Adaptive PWDG for the Helmholtz Equation

**Shelvean Kapita**  
Department of Mathematics, Texas A&M University  
**August 2026**

This repository contains the cleaned research implementation accompanying

> **Direction-Adaptive Plane-Wave Discontinuous Galerkin Methods for the Helmholtz Equation**

together with the manuscript itself under [`paper/`](paper/) —
[LaTeX source](paper/Kapita_Direction_Adaptive_PWDG_Helmholtz.tex),
[compiled PDF](paper/Kapita_Direction_Adaptive_PWDG_Helmholtz.pdf), and the
figures it includes.  [`REPRODUCE.md`](REPRODUCE.md) maps every numbered table
and figure of the PDF onto the script that produces it.

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

audited_source/
    the original pre-refactor source that produced the manuscript numbers
    (import closure of the three manuscript drivers only)

validation/
    verification record for the audited transmission driver

paper/
    Kapita_Direction_Adaptive_PWDG_Helmholtz.tex   manuscript source
    Kapita_Direction_Adaptive_PWDG_Helmholtz.pdf   compiled manuscript
    sn-jnl.cls, sn-mathphys-num.bst                Springer Nature style files
    figures/                 figures included by the manuscript

results/                     created at run time by the drivers (git-ignored)
generated/                   created at run time by audited_source (git-ignored)

reproduce.py                 one-command launcher for the core/ experiments
reproduce_manuscript.py      launcher that prefers the audited drivers
verify_table3_exact.py       asserts the four submitted Table 3 rows
```

Two source trees are present.  `core/` + `experiments/` is the cleaned,
readable implementation.  `audited_source/` is the original source that
produced the printed manuscript values.  Where a table value is at roundoff
level the two can differ in the last digits, so **prefer `audited_source/` when
the purpose is to regenerate a printed number** — `REPRODUCE.md` shows a
measured instance and records what each tree reproduces.

Five documents sit at the top level.  `README.md` (this file) explains how to
install and run.  [`METHOD_MAP.md`](METHOD_MAP.md) locates the mathematics of
the paper in the source.  [`REPRODUCE.md`](REPRODUCE.md) maps the numbered
tables and figures onto the drivers that produce them, and states which ones
this archive does not regenerate.  [`VALIDATION.md`](VALIDATION.md) records the
checks made when the source was cleaned.  [`PAPER_REPRODUCTION.md`](PAPER_REPRODUCTION.md)
is the referee-facing guide to the one-command launchers.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run commands from the top level of the archive.

## Main experiments

Each driver below is described here in terms of what it computes.  For the
output-side view — which script writes which file, and which table or figure of
the manuscript that file becomes — see [`REPRODUCE.md`](REPRODUCE.md).  All
drivers write into `results/`, which is not tracked by git; the committed
reference outputs are under `data/`.

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

## Citation

If you use this code or refer to these experiments, please cite the
accompanying manuscript:

> Shelvean Kapita, *Direction-Adaptive Plane-Wave Discontinuous Galerkin
> Methods for the Helmholtz Equation*, Department of Mathematics, Texas A&M
> University, August 2026.

`CITATION.txt` carries the same reference in plain text, and the manuscript
itself is under [`paper/`](paper/).

## License

The code, data and manuscript in this repository are released under
[CC0 1.0 Universal](LICENSE): the author waives copyright and related rights
worldwide, so no permission is needed to use, modify or redistribute them.
Attribution is not legally required; a citation of the manuscript is the
customary scholarly courtesy.

Two files are **not** the author's to license and are not covered by the CC0
waiver: `paper/sn-jnl.cls` and `paper/sn-mathphys-num.bst` are the Springer
Nature journal class and bibliography style, redistributed here only so the
manuscript compiles as submitted.  They remain subject to Springer Nature's own
terms.
