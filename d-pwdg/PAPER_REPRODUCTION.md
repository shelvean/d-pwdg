# Paper reproduction guide

**Manuscript:** *Direction-Adaptive Plane-Wave Discontinuous Galerkin Methods for the Helmholtz Equation*  
**Author:** Shelvean Kapita  
**Date:** August 2026

This directory is the referee-facing reproducibility package.  Run commands from its top level.
The `core/` drivers write below `results/` and the audited drivers below `generated/`; neither is tracked
by git.  The `data/` directory is read-only reference/audit data.

## Environment

Recommended: Python 3.10--3.13.  Install with

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

A tested environment on 18 August 2026 used Python 3.13.5, NumPy 2.3.5,
SciPy 1.17.0, pandas 2.2.3 and Matplotlib 3.10.8.

## One-command interface

```bash
python reproduce.py --list
python reproduce.py --quick
python reproduce.py --paper
```

`--paper` includes expensive nonlinear/adaptive calculations.  For practical checking,
run individual experiments with `python reproduce.py --experiment NAME`.

A failing step does not abort a group run.  The launcher records the failure, skips only
the steps that consume its outputs (the ordered Fig. 7 chain), runs everything else, and
ends with a per-step summary and a nonzero exit code.  Pass `--stop-on-fail` to abort at
the first failure instead.

## Manuscript-to-code map

| Manuscript result | Reproduction command | Main output |
|---|---|---|
| Uniform PWDG vs graph--Riesz conditioning | `python reproduce.py --experiment table1` | terminal table |
| Three hidden plane waves | `python reproduce.py --experiment table2` | `generated/threewave_large_mesh_results.csv` |
| Propagating/evanescent transmission | `python reproduce.py --experiment table3` | `generated/transmission_omega12_analytic_final.csv` |
| Circular DtN centered Hankel direction recovery | `python reproduce.py --experiment dtn-centered` | `results_dtn/dtn_centered_*.csv`, `..._directions.png` |
| Circular DtN off-center Hankel direction recovery | `python reproduce.py --experiment dtn-offcenter` | `results_dtn/dtn_offcenter_*.csv`, `..._directions.png` |
| Part B variable projection Hankel check | `python reproduce.py --experiment variable-projection` | `generated/hankel_direct_multidir_joint.csv` |
| Finite-direction fixed-capacity comparison | `python reproduce.py --experiment finite-capacity` | `results/finite_direction_capacity_M1_M8.csv` |
| ENRICH--MOVE continuation through M=20 | `python reproduce.py --experiment finite-directions` | `results/finite_direction_enrichment*.csv` |
| Double-precision DtN trace-cutoff audit | `python reproduce.py --experiment dtn-tau-audit` | `results/precision_audit/tau_sweep_audited.csv` |
| Hybrid h-direction adaptivity | `python reproduce.py --experiment hybrid` | `results/adaptive_threshold/*.csv` |

The manuscript figures based on long precision runs and the finite-direction continuation
are also supplied under `data/audited/` as an immutable record of the reported calculations.

## Precision audit qualification

The **double-precision** cutoff sweep is fully rerunnable using
`experiments/precision_audit/experiment_dtn_tau_sweep.py`.
The paper additionally reports trace spectra computed at 50 decimal digits and selected
higher-arithmetic solves.  Their audited CSV/figure outputs are retained in `data/audited/`.
The standalone high-precision generator was not present in the surviving cleaned source tree,
so this archive does not claim that those particular data can presently be regenerated from
source.  This limitation is explicit to avoid overstating reproducibility.

## Determinism

The nonlinear problems are nonconvex.  Fixed Sobol seeds, starting fans, continuation schedules,
cutoffs, and stopping tolerances are part of the experiment definitions.  Exact solutions are
used to impose data and measure errors; they are not supplied to the nonlinear direction search
except in scripts explicitly labelled `oracle`.

## Integrity

`SHA256SUMS.txt` records hashes for every source and reference-data file in this archive.

## Audited manuscript source

The directory `audited_source/` contains the original Python source used during the
manuscript experiments, before the readability refactor in `core/`.  For numerical
values quoted in the manuscript, prefer these audited drivers when an audited driver
exists.  In particular:

```bash
python reproduce_manuscript.py transmission-and-conditioning
python reproduce_manuscript.py threewave-large-mesh
```

A verification run of the audited transmission driver on this system reproduced the
reported 8-triangle errors `1.4052063417e-15` (69 degrees) and
`1.9963819849e-15` (29 degrees), and the 32-triangle errors
`1.1265076008e-15` and `4.2577038056e-15`.  The resulting CSV is preserved in
`validation/transmission_omega12_analytic_final.csv`.

The structured `core/` tree is retained because it is substantially easier to read and
audit mathematically, but tiny changes in optimization stopping/assembly order can move
roundoff-level results by several digits.  It should not be substituted for the audited
source when the purpose is to regenerate the exact printed table values.

## Table 3 verification

For the submitted Table 3 transmission values, use

    python verify_table3_exact.py            # manuscript accuracy (default)
    python verify_table3_exact.py --bitwise  # validation-machine digit record

The default mode asserts what Table 3 claims and is machine independent: the
problem sizes exactly, both relative L2 errors at machine precision and of the
same roundoff order as the reference record, the transmitted state at its
analytic tangential-phase-matching value in the propagating and the evanescent
case, the incident and reflected directions exactly, kappa_GR to ten digits,
kappa_2(A) within the roundoff floor of a condition number that size, and the
nonlinear evaluation counts within two.

`--bitwise` is the original digit-for-digit comparison at `rtol = 5e-13`.  It
is a record of the validation machine: BLAS kernel dispatch differs across
CPUs, so the last bits of roundoff-limited quantities are hardware dependent
even with identical numpy/scipy versions.

Both modes invoke the audited analytic-Jacobian continuation in
`audited_source/transmission_omega12_experiments.py`.  The readable `core/`
transmission optimizer uses a finite-difference Jacobian and is not the source
of the printed roundoff-level L2 errors.

The exact manuscript configuration is:

- initial complex-angle starts `STARTS_A` in the audited driver;
- frequency continuation `1, 2, 4, 6, 8, 10, 12`;
- `N=2`: optimization edge quadrature `nq=16`;
- `N=4`: optimization edge quadrature `nq=14`;
- analytic Jacobian from `all_analytic`;
- SciPy `least_squares`, method `trf`, `x_scale='jac'`;
- `ftol=xtol=gtol=1e-12`;
- `max_nfev=120` per continuation stage;
- final state reassembled with `nq=max(nq,24)`;
- relative L2 error evaluated by `l2_error(s,20)`.

On the tested environment `--bitwise` reproduces all four submitted rows
exactly to the stored floating-point values; see
`validation/table3_exact_verification.txt`.
