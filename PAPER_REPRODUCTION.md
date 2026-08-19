# Paper reproduction guide

**Manuscript:** *Direction-Adaptive Plane-Wave Discontinuous Galerkin Methods for the Helmholtz Equation*  
**Author:** Shelvean Kapita  
**Date:** August 2026

This directory is the referee-facing reproducibility package.  Run commands from its top level.
All newly generated files are written below `results/`; the `data/` directory is read-only reference/audit data.

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

## Manuscript-to-code map

| Manuscript result | Reproduction command | Main output |
|---|---|---|
| Uniform PWDG vs graph--Riesz conditioning | `cd paper_auxiliary && python uniform_conditioning_graph_riesz.py` | terminal table |
| Three hidden plane waves | `cd paper_auxiliary && python test_threewave_auto.py` | terminal table |
| Propagating/evanescent transmission | `python reproduce.py --experiment transmission` | `results/transmission_recovery.csv` |
| Circular DtN centered Hankel direction recovery | `python reproduce.py --experiment dtn-centered` | DtN continuation CSV/plots |
| Circular DtN off-center Hankel direction recovery | `python reproduce.py --experiment dtn-offcenter` | DtN continuation CSV/plots |
| Part B variable projection Hankel check | `python reproduce.py --experiment variable-projection` | `results/variable_projection_hankel.csv` |
| Finite-direction fixed-capacity comparison | `python reproduce.py --experiment finite-capacity` | `results/finite_direction_capacity_M1_M8.csv` |
| ENRICH--MOVE continuation through M=20 | `python reproduce.py --experiment finite-directions` | continuation CSVs |
| Double-precision DtN trace-cutoff audit | `python reproduce.py --experiment dtn-tau-audit` | `results/precision_audit/tau_sweep_audited.csv` |
| Hybrid h-direction adaptivity | `python reproduce.py --experiment hybrid` | hybrid/pure-h history CSVs |

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
