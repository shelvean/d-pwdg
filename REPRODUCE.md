# Reproducing the manuscript results

**Shelvean Kapita — August 2026**

This repository holds only what the submitted manuscript needs. Every source
file here produces a numbered table or figure, or is imported by something that
does — there are no development or diagnostic scripts. This file maps each
numbered item of `paper/Kapita_Direction_Adaptive_PWDG_Helmholtz.pdf` onto the
driver that produces it; `METHOD_MAP.md` maps the mathematics onto the source.

Numbering follows the compiled PDF supplied under `paper/`.

## Two implementations, and which produced the paper

`audited_source/` is the implementation that produced the printed values. It
optimizes the directions with the **analytic** Jacobian — the Wirtinger
derivatives of Section 7.1, which the paper describes as "analytic residual
derivatives". Tables 1, 2 and 3 and Fig. 1 come from here.

`core/` + `experiments/` is the readable implementation, retained because it is
the **only** implementation of the DtN experiments (Tables 4 and 6, Fig. 4),
the finite-direction continuation (Table 7, Fig. 7) and the hybrid benchmark
(Table 8, Fig. 8). The audited tree contains no DtN code at all. It optimizes
with a finite-difference Jacobian, which on a zero-residual problem sets a floor
on how exactly the directions can be identified; that is why Table 3 is taken
from the audited driver and asserted by `verify_table3_exact.py` rather than
recomputed here.

## Conventions

| Directory | Role |
| --- | --- |
| `results/`, `generated/` | Created at run time, both git-ignored. `core/` drivers write to `results/`, audited drivers to `generated/`. |
| `data/` | Reference outputs from the runs reported in the paper, committed so a new run can be compared. |
| `data/audited/` | The audited published record. Where the paper and `data/audited/` disagree, `data/audited/` is authoritative — see `data/audited/README.md`. |
| `validation/` | Verification records for the audited transmission driver, including the Table 3 pass record. |

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python reproduce.py --list       # every entry, with the item it produces
python reproduce.py --quick      # Tables 1 and 3 and Fig. 1
python reproduce.py --paper      # everything rerunnable
```

The drivers are deterministic: initial states and continuation schedules are
part of the experiment definition, not free parameters.

## Map from paper items to code

| Paper item | `reproduce.py --experiment` | Driver | Writes |
| --- | --- | --- | --- |
| Table 1 — uniform vs. graph–Riesz conditioning | `table1` | `audited_source/uniform_conditioning_graph_riesz.py` | terminal table |
| Table 2 — three hidden plane waves | `table2` | `audited_source/test_threewave_large_mesh.py` | `generated/threewave_large_mesh_results.csv` |
| Table 3 — Part A transmission recovery | `table3` | `verify_table3_exact.py`, which runs the audited continuation and **asserts** all four submitted rows | `generated/transmission_omega12_analytic_final.csv` |
| Fig. 1 — transmission fields, ω=12 | `fig1` | `audited_source/plot_transmission_visualization.py` | `generated/transmission_solution_{69,29}deg_2d.png` |
| Table 4 / Fig. 4 — DtN Hankel tests, learned wave vectors | `dtn-centered`, `dtn-offcenter` | `experiments/experiment_dtn_direction_recovery.py` | `results_dtn/dtn_<case>_*.csv`, `..._directions.png` |
| Table 6 — trace-cutoff sweep | `dtn-tau-audit` | `experiments/precision_audit/experiment_dtn_tau_sweep.py` | `results/precision_audit/tau_sweep_audited.csv` |
| Table 7 — finite-direction capacity | `finite-capacity` | `experiments/experiment_finite_direction_capacity.py` | `results/finite_direction_capacity_M1_M8.csv` |
| Fig. 7 — continuation in the number of exact directions | `finite-directions` | the ordered three-step chain below | `results/finite_direction_enrichment*.csv` |
| Table 8 / Fig. 8 — threshold-driven hybrid benchmark | `hybrid` | `experiments/experiment_adaptive_threshold.py` | `results/adaptive_threshold/*.csv` |
| §7.9 variable-projection check (prose, no table) | `variable-projection` | `audited_source/direct_hankel_multidir_joint.py` | `generated/hankel_direct_multidir_joint.csv` |
| Table 5 / Fig. 5 — audited DtN precision continuation | — | *multi-precision runs not in this archive* | record in `data/audited/dtn_precision_audited.{csv,png}` |
| Fig. 6 — local trace rank at 50 decimal digits | — | *50-digit runs not in this archive* | record in `data/audited/high_precision_trace_spectrum.csv` |
| Figs. 2, 3, 8 — rendered figures | — | *no plotting driver survives*; the quantities behind them are reproducible above | committed under `paper/figures/` |

The DtN driver defaults to `--nr 2 --ntheta 12`; the paper runs used
`--k 8 --nr 1 --ntheta 8 --p 3`, which is what `reproduce.py` passes.

## Ordered chain for Fig. 7

Each step reads the converged state written by the previous one, so run them in
order — `reproduce.py --experiment finite-directions` does this for you:

```bash
python -m experiments.experiment_finite_direction_enrichment        # M = 1..10
python -m experiments.polish_finite_direction_enrichment            # re-solve at tighter quadrature
python -m experiments.experiment_finite_direction_enrichment_11_20  # M = 11..20
```

The polish step does not change the approximation space. It restarts each
converged angle state with higher edge quadrature and a smaller
finite-difference step.

## What this archive does not regenerate

1. **The multi-precision results behind Table 5 and Fig. 6.** The 80-bit and
   binary128 continuation and the 50-digit trace eigensolves were carried out
   outside this archive and need arbitrary-precision arithmetic (`mpmath`),
   deliberately not in `requirements.txt`. The double-precision half of that
   story *is* reproducible, via the Table 6 sweep.
2. **Figs. 2, 3 and 8 as images.** The quantities behind them are reproducible
   (Fig. 8 from the hybrid CSVs, Figs. 2–3 from the transmission and DtN
   solvers), but the plotting code that produced those committed PNGs did not
   survive. Fig. 1 does regenerate, and Fig. 4 is produced by the DtN driver.

## Verification runs

Everything below was produced by running the code in this repository, on
Python 3.11.15 with NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.5 — deliberately
*not* the environment in `ENVIRONMENT_TESTED.txt` (Python 3.13.5, NumPy 2.3.5,
SciPy 1.17.0), so agreement is across versions rather than within one.

| Run | Result |
| --- | --- |
| Table 3 checker | **PASS on all four submitted rows** to its own `rtol = 5e-13`: `1.4052063417077849e-15` / `1.9963819848785935e-15` (N=2) and `1.1265076008304821e-15` / `4.2577038056364852e-15` (N=4), with `ndof`, `cum_nfev`, `κ₂(A)` and `κ_GR` all matching. Output identical to `validation/table3_exact_verification.txt`. |
| Audited transmission driver | **Byte-identical** to `validation/transmission_omega12_analytic_final.csv` and to the continuation-stages record. |
| Table 1 | Matches the manuscript to all printed digits for `p ≤ 23`, and `κ_GR` through `p = 27`. The two entries with `κ₂(A) ≥ 10¹⁴` differ in the second digit, which is what a condition number of that size means. At `p = 29` the smallest graph eigenvalue is negative (`-2.37e-14`, manuscript `-2.0e-14`) and `κ_GR` is correctly undefined. |
| Table 2 | On 72 triangles the uniform errors, `N_fev`, `κ₂(A)` and `κ_GR` match to every printed digit. Directions recovered as `251.000000000000, 123.000000000000, 17.000000000000`. |
| Fig. 1 | Renders both panels; visually identical to the committed figure. |
| Table 4 (centered) | Relative `L²` `7.663e-2` and mean angle error `7.00e-6°`, against the manuscript's `7.663×10⁻²` and `7×10⁻⁶°`. |
| Table 6 | All 39 audited rows matched on `(p, τ)`. Identical for `τ ≥ 10⁻¹⁴`; diverging by up to 52% in relative error at `τ ≤ 10⁻¹⁶` and high `p`, where retained modes sit at the roundoff boundary and eigenvalue ordering is version-dependent. This is the paper's own thesis about `τ_rank`. `VALIDATION.md`'s spot check reproduces exactly. |
| §7.9 variable projection | One ray from the centroid-radial start gives `J = 0.4560437374` and `E_L2 = 0.3546188`, against the manuscript's `4.5604×10⁻¹` and `3.5462×10⁻¹`. |

Not rerun end to end here: the hybrid benchmark (Table 8, the expensive one)
and the full finite-direction continuation chain.
