# Reproducing the manuscript results

**Shelvean Kapita — August 2026**

This file records which script produces which numbered table and figure of
`paper/Kapita_Direction_Adaptive_PWDG_Helmholtz.pdf`, where each script writes
its output, and which files in this repository hold the published record.
`METHOD_MAP.md` maps the *mathematics* of the paper onto the source; this file
maps the *outputs*.

Numbering follows the compiled PDF supplied under `paper/`.

## Conventions

Three different directories are involved and they are not interchangeable.

| Directory | Role |
| --- | --- |
| `results/` | Created at run time by the drivers. Git-ignored. Everything a fresh run produces lands here. |
| `data/` | Reference outputs from the runs reported in the paper, committed so a new run can be compared against them. |
| `data/audited/` | The audited published record. Where a number in the paper and a number in `data/audited/` disagree, `data/audited/` is authoritative — see `data/audited/README.md`. |

Install and run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m experiments.experiment_transmission     # module form, not path form
```

The drivers are deterministic: initial states and continuation schedules are
part of the experiment definition, not free parameters (see the
"Reproducibility conventions" section of `README.md`).

## Map from paper items to code

| Paper item | Produced by | Writes | Reference copy |
| --- | --- | --- | --- |
| Table 1 — uniform vs. graph–Riesz conditioning | `audited_source/uniform_conditioning_graph_riesz.py` (also emitted by the transmission driver) | terminal table; `generated/uniform_conditioning_graph_riesz_omega12.csv` | — |
| Table 2 — three hidden plane waves | `python reproduce_manuscript.py threewave-large-mesh` | `generated/threewave_large_mesh_results.csv` | — |
| Fig. 1 — transmission fields, ω=12 | `cd audited_source && python plot_transmission_visualization.py` | `generated/transmission_solution_{69,29}deg_2d.png` | `paper/figures/transmission_solution_{69,29}deg_2d.png` |
| Table 3 — Part A transmission recovery | `python reproduce_manuscript.py transmission-and-conditioning` (authoritative) or `experiments/experiment_transmission.py` (see the caveat below) | `generated/transmission_omega12_analytic_final.csv` / `results/transmission_recovery.csv` | `validation/transmission_omega12_analytic_final.csv` |
| Fig. 2 — the 8- and 32-triangle meshes | *no plotting driver* | — | `paper/figures/transmission_meshes_8_32.png` |
| Fig. 3 — DtN geometry and scattering reference field | *no plotting driver* (mesh class is `core/dtn_pwdg.py: PolarAnnulusMesh`) | — | `paper/figures/dtn_mesh_and_plane_wave_scattering_M20_jet.png` |
| Table 4 — DtN Hankel tests on the exact annulus | `experiments/experiment_dtn_direction_recovery.py --case centered` and `--case offcenter` | `<outdir>/dtn_<case>_continuation.csv`, `<outdir>/dtn_<case>_outer_directions.csv` | — |
| Fig. 4 — learned wave vectors on the DtN boundary | same script, same two cases | `<outdir>/dtn_<case>_directions.png` | `paper/figures/dtn_{centered,offcenter}_directions.png` |
| Table 5 — audited DtN precision continuation | *multi-precision runs not in this archive* | — | `data/audited/dtn_precision_audited.csv` |
| Fig. 5 — precision continuation, all methods | *same* | — | `data/audited/dtn_precision_audited.png` |
| Table 6 — effect of decreasing the trace cutoff | `experiments/precision_audit/experiment_dtn_tau_sweep.py` | `results/precision_audit/tau_sweep_audited.csv` | `data/audited/tau_precision_audit.csv`, `data/audited/{p29_tau_sweep,tau_vs_precision}.png` |
| Fig. 6 — local trace rank at 50 decimal digits | *50-digit runs not in this archive* | — | `data/audited/high_precision_trace_spectrum.csv`, `data/audited/double_vs_mp50_trace_rank.csv`, `data/audited/trace_rank_recovery_50digit.png` |
| Table 7 — finite-direction capacity | `experiments/experiment_finite_direction_capacity.py` (M=1…8; higher M via the continuation chain below) | `results/finite_direction_capacity_M1_M8.csv` | — |
| Fig. 7 — continuation in the number of exact directions | the three-step ENRICH–MOVE chain below | `results/finite_direction_enrichment*.csv` | `data/audited/finite_direction_continuation_audited.csv` / `.png` |
| Table 8 — threshold-driven hybrid benchmark | `experiments/experiment_adaptive_threshold.py` | `results/adaptive_threshold/{hybrid_threshold,pure_h_threshold,hybrid_threshold_partial}.csv` | `data/hybrid_threshold_comparison.csv`, `data/pure_h_threshold.csv`, `data/hybrid_stage2_h_threshold.csv`, `data/hybrid_threshold_partial.csv` |
| Fig. 8 — relative graph error vs. coefficients | same script; *no plotting driver* for the figure itself | same CSVs | `paper/figures/hybrid_threshold_comparison_scientific_ticks_grid.png` |
| §7.9 variable-projection check (prose, no table) | `experiments/experiment_variable_projection_hankel.py` | `results/variable_projection_hankel.csv` | `data/hankel_direct_joint.csv` |

`experiment_dtn_direction_recovery.py` writes to `--outdir`, default
`results_dtn`. The paper runs used `--k 8 --nr 1 --ntheta 8 --p 3`; the script
defaults are `--nr 2 --ntheta 12`, so pass the flags explicitly:

```bash
python -m experiments.experiment_dtn_direction_recovery --case centered  --k 8 --nr 1 --ntheta 8 --p 3
python -m experiments.experiment_dtn_direction_recovery --case offcenter --k 8 --nr 1 --ntheta 8 --p 3
```

## Ordered chain for Fig. 7

The finite-direction continuation is not a single script. Each step reads the
converged state written by the previous one, so run them in this order:

```bash
python -m experiments.experiment_finite_direction_enrichment        # M = 1..10   -> results/finite_direction_enrichment.csv
python -m experiments.polish_finite_direction_enrichment            # re-solve at tighter quadrature
                                                                    #             -> results/finite_direction_enrichment_polished.csv
python -m experiments.experiment_finite_direction_enrichment_11_20  # M = 11..20  -> results/finite_direction_enrichment_11_20.csv
```

The polish step does not change the approximation space. It restarts each
converged angle state with higher edge quadrature and a smaller
finite-difference step, so the reported recovery errors are limited by the
continuation rather than by the preliminary solve.

## Gaps — what this archive does not regenerate

These are stated so that a reader does not go looking for a driver that is not
here.

1. ~~Tables 1 and 2 have no dedicated driver.~~ **Closed.** Both drivers are
   now under `audited_source/` and both were rerun — see "Verification runs"
   below.
2. **Figs. 2, 3 and 8 have no plotting driver.** The underlying quantities are
   reproducible (Fig. 8 from the `experiment_adaptive_threshold.py` CSVs;
   Figs. 2–3 from the transmission and DtN solvers), but the plotting code that
   produced those committed PNGs is not in the archive. Fig. 1 *is* now
   regenerable — `audited_source/plot_transmission_visualization.py`, verified
   to render the committed figure. Figs. 4, 5, 6 and 7 are likewise available
   as images: Fig. 4 from `experiment_dtn_direction_recovery.py`, and Figs. 5–7
   as the audited PNGs under `data/audited/`.
3. **The multi-precision results behind Table 5 and Fig. 6 are records, not
   runs.** The 80-bit and binary128 continuation and the 50-digit trace
   eigensolves were carried out outside this archive and need arbitrary-
   precision arithmetic (`mpmath`), which is deliberately not in
   `requirements.txt`. The double-precision half of that story *is*
   reproducible, via `experiments/precision_audit/experiment_dtn_tau_sweep.py`.
4. **`paper/figures/exact_circular_dtn_mesh_studies.png` is unreferenced.** No
   `\includegraphics` in the current `.tex` uses it; it is kept because it
   shipped with the submission bundle.

## Scripts with no current paper item

Two drivers are retained for reproducibility but do not feed a table or figure
in the present version of the manuscript:

| Script | Writes | Reference copy | Status |
| --- | --- | --- | --- |
| `experiments/experiment_high_frequency_hankel.py` | `results/high_frequency_hankel.csv` | `data/high_frequency_hankel.csv` | Fixed-budget stress test: mesh and five local rays held fixed while κ increases. |
| `experiments/experiment_nonlinear_history.py` | `results/two_plane_nonlinear_history.csv`, `.../two_plane_nonlinear_convergence.{png,pdf}` | `data/two_plane_nonlinear_history_ls.csv` | Nonlinear convergence history for the two-direction solve. `METHOD_MAP.md` refers to a convergence plot; no such figure is included in the current `.tex`. |
| `experiments/experiment_crossing_waves.py` | `results/two_plane_crossing.csv` | `data/two_plane_crossing.csv` | One-, two- and three-direction comparison for two crossing waves (37°, 128°). Supports the implementation checks of §7.1. |
| `experiments/experiment_dtn_boundary_direction_oracle.py` | `<outdir>/dtn_<case>_boundary_oracle.csv`, `dtn_boundary_oracle_summary.csv`, two PNGs | — | Oracle diagnostic, not the automatic algorithm: the best single-plane-wave fit to the *exact* Cauchy trace on each arc, used as a reference for the learned directions. |

## Cost

`experiments/experiment_adaptive_threshold.py` is the expensive one. Every
marked element may require two MOVE trials and up to six ENRICH trials, and
each accepted trial is evaluated through a global solve — the paper notes that
the TEST step is not yet cheaper than pure `h`-adaptivity in elapsed time. It
writes `hybrid_threshold_partial.csv` incrementally so an interrupted run does
not lose completed cycles. The finite-direction chain and the τ sweep are
minutes-scale; the remaining drivers are seconds-scale.

## Two source trees, and which one is authoritative

The repository now carries two implementations of the same mathematics.

`core/` + `experiments/` is the cleaned, readable tree described by
`METHOD_MAP.md`. `audited_source/` is the original pre-refactor source that
actually produced the numbers printed in the manuscript, together with
`reproduce_manuscript.py` as its entry point.

**For regenerating a printed table value, prefer `audited_source/`.** The two
trees agree on the mathematics but not always on the last digits: stopping
rules and assembly order differ slightly, which is invisible except where the
reported quantity is itself at roundoff level. The transmission experiment is
the clearest case — see the caveat in the verification table below.

`audited_source/` holds only the import closure of the manuscript drivers
(14 files), not the full legacy tree.

## Verification runs

Every claim below was produced by running the code in this repository, on
Python 3.11.15 with NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.5 — deliberately
*not* the environment in `ENVIRONMENT_TESTED.txt` (Python 3.13.5, NumPy 2.3.5,
SciPy 1.17.0), so agreement is across versions rather than within one.

| Run | Command | Result |
| --- | --- | --- |
| Audited transmission (Table 3) | `python reproduce_manuscript.py transmission-and-conditioning` | **Byte-identical** to `validation/transmission_omega12_analytic_final.csv` and to the continuation-stages record. Reproduces the manuscript's 8-triangle errors `1.4052063417e-15` (69°) and `1.9963819849e-15` (29°), and the 32-triangle `1.1265076008e-15` / `4.2577038056e-15`. |
| Conditioning (Table 1) | `cd audited_source && python uniform_conditioning_graph_riesz.py` | Matches the manuscript to all printed digits for `p ≤ 23`, and `κ_GR` through `p = 27`. The two entries where `κ₂(A) ≥ 10¹⁴` differ in the second digit — `3.49e14` vs. `3.43e14` at `p=27`, `2.99e17` vs. `2.48e17` at `p=29` — which is what a condition number of that size means. The qualitative claim reproduces exactly: at `p=29` the smallest graph eigenvalue is negative (`-2.37e-14`, manuscript `-2.0e-14`) and `κ_GR` is correctly reported as undefined. |
| Three hidden waves (Table 2) | `python reproduce_manuscript.py threewave-large-mesh` | On 72 triangles, the uniform errors, `N_fev`, `κ₂(A)` and `κ_GR` all match to every printed digit (e.g. `κ=8`: uniform `0.37541`, `N_fev` 8, `κ₂` 37.610, `κ_GR` 5.983). The optimized-error column differs in the second digit at the `10⁻¹⁵` level, as roundoff-level quantities do. All three directions recovered as `251.000000000000, 123.000000000000, 17.000000000000`. |
| Crossing waves | `python reproduce.py --experiment crossing` | Every value **identical to the last digit** to the committed `data/two_plane_crossing.csv`. Only the CSV header names differ, having been renamed during the cleanup. |
| DtN trace-cutoff sweep (Table 6) | `python reproduce.py --experiment dtn-tau-audit` | All 39 audited rows matched on `(p, τ)`. For `τ ≥ 10⁻¹⁴` the retained rank, effective dimension and relative error are identical. For `τ ≤ 10⁻¹⁶` at high `p` they diverge, by up to 52% in relative error, because those cutoffs retain modes at the roundoff boundary where eigenvalue ordering is version-dependent. This is the paper's own thesis about `τ_rank` rather than a contradiction of it. `VALIDATION.md`'s spot check (`p=27`, `τ=10⁻¹²` → 184 effective dofs, rank 23, `7.895e-4`) reproduces exactly. |
| Fig. 1 plot | `cd audited_source && python plot_transmission_visualization.py` | Renders both panels; visually identical to the committed `paper/figures/transmission_solution_69deg_2d.png` (regenerated at higher resolution, so not byte-identical). |
| Variable projection | `python reproduce.py --experiment variable-projection` | Centroid-radial start gives `J = 0.456044`, `E_L2 = 0.354619`, against the manuscript's `4.5604e-1` and `3.5462e-1`. Blind Sobol converges to the poorer basin as reported. |
| **Cleaned-tree transmission** | `python reproduce.py --experiment transmission` | **Does not reproduce the manuscript's figures.** Relative `L²` comes out `6.58e-12` (69°) and `2.92e-11` (29°), against the paper's `1.41e-15` and `2.00e-15`. Recovered angles and both condition numbers agree to ~6 digits; only the roundoff-level error column moves. This is the concrete instance of the caveat above — use `reproduce_manuscript.py` for this table. |

Not rerun here: the hybrid adaptive benchmark (`--experiment hybrid`, the
expensive one) and the finite-direction continuation chain.
