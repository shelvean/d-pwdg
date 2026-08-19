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
| Table 1 — uniform vs. graph–Riesz conditioning | *no dedicated driver* (see gaps below) | — | — |
| Table 2 — three hidden plane waves | *no dedicated driver* (see gaps below) | — | — |
| Fig. 1 — transmission fields, ω=12 | *no plotting driver* | — | `paper/figures/transmission_solution_{69,29}deg_2d.png` |
| Table 3 — Part A transmission recovery | `experiments/experiment_transmission.py` | `results/transmission_recovery.csv` | — |
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

1. **Tables 1 and 2 have no dedicated driver.** The conditioning pair
   `κ₂(A)` and `κ_GR` in Table 1 comes from `graph_data` in
   `core/direction_adaptive.py`, and the three-wave recovery of Table 2
   (directions 17°, 123°, 251° on 72 triangles) uses the same Part A machinery
   as the crossing-wave driver. Both were run from short ad-hoc scripts that
   are not part of the cleaned archive.
2. **The four rendered figures — Figs. 1, 2, 3 and 8 — have no plotting
   driver.** The underlying quantities are reproducible (Fig. 8 from the
   `experiment_adaptive_threshold.py` CSVs; Figs. 1–3 from the transmission and
   DtN solvers), but the plotting code that produced the committed PNGs is not
   in the archive. Figs. 4, 5, 6 and 7 *are* reproducible as images: Fig. 4
   from `experiment_dtn_direction_recovery.py`, and Figs. 5–7 as the audited
   PNGs under `data/audited/`.
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
