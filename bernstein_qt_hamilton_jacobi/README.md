# C0 Bernstein Quasi-Trefftz Method for Stationary Hamilton–Jacobi Equations

**Shelvean Kapita**
Department of Mathematics, Texas A&M University

This subproject holds the manuscript source, the Python implementation, the
benchmark data and the figures for

> **A High-Order C0 Bernstein Quasi-Trefftz Method for Stationary
> Hamilton–Jacobi Equations**

It is self-contained: nothing here imports from the direction-adaptive PWDG
code at the top level of the repository, and the top-level launchers do not
run anything in this directory.

The method is a global C0 Bernstein degree-p spline space on a triangulation,
with the Hamiltonian residual tested against Bernstein polynomials of degree
p-1.  The quasi-Trefftz (qT) state is obtained by an implicit conforming
condensation: the assembled C0 coordinates are split into retained trace
coordinates and dependent coordinates, one rank-revealing QR of the transposed
dependent Jacobian selects a square independent moment block, and that block is
solved to solver tolerance by sparse Newton with backtracking.  The eikonal
equation is the principal application; two non-eikonal manufactured
Hamiltonians (anisotropic-quadratic and quartic-convex) check that the
construction is genuinely Hamilton–Jacobi.

## Layout

```text
code/       Python implementation and experiment/plot drivers
data/       CSV/NPZ outputs that the manuscript tables and graphs are built from
figures/    figures included by the manuscript (PDF, with PNG copies)
paper/      main.tex and the two compiled PDFs of the submitted manuscript
package_notes/
            the README files shipped inside the submission package, verbatim
```

The scripts locate `data/` and `figures/` relative to their own location
(`Path(__file__).resolve().parents[1]`), so `code/`, `data/` and `figures/`
must stay siblings.  Run everything from this directory, for example

```bash
python code/run_examples.py
```

The Journal of Scientific Computing submission archive was flat, because
Editorial Manager rejects ZIP files with subfolders.  This directory restores
the original `code/ data/ figures/` organization that the scripts assume.  The
only edit made to the manuscript source for that purpose is a single
`\graphicspath{{../figures/}}` line after `\usepackage{graphicx}` in
`paper/main.tex`; the figure file names in the `\includegraphics` calls are the
flat ones from the submission.  The Python scripts and data files are copied
unchanged.

## Building the manuscript

`paper/main.tex` uses the `elsarticle` class in preprint format with the
journal field set to Journal of Scientific Computing.  Run twice from
`paper/`:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The bibliography is an inline `thebibliography` environment, so no BibTeX pass
is needed.  `Kapita_Bernstein_QuasiTrefftz_Hamilton_Jacobi_JSC_ready.pdf` is the
PDF compiled from this source with the figures in a `figures/` subfolder and
`Kapita_Bernstein_QuasiTrefftz_Hamilton_Jacobi_JSC_flat_compiled.pdf` is the
same source compiled from the flat submission archive; the two differ only in
the compile timestamp and in the figure paths recorded inside the PDF.  A
compiled copy of the manuscript is also kept with the other manuscripts under
the top-level [`paper/`](../paper/) directory.

## Installation

NumPy, SciPy, pandas and Matplotlib are needed by everything; numba is imported
only by the Marmousi fast-sweeping reference and therefore by the Marmousi
drivers.

```bash
python -m pip install -r requirements.txt
```

## Code

### Solvers

| file | contents |
| --- | --- |
| `code/optimized_hj.py` | production solver: batched element values, gradients, Hamiltonian evaluations and moment contractions; analytic Jacobians assembled into a fixed CSR pattern; column-scaled sparse Gauss–Newton steps solved with LSMR; backtracking on the true nonlinear residual; exact triangular Bernstein degree elevation for p to p+2 continuation. Contains the factored eikonal (`FastFactoredEikonalPminus1`), general Hamilton–Jacobi and multiplicatively factored point-source (`FastMultiplicativeFactoredBB`) solvers. |
| `code/conforming_embedded_qt.py` | the implicit conforming qT condensation (`ConformingEmbeddedQT`): trace/dependent split, rank-revealing selection of the independent moment block, sparse Newton on the selected equations, complementary consistency defect. |
| `code/qt_projection_utils.py` | parent-space L2 projection used as the branch seed. |
| `code/eikonal_c0_pminus1.py` | the original stand-alone global C0 Bernstein degree-(p-1) eikonal solver, with a command-line interface (`--n --p --kind --boundary --json`). |
| `code/heterogeneous_stratified.py` | the stratified heterogeneous slowness model and its reference phase. |
| `code/high_frequency_benchmarks.py` | Potter–Cameron radial and linear-speed point-source problems (`RadialS1`, `RadialS2`, `LinearSpeed`), the factored C0 eikonal solver used for them, the two-source first-arrival interface, and a Cartesian fast-sweeping reference with a velocity-file loader. |
| `code/treister_haber_benchmarks.py` | Treister–Haber benchmark cases with a command-line interface. |
| `code/marmousi_bb_core.py` | Bernstein branch correction on the Marmousi model after causal branch selection. |
| `code/marmousi_fast_sweeping_reference.py` | first-order causal fast sweeping on a Cartesian grid, used only to select and reference the viscosity first-arrival branch. |
| `code/plot_style.py` | shared Matplotlib style for the figures. |

### Experiment drivers and the data they write

| driver | writes |
| --- | --- |
| `code/run_c0_qt_embedded_experiment.py` | `data/c0_qt_embedded_rotating_anisotropic_{p,h}_sweep.csv` (full C0 parent space versus embedded qT manifold on the rotating-anisotropy Hamiltonian) |
| `code/run_c0_qt_first_experiment.py` | `data/c0_qt_rotating_anisotropic_{p,h}_sweep.csv` (earlier full-residual branch-correction prototype on the same problem) |
| `code/run_embedded_qt_core_benchmarks.py` | `data/general_hj_benchmarks_embedded_qt.csv`, `data/smooth_p_odd_embedded_qt.csv`, `data/smooth_h_p5_embedded_qt.csv`, `data/polynomial_recovery_odd_embedded_qt.csv`, `data/focal_caustic_p_sweep_embedded_qt.csv` |
| `code/hj_general_benchmarks.py` | `data/general_hj_benchmarks.csv` (anisotropic-quadratic and quartic-convex sweeps) |
| `code/run_examples.py` | `data/polynomial_recovery_odd.csv`, `data/smooth_p_odd.csv`, `data/smooth_h_p5.csv`, `data/smooth_inflow_p5.csv` |
| `code/run_stratified_heterogeneous.py` | `data/stratified_p5_h_sweep.csv`, `data/stratified_n4_p5_solution.npz` |
| `code/run_high_frequency_benchmarks.py` | `data/high_frequency_benchmark_results.csv`, `data/linear_speed_two_source_p5_grid.npz` |
| `code/run_focal_caustic_sweep.py` | `data/focal_caustic_p_sweep.csv` |
| `code/run_carrier_frequency_sweep.py` | `data/carrier_frequency_sweep.csv` |
| `code/run_treister_haber_odd.py` | `data/treister_haber_case1_odd_h1.csv` |
| `code/run_marmousi_smooth.py` | `reproduced_marmousi/` (see below) |
| `code/run_marmousi_c0_qt_compare.py` | `data/marmousi_c0_qt_{p,h}_sweep.csv` |

Files with `_embedded_qt` in the name come from the embedded qT condensation,
which is the main smooth-branch evidence in the manuscript.  The older CSV
files without that suffix are retained unchanged for auditability; they were
produced by the earlier full-residual Gauss–Newton/LSMR branch-correction
prototype.  `data/pointsource_p_sweep.csv` is the external point-source
odd-degree sweep; no driver in the package writes that file, so it is kept as
a data record only.

All degree sweeps use the single odd sequence p = 3, 5, 7, 9, 11, 13.

### Figure drivers

| driver | reads | writes |
| --- | --- | --- |
| `code/regenerate_odd_figures.py` | `smooth_p_odd_embedded_qt.csv`, `smooth_h_p5_embedded_qt.csv`, `focal_caustic_p_sweep_embedded_qt.csv`, `carrier_frequency_sweep.csv`, `stratified_n4_p5_solution.npz`, `linear_speed_two_source_p5_grid.npz` | `smooth_p_convergence.pdf`, `smooth_h_convergence.pdf`, `focal_p_convergence.pdf`, `carrier_frequency.pdf`, `stratified_p5_solution.png`, `linear_speed_two_source_p5.png` |
| `code/plot_c0_qt_first_experiment.py` | `c0_qt_embedded_rotating_anisotropic_{p,h}_sweep.csv` | `c0_qt_first_experiment.pdf/.png` |
| `code/plot_focal_caustic.py` | recomputes the p=5 factored focal solve | `focal_caustic_p5.pdf` |
| `code/plot_raw_marmousi_lowp.py` | `marmousi_raw_p{3,5,7}.csv` and an external raw reference | `marmousi_raw_lowp.pdf/.png` |
| `code/run_marmousi_smooth.py` | external velocity file | `marmousi_smooth_bb_panels`, `marmousi_h_convergence`, `marmousi_residual_convergence` (PDF and PNG) |

`figures/c0_qt_condensation_schematic.pdf` is a drawn schematic of the
condensation and is not produced by a script.

## Marmousi benchmark

The Marmousi velocity arrays are not redistributed.  The benchmark uses a
surface source at x = 4.6 km, a native 8 m causal reference grid, a 73 x 25
coarse causal branch selector, and p = 7 Bernstein h-continuation.  The
numbers used in the manuscript are `data/marmousi_smooth_h_sweep_p7.csv` and
`data/marmousi_raw_h_sweep_p7.csv`.  To recompute them from a downloaded
376 x 1151 velocity file:

```bash
python code/run_marmousi_smooth.py \
  --velocity /path/to/vel_marmousi_smooth400_376x1151.csv.txt \
  --output-dir reproduced_marmousi
```

Add `--raw-velocity /path/to/vel_marmousi_376x1151.csv.txt` to repeat the
unsmoothed robustness sweep.  The script recomputes the native causal
reference, the coarse branch selector, all six p = 7 h-continuation levels, the
CSV table and the three Marmousi figures.  The `reproduced_marmousi/` output
directory is git-ignored.

Two scripts keep the absolute paths of the machine the runs were made on and
must be edited before use: `code/run_marmousi_c0_qt_compare.py` reads the
smoothed velocity file from `/mnt/data/`, and `code/plot_raw_marmousi_lowp.py`
reads `/mnt/data/marmousi_reference_raw.npz`.  They are left as they were run
rather than silently changed.

## Scope note

The theory concerns smooth classical branches.  The global least-squares
estimate is conditional on stability of the assembled linearized
characteristic map.  The formulation is not a monotone viscosity-solution
scheme.  The focal experiment uses a known singular factor and the two-source
experiment uses explicit branchwise minimization; automatic generation of
multivalued post-caustic phases is not claimed.  The Marmousi and selected
first-arrival stress tests use the earlier full-residual correction as a
branch-preserving hybrid after causal selection and are not presented as
evidence for the reduced qT manifold.

## Citation

> Shelvean Kapita, *A High-Order C0 Bernstein Quasi-Trefftz Method for
> Stationary Hamilton–Jacobi Equations*, Department of Mathematics, Texas A&M
> University.

## License

Released under the repository's [CC0 1.0 Universal](../LICENSE) waiver, like
the rest of the repository.  The `elsarticle` class that `main.tex` uses is not
included here; it is part of every standard TeX distribution.
