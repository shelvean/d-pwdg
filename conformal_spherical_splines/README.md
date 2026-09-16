# Conformal spherical splines for the Laplace-Beltrami operator on genus-zero surfaces

Code and data for the experiments in

> S. Kapita, *Conformal Spherical Splines for the Laplace-Beltrami Operator on
> Genus-Zero Surfaces: Construction, Algorithm, and Experiments*, submitted to
> SIAM Journal on Scientific Computing, 2026.

Python 3.10 or later with NumPy, SciPy, SymPy, Matplotlib and the `ismember`
package (`python -m pip install -r requirements.txt`). All scripts run
from this directory. Tables refer to the main article (T) and the supplement
(SM).

## Reproducing the tables

| Script | Produces |
|---|---|
| `test_sphere.py` | algebraic checks, SM4 |
| `run_numerics.py` | sphere parity table (T3), curved-surface eigenvalues (T4), geometry diagnostics (SM Table 4.1) |
| `poisson_surface.py` | Poisson Tests 1 and 2 (`tab_poisson.tex`: T2, SM Table 5.1) |
| `poisson_test1_ext.py` | Test 1 extended to level 4 (T1); `poisson_rows_ext.jsonl` |
| `poisson_prescribed.py` | Test 1 for the prescribed factor with no embedding (SM Table 5.2) |
| `compare_fem.py`, `make_compare.py` | surface FEM comparison (T5); `compare_fem.log` |
| `tol_sparsity.py` | rank tolerance and sparsity study (SM Table 2.1); `tol_sparsity.json` |
| `make_poisson_fig.py` | Test 1 convergence plot (SM Figure 5.1) |
| `viz_figures.py` | eigenfunction renderings (Figure 1; needs Mayavi) |
| `rbc_radial_graph_check.py` | radial-graph check for the Evans-Fung cell (Section 2) |

The scripts read and write their logs in the working directory; the logs in
`data/` are the runs behind the paper (copy them up one level to regenerate
the tables and figures without rerunning). `compare_fem.py` is resumable and
appends to `compare_fem.log`; the level-4 and `d = 6` runs take minutes each
in this pure-Python implementation.

## Modules

- `sphsplines.py`: spherical Bernstein-Bezier polynomials, icosahedral
  triangulations, radial-projection quadrature, broken mass and stiffness
  assembly, the edge smoothness matrix, interpolation, the reduced pencil.
- `prolong.py`: null-space matrix by sparse rank-revealing elimination in the
  locality order.
- `smesh.py`, `barynets.py`, `smoothfast.py`, `aad_assembly.py`: mesh,
  barycentric and assembly utilities (product-to-moment assembly).
- `spheroid.py`, `revsurf.py`, `rbc.py`: conformal factor of the spheroid, a
  general surface of revolution (Mercator construction), and the Evans-Fung
  red blood cell.
- `surface_fem.py`: parametric surface finite elements (Dziuk P1,
  isoparametric P2 and P3) used for the comparison.

## Data

`data/` holds the raw outputs behind every table: JSON-lines logs of the
Poisson and comparison runs, the tolerance and sparsity study, and the
radial-graph check output.

## License

BSD 2-Clause; see `LICENSE`.
