# Recovered code for `quotients-13.pdf`

Manuscript: *Conformal Bernstein–Bézier Splines on Compact Two-Manifolds of Positive Genus: Construction, Algorithm, and Experiments* (35 pages, supplied September 20, 2026).

**Status:** This is a consolidated archive of **existing code located in the author's saved files** plus the clearly marked `hyperbolic/spectrum_driver_q13.py` convenience script. This is **not established to be the exact, complete reproducibility archive used to make every figure and table of `quotients-13.pdf`.** Do not submit it as complete manuscript reproducibility without filling the gaps below and comparing outputs with the paper. The two original recovered source archives are `genus1plus-code-session.zip` and `hypsplines_revised_code.zip` (September 17, 2026). Their source files are retained unmodified apart from grouping in subdirectories. The later manuscript has experiments beyond those original packages.

## Installation

Use Python 3.10+ and install `numpy scipy matplotlib sympy`. For optional 3-D renders also install `mayavi vtk` and use a suitable offscreen display such as `xvfb-run` on Linux. Run the commands from their respective subdirectories since the original code imports sibling modules by name.

## Flat quotients (`flat/`)

- `flatquot.py`, `prolong.py`: coefficient constraints, mesh, assembly, Poisson and eigenvalue routines.
- `gb_flat.py`: flat portion of Table 3.
- `poisson_flat.py`: flat/conformal torus and Klein-bottle Poisson experiments (Tables 5--6 in the updated manuscript; older README numbers differ).
- `mobius_eig.py`: conformal Möbius spectrum (Table 7).
- `flat_figs.py`, `flat_maya.py`, `gallery_maya.py`, `mobius_fig.py`: corresponding display/field figures.
- `data/`: original archived JSON results.

```bash
cd flat
python gb_flat.py
python poisson_flat.py torus
python poisson_flat.py ctorus
python poisson_flat.py klein
python mobius_eig.py
```

The larger Poisson sweeps can take significant time.

## Hyperbolic quotients (`hyperbolic/`)

- `hypgeom.py`, `hypbb.py`, `hypmesh.py`, `hypc1.py`, `prolong.py`: geometry, basis, assembly, smoothness constraints.
- `test_bolza.py`, `test_hypbb.py`: core group/geometry and Bernstein/gluing checks.
- `solve_bolza.py`: first-eigenvalue convergence on the Bolza surface (Table 10 in updated manuscript).
- `hypsurf.py`: general regular even-sided polygons with opposite-edge pairings (the geometric family in Table 12).
- `klein.py`, `nonor3.py`: other quotient geometries; `make_figs.py`, `mayafigs.py`: figure utilities.
- `spectrum_driver_q13.py`: **new convenience driver**, not an archived original experiment script; computes a selected Bolza or higher-genus spectrum with the recovered code.

```bash
cd hyperbolic
python test_bolza.py
python test_hypbb.py
python solve_bolza.py
python spectrum_driver_q13.py bolza --degree 4 --level 1 --quadrature 44
python spectrum_driver_q13.py genus --genus 3 --degree 4 --level 1 --quadrature 24
```

**Limits and missing code for this PDF**

The located files do not establish the complete implementation behind the embedded-discocyte torus eigenvalue study and renderings (Table 8, Figures 9--10); its equal-degree **surface FEM comparison** (Table 9); the Bolza **intrinsic-edge-length triangulation** experiment (Table 1); all of the updated conditioning table; or exact scripts/data for every displayed eigenfunction and the complete genus ladder. Some of the original hyperbolic package includes extra related calculations (biharmonic and Selberg) not reported in this 35-page manuscript. Their presence in this archive is not evidence that they produced a particular table in `quotients-13.pdf`.

Original flat README uses **older table numbers**. Consult the current PDF for the latest definitions, parameters, quadrature settings and table numbering, and compare outputs quantitatively rather than copying figures or reporting results as rerun. The optional new spectrum driver has not been exercised over every case in Tables 10--12.

## Checks performed when consolidating

- `python flat/gb_flat.py`: completed, Euler characteristics and boundary cycles as expected.
- `python hyperbolic/test_bolza.py`: 28/28 tests passed.
- `python hyperbolic/test_hypbb.py`: 26/26 tests passed.
- A standalone genus-two, degree-four, level-one spectrum from `hypsurf.assemble_n()` yielded 254 DOFs and first nonzero eigenvalue about 3.85213851 at quadrature 16; the value is quadrature-dependent on the coarse polygon. This is **not** claimed as a reproduction of Table 12 at its specified quadrature.
- The full Poisson and eigenvalue sweeps were **not completed** as part of this consolidation.
