# Code for "Conformal Bernstein-Bezier Splines on Compact Two-Manifolds of Positive Genus"

Flat layout; run every command from this directory. Python 3.10+, numpy,
scipy, matplotlib; Mayavi and VTK under `xvfb-run` for the 3D renderings
(`xvfb-run -a -s "-screen 0 1500x1100x24" python3 <script>`). Set
`QT_QPA_PLATFORM=offscreen` when a module that imports Mayavi is used
without a display. No caches or build products are included; the JSON files
are the logs behind the tables.

## Modules

| file | contents |
|---|---|
| `flatquot.py` | flat quotients (torus, Klein bottle, Mobius band): criss-cross mesh, pairings, C^r rows across interior and paired edges, weighted assembly by conical Gauss rule, bordered Poisson solve, eigensolve, error routine |
| `flatexact.py` | closed-form assembly on the flat quotients with the weight replaced by its Bernstein interpolant (Section 3.4) |
| `metric_assembly.py` | assembly on the flat torus with a general metric tensor (not used by the paper; kept for the trefoil example) |
| `prolong.py` | rank-revealing elimination: null-space matrix Z of the smoothness matrix J |
| `hypgeom.py` | hyperboloid model: isometries, Bolza octagon, regular polygons and pairings, Klein chart, Duffy/Gauss rule, angle defect |
| `hypbb.py` | Bernstein-Bezier forms on the hyperboloid, tangential gradients, local matrices, quadrature points with the far-corner collapse |
| `hypmesh.py` | octagon mesh, geodesic refinement, degree-of-freedom identification across paired sides, assembly on the Bolza surface |
| `hypsurf.py` | regular n-gon surfaces with arbitrary side pairing (genus ladder) |
| `klein.py`, `nonor3.py` | Klein quartic and the nonorientable N_3 (polygon figure and the N_3 rows of Table gb) |
| `intrinsic.py` | intrinsic presentation: triangles placed from edge lengths, transition isometries per edge, assembly without polygon or group |
| `iso_fem.py` | isoparametric surface finite elements of degree k on a parametrized periodic surface, geometry of degree k or 1 |
| `dtorus.py` | the discocyte torus (Evans-Fung profile revolved), conformal variable, weight; `dtorus_trig_backup.py` the earlier trigonometric profile |
| `pants_mesh.py` | right-angled hexagons and the pants layouts in the Poincare disk |
| `flat_maya.py`, `flat_figs.py`, `mayafigs.py` | Mayavi renderers for the flat quotients and the hyperbolic panels |

## Tables of quotients.tex

| table | command | log |
|---|---|---|
| dim (flat rows) | `python3 run_cond.py` (first block) | printed |
| dim (Bolza rows) | `python3 run_bolza.py` (dimensions printed with each row) | printed |
| gb (flat rows) | `python3 gb_flat.py` | printed |
| gb (Bolza row) | `python3 run_gb_bolza.py` | printed |
| cond | `python3 run_cond.py` (second block) | printed |
| poisson-torus, poisson-klein | `python3 poisson_flat.py torus`, `python3 poisson_flat.py ctorus`, `python3 poisson_flat.py klein` | `poisson_*.json` |
| mobius | `python3 mobius_eig.py` | `mobius_eig.json` |
| dtorus, closed-form comparison, isocmp | `python3 run_dtorus_tables.py` | `dtorus_table.json`, `dtorus_exact.json`, `iso_sweep.json` |
| bolza, bolzaspec, odd-degree zero modes | `python3 run_bolza.py` | `bolza_conv.json`, `bolza_spectrum.json` |
| intrinsic | `python3 run_intrinsic.py` | `bolza_intrinsic.json` |
| ladder | `python3 run_ladder.py` | `genus_ladder.json` |

## Figures of quotients.tex (raw Mayavi panels; labels are set in LaTeX)

| figure | command | panels |
|---|---|---|
| flat-quotient gallery | `xvfb-run ... python3 gallery_maya.py` | `p_torus, p_clifford, p_klein, p_mobius` |
| polygons, Bolza meshes, Bolza modes | `xvfb-run ... python3 mayafigs.py` | `p_poly_*, p_mesh0..2, p_mode1, p_mode5` |
| pretzel | `xvfb-run ... python3 pretzel2.py` (needs `pretzel.py`) | `p_pretzel_mesh` |
| pants decomposition | `python3 pants_mesh.py` | `p_hex1, p_hex2, p_hex4` |
| Mobius eigenfunctions | `xvfb-run ... python3 mobius_fig.py` | `p_mobius_mode1, 9, 25` |
| discocyte torus mesh and cutaway | `xvfb-run ... python3 dtorus_fig.py`, `xvfb-run ... python3 dtorus_cut.py` | `p_dtorus_mesh, p_dtorus_cut` |
| discocyte eigenfunctions | `xvfb-run ... python3 dtorus_modes.py` | `p_dtorus_m1, m12, m30` |

The schematics (pairings of the square, the octagon pairing and corner walk,
the C^r condition across a paired edge) are TikZ in the manuscript source.

## Not in the paper

`trefoil.py`, `trefoil_fig.py`, `tube_torus.py`, `embedded_eigs.py`,
`dtorus_rbc*.py`, `dtorus_run.py`, `dtorus_high.json`, `iso_dtorus.json`,
`iso_tube.json`, `trefoil_eigs*.json`: the trefoil tube, the tube torus and
earlier profile runs, kept for the follow-up paper.
