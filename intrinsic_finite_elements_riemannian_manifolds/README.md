# Reproducibility package

Code for every table and figure of "Intrinsic finite elements for PDEs and spectra on Riemannian manifolds"
(S. Kapita). Python 3.10 or later; `pip install -r requirements.txt`. The figure scripts typeset their text with LaTeX
(`text.usetex`), so a TeX installation is needed for the figures only. Run every command from this folder.
Intermediate data go to `run/`, tables to `results/` or the terminal, figures to `figures/`.
Times are for one core of a 2024 laptop-class machine.

## Layout
- `riemannfem/`: the library of the paper. Bernstein scalar spaces, trimmed form spaces, entity-based assembly of `Z_k`
  and `D_k`, metric-dependent Hodge matrices, Duffy-Gauss quadrature, the periodic torus, the five-cell
  Poincare mesh, and the solid tubes.
- `trefoil_hp.py`: two-dimensional solver on the torus with the trefoil-tube metric.
- `bb3d.py`: a plain C0 Bernstein tetrahedral assembler, independent of `riemannfem`, used for the high-degree
  scalar spectra and the eigenfunction figures.
- `hypforms.py`, `sphereforms.py`, `metriccompare.py`: model-coordinate code for the Seifert-Weber scalar comparison.

## Two-dimensional experiments (trefoil tube surface)
| Item | Command | Check value | Time |
|---|---|---|---|
| Poisson table, p = 2..6 on 16x4, 32x8, 64x16 | `python trefoil_poisson_uniform.py` | p = 6, finest mesh: L2 error 2.793e-09 | 3 min |
| Spectrum table | `python run_spec.py NT NS 5 12` for (NT,NS) = (48,8), (72,8), (96,8), (144,12), (24,18) | lambda_480 = 155.7335, 154.3761, 154.3059, 154.2991, 190.8084 | 1 to 4 min each |
| Eigenfunction figure | `python run_spec.py 144 12 5 12` then `python trefoil_modes_figure.py` | `figures/trefoil_selected_modes.png` | 5 min |

## Three-torus with variable metric
| Item | Command | Check value | Time |
|---|---|---|---|
| Scalar tables (isotropic, anisotropic) | `python scalar_high_order.py --N N --p P [--anisotropic]` | N = 2, P = 2: L2 error 1.4425e-01 | seconds to minutes |
| Scalar assembly against D_0^T M_1 D_0 | `python test_scalar_form_consistency.py` | relative differences below 1e-14 | 10 s |
| One-form tables | `python oneform_high_order.py --N N --r R --solver schur --results results/oneform.jsonl` | N = 5, R = 3: L2 error 1.7157e-02 | up to 2 min |
| One-form rows N = 6, R = 3 and N = 4, R = 4 | `python oneform_staged.py N R assemble`, `python oneform_solve_ck.py N R 3000`, `python oneform_staged.py N R errors` | 1.0045e-02 and 5.2710e-03 | 4 min, 2 min |
| Source-term check against finite differences | `python audit_source.py` | differences near 3e-08 | 1 min |
| Broken best approximation, constant-form test, matrix defects | `python deep_audit.py` | ratios 1.55 to 1.80 | 10 min |
| Metric replacement tables and ratio intervals | `python metric_replacement.py` and `python metric_replacement.py --anisotropic` | N = 2, p = 2: 34.573394 and 33.277525, interval [0.415, 2.411] | 5 min |
| Reference moment conditioning table | `python test_bernstein_reference.py` | r = 4, k = 1: 9.5e+04 to 4.4e+03 | 1 min |
| Eigenfunction figure | `python torus3.py 6 4 12`, `python torusviz.py`, `python torusfig.py` | first eigenvalue 33.8162 | 6 min |

## Poincare dodecahedral space
| Item | Command | Check value | Time |
|---|---|---|---|
| Trimmed-space spectrum table | `python pds_sparse_refined.py --r R --level L` for R = 2, 3 and L = 0, 1 | 257.7958, 180.5860, 172.8157, 168.3716 | seconds |
| Coexact one-form level | `python pds_sparse_1forms.py --r 3 --level 1` | 4.000004, six members | 1 min |
| Dimensions and cohomology table | `python pds_highorder_complex.py`, `python pds_r4_complex.py` | (1,6,10,5), cohomology (1,0,0,1) | 3 min |
| Degree fourteen on five cells | `python pds.py 14 140` | 167.99999999999 (13 members), then 21, 25, 31, 33 | 2 min |
| Facet pairing table | `python pds_pairings.py` | ten pairings | 1 s |
| Eigenfunction figure | after `pds.py 14 140`: `python pdsviz.py`, `python pdsfig.py` | | 2 min |

## Solid tubes
| Item | Command | Check value | Time |
|---|---|---|---|
| Volume eigenvalue table | `python solid_torus_volume_modes.py --kind all --p 1 --q 3 --no-plot`, again with `--p 2 --q 5` | round torus, Neumann: 0.2126307 and 0.2078609 | 3 min |
| Eigenfunction figure | `python tube.py trefoil 96 3 3 8 neumann`, `python tube.py trefoil 96 3 3 8 dirichlet`, `python tubefig.py` | Neumann 0.0479, Dirichlet 48.16 | 5 min |

## Seifert-Weber space
| Item | Command | Check value | Time |
|---|---|---|---|
| Scalar comparison, model metric | `python hypforms.py 6 1` | volume 11.199065, first level 9.5701 (six members) | 3 min |
| Scalar comparison, piecewise-flat metric | `python reproduce_metric_comparison.py` | volume 19.857 on 60 cells | 2 min |
| Scalar spectrum with S_6^0 and figure | `python sw.py 6 1 60`, `python swviz.py`, `python swfig.py` | volume 11.1990647, levels 9.5701, 15.3637, 19.3280 | 6 min |
| Coexact one-form table | `python sw_forms.py --r R --level L --k 14 --sigma 3.0` for (L,R) = (0,3), (0,4), (0,5), (1,2), (1,3) | (0,4): 2.040767 and 2.040998; (1,3): 2.039325 to 2.039478 | 20 s to 4 min |
| Coexact row 480 cells, r = 4 | `python sw_r4_chunks.py 4 1 0:240`, `... 240:480`, `python sw_r4_chunks.py 4 1 solve 22 3.0` | 2.039273 to 2.039275 | 11 min |
| Multiplicities below 64 | `python sw_forms.py --r 4 --level 0 --dense` | 203 eigenvalues below 66, clusters of sizes 3 to 6, kernel 631 | 2 min |
| Residual and exact-form component | `python sw_orth.py` | below 1e-9 and 6e-11 | 1 min |
| Chart constants table | `python mesh_constants.py` | S^3/Gamma anisotropy 4.00 | 2 min |

The convergence plot of the scalar torus problem is drawn in the manuscript from the numbers of the scalar table.
