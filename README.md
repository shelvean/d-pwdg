# Research code and manuscripts by Shelvean Kapita

Department of Mathematics, Texas A&M University

This repository collects the research code and manuscripts for the papers
listed below.  Each paper with code lives in its own self-contained
subproject with its own README, installation notes and reproduction guide;
nothing in one subproject imports from another.  Run every command from
inside the subproject it belongs to.

| Paper | Where |
|---|---|
| **Direction-Adaptive Plane-Wave Discontinuous Galerkin Methods for the Helmholtz Equation** | [`direction_adaptive_pwdg_helmholtz/`](direction_adaptive_pwdg_helmholtz/): code, data, reproduction guides, and the manuscript ([LaTeX source](direction_adaptive_pwdg_helmholtz/paper/Kapita_Direction_Adaptive_PWDG_Helmholtz.tex), [PDF](direction_adaptive_pwdg_helmholtz/paper/Kapita_Direction_Adaptive_PWDG_Helmholtz.pdf)) |
| **A High-Order C0 Bernstein Quasi-Trefftz Method for Stationary Hamilton–Jacobi Equations** | [`bernstein_quasi_trefftz_hamilton_jacobi/`](bernstein_quasi_trefftz_hamilton_jacobi/): LaTeX source, code, data and figures; a compiled PDF is also under [`paper/`](paper/Kapita_Bernstein_QuasiTrefftz_Hamilton_Jacobi.pdf) |
| **Conformal Bernstein–Bézier Splines on Compact Two-Manifolds of Positive Genus: Construction, Algorithm, and Experiments** | [`conformal_bb_splines/`](conformal_bb_splines/): code and JSON logs for every table and figure |
| **Intrinsic Finite Elements for PDEs and Spectra on Riemannian Manifolds** | [`intrinsic_finite_elements_riemannian_manifolds/`](intrinsic_finite_elements_riemannian_manifolds/): the `riemannfem` library and the scripts for every table and figure |

## Layout

```text
direction_adaptive_pwdg_helmholtz/              Helmholtz PWDG subproject (code + manuscript)
bernstein_quasi_trefftz_hamilton_jacobi/        Bernstein qT Hamilton-Jacobi subproject (code + manuscript)
intrinsic_finite_elements_riemannian_manifolds/ intrinsic FEM subproject (code)
conformal_bb_splines/                           conformal Bernstein-Bezier splines subproject (code)
paper/                                          compiled PDF of the Bernstein qT Hamilton-Jacobi manuscript
CITATION.txt                                    plain-text references for every manuscript
LICENSE                                         CC0 1.0 Universal, repository-wide
```

## Citation

`CITATION.txt` lists the reference for each manuscript.  Each subproject's
README repeats the citation for its own paper.

## License

The code, data and manuscripts in this repository are released under
[CC0 1.0 Universal](LICENSE): the author waives copyright and related rights
worldwide, so no permission is needed to use, modify or redistribute them.
Attribution is not legally required; a citation of the relevant manuscript is
the customary scholarly courtesy.

Two files are **not** the author's to license and are not covered by the CC0
waiver: `direction_adaptive_pwdg_helmholtz/paper/sn-jnl.cls` and
`direction_adaptive_pwdg_helmholtz/paper/sn-mathphys-num.bst` are the Springer
Nature journal class and bibliography style, redistributed only so that
manuscript compiles as submitted.  They remain subject to Springer Nature's
own terms.
