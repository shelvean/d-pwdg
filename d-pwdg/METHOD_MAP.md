# Map from the manuscript to the code

**Author:** Shelvean Kapita  
**Date:** August 2026

This file records where the principal mathematical objects in the manuscript appear in the reproducibility code.  It is intended to make the implementation auditable without requiring the reader to infer the variational structure from generic finite element software.

## Local Trefftz functions

The local basis is represented in `core/direction_adaptive.py` by `PWDGDictionary._basis`.  The function evaluated on element `K` is

```text
exp(i q . (x - x_K)).
```

The shift by the element centroid `x_K` does not alter the Trefftz property and avoids unnecessarily large phases in the basis evaluation.

Real propagation directions are constructed by `q_real`.  Complex directions are constructed by `q_complex_angle`.  The latter uses the same analytic formula with a complex angle, so propagating and evanescent states do not require separate implementation branches.

## PWDG matrix and right-hand side

`PWDGDictionary.assemble` loops over the mesh skeleton.  On an interior edge it assembles the four element-pair blocks corresponding to averages, jumps, the value-jump penalty, and the normal-flux-jump penalty.  On a boundary edge it assembles the all-Dirichlet PWDG term and the boundary data.

The code uses

```text
ALPHA = BETA = 0.5
```

in the symmetric all-Dirichlet experiments, in agreement with the manuscript.

## Physical skeleton residual

`physical_skeleton_residual` and `physical_skeleton_residual_vector` evaluate the same weighted residual used as the nonlinear objective.  The residual vector is split into real and imaginary parts only at the interface with SciPy's real nonlinear least-squares solver.

For an interior edge the two blocks are

```text
sqrt(alpha*xi) [u]
sqrt(beta/xi) [grad u . n]
```

and on a Dirichlet boundary the block is

```text
sqrt(alpha*k) (u-g).
```

In the all-Dirichlet experiments, the square of this norm is the graph error.

## Graph-Riesz conditioning

`graph_data` constructs the graph-Riesz scaling used for the conditioning diagnostic.  The raw Euclidean matrix condition number and the graph-Riesz condition number are intentionally kept separate in the output.

## Part A

Part A keeps the PWDG equations as the state equation.  A trial direction field is inserted in `PWDGDictionary`, `solve()` computes the constrained coefficient vector, and the nonlinear optimizer evaluates the physical skeleton residual of that PWDG solution.

The transmission-specific automatic searches are in the lower half of `core/direction_adaptive.py`, notably `q_complex_angle`, `region_graph_greedy_auto`, `optimize_region_directions`, and `optimize_region_directions_ls`.

## Part B and variable projection

`audited_source/direct_hankel_multidir_joint.py` implements the unconstrained Trefftz least-squares formulation and applies it to the one-ray outgoing Hankel problem.  The linear coefficients are eliminated so that the nonlinear optimizer receives only the direction variables.

## Transmission experiment

`audited_source/exact.py` contains the exact two-material transmission field used only for boundary data and error measurement.  `audited_source/transmission_omega12_experiments.py` runs frequency continuation in the three complex-angle variables, with the analytic direction Jacobian of `audited_source/wirtinger_pwdg_tests.py`.  The recovered upper-material state is compared after the solve with the Snell angle in the propagating case and with the analytic hyperbolic decay parameter in the evanescent case.  `verify_table3_exact.py` asserts the four submitted rows.

## Analytic direction derivatives

`audited_source/wirtinger_pwdg_tests.py` assembles the exact real-parameter derivative of the PWDG matrix and right-hand side (`assemble_directional_derivative`), the coefficient sensitivity (`solution_sensitivity`), and the residual Jacobian column (`residual_vector_and_jac_column`).  `all_analytic` assembles the full Jacobian.  This is the "analytic residual derivatives" of Section 7.1, and it is what allows the transmission recovery to reach roundoff.

## Hybrid adaptivity

`experiments/experiment_adaptive_threshold.py` contains the full L-shaped benchmark and uses the following sequence:

```text
SOLVE -> ESTIMATE -> MARK -> TEST -> ACT -> SOLVE.
```

`Solver.indicators` performs ESTIMATE by distributing positive skeleton contributions to elements.  `dorfer` performs MARK.  The first stage tests MOVE, ENRICH, and REFINE.  After cycle 14, the local direction field is frozen and the code continues with residual-driven h-refinement.

The supplied reference data record the calculation reported in the paper, including the final comparison at the five-percent relative graph-error threshold.
