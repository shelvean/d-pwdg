# Embedded C0 quasi-Trefftz revision

This revision makes the quasi-Trefftz reduction explicit in both the manuscript and code.

## Defining change

The qT state is no longer defined by minimizing all Hamiltonian moments over the full C0 parent space.  Instead:

1. Assemble the global C0 Bernstein space first, so every interior-edge coefficient has one shared global index.
2. Split global coordinates into retained trace coordinates and dependent conforming coordinates.
3. Assemble the degree-(p-1) Hamiltonian moment map and its Jacobian with respect to the dependent coordinates.
4. Use one rank-revealing QR factorization of the transposed dependent Jacobian to select a square independent moment block.
5. Solve those selected nonlinear moment equations to solver tolerance by sparse Newton/backtracking.
6. Treat the unselected moment rows as a complementary consistency defect.

Thus the computed qT state lies on an implicit equation-adapted manifold contained in the already assembled C0 parent space.  Shared interior-edge coefficients are dependent, not deleted and not duplicated.

## New code

- `code/conforming_embedded_qt.py`: implicit conforming qT condensation.
- `code/qt_projection_utils.py`: parent-space L2 projection used as a branch seed.
- `code/run_c0_qt_embedded_experiment.py`: p- and h-sweeps comparing the full parent space with the embedded qT manifold.
- `code/run_embedded_qt_core_benchmarks.py`: embedded-qT reruns of the smooth non-eikonal, smooth eikonal, polynomial-recovery, and focal tests.

## New data

The new CSV files have `_embedded_qt` in their names.  Older CSV files are retained unchanged for auditability and for the previously computed full-residual branch-correction prototype.

## Solver distinction

The main smooth-branch evidence now uses the embedded qT condensation.  Marmousi and selected first-arrival stress tests retain the earlier inexact full-residual Gauss-Newton/LSMR correction because it is being used as a branch-preserving hybrid after causal selection; the manuscript explicitly does not present those runs as evidence for the reduced qT manifold.
