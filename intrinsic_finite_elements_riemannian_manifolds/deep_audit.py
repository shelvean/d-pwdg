"""Independent diagnostic for the one-form experiment in focm_oneform_hp.

Put this script next to oneform_high_order.py, or run with PYTHONPATH set to
its containing directory. It does not change the manufactured PDE or the
previously published numerical results.
"""
from __future__ import annotations
import argparse
import json
import time
import warnings
import numpy as np

from oneform_high_order import analytic_oneform, B_ANISO
from riemannfem.periodic_scalar import periodic_freudenthal_mesh, periodic_tetrahedral_facet_pairings
from riemannfem.torus_metric import AffineTetrahedron
from riemannfem.reference import simplex_duffy
from riemannfem.forms import TrimmedPolynomialFormSpace
from riemannfem.bernstein_forms import bernstein_change
from riemannfem.entity_assembly import SparseEntityComplex


def best_broken_l2(N: int, r: int, q: int = 12) -> dict:
    """L2_g best approximation on the entire *broken* P_r^- Lambda^1 space.

    Independent of the global continuity constraints and PDE solver.
    It is a strict lower bound on the error of every conforming solution
    from the same reference one-form space on this mesh.
    """
    cells = periodic_freudenthal_mesh(N)
    V = TrimmedPolynomialFormSpace(3, r, 1)
    C, _, _ = bernstein_change(3, r, 1)
    lam, w = simplex_duffy(3, q)
    points = lam[:, 1:]
    values = np.asarray([C.T @ V.values(x) for x in points])
    error_sq = exact_sq = 0.0
    for verts in cells:
        K = AffineTetrahedron(verts / N)
        E = K.E
        Einv = np.linalg.inv(E)
        x = K.physical_coordinates(points)
        alpha, _, _, _, phi = analytic_oneform(x, np.eye(3))
        alpha_ref = alpha @ E
        contrav = Einv @ Einv.T
        wt = w * abs(np.linalg.det(E)) * np.exp(phi)
        mass = np.einsum('q,qai,ij,qbj->ab', wt, values, contrav, values, optimize=True)
        load = np.einsum('q,qai,ij,qj->a', wt, values, contrav, alpha_ref, optimize=True)
        coeff = np.linalg.solve(mass, load)
        diff = alpha_ref - np.einsum('qai,a->qi', values, coeff)
        error_sq += float(np.einsum('q,qi,ij,qj->', wt, diff, contrav, diff))
        exact_sq += float(np.einsum('q,qi,ij,qj->', wt, alpha_ref, contrav, alpha_ref))
    return {'N': N, 'r': r, 'quadrature': q,
            'best_broken_absolute_L2': float(np.sqrt(error_sq)),
            'best_broken_relative_L2': float(np.sqrt(error_sq/exact_sq))}


def flat_constant_patch(r: int) -> dict:
    """Solve (-Delta + I) alpha = alpha for alpha=dx_1 on the flat torus."""
    import oneform_high_order as mod
    original = mod.analytic_oneform
    def constant_field(x, B):
        x = np.atleast_2d(x)
        n = len(x)
        a = np.zeros((n, 3)); a[:, 0] = 1.
        return a, np.zeros((n, 3, 3)), np.zeros(n), a.copy(), np.zeros(n)
    try:
        mod.analytic_oneform = constant_field
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # relative d,delta norms are undefined
            result = mod.run(2, r, q=max(5, r+3), error_q=max(7, r+5), solver='direct')
    finally:
        mod.analytic_oneform = original
    return {'r': r, 'absolute_L2': result['L2_1_error'],
            'absolute_d_error': result['d_1_error'],
            'absolute_delta_error': result['delta_aux_error'],
            'algebraic_residual': result['residual']}


def matrix_defects(r: int) -> dict:
    cells = periodic_freudenthal_mesh(2)
    pairs = periodic_tetrahedral_facet_pairings(cells, 2)
    C = SparseEntityComplex(len(cells), r, pairs, reference_basis='bernstein')
    return {'r': r, 'reference_moment_condition_numbers': [float(v) for v in C.conds],
            'd_squared_max_entries': [float(v) for v in C.d2_defects],
            'derivative_invariance_max_entries': [float(v) for v in C.defects]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='omit r=5 conditioning and r=4,N=3 best-approx test')
    args = parser.parse_args()
    out = {'flat_constant_patch': [flat_constant_patch(r) for r in (1,2,3)],
           'broken_best_approximation': [best_broken_l2(N,r) for N,r in
                 ([(2,3),(2,4),(3,3)] if args.quick else [(2,3),(2,4),(3,3),(3,4)])],
           'matrix_audit': [matrix_defects(r) for r in
                 ((3,4) if args.quick else (3,4,5))]}
    print('AUDIT_JSON_START')
    print(json.dumps(out, indent=2, allow_nan=False))
    print('AUDIT_JSON_END')


if __name__ == '__main__':
    main()
