"""Variable-projection utilities for Trefftz skeleton least squares.

Author
------
Shelvean Kapita
Department of Mathematics, Texas A&M University

Date
----
August 2026

This module implements Part B of the manuscript in a form that is independent
of the PWDG state equation.  For a fixed set of local Trefftz directions, the
weighted skeleton residual can be written

    ||D(Z)c - b||_2^2.

The complex coefficient vector ``c`` is linear and is therefore eliminated by
least squares.  Only the direction parameters ``Z`` are passed to the nonlinear
optimizer.  This is the standard variable-projection idea specialized to the
Trefftz skeleton residual used in the paper.

The implementation below keeps the residual blocks visible: value jumps,
normal-flux jumps, and boundary Dirichlet mismatch.  This makes it easy to
compare each line of code with the definition of the functional in the paper.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

from .pwdg import ALPHA, BETA

__author__ = "Shelvean Kapita"
__date__ = "August 2026"


def build_trace_least_squares(state):
    """Assemble ``D`` and ``b`` for the weighted all-Dirichlet skeleton residual.

    Parameters
    ----------
    state:
        A :class:`core.direction_adaptive.PWDGDictionary` whose local wave
        vectors have already been specified.  The PWDG matrix does not need to
        be assembled and the Galerkin equations are not used.

    Returns
    -------
    D, b:
        Complex matrix and data vector such that ``||D@c-b||_2`` is the
        quadrature approximation of the physical skeleton residual.
    """
    rows = []
    rhs = []
    ncoeff = state.ndof

    for edge_index in range(state.et.shape[0]):
        points, weights, normal0, _ = state.edge(edge_index)
        e0, e1 = state.et[edge_index]
        sqrt_weights = np.sqrt(weights)

        if e1 >= 0:
            outward0 = state._outward(e0, normal0, points)
            xi = 0.5 * (state.kap[e0] + state.kap[e1])

            # --------------------------------------------------------------
            # Value jump: sqrt(alpha*xi) [u]
            # --------------------------------------------------------------
            value_block = np.zeros((len(points), ncoeff), dtype=complex)
            value_block[:, state.sl(e0)] = state._basis(e0, points)
            value_block[:, state.sl(e1)] = -state._basis(e1, points)
            value_scale = np.sqrt(ALPHA * xi) * sqrt_weights
            rows.append(value_scale[:, None] * value_block)
            rhs.append(np.zeros(len(points), dtype=complex))

            # --------------------------------------------------------------
            # Normal-flux jump: sqrt(beta/xi) [grad u . n]
            # --------------------------------------------------------------
            flux_block = np.zeros((len(points), ncoeff), dtype=complex)
            for element, outward in ((e0, outward0), (e1, -outward0)):
                basis = state._basis(element, points)
                normal_derivatives = 1j * basis * (state.q[element] @ outward)[None, :]
                flux_block[:, state.sl(element)] = normal_derivatives

            flux_scale = np.sqrt(BETA / xi) * sqrt_weights
            rows.append(flux_scale[:, None] * flux_block)
            rhs.append(np.zeros(len(points), dtype=complex))

        else:
            # --------------------------------------------------------------
            # Dirichlet boundary mismatch: sqrt(alpha*k) (u-g)
            # --------------------------------------------------------------
            element = e0
            boundary_block = np.zeros((len(points), ncoeff), dtype=complex)
            boundary_block[:, state.sl(element)] = state._basis(element, points)
            boundary_scale = np.sqrt(ALPHA * state.kap[element]) * sqrt_weights
            rows.append(boundary_scale[:, None] * boundary_block)

            exact_trace = state.exact(points[:, 0], points[:, 1])
            rhs.append(boundary_scale * exact_trace)

    return np.vstack(rows), np.concatenate(rhs)


def eliminate_coefficients(D, b, rcond:
    float = 1.0e-12):
    """Return the minimum-residual coefficient vector for fixed directions.

    A rank-revealing QR factorization is used when the matrix has full column
    rank.  A truncated SVD is used as a stable fallback when local direction
    redundancy makes the least-squares matrix rank deficient.
    """
    m, n = D.shape
    Q, R, piv = la.qr(D, mode="economic", pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(R))
    scale = diagonal[0] if len(diagonal) else 0.0
    rank = int(np.sum(diagonal > rcond * scale)) if scale > 0.0 else 0

    if rank == n:
        projected_rhs = Q.conj().T @ b
        pivoted_coefficients = la.solve_triangular(
            R,
            projected_rhs,
            lower=False,
            check_finite=False,
        )
        coefficients = np.empty(n, dtype=complex)
        coefficients[piv] = pivoted_coefficients
        return coefficients, rank

    # Rank-deficient fallback.
    U, singular_values, Vh = la.svd(D, full_matrices=False, check_finite=False)
    if len(singular_values) == 0:
        return np.zeros(n, dtype=complex), 0
    keep = singular_values > rcond * singular_values[0]
    coefficients = (
        Vh[keep].conj().T
        @ ((U[:, keep].conj().T @ b) / singular_values[keep])
    )
    return coefficients, int(np.sum(keep))


def projected_state(state, rcond:
    float = 1.0e-12):
    """Eliminate the coefficients and attach the LS solution to ``state``."""
    D, b = build_trace_least_squares(state)
    coefficients, rank = eliminate_coefficients(D, b, rcond=rcond)
    state.u = coefficients
    residual = D @ coefficients - b
    return state, residual, rank


def real_projected_residual(state, rcond:
    float = 1.0e-12):
    """Return the reduced residual as a real vector for SciPy optimization."""
    state, residual, _ = projected_state(state, rcond=rcond)
    return np.concatenate((residual.real, residual.imag))
