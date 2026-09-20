
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from .forms import (
    TrimmedPolynomialFormSpace,
    simplex_vertex_permutation_affine,
)


@dataclass(frozen=True)
class FacetPairing:
    """
    Identification of two tetrahedral facets.

    vertex_permutation is a tuple p of length 3.  If the plus facet uses its
    canonical local vertex order q_plus and the minus facet uses q_minus, then
    vertex j of q_plus is identified with vertex p[j] of q_minus.
    """
    cell_plus: int
    facet_plus: int
    cell_minus: int
    facet_minus: int
    vertex_permutation: tuple[int, int, int]


def trace_constraint_matrix(ncells: int, r: int, k: int, pairings):
    """
    Broken-to-global conformity constraints for P_r^- Lambda^k on tetrahedra.

    The local algebraic basis need not be entity based.  Continuity is imposed
    exactly through trace pullbacks:
        R_+ c_+ - T_F R_- c_- = 0.
    """
    V = TrimmedPolynomialFormSpace(3, r, k)
    nloc = V.nloc

    if k == 3:
        return np.zeros((0, ncells*nloc))

    W = TrimmedPolynomialFormSpace(2, r, k)
    blocks = []

    for p in pairings:
        Rplus = V.facet_trace_matrix(p.facet_plus)
        Rminus = V.facet_trace_matrix(p.facet_minus)

        A, b = simplex_vertex_permutation_affine(p.vertex_permutation)
        T = W.pullback_matrix(A, b)

        B = np.zeros((W.nloc, ncells*nloc))
        a0 = p.cell_plus*nloc
        b0 = p.cell_minus*nloc
        B[:, a0:a0+nloc] = Rplus
        B[:, b0:b0+nloc] = -T @ Rminus
        blocks.append(B)

    if not blocks:
        return np.zeros((0, ncells*nloc))
    return np.vstack(blocks)


class DenseConstrainedComplex:
    """
    Exact reference implementation of global assembly by trace constraints.

    P_k is an orthonormal basis for the nullspace of the facet constraint
    matrix.  It injects global coefficients into the broken cell space.

    This implementation is deliberately dense so that every algebraic identity
    can be audited.  It is intended for modest quotient meshes and regression
    tests.  A sparse entity-based assembly will use the same trace maps.
    """

    def __init__(self, ncells: int, r: int, pairings, rcond=1e-10):
        self.ncells = int(ncells)
        self.r = int(r)
        self.pairings = list(pairings)
        self.rcond = float(rcond)

        self.spaces = [TrimmedPolynomialFormSpace(3, r, k) for k in range(4)]
        self.constraints = []
        self.P = []

        for k, V in enumerate(self.spaces):
            C = trace_constraint_matrix(self.ncells, r, k, self.pairings)
            self.constraints.append(C)
            if C.shape[0] == 0:
                P = np.eye(self.ncells*V.nloc)
            else:
                P = la.null_space(C, rcond=self.rcond)
            self.P.append(P)

        self.D = []
        self.derivative_invariance_defect = []
        for k in range(3):
            Dl = self.spaces[k].exterior_derivative_matrix()
            Dbroken = np.kron(np.eye(self.ncells), Dl)

            Y = Dbroken @ self.P[k]
            coeff = self.P[k+1].T @ Y
            residual = Y - self.P[k+1] @ coeff

            self.D.append(coeff)
            self.derivative_invariance_defect.append(
                float(np.linalg.norm(residual, ord=np.inf))
            )

    @property
    def dimensions(self):
        return tuple(P.shape[1] for P in self.P)

    @property
    def d2_defects(self):
        return tuple(
            float(np.linalg.norm(self.D[k+1] @ self.D[k], ord=np.inf))
            for k in range(2)
        )

    def betti_numbers(self, tol=1e-9):
        ranks = [np.linalg.matrix_rank(D, tol=tol) for D in self.D]
        dims = self.dimensions
        b = []
        for k in range(4):
            rin = ranks[k-1] if k > 0 else 0
            rout = ranks[k] if k < 3 else 0
            b.append(int(dims[k] - rin - rout))
        return tuple(b)

    def assemble_hodge_masses(self, local_masses):
        """
        local_masses[k][cell] is the local Hodge matrix for k-forms.
        """
        M = []
        for k in range(4):
            mats = local_masses[k]
            if len(mats) != self.ncells:
                raise ValueError("wrong number of cell matrices")
            Mbroken = la.block_diag(*mats)
            P = self.P[k]
            Mg = P.T @ Mbroken @ P
            M.append(0.5*(Mg+Mg.T))
        return M

    def hodge_laplacian(self, M, k):
        """
        Matrix for
            (d u, d v) + (delta u, delta v)
        in the global coefficient basis.
        """
        Mk = M[k]
        A = np.zeros_like(Mk)

        if k < 3:
            Dk = self.D[k]
            A += Dk.T @ M[k+1] @ Dk

        if k > 0:
            Dm = self.D[k-1]
            X = la.solve(M[k-1], Dm.T @ Mk, assume_a="pos")
            A += Mk @ Dm @ X

        return 0.5*(A+A.T)

    def hodge_spectrum(self, M, k):
        A = self.hodge_laplacian(M, k)
        return la.eigh(A, M[k], eigvals_only=True)
