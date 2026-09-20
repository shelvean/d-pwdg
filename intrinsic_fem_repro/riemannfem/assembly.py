
from __future__ import annotations
import numpy as np
import scipy.sparse as sp


class ConstraintOperator:
    """Injection from independent global coefficients to broken coefficients.

    If c is the global coefficient vector, P c is the concatenated vector of
    local cell coefficients.  P may contain permutations, signs, or general
    small trace transformations.
    """
    def __init__(self, P):
        self.P = sp.csr_matrix(P)

    @property
    def nbroken(self):
        return self.P.shape[0]

    @property
    def ndof(self):
        return self.P.shape[1]


def block_diagonal(local_matrices):
    return sp.block_diag([sp.csr_matrix(A) for A in local_matrices], format="csr")


def assemble_constrained(local_matrices, constraints: ConstraintOperator):
    """Variational assembly A = P^T A_broken P."""
    Ab = block_diagonal(local_matrices)
    P = constraints.P
    if Ab.shape[0] != P.shape[0]:
        raise ValueError("broken matrix and constraint operator dimensions differ")
    return (P.T @ Ab @ P).tocsr()


def boolean_constraint(cell_dofs, ndof=None):
    """Build P for pure coefficient identification.

    cell_dofs[c,a] is the global index of local coefficient a on cell c.
    """
    cell_dofs = np.asarray(cell_dofs, dtype=int)
    nbroken = cell_dofs.size
    ndof = int(ndof if ndof is not None else cell_dofs.max()+1)
    rows = np.arange(nbroken, dtype=int)
    cols = cell_dofs.ravel()
    data = np.ones(nbroken)
    return ConstraintOperator(sp.coo_matrix((data,(rows,cols)),
                                            shape=(nbroken,ndof)))
