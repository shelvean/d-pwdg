"""Bernstein-coordinate orthonormalization of trimmed reference form spaces.

The reference forms retain their algebraic definition P_r^- Lambda^k.
For each coordinate k-form component, a total-degree-r Bernstein expansion
is formed exactly by degree elevation of monomials. A thin QR of the resulting
coefficient matrix gives a well-scaled basis for that same polynomial space.

The coordinate-change matrix C satisfies
    coefficients_in_algebraic_basis = C @ coefficients_in_bernstein_basis.
Global unknowns remain entity moments and do not depend on the local basis.
"""
from __future__ import annotations
from functools import lru_cache
from math import factorial
import numpy as np
import scipy.linalg as la
from .forms import TrimmedPolynomialFormSpace, monomial_indices, local_hodge_mass


def _multinomial(alpha):
    return factorial(sum(alpha))/np.prod([factorial(int(a)) for a in alpha])


@lru_cache(maxsize=None)
def monomial_to_bernstein(n:int, r:int, k:int):
    """Coordinate conversion P_r Lambda^k, monomials to degree-r Bernstein."""
    V=TrimmedPolynomialFormSpace(n,r,k)
    alphas=monomial_indices(n+1,r,homogeneous=True)
    index={a:i for i,a in enumerate(alphas)}
    wedges=V.ambient.wedges
    windex={I:j for j,I in enumerate(wedges)}
    nw=len(wedges)
    H=np.zeros((len(alphas)*nw,V.ambient.dim))
    for col,(beta,I) in enumerate(V.ambient.keys):
        d=sum(beta)
        base=(0,)+tuple(beta)
        for gamma in monomial_indices(n+1,r-d,homogeneous=True):
            alpha=tuple(b+g for b,g in zip(base,gamma))
            coeff=_multinomial(gamma)/_multinomial(alpha)
            H[index[alpha]*nw+windex[I],col] += coeff
    return H


@lru_cache(maxsize=None)
def bernstein_change(n:int,r:int,k:int):
    """Return C,Cinv,Q for algebraic-to-Bernstein-orthonormal basis.

    For an algebraic basis matrix B and monomial-to-Bernstein converter H,
    H B C=Q with Q.T Q=I in the Euclidean Bernstein coefficient metric.
    The matrix C maps new coefficients to the old algebraic coefficients.
    """
    V=TrimmedPolynomialFormSpace(n,r,k)
    H=monomial_to_bernstein(n,r,k) @ V.B
    Q,R=la.qr(H,mode='economic')
    C=la.solve_triangular(R,np.eye(V.nloc))
    # Reconstruct rather than invert C: Q and R define the exact relation.
    return C,R,Q


@lru_cache(maxsize=None)
def reference_entity_dofs_bernstein(r:int,k:int):
    from .entity_assembly import reference_entity_dofs
    E,_,entries,cond_old=reference_entity_dofs(r,k)
    C,_,_=bernstein_change(3,r,k)
    Eb=E@C
    Ein=la.solve(Eb,np.eye(Eb.shape[0]))
    cond=float(np.linalg.cond(Eb))
    return Eb,Ein,entries,cond


def local_hodge_mass_bernstein(space:TrimmedPolynomialFormSpace,metric,q=None):
    """Compute mass with polynomial values expressed in Bernstein-QR basis.

    Quadrature uses the intrinsic reference-cell metric and never an ambient
    Euclidean embedding. The algorithm evaluates a transformed form basis at
    each quadrature node. It shares the same quadrature and tensor layer as
    the original algebraic basis.
    """
    from .forms import wedge_metric_matrix
    from .reference import simplex_duffy
    if metric.dim != space.n: raise ValueError('metric dimension mismatch')
    C,_,_=bernstein_change(space.n,space.r,space.k)
    q=int(q or max(space.r+3,5))
    lams,weights=simplex_duffy(space.n,q)
    M=np.zeros((space.nloc,space.nloc))
    for lam,w in zip(lams,weights):
        x=lam[1:]
        G=np.asarray(metric.metric(x),float)
        if G.shape!=(space.n,space.n): raise ValueError('invalid metric shape')
        det=np.linalg.det(G)
        if det<=0: raise ValueError('metric must be positive definite')
        H=wedge_metric_matrix(np.linalg.inv(G),space.k)
        V=C.T@space.values(x)
        M+=w*np.sqrt(det)*(V@H@V.T)
    return .5*(M+M.T)


def reference_differential_bernstein(n:int,r:int,k:int):
    """Matrix d in the Bernstein-QR bases at consecutive form degrees."""
    if k==n:return np.zeros((0,TrimmedPolynomialFormSpace(n,r,k).nloc))
    V=TrimmedPolynomialFormSpace(n,r,k)
    C,_,_=bernstein_change(n,r,k)
    _,Rnext,_=bernstein_change(n,r,k+1)
    return Rnext@V.exterior_derivative_matrix()@C


def reference_affine_pullback_bernstein(source, A,b,target=None):
    """Pullback in Bernstein-QR coordinates (source and target form spaces)."""
    import numpy as np
    if target is None:target=source
    T=source.pullback_matrix(A,b,target=target)
    C,_,_=bernstein_change(source.n,source.r,source.k)
    _,Rt,_=bernstein_change(target.n,target.r,target.k)
    return Rt@T@C
