"""Sparse C0 Bernstein assembly on a periodic tetrahedral mesh.

Local scalar coefficients are joined by exact barycentric trace agreement.
All metric data come from intrinsic element chart tensors; no embedding occurs.
"""
from __future__ import annotations
import itertools
from functools import lru_cache
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .reference import SimplexBernstein, simplex_duffy
from .assembly import ConstraintOperator, assemble_constrained
from .torus_metric import AffineTetrahedron, conformal_factor, exact_solution, rhs


def periodic_freudenthal_mesh(N):
    """Conforming six-tetrahedra-per-cube mesh of the periodic coordinate torus."""
    N=int(N)
    if N<2: raise ValueError('use N>=2')
    I=np.eye(3,dtype=int)
    cells=[]
    for a in itertools.product(range(N),repeat=3):
        o=np.asarray(a,dtype=int)
        for perm in itertools.permutations(range(3)):
            V=[o.copy()]
            for j in perm:
                V.append(V[-1]+I[j])
            cells.append(np.asarray(V,dtype=int))
    return np.asarray(cells,dtype=int)


def periodic_bernstein_numbering(cells,N,p):
    """Map domain-point Bernstein coefficients to periodic global indices.

    Each point is represented *exactly* by its integer weighted-coordinate
    numerator, avoiding tolerance-dependent geometric matching.
    """
    ref=SimplexBernstein(3,p)
    table={}
    indices=np.empty((len(cells),ref.nloc),dtype=int)
    modulus=N*p
    for c,V in enumerate(cells):
        for j,alpha in enumerate(ref.indices):
            key=tuple((np.asarray(alpha,dtype=int)@V)%modulus)
            if key not in table: table[key]=len(table)
            indices[c,j]=table[key]
    return indices,table


@lru_cache(maxsize=12)
def reference_evaluations(p,q):
    B=SimplexBernstein(3,p)
    lam,w=simplex_duffy(3,q)
    vals=[];grads=[]
    for z in lam:
        v,d=B.values_grads(z)
        vals.append(v);grads.append(d)
    return lam[:,1:],w,np.asarray(vals),np.asarray(grads)


def _cell_terms(vertices,p,q,eval_data,base_metric=None):
    xhat,w,B,dB=eval_data
    cell=AffineTetrahedron(vertices)
    E=cell.E
    J=np.abs(np.linalg.det(E))
    Einv=np.linalg.inv(E)
    gx=cell.physical_coordinates(xhat)
    phi,_=conformal_factor(gx)
    Bphys=np.eye(3) if base_metric is None else np.asarray(base_metric,float)
    Binv=np.linalg.inv(Bphys)
    sf=np.sqrt(np.linalg.det(Bphys))
    f=rhs(gx,base_metric=base_metric)
    mass_w=w*J*sf*np.exp(3*phi)
    stiff_w=w*J*sf*np.exp(phi)
    gp=np.einsum('qai,ij->qaj',dB,Einv,optimize=True)
    gpB=np.einsum('qai,ij->qaj',gp,Binv,optimize=True)
    M=B.T@(mass_w[:,None]*B)
    K=sum(gpB[:,:,j].T@(stiff_w[:,None]*gp[:,:,j]) for j in range(3))
    load=B.T@(mass_w*f)
    return M,K,load


def solve_manufactured(N,p,q=None,evaluate_errors=True,base_metric=None):
    """Solve (-Delta_g+I)u=f with periodic C0 Bernstein elements."""
    N=int(N);p=int(p)
    if p<1:raise ValueError('p>=1 required')
    q=int(q or max(p+4,6))
    if base_metric is not None:
        base_metric=np.asarray(base_metric,float)
        if base_metric.shape!=(3,3) or np.linalg.eigvalsh(base_metric).min()<=0:
            raise ValueError('base metric must be a symmetric SPD 3x3 matrix')
    cells=periodic_freudenthal_mesh(N)
    ldofs,labels=periodic_bernstein_numbering(cells,N,p)
    ndof=len(labels)
    data=reference_evaluations(p,q)
    masses=[];stiffs=[];loads=[]
    for V in cells:
        M,K,b=_cell_terms(V/N,p,q,data,base_metric)
        masses.append(M);stiffs.append(K);loads.append(b)
    from .assembly import boolean_constraint
    P=boolean_constraint(ldofs,ndof)
    # Keep the same constrained variational assembly API as the form complex.
    M=assemble_constrained(masses,P)
    K=assemble_constrained(stiffs,P)
    L=P.P.T@np.concatenate(loads)
    A=K+M
    u=spla.spsolve(A,L)
    residual=np.linalg.norm(A@u-L)/max(1.,np.linalg.norm(L))
    result={'N':N,'p':p,'q':q,'cells':len(cells),'dofs':ndof,
            'residual':float(residual),'h':float(np.sqrt(3)/N),
            'metric':'isotropic conformal' if base_metric is None else 'anisotropic conformal'}
    if evaluate_errors:
        result.update(error_norms(cells,ldofs,u,N,p,q+2,base_metric=base_metric))
    return result


def error_norms(cells,ldofs,u,N,p,q,base_metric=None):
    xhat,w,B,dB=reference_evaluations(p,q)
    L2=H1=ueL2=0.
    Bphys=np.eye(3) if base_metric is None else np.asarray(base_metric,float)
    Binv=np.linalg.inv(Bphys)
    sf=np.sqrt(np.linalg.det(Bphys))
    for c,Vint in enumerate(cells):
        cell=AffineTetrahedron(Vint/N)
        E=cell.E
        J=abs(np.linalg.det(E))
        phi,_=conformal_factor(cell.physical_coordinates(xhat))
        ue,ge,_=exact_solution(cell.physical_coordinates(xhat))
        coefficients=u[ldofs[c]]
        uh=B@coefficients
        gh=np.einsum('qai,ij,a->qj',dB,np.linalg.inv(E),coefficients,optimize=True)
        value_error=ue-uh
        gradient_error=ge-gh
        wm=w*J*sf*np.exp(3*phi)
        wk=w*J*sf*np.exp(phi)
        L2+=float(np.dot(wm,value_error**2))
        H1+=float(np.dot(wk,np.einsum('qi,ij,qj->q',
                          gradient_error,Binv,gradient_error,optimize=True)))
        ueL2+=float(np.dot(wm,ue**2))
    return {'L2_error':float(np.sqrt(L2)),
            'H1_seminorm_error':float(np.sqrt(H1)),
            'relative_L2_error':float(np.sqrt(L2/ueL2))}


def periodic_tetrahedral_facet_pairings(cells,N):
    """Face pairings for the Freudenthal tetrahedra on the periodic 3-torus.

    Opposite cube faces are paired by coordinate translations. All other
    pairings are ordinary interior facets. Every matching is combinatorial,
    using exact integer coordinates in the periodic chart.
    """
    from collections import defaultdict
    from .global_complex import FacetPairing
    groups=defaultdict(list)
    for c,V in enumerate(cells):
        for f in range(4):
            vs=tuple(j for j in range(4) if j!=f)
            face=V[list(vs)]
            key=(tuple((face.sum(axis=0)%(3*N)).tolist()),
                 tuple(sorted(tuple((v%N).tolist()) for v in face)))
            groups[key].append((c,f,vs,face))
    pairs=[]
    for key,incidences in groups.items():
        if len(incidences)!=2:
            raise ValueError(f'non-manifold facet incidence {key}: {len(incidences)}')
        c0,f0,vs0,P=incidences[0]
        c1,f1,vs1,Q=incidences[1]
        perm=[]
        for v in P:
            matches=[j for j,q in enumerate(Q) if np.all((v-q)%N==0)]
            if len(matches)!=1:raise ValueError('nonunique facet vertex matching')
            perm.append(matches[0])
        pairs.append(FacetPairing(c0,f0,c1,f1,tuple(perm)))
    return pairs
