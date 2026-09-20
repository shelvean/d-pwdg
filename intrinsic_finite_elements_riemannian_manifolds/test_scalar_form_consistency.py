"""Scalar gradient-form identity on the *same* intrinsically curved mesh.

A generic scalar Bernstein implementation and a separately assembled trimmed
complex should give identical generalized Laplace-Beltrami eigenvalues.
This test exercises the metric tensor, facet assembly and exterior derivative.
"""
import numpy as np
import scipy.linalg as la
from riemannfem.periodic_scalar import (periodic_freudenthal_mesh,
    periodic_tetrahedral_facet_pairings,periodic_bernstein_numbering,
    reference_evaluations,_cell_terms)
from riemannfem.torus_metric import AffineTetrahedron
from riemannfem.assembly import boolean_constraint,assemble_constrained
from riemannfem.entity_assembly import SparseEntityComplex
from riemannfem.bernstein_forms import local_hodge_mass_bernstein


def check(Bphys=None):
    N=2;r=2;q=6
    cells=periodic_freudenthal_mesh(N)
    pairings=periodic_tetrahedral_facet_pairings(cells,N)
    entity=SparseEntityComplex(len(cells),r,pairings,reference_basis='bernstein')
    assert entity.dimensions[0]==64
    G=[AffineTetrahedron(V/N).pulled_back_metric(Bphys) for V in cells]
    M0=entity.assemble_mass(0,[local_hodge_mass_bernstein(entity.spaces[0],g,q)
                                for g in G],local_basis='bernstein')
    M1=entity.assemble_mass(1,[local_hodge_mass_bernstein(entity.spaces[1],g,q)
                                for g in G],local_basis='bernstein')
    D0=entity.D[0]
    K0=(D0.T@M1@D0).toarray()
    lam_form=la.eigh(K0,M0.toarray(),eigvals_only=True)
    numbering,labels=periodic_bernstein_numbering(cells,N,r)
    constr=boolean_constraint(numbering,len(labels))
    mats=[];stiffs=[]
    eval_data=reference_evaluations(r,q)
    for V in cells:
        m,k,_=_cell_terms(V/N,r,q,eval_data,Bphys)
        mats.append(m);stiffs.append(k)
    Ms=assemble_constrained(mats,constr)
    Ks=assemble_constrained(stiffs,constr)
    lam_scalar=la.eigh(Ks.toarray(),Ms.toarray(),eigvals_only=True)
    rel=np.max(np.abs(lam_form-lam_scalar))/max(1.,np.max(np.abs(lam_scalar)))
    print('metric:', 'isotropic' if Bphys is None else 'anisotropic',
          'scalar vs form generalized spectrum relative difference',rel,
          'lowest positive:',lam_scalar[1],flush=True)
    assert rel<5e-10
    return rel

if __name__=='__main__':
    check()
    check(np.array([[1.20,.17,.04],[.17,.91,.08],[.04,.08,1.10]]))
    print('SCALAR/FORM INTRINSIC CONSISTENCY TESTS PASSED')
