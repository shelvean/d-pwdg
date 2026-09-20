from __future__ import annotations
import numpy as np
from riemannfem.bernstein_forms import (bernstein_change,reference_entity_dofs_bernstein,
    local_hodge_mass_bernstein,reference_differential_bernstein,reference_affine_pullback_bernstein)
from riemannfem.forms import TrimmedPolynomialFormSpace, local_hodge_mass, simplex_vertex_permutation_affine
from riemannfem.entity_assembly import reference_entity_dofs
from riemannfem.metric import CallableMetric


def main():
    print('Reference moment matrix conditioning: algebraic -> Bernstein QR')
    for r in (1,2,3,4):
        for k in range(4):
            V=TrimmedPolynomialFormSpace(3,r,k)
            C,R,Q=bernstein_change(3,r,k)
            Eb,Ein,_,co=reference_entity_dofs_bernstein(r,k)
            Ea,_,_,co_old=reference_entity_dofs(r,k)
            iderr=np.max(np.abs(Eb@Ein-np.eye(V.nloc)))
            qerr=np.max(np.abs(Q.T@Q-np.eye(V.nloc)))
            print(f'r={r}, k={k}, n={V.nloc}: {co_old:.4e} -> {co:.4e}; '
                  f'identity={iderr:.2e}, QR={qerr:.2e}',flush=True)
            assert np.linalg.norm(Eb-Ea@C,np.inf)<2e-8
            assert iderr < (2e-8 if r==4 else 1e-10)
            assert qerr<1e-11
            assert co < co_old * 1.01

        Ds=[reference_differential_bernstein(3,r,k) for k in range(3)]
        for k in (0,1):
            defect=np.linalg.norm(Ds[k+1]@Ds[k],np.inf)
            print('  r',r,'k',k,'d^2 defect',defect)
            assert defect < (2e-9 if r==4 else 1e-10)
        if r>3:continue
        A,b=simplex_vertex_permutation_affine((0,2,1,3))
        Ts=[reference_affine_pullback_bernstein(TrimmedPolynomialFormSpace(3,r,k),A,b)
            for k in range(4)]
        for k in range(3):
            err=np.linalg.norm(Ds[k]@Ts[k]-Ts[k+1]@Ds[k],np.inf)
            assert err < 1e-9

    def g(x):
        s=1+.12*np.sin(x[0]+1.7*x[1]-.8*x[2])
        return s*s*np.array([[1.15,.08,.03],[.08,.95,.05],[.03,.05,1.08]])
    G=CallableMetric(3,g)
    for k in range(4):
        V=TrimmedPolynomialFormSpace(3,3,k)
        C,_,_=bernstein_change(3,3,k)
        M_old=local_hodge_mass(V,G,q=7)
        M_new=local_hodge_mass_bernstein(V,G,q=7)
        rel=np.linalg.norm(M_new-C.T@M_old@C)/np.linalg.norm(M_new)
        print('k',k,'variable metric mass transform residual',rel)
        assert rel<1e-12
    print('BERNSTEIN REFERENCE TESTS PASSED')
if __name__=='__main__':main()
