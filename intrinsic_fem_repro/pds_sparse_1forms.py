"""Positive coexact 1-form spectrum using the sparse high-order complex.

The operator d_1^* d_1 has a large kernel of exact forms.  Shift-invert near
lambda=4 isolates low positive coexact modes; these are NOT the full Hodge
spectrum and must not be reported as such.
"""
import argparse, time
import numpy as np
import scipy.sparse.linalg as sla
from riemannfem.spherical import poincare_homology_sphere
from riemannfem.entity_assembly import SparseEntityComplex
from riemannfem.forms import local_hodge_mass


def run(r=3,level=1,q=7,sigma=4.1):
    t=time.perf_counter()
    mesh=poincare_homology_sphere(level)
    C=SparseEntityComplex(len(mesh['cells']),r,mesh['pairings'])
    masses={}
    for k in (1,2):
        masses[k]=C.assemble_mass(k,[local_hodge_mass(C.spaces[k],g,q=q)
                                     for g in mesh['metrics']])
    A=(C.D[1].T@masses[2]@C.D[1]).tocsr()
    lam=sla.eigsh(A,M=masses[1],k=min(12,C.dimensions[1]-2),
                  sigma=sigma,which='LM',return_eigenvectors=False,tol=1e-10)
    lam=np.sort(np.real(lam))
    print(f'r={r} level={level} cells={len(mesh["cells"])} '
          f'1-form DOFs={C.dimensions[1]} 2-form DOFs={C.dimensions[2]}')
    print('coexact 1-form low positive eigenvalues near shift',lam)
    print('seconds',time.perf_counter()-t)
    return lam

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--r',type=int,default=3)
    p.add_argument('--level',type=int,default=1)
    p.add_argument('--quad',type=int,default=7)
    args=p.parse_args()
    run(args.r,args.level,args.quad)
