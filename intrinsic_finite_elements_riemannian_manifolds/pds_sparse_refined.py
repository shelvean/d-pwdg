"""Sparse 40-cell Poincare quotient: dimensions, complex and scalar spectrum."""
from __future__ import annotations
import argparse, time
import numpy as np
import scipy.sparse.linalg as sla
from riemannfem.spherical import poincare_homology_sphere
from riemannfem.entity_assembly import SparseEntityComplex
from riemannfem.forms import local_hodge_mass
from riemannfem.integration import mesh_volume


def run(r=3,level=1,q=7,spectrum=True):
    t0=time.perf_counter()
    mesh=poincare_homology_sphere(level)
    C=SparseEntityComplex(len(mesh['cells']),r,mesh['pairings'])
    print(f'r={r} level={level} cells={len(mesh["cells"])}',flush=True)
    print('entity orbits (V,E,F,T)',C.orbits.counts(),flush=True)
    print('global dimensions',C.dimensions,flush=True)
    print('global D nnz',[D.nnz for D in C.D], 'P nnz',[P.nnz for P in C.P],flush=True)
    print('d preserves constraints (max-entry)',C.defects,flush=True)
    print('d*d defects (max-entry)',C.d2_defects,flush=True)
    print('metric integration volume',mesh_volume(mesh['metrics'],q),
          'exact',mesh['exact_volume'],flush=True)
    result={'r':r,'level':level,'cells':len(mesh['cells']),
            'entities':C.orbits.counts(),'dimensions':C.dimensions,
            'P_nnz':[P.nnz for P in C.P], 'D_nnz':[D.nnz for D in C.D],
            'd_invariance':C.defects,'d2':C.d2_defects}
    if spectrum:
        M0=C.assemble_mass(0,[local_hodge_mass(C.spaces[0],g,q=q) for g in mesh['metrics']])
        M1=C.assemble_mass(1,[local_hodge_mass(C.spaces[1],g,q=q) for g in mesh['metrics']])
        K=(C.D[0].T@M1@C.D[0]).tocsr()
        lam=sla.eigsh(K,M=M0,k=min(16,M0.shape[0]-2),sigma=-0.1,
                       which='LM',return_eigenvectors=False,tol=1e-10)
        lam=np.sort(np.real(lam))
        print('first scalar eigenvalues',lam,flush=True)
        print('first nonzero cluster, target 168',lam[1:min(14,len(lam))],flush=True)
        result['scalar_eigenvalues']=lam.tolist()
    result['seconds']=time.perf_counter()-t0
    print(f'wall time {result["seconds"]:.2f}s',flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--r',type=int,default=3)
    p.add_argument('--level',type=int,default=1)
    p.add_argument('--quad',type=int,default=7)
    p.add_argument('--no-spectrum',action='store_true')
    args=p.parse_args()
    run(args.r,args.level,args.quad,not args.no_spectrum)
