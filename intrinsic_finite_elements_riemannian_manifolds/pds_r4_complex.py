"""A level-1, order-4 reference complex without dense global rank/SVD."""
from __future__ import annotations
import time
from riemannfem.spherical import poincare_homology_sphere
from riemannfem.entity_assembly import SparseEntityComplex

if __name__=='__main__':
    t=time.perf_counter()
    mesh=poincare_homology_sphere(level=1)
    C=SparseEntityComplex(len(mesh['cells']),4,mesh['pairings'],reference_basis='bernstein')
    print('cells',len(mesh['cells']))
    print('quotient entity counts',C.orbits.counts())
    print('r=4 global dimensions',C.dimensions)
    print('reference moment condition numbers',C.conds)
    print('D^2 maximum entry',C.d2_defects)
    print('d conformity maximum entry',C.defects)
    print('wall_seconds',time.perf_counter()-t)
    assert C.dimensions==(428,1628,2000,800)
    assert max(C.defects)<1e-9 and max(C.d2_defects)<1e-9
