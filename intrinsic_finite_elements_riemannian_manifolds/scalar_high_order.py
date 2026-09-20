"""High-order scalar Laplace--Beltrami manufactured-solution benchmark.

Run from the package root: python scalar_high_order.py --N 5 --p 5
The manifold is the intrinsic periodic three-torus with g=e^(2phi) B.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from riemannfem.periodic_scalar import solve_manufactured
from riemannfem.torus_metric import scalar_curvature
from oneform_high_order import B_ANISO


def run(N,p,anisotropic=False,quad=None):
    t0=time.perf_counter()
    q=quad if quad else max(p+5,7)
    B=B_ANISO if anisotropic else None
    result=solve_manufactured(N,p,q=q,base_metric=B)
    return {**result,'metric': 'anisotropic' if anisotropic else 'isotropic',
            'seconds': round(time.perf_counter()-t0,2)}

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--N',type=int,default=5)
    ap.add_argument('--p',type=int,default=5)
    ap.add_argument('--quad',type=int)
    ap.add_argument('--anisotropic',action='store_true')
    ap.add_argument('--results',default='results/scalar.jsonl')
    a=ap.parse_args()
    if a.N<2 or a.p<1:ap.error('N>=2, p>=1 required')
    res=run(a.N,a.p,a.anisotropic,a.quad)
    print(json.dumps(res),flush=True)
    dest=Path(a.results);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('a',encoding='utf8') as f:f.write(json.dumps(res)+'\n')
