"""Matched exact hyperbolic / piecewise-flat scalar calculation.

Usage
-----
python reproduce_metric_comparison.py --level 0 --degree 2 --nq 8
python reproduce_metric_comparison.py --level 3 --degree 2 --nq 8

The level-3 experiment has 30,720 tetrahedra and requires considerably more
memory and time than the level-0 smoke test. Both runs use the same cells,
identifications, polynomial degree and quadrature; only metric provider differs.
"""
import argparse
import time
import numpy as np
import hypforms
import metriccompare


def manifold_data(level):
    a, rho, U, Fd, hv, hf, faces = hypforms.geometry()
    cells = []
    origin = np.array([1., 0., 0., 0.])
    for k, ids in enumerate(faces):
        for i in range(len(ids)):
            cells.append(np.stack([origin, hf[k], hv[ids[i]],
                                   hv[ids[(i+1) % len(ids)]]], axis=1))
    for _ in range(level):
        cells = hypforms.refine(cells)
    group = [np.eye(4)] + [hypforms.pairing(f, a) for f in Fd]
    return cells, group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', type=int, default=0)
    ap.add_argument('--degree', type=int, default=2)
    ap.add_argument('--nq', type=int, default=None)
    ap.add_argument('--neig', type=int, default=8)
    args = ap.parse_args()
    cells, group = manifold_data(args.level)
    print('tetrahedra:',len(cells),flush=True)
    for mode in ('model', 'regge'):
        t0=time.perf_counter()
        out=metriccompare.build(cells,group,-1,args.degree,
                                mode=mode,nq=args.nq,neig=args.neig)
        print(mode,'DOFs',out['ndof'],'volume',out['vol'],
              'first positive eigenvalue',out['lam'][1],
              'seconds',time.perf_counter()-t0,flush=True)

if __name__ == '__main__':
    main()
