"""
compare_fem.py: conformal spherical splines vs parametric surface FEM
(Dziuk P1, isoparametric P2, P3) on the spheroid and the dumbbell.

Poisson Test 1 (u~ = e^{x1} x2 + x1 x3^3 pushed to M): L2(M) and H1(M)
errors and wall-clock time against degrees of freedom.  Eigenvalues on the
spheroid: max error of the first 16 against the spherical-harmonic
reference (L = 60) and time of the shift-invert solve.

Timing convention: for the spline method the geometry-independent setup
(smoothness matrix, null-space matrix, round-sphere stiffness) is timed
separately from the per-surface work (weight at the quadrature points,
load vector, constraint vector, bordered solve); the FEM time is assembly
plus solve on the given surface.  All times are single-threaded wall
clock on the same machine in one session.
"""
import json, math, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from sphsplines import (icosphere, meshsize_sphere, smoothness_sphere,
                        assemble_sphere, interp_sphere, sphere_pencil)
from prolong import column_order, build_prolongation
from poisson_surface import manufactured, geometries, load_vector, errors as sp_errors
from spheroid import sh_reference
import surface_fem as fem

EXPR = 'exp(x1) * x2 + x3**3 * x1'
ut, grad_ut, lap_ut = manufactured(EXPR)
G = geometries()
rows = []


def spline_poisson(name, wgt, d, r, lev):
    v, t = icosphere(lev)
    m = (d + 1) * (d + 2) // 2
    t0 = time.time()
    J = smoothness_sphere(v, t, d, r)
    Z, D, Fr = build_prolongation(J, m * t.shape[0], column_order(v, t, d))
    _, K = assemble_sphere(v, t, d, q=12, weight=None)
    A = (Z.T @ K @ Z).tocsr()
    t_setup = time.time() - t0
    t0 = time.time()
    Fv = load_vector(v, t, d, wgt, lap_ut, q=12)
    c1 = interp_sphere(v, t, d, lambda p: np.ones(p.shape[1]))
    # constraint vector M_w c1 without the full weighted mass matrix:
    # it is the load vector of the function 1 with weight w
    Mwc1 = load_vector(v, t, d, wgt, lambda p: -(wgt(p) if wgt is not None else np.ones(p.shape[1])), q=12)
    g = np.asarray(Z.T @ Mwc1).ravel()
    b = np.asarray(Z.T @ Fv).ravel()
    n = A.shape[0]
    Aug = sp.bmat([[A, sp.csr_matrix(g[:, None])], [sp.csr_matrix(g[None, :]), None]], format='csc')
    sol = spla.spsolve(Aug, np.concatenate([b, [0.0]]))
    c = np.asarray(Z @ sol[:n]).ravel()
    t_surf = time.time() - t0
    e0, e1 = sp_errors(v, t, d, c, wgt, ut, grad_ut)
    return dict(method='spline', geo=name, deg=d, r=r, lev=lev, h=meshsize_sphere(v, t),
                dof=n, e0=e0, e1=e1, t_setup=t_setup, t_surf=t_surf)


def fem_poisson(name, wgt, k, lev):
    embed = (lambda p: p) if wgt is None else wgt.embed
    f = lambda s: -lap_ut(s) / (wgt(s) if wgt is not None else 1.0)
    ref = fem.RefElement(k, 6)
    t0 = time.time()
    mesh = fem.build_mesh(lev, k, embed)
    t_mesh = time.time() - t0
    uh, info, (K, M, qd) = fem.solve_poisson(mesh, ref, f)
    e0, e1 = fem.errors(mesh, ref, uh, qd, ut, grad_ut, wgt, None if wgt is None else embed)
    return dict(method=f'fem', geo=name, deg=k, r=0, lev=lev, h=mesh['h'],
                dof=info['dim'], e0=e0, e1=e1, t_setup=t_mesh, t_surf=info['time'])


def done_keys(path):
    keys=set()
    try:
        for line in open(path):
            if line.startswith('{'):
                r=json.loads(line); keys.add((r['method'],r.get('geo',''),r['deg'],r['lev']))
    except FileNotFoundError: pass
    return keys


if __name__ == '__main__':
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else 'all'
    done = done_keys('compare_fem.log')
    log = open('compare_fem.log', 'a')
    def emit(row):
        print(json.dumps(row)); log.write(json.dumps(row)+'\n'); log.flush()
    for name in (['spheroid', 'dumbbell'] if only in ('all','spheroid','dumbbell') else []):
        if only != 'all' and name != only: continue
        wgt = G[name]
        for d, r, levels in [(2, 0, [1, 2, 3, 4]), (4, 0, [1, 2, 3, 4]), (6, 1, [1, 2, 3])]:
            for lev in levels:
                if ('spline',name,d,lev) in done: continue
                row = spline_poisson(name, wgt, d, r, lev); emit(row)
        for k, levels in [(1, [1, 2, 3, 4, 5]), (2, [1, 2, 3, 4, 5]), (3, [1, 2, 3, 4])]:
            for lev in levels:
                if ('fem',name,k,lev) in done: continue
                row = fem_poisson(name, wgt, k, lev); emit(row)
    if only not in ('all','eigs'): sys.exit(0)

    # eigenvalues on the spheroid
    sw = G['spheroid']
    ref = sh_reference(sw.w_of_Theta, L=60, nG=400, k=16)
    erows = []
    for d, r, levels in [(2, 0, [1, 2, 3]), (4, 0, [1, 2, 3]), (6, 1, [1, 2])]:
        for lev in levels:
            v, t = icosphere(lev)
            t0 = time.time()
            Z, Kz, Mz, S, M, K = sphere_pencil(v, t, d, r, q=12, weight=sw)
            t_asm = time.time() - t0
            t0 = time.time()
            vals = np.sort(spla.eigsh(Kz.tocsc(), k=16, M=Mz.tocsc(), sigma=-1.0, which='LM', return_eigenvectors=False))
            t_eig = time.time() - t0
            row = dict(method='spline', deg=d, r=r, lev=lev, dof=Kz.shape[0],
                       err=float(np.abs(vals - ref).max()), t_asm=t_asm, t_eig=t_eig)
            erows.append(row); print(json.dumps(row))
    for k, levels in [(1, [1, 2, 3, 4, 5]), (2, [1, 2, 3, 4]), (3, [1, 2, 3, 4])]:
        for lev in levels:
            ref_el = fem.RefElement(k, 6)
            t0 = time.time()
            mesh = fem.build_mesh(lev, k, sw.embed)
            K, M, F, qd = fem.assemble(mesh, ref_el, None)
            t_asm = time.time() - t0
            vals, t_eig = fem.eigenvalues(K, M, 16)
            row = dict(method='fem', deg=k, r=0, lev=lev, dof=K.shape[0],
                       err=float(np.abs(vals - ref).max()), t_asm=t_asm, t_eig=t_eig)
            erows.append(row); print(json.dumps(row))
    json.dump(erows, open('compare_eigs.json', 'w'), indent=1)
