#!/usr/bin/env python3
"""Conforming embedded quasi-Trefftz reduction for stationary HJ equations.

The parent coordinates are the already assembled global C0 Bernstein coordinates.
All outer-boundary Bernstein coefficients are retained as trace coordinates.  The
remaining (dependent) C0 coordinates -- including coefficients on interior mesh
edges, which are single shared global unknowns -- are determined by a square,
full-structural-rank subset of the degree-(p-1) Hamiltonian moment equations.

Thus the nonlinear qT map is an implicit condensation

    c_boundary  ->  c_dependent(c_boundary)

inside S_p^0(Delta).  The selected moment equations are solved to nonlinear
solver tolerance; the unselected moments are retained as a consistency defect.
No interelement coefficient is duplicated, so C0 conformity is exact throughout.
"""
from __future__ import annotations
import math, time
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse.linalg import spsolve
import scipy.linalg as la


def _row_matching(J):
    """Select one structurally independent residual row per dependent column."""
    A = J.copy().tocsr()
    A.data[:] = 1.0
    # Rows of A.T are dependent columns, columns of A.T are residual rows.
    m = maximum_bipartite_matching(A.T, perm_type='column')
    # With perm_type='column', result has one entry per row of A.T.
    rows = np.asarray(m, dtype=int)
    if rows.shape[0] != A.shape[1] or np.any(rows < 0):
        # alternate orientation, defensive fallback
        m2 = maximum_bipartite_matching(A, perm_type='row')
        rows = np.asarray(m2, dtype=int)
    if rows.shape[0] != A.shape[1] or np.any(rows < 0):
        raise RuntimeError('Could not find a full structural row matching for qT condensation')
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError('Structural matching returned duplicate residual rows')
    return rows


class ConformingEmbeddedQT:
    """Implicit C0-qT condensation of a FastHamiltonJacobiPminus1 model.

    The model's boundary B-coefficients are the retained coordinates.  All other
    globally assembled C0 coefficients are dependent qT coordinates.
    """
    def __init__(self, model, seed_full=None):
        self.model = model
        self.trace = np.asarray(model.fixed, dtype=int).copy()
        self.dependent = np.asarray(model.free, dtype=int).copy()
        self.seed = model.initial_coefficients() if seed_full is None else np.asarray(seed_full,float).copy()
        self.seed[self.trace] = model.fixed_vals
        model._base = self.seed.copy(); model._cache_z = None
        z = self.seed[self.dependent].copy()
        J = model.jacobian(z)
        # Numerical row pivoting of J^T selects a well-conditioned independent
        # subset of moment equations for the implicit qT condensation.
        # This is done once per branch/degree, not during Newton iterations.
        _Q,_R,_piv = la.qr(J.toarray().T, mode='economic', pivoting=True, check_finite=False)
        self.selected_rows = np.asarray(_piv[:J.shape[1]], dtype=int)
        self.remaining_rows = np.setdiff1d(np.arange(J.shape[0],dtype=int), self.selected_rows, assume_unique=False)
        self.structural_square_shape = (len(self.selected_rows), len(self.dependent))

    def solve_trace(self, trace_values=None, initial_full=None, max_iter=40,
                    atol=1e-12, rtol=1e-11, step_tol=1e-13, verbose=0):
        """Solve selected qT moments exactly for dependent C0 coordinates."""
        m=self.model
        c = self.seed.copy() if initial_full is None else np.asarray(initial_full,float).copy()
        tv = m.fixed_vals if trace_values is None else np.asarray(trace_values,float)
        c[self.trace] = tv
        m._base = c.copy(); m._base[self.trace] = tv; m._cache_z=None
        z = m._base[self.dependent].copy()
        t0=time.perf_counter(); nfev=0; njev=0; success=False; msg='maximum iterations reached'
        Rall=m.residual(z); nfev+=1; R=Rall[self.selected_rows]
        r0=max(np.linalg.norm(R),1e-30)
        for it in range(max_iter):
            rn=float(np.linalg.norm(R))
            if verbose: print('qT it',it,'selected rms',rn/math.sqrt(max(1,len(R))))
            if rn <= atol + rtol*r0:
                success=True; msg='selected qT moment tolerance reached'; break
            Jall=m.jacobian(z); njev+=1
            A=Jall[self.selected_rows,:].tocsc()
            try:
                step=spsolve(A,-R)
            except Exception:
                step=np.full_like(z,np.nan)
            if not np.all(np.isfinite(step)):
                msg='singular qT pivot block'; break
            if np.linalg.norm(step) <= step_tol*(1+np.linalg.norm(z)):
                success=rn <= 100*(atol+rtol*r0); msg='qT step tolerance reached'; break
            phi0=.5*rn*rn; alpha=1.0; accepted=False
            for _ in range(14):
                zt=z+alpha*step
                Rtall=m.residual(zt); nfev+=1; Rt=Rtall[self.selected_rows]
                if .5*np.dot(Rt,Rt) < phi0:
                    z=zt;Rall=Rtall;R=Rt;accepted=True;break
                alpha*=.5
            if not accepted:
                msg='qT line search stalled';break
        m._base[self.dependent]=z
        # refresh the complete residual at accepted state
        Rall=m.residual(z)
        selected_rms=float(np.linalg.norm(Rall[self.selected_rows])/math.sqrt(max(1,len(self.selected_rows))))
        remaining_rms=float(np.linalg.norm(Rall[self.remaining_rows])/math.sqrt(max(1,len(self.remaining_rows)))) if len(self.remaining_rows) else 0.0
        full_rms=float(np.linalg.norm(Rall)/math.sqrt(max(1,len(Rall))))
        sec=time.perf_counter()-t0
        out=m._base.copy()
        sol=SimpleNamespace(success=success,nfev=nfev,njev=njev,message=msg,
                            selected_rms=selected_rms,remaining_rms=remaining_rms,
                            full_rms=full_rms,seconds=sec)
        return out,sol

    @property
    def parent_dim(self): return len(self.model.asm.keys)
    @property
    def trace_dim(self): return len(self.trace)
    @property
    def dependent_dim(self): return len(self.dependent)
