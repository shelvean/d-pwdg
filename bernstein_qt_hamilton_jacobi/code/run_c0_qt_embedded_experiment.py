#!/usr/bin/env python3
"""Experiment 1: full C0 parent approximation versus genuine embedded C0-qT reduction.

The qT state is not obtained by minimizing all Hamiltonian moments in the parent
space.  Instead, after global C0 assembly, outer-boundary B-coefficients are kept
as trace coordinates and all remaining C0 coordinates are implicitly condensed
by an independent square subset of the p-1 Hamiltonian moment equations.
Selected moments are solved to nonlinear tolerance; unselected moments are a
consistency defect.  C0 continuity is exact because all shared edge coefficients
have one global index before the qT reduction is constructed.
"""
from __future__ import annotations
import math, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
from optimized_hj import FastHamiltonJacobiPminus1
from eikonal_c0_pminus1 import triangle_quadrature, bernstein_values
from conforming_embedded_qt import ConformingEmbeddedQT
from run_c0_qt_first_experiment import RotatingAnisotropicHJ, full_c0_projection

DEG=(3,5,7,9,11,13)
HLEVELS=(2,3,4,5,6,7)


def all_residual_rms(model,c):
    model._base=np.asarray(c).copy(); model._cache_z=None
    R=model.residual(model._base[model.free])
    return float(np.linalg.norm(R)/math.sqrt(len(R)))


def run_one(n,p,prob):
    m=FastHamiltonJacobiPminus1(n,p,prob)
    cf,tp=full_c0_projection(m); f2,f1=m.errors(cf); fr=all_residual_rms(m,cf)
    t0=time.perf_counter(); qt=ConformingEmbeddedQT(m,cf); setup=time.perf_counter()-t0
    cq,sol=qt.solve_trace(initial_full=cf,max_iter=50,atol=2e-13,rtol=2e-11)
    q2,q1=m.errors(cq)
    return dict(n=n,ntri=len(m.tris),p=p,
                global_c0_dofs=qt.parent_dim,global_qt_trace_dim=qt.trace_dim,
                dependent_c0_dofs=qt.dependent_dim,global_compression=qt.parent_dim/qt.trace_dim,
                local_parent_dim=(p+1)*(p+2)//2,local_triangle_qt_dim=p+1,
                c0_relL2=f2,qt_relL2=q2,L2_ratio=q2/f2,
                c0_relH1=f1,qt_relH1=q1,H1_ratio=q1/f1,
                c0_residual_rms=fr,qt_selected_rms=sol.selected_rms,
                qt_remaining_rms=sol.remaining_rms,qt_full_rms=sol.full_rms,
                c0_projection_time=tp,qt_setup_time=setup,qt_condensation_time=sol.seconds,
                qt_nfev=sol.nfev,qt_success=sol.success)


def main():
    prob=RotatingAnisotropicHJ(); out=ROOT/'data'; out.mkdir(exist_ok=True)
    rp=[]
    for p in DEG:
        r=run_one(4,p,prob); print('p',r,flush=True); rp.append(r)
    pd.DataFrame(rp).to_csv(out/'c0_qt_embedded_rotating_anisotropic_p_sweep.csv',index=False)
    rh=[]
    for n in HLEVELS:
        r=run_one(n,7,prob); print('h',r,flush=True); rh.append(r)
    pd.DataFrame(rh).to_csv(out/'c0_qt_embedded_rotating_anisotropic_h_sweep.csv',index=False)

if __name__=='__main__': main()
