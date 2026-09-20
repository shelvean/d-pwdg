#!/usr/bin/env python3
from pathlib import Path
import math,sys,time
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from conforming_embedded_qt import ConformingEmbeddedQT
from qt_projection_utils import project_unknown
from optimized_hj import FastHamiltonJacobiPminus1,FastFactoredEikonalPminus1
from high_frequency_benchmarks import ZeroFactor,RadialFactor
from eikonal_c0_pminus1 import ManufacturedEikonal,PolynomialEikonal
from hj_general_benchmarks import AnisotropicQuadraticHJ,QuarticHJ
from run_focal_caustic_sweep import FocalCaustic
DEG=(3,5,7,9,11,13)

def rec(model,c,sol,extra=None):
    e=model.errors(c); R=model.residual(c[model.free])
    d=dict(n=getattr(model,'n',None),ntri=len(model.tris),p=model.p,full_c0_dofs=len(model.asm.keys),
           retained_trace_dim=len(model.fixed),dependent_dofs=len(model.free),
           relL2=e[0],relH1=e[1],absLinf=e[2] if len(e)>2 else np.nan,
           selected_rms=sol.selected_rms,remaining_rms=sol.remaining_rms,full_residual_rms=sol.full_rms,
           nfev=sol.nfev,seconds=sol.seconds,success=sol.success,solver='embedded C0-qT sparse Newton')
    if extra:d.update(extra)
    return d

def solve_qt(model, target_fun):
    seed,_=project_unknown(model,target_fun)
    q=ConformingEmbeddedQT(model,seed)
    c,s=q.solve_trace(initial_full=seed,max_iter=50,atol=2e-13,rtol=2e-11)
    return c,s

# general HJ
rows=[]
for prob in (AnisotropicQuadraticHJ(),QuarticHJ()):
  for p in DEG:
    m=FastHamiltonJacobiPminus1(3,p,prob);c,s=solve_qt(m,prob.u);r=rec(m,c,s,{'problem':prob.name});rows.append(r);print('general',r,flush=True)
pd.DataFrame(rows).to_csv(ROOT/'data'/'general_hj_benchmarks_embedded_qt.csv',index=False)
# smooth eikonal p
prob=ManufacturedEikonal();rows=[]
for p in DEG:
 m=FastFactoredEikonalPminus1(3,p,prob,ZeroFactor(),bounds=((0,1),(0,1)));c,s=solve_qt(m,prob.u);r=rec(m,c,s);rows.append(r);print('smoothp',r,flush=True)
pd.DataFrame(rows).to_csv(ROOT/'data'/'smooth_p_odd_embedded_qt.csv',index=False)
# smooth h p5
rows=[]
for n in (2,3,4,5,6,7):
 m=FastFactoredEikonalPminus1(n,5,prob,ZeroFactor(),bounds=((0,1),(0,1)));c,s=solve_qt(m,prob.u);r=rec(m,c,s);rows.append(r);print('smoothh',r,flush=True)
pd.DataFrame(rows).to_csv(ROOT/'data'/'smooth_h_p5_embedded_qt.csv',index=False)
# polynomial
prob2=PolynomialEikonal();rows=[]
for p in DEG:
 m=FastFactoredEikonalPminus1(2,p,prob2,ZeroFactor(),bounds=((0,1),(0,1)));c,s=solve_qt(m,prob2.u);r=rec(m,c,s);rows.append(r);print('poly',r,flush=True)
pd.DataFrame(rows).to_csv(ROOT/'data'/'polynomial_recovery_odd_embedded_qt.csv',index=False)
# focal
prob3=FocalCaustic();fac=RadialFactor((0,0),-1.0);rows=[]
for p in DEG:
 m=FastFactoredEikonalPminus1(4,p,prob3,fac,bounds=((-1,1),(-1,1)));target=lambda x,pr=prob3,fa=fac:pr.u(x)-fa.u(x);c,s=solve_qt(m,target);r=rec(m,c,s);rows.append(r);print('focal',r,flush=True)
pd.DataFrame(rows).to_csv(ROOT/'data'/'focal_caustic_p_sweep_embedded_qt.csv',index=False)
