from pathlib import Path
import pandas as pd
from optimized_hj import FastFactoredEikonalPminus1,elevate_global
from high_frequency_benchmarks import ZeroFactor
from eikonal_c0_pminus1 import PolynomialEikonal,ManufacturedEikonal,C0EikonalPminus1
ROOT=Path(__file__).resolve().parents[1]; DEG=(3,5,7,9,11,13)
# exact polynomial recovery
rows=[];oldm=oldc=None;prob=PolynomialEikonal()
for p in DEG:
    m=FastFactoredEikonalPminus1(2,p,prob,ZeroFactor(),bounds=((0,1),(0,1)));init=elevate_global(oldm,oldc,m) if oldm is not None else None;c,s,t=m.solve(initial_full=init);rows.append(m.summary(c,s,t));oldm,oldc=m,c
pd.DataFrame(rows).to_csv(ROOT/'data'/'polynomial_recovery_odd.csv',index=False)
# smooth degree sweep
rows=[];oldm=oldc=None;prob=ManufacturedEikonal()
for p in DEG:
    m=FastFactoredEikonalPminus1(3,p,prob,ZeroFactor(),bounds=((0,1),(0,1)));init=elevate_global(oldm,oldc,m) if oldm is not None else None;c,s,t=m.solve(initial_full=init);rows.append(m.summary(c,s,t));oldm,oldc=m,c
pd.DataFrame(rows).to_csv(ROOT/'data'/'smooth_p_odd.csv',index=False)
# smooth six-level h sweep at p=5
rows=[]
for n in (2,3,4,5,6,7):
    m=FastFactoredEikonalPminus1(n,5,prob,ZeroFactor(),bounds=((0,1),(0,1)));c,s,t=m.solve();rows.append(m.summary(c,s,t))
pd.DataFrame(rows).to_csv(ROOT/'data'/'smooth_h_p5.csv',index=False)
# inflow-only validation retains the original boundary-mode implementation
m=C0EikonalPminus1(3,5,prob,boundary='inflow');c,s,t=m.solve(max_nfev=120);pd.DataFrame([m.summary(c,s,t)]).to_csv(ROOT/'data'/'smooth_inflow_p5.csv',index=False)
