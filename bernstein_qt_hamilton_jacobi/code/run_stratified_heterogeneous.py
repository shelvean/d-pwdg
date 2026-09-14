from pathlib import Path
import numpy as np, pandas as pd
from optimized_hj import FastFactoredEikonalPminus1
from high_frequency_benchmarks import ZeroFactor
from heterogeneous_stratified import StratifiedEikonal
ROOT=Path(__file__).resolve().parents[1]
problem=StratifiedEikonal(alpha=.8); rows=[]
for n in (2,3,4,5,6,7):
    m=FastFactoredEikonalPminus1(n,5,problem,ZeroFactor(),bounds=((0,1),(0,1)))
    c,sol,sec=m.solve(max_nfev=40,gtol=1e-13,rtol=2e-11,lsmr_tol=1e-10)
    row=m.summary(c,sol,sec);rows.append(row);print(row,flush=True)
    if n==4:np.savez(ROOT/'data'/'stratified_n4_p5_solution.npz',c=c,verts=m.verts,tris=m.tris,l2g=m.asm.l2g,p=m.p)
pd.DataFrame(rows).to_csv(ROOT/'data'/'stratified_p5_h_sweep.csv',index=False)
