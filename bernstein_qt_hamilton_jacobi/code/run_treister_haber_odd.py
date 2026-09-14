from pathlib import Path
import pandas as pd
from optimized_hj import FastMultiplicativeFactoredBB,elevate_global
from treister_haber_benchmarks import THCase1
ROOT=Path(__file__).resolve().parents[1]
settings=((3,1e-9),(5,1e-9),(7,1e-10),(9,1e-11),(11,1e-12),(13,1e-12))
prob=THCase1();rows=[];oldm=oldc=None
for p,gtol in settings:
    m=FastMultiplicativeFactoredBB(4,8,p,prob);init=elevate_global(oldm,oldc,m) if oldm is not None else None
    c,sol,sec=m.solve(initial_full=init,max_nfev=15,gtol=gtol,rtol=1e-11,lsmr_tol=3e-10)
    row=m.summary(c,sol,sec);row['gtol']=gtol;rows.append(row);print(row,flush=True);oldm,oldc=m,c
pd.DataFrame(rows).to_csv(ROOT/'data'/'treister_haber_case1_odd_h1.csv',index=False)
