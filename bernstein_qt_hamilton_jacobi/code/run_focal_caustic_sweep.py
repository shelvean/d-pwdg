from pathlib import Path
import numpy as np, pandas as pd
from optimized_hj import FastFactoredEikonalPminus1, elevate_global
from high_frequency_benchmarks import RadialFactor

ROOT=Path(__file__).resolve().parents[1]
DEGREES=(3,5,7,9,11,13)

class FocalCaustic:
    def __init__(self, center=(0.0,0.0), amp=0.035): self.center=np.asarray(center,float); self.amp=amp
    def u(self,x):
        z=x-self.center; r=np.linalg.norm(z,axis=1); X=x[:,0];Y=x[:,1]
        return -r+self.amp*np.sin(np.pi*X)*np.sin(np.pi*Y)
    def grad(self,x):
        z=x-self.center;r=np.linalg.norm(z,axis=1);g=np.zeros_like(z);m=r>1e-13;g[m]=-z[m]/r[m,None]
        X=x[:,0];Y=x[:,1];a=self.amp
        g[:,0]+=a*np.pi*np.cos(np.pi*X)*np.sin(np.pi*Y);g[:,1]+=a*np.pi*np.sin(np.pi*X)*np.cos(np.pi*Y);return g
    def n2(self,x):g=self.grad(x);return np.sum(g*g,axis=1)

prob=FocalCaustic(); fac=RadialFactor((0,0),-1.0); rows=[]; oldm=oldc=None
for p in DEGREES:
    m=FastFactoredEikonalPminus1(4,p,prob,fac,bounds=((-1,1),(-1,1)))
    init=elevate_global(oldm,oldc,m) if oldm is not None else None
    c,sol,sec=m.solve(max_nfev=35,initial_full=init,gtol=1e-14,rtol=2e-12,lsmr_tol=3e-12)
    row=m.summary(c,sol,sec); row['message']=sol.message; rows.append(row); print(row,flush=True)
    oldm,oldc=m,c
pd.DataFrame(rows).to_csv(ROOT/'data'/'focal_caustic_p_sweep.csv',index=False)
