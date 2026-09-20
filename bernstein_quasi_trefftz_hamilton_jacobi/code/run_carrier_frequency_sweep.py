from pathlib import Path
import math, numpy as np, pandas as pd
from optimized_hj import FastFactoredEikonalPminus1
from high_frequency_benchmarks import ZeroFactor
from eikonal_c0_pminus1 import ManufacturedEikonal,triangle_quadrature,bernstein_values
ROOT=Path(__file__).resolve().parents[1]
prob=ManufacturedEikonal();m=FastFactoredEikonalPminus1(3,5,prob,ZeroFactor(),bounds=((0,1),(0,1)))
c,sol,sec=m.solve(max_nfev=35,gtol=1e-14,rtol=2e-12,lsmr_tol=3e-12)
ref,wref=triangle_quadrature(18);lam=np.column_stack([1-ref[:,0]-ref[:,1],ref[:,0],ref[:,1]]);B=bernstein_values(5,lam)
rows=[]
for om in (50,100,250,500,1000,2000,5000,10000,20000):
    E=N=phaseE=phaseN=0.
    for K,tri in enumerate(m.tris):
        V=m.verts[tri];J=np.column_stack((V[1]-V[0],V[2]-V[0]));det=abs(np.linalg.det(J));pts=V[0]+ref@J.T;W=wref*det
        uh=B@c[m.asm.l2g[K]];ue=prob.u(pts);wh=np.exp(1j*om*uh);we=np.exp(1j*om*ue)
        E+=np.sum(W*np.abs(wh-we)**2);N+=np.sum(W*np.abs(we)**2);phaseE+=np.sum(W*(uh-ue)**2);phaseN+=np.sum(W*ue**2)
    rp=math.sqrt(phaseE/phaseN);rows.append(dict(omega=om,relative_field_L2=math.sqrt(E/N),relative_phase_L2=rp,omega_times_rel_phase=om*rp))
df=pd.DataFrame(rows);print(df.to_string(index=False));df.to_csv(ROOT/'data'/'carrier_frequency_sweep.csv',index=False)
