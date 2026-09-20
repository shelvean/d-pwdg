from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from optimized_hj import FastFactoredEikonalPminus1,elevate_global
from high_frequency_benchmarks import RadialS1,RadialS2,LinearSpeed,RadialFactor,ZeroFactor
from eikonal_c0_pminus1 import bernstein_values
ROOT=Path(__file__).resolve().parents[1];DEG=(3,5,7,9,11,13);rows=[]
configs=[
 ('Potter-Cameron s1',RadialS1(),lambda q:RadialFactor(q.source,1.0),((-1,1),(-1,1))),
 ('Potter-Cameron s2',RadialS2(),lambda q:ZeroFactor(),((-1,1),(-1,1))),
 ('linear-speed single source',LinearSpeed(source=(0,0),v=(.5,.25)),lambda q:RadialFactor(q.source,q.si),((0,1),(0,1))),]
for name,prob,ff,bounds in configs:
    oldm=oldc=None
    for p in DEG:
        m=FastFactoredEikonalPminus1(4,p,prob,ff(prob),bounds=bounds);init=elevate_global(oldm,oldc,m) if oldm is not None else None;c,s,t=m.solve(initial_full=init,max_nfev=45);r=m.summary(c,s,t);r['benchmark']=name;rows.append(r);print(r,flush=True);oldm,oldc=m,c
# two-source p=5
probs=[LinearSpeed(source=(0,0),v=(.5,.25)),LinearSpeed(source=(.8,0),v=(.5,.25))];branches=[]
for prob in probs:
    m=FastFactoredEikonalPminus1(4,5,prob,RadialFactor(prob.source,prob.si),bounds=((0,1),(0,1)));c,s,t=m.solve();branches.append((m,c,t))

def evaluate(m,c,pts):
    U=np.full(len(pts),np.nan)
    for K,tri in enumerate(m.tris):
        V=m.verts[tri];A=np.column_stack((V[1]-V[0],V[2]-V[0]));rs=np.linalg.solve(A,(pts-V[0]).T);l1,l2=rs[0],rs[1];l0=1-l1-l2;mask=(l0>=-1e-10)&(l1>=-1e-10)&(l2>=-1e-10)
        if mask.any():
            lam=np.column_stack([l0[mask],l1[mask],l2[mask]]);B=bernstein_values(m.p,lam);U[mask]=m.factor.u(pts[mask])+B@c[m.asm.l2g[K]]
    return U
xx=np.linspace(0,1,251);yy=np.linspace(0,1,251);X,Y=np.meshgrid(xx,yy);pts=np.column_stack([X.ravel(),Y.ravel()]);U1=evaluate(branches[0][0],branches[0][1],pts);U2=evaluate(branches[1][0],branches[1][1],pts);U=np.minimum(U1,U2);ex=np.minimum(probs[0].u(pts),probs[1].u(pts));er=U-ex
rows.append(dict(benchmark='linear-speed two source',p=5,n=4,relL2=float(np.linalg.norm(er)/np.linalg.norm(ex)),absLinf=float(np.max(np.abs(er))),seconds=branches[0][2]+branches[1][2],full_c0_dofs=branches[0][0].asm.dof_coords.shape[0],retained_local_dim=6))
pd.DataFrame(rows).to_csv(ROOT/'data'/'high_frequency_benchmark_results.csv',index=False);np.savez(ROOT/'data'/'linear_speed_two_source_p5_grid.npz',X=X,Y=Y,U=U.reshape(X.shape),exact=ex.reshape(X.shape))
