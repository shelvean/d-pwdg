import sys, os, time
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.linalg as la
from scipy.optimize import least_squares
from scipy.stats import qmc
from pwdg import PWDG
from errors import l2_error as std_l2
from direction_adaptive_automatic import l2_error, graph_data, physical_skeleton_residual_vector
from wirtinger_pwdg_tests import ThreeWave, make_family_state, fams_all, parameter_dq, assemble_directional_derivative, solution_sensitivity, residual_vector_and_jac_column


def optimize_real_angles(n,k,seed=47,nq=14,max_nfev=120):
    ex=ThreeWave(k)
    m=3
    sob=qmc.Sobol(1,scramble=True,seed=seed).random_base2(2)[:m,0]
    x0=2*np.pi*sob
    fams=fams_all(n,m)
    cache={'x':None,'s':None,'fb':None,'r':None}
    counts={'state_assemblies':0,'jac_calls':0,'sensitivity_solves':0}
    hist=[]
    def unpack(x):
        z=np.zeros((m,2)); z[:,0]=np.asarray(x)%(2*np.pi); return z
    def state(x):
        if cache['x'] is None or not np.array_equal(np.asarray(x),cache['x']):
            z=unpack(x)
            s,fb=make_family_state(n,ex,z,fams,nq)
            r=physical_skeleton_residual_vector(s)
            cache.update(x=np.array(x,copy=True),s=s,fb=fb,r=r)
            counts['state_assemblies']+=1
        return unpack(x),cache['s'],cache['fb'],cache['r']
    def fun(x): return state(x)[3]
    def jac(x):
        counts['jac_calls']+=1
        z,s,fb,r=state(x)
        J=np.empty((len(r),m))
        for j in range(m):
            p=2*j # theta derivative
            dq=parameter_dq(s,fb,z,p)
            dA,dF=assemble_directional_derivative(s,dq)
            dc=solution_sensitivity(s,dA,dF)
            _,dr=residual_vector_and_jac_column(s,dq,dc)
            J[:,j]=dr
            counts['sensitivity_solves']+=1
        return J
    def cb(x,*args,**kwargs):
        z,s,fb,r=state(x)
        hist.append((len(hist),float(np.dot(r,r)),np.degrees(z[:,0]).copy()))
    t0=time.perf_counter()
    rr=least_squares(fun,x0,jac=jac,method='trf',x_scale='jac',ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=max_nfev,callback=cb)
    secs=time.perf_counter()-t0
    z=unpack(rr.x)
    s,fb=make_family_state(n,ex,z,fams,max(nq,24))
    # uniform baseline
    b=PWDG(n,ex,p=3,nq=24); A0,F0=b.assemble(); b.solve()
    # graph data automatic
    gd=graph_data(s.A)
    exact=np.array([17.,123.,251.])
    found=np.sort(np.degrees(z[:,0])%360)
    exactsort=np.sort(exact)
    # assignment by brute permutations for max/mean angle error
    import itertools
    def cdiff(a,b): return abs((a-b+180)%360-180)
    best=None
    for perm in itertools.permutations(found):
        errs=np.array([cdiff(perm[i],exactsort[i]) for i in range(3)])
        val=errs.max()
        if best is None or val<best[0]: best=(val,errs,np.array(perm))
    return dict(
        n=n, triangles=s.nt, k=k, ndof=s.ndof,
        uniform_error=std_l2(b,16), automatic_error=l2_error(s,16),
        accepted_iterations=len(hist), nfev=rr.nfev, njev=rr.njev,
        state_assemblies=counts['state_assemblies'], seconds=secs,
        raw_condition=gd['cond_raw'], graph_condition=gd['cond_gr'],
        max_angle_error_deg=float(best[0]), mean_angle_error_deg=float(best[1].mean()),
        start_deg=';'.join(f'{v:.6f}' for v in np.degrees(x0)),
        found_deg=';'.join(f'{v:.12f}' for v in np.degrees(z[:,0])%360),
        J=float(np.dot(physical_skeleton_residual_vector(s),physical_skeleton_residual_vector(s))),
        success=rr.success, message=rr.message
    ), hist

rows=[]; histories=[]
for n in [4,6]:
    for k in [4.,8.,12.]:
        print('RUN',n,k,flush=True)
        r,h=optimize_real_angles(n,k)
        rows.append(r)
        for it,J,a in h:
            histories.append(dict(n=n,k=k,iteration=it,J=J,angles_deg=';'.join(map(str,a))))
        print(r,flush=True)

OUT=Path(__file__).resolve().parent.parent/'generated'; OUT.mkdir(exist_ok=True)
out=str(OUT/'threewave_large_mesh_results.csv')
pd.DataFrame(rows).to_csv(out,index=False)
pd.DataFrame(histories).to_csv(OUT/'threewave_large_mesh_J_history.csv',index=False)
print(out)
