import math, time, json, os, sys, numpy as np
from poisson_surface import manufactured, solve_poisson, errors, geometries
from sphsplines import icosphere, meshsize_sphere
ut,grad_ut,lap_ut=manufactured('exp(x1) * x2 + x3**3 * x1')
cases=[(2,0,[1,2,3,4]),(4,0,[1,2,3,4]),(6,1,[1,2,3,4])]
out='poisson_rows_ext.jsonl'
done=set()
if os.path.exists(out):
    for line in open(out):
        r=json.loads(line); done.add((r['geo'],r['d'],r['lev']))
geos=geometries(); cache={}
for name,wgt in geos.items():
    for d,r,levels in cases:
        for lev in levels:
            if (name,d,lev) in done: continue
            v,t=icosphere(lev); h=meshsize_sphere(v,t); t0=time.time()
            c,info=solve_poisson(v,t,d,r,wgt,lap_ut,cache=cache)
            e0,e1=errors(v,t,d,c,wgt,ut,grad_ut)
            row=dict(geo=name,d=d,r=r,lev=lev,h=h,dim=info['dim'],e0=e0,e1=e1,resid=info['resid'])
            open(out,'a').write(json.dumps(row)+'\n')
            print(name,d,r,lev,f"{e0:.3e} {e1:.3e} {time.time()-t0:.1f}s",flush=True)
