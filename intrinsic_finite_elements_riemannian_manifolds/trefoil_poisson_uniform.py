import sys,json,numpy as np
sys.path.insert(0,'.');import trefoil_hp as T
res=[]
for p in (2,3,4,5,6):
    prev=None
    for Nu,Nv in ((16,4),(32,8),(64,16)):
        r,_,_=T.poisson_run(Nu,Nv,p,q=max(8,p+4),qe=max(10,p+6))
        if prev:r['oL2']=float(np.log(prev['L2']/r['L2'])/np.log(2));r['oH1']=float(np.log(prev['H1']/r['H1'])/np.log(2))
        prev=r;res.append(r);print(Nu,Nv,p,r['dofs'],'%.4e'%r['L2'],'%.2f'%r.get('oL2',0),'%.4e'%r['H1'],'%.2f'%r.get('oH1',0),flush=True)
json.dump(res,open('run/poisson_sweep_uniform.json','w'),indent=1,default=float)
