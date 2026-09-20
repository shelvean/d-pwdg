import sys,time,json,numpy as np
sys.path.insert(0,'.')
import trefoil_hp as T
Nu,Nv,p,q=map(int,sys.argv[1:5])
t=time.time()
res,data,(vals,vecs)=T.spectrum_run(Nu,Nv,p,481,q,save_vectors=True)
print(Nu,Nv,p,'dofs',res['dofs'],'time',time.time()-t)
for i in [1,2,10,40,80,120,240,320,400,480]: print(i,vals[i])
print('res',res['relative_residuals'],res['zero_mode_abs_residual'])
np.savez_compressed(f'run/spec_{Nu}_{Nv}_{p}.npz',vals=vals,vecs=vecs,gd=data['gd'])
