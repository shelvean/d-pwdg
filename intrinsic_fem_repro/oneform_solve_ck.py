"""Schur-complement CG for the mixed one-form problem with a fill-reducing symmetric ordering and
checkpointing of the iterate, so that a run can continue across time-limited calls."""
import sys,os,time,pickle,numpy as np,scipy.sparse as sp,scipy.sparse.linalg as spla
import oneform_high_order as O
N,r,budget=int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]);tag=f'results/stage_N{N}_r{r}';T0=time.perf_counter()
cells=O.periodic_freudenthal_mesh(N);pairs=O.periodic_tetrahedral_facet_pairings(cells,N);C=O.SparseEntityComplex(len(cells),r,pairs,reference_basis='bernstein')
(M0,M1,M2),F=pickle.load(open(tag+'_MF.pkl','rb'));D0,D1=C.D[:2]
A=(D1.T@M2@D1+M1).tocsc();B=(M1@D0).tocsc();Bt=B.T.tocsr()
f0=spla.splu(M0.tocsc(),permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.0);fA=spla.splu(A,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.0)
print('factorized',round(time.perf_counter()-T0),'s',flush=True)
mv=lambda x:A@x+B@f0.solve(Bt@x);n=A.shape[0];ck=tag+'_ck.npz'
x=np.load(ck)['x'] if os.path.exists(ck) else np.zeros(n);it0=int(np.load(ck)['it']) if os.path.exists(ck) else 0
# preconditioned CG written out, so that it can stop on a time budget
rv=F-mv(x);z=fA.solve(rv);p=z.copy();rz=rv@z;nF=np.linalg.norm(F);it=it0;rtol=2e-11
while np.linalg.norm(rv)>rtol*nF and time.perf_counter()-T0<budget:
    Ap=mv(p);a=rz/(p@Ap);x+=a*p;rv-=a*Ap;z=fA.solve(rv);rz2=rv@z;p=z+(rz2/rz)*p;rz=rz2;it+=1
rel=np.linalg.norm(F-mv(x))/nF
if rel>rtol*1.5:
    np.savez(ck,x=x,it=it);print('checkpoint at iteration',it,'relative residual %.2e'%rel,round(time.perf_counter()-T0),'s')
else:
    sig=f0.solve(Bt@x);res=float(np.linalg.norm(np.r_[-M0@sig+Bt@x,B@sig+A@x-F])/max(1.,nF))
    np.savez(tag+'_sol.npz',sig=sig,al=x,res=res,it=it);print('converged: iterations',it,'residual %.2e'%res,round(time.perf_counter()-T0),'s')
