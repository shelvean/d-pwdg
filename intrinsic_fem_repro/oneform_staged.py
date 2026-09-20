"""Staged version of oneform_high_order.run: assemble | solve | errors, with files in between."""
import sys,time,json,pickle,numpy as np,scipy.sparse as sp
import oneform_high_order as O
N,r,stage=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3];q=max(6,r+4);eq=q+2;tag=f'results/stage_N{N}_r{r}';t=time.perf_counter()
B=np.eye(3);cells=O.periodic_freudenthal_mesh(N);pairs=O.periodic_tetrahedral_facet_pairings(cells,N)
C=O.SparseEntityComplex(len(cells),r,pairs,reference_basis='bernstein');print('complex',round(time.perf_counter()-t),'s',flush=True)
if stage=='assemble':
    M,F=O.local_and_rhs(C,cells,N,r,B,q);pickle.dump((M,F),open(tag+'_MF.pkl','wb'));print('assembled',round(time.perf_counter()-t),'s')
elif stage=='solve':
    M,F=pickle.load(open(tag+'_MF.pkl','rb'));sig,al,res,it=O.solve_mixed(M,F,C.D[0],C.D[1],method='schur')
    np.savez(tag+'_sol.npz',sig=sig,al=al,res=res,it=it);print('solved it',it,'res',res,round(time.perf_counter()-t),'s')
else:
    S=np.load(tag+'_sol.npz');err=O.errors(C,cells,N,r,B,S['al'],S['sig'],eq)
    out=dict(N=N,r=r,cells=len(cells),dofs_1=int(len(S['al'])),quad=q,error_quad=eq,iterations=int(S['it']),residual=float(S['res']),**err)
    open('results/new.jsonl','a').write(json.dumps(out)+'\n');print(json.dumps(out)[:400],round(time.perf_counter()-t),'s')
