"""r on the refined Seifert-Weber mesh in restartable stages: local mass blocks are computed in
chunks of cells and saved; the last stage assembles and solves."""
import sys,time,pickle,numpy as np,scipy.sparse.linalg as sla
from sw_forms import seifert_weber,SparseEntityComplex,local_hodge_mass
r,level,stage=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3];t=time.perf_counter()
cells,pairings,metrics=seifert_weber(level);C=SparseEntityComplex(len(cells),r,pairings);q=r+4
if stage!='solve':
    a,b=map(int,stage.split(':'))
    out={k:[local_hodge_mass(C.spaces[k],metrics[i],q=q) for i in range(a,b)] for k in (1,2)}
    pickle.dump(out,open(f'run/swblk_r{r}_L{level}_{a}_{b}.pkl','wb'));print('cells',a,b,'done',round(time.perf_counter()-t),'s')
else:
    import glob
    files=sorted(glob.glob(f'run/swblk_r{r}_L{level}_*.pkl'),key=lambda f:int(f.split('_')[-2]));loc={1:[],2:[]}
    for f in files:
        d=pickle.load(open(f,'rb'));loc[1]+=d[1];loc[2]+=d[2]
    assert len(loc[1])==len(cells)
    M1=C.assemble_mass(1,loc[1]);M2=C.assemble_mass(2,loc[2]);A=(C.D[1].T@M2@C.D[1]).tocsc()
    print('dims',C.dimensions,'assembled',round(time.perf_counter()-t),'s',flush=True)
    lam=np.sort(sla.eigsh(A,M=M1.tocsc(),k=int(sys.argv[4]),sigma=float(sys.argv[5]),which='LM',tol=1e-10,return_eigenvectors=False))
    print(np.array2string(lam,precision=6,max_line_width=150));print('seconds',round(time.perf_counter()-t))
    np.save(f'run/swlam_r{r}_L{level}.npy',lam)
