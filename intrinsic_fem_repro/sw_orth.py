import numpy as np,scipy.sparse.linalg as sla,scipy.linalg as la,time
from sw_forms import seifert_weber,SparseEntityComplex,local_hodge_mass
r,level=4,0;t=time.time();cells,pairings,metrics=seifert_weber(level);C=SparseEntityComplex(len(cells),r,pairings)
M1=C.assemble_mass(1,[local_hodge_mass(C.spaces[1],g,q=r+4) for g in metrics]);M2=C.assemble_mass(2,[local_hodge_mass(C.spaces[2],g,q=r+4) for g in metrics])
A=(C.D[1].T@M2@C.D[1]).tocsc();lam,U=sla.eigsh(A,M=M1.tocsc(),k=12,sigma=3.0,which='LM',tol=1e-12);o=np.argsort(lam);lam=lam[o];U=U[:,o]
D0=C.D[0].toarray();M1d=M1.toarray();G=D0.T@M1d@D0
for i in range(6):
    u=U[:,i];a=la.lstsq(G,D0.T@M1d@u)[0];e=D0@a
    res=np.linalg.norm(A@u-lam[i]*(M1@u))/np.linalg.norm(lam[i]*(M1@u))
    print(f'{lam[i]:.6f}  exact-part fraction {np.sqrt(e@M1d@e)/np.sqrt(u@M1d@u):.2e}  relative residual {res:.2e}')
# metric comparison constants for this mesh
X=np.array([[.25,.25,.25],[.05,.05,.05],[.85,.05,.05],[.05,.85,.05],[.05,.05,.85]]);ev=np.array([np.linalg.eigvalsh(g(x)) for g in metrics for x in X])
print('SW 60 cells: eigenvalues of G_K in',ev.min().round(4),ev.max().round(4),'ratio',(ev.max()/ev.min()).round(2),'max pointwise anisotropy',(ev[:,2]/ev[:,0]).max().round(2),'[%ds]'%(time.time()-t))
