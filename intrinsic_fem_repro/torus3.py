import sys,itertools,time,numpy as np
from scipy.sparse.linalg import eigsh
import bb3d
N,p,nm=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]);h=1/N;PERMS=list(itertools.permutations(range(3)))
vidx=lambda i,j,k:((i%N)*N+(j%N))*N+(k%N)
cells=[];orig=[];Es=[]
for i,j,k in itertools.product(range(N),repeat=3):
    for pm in PERMS:
        v=[np.array([i,j,k])]
        for ax in pm:
            e=np.zeros(3,int);e[ax]=1;v.append(v[-1]+e)
        cells.append([vidx(*x) for x in v]);orig.append(v[0]*h);Es.append(np.stack([(v[m]-v[0])*h for m in (1,2,3)],1))
cells=np.array(cells);orig=np.array(orig);Es=np.array(Es)
phi=lambda x:0.12*np.prod(np.sin(2*np.pi*x),-1)+0.08*np.cos(2*np.pi*(x[...,0]+x[...,1]))
def mf(idx,X):
    x=orig[idx][:,None,:]+np.einsum('cij,qj->cqi',Es[idx],X);EtE=np.einsum('cki,ckj->cij',Es[idx],Es[idx])
    return np.exp(2*phi(x))[...,None,None]*EtE[:,None]
gd,ndof,keys=bb3d.dofs_by_vertex_ids(cells,p);t=time.time();K,M,vol=bb3d.assemble(gd,ndof,p,p+5,mf)
print('dofs',ndof,'volume',vol,time.time()-t,flush=True)
vals,vecs=eigsh(K.tocsc(),M=M.tocsc(),k=nm,sigma=-1.,which='LM');o=np.argsort(vals);vals=vals[o];vecs=vecs[:,o]
print(np.round(vals,3));np.savez_compressed(f'run/torus3_{N}_{p}.npz',vals=vals,vecs=vecs,gd=gd)
