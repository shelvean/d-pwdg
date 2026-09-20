"""Seifert-Weber dodecahedral space: geometry and pairings from hypforms.py, ordinary Bernstein
degree p in the affine chart y = V b, x = y/r, metric G = E^T [Q/r^2 + (Qy)(Qy)^T/r^4] E."""
import sys,time,numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh
sys.path.insert(0,'.')
import hypforms as H, bb3d
Qm=np.diag([-1.,1,1,1])
def cells_and_gens(level):
    a,rho,U,Fd,hv,hf,faces=H.geometry();O=np.array([1.,0,0,0]);cells=[]
    for k,idx in enumerate(faces):
        for i in range(5):cells.append(np.stack([O,hf[k],hv[idx[i]],hv[idx[(i+1)%5]]],1))
    for _ in range(level):cells=H.refine(cells)
    return np.array(cells),[np.eye(4)]+[H.pairing(f,a) for f in Fd],(a,rho,U,Fd,hv,hf,faces)
def dofmap(cells,gens,p):
    A=np.array(bb3d.inds3(p),float)/p;pts=np.concatenate([(V@A.T).T for V in cells]);par=np.arange(len(pts));tr=cKDTree(pts)
    def find(i):
        while par[i]!=i:par[i]=par[par[i]];i=par[i]
        return i
    for g in gens:
        for i,nb in enumerate(tr.query_ball_point(pts@g.T,1e-8)):
            for j in nb:
                x,y=find(i),find(j)
                if x!=y:par[y]=x
    _,gid=np.unique([find(i) for i in range(len(pts))],return_inverse=True);return gid.reshape(len(cells),-1),gid.max()+1
if __name__=='__main__':
    p,level,nm=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]);t=time.time()
    cells,gens,_=cells_and_gens(level);gd,ndof=dofmap(cells,gens,p)
    E=np.stack([cells[:,:,j]-cells[:,:,0] for j in (1,2,3)],-1)                 # (nc,4,3)
    def mf(idx,X):
        y=cells[idx,:,0][:,None,:]+np.einsum('cki,qi->cqk',E[idx],X);Qy=y*np.diag(Qm);r2=-(y*Qy).sum(-1)
        P=Qm[None,None]/r2[...,None,None]+Qy[...,:,None]*Qy[...,None,:]/(r2**2)[...,None,None]
        return np.einsum('cki,cqkl,clj->cqij',E[idx],P,E[idx])
    K,M,vol=bb3d.assemble(gd,ndof,p,p+6,mf,batch=16);print('cells',len(cells),'dofs',ndof,'volume',vol,'asm',time.time()-t,flush=True)
    vals,vecs=eigsh(K.tocsc(),M=M.tocsc(),k=nm,sigma=-1.,which='LM');o=np.argsort(vals);vals=vals[o];vecs=vecs[:,o]
    print(np.round(vals,4));np.savez_compressed(f'run/sw_{p}_{level}.npz',vals=vals,vecs=vecs,gd=gd)
