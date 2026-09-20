import numpy as np,bb3d
from sw import cells_and_gens,H,Qm
p,level=6,1;D=np.load(f'run/sw_{p}_{level}.npz');vals,vecs,gd=D['vals'],D['vecs'],D['gd']
cells,gens,(a_in,rho,U,Fd,hv,hf,faces)=cells_and_gens(level);CINV=np.linalg.inv(cells)
def ev(C,y,chunk=1500):
    out=np.empty((len(y),C.shape[1]))
    for s in range(0,len(y),chunk):
        yy=y[s:s+chunk];lam=np.einsum('cij,nj->nci',CINV,yy);lam/=lam.sum(2,keepdims=True);t=lam.min(2).argmax(1)
        l=np.clip(lam[np.arange(len(yy)),t],0,None);l/=l.sum(1,keepdims=True);B=bb3d.bern(p,l,grad=False)
        out[s:s+chunk]=np.einsum('na,nam->nm',B,C[gd[t]])
    return out
def rot(axis,ang):
    f=axis/np.linalg.norm(axis);Kx=np.array([[0,-f[2],f[1]],[f[2],0,-f[0]],[-f[1],f[0],0]]);R=np.eye(4)
    R[1:,1:]=np.eye(3)+np.sin(ang)*Kx+(1-np.cos(ang))*Kx@Kx;return R
def level_of(lam):
    sel=np.where(np.abs(vals-lam)<2e-3*lam)[0];return vecs[:,sel],vals[sel].mean()
def inv_U(C,Rs,npts=900):
    rng=np.random.default_rng(0);c=rng.integers(len(cells),size=npts);b=rng.dirichlet(np.ones(4),size=npts)
    y=np.einsum('nij,nj->ni',cells[c],b);Phi=ev(C,y);A=np.zeros((C.shape[1],)*2)
    for R in Rs:A+=np.linalg.lstsq(Phi,ev(C,y@R.T),rcond=None)[0]
    A/=len(Rs);w,Uu=np.linalg.eigh((A+A.T)/2);return Uu[:,w>0.5],np.sort(w)
AX5=Fd[0];AX3=U[0];C5=[rot(AX5,2*np.pi*j/5) for j in range(5)];C3=[rot(AX3,2*np.pi*j/3) for j in range(3)]
if __name__=='__main__':
    for lam in (9.5701,15.3637,19.328,19.4929,32.8616,34.3539,45.7429):
        C,v=level_of(lam);u5,w5=inv_U(C,C5);u3,w3=inv_U(C,C3)
        print(f'{v:9.4f} dim {C.shape[1]}  five-fold invariants {u5.shape[1]}  three-fold invariants {u3.shape[1]}  projector spectrum {np.round(w5,3)}')
