import numpy as np,bb3d
from pds import V,one,adj,tets,aligned,reps,qmul
p=14;D=np.load(f'run/pds_{p}.npz');vals,vecs,gd=D['vals'],D['vecs'],D['gd']
AT=np.array([a[1] for a in aligned]);ORB=np.array([a[0] for a in aligned])
TINV=np.linalg.inv(np.transpose(V[AT],(0,2,1)))            # columns = aligned vertices
def ev(C,x,chunk=4000):
    """C (ndof,m) coefficient columns, x (n,4) points on S^3 -> (n,m)."""
    out=np.empty((len(x),C.shape[1]))
    for s in range(0,len(x),chunk):
        xx=x[s:s+chunk];lam=np.einsum('tij,nj->nti',TINV,xx);t=lam.min(2).argmax(1)
        l=lam[np.arange(len(xx)),t];l=np.clip(l,0,None);l/=l.sum(1,keepdims=True);B=bb3d.bern(p,l,grad=False)
        for o in range(5):
            m=ORB[t]==o
            if m.any():out[s:s+chunk][m]=B[m]@C[gd[o]]
    return out
def conj(g,x):
    gb=g*np.array([1,-1,-1,-1.]);return qmul(qmul(g[None,:],x),gb[None,:])
def level(lev):
    sel=np.where(np.abs(vals-lev)<0.01*lev)[0];return vecs[:,sel],vals[sel]
def invariant(C,elems,npts=1200,seed=0):
    """Coefficient combinations of columns of C invariant under conjugation by elems."""
    rng=np.random.default_rng(seed);x=rng.normal(size=(npts,4));x/=np.linalg.norm(x,axis=1,keepdims=True)
    Phi=ev(C,x);A=np.zeros((C.shape[1],)*2)
    for g in elems:A+=np.linalg.lstsq(Phi,ev(C,conj(g,x)),rcond=None)[0]
    A/=len(elems);A=(A+A.T)/2;w,U=np.linalg.eigh(A);keep=w>0.5
    return C@U[:,keep],w
