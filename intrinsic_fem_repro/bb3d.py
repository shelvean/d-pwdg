"""Minimal C0 Bernstein tetrahedral FEM with a prescribed metric (for figures)."""
import numpy as np, math, itertools, scipy.sparse as sp
from numpy.polynomial.legendre import leggauss

def duffy(q):
    a,w=leggauss(q);a=(a+1)/2;w=w/2
    t1,t2,t3=np.meshgrid(a,a,a,indexing='ij');w3=w[:,None,None]*w[None,:,None]*w[None,None,:]
    X=np.stack([t1,t2*(1-t1),t3*(1-t1)*(1-t2)],-1).reshape(-1,3)
    ww=(w3*(1-t1)**2*(1-t2)).ravel();assert abs(ww.sum()-1/6)<1e-13
    return X,ww
def inds3(p):
    return [(i,j,k,p-i-j-k) for i in range(p,-1,-1) for j in range(p-i,-1,-1) for k in range(p-i-j,-1,-1)]
def bern(p,lam,grad=True):
    """lam (n,4). Returns B (n,nloc) and dB/dlambda (n,nloc,4)."""
    I=np.array(inds3(p));n=len(lam)
    P=np.ones((4,p+1,n))
    for e in range(1,p+1):P[:,e,:]=P[:,e-1,:]*lam.T
    coef=np.array([math.factorial(p)/np.prod([math.factorial(a) for a in al]) for al in I])
    fac=[P[j][I[:,j]] for j in range(4)]                      # each (nloc,n)
    B=(coef[:,None]*fac[0]*fac[1]*fac[2]*fac[3]).T
    if not grad:return B
    dB=np.zeros((n,len(I),4))
    for j in range(4):
        d=I[:,j][:,None]*P[j][np.maximum(I[:,j]-1,0)]
        others=np.ones_like(d)
        for k in range(4):
            if k!=j:others=others*fac[k]
        dB[:,:,j]=(coef[:,None]*d*others).T
    return B,dB
def assemble(gd,ndof,p,q,metric_fn,batch=128):
    """metric_fn(cell_index_array, Xhat) -> G (c,nq,3,3) in reference coordinates."""
    X,w=duffy(q);lam=np.c_[1-X.sum(1),X];B,dBl=bern(p,lam)
    D=dBl[:,:,1:]-dBl[:,:,[0]];nloc=B.shape[1];Dm=D.transpose(1,0,2).reshape(nloc,-1)
    nc=len(gd);Ks=[];Ms=[];vol=0.
    for s in range(0,nc,batch):
        idx=np.arange(s,min(nc,s+batch));G=metric_fn(idx,X)
        rho=np.sqrt(np.linalg.det(G));C=(w*rho)[...,None,None]*np.linalg.inv(G)
        Tm=np.einsum('qai,cqij->caqj',D,C,optimize=True).reshape(len(idx),nloc,-1)
        Ks.append(Tm@Dm.T);Ms.append((B.T[None]*(w*rho)[:,None,:])@B);vol+=(w*rho).sum()
    K=np.concatenate(Ks);M=np.concatenate(Ms)
    r=np.repeat(gd,nloc,axis=1).ravel();c=np.tile(gd,(1,nloc)).ravel()
    return (sp.coo_matrix((K.ravel(),(r,c)),shape=(ndof,ndof)).tocsr(),
            sp.coo_matrix((M.ravel(),(r,c)),shape=(ndof,ndof)).tocsr(),vol)
def dofs_by_vertex_ids(cells,p):
    I=inds3(p);keys={};gd=np.empty((len(cells),len(I)),int);supp={}
    for c,conn in enumerate(cells):
        for a,al in enumerate(I):
            s=[j for j in range(4) if al[j]>0]
            if len(s)==4:key=('t',c,al)
            else:
                o=sorted(s,key=lambda j:conn[j]);key=(tuple(int(conn[j]) for j in o),tuple(al[j] for j in o))
            if key not in keys:keys[key]=len(keys)
            gd[c,a]=keys[key]
    return gd,len(keys),keys
def boundary_dofs(cells,keys):
    from collections import Counter
    cnt=Counter()
    for conn in cells:
        for f in itertools.combinations(sorted(int(v) for v in conn),3):cnt[f]+=1
    sub=set()
    for f,n in cnt.items():
        if n==1:
            for r in (1,2,3):
                for s in itertools.combinations(f,r):sub.add(s)
    return np.array(sorted(i for k,i in keys.items() if k[0]!='t' and k[0] in sub),int)
