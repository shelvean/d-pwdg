"""Poincare dodecahedral space S^3/Gamma: five geodesic tetrahedra of the 600-cell quotient,
radial chart, C0 Bernstein degree p, DOFs identified by Gamma-orbits of domain points."""
import sys,itertools,time,numpy as np,scipy.linalg as la
import bb3d
phi=(1+5**.5)/2
def qmul(a,b):
    a0,a1,a2,a3=[a[...,i] for i in range(4)];b0,b1,b2,b3=[b[...,i] for i in range(4)]
    return np.stack([a0*b0-a1*b1-a2*b2-a3*b3,a0*b1+a1*b0+a2*b3-a3*b2,a0*b2-a1*b3+a2*b0+a3*b1,a0*b3+a1*b2-a2*b1+a3*b0],-1)
def group():
    V=[]
    for i in range(4):
        for s in (1,-1):
            e=[0,0,0,0];e[i]=s;V.append(e)
    for s in itertools.product((.5,-.5),repeat=4):V.append(list(s))
    even=[p for p in itertools.permutations(range(4)) if sum(1 for i in range(4) for j in range(i) if p[j]>p[i])%2==0]
    base=[phi/2,.5,1/(2*phi),0]
    for p in even:
        for s in itertools.product((1,-1),repeat=3):
            v=[0]*4;sg=iter(s)
            for src,dst in enumerate(p):v[dst]=base[src]*(next(sg) if base[src]!=0 else 1)
            V.append(v)
    V=np.unique(np.round(np.array(V),12),axis=0);assert len(V)==120
    return V
V=group();one=int(np.argmin(np.linalg.norm(V-[1,0,0,0],axis=1)))
Gm=V@V.T;adj=np.abs(Gm-phi/2)<1e-9;assert adj.sum()==120*12
tets=[]
for i in range(120):
    ni=[j for j in np.where(adj[i])[0] if j>i]
    for j,k,l in itertools.combinations(ni,3):
        if adj[j,k] and adj[j,l] and adj[k,l]:tets.append((i,j,k,l))
assert len(tets)==600
def vid(P):   # indices of points P (n,4) among V
    return np.argmin(np.linalg.norm(P[:,None,:]-V[None],axis=2),axis=1)
LM=np.array([vid(qmul(V[g][None,:],V)) for g in range(120)])        # LM[g,i]=index of g*v_i
tetset={frozenset(t):n for n,t in enumerate(tets)};seen={};reps=[]
aligned=[None]*600                                                   # (orbit, vertex tuple aligned with rep order)
for t in tets:
    if frozenset(t) in seen:continue
    reps.append(t)
    for g in range(120):
        img=tuple(int(LM[g,i]) for i in t);f=frozenset(img);seen[f]=1;aligned[tetset[f]]=(len(reps)-1,img)
assert len(reps)==5 and all(a is not None for a in aligned)
def orbit_key(y):
    im=np.round(qmul(V,y[None,:]),8)+0.0
    return tuple(im[np.lexsort(im.T[::-1])[0]])
def build(p):
    I=bb3d.inds3(p);keys={};gd=np.empty((5,len(I)),int)
    for c,t in enumerate(reps):
        for a,al in enumerate(I):
            k=orbit_key(sum(al[j]/p*V[t[j]] for j in range(4)))
            if k not in keys:keys[k]=len(keys)
            gd[c,a]=keys[k]
    return gd,len(keys)
def metric_fn(idx,X):
    out=[]
    for c in idx:
        t=reps[c];E=np.stack([V[t[j]]-V[t[0]] for j in (1,2,3)],1);y=V[t[0]]+X@E.T;r2=(y*y).sum(1)
        P=np.eye(4)[None]/r2[:,None,None]-y[:,:,None]*y[:,None,:]/(r2**2)[:,None,None]
        out.append(np.einsum('ki,qkl,lj->qij',E,P,E))
    return np.array(out)
if __name__=='__main__':
    p=int(sys.argv[1]);t0=time.time();gd,ndof=build(p)
    K,M,vol=bb3d.assemble(gd,ndof,p,p+6,metric_fn,batch=1)
    print('p',p,'dofs',ndof,'volume error',vol-2*np.pi**2/120,'asm',time.time()-t0,flush=True)
    vals,vecs=la.eigh(K.toarray(),M.toarray());n=int(sys.argv[2])
    ex=np.concatenate([[k*(k+2)]*m for k,m in ((0,1),(12,13),(20,21),(24,25),(30,31),(32,33),(36,37))])
    for lev in (168,440,624,960,1088,1368):
        sel=np.abs(vals-lev)<0.02*lev
        print(lev,'count',sel.sum(),'range',vals[sel].min() if sel.any() else None,vals[sel].max() if sel.any() else None)
    np.savez_compressed(f'run/pds_{p}.npz',vals=vals[:n],vecs=vecs[:,:n],gd=gd)
