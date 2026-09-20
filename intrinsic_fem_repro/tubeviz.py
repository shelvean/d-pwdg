import sys,numpy as np
sys.path.insert(0,'.');import trefoil_hp as T
import bb3d
TAU=2*np.pi;a=0.35
class Tube:
    def __init__(s,f,Ns,nr,p):
        d=np.load(f);s.gd=d['gd'];s.coords=d['coords'];s.pts=d['pts'];s.tri=d['tri'];s.Ns=Ns;s.p=p;s.nt=len(s.tri)
        s.rin=a*np.cos(np.pi/(6*nr))
        P=s.pts[s.tri];s.T0=P[:,0];s.Tinv=np.linalg.inv(np.stack([P[:,1]-P[:,0],P[:,2]-P[:,0]],-1))
    def ev(s,c,S,U,Vv):
        sh=S.shape;S=S.ravel()%TAU;U=U.ravel();Vv=Vv.ravel();ds=TAU/s.Ns
        k=np.minimum((S/ds).astype(int),s.Ns-1)
        l=np.einsum('tij,ntj->nti',s.Tinv,np.stack([U,Vv],-1)[:,None,:]-s.T0[None])
        m=np.minimum(np.minimum(l[...,0],l[...,1]),1-l[...,0]-l[...,1]);ti=m.argmax(1);ok=m.max(1)>-1e-9
        x=np.stack([S,U,Vv],-1);best=np.full(len(S),-1e9);lam=np.zeros((len(S),4));cell=np.zeros(len(S),int)
        for j in range(3):
            ce=(k*s.nt+ti)*3+j;A=s.coords[ce];M=np.stack([A[:,1]-A[:,0],A[:,2]-A[:,0],A[:,3]-A[:,0]],-1)
            l3=np.linalg.solve(M,(x-A[:,0])[...,None])[...,0];l4=np.c_[1-l3.sum(1),l3];mm=l4.min(1)
            up=mm>best;best[up]=mm[up];lam[up]=l4[up];cell[up]=ce[up]
        B=bb3d.bern(s.p,lam,grad=False);val=(B*c[s.gd[cell]]).sum(1);val[~ok]=np.nan
        return val.reshape(sh)
def frame(t):
    c0,cp,cpp,_=T.curve_derivatives(t);tt=cp/np.linalg.norm(cp,axis=-1,keepdims=True)
    nr=cpp-(cpp*tt).sum(-1,keepdims=True)*tt;n=nr/np.linalg.norm(nr,axis=-1,keepdims=True)
    return c0,tt,n,np.cross(tt,n)
