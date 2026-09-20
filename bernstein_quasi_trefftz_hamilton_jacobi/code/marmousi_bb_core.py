from __future__ import annotations
import sys, math, time
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import lsmr
from scipy.interpolate import RegularGridInterpolator

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))
from eikonal_c0_pminus1 import multiindices2, bernstein_values, bernstein_gradients, triangle_quadrature, tri_geometry, build_c0_assembly
from high_frequency_benchmarks import bernstein_1d_matrix, RadialFactor
from optimized_hj import elevate_local
from marmousi_fast_sweeping_reference import fast_sweep

def rect_mesh(nx,nz,bounds=((0.,9.2),(0.,3.0))):
    (xmin,xmax),(zmin,zmax)=bounds
    xs=np.linspace(xmin,xmax,nx+1); zs=np.linspace(zmin,zmax,nz+1)
    verts=np.array([(x,z) for z in zs for x in xs],float)
    def vid(i,j): return j*(nx+1)+i
    tris=[]
    for j in range(nz):
        for i in range(nx):
            v00,v10,v01,v11=vid(i,j),vid(i+1,j),vid(i,j+1),vid(i+1,j+1)
            tris += [(v00,v10,v11),(v00,v11,v01)]
    return verts,np.asarray(tris,int)

def boundary_edges(tris):
    d={}
    for tri in tris:
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            e=tuple(sorted((int(a),int(b))));d[e]=d.get(e,0)+1
    return [e for e,c in d.items() if c==1]

def elevate_global_rect(old, old_c, new):
    out=np.zeros(len(new.asm.keys)); cnt=np.zeros(len(new.asm.keys))
    for K in range(len(old.tris)):
        el=elevate_local(old_c[old.asm.l2g[K]],old.p,new.p)
        gids=new.asm.l2g[K]
        np.add.at(out,gids,el);np.add.at(cnt,gids,1)
    out/=np.maximum(cnt,1)
    out[new.fixed]=new.fixed_vals
    return out

class MarmProblem:
    def __init__(self, velocity, xvel, zvel, Tseed, xseed, zseed):
        self.velocity=velocity
        self.sinterp=RegularGridInterpolator((zvel,xvel),1.0/velocity,bounds_error=False,fill_value=None)
        self.tinterp=RegularGridInterpolator((zseed,xseed),Tseed,bounds_error=False,fill_value=None)
    def n2(self,pts):
        q=np.column_stack([pts[:,1],pts[:,0]])
        s=self.sinterp(q);return s*s
    def u(self,pts):
        q=np.column_stack([pts[:,1],pts[:,0]])
        return self.tinterp(q)

class RectFastFactored:
    def __init__(self,nx,nz,p,problem,factor,bounds=((0.,9.2),(0.,3.0)),qorder=None):
        self.nx=nx;self.nz=nz;self.p=p;self.problem=problem;self.factor=factor;self.bounds=bounds
        self.verts,self.tris=rect_mesh(nx,nz,bounds)
        self.asm=build_c0_assembly(self.verts,self.tris,p)
        self.Nd=len(multiindices2(p));self.r=p-1;self.Nr=len(multiindices2(self.r))
        self.qorder=qorder or max(8,min(16,p+3))
        ref,wref=triangle_quadrature(self.qorder)
        lam=np.column_stack([1-ref[:,0]-ref[:,1],ref[:,0],ref[:,1]])
        self.B=bernstein_values(p,lam);self.Q=bernstein_values(self.r,lam)
        E=len(self.tris);nq=len(ref)
        self.G=[];self.W=[];self.PTS=[]
        # list storage to reduce peak contiguous allocation hassles
        for tri in self.tris:
            V=self.verts[tri];_,gl=tri_geometry(V);J=np.column_stack((V[1]-V[0],V[2]-V[0]));det=abs(np.linalg.det(J))
            pts=V[0]+ref@J.T
            self.PTS.append(pts);self.G.append(bernstein_gradients(p,lam,gl));self.W.append(wref*det)
        self.PTS=np.asarray(self.PTS);self.G=np.asarray(self.G);self.W=np.asarray(self.W)
        self.GFAC=self.factor.grad(self.PTS.reshape(-1,2)).reshape(E,nq,2)
        self.N2=self.problem.n2(self.PTS.reshape(-1,2)).reshape(E,nq)
        self.fixed,self.fixed_vals=self._boundary_coefficients()
        allidx=np.arange(len(self.asm.keys));mask=np.ones(len(allidx),bool);mask[self.fixed]=False;self.free=allidx[mask]
        self.g2free=np.full(len(allidx),-1,int);self.g2free[self.free]=np.arange(len(self.free));self.local_free=self.g2free[self.asm.l2g]
        pe=[];pb=[];pj=[]
        for K in range(E):
            js=np.nonzero(self.local_free[K]>=0)[0]
            if not len(js): continue
            bb=np.repeat(np.arange(self.Nr),len(js));jj=np.tile(js,self.Nr)
            pe.append(np.full(len(bb),K,int));pb.append(bb);pj.append(jj)
        self.pe=np.concatenate(pe);self.pb=np.concatenate(pb);self.pj=np.concatenate(pj)
        self.prows=self.pe*self.Nr+self.pb;self.pcols=self.local_free[self.pe,self.pj]
        self._base=None;self._cz=None;self._R=None;self._gh=None;self._J=None
    def tau_seed(self,x): return self.problem.u(x)-self.factor.u(x)
    def _boundary_coefficients(self):
        key_to_gid={k:i for i,k in enumerate(self.asm.keys)}
        # factor removes the source singularity; anchor only the source value tau(xs)=0
        d=np.linalg.norm(self.verts-self.factor.source[None,:],axis=1)
        v=int(np.argmin(d))
        if d[v] > 1e-10:
            raise ValueError('source must be a mesh vertex')
        gid=key_to_gid.get(('v',v))
        return np.asarray([gid],int),np.asarray([0.0])
    def initial_coefficients(self):
        # Elementwise Bernstein interpolation of the causal seed, assembled C0 by averaging.
        inds=multiindices2(self.p)
        lam=np.asarray(inds,float)/self.p
        A=bernstein_values(self.p,lam)
        out=np.zeros(len(self.asm.keys));cnt=np.zeros(len(self.asm.keys))
        for K,tri in enumerate(self.tris):
            V=self.verts[tri]; pts=lam@V
            vals=self.tau_seed(pts)
            ck=la.solve(A,vals,check_finite=False)
            gids=self.asm.l2g[K];np.add.at(out,gids,ck);np.add.at(cnt,gids,1.0)
        out/=np.maximum(cnt,1.0);out[self.fixed]=self.fixed_vals
        return out
    def unpack(self,z):c=self._base.copy();c[self.free]=z;return c
    def state(self,z):
        if self._cz is not None and np.array_equal(z,self._cz): return self._R,self._gh
        c=self.unpack(z); C=c[self.asm.l2g]
        gt=np.einsum('eqjd,ej->eqd',self.G,C,optimize=True);gh=gt+self.GFAC
        F=.5*(np.einsum('eqd,eqd->eq',gh,gh,optimize=True)-self.N2)
        R=np.einsum('qb,eq,eq->eb',self.Q,self.W,F,optimize=True).ravel()
        self._cz=z.copy();self._R=R;self._gh=gh;self._J=None
        return R,gh
    def jac(self,z):
        R,gh=self.state(z)
        if self._J is not None:return self._J
        adv=np.einsum('eqd,eqjd->eqj',gh,self.G,optimize=True)
        Jloc=np.einsum('qb,eq,eqj->ebj',self.Q,self.W,adv,optimize=True)
        data=Jloc[self.pe,self.pb,self.pj]
        self._J=sp.csr_matrix((data,(self.prows,self.pcols)),shape=(len(R),len(self.free)))
        return self._J
    def solve(self,initial_full=None,maxit=30,verbose=0,lsmr_tol=1e-7):
        self._base=self.initial_coefficients() if initial_full is None else np.asarray(initial_full).copy();self._base[self.fixed]=self.fixed_vals
        z=self._base[self.free].copy();self._cz=None;t0=time.perf_counter();R,_=self.state(z);cost=.5*R@R;cost0=cost;nfev=1;njev=0
        for it in range(maxit):
            J=self.jac(z);njev+=1;g=np.asarray(J.T@R).ravel();col=np.sqrt(np.asarray(J.power(2).sum(axis=0)).ravel());col=np.maximum(col,1e-13)
            opt=np.linalg.norm(g/col,np.inf)
            if verbose:print('it',it,'cost',cost,'rms',np.linalg.norm(R)/math.sqrt(len(R)),'opt',opt)
            if opt<1e-8 or np.linalg.norm(R)<1e-9*max(1,np.sqrt(2*cost0)):break
            Js=J@sp.diags(1/col)
            y=lsmr(Js,-R,atol=lsmr_tol,btol=lsmr_tol,maxiter=80)[0]
            step=y/col
            alpha=1.;accepted=False
            for _ in range(10):
                zt=z+alpha*step;Rt,_=self.state(zt);nfev+=1;ct=.5*Rt@Rt
                if ct<cost:
                    z,R,cost=zt,Rt,ct;accepted=True;break
                alpha*=.5
            if not accepted:break
        dt=time.perf_counter()-t0;self._base[self.free]=z
        return self._base.copy(),SimpleNamespace(cost=cost,nfev=nfev,njev=njev),dt
    def eval_grid(self,c,xg,zg):
        X,Z=np.meshgrid(xg,zg); pts=np.column_stack([X.ravel(),Z.ravel()])
        (xmin,xmax),(zmin,zmax)=self.bounds;hx=(xmax-xmin)/self.nx;hz=(zmax-zmin)/self.nz
        ii=np.minimum(((pts[:,0]-xmin)/hx).astype(int),self.nx-1);jj=np.minimum(((pts[:,1]-zmin)/hz).astype(int),self.nz-1)
        rr=(pts[:,0]-(xmin+ii*hx))/hx; tt=(pts[:,1]-(zmin+jj*hz))/hz
        lower=tt<=rr
        K=2*(jj*self.nx+ii)+(~lower).astype(int)
        lam=np.empty((len(pts),3))
        lam[lower,0]=1-rr[lower];lam[lower,1]=rr[lower]-tt[lower];lam[lower,2]=tt[lower]
        up=~lower;lam[up,0]=1-tt[up];lam[up,1]=rr[up];lam[up,2]=tt[up]-rr[up]
        out=np.empty(len(pts))
        for kk in np.unique(K):
            m=K==kk;B=bernstein_values(self.p,lam[m]);out[m]=B@c[self.asm.l2g[kk]]
        out += self.factor.u(pts)
        return out.reshape(len(zg),len(xg))

def make_seed(vfine,x,z,nxc=72,nzc=24,src=(4.6,0.0)):
    xi=np.linspace(x[0],x[-1],nxc+1);zi=np.linspace(z[0],z[-1],nzc+1)
    vinterp=RegularGridInterpolator((z,x),vfine,bounds_error=False,fill_value=None)
    X,Z=np.meshgrid(xi,zi);vc=vinterp(np.column_stack([Z.ravel(),X.ravel()])).reshape(len(zi),len(xi))
    s=1/vc;hx=xi[1]-xi[0];hz=zi[1]-zi[0];sx=np.argmin(abs(xi-src[0]));sz=np.argmin(abs(zi-src[1]))
    T,ncy,mc=fast_sweep(s,hx,hz,sz,sx,100,1e-12)
    return xi,zi,vc,T,ncy

