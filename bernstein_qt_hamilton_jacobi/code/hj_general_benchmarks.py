from __future__ import annotations
import time, math, csv
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.optimize import least_squares

from eikonal_c0_pminus1 import (
    multiindices2, bernstein_values, bernstein_gradients,
    triangle_quadrature, structured_tri_mesh, tri_geometry, build_c0_assembly,
)

Array=np.ndarray

class AnisotropicQuadraticHJ:
    """Smooth stationary HJ benchmark H(x,p)=1/2 p^T A(x)p-f(x)."""
    def u(self,x:Array)->Array:
        X=x[:,0];Y=x[:,1]
        return X+0.30*Y+0.055*np.sin(np.pi*X)*np.sin(np.pi*Y)
    def grad(self,x:Array)->Array:
        X=x[:,0];Y=x[:,1]; a=0.055
        return np.column_stack([
            1+a*np.pi*np.cos(np.pi*X)*np.sin(np.pi*Y),
            0.30+a*np.pi*np.sin(np.pi*X)*np.cos(np.pi*Y)])
    def Avec(self,x:Array,p:Array)->Array:
        X=x[:,0];Y=x[:,1]
        A11=1.10+0.18*X
        A22=0.85+0.12*Y
        A12=0.10+0.04*np.sin(np.pi*X)*np.sin(np.pi*Y)
        return np.column_stack([A11*p[:,0]+A12*p[:,1], A12*p[:,0]+A22*p[:,1]])
    def f(self,x:Array)->Array:
        g=self.grad(x); Ag=self.Avec(x,g); return 0.5*np.sum(g*Ag,axis=1)
    def H(self,x:Array,p:Array)->Array:
        Ap=self.Avec(x,p); return 0.5*np.sum(p*Ap,axis=1)-self.f(x)
    def Hp(self,x:Array,p:Array)->Array:
        return self.Avec(x,p)
    name='anisotropic quadratic'

class QuarticHJ:
    """Genuinely nonquadratic convex benchmark H=1/4(px^4+py^4)+mu/2 |p|^2-f(x)."""
    mu=0.35
    def u(self,x:Array)->Array:
        X=x[:,0];Y=x[:,1]
        return X+0.22*Y+0.045*np.sin(np.pi*X)*np.sin(np.pi*Y)
    def grad(self,x:Array)->Array:
        X=x[:,0];Y=x[:,1]; a=0.045
        return np.column_stack([
            1+a*np.pi*np.cos(np.pi*X)*np.sin(np.pi*Y),
            0.22+a*np.pi*np.sin(np.pi*X)*np.cos(np.pi*Y)])
    def phi(self,p:Array)->Array:
        return 0.25*(p[:,0]**4+p[:,1]**4)+0.5*self.mu*np.sum(p*p,axis=1)
    def f(self,x:Array)->Array:
        return self.phi(self.grad(x))
    def H(self,x:Array,p:Array)->Array:
        return self.phi(p)-self.f(x)
    def Hp(self,x:Array,p:Array)->Array:
        return np.column_stack([p[:,0]**3+self.mu*p[:,0], p[:,1]**3+self.mu*p[:,1]])
    name='quartic convex'

class C0HamiltonJacobiPminus1:
    def __init__(self,n:int,p:int,problem,quad_order=None):
        self.n=n;self.p=p;self.problem=problem
        self.verts,self.tris=structured_tri_mesh(n)
        self.asm=build_c0_assembly(self.verts,self.tris,p)
        self.inds=multiindices2(p);self.Nd=len(self.inds)
        self.r=p-1;self.test_inds=multiindices2(self.r);self.Nr=len(self.test_inds)
        qorder=quad_order or max(10,2*p+5)
        ref,wref=triangle_quadrature(qorder)
        lam=np.column_stack([1-ref[:,0]-ref[:,1],ref[:,0],ref[:,1]])
        self.B=bernstein_values(p,lam);self.Q=bernstein_values(self.r,lam)
        self.elements=[]
        for tri in self.tris:
            V=self.verts[tri];_,gl=tri_geometry(V)
            J=np.column_stack((V[1]-V[0],V[2]-V[0]));det=abs(np.linalg.det(J))
            pts=V[0]+ref@J.T;G=bernstein_gradients(p,lam,gl);W=wref*det
            self.elements.append((pts,G,W))
        tol=1e-13; fixed=[]; vals=[]
        for gid,x in enumerate(self.asm.dof_coords):
            if abs(x[0])<tol or abs(x[0]-1)<tol or abs(x[1])<tol or abs(x[1]-1)<tol:
                fixed.append(gid);vals.append(float(problem.u(np.asarray([x]))[0]))
        self.fixed=np.asarray(fixed,int);self.fixed_vals=np.asarray(vals,float)
        allidx=np.arange(len(self.asm.keys));mask=np.ones(len(allidx),bool);mask[self.fixed]=False
        self.free=allidx[mask];self.free_pos={int(g):i for i,g in enumerate(self.free)}

    def initial_coefficients(self):
        x=self.asm.dof_coords
        # common affine characteristic background; exact boundary values override it
        c=x[:,0]+0.26*x[:,1]
        c[self.fixed]=self.fixed_vals
        return c
    def unpack(self,z):
        c=self._base.copy();c[self.free]=z;return c
    def residual_and_jac(self,z,want_jac=True):
        c=self.unpack(z);R=np.zeros(len(self.tris)*self.Nr)
        Jm=np.zeros((len(R),len(self.free))) if want_jac else None
        for K in range(len(self.tris)):
            gids=self.asm.l2g[K];ck=c[gids];pts,G,W=self.elements[K]
            gh=np.einsum('qjd,j->qd',G,ck,optimize=True)
            F=self.problem.H(pts,gh);rows=slice(K*self.Nr,(K+1)*self.Nr)
            R[rows]=self.Q.T@(W*F)
            if want_jac:
                a=self.problem.Hp(pts,gh)
                adv=np.einsum('qd,qjd->qj',a,G,optimize=True)
                Jloc=self.Q.T@(W[:,None]*adv)
                for j,g in enumerate(gids):
                    pos=self.free_pos.get(int(g))
                    if pos is not None: Jm[rows,pos]+=Jloc[:,j]
        return R,Jm
    def solve(self,max_nfev=120):
        self._base=self.initial_coefficients();z0=self._base[self.free].copy()
        fun=lambda z:self.residual_and_jac(z,False)[0]
        jac=lambda z:self.residual_and_jac(z,True)[1]
        t=time.perf_counter()
        sol=least_squares(fun,z0,jac=jac,method='trf',x_scale='jac',max_nfev=max_nfev,
                          xtol=1e-12,ftol=1e-12,gtol=1e-12)
        return self.unpack(sol.x),sol,time.perf_counter()-t
    def errors(self,c,order=None):
        q=order or max(12,2*self.p+6)
        ref,wref=triangle_quadrature(q);lam=np.column_stack([1-ref[:,0]-ref[:,1],ref[:,0],ref[:,1]])
        B=bernstein_values(self.p,lam)
        l2=0.;l2u=0.;h1=0.;h1u=0.
        for K,tri in enumerate(self.tris):
            V=self.verts[tri];_,gl=tri_geometry(V)
            J=np.column_stack((V[1]-V[0],V[2]-V[0]));det=abs(np.linalg.det(J));pts=V[0]+ref@J.T;W=wref*det
            G=bernstein_gradients(self.p,lam,gl);ck=c[self.asm.l2g[K]]
            uh=B@ck;gh=np.einsum('qjd,j->qd',G,ck,optimize=True)
            ue=self.problem.u(pts);ge=self.problem.grad(pts)
            l2+=np.sum(W*(uh-ue)**2);l2u+=np.sum(W*ue**2)
            h1+=np.sum(W*np.sum((gh-ge)**2,axis=1));h1u+=np.sum(W*np.sum(ge**2,axis=1))
        return math.sqrt(l2/l2u),math.sqrt(h1/h1u)
    def row_scaled_sigma_min(self,c):
        self._base=c.copy();R,J=self.residual_and_jac(c[self.free],True)
        if J.shape[1]==0:return np.nan
        rown=np.linalg.norm(J,axis=1);rown[rown==0]=1
        coln=np.linalg.norm(J/rown[:,None],axis=0);coln[coln==0]=1
        s=la.svdvals((J/rown[:,None])/coln[None,:])
        return float(s[-1]) if len(s) else np.nan


def run(out):
    import pandas as pd
    from optimized_hj import FastHamiltonJacobiPminus1, elevate_global
    rows=[]
    for prob in (AnisotropicQuadraticHJ(),QuarticHJ()):
        oldm=oldc=None
        for p in (3,5,7,9,11,13):
            S=FastHamiltonJacobiPminus1(3,p,prob)
            init=elevate_global(oldm,oldc,S) if oldm is not None else None
            c,sol,sec=S.solve(initial_full=init,max_nfev=40)
            e2,e1=S.errors(c); R=S.residual(c[S.free])
            rows.append(dict(problem=prob.name,n=3,ntri=18,p=p,dofs=len(S.asm.keys),free_dofs=len(S.free),
                relL2=e2,relH1=e1,residual_rms=float(la.norm(R)/math.sqrt(len(R))),
                nfev=int(sol.nfev),seconds=sec,success=bool(sol.success),message=sol.message))
            print(rows[-1],flush=True); oldm,oldc=S,c
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)

if __name__=='__main__':
    run(Path(__file__).resolve().parents[1]/'data'/'general_hj_benchmarks.csv')
