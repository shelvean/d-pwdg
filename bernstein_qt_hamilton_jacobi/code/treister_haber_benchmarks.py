from __future__ import annotations

import argparse, csv, math, time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.optimize import least_squares

from eikonal_c0_pminus1 import (
    multiindices2, bernstein_values, bernstein_gradients,
    triangle_quadrature, tri_geometry, build_c0_assembly,
)

Array=np.ndarray


def rectangular_tri_mesh(nx:int, ny:int, bounds=((0.0,4.0),(0.0,8.0))):
    (xmin,xmax),(ymin,ymax)=bounds
    xs=np.linspace(xmin,xmax,nx+1); ys=np.linspace(ymin,ymax,ny+1)
    verts=np.array([(x,y) for y in ys for x in xs],float)
    def vid(i,j): return j*(nx+1)+i
    tris=[]
    for j in range(ny):
        for i in range(nx):
            v00,v10,v01,v11=vid(i,j),vid(i+1,j),vid(i,j+1),vid(i+1,j+1)
            if (i+j)%2==0:
                tris += [(v00,v10,v11),(v00,v11,v01)]
            else:
                tris += [(v00,v10,v01),(v10,v11,v01)]
    return verts,np.asarray(tris,int)


class THCase1:
    name='TH1_squared_slowness_gradient'
    def __init__(self,a=-0.4,s0=2.0,source=(0.0,4.0)):
        self.a=float(a); self.s0=float(s0); self.source=np.asarray(source,float)
    def n2(self,x):
        return self.s0**2 + 2*self.a*(x[:,0]-self.source[0])
    def tau(self,x):
        z=x-self.source; r2=np.sum(z*z,axis=1)
        S2=self.s0**2 + self.a*z[:,0]
        disc=np.maximum(S2*S2 - (self.a**2)*r2,0.0)
        sig2=2*r2/np.maximum(S2+np.sqrt(disc),1e-30)
        sig=np.sqrt(np.maximum(sig2,0.0))
        return S2*sig - (self.a**2)*sig**3/6.0
    def psi_source(self): return self.s0


class THCase2:
    name='TH2_velocity_gradient'
    def __init__(self,a=1.0,s0=2.0,source=(0.0,4.0)):
        self.a=float(a); self.s0=float(s0); self.source=np.asarray(source,float)
    def slowness(self,x):
        return 1.0/(1.0/self.s0 + self.a*(x[:,0]-self.source[0]))
    def n2(self,x):
        s=self.slowness(x); return s*s
    def tau(self,x):
        z=x-self.source; r2=np.sum(z*z,axis=1); s=self.slowness(x)
        arg=1.0+0.5*self.s0*(self.a**2)*s*r2
        return np.arccosh(np.maximum(arg,1.0))/self.a
    def psi_source(self): return self.s0


class THCase3:
    name='TH3_gaussian_factor'
    def __init__(self,source=(1.0,2.0),x1=(4.0/3.0,2.0),sigma=(0.1,0.4)):
        self.source=np.asarray(source,float); self.x1=np.asarray(x1,float)
        self.Sigma=np.diag(np.asarray(sigma,float))
    def psi(self,x):
        z=x-self.x1
        q=np.einsum('ni,ij,nj->n',z,self.Sigma,z)
        return 0.5*np.exp(-q)+0.5
    def gradpsi(self,x):
        z=x-self.x1; ps=self.psi(x)
        # psi-1/2 = 1/2 exp(-q), grad = -2 Sigma z (psi-1/2)
        return -2.0*((ps-0.5)[:,None])*(z@self.Sigma.T)
    def tau(self,x):
        z=x-self.source; r=np.linalg.norm(z,axis=1)
        return r*self.psi(x)
    def n2(self,x):
        z=x-self.source; r=np.linalg.norm(z,axis=1)
        gr=np.zeros_like(z); m=r>1e-14; gr[m]=z[m]/r[m,None]
        ps=self.psi(x); gp=self.gradpsi(x)
        gt=ps[:,None]*gr + r[:,None]*gp
        # limiting slowness at source is psi(source)
        bad=~m
        if np.any(bad):
            # direction-independent limiting magnitude
            out=np.sum(gt*gt,axis=1); out[bad]=self.psi(x[bad])**2; return out
        return np.sum(gt*gt,axis=1)
    def psi_source(self): return float(self.psi(self.source[None,:])[0])


class MultiplicativeFactoredBB:
    """Solve tau = r psi_h by global C0 degree-p Bernstein p-1 residual moments.

    Only psi(x_s)=lim tau/r is fixed.  This mirrors the point-source information
    used in factored causal solvers; no exact outer-boundary trace is supplied.
    """
    def __init__(self,nx:int,ny:int,p:int,problem,quad_order=None,bounds=((0.0,4.0),(0.0,8.0))):
        self.nx,self.ny,self.p=int(nx),int(ny),int(p); self.problem=problem; self.bounds=bounds
        self.verts,self.tris=rectangular_tri_mesh(nx,ny,bounds)
        self.asm=build_c0_assembly(self.verts,self.tris,p)
        self.inds=multiindices2(p); self.Nd=len(self.inds)
        self.rdeg=p-1; self.Nr=len(multiindices2(self.rdeg))
        self.qorder=quad_order or max(8,2*p+3)
        ref,wref=triangle_quadrature(self.qorder); self.ref,self.wref=ref,wref
        lam=np.column_stack([1-ref[:,0]-ref[:,1],ref[:,0],ref[:,1]])
        self.B=bernstein_values(p,lam); self.Q=bernstein_values(self.rdeg,lam)
        self.elements=[]
        for K,tri in enumerate(self.tris):
            V=self.verts[tri]; _,gl=tri_geometry(V)
            J=np.column_stack((V[1]-V[0],V[2]-V[0])); det=abs(np.linalg.det(J))
            pts=V[0]+ref@J.T; G=bernstein_gradients(p,lam,gl); W=wref*det
            z=pts-self.problem.source; rr=np.linalg.norm(z,axis=1)
            gr=np.zeros_like(z); m=rr>1e-14; gr[m]=z[m]/rr[m,None]
            self.elements.append((pts,G,W,rr,gr))
        # Source must be a degree-p domain point. On the chosen meshes it is a vertex.
        d=np.linalg.norm(self.asm.dof_coords-self.problem.source[None,:],axis=1)
        self.source_gid=int(np.argmin(d))
        if d[self.source_gid]>1e-11:
            raise ValueError(f'source is not a global domain point (distance {d[self.source_gid]})')
        self.fixed=np.asarray([self.source_gid],int)
        self.fixed_vals=np.asarray([self.problem.psi_source()],float)
        allidx=np.arange(len(self.asm.keys)); mask=np.ones(len(allidx),bool); mask[self.fixed]=False
        self.free=allidx[mask]; self.free_pos={int(g):i for i,g in enumerate(self.free)}
        self._base=self.initial_coefficients()

    def initial_coefficients(self):
        # Constant source-slowness factor is the standard smooth first guess.
        c=np.full(len(self.asm.keys),self.problem.psi_source(),float)
        c[self.fixed]=self.fixed_vals
        return c

    def unpack(self,z):
        c=self._base.copy(); c[self.free]=z; c[self.fixed]=self.fixed_vals; return c

    def residual_and_jac(self,z,want_jac=True):
        c=self.unpack(z); R=np.zeros(len(self.tris)*self.Nr,float)
        Jmat=np.zeros((len(R),len(self.free)),float) if want_jac else None
        for K,tri in enumerate(self.tris):
            gids=self.asm.l2g[K]; ck=c[gids]
            pts,G,W,rr,gr=self.elements[K]
            psi=self.B@ck; gpsi=np.einsum('qjd,j->qd',G,ck,optimize=True)
            gtau=psi[:,None]*gr + rr[:,None]*gpsi
            F=0.5*(np.sum(gtau*gtau,axis=1)-self.problem.n2(pts))
            rows=slice(K*self.Nr,(K+1)*self.Nr)
            R[rows]=self.Q.T@(W*F)
            if want_jac:
                # d grad(tau)/d c_j = B_j grad r + r grad B_j
                dgtau=self.B[:,:,None]*gr[:,None,:] + rr[:,None,None]*G
                adv=np.einsum('qd,qjd->qj',gtau,dgtau,optimize=True)
                Jloc=self.Q.T@(W[:,None]*adv)
                for j,g in enumerate(gids):
                    pos=self.free_pos.get(int(g))
                    if pos is not None: Jmat[rows,pos]+=Jloc[:,j]
        return R,Jmat

    def solve(self,max_nfev=80,verbose=0):
        self._base=self.initial_coefficients(); z0=self._base[self.free].copy()
        fun=lambda z:self.residual_and_jac(z,False)[0]
        jac=lambda z:self.residual_and_jac(z,True)[1]
        t=time.perf_counter()
        sol=least_squares(fun,z0,jac=jac,method='trf',x_scale='jac',max_nfev=max_nfev,
                          xtol=1e-12,ftol=1e-12,gtol=1e-12,verbose=verbose)
        return self.unpack(sol.x),sol,time.perf_counter()-t

    def errors(self,c,order=None):
        order=order or max(12,2*self.p+5)
        ref,wref=triangle_quadrature(order)
        lam=np.column_stack([1-ref[:,0]-ref[:,1],ref[:,0],ref[:,1]])
        B=bernstein_values(self.p,lam)
        int_e2=0.0; area=0.0; linf=0.0; int_u2=0.0
        for K,tri in enumerate(self.tris):
            V=self.verts[tri]; J=np.column_stack((V[1]-V[0],V[2]-V[0])); det=abs(np.linalg.det(J))
            pts=V[0]+ref@J.T; W=wref*det
            z=pts-self.problem.source; rr=np.linalg.norm(z,axis=1)
            psi=B@c[self.asm.l2g[K]]; uh=rr*psi; ue=self.problem.tau(pts); e=uh-ue
            int_e2 += np.sum(W*e*e); int_u2 += np.sum(W*ue*ue); area += np.sum(W)
            linf=max(linf,float(np.max(np.abs(e))))
        rms=math.sqrt(int_e2/area)
        rel=math.sqrt(int_e2/max(int_u2,1e-30))
        return linf,rms,rel

    def summary(self,c,sol,seconds):
        linf,rms,rel=self.errors(c); self._base=c.copy(); R,_=self.residual_and_jac(c[self.free],False)
        hx=(self.bounds[0][1]-self.bounds[0][0])/self.nx; hy=(self.bounds[1][1]-self.bounds[1][0])/self.ny
        return dict(case=self.problem.name,nx=self.nx,ny=self.ny,h=max(hx,hy),ntri=len(self.tris),p=self.p,
                    c0_dofs=len(self.asm.keys),free_dofs=len(self.free),residual_rows=len(R),
                    absLinf=linf,meanL2=rms,relL2=rel,residual_rms=float(la.norm(R)/math.sqrt(len(R))),
                    success=bool(sol.success),nfev=int(sol.nfev),seconds=float(seconds))


def problem_from_name(name):
    return {'case1':THCase1,'case2':THCase2,'case3':THCase3}[name]()


def run_sweep(cases=('case1','case2','case3'), meshes=((4,8),(8,16)), ps=(2,3,4,5), max_nfev=80):
    rows=[]
    for case in cases:
        for nx,ny in meshes:
            for p in ps:
                prob=problem_from_name(case); m=MultiplicativeFactoredBB(nx,ny,p,prob)
                c,sol,sec=m.solve(max_nfev=max_nfev)
                s=m.summary(c,sol,sec); rows.append(s)
                print(s,flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cases',nargs='+',default=['case1','case2','case3'])
    ap.add_argument('--mesh',nargs=2,type=int,default=[4,8])
    ap.add_argument('--p',nargs='+',type=int,default=[2,3,4,5])
    ap.add_argument('--max-nfev',type=int,default=80)
    ap.add_argument('--out',default='treister_haber_bb.csv')
    a=ap.parse_args()
    rows=run_sweep(tuple(a.cases),(tuple(a.mesh),),tuple(a.p),a.max_nfev)
    if rows:
        with open(a.out,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

if __name__=='__main__': main()
