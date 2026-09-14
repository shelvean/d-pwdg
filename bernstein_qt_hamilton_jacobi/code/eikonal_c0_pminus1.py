from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from scipy.optimize import least_squares
from scipy.special import roots_legendre, gammaln

Array = np.ndarray


def multiindices2(p: int) -> List[Tuple[int, int, int]]:
    return [(a, b, p-a-b) for a in range(p+1) for b in range(p+1-a)]


def multinomial_coeff(alpha: Sequence[int]) -> float:
    p = sum(alpha)
    return float(np.exp(gammaln(p+1) - sum(gammaln(a+1) for a in alpha)))


def bernstein_values(p: int, lam: Array) -> Array:
    """Values of all degree-p triangle Bernstein polynomials.

    lam has shape (nq,3); result (nq,Nd).
    """
    inds = multiindices2(p)
    out = np.empty((lam.shape[0], len(inds)), float)
    for j, a in enumerate(inds):
        val = np.full(lam.shape[0], multinomial_coeff(a), float)
        for k, ak in enumerate(a):
            if ak:
                val *= lam[:, k] ** ak
        out[:, j] = val
    return out


def bernstein_gradients(p: int, lam: Array, grad_lam: Array) -> Array:
    """Physical gradients, shape (nq,Nd,2)."""
    inds = multiindices2(p)
    nq = lam.shape[0]
    G = np.zeros((nq, len(inds), 2), float)
    if p == 0:
        return G
    inds_m = multiindices2(p-1)
    idx_m = {a:i for i,a in enumerate(inds_m)}
    Bm = bernstein_values(p-1, lam)
    for j, a in enumerate(inds):
        for k in range(3):
            if a[k] > 0:
                b = list(a); b[k] -= 1
                G[:, j, :] += p * Bm[:, idx_m[tuple(b)], None] * grad_lam[k]
    return G


def triangle_quadrature(order: int) -> Tuple[Array, Array]:
    """Tensor Gauss rule on reference triangle (0,0),(1,0),(0,1)."""
    x, w = roots_legendre(order)
    r = 0.5*(x+1.0); wr = 0.5*w
    t = r.copy(); wt = wr.copy()
    rr=[]; ss=[]; ww=[]
    for i, ri in enumerate(r):
        for j, tj in enumerate(t):
            rr.append(ri)
            ss.append((1.0-ri)*tj)
            ww.append(wr[i]*wt[j]*(1.0-ri))
    pts = np.column_stack([rr,ss])
    return pts, np.asarray(ww)


def structured_tri_mesh(n: int) -> Tuple[Array, Array]:
    verts = np.array([(i/n, j/n) for j in range(n+1) for i in range(n+1)], float)
    def vid(i,j): return j*(n+1)+i
    tris=[]
    for j in range(n):
        for i in range(n):
            v00,v10,v01,v11=vid(i,j),vid(i+1,j),vid(i,j+1),vid(i+1,j+1)
            # alternate diagonal to avoid a directional artifact
            if (i+j)%2==0:
                tris += [(v00,v10,v11),(v00,v11,v01)]
            else:
                tris += [(v00,v10,v01),(v10,v11,v01)]
    return verts, np.asarray(tris, int)


def tri_geometry(V: Array) -> Tuple[float, Array]:
    J = np.column_stack((V[1]-V[0], V[2]-V[0]))
    det = np.linalg.det(J)
    area = abs(det)/2.0
    invJT = np.linalg.inv(J).T
    # lambda1=r, lambda2=s, lambda0=1-r-s
    g1 = invJT[:,0]
    g2 = invJT[:,1]
    g0 = -g1-g2
    return area, np.vstack([g0,g1,g2])


def canonical_edge_key(v0: int, v1: int, pos_from_v0: int, p: int) -> Tuple:
    if v0 < v1:
        return ('e', v0, v1, pos_from_v0)
    return ('e', v1, v0, p-pos_from_v0)


@dataclass
class C0Assembly:
    verts: Array
    tris: Array
    p: int
    l2g: Array
    dof_coords: Array
    keys: List[Tuple]


def build_c0_assembly(verts: Array, tris: Array, p: int) -> C0Assembly:
    inds = multiindices2(p)
    key_to_gid: Dict[Tuple,int] = {}
    keys: List[Tuple] = []
    coords: List[Tuple[float,float]] = []
    l2g = np.empty((len(tris), len(inds)), int)

    for K, tri in enumerate(tris):
        V = verts[tri]
        for j,a in enumerate(inds):
            nz = [i for i,x in enumerate(a) if x>0]
            if len(nz)==1:  # vertex
                loc=nz[0]; key=('v', int(tri[loc]))
            elif len(nz)==2 and 0 in a:  # edge
                zero=a.index(0); locs=[q for q in range(3) if q!=zero]
                u,v=locs
                # position measured from local u vertex toward v: alpha_v
                key=canonical_edge_key(int(tri[u]),int(tri[v]),a[v],p)
            elif len(nz)==2:  # can occur only with one zero, handled above
                raise RuntimeError('unexpected edge case')
            else:  # triangle interior
                key=('t', K, a[0],a[1],a[2])
            if key not in key_to_gid:
                gid=len(keys); key_to_gid[key]=gid; keys.append(key)
                x=(a[0]*V[0]+a[1]*V[1]+a[2]*V[2])/p if p>0 else V.mean(axis=0)
                coords.append(tuple(x))
            l2g[K,j]=key_to_gid[key]
    return C0Assembly(verts,tris,p,l2g,np.asarray(coords,float),keys)


class ManufacturedEikonal:
    """Smooth single-arrival manufactured solution with positive x-slowness direction."""
    def __init__(self, amp: float = 0.06): self.amp=amp
    def u(self, x: Array) -> Array:
        X=x[:,0]; Y=x[:,1]; a=self.amp
        return X + 0.35*Y + a*np.sin(np.pi*X)*np.sin(np.pi*Y)
    def grad(self, x: Array) -> Array:
        X=x[:,0]; Y=x[:,1]; a=self.amp
        ux=1 + a*np.pi*np.cos(np.pi*X)*np.sin(np.pi*Y)
        uy=0.35 + a*np.pi*np.sin(np.pi*X)*np.cos(np.pi*Y)
        return np.column_stack([ux,uy])
    def n2(self, x: Array) -> Array:
        g=self.grad(x); return np.sum(g*g,axis=1)
    def inflow_trace(self, y: Array) -> Array:
        # x=0 -> exact trace is affine, hence these are exact Bernstein B-coefficients
        return 0.35*y


class PolynomialEikonal:
    """Quadratic exact solution used for exact-recovery validation."""
    def __init__(self, eps: float = 0.08): self.eps=eps
    def u(self,x:Array)->Array:
        X=x[:,0];Y=x[:,1];return X+0.35*Y+self.eps*X*Y
    def grad(self,x:Array)->Array:
        X=x[:,0];Y=x[:,1];return np.column_stack([1+self.eps*Y,0.35+self.eps*X])
    def n2(self,x:Array)->Array:
        g=self.grad(x);return np.sum(g*g,axis=1)
    def inflow_trace(self,y:Array)->Array:return 0.35*y


class C0EikonalPminus1:
    def __init__(self,n:int,p:int,problem,quad_order:int|None=None,boundary:str='all'):
        if p<1: raise ValueError('p must be >=1')
        self.n=n;self.p=p;self.problem=problem;self.boundary=boundary
        self.verts,self.tris=structured_tri_mesh(n)
        self.asm=build_c0_assembly(self.verts,self.tris,p)
        self.inds=multiindices2(p);self.Nd=len(self.inds)
        self.r=p-1; self.test_inds=multiindices2(self.r); self.Nr=len(self.test_inds)
        self.qorder=quad_order or max(8,2*p+3)
        ref,wref=triangle_quadrature(self.qorder)
        self.ref=ref;self.wref=wref
        lam=np.column_stack([1-ref[:,0]-ref[:,1],ref[:,0],ref[:,1]])
        self.B=bernstein_values(p,lam)
        self.Q=bernstein_values(self.r,lam)
        self.elements=[]
        for K,tri in enumerate(self.tris):
            V=self.verts[tri]
            area,gl=tri_geometry(V)
            J=np.column_stack((V[1]-V[0],V[2]-V[0])); det=abs(np.linalg.det(J))
            pts=V[0]+ref@J.T
            G=bernstein_gradients(p,lam,gl)
            W=wref*det
            self.elements.append((pts,G,W))
        self.fixed,self.fixed_vals=self._boundary_dofs()
        allidx=np.arange(len(self.asm.keys)); mask=np.ones(len(allidx),bool);mask[self.fixed]=False
        self.free=allidx[mask]
        self.free_pos={int(g):i for i,g in enumerate(self.free)}

    def _boundary_dofs(self):
        fixed=[];vals=[]
        tol=1e-13
        for gid,x in enumerate(self.asm.dof_coords):
            on_inflow=abs(x[0])<tol
            on_boundary=on_inflow or abs(x[0]-1)<tol or abs(x[1])<tol or abs(x[1]-1)<tol
            take=on_boundary if self.boundary=='all' else on_inflow
            if take:
                fixed.append(gid)
                if self.boundary=='inflow':
                    vals.append(float(self.problem.inflow_trace(np.array([x[1]]))[0]))
                else:
                    vals.append(float(self.problem.u(np.asarray([x]))[0]))
        return np.asarray(fixed,int),np.asarray(vals,float)

    def initial_coefficients(self)->Array:
        # affine background u=x+0.35y is represented exactly by B-coefficients = values at domain points
        x=self.asm.dof_coords
        c=x[:,0]+0.35*x[:,1]
        c[self.fixed]=self.fixed_vals
        return c

    def unpack(self,z:Array)->Array:
        c=self._base.copy(); c[self.free]=z; return c

    def residual_and_jac(self,z:Array,want_jac=True):
        c=self.unpack(z)
        nrows=len(self.tris)*self.Nr
        R=np.zeros(nrows,float)
        Jmat=np.zeros((nrows,len(self.free)),float) if want_jac else None
        for K,tri in enumerate(self.tris):
            gids=self.asm.l2g[K]; ck=c[gids]
            pts,G,W=self.elements[K]
            gh=np.einsum('qjd,j->qd',G,ck,optimize=True)
            Fq=0.5*(np.sum(gh*gh,axis=1)-self.problem.n2(pts))
            rows=slice(K*self.Nr,(K+1)*self.Nr)
            R[rows]=self.Q.T@(W*Fq)
            if want_jac:
                # local Newton matrix: int q_beta (grad uh . grad B_j)
                adv=np.einsum('qd,qjd->qj',gh,G,optimize=True)
                Jloc=self.Q.T@(W[:,None]*adv)
                for j,g in enumerate(gids):
                    pos=self.free_pos.get(int(g))
                    if pos is not None:
                        Jmat[rows,pos]+=Jloc[:,j]
        return R,Jmat

    def solve(self,max_nfev:int=80,verbose:int=0,xtol=1e-12,ftol=1e-12,gtol=1e-12):
        self._base=self.initial_coefficients()
        z0=self._base[self.free].copy()
        def fun(z): return self.residual_and_jac(z,False)[0]
        def jac(z): return self.residual_and_jac(z,True)[1]
        t=time.perf_counter()
        sol=least_squares(fun,z0,jac=jac,method='trf',x_scale='jac',max_nfev=max_nfev,
                          xtol=xtol,ftol=ftol,gtol=gtol,verbose=verbose)
        elapsed=time.perf_counter()-t
        c=self.unpack(sol.x)
        return c,sol,elapsed

    def errors(self,c:Array,order:int|None=None):
        order=order or max(10,2*self.p+4)
        ref,wref=triangle_quadrature(order)
        lam=np.column_stack([1-ref[:,0]-ref[:,1],ref[:,0],ref[:,1]])
        B=bernstein_values(self.p,lam)
        L2=H1=Nu=Ng=0.0
        for K,tri in enumerate(self.tris):
            V=self.verts[tri]; area,gl=tri_geometry(V); J=np.column_stack((V[1]-V[0],V[2]-V[0]));det=abs(np.linalg.det(J))
            pts=V[0]+ref@J.T;W=wref*det;G=bernstein_gradients(self.p,lam,gl)
            ck=c[self.asm.l2g[K]]; uh=B@ck; gh=np.einsum('qjd,j->qd',G,ck,optimize=True)
            ue=self.problem.u(pts); ge=self.problem.grad(pts)
            L2+=np.sum(W*(uh-ue)**2);H1+=np.sum(W*np.sum((gh-ge)**2,axis=1));Nu+=np.sum(W*ue**2);Ng+=np.sum(W*np.sum(ge**2,axis=1))
        return math.sqrt(L2/Nu),math.sqrt(H1/Ng)

    def local_elimination_stats(self,c:Array):
        """Characteristic-aware p-1 deep-block Jacobian ranks/conditioning.

        For each triangle choose the facet whose barycentric normal has largest
        |grad u_h . grad lambda_i| at the centroid. Deep coefficients are those
        with alpha_i >= 1, i.e. lambda_i P_{p-1}.
        """
        stats=[]
        lamc=np.array([[1/3,1/3,1/3]],float)
        Bc=bernstein_values(self.p,lamc); Qc=None
        # reuse accurate quadrature for moment matrix
        lam=np.column_stack([1-self.ref[:,0]-self.ref[:,1],self.ref[:,0],self.ref[:,1]])
        for K,tri in enumerate(self.tris):
            gids=self.asm.l2g[K];ck=c[gids];pts,G,W=self.elements[K]
            gh=np.einsum('qjd,j->qd',G,ck,optimize=True)
            # centroid gradient from basis at centroid
            _,gl=tri_geometry(self.verts[tri]); Gcent=bernstein_gradients(self.p,lamc,gl)[0]
            gc=Gcent.T@ck
            trans=np.abs(gl@gc)
            facet=int(np.argmax(trans))
            deep=[j for j,a in enumerate(self.inds) if a[facet]>=1]
            assert len(deep)==self.Nr
            adv=np.einsum('qd,qjd->qj',gh,G,optimize=True)
            Jloc=self.Q.T@(W[:,None]*adv[:,deep])
            s=la.svdvals(Jloc)
            tol=max(Jloc.shape)*np.finfo(float).eps*(s[0] if len(s) else 1.0)
            rank=int(np.sum(s>tol)); cond=float(s[0]/s[-1]) if rank==len(s) and s[-1]>0 else float('inf')
            stats.append(dict(K=K,facet=facet,transversality=float(trans[facet]),rank=rank,size=len(deep),cond=cond,smin=float(s[-1])))
        return stats

    def summary(self,c,sol,elapsed):
        eL2,eH1=self.errors(c);R,_=self.residual_and_jac(c[self.free],False)
        stats=self.local_elimination_stats(c)
        return dict(n=self.n,ntri=len(self.tris),p=self.p,test_degree=self.p-1,
                    full_c0_dofs=len(self.asm.keys),fixed_boundary_dofs=len(self.fixed),boundary=self.boundary,free_dofs=len(self.free),
                    residual_rows=len(self.tris)*self.Nr,retained_local_dim=self.p+1,
                    relL2=eL2,relH1=eH1,residual_l2=float(la.norm(R)),residual_rms=float(la.norm(R)/math.sqrt(len(R))),
                    success=bool(sol.success),nfev=int(sol.nfev),njev=int(sol.njev) if sol.njev is not None else None,
                    cost=float(sol.cost),optimality=float(sol.optimality),seconds=elapsed,
                    local_blocks_full_rank=all(s['rank']==s['size'] for s in stats),
                    min_transversality=min(s['transversality'] for s in stats),
                    max_local_deep_cond=max(s['cond'] for s in stats))


def run_case(n=3,p=4,kind='smooth',amp=0.06,max_nfev=80,verbose=0,boundary='all'):
    prob=ManufacturedEikonal(amp) if kind=='smooth' else PolynomialEikonal()
    model=C0EikonalPminus1(n,p,prob,boundary=boundary)
    c,sol,t=model.solve(max_nfev=max_nfev,verbose=verbose)
    return model.summary(c,sol,t),c,model


def main():
    ap=argparse.ArgumentParser(description='Global C0 Bernstein p-1 quasi-Trefftz eikonal solver')
    ap.add_argument('--n',type=int,default=3)
    ap.add_argument('--p',type=int,default=4)
    ap.add_argument('--kind',choices=['smooth','polynomial'],default='smooth')
    ap.add_argument('--amp',type=float,default=0.06)
    ap.add_argument('--max-nfev',type=int,default=80)
    ap.add_argument('--boundary',choices=['all','inflow'],default='all')
    ap.add_argument('--verbose',type=int,default=0)
    ap.add_argument('--json',type=str,default='')
    a=ap.parse_args()
    s,c,m=run_case(a.n,a.p,a.kind,a.amp,a.max_nfev,a.verbose,a.boundary)
    print(json.dumps(s,indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(s,indent=2))

if __name__=='__main__': main()
