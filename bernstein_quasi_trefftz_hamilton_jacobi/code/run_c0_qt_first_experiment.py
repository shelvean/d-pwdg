#!/usr/bin/env python3
"""Experiment 1: full C0 parent space versus C0-quasi-Trefftz correction.

The problem is a smooth, genuinely two-dimensional stationary Hamilton-Jacobi equation
with a rotating anisotropic quadratic Hamiltonian.  For each mesh/degree we first compute
the L2-best approximation in the full C0 Bernstein parent space (with exact boundary trace),
then use that coefficient vector as the initial state for the p-1 Hamiltonian moment solve.
This isolates the cost of imposing the qT equations from branch-selection effects.
"""
from __future__ import annotations
import math, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
from optimized_hj import FastHamiltonJacobiPminus1
from eikonal_c0_pminus1 import triangle_quadrature, bernstein_values

DEG=(3,5,7,9,11,13)
HLEVELS=(2,3,4,5,6,7)

class RotatingAnisotropicHJ:
    name='rotating anisotropic HJ'
    def u(self,x):
        X=x[:,0];Y=x[:,1]
        return (X+0.32*Y
                +0.055*np.sin(2*np.pi*X)*np.sin(np.pi*Y)
                +0.025*np.sin(np.pi*X)*np.cos(2*np.pi*Y))
    def grad(self,x):
        X=x[:,0];Y=x[:,1]
        ux=(1+0.11*np.pi*np.cos(2*np.pi*X)*np.sin(np.pi*Y)
              +0.025*np.pi*np.cos(np.pi*X)*np.cos(2*np.pi*Y))
        uy=(0.32+0.055*np.pi*np.sin(2*np.pi*X)*np.cos(np.pi*Y)
              -0.05*np.pi*np.sin(np.pi*X)*np.sin(2*np.pi*Y))
        return np.column_stack([ux,uy])
    def Avec(self,x,p):
        X=x[:,0];Y=x[:,1]
        theta=0.35*np.sin(np.pi*X)*np.cos(np.pi*Y)
        c=np.cos(theta);s=np.sin(theta)
        rho=3.0+0.5*np.sin(np.pi*X)*np.sin(np.pi*Y)
        A11=c*c+rho*s*s; A22=s*s+rho*c*c; A12=(1-rho)*c*s
        return np.column_stack([A11*p[:,0]+A12*p[:,1],A12*p[:,0]+A22*p[:,1]])
    def f(self,x):
        g=self.grad(x);return 0.5*np.sum(g*self.Avec(x,g),axis=1)
    def H(self,x,p):
        return 0.5*np.sum(p*self.Avec(x,p),axis=1)-self.f(x)
    def Hp(self,x,p):return self.Avec(x,p)


def full_c0_projection(model):
    p=model.p;ref,wref=triangle_quadrature(max(12,2*p+6))
    lam=np.column_stack([1-ref[:,0]-ref[:,1],ref[:,0],ref[:,1]])
    B=bernstein_values(p,lam);nd=len(model.asm.keys)
    rows=[];cols=[];vals=[];rhs=np.zeros(nd)
    for K,tri in enumerate(model.tris):
        V=model.verts[tri];J=np.column_stack((V[1]-V[0],V[2]-V[0]));det=abs(np.linalg.det(J))
        pts=V[0]+ref@J.T;W=wref*det;ue=model.problem.u(pts)
        M=B.T@(W[:,None]*B);b=B.T@(W*ue);g=model.asm.l2g[K]
        rows.append(np.repeat(g,len(g)));cols.append(np.tile(g,len(g)));vals.append(M.ravel());np.add.at(rhs,g,b)
    A=sp.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(nd,nd)).tocsr()
    c=np.zeros(nd);c[model.fixed]=model.fixed_vals;free=model.free
    b=rhs[free]-A[free][:,model.fixed]@c[model.fixed]
    t0=time.perf_counter();c[free]=spsolve(A[free][:,free],b);return c,time.perf_counter()-t0


def residual_rms(model,c):
    model._base=np.asarray(c).copy();model._base[model.fixed]=model.fixed_vals;model._cache_z=None
    R=model.residual(model._base[model.free]);return float(np.linalg.norm(R)/math.sqrt(len(R)))


def run_one(n,p,prob):
    m=FastHamiltonJacobiPminus1(n,p,prob)
    cf,tp=full_c0_projection(m);f2,f1=m.errors(cf);fr=residual_rms(m,cf)
    cq,sol,tq=m.solve(initial_full=cf,max_nfev=40,lsmr_tol=2e-12);q2,q1=m.errors(cq);qr=residual_rms(m,cq)
    Np=(p+1)*(p+2)//2;Nq=p+1
    return dict(n=n,ntri=len(m.tris),p=p,global_c0_dofs=len(cf),free_c0_dofs=len(m.free),
                local_c0_dim=Np,local_qt_dim=Nq,local_compression=Np/Nq,
                c0_relL2=f2,qt_relL2=q2,L2_ratio=q2/f2,c0_relH1=f1,qt_relH1=q1,H1_ratio=q1/f1,
                c0_residual=fr,qt_residual=qr,residual_ratio=qr/fr,
                c0_projection_time=tp,qt_correction_time=tq,qt_nfev=int(sol.nfev))


def main():
    prob=RotatingAnisotropicHJ();out=ROOT/'data';out.mkdir(exist_ok=True)
    rp=[]
    for p in DEG:
        r=run_one(4,p,prob);print('p',r,flush=True);rp.append(r)
    pd.DataFrame(rp).to_csv(out/'c0_qt_rotating_anisotropic_p_sweep.csv',index=False)
    rh=[]
    for n in HLEVELS:
        r=run_one(n,7,prob);print('h',r,flush=True);rh.append(r)
    pd.DataFrame(rh).to_csv(out/'c0_qt_rotating_anisotropic_h_sweep.csv',index=False)

if __name__=='__main__':main()
