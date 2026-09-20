#!/usr/bin/env python3
"""C0 parent-space versus C0-quasi-Trefftz correction on smoothed Marmousi.

This is an approximation-space experiment, separate from the later causal branch-selection
Marmousi experiment.  The native causal reference supplies the *outer trace* in both methods.
The full C0 benchmark is the L2 projection of the factored reference branch into S_p^0.
Starting from that parent-space approximant, the C0-qT result applies the degree-(p-1)
Hamiltonian moment correction with the same exact C0 boundary trace.  Thus the experiment
isolates the qT restriction; it does not test branch selection.
"""
from __future__ import annotations
import math, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
from marmousi_bb_core import MarmProblem, RectFastFactored, make_seed, boundary_edges
from high_frequency_benchmarks import RadialFactor, bernstein_1d_matrix
from eikonal_c0_pminus1 import bernstein_values, triangle_quadrature
from marmousi_fast_sweeping_reference import fast_sweep

VX=Path('/mnt/data/vel_marmousi_smooth400_376x1151.csv.txt')
DOMAIN_X=9.2; DOMAIN_Z=3.0; SRC=np.array([4.6,0.0])
DEG=(3,5,7,9,11,13); HSEQ=((6,2),(12,4),(18,6),(24,8),(30,10),(36,12))


def load():
    v=np.loadtxt(VX,delimiter=',')/1000.0
    z=np.linspace(0,DOMAIN_Z,v.shape[0]); x=np.linspace(0,DOMAIN_X,v.shape[1])
    ix=np.argmin(abs(x-SRC[0])); T,_,_=fast_sweep(1/v,x[1]-x[0],z[1]-z[0],0,ix,100,1e-12)
    xc,zc,vc,Tseed,_=make_seed(v,x,z,72,24,tuple(SRC)); prob=MarmProblem(v,x,z,Tseed,xc,zc)
    fac=RadialFactor(SRC,1/float(v[0,ix])); interp=RegularGridInterpolator((z,x),T,bounds_error=False,fill_value=None)
    return x,z,T,prob,fac,interp


def tau_ref(model,interp,pts):
    return interp(np.column_stack([pts[:,1],pts[:,0]]))-model.factor.u(pts)


def impose_reference_boundary(model,interp):
    kv={}; key_to_gid={k:i for i,k in enumerate(model.asm.keys)}
    bedges=boundary_edges(model.tris); bverts=sorted(set(v for e in bedges for v in e))
    for v in bverts:
        gid=key_to_gid.get(('v',int(v)))
        if gid is not None: kv[gid]=float(tau_ref(model,interp,model.verts[[v]])[0])
    ts=np.linspace(0,1,model.p+1); B1=bernstein_1d_matrix(model.p,ts)
    for a,b in bedges:
        va,vb=model.verts[a],model.verts[b]; pts=(1-ts[:,None])*va+ts[:,None]*vb
        coeff=la.solve(B1,tau_ref(model,interp,pts),check_finite=False)
        for k in range(1,model.p):
            gid=key_to_gid.get(('e',int(a),int(b),k))
            if gid is not None: kv[gid]=float(coeff[k])
    model.fixed=np.array(sorted(kv),int); model.fixed_vals=np.array([kv[g] for g in model.fixed])
    allidx=np.arange(len(model.asm.keys)); mask=np.ones(len(allidx),bool);mask[model.fixed]=False;model.free=allidx[mask]
    model.g2free=np.full(len(allidx),-1,int);model.g2free[model.free]=np.arange(len(model.free));model.local_free=model.g2free[model.asm.l2g]
    pe=[];pb=[];pj=[]
    for K in range(len(model.tris)):
        js=np.nonzero(model.local_free[K]>=0)[0]
        if not len(js):continue
        bb=np.repeat(np.arange(model.Nr),len(js));jj=np.tile(js,model.Nr)
        pe.append(np.full(len(bb),K,int));pb.append(bb);pj.append(jj)
    model.pe=np.concatenate(pe);model.pb=np.concatenate(pb);model.pj=np.concatenate(pj)
    model.prows=model.pe*model.Nr+model.pb;model.pcols=model.local_free[model.pe,model.pj]
    model._base=None;model._cz=None;model._R=None;model._gh=None;model._J=None


def full_c0_projection(model,interp):
    p=model.p; nd=len(model.asm.keys); ref,wref=triangle_quadrature(max(10,min(18,p+4)))
    lam=np.column_stack([1-ref[:,0]-ref[:,1],ref[:,0],ref[:,1]]);B=bernstein_values(p,lam)
    rr=[];cc=[];vv=[];rhs=np.zeros(nd)
    for K,tri in enumerate(model.tris):
        V=model.verts[tri];J=np.column_stack((V[1]-V[0],V[2]-V[0]));det=abs(np.linalg.det(J));pts=V[0]+ref@J.T;W=wref*det
        t=tau_ref(model,interp,pts); M=B.T@(W[:,None]*B); b=B.T@(W*t); gids=model.asm.l2g[K]
        rr.append(np.repeat(gids,len(gids)));cc.append(np.tile(gids,len(gids)));vv.append(M.ravel());np.add.at(rhs,gids,b)
    M=sp.coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(nd,nd)).tocsr()
    c=np.zeros(nd);c[model.fixed]=model.fixed_vals;free=model.free
    b=rhs[free]-M[free][:,model.fixed]@c[model.fixed]
    t0=time.perf_counter();c[free]=spsolve(M[free][:,free],b);return c,time.perf_counter()-t0


def metrics(model,c,Tref,x,z):
    U=model.eval_grid(c,x,z);e=U-Tref;den=np.sqrt(np.mean(Tref*Tref));rel=float(np.sqrt(np.mean(e*e))/den);mae=float(np.mean(abs(e)))
    model._base=np.asarray(c).copy();model._base[model.fixed]=model.fixed_vals;model._cz=None
    R,_=model.state(model._base[model.free]);res=float(np.linalg.norm(R)/math.sqrt(len(R)))
    return rel,mae,res


def one(nx,nz,p,Tref,x,z,prob,fac,interp):
    m=RectFastFactored(nx,nz,p,prob,fac,qorder=max(10,min(16,p+3)));impose_reference_boundary(m,interp)
    cp,tp=full_c0_projection(m,interp); fr,fmae,fres=metrics(m,cp,Tref,x,z)
    cq,sol,tq=m.solve(cp,maxit=8,lsmr_tol=1e-8); qr,qmae,qres=metrics(m,cq,Tref,x,z)
    Np=(p+1)*(p+2)//2;Nq=p+1
    return dict(nx=nx,nz=nz,ntri=len(m.tris),p=p,c0_dofs=len(cp),free_c0_dofs=len(m.free),local_c0_dim=Np,local_qt_dim=Nq,
                local_compression=Np/Nq,c0_relrmse=fr,qt_relrmse=qr,error_ratio=qr/fr,c0_mae=fmae,qt_mae=qmae,
                c0_residual=fres,qt_residual=qres,residual_ratio=qres/fres,c0_projection_time=tp,qt_correction_time=tq)


def main():
    x,z,Tref,prob,fac,interp=load();out=ROOT/'data';out.mkdir(exist_ok=True)
    rp=[]
    for p in DEG:
        row=one(12,4,p,Tref,x,z,prob,fac,interp);print('p',row,flush=True);rp.append(row)
    pd.DataFrame(rp).to_csv(out/'marmousi_c0_qt_p_sweep.csv',index=False)
    rh=[]
    for nx,nz in HSEQ:
        row=one(nx,nz,7,Tref,x,z,prob,fac,interp);print('h',row,flush=True);rh.append(row)
    pd.DataFrame(rh).to_csv(out/'marmousi_c0_qt_h_sweep.csv',index=False)
if __name__=='__main__':main()
