"""Direct target-frequency full-space Hankel LS with multiple free real directions per element.

N=2 mesh, kappa=8, source=(-1.6,0.3).  Each element has r local directions.
The directions are initialized by a small symmetric fan about the centroid-radial ray;
all coefficients start from zero.  All coefficients and all directions are then optimized
simultaneously in one nonlinear least-squares solve.  No frequency continuation.
"""
import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.optimize import least_squares
from pathlib import Path

from exact import Hankel
from pwdg import square_mesh, ALPHA, BETA
from direction_adaptive_automatic import PWDGDictionary, l2_error
from revision_experiments import trace_ls_system

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'generated'; OUT.mkdir(exist_ok=True)
N_MESH=2; KAPPA=8.0; SOURCE=(-1.6,0.3); NQ=14


def q_and_qprime(k,z):
    q=k*np.array([np.cos(z),np.sin(z)],complex)
    qp=k*np.array([-np.sin(z),np.cos(z)],complex)
    return q,qp


def offsets_for_r(r):
    # Moderate angular fan, wide enough to represent local wavefront curvature without near-coalescence.
    if r==1: return np.array([0.0])
    # fixed total span 40 degrees for r<=5, 50 thereafter
    span=40.0 if r<=5 else 50.0
    return np.deg2rad(np.linspace(-span/2,span/2,r))


def build_state(ex,z,r,nq=NQ):
    z=np.asarray(z,complex).reshape(-1)
    v,t=square_mesh(N_MESH); nt=len(t); cent=v[t].mean(axis=1)
    if len(z)!=nt*r: raise ValueError('z length must nt*r')
    kap=np.asarray(ex.wavenumber(cent[:,1]),float)
    q=[]; qp=[]
    for e,k in enumerate(kap):
        qe=[]; qpe=[]
        for ell in range(r):
            zz=z[e*r+ell]
            qq,qqp=q_and_qprime(k,zz); qe.append(qq);qpe.append(qqp)
        q.append(np.array(qe));qp.append(np.array(qpe))
    s=PWDGDictionary(N_MESH,ex,q,nq=nq)
    D,b=trace_ls_system(s)
    return s,D,b,qp


def residual_and_Kz(ex,z,c,r,nq=NQ):
    s,D,b,qp=build_state(ex,z,r,nq)
    c=np.asarray(c,complex); res=D@c-b
    m=len(z); blocks=[[] for _ in range(m)]
    for f in range(s.et.shape[0]):
        pts,wq,nrm0,L=s.edge(f); e0,e1=s.et[f]; sw=np.sqrt(wq)
        if e1>=0:
            n0=s._outward(e0,nrm0,pts); xi=0.5*(s.kap[e0]+s.kap[e1])
            du_by=[]; dg_by=[]
            for e in (e0,e1):
                R=pts-s.cent[e]; ph=s._basis(e,pts); cl=c[s.sl(e)]
                dume={}; dgme={}
                for ell in range(r):
                    j=e*r+ell; qq=s.q[e][ell]; qqp=qp[e][ell]
                    dot=R@qqp; dphi=1j*dot*ph[:,ell]
                    dgrad=(1j*qqp[None,:]-dot[:,None]*qq[None,:])*ph[:,ell,None]
                    dume[j]=cl[ell]*dphi; dgme[j]=cl[ell]*dgrad
                du_by.append(dume);dg_by.append(dgme)
            for j in range(m):
                du0=du_by[0].get(j,np.zeros(len(pts),complex));du1=du_by[1].get(j,np.zeros(len(pts),complex))
                dg0=dg_by[0].get(j,np.zeros((len(pts),2),complex));dg1=dg_by[1].get(j,np.zeros((len(pts),2),complex))
                bru=np.sqrt(ALPHA*xi)*sw*(du0-du1)
                brf=np.sqrt(BETA/xi)*sw*(dg0@n0+dg1@(-n0))
                blocks[j].extend([bru,brf])
        else:
            e=e0;R=pts-s.cent[e];ph=s._basis(e,pts);cl=c[s.sl(e)]
            dume={}
            for ell in range(r):
                j=e*r+ell; qqp=qp[e][ell];dot=R@qqp
                dume[j]=cl[ell]*(1j*dot*ph[:,ell])
            for j in range(m):
                du=dume.get(j,np.zeros(len(pts),complex))
                blocks[j].append(np.sqrt(ALPHA*s.kap[e])*sw*du)
    Kz=np.column_stack([np.concatenate(blocks[j]) for j in range(m)])
    return s,D,b,res,Kz


def solve_r(r,max_nfev=1200):
    ex=Hankel(KAPPA,source=SOURCE)
    v,t=square_mesh(N_MESH);cent=v[t].mean(axis=1)
    d=cent-np.asarray(SOURCE);rad=np.arctan2(d[:,1],d[:,0])%(2*np.pi)
    offs=offsets_for_r(r)
    z0=np.concatenate([rad[e]+offs for e in range(len(t))]).astype(complex)
    s,D,b,qp=build_state(ex,z0,r,NQ); ndof=D.shape[1]; m=len(z0)
    # zero coefficient start; real directions only
    x0=np.r_[np.zeros(ndof),np.zeros(ndof),z0.real]
    def unpack(x):
        c=x[:ndof]+1j*x[ndof:2*ndof]
        z=x[2*ndof:].astype(complex)
        return c,z
    def fun(x):
        c,z=unpack(x);res=residual_and_Kz(ex,z,c,r,NQ)[3]
        return np.r_[res.real,res.imag]
    def jac(x):
        c,z=unpack(x);_,Dl,_,res,Kz=residual_and_Kz(ex,z,c,r,NQ)
        return np.hstack([np.vstack([Dl.real,Dl.imag]),np.vstack([-Dl.imag,Dl.real]),np.vstack([Kz.real,Kz.imag])])
    rr=least_squares(fun,x0,jac=jac,method='trf',x_scale='jac',ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=max_nfev)
    c,z=unpack(rr.x);s,D,b,res,Kz=residual_and_Kz(ex,z,c,r,20);s.u=c
    J=float(np.vdot(res,res).real);e=l2_error(s,14)
    grad=np.column_stack([D,Kz]).conj().T@res
    # angle spread relative radial for diagnostics
    zr=z.real.reshape(len(t),r)
    delta=(zr-rad[:,None]+np.pi)%(2*np.pi)-np.pi
    return dict(r=r,nominal_dofs=ndof,direction_variables=m,nfev=rr.nfev,J_Q=J,relative_L2=e,
                grad_norm=float(la.norm(grad)),success=rr.success,status=rr.status,
                mean_abs_offset_deg=float(np.degrees(np.mean(np.abs(delta)))),
                max_abs_offset_deg=float(np.degrees(np.max(np.abs(delta)))),
                theta_deg=';'.join(f'{x:.10g}' for x in (np.degrees(z.real)%360)))


def main():
    rows=[]
    for r in [1,2,3,4,5]:
        o=solve_r(r)
        rows.append(o);print(o)
    df=pd.DataFrame(rows);df.to_csv(OUT/'hankel_direct_multidir_joint.csv',index=False)
    print('\n',df[['r','nominal_dofs','direction_variables','nfev','J_Q','relative_L2','grad_norm']].to_string(index=False))

if __name__=='__main__': main()
