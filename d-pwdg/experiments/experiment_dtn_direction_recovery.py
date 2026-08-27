"""Direction recovery on an exact circular DtN boundary.

Author
------
Shelvean Kapita
Department of Mathematics, Texas A&M University

Date
----
August 2026

Three experiments are provided:
  1. centered outgoing Hankel wave (calibration),
  2. off-center outgoing Hankel wave (geometry-vs-physics test),
  3. sound-soft disk scattering (genuine scattering field).

The annulus is represented exactly in polar coordinates, with a=0.5 and R=1,
as in the dissertation and Kapita--Monk JCAM experiment.  Direction variables
are optimized in the Part-A sense: for every angle state, the DtN-PWDG
Galerkin equations are solved first, and the positive skeleton residual is then
minimized.

The preliminary optimizer uses centered finite-difference Jacobians supplied by
SciPy.  This script is intended to establish and validate the DtN experiment
before the analytic direction derivative is folded into the main solver.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import qmc

from core.dtn_pwdg import (
    PolarAnnulusMesh, DtNPWDG, Hankel2Field, SoundSoftDiskScattering,
    angle_error_deg,
)

__author__ = "Shelvean Kapita"
__date__ = "August 2026"


def sobol_angles(n, seed=17):
    sampler=qmc.Sobol(d=1,scramble=True,seed=seed)
    x=sampler.random(n).ravel()
    return 2*np.pi*x


def solve_angles(mesh, k, exact, start, p=1, dtn_N=None, max_nfev=120, fan_halfwidth_deg=18.0):
    last={'solver':None}
    if p == 1:
        offsets=np.array([0.0])
    else:
        offsets=np.deg2rad(np.linspace(-fan_halfwidth_deg,fan_halfwidth_deg,p))
    def expand(center):
        return (np.asarray(center)[:,None] + offsets[None,:]).ravel()
    def residual(center):
        theta=expand(center)
        s=DtNPWDG(mesh,k,theta,exact,p_per_element=p,dtn_N=dtn_N,edge_order=16)
        s.assemble()
        s.solve()
        last['solver']=s
        return s.physical_residual_vector()
    opt=least_squares(residual,start,method='trf',jac='2-point',
                      max_nfev=max_nfev,xtol=2e-9,ftol=2e-9,gtol=2e-9,
                      x_scale='jac',verbose=0)
    s=DtNPWDG(mesh,k,expand(opt.x),exact,p_per_element=p,dtn_N=dtn_N,edge_order=20)
    s.assemble()
    s.solve()
    return opt,s


def continuation(case='offcenter', target_k=8.0, nr=2, ntheta=12, seed=17, p=3):
    mesh=PolarAnnulusMesh(a=0.5,R=1.0,nr=nr,ntheta=ntheta)
    start=sobol_angles(mesh.ne,seed=seed)
    schedule=[1.0,2.0,4.0] + ([target_k] if target_k>4.0 else []) if target_k>=4 else [1.0,target_k]
    rows=[]
    opt=None
    s=None
    for k in schedule:
        if case=='centered':
            exact=Hankel2Field(k,(0.0,0.0))
        elif case=='offcenter':
            exact=Hankel2Field(k,(0.18,-0.12))
        elif case=='scattering':
            exact=SoundSoftDiskScattering(k,a=0.5,series_N=100)
        else:
            raise ValueError(case)
        N=max(16,int(np.ceil(1.5*k*mesh.R)+4))
        opt,s=solve_angles(mesh,k,exact,start,p=p,dtn_N=N,max_nfev=80)
        start=np.mod(opt.x,2*np.pi)
        rows.append(dict(case=case,k=k,dtn_N=N,nfev=opt.nfev,cost=2*opt.cost,
                         rel_l2=s.relative_l2_error(),**s.graph_conditioning()))
        print(rows[-1])
    return mesh,opt,s,pd.DataFrame(rows)


def outer_direction_diagnostics(mesh,s,case):
    outer=[]
    for e,el in enumerate(mesh.elements):
        if abs(el.r1-mesh.R)<1e-13:
            th=0.5*(el.th0+el.th1)
            x=mesh.R*np.cos(th)
            y=mesh.R*np.sin(th)
            learned=np.mod(s.angles[e,s.p//2],2*np.pi)
            if isinstance(s.exact,Hankel2Field):
                ref=float(s.exact.wavevector_angle(x,y))
            else:
                # For scattering there is generally no single exact ray.  Use
                # the local phase gradient estimated from the exact field by a
                # small centered difference as a diagnostic only.
                h=2e-6
                u=s.exact(x,y)
                ux=(s.exact(x+h,y)-s.exact(x-h,y))/(2*h)
                uy=(s.exact(x,y+h)-s.exact(x,y-h))/(2*h)
                qx=np.imag(ux/u)
                qy=np.imag(uy/u)
                ref=float(np.mod(np.arctan2(qy,qx),2*np.pi))
            normal=np.mod(th,2*np.pi)
            outer.append(dict(element=e,theta_boundary=np.rad2deg(th),
                              learned_deg=np.rad2deg(learned),
                              exact_wavevector_deg=np.rad2deg(ref),
                              error_to_exact_deg=float(angle_error_deg(learned,ref)),
                              error_to_outward_normal_deg=float(angle_error_deg(learned,normal))))
    return pd.DataFrame(outer)


def make_plots(mesh,s,diag,outdir,stem):
    outdir=Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    # Direction field on the annulus.  Arrow is the wave vector q; for Hankel-2
    # outgoing waves the physical propagation direction is opposite the arrow.
    fig,ax=plt.subplots(figsize=(6.2,6.2))
    tt=np.linspace(0,2*np.pi,500)
    ax.plot(mesh.a*np.cos(tt),mesh.a*np.sin(tt),'k-',lw=1.2)
    ax.plot(mesh.R*np.cos(tt),mesh.R*np.sin(tt),'k-',lw=1.2)
    for r in mesh.rs[1:
        -1]: ax.plot(r*np.cos(tt),r*np.sin(tt),'k-',lw=.35,alpha=.45)
    for th in mesh.ths[:
        -1]: ax.plot([mesh.a*np.cos(th),mesh.R*np.cos(th)],
                                     [mesh.a*np.sin(th),mesh.R*np.sin(th)],'k-',lw=.35,alpha=.45)
    c=mesh.centers
    ang=s.angles[:,s.p//2]
    ax.quiver(c[:,0],c[:,1],np.cos(ang),np.sin(ang),angles='xy',scale_units='xy',scale=8.0,width=.005)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xticks([-1,-.5,0,.5,1])
    ax.set_yticks([-1,-.5,0,.5,1])
    fig.tight_layout()
    fig.savefig(outdir/f'{stem}_directions.png',dpi=220)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(6.6,4.3))
    ax.plot(diag.theta_boundary,diag.error_to_exact_deg,'o-',label='learned vs exact local wave vector')
    ax.plot(diag.theta_boundary,diag.error_to_outward_normal_deg,'s--',label='learned vs outward circle normal')
    ax.set_xlabel('boundary polar angle (degrees)')
    ax.set_ylabel('angular difference (degrees)')
    ax.set_xticks([0,90,180,270,360])
    ymax=max(1.0,float(np.nanmax([diag.error_to_exact_deg.max(),diag.error_to_outward_normal_deg.max()])))
    ax.set_yticks(np.linspace(0,np.ceil(ymax/10)*10 if ymax>10 else ymax,5))
    ax.grid(True,which='both',alpha=.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(outdir/f'{stem}_boundary_errors.png',dpi=220)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--case',choices=['centered','offcenter','scattering'],default='offcenter')
    ap.add_argument('--k',type=float,default=8.0)
    ap.add_argument('--nr',type=int,default=2)
    ap.add_argument('--ntheta',type=int,default=12)
    ap.add_argument('--p',type=int,default=3)
    ap.add_argument('--outdir',default='results_dtn')
    args=ap.parse_args()
    mesh,opt,s,hist=continuation(args.case,args.k,args.nr,args.ntheta,p=args.p)
    diag=outer_direction_diagnostics(mesh,s,args.case)
    out=Path(args.outdir)
    out.mkdir(parents=True,exist_ok=True)
    hist.to_csv(out/f'dtn_{args.case}_continuation.csv',index=False)
    diag.to_csv(out/f'dtn_{args.case}_outer_directions.csv',index=False)
    make_plots(mesh,s,diag,out,f'dtn_{args.case}')
    print('\nOuter-boundary direction diagnostics:')
    print(diag.to_string(index=False))
    print('\nmean exact-direction error =',diag.error_to_exact_deg.mean())
    print('max exact-direction error  =',diag.error_to_exact_deg.max())
    print('relative L2 error          =',s.relative_l2_error())

if __name__=='__main__':
    main()
