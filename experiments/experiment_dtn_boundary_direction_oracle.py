"""Local Cauchy-trace direction diagnostic on the exact circular DtN boundary.

Author
------
Shelvean Kapita
Department of Mathematics, Texas A&M University

Date
----
August 2026

This script is a *diagnostic/oracle experiment*, not the automatic PWDG
algorithm.  It asks what single plane-wave direction best represents the exact
Cauchy trace on each arc of the artificial circle.  The result gives a clean
reference against which the directions learned by the nonlinear DtN-PWDG solve
can later be compared.

The geometry is the exact annulus a=0.5, R=1 used in the dissertation and the
Kapita--Monk JCAM paper.  No polygonal boundary approximation is made.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import hankel2, jv, h2vp

from core.quadrature import gauss01
from core.dtn_pwdg import Hankel2Field, SoundSoftDiskScattering, angle_error_deg

__author__ = "Shelvean Kapita"
__date__ = "August 2026"


def scattering_gradient(field, x, y):
    """Analytic gradient of the circular sound-soft scattered field."""
    x=np.asarray(x)
    y=np.asarray(y)
    r=np.hypot(x,y)
    th=np.arctan2(y,x)
    k=field.k
    a=field.a
    ur=-(jv(0,k*a)/hankel2(0,k*a))*k*h2vp(0,k*r,1)
    uth=np.zeros_like(r,dtype=complex)
    for m in range(1,field.series_N+1):
        coef=-2.0*(1j**m)*jv(m,k*a)/hankel2(m,k*a)
        H=hankel2(m,k*r)
        ur += coef*k*h2vp(m,k*r,1)*np.cos(m*th)
        uth += coef*H*(-m*np.sin(m*th))
    ux=np.cos(th)*ur - np.sin(th)*uth/r
    uy=np.sin(th)*ur + np.cos(th)*uth/r
    return np.column_stack((ux,uy))


def exact_cauchy(field, pts, normals):
    u=field(pts[:,0],pts[:,1])
    if isinstance(field,Hankel2Field):
        grad=field.gradient(pts[:,0],pts[:,1])
    else:
        grad=scattering_gradient(field,pts[:,0],pts[:,1])
    dn=np.einsum('qc,qc->q',grad,normals)
    return u,dn,grad


def best_plane_wave_on_arc(field, R, th0, th1, nq=28):
    s,w=gauss01(nq)
    th=th0+(th1-th0)*s
    pts=np.column_stack((R*np.cos(th),R*np.sin(th)))
    n=np.column_stack((np.cos(th),np.sin(th)))
    wds=w*R*(th1-th0)
    u,dn,grad=exact_cauchy(field,pts,n)
    thc=.5*(th0+th1)
    xc=np.array([R*np.cos(thc),R*np.sin(thc)])
    k=field.k

    # Cauchy metric: k||u-v||^2 + k^{-1}||dn u-dn v||^2.
    b=np.concatenate((np.sqrt(k*wds)*u, np.sqrt(wds/k)*dn))
    bnorm2=float(np.vdot(b,b).real)
    def reduced(theta):
        d=np.array([np.cos(theta),np.sin(theta)])
        ph=np.exp(1j*k*((pts-xc)@d))
        dph=1j*k*(n@d)*ph
        aa=np.concatenate((np.sqrt(k*wds)*ph,np.sqrt(wds/k)*dph))
        c=np.vdot(aa,b)/np.vdot(aa,aa)
        r=aa*c-b
        return float(np.vdot(r,r).real/bnorm2)

    grid=np.linspace(0,2*np.pi,181,endpoint=False)
    vals=np.array([reduced(t) for t in grid])
    j=int(np.argmin(vals))
    h=2*np.pi/181
    lo=grid[j]-1.5*h
    hi=grid[j]+1.5*h
    # unwrap around grid minimizer for a bounded local polish
    f=lambda z: reduced(np.mod(z,2*np.pi))
    opt=minimize_scalar(f,bounds=(lo,hi),method='bounded',options={'xatol':1e-12})
    learned=float(np.mod(opt.x,2*np.pi))

    # Exact local phase-gradient direction at arc midpoint.
    pm=xc[None,:]
    nm=np.array([[np.cos(thc),np.sin(thc)]])
    um,dnm,gm=exact_cauchy(field,pm,nm)
    qloc=np.imag(gm[0]/um[0])
    exact_ang=float(np.mod(np.arctan2(qloc[1],qloc[0]),2*np.pi))
    return learned,exact_ang,float(opt.fun)


def run_case(case,k=8.0,a=.5,R=1.0,ntheta=24):
    if case=='centered':
        field=Hankel2Field(k,(0.,0.))
    elif case=='offcenter':
        field=Hankel2Field(k,(0.18,-0.12))
    elif case=='scattering':
        field=SoundSoftDiskScattering(k,a=a,series_N=100)
    else:
        raise ValueError(case)
    ths=np.linspace(0,2*np.pi,ntheta+1)
    rows=[]
    for j in range(ntheta):
        th0,th1=ths[j],ths[j+1]
        thc=.5*(th0+th1)
        learned,exact,res=best_plane_wave_on_arc(field,R,th0,th1)
        outward=np.mod(thc,2*np.pi)
        rows.append(dict(case=case,k=k,arc=j,boundary_angle_deg=np.rad2deg(thc),
                         learned_deg=np.rad2deg(learned),
                         exact_phase_deg=np.rad2deg(exact),
                         error_exact_deg=float(angle_error_deg(learned,exact)),
                         error_outward_normal_deg=float(angle_error_deg(learned,outward)),
                         local_relative_cauchy_residual=res))
    return field,pd.DataFrame(rows)


def plots(field,df,outdir,case,R=1.0):
    out=Path(outdir)
    out.mkdir(parents=True,exist_ok=True)
    # Circular arrow plot: learned q vectors and exact local q vectors.
    th=np.deg2rad(df.boundary_angle_deg.to_numpy())
    x=R*np.cos(th)
    y=R*np.sin(th)
    al=np.deg2rad(df.learned_deg.to_numpy())
    ae=np.deg2rad(df.exact_phase_deg.to_numpy())
    tt=np.linspace(0,2*np.pi,500)
    fig,ax=plt.subplots(figsize=(6.4,6.4))
    ax.plot(R*np.cos(tt),R*np.sin(tt),'k-',lw=1.1)
    ax.plot(.5*np.cos(tt),.5*np.sin(tt),'k-',lw=1.1)
    ax.quiver(x,y,np.cos(ae),np.sin(ae),angles='xy',scale_units='xy',scale=5.0,width=.004,label='exact local phase')
    ax.quiver(x,y,np.cos(al),np.sin(al),angles='xy',scale_units='xy',scale=5.0,width=.007,alpha=.65,label='best local plane wave')
    ax.set_aspect('equal')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xticks([-1,-.5,0,.5,1])
    ax.set_yticks([-1,-.5,0,.5,1])
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(out/f'dtn_{case}_boundary_directions.png',dpi=240)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.0,4.4))
    ax.plot(df.boundary_angle_deg,df.error_exact_deg,'o-',label='vs exact local phase')
    ax.plot(df.boundary_angle_deg,df.error_outward_normal_deg,'s--',label='vs outward circle normal')
    ax.set_xlabel('boundary polar angle (degrees)')
    ax.set_ylabel('angular difference (degrees)')
    ax.set_xticks([0,90,180,270,360])
    ymax=max(1.,df.error_exact_deg.max(),df.error_outward_normal_deg.max())
    top=max(5.,np.ceil(ymax/10.)*10.)
    ax.set_yticks(np.linspace(0,top,5))
    ax.grid(True,alpha=.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out/f'dtn_{case}_boundary_direction_error.png',dpi=240)
    plt.close(fig)


def main():
    out=Path('/mnt/data/dtn_work/results_oracle')
    out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for case in ('centered','offcenter','scattering'):
        field,df=run_case(case,k=8.,ntheta=24)
        df.to_csv(out/f'dtn_{case}_boundary_oracle.csv',index=False)
        plots(field,df,out,case)
        summary.append(dict(case=case,
                            mean_angle_error_deg=df.error_exact_deg.mean(),
                            max_angle_error_deg=df.error_exact_deg.max(),
                            mean_normal_difference_deg=df.error_outward_normal_deg.mean(),
                            max_local_cauchy_residual=df.local_relative_cauchy_residual.max()))
    sdf=pd.DataFrame(summary)
    sdf.to_csv(out/'dtn_boundary_oracle_summary.csv',index=False)
    print(sdf.to_string(index=False))

if __name__=='__main__':
    main()
