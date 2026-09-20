"""First-order causal fast sweeping on a rectangular Cartesian grid.

Used only to select/reference the viscosity first-arrival branch in the Marmousi
experiment.  Coordinates may have unequal x/z spacings.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit

@njit(cache=True)
def _local_update(a,b,s,hx,hz):
    one=min(a+s*hx,b+s*hz)
    if not math.isfinite(a) and not math.isfinite(b):
        return 1e30
    if not math.isfinite(a): return b+s*hz
    if not math.isfinite(b): return a+s*hx
    A=1.0/(hx*hx)+1.0/(hz*hz)
    B=a/(hx*hx)+b/(hz*hz)
    C=a*a/(hx*hx)+b*b/(hz*hz)-s*s
    D=B*B-A*C
    if D<=0.0: return one
    t=(B+math.sqrt(D))/A
    if t<max(a,b): return one
    return min(t,one) if one<max(a,b) else t

@njit(cache=True)
def _sweep_once(T,s,hx,hz,iz0,iz1,izstep,ix0,ix1,ixstep,sz,sx):
    nz,nx=T.shape;maxchg=0.0
    for iz in range(iz0,iz1,izstep):
        for ix in range(ix0,ix1,ixstep):
            if iz==sz and ix==sx: continue
            a=1e30
            if ix>0 and T[iz,ix-1]<a: a=T[iz,ix-1]
            if ix<nx-1 and T[iz,ix+1]<a: a=T[iz,ix+1]
            b=1e30
            if iz>0 and T[iz-1,ix]<b: b=T[iz-1,ix]
            if iz<nz-1 and T[iz+1,ix]<b: b=T[iz+1,ix]
            tn=_local_update(a,b,s[iz,ix],hx,hz)
            if tn<T[iz,ix]:
                d=T[iz,ix]-tn
                if d>maxchg:maxchg=d
                T[iz,ix]=tn
    return maxchg

@njit(cache=True)
def fast_sweep(slowness,hx,hz,source_iz,source_ix,maxcycles=100,tol=1e-12):
    """Solve |grad T|=slowness with T(source)=0 by four-direction sweeping."""
    nz,nx=slowness.shape
    T=np.full((nz,nx),1e30)
    T[source_iz,source_ix]=0.0
    for cyc in range(maxcycles):
        mc=0.0
        mc=max(mc,_sweep_once(T,slowness,hx,hz,0,nz,1,0,nx,1,source_iz,source_ix))
        mc=max(mc,_sweep_once(T,slowness,hx,hz,0,nz,1,nx-1,-1,-1,source_iz,source_ix))
        mc=max(mc,_sweep_once(T,slowness,hx,hz,nz-1,-1,-1,0,nx,1,source_iz,source_ix))
        mc=max(mc,_sweep_once(T,slowness,hx,hz,nz-1,-1,-1,nx-1,-1,-1,source_iz,source_ix))
        if mc<tol:
            return T,cyc+1,mc
    return T,maxcycles,mc
