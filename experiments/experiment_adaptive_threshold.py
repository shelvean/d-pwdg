"""Two-stage direction/mesh adaptive PWDG benchmark on an L-shaped domain.

Author
------
Shelvean Kapita
Department of Mathematics, Texas A&M University

Date
----
August 2026

Purpose
-------
This script is the reference implementation for the prescribed-tolerance
adaptive experiment in the manuscript.  The exact field contains a re-entrant
corner singularity and two exterior cylindrical waves.  The example therefore
contains two distinct approximation difficulties: the local direction set must
represent crossing wave fronts, while mesh refinement is still required near
the singular corner.

The adaptive loop is

    SOLVE -> ESTIMATE -> MARK -> TEST -> ACT -> SOLVE.

ESTIMATE distributes the positive skeleton residual to element indicators.
MARK uses Doerfler marking.  During the direction-learning stage, TEST compares
three possible actions on each marked element:

* MOVE: rotate the current local direction set by +/-15 degrees
* ENRICH: add one of six candidate directions if it is sufficiently separated
  from the existing rays
* REFINE: bisect the marked element and close the mesh conformingly.

After cycle 14 the learned directions are frozen.  The second stage then uses
only residual-driven h-refinement until the relative graph error is below five
percent.  A pure h-adaptive calculation, starting from the same mesh with five
equispaced directions per element, is run for comparison.

The script writes intermediate CSV files and a checkpoint in
``results/adaptive_threshold``.  Checkpointing is useful because the TEST phase
requires several complete trial solves for every marked element.

Run
---
From the top level of the code archive,

    python -m experiments.experiment_adaptive_threshold

The calculation is intentionally expensive.  The CSV files supplied in the
``data`` directory are the reference results reported in the manuscript.
"""

from __future__ import annotations

from pathlib import Path

__author__ = "Shelvean Kapita"
__date__ = "August 2026"

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "adaptive_threshold"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import csv
import math
import os
import pickle
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import h1vp, hankel1, jv, jvp

# Numerical parameters used in the manuscript experiment.
KAPPA = 12.0
ALPHA = BETA = 0.5
THETA_DORFLER = 0.4
MOVE_THR = 0.20
ENRICH_THR = 0.40
MOVE_DEG = 15.0
SEP_DEG = 7.0
P0 = 5
PMAX = 20  # nominal cap; the reported calculation reaches only p_K = 8.
TARGET = 0.05
MAX_CYCLES = 22
NQ = 18
DIRECTION_STAGE_END_CYCLE = 14

# -----------------------------------------------------------------------------
# Exact manufactured solution and derivatives
# -----------------------------------------------------------------------------
x1=np.array([-1.45,0.55])
x2=np.array([0.55,1.45])
def exact_val(x,y):
    x=np.asarray(x)
    y=np.asarray(y)
    r=np.hypot(x,y)
    th=np.arctan2(y,x)
    # on L-shaped domain theta convention [0, 3pi/2] around reentrant corner,
    # with branch cut along positive x axis; map negative theta into [0,2pi)
    th=np.where(th<0, th+2*np.pi, th)
    # removed quadrant is x>=0,y<=0, so physical theta spans [0,3pi/2]
    term=np.zeros_like(r,dtype=complex)
    nz=r>1e-14
    term[nz]=2*jv(2/3,KAPPA*r[nz])*np.sin(2*th[nz]/3)
    # r=0 limiting value 0
    R1=np.hypot(x-x1[0],y-x1[1])
    R2=np.hypot(x-x2[0],y-x2[1])
    return term + 4*hankel1(0,KAPPA*R1)+hankel1(0,KAPPA*R2)

def exact_grad(x,y):
    # analytic Bessel polar gradient + Hankel gradients
    x=np.asarray(x)
    y=np.asarray(y)
    r=np.hypot(x,y)
    th=np.arctan2(y,x)
    th=np.where(th<0,th+2*np.pi,th)
    gx=np.zeros_like(r,dtype=complex)
    gy=np.zeros_like(r,dtype=complex)
    nz=r>1e-10
    rr=r[nz]
    tt=th[nz]
    ur=2*KAPPA*jvp(2/3,KAPPA*rr,1)*np.sin(2*tt/3)
    uth=2*jv(2/3,KAPPA*rr)*(2/3)*np.cos(2*tt/3)
    gx[nz]+=ur*np.cos(tt)-uth*np.sin(tt)/rr
    gy[nz]+=ur*np.sin(tt)+uth*np.cos(tt)/rr
    for amp,src in ((4.0,x1),(1.0,x2)):
        dx=x-src[0]
        dy=y-src[1]
        R=np.hypot(dx,dy)
        fac=amp*KAPPA*h1vp(0,KAPPA*R,1)/R
        gx += fac*dx
        gy += fac*dy
    return np.stack([gx,gy],axis=-1)

class Exact:
    def __call__(self,x,y):
        return exact_val(x,y)
    def wavenumber(self,y):
        return KAPPA*np.ones_like(np.asarray(y,dtype=float))
exact=Exact()

# -----------------------------------------------------------------------------
# Initial L-shaped mesh and mesh-connectivity utilities
# -----------------------------------------------------------------------------
def initial_l_mesh():
    xs=np.linspace(-1,1,5)
    ys=np.linspace(-1,1,5)
    verts=[]
    vid={}
    def V(x,y):
        key=(round(float(x),12),round(float(y),12))
        if key not in vid:
            vid[key]=len(verts)
            verts.append([x,y])
        return vid[key]
    tris=[]
    for j in range(4):
      for i in range(4):
        xc=(xs[i]+xs[i+1])/2
        yc=(ys[j]+ys[j+1])/2
        if xc>=0 and yc<=0:
            continue
        a=V(xs[i],ys[j])
        b=V(xs[i+1],ys[j])
        c=V(xs[i+1],ys[j+1])
        d=V(xs[i],ys[j+1])
        # alternate diagonals mildly to avoid directional bias
        if (i+j)%2==0:
            tris += [[a,b,c],[a,c,d]]
        else:
            tris += [[a,b,d],[b,c,d]]
    return np.array(verts,float), np.array(tris,int)

def tri_area(v,t):
    A=v[t]
    return 0.5*abs(np.cross(A[1]-A[0],A[2]-A[0]))

def edges(t):
    d={}
    for k,T in enumerate(t):
        for a,b in ((T[0],T[1]),(T[1],T[2]),(T[2],T[0])):
            key=tuple(sorted((int(a),int(b))))
            d.setdefault(key,[]).append(k)
    ev=[]
    et=[]
    for key,adj in d.items():
        ev.append(key)
        et.append([adj[0],adj[1] if len(adj)>1 else -1])
    return np.array(ev,int),np.array(et,int)

def edge_map(t):
    d={}
    for k,T in enumerate(t):
        for a,b in ((T[0],T[1]),(T[1],T[2]),(T[2],T[0])):
            d.setdefault(tuple(sorted((int(a),int(b)))),[]).append(k)
    return d

def longest_edge(v,T):
    E=[tuple(sorted((int(T[0]),int(T[1])))),tuple(sorted((int(T[1]),int(T[2])))),tuple(sorted((int(T[2]),int(T[0]))))]
    return max(E,key=lambda e: np.linalg.norm(v[e[1]]-v[e[0]]))

def bisect_conforming(v,t,marked,dirs):
    # Mark the longest edge of every requested triangle. Any triangle incident to a
    # marked edge is subdivided using all of its marked edges, so neighboring
    # elements share the same midpoint and the resulting mesh is conforming.
    split_edges={longest_edge(v,t[int(k)]) for k in marked}
    verts=v.tolist()
    midpoint={}
    for e in split_edges:
        midpoint[e]=len(verts)
        verts.append(((v[e[0]]+v[e[1]])/2).tolist())
    VV=lambda: np.asarray(verts,float)
    newt=[]
    newdirs=[]
    def add(tri,d):
        tri=np.array(tri,int)
        A=VV()[tri]
        if np.cross(A[1]-A[0],A[2]-A[0])<0:
            tri[[1,2]]=tri[[2,1]]
        newt.append(tri.tolist())
        newdirs.append(d.copy())
    for k,T in enumerate(t):
        a,b,c=map(int,T)
        E=[tuple(sorted((a,b))),tuple(sorted((b,c))),tuple(sorted((c,a)))]
        se=[e for e in E if e in split_edges]
        if len(se)==0:
            add([a,b,c],dirs[k])
        elif len(se)==1:
            u,w=se[0]
            z=[q for q in (a,b,c) if q not in (u,w)][0]
            m=midpoint[se[0]]
            add([z,u,m],dirs[k])
            add([z,m,w],dirs[k])
        elif len(se)==3:
            mab=midpoint[tuple(sorted((a,b)))]
            mbc=midpoint[tuple(sorted((b,c)))]
            mca=midpoint[tuple(sorted((c,a)))]
            add([a,mab,mca],dirs[k])
            add([mab,b,mbc],dirs[k])
            add([mca,mbc,c],dirs[k])
            add([mab,mbc,mca],dirs[k])
        else:
            # two split edges share one vertex u; v1,v2 are the other vertices
            e1,e2=se
            common=list(set(e1).intersection(e2))[0]
            others=[q for q in (a,b,c) if q!=common]
            v1,v2=others
            m1=midpoint[tuple(sorted((common,v1)))]
            m2=midpoint[tuple(sorted((common,v2)))]
            add([common,m1,m2],dirs[k])
            add([m1,v1,v2],dirs[k])
            add([m1,v2,m2],dirs[k])
    return np.asarray(verts,float),np.asarray(newt,int),newdirs

# ---- quadrature ----
gs,gw=leggauss(NQ)
gs=(gs+1)/2
gw=gw/2

def q_real(theta):
    return KAPPA*np.array([np.cos(theta),np.sin(theta)],complex)

def build_edges(t):
    return edges(t)

# -----------------------------------------------------------------------------
# PWDG assembly and solution on a general triangular mesh
# -----------------------------------------------------------------------------

class Solver:
    def __init__(self,v,t,dirs):
        self.v=v
        self.t=t
        self.dirs=dirs
        self.ev,self.et=build_edges(t)
        self.nt=len(t)
        self.cent=v[t].mean(axis=1)
        self.nloc=np.array([len(x) for x in dirs])
        self.off=np.r_[0,np.cumsum(self.nloc)]
        self.ndof=int(self.off[-1])
    def sl(self,e):
        return slice(self.off[e],self.off[e+1])
    def basis(self,e,P):
        q=np.array([q_real(th) for th in self.dirs[e]])
        return np.exp(1j*(P-self.cent[e])@q.T),q
    def edgeq(self,f):
        a,b=self.v[self.ev[f]]
        d=b-a
        L=np.linalg.norm(d)
        P=a+gs[:,None]*d
        n=np.array([d[1],-d[0]])/L
        return P,gw*L,n
    def outn(self,e,n,P):
        return n if n@(P.mean(axis=0)-self.cent[e])>0 else -n
    def assemble(self):
        rows=[]
        cols=[]
        vals=[]
        F=np.zeros(self.ndof,complex)
        def push(te,tr,B):
            rr=np.arange(self.off[te],self.off[te+1])
            cc=np.arange(self.off[tr],self.off[tr+1])
            rows.append(np.repeat(rr,len(cc)))
            cols.append(np.tile(cc,len(rr)))
            vals.append(B.ravel())
        for f in range(len(self.et)):
            P,w,n0=self.edgeq(f)
            e0,e1=self.et[f]
            if e1>=0:
                n=self.outn(e0,n0,P)
                for ea,na in ((e0,n),(e1,-n)):
                  pa,qa=self.basis(ea,P)
                  for eb,nb in ((e0,n),(e1,-n)):
                    pb,qb=self.basis(eb,P)
                    G=np.einsum('q,ql,qm->lm',w,pa,np.conj(pb),optimize=True)
                    qan=qa@na
                    qab=qa@nb
                    qbn=np.conj(qb@nb)
                    C=(-0.5j*qbn[None,:]-0.5j*qab[:,None]+1j*(BETA/KAPPA)*qan[:,None]*qbn[None,:]+1j*KAPPA*ALPHA*float(na@nb))
                    push(eb,ea,(C*G).T)
            else:
                e=e0
                n=self.outn(e,n0,P)
                pa,qa=self.basis(e,P)
                qn=qa@n
                G=np.einsum('q,ql,qm->lm',w,pa,np.conj(pa),optimize=True)
                C=(-1j*qn[:,None]+1j*ALPHA*KAPPA)*np.ones((1,len(qn)),complex)
                push(e,e,(C*G).T)
                g=exact_val(P[:,0],P[:,1])
                rhs=np.einsum('q,q,qm->m',w,g,np.conj(pa),optimize=True)
                F[self.sl(e)] += 1j*np.conj(qn)*rhs + 1j*ALPHA*KAPPA*rhs
        A=sp.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(self.ndof,self.ndof)).tocsc()
        self.A=A
        self.F=F
    def solve(self):
        self.assemble()
        self.u=spla.spsolve(self.A,self.F)
        return self
    def uh(self,e,P):
        B,q=self.basis(e,P)
        return B@self.u[self.sl(e)]
    def grad(self,e,P):
        B,q=self.basis(e,P)
        c=self.u[self.sl(e)]
        return (B*(1j*c)[None,:])@q
    def indicators(self):
        eta=np.zeros(self.nt)
        for f in range(len(self.et)):
            P,w,n0=self.edgeq(f)
            e0,e1=self.et[f]
            if e1>=0:
                n=self.outn(e0,n0,P)
                ju=self.uh(e0,P)-self.uh(e1,P)
                jf=self.grad(e0,P)@n+self.grad(e1,P)@(-n)
                z=np.sum(w*(ALPHA*KAPPA*np.abs(ju)**2+BETA/KAPPA*np.abs(jf)**2))
                eta[e0]+=0.5*z
                eta[e1]+=0.5*z
            else:
                r=self.uh(e0,P)-exact_val(P[:,0],P[:,1])
                eta[e0]+=np.sum(w*ALPHA*KAPPA*np.abs(r)**2)
        return eta

def solve(v,t,dirs):
    return Solver(v,t,dirs).solve()

def dorfer(eta):
    order=np.argsort(-eta)
    target=THETA_DORFLER*eta.sum()
    s=0
    out=[]
    for k in order:
        out.append(int(k))
        s+=eta[k]
        if s>=target:
            break
    return out

def graph_norm_exact_boundary(v,t):
    # normalization sqrt(alpha*k ||g||^2_boundary), exactly the only boundary trace term of exact all-Dirichlet field
    ev,et=edges(t)
    val=0
    for f,(e0,e1) in enumerate(et):
        if e1>=0:
            continue
        a,b=v[ev[f]]
        d=b-a
        L=np.linalg.norm(d)
        P=a+gs[:,None]*d
        w=gw*L
        g=exact_val(P[:,0],P[:,1])
        val+=np.sum(w*ALPHA*KAPPA*np.abs(g)**2)
    return math.sqrt(val)

def metrics(s):
    eta=s.indicators()
    J=float(eta.sum())
    return eta,J,math.sqrt(J)

def min_sep(angle,arr):
    if len(arr)==0:
        return 999
    d=np.abs(np.angle(np.exp(1j*(np.asarray(arr)-angle))))
    return np.degrees(d.min())

# -----------------------------------------------------------------------------
# Two-stage hybrid adaptive loop
# -----------------------------------------------------------------------------

def hybrid_run():
    ck=str(OUTPUT_DIR/'hybrid_checkpoint.pkl')
    if os.path.exists(ck):
        with open(ck,'rb') as f:
            state=pickle.load(f)
        v,t,dirs,hist,norm,startcyc=state
        print('RESUME HYBRID at',startcyc,'nelem',len(t),'ndof',sum(map(len,dirs)),flush=True)
    else:
        v,t=initial_l_mesh()
        base=2*np.pi*np.arange(P0)/P0
        dirs=[base.copy() for _ in range(len(t))]
        norm=graph_norm_exact_boundary(v,t)
        hist=[]
        startcyc=0
    for cyc in range(startcyc,MAX_CYCLES+1):
        tic=time.time()
        s=solve(v,t,dirs)
        eta,J,ge=metrics(s)
        rel=ge/norm
        rec=dict(cycle=cyc,nelem=len(t),ndof=s.ndof,J=J,graph_error=ge,relative_graph_error=rel,mean_p=float(np.mean([len(x) for x in dirs])),max_p=max(map(len,dirs)),solve_sec=time.time()-tic)
        print('HYB',rec,flush=True)
        hist.append(rec)
        save(str(OUTPUT_DIR/'hybrid_threshold_partial.csv'),hist)
        if rel<=TARGET or cyc==MAX_CYCLES:
            if os.path.exists(ck):
                os.remove(ck)
            break
        marked=dorfer(eta)

        # Stage 2: after the direction-learning phase, keep the learned local
        # directions fixed and continue with ordinary residual-driven h-refinement.
        # This is the two-stage strategy used for the final threshold comparison.
        if cyc >= DIRECTION_STAGE_END_CYCLE:
            v,t,dirs=bisect_conforming(v,t,marked,dirs)
            hist[-1].update(marked=len(marked),move=0,enrich=0,refine=len(marked))
            save(str(OUTPUT_DIR/'hybrid_threshold_partial.csv'),hist)
            with open(ck,'wb') as f:
                pickle.dump((v,t,dirs,hist,norm,cyc+1),f)
            continue

        actions={}
        for K in marked:
            bestmove=(J,None)
            besten=(J,None)
            for deg in (-MOVE_DEG,MOVE_DEG):
                td=[x.copy() for x in dirs]
                td[K]=(td[K]+np.radians(deg))%(2*np.pi)
                try:
                    ss=solve(v,t,td)
                    _,Jt,_=metrics(ss)
                    if Jt<bestmove[0]:
                        bestmove=(Jt,td[K])
                except Exception:
                    pass
            if len(dirs[K])<PMAX:
                for ang in 2*np.pi*np.arange(6)/6:
                    if min_sep(ang,dirs[K])<SEP_DEG:
                        continue
                    td=[x.copy() for x in dirs]
                    td[K]=np.r_[td[K],ang]
                    try:
                        ss=solve(v,t,td)
                        _,Jt,_=metrics(ss)
                        if Jt<besten[0]:
                            besten=(Jt,td[K])
                    except Exception:
                        pass
            gm=(J-bestmove[0])/max(eta[K],1e-30)
            gei=(J-besten[0])/max(eta[K],1e-30)
            if gm>=MOVE_THR or gei>=ENRICH_THR:
                if gm>=MOVE_THR and gm>=gei:
                    actions[K]=('MOVE',bestmove[1],gm)
                else:
                    actions[K]=('ENRICH',besten[1],gei)
            else:
                actions[K]=('REFINE',None,0)
        newdirs=[x.copy() for x in dirs]
        ref=[]
        for K,(a,d,gain) in actions.items():
            if a in ('MOVE','ENRICH') and d is not None:
                newdirs[K]=d
            else:
                ref.append(K)
        if not ref or len(ref)<len(marked):
            try:
                st=solve(v,t,newdirs)
                _,Jbatch,_=metrics(st)
                if Jbatch>J:
                    ref=list(marked)
                    newdirs=[x.copy() for x in dirs]
            except Exception:
                ref=list(marked)
                newdirs=[x.copy() for x in dirs]
        dirs=newdirs
        if ref:
            v,t,dirs=bisect_conforming(v,t,ref,dirs)
        hist[-1].update(marked=len(marked),move=sum(a[0]=='MOVE' for a in actions.values()),enrich=sum(a[0]=='ENRICH' for a in actions.values()),refine=sum(a[0]=='REFINE' for a in actions.values()))
        save(str(OUTPUT_DIR/'hybrid_threshold_partial.csv'),hist)
        with open(ck,'wb') as f:
            pickle.dump((v,t,dirs,hist,norm,cyc+1),f)
        if s.ndof>7000:
            print('hybrid dof safety stop')
            break
    return hist,norm

# -----------------------------------------------------------------------------
# Pure h-adaptive reference calculation
# -----------------------------------------------------------------------------

def pure_h_run(norm):
    v,t=initial_l_mesh()
    base=2*np.pi*np.arange(P0)/P0
    dirs=[base.copy() for _ in range(len(t))]
    hist=[]
    for cyc in range(MAX_CYCLES+5):
        tic=time.time()
        s=solve(v,t,dirs)
        eta,J,ge=metrics(s)
        rel=ge/norm
        rec=dict(cycle=cyc,nelem=len(t),ndof=s.ndof,J=J,graph_error=ge,relative_graph_error=rel,solve_sec=time.time()-tic)
        print('PURE',rec,flush=True)
        hist.append(rec)
        if rel<=TARGET:
            break
        marked=dorfer(eta)
        v,t,dirs=bisect_conforming(v,t,marked,dirs)
        if s.ndof>15000:
            print('pure dof safety stop')
            break
    return hist

def save(name,hist):
    keys=[]
    for r in hist:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(name,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        w.writerows(hist)

if __name__=='__main__':
    print('initial triangles',len(initial_l_mesh()[1]))
    h,norm=hybrid_run()
    p=pure_h_run(norm)
    save(str(OUTPUT_DIR/'hybrid_threshold.csv'),h)
    save(str(OUTPUT_DIR/'pure_h_threshold.csv'),p)
    print('NORMALIZER',norm,'TARGET',TARGET)
