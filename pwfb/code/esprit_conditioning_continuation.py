"""Conditioning-aware continuation of the direct ESPRIT capacity experiment.

Extends the deterministic nested ray family used in the manuscript and records
both recovery accuracy and the conditioning quantities entering the ESPRIT
stability theorem.  The principal condition number is
    kappa_H = sigma_1(H) / sigma_M(H),
for the rank-M signal Hankel matrix.  We also record the condition number of the
shift least-squares matrix U_M[:-1,:].
"""
from pathlib import Path
import csv, time
import numpy as np
from scipy.linalg import hankel, svd
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
OLD_DIR=np.array([17,123,251,68,191,315,39,158,286,340,92,224,5,145,273,327,54,178,235,301],float)
OLD_PHASE=np.array([0,.37,-.51,.81,-1.04,1.33,-1.52,.22,1.71,-.93,.58,-1.22,1.48,-.31,.96,-1.73,.11,1.18,-.77,1.91],float)

def circdist(a,b):
    d=np.abs(a-b)%360.0
    return np.minimum(d,360.0-d)

def extend_maximin(nmax=700, step=.05):
    d=list(OLD_DIR)
    cand=np.arange(0.,360.,step)
    # remove exact grid locations already used when present
    md=np.full(cand.shape, np.inf)
    for x in d:
        md=np.minimum(md,circdist(cand,x))
    used=np.zeros(cand.shape,dtype=bool)
    for x in d:
        j=int(np.argmin(circdist(cand,x)))
        if circdist(cand[j],x)<1e-12: used[j]=True; md[j]=-1
    while len(d)<nmax:
        j=int(np.argmax(md)); x=float(cand[j]); d.append(x); used[j]=True; md[j]=-1
        md=np.minimum(md,circdist(cand,x)); md[used]=-1
    return np.asarray(d)

MAXM=820
D=extend_maximin(MAXM)
P=np.empty(MAXM); P[:20]=OLD_PHASE
j=np.arange(20,MAXM); P[20:]=np.mod(.731*j+.173*j*j,2*np.pi)-np.pi
C=np.exp(1j*P)

def solve(N,q):
    z=np.exp(-1j*np.deg2rad(D[:q])); m=np.arange(N)
    g=np.sum(C[:q,None]*z[:,None]**m[None,:],axis=0)
    L=(N+1)//2; K=N-L+1
    if q>min(K,L-1):
        return dict(status='sample_limit',N=N,M=q,L=L,K=K)
    H=hankel(g[:L],g[L-1:L+K-1])
    U,s,_=svd(H,full_matrices=False,lapack_driver='gesvd')
    Uq=U[:,:q]
    A=Uq[:-1,:]
    # Since Uq has orthonormal columns, A^*A=I-r^*r with r=last row.
    rownorm2=float(np.vdot(Uq[-1,:],Uq[-1,:]).real)
    minsv_shift=np.sqrt(max(0.0,1.0-rownorm2))
    kappa_shift=np.inf if minsv_shift==0 else 1.0/minsv_shift
    S=np.linalg.lstsq(A,Uq[1:,:],rcond=None)[0]
    nodes=np.linalg.eigvals(S)
    est=(-np.angle(nodes)*180/np.pi)%360.; tru=D[:q]%360.
    dd=np.abs(est[:,None]-tru[None,:])%360.; dd=np.minimum(dd,360.-dd)
    rr,cc=linear_sum_assignment(dd); e=dd[rr,cc]
    ratio=float(s[q-1]/s[0]); kH=np.inf if ratio==0 else 1.0/ratio
    return dict(status='valid',N=N,M=q,L=L,K=K,
                max_angle_error_deg=float(e.max()),mean_angle_error_deg=float(e.mean()),
                sigma_q_over_sigma_1=ratio,kappa_H=kH,kappa_shift=float(kappa_shift),
                node_modulus_min=float(np.abs(nodes).min()),node_modulus_max=float(np.abs(nodes).max()))

cases=[]
for q in [700]: cases.append((1401,q))
for q in [600,620,640,660,680,700,720,740,750,760,770,780,790,795,798,800]: cases.append((1601,q))
rows=[]
for N,q in cases:
    t=time.time(); r=solve(N,q); r['seconds']=time.time()-t; rows.append(r)
    print(r, flush=True)

cols=['N','M','L','K','status','max_angle_error_deg','mean_angle_error_deg','sigma_q_over_sigma_1','kappa_H','kappa_shift','node_modulus_min','node_modulus_max','seconds']
with (DATA/'esprit_conditioning_continuation_part4_reproduced.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({k:r.get(k,'') for k in cols})
