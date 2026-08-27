"""Conditioning-aware long-window ESPRIT capacity experiment.

Reproduces the selected rows in Table 'Conditioning-aware continuation of direct
ESPRIT'.  The first 20 directions/phases are those of Kapita (2026); subsequent
rays are added deterministically by circular maximin selection on a 0.05-degree
grid.  All calculations are IEEE double precision.

Outputs: esprit_conditioning_long_windows.csv
"""
from pathlib import Path
import csv
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

def extend_maximin(nmax=1000,step=.05):
    directions=list(OLD_DIR)
    cand=np.arange(0.,360.,step)
    mind=np.full(cand.shape,np.inf)
    used=np.zeros(cand.shape,dtype=bool)
    for x in directions:
        mind=np.minimum(mind,circdist(cand,x))
        j=int(np.argmin(circdist(cand,x)))
        if circdist(cand[j],x)<1e-12:
            used[j]=True; mind[j]=-1
    while len(directions)<nmax:
        j=int(np.argmax(mind)); x=float(cand[j])
        directions.append(x); used[j]=True; mind[j]=-1
        mind=np.minimum(mind,circdist(cand,x)); mind[used]=-1
    return np.asarray(directions)

D=extend_maximin(1000)
P=np.empty(1000); P[:20]=OLD_PHASE
j=np.arange(20,1000)
P[20:]=np.mod(.731*j+.173*j*j,2*np.pi)-np.pi
C=np.exp(1j*P)

def solve(N,M):
    z=np.exp(-1j*np.deg2rad(D[:M])); m=np.arange(N)
    g=np.sum(C[:M,None]*z[:,None]**m[None,:],axis=0)
    L=(N+1)//2; K=N-L+1
    H=hankel(g[:L],g[L-1:L+K-1])
    U,s,_=svd(H,full_matrices=False,lapack_driver='gesvd')
    UM=U[:,:M]
    # U_M[:-1]^* U_M[:-1] = I-r^*r, r = last row.
    rownorm2=float(np.vdot(UM[-1,:],UM[-1,:]).real)
    sigma_shift_min=np.sqrt(max(0.,1.-rownorm2))
    kappa_shift=np.inf if sigma_shift_min==0 else 1./sigma_shift_min
    S=np.linalg.lstsq(UM[:-1,:],UM[1:,:],rcond=None)[0]
    nodes=np.linalg.eigvals(S)
    est=(-np.angle(nodes)*180/np.pi)%360.; true=D[:M]%360.
    d=np.abs(est[:,None]-true[None,:])%360.; d=np.minimum(d,360.-d)
    rr,cc=linear_sum_assignment(d); e=d[rr,cc]
    ratio=float(s[M-1]/s[0])
    return [N,M,float(e.max()),float(e.mean()),ratio,float(1./ratio),float(kappa_shift),
            float(np.abs(nodes).min()),float(np.abs(nodes).max())]

CASES=[
 (1001,400),(1001,450),(1001,470),(1001,480),
 (1401,500),(1401,580),(1401,600),
 (1601,600),
 (1801,760),
 (2001,800),(2001,880),(2001,900),
 (2201,900),(2201,920),(2201,940),
]
rows=[]
for case in CASES:
    row=solve(*case); rows.append(row); print(row,flush=True)
with (DATA/'esprit_conditioning_long_windows_reproduced.csv').open('w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['N','q','max_angle_error_deg','mean_angle_error_deg','sigma_q_over_sigma_1',
                'kappa_H','kappa_shift','node_modulus_min','node_modulus_max'])
    w.writerows(rows)
