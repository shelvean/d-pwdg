from pathlib import Path
import csv, time
import numpy as np
from scipy.linalg import hankel, svd
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
OLD_DIR=np.array([17,123,251,68,191,315,39,158,286,340,92,224,5,145,273,327,54,178,235,301],float)
OLD_PHASE=np.array([0,.37,-.51,.81,-1.04,1.33,-1.52,.22,1.71,-.93,.58,-1.22,1.48,-.31,.96,-1.73,.11,1.18,-.77,1.91],float)
def cd(a,b):
 d=np.abs(a-b)%360.; return np.minimum(d,360.-d)
def extend(nmax=1000,step=.05):
 d=list(OLD_DIR); cand=np.arange(0.,360.,step); md=np.full(cand.shape,np.inf); used=np.zeros(cand.shape,bool)
 for x in d: md=np.minimum(md,cd(cand,x))
 for x in d:
  j=int(np.argmin(cd(cand,x)))
  if cd(cand[j],x)<1e-12: used[j]=True; md[j]=-1
 while len(d)<nmax:
  j=int(np.argmax(md)); x=float(cand[j]); d.append(x); used[j]=True; md[j]=-1; md=np.minimum(md,cd(cand,x)); md[used]=-1
 return np.asarray(d)
D=extend(1100); P=np.empty(1100); P[:20]=OLD_PHASE; jj=np.arange(20,1100); P[20:]=np.mod(.731*jj+.173*jj*jj,2*np.pi)-np.pi; C=np.exp(1j*P)
def solve(N,q):
 z=np.exp(-1j*np.deg2rad(D[:q])); m=np.arange(N); g=np.sum(C[:q,None]*z[:,None]**m[None,:],axis=0)
 L=(N+1)//2; K=N-L+1; H=hankel(g[:L],g[L-1:L+K-1]); U,s,_=svd(H,full_matrices=False,lapack_driver='gesvd'); Uq=U[:,:q]
 rownorm2=float(np.vdot(Uq[-1,:],Uq[-1,:]).real); minsv=np.sqrt(max(0.,1.-rownorm2)); kshift=np.inf if minsv==0 else 1/minsv
 S=np.linalg.lstsq(Uq[:-1,:],Uq[1:,:],rcond=None)[0]; nodes=np.linalg.eigvals(S)
 est=(-np.angle(nodes)*180/np.pi)%360.; tru=D[:q]%360.; dd=np.abs(est[:,None]-tru[None,:])%360.; dd=np.minimum(dd,360.-dd); rr,cc=linear_sum_assignment(dd); e=dd[rr,cc]
 rat=float(s[q-1]/s[0]); return [N,q,float(e.max()),float(e.mean()),rat,float(1/rat),float(kshift),float(np.abs(nodes).min()),float(np.abs(nodes).max())]
cases=[(2201,940)]
rows=[]
for N,q in cases:
 t=time.time(); r=solve(N,q); rows.append(r+[time.time()-t]); print(r,flush=True)
with (DATA/'esprit_conditioning_selected_part7_reproduced.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['N','q','max_angle_error_deg','mean_angle_error_deg','sigma_q_over_sigma_1','kappa_H','kappa_shift','node_modulus_min','node_modulus_max','seconds']); w.writerows(rows)
