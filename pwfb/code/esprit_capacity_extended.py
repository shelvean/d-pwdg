"""Extended finite-direction capacity experiment for direct target-frequency ESPRIT.

The first 20 rays and phases are exactly those of Kapita (2026), arXiv:2608.18380.
The sequence is extended deterministically by a circular maximin rule.  A fixed
window of N=401 consecutive modal coefficients gives a balanced 201x201 Hankel
matrix and admits ranks q <= 200 by sample count.  The experiment therefore
separates the old residual-search failure at q=20 from the eventual numerical
conditioning limit of direct ESPRIT in IEEE double precision.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from scipy.linalg import hankel, svd
from scipy.optimize import linear_sum_assignment

plt.rcParams.update({
    "pgf.texsystem":"pdflatex", "pgf.rcfonts":False,
    "font.family":"serif", "text.usetex":True,
    "pgf.preamble":r"\usepackage{amsmath,amssymb}",
    "font.size":9, "axes.labelsize":9, "xtick.labelsize":8,
    "ytick.labelsize":8, "axes.titlesize":9,
})
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"figures"; OUT.mkdir(exist_ok=True)
DATA=ROOT/"data"; DATA.mkdir(exist_ok=True)
N=401
OLD_DIR=np.array([17,123,251,68,191,315,39,158,286,340,92,224,5,145,273,327,54,178,235,301],float)
OLD_PHASE=np.array([0,.37,-.51,.81,-1.04,1.33,-1.52,.22,1.71,-.93,.58,-1.22,1.48,-.31,.96,-1.73,.11,1.18,-.77,1.91],float)

def extend_maximin(nmax=200, step=.05):
    d=list(OLD_DIR); cand=np.arange(0.,360.,step)
    while len(d)<nmax:
        a=np.asarray(d)
        delta=np.abs(cand[:,None]-a[None,:]) % 360.
        sep=np.minimum(delta,360.-delta).min(axis=1)
        j=int(np.argmax(sep)); d.append(float(cand[j])); cand=np.delete(cand,j)
    return np.asarray(d)
D=extend_maximin(200)
P=np.empty(200); P[:20]=OLD_PHASE
j=np.arange(20,200); P[20:]=np.mod(.731*j+.173*j*j,2*np.pi)-np.pi
C=np.exp(1j*P)

def solve(q):
    z=np.exp(-1j*np.deg2rad(D[:q])); m=np.arange(N)
    g=np.sum(C[:q,None]*z[:,None]**m[None,:],axis=0)
    L=(N+1)//2; K=N-L+1
    H=hankel(g[:L],g[L-1:L+K-1])
    U,s,_=svd(H,full_matrices=False,lapack_driver="gesvd")
    Uq=U[:,:q]
    S=np.linalg.lstsq(Uq[:-1,:],Uq[1:,:],rcond=None)[0]
    nodes=np.linalg.eigvals(S)
    est=(-np.angle(nodes)*180/np.pi)%360.; tru=D[:q]%360.
    dd=np.abs(est[:,None]-tru[None,:])%360.; dd=np.minimum(dd,360.-dd)
    rr,cc=linear_sum_assignment(dd); e=dd[rr,cc]
    return float(e.max()),float(e.mean()),float(s[q-1]/s[0]),float(np.abs(nodes).min()),float(np.abs(nodes).max())

# Dense enough for a smooth capacity plot, with every point near breakdown.
Q=list(range(1,171,2)) + list(range(172,201))
Q=sorted(set(Q+[19,20,40,80,100,120,140,160,170]))
rows=[]
for q in Q:
    r=solve(q); rows.append((q,)+r)
    if q in [19,20,40,80,120,140,160,170,174,180,184,188,192,194,196,197,198,199,200]:
        print(q, r)

csv_path=DATA/"esprit_capacity_N401_reproduced.csv"
with csv_path.open("w",newline="") as f:
    w=csv.writer(f); w.writerow(["q","max_angle_error_deg","mean_angle_error_deg","sigma_q_over_sigma_1","node_modulus_min","node_modulus_max"]); w.writerows(rows)

QARR=np.array([r[0] for r in rows]); err=np.array([r[1] for r in rows]); sig=np.array([r[3] for r in rows]); kap=1.0/sig
fig,(a,b)=plt.subplots(1,2,figsize=(6.2,2.55))
a.semilogy(QARR,err,"k.-",lw=.85,ms=2.8); a.axvline(19,color="0.55",ls="--",lw=.8)
a.set_xlabel(r"number of rays $q$"); a.set_ylabel(r"maximum angle error (degrees)"); a.set_title(r"(a) direct ESPRIT, $N=401$")
a.set_xlim(1,201); a.set_xticks([1,40,80,120,160,200]); a.grid(True,which="major",lw=.3,color="0.85")
a.text(18,2e-2,r"residual path: 19",rotation=90,ha="right",va="bottom",fontsize=7.5,color="0.35")
# mark onset and breakdown
for q,lab in [(174,r"$174$"),(196,r"$196$"),(198,r"$198$")]:
    idx=np.where(QARR==q)[0][0]; a.plot(QARR[idx],err[idx],"ko",ms=3.4); a.annotate(lab,(QARR[idx],err[idx]),xytext=(4,5),textcoords="offset points",fontsize=7)
b.semilogy(QARR,kap,"k.-",lw=.85,ms=2.8); b.axvline(19,color="0.55",ls="--",lw=.8)
b.set_xlabel(r"number of rays $q$"); b.set_ylabel(r"$\kappa_H=\sigma_1(H)/\sigma_q(H)$"); b.set_title(r"(b) signal-Hankel conditioning")
b.set_xlim(1,201); b.set_xticks([1,40,80,120,160,200]); b.grid(True,which="major",lw=.3,color="0.85")
for q in [174,196,198]:
    idx=np.where(QARR==q)[0][0]; b.plot(QARR[idx],kap[idx],"ko",ms=3.4)
fig.subplots_adjust(left=.11,right=.985,bottom=.20,top=.88,wspace=.34)
fig.savefig(OUT/"esprit_capacity_N401_reproduced.pdf")
plt.close(fig)
