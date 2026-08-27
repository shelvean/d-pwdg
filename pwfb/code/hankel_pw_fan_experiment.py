import numpy as np, pandas as pd
from scipy.special import jv,jvp,hankel1,h1vp
from scipy.optimize import minimize_scalar, least_squares
from pathlib import Path
from numpy.polynomial.legendre import leggauss

K=16.; a=.5; R=1.; h=.44; rc=.75
cent_angles=np.deg2rad(np.arange(22.5,360,45)); centers=np.c_[rc*np.cos(cent_angles),rc*np.sin(cent_angles)]
sources=[0.,.15,.25,.35]; ps=[1,3,5,7,9,13,17,21]
M=35;m=np.arange(-M,M+1);tau2=2*np.pi*h*K*(jv(m,K*h)**2+jvp(m,K*h)**2); w=np.sqrt(tau2/tau2.max())
ng=42; gx,gw=leggauss(ng)

def modal(center,sx):
 V=np.array([sx,0])-center; D=np.linalg.norm(V); ph=np.arctan2(V[1],V[0]);
 if D<=h: return None,D
 return hankel1(m,K*D)*np.exp(-1j*m*ph),D

def PW(th): return (1j**m[:,None])*np.exp(-1j*m[:,None]*th[None,:])
def fit(th,am):
 A=w[:,None]*PW(th); b=w*am; c=np.linalg.lstsq(A,b,rcond=1e-13)[0]; r=A@c-b
 return np.linalg.norm(r)/np.linalg.norm(b),c

def cond_rank(th):
 A=w[:,None]*PW(th); ev=np.linalg.eigvalsh(A.conj().T@A); tol=1e-12*ev.max(); rk=np.sum(ev>tol)
 return (ev.max()/ev[ev>tol].min() if rk else np.inf),int(rk)

def principal(am):
 # optimize one direction starting from local phase by coarse angular dictionary, no geometric truth used
 grid=np.linspace(0,2*np.pi,720,endpoint=False); errs=[]
 for t in grid: errs.append(fit(np.array([t]),am)[0])
 t0=grid[np.argmin(errs)]
 def f(x): return fit(np.array([x%(2*np.pi)]),am)[0]
 r=minimize_scalar(f,bounds=(t0-np.deg2rad(2),t0+np.deg2rad(2)),method='bounded',options={'xatol':1e-13})
 return r.x%(2*np.pi)

def fan(center,beta,p):
 if p==1:return np.array([center])
 return (center+np.linspace(-beta,beta,p))%(2*np.pi)

def exact_u(X,sx): return hankel1(0,K*np.linalg.norm(X-np.array([sx,0]),axis=1))
def sector(j):
 th0=j*np.pi/4;th1=(j+1)*np.pi/4; rr=.75+.25*gx;rw=.25*gw; tt=(th0+th1)/2+(th1-th0)/2*gx;tw=(th1-th0)/2*gw
 RR,TT=np.meshgrid(rr,tt,indexing='ij');W=np.outer(rw,tw)*RR;X=np.c_[(RR*np.cos(TT)).ravel(),(RR*np.sin(TT)).ravel()]
 return X,W.ravel()

rows=[];arr=[]
for sx in sources:
 for j,cK in enumerate(centers):
  am,D=modal(cK,sx)
  if am is None: continue
  th0=principal(am); truth=np.arctan2(cK[1],cK[0]-sx)%(2*np.pi)
  X,W=sector(j); u=exact_u(X,sx); den=np.sqrt(np.sum(W*abs(u)**2))
  for p in ps:
   if p==1: beta=0.; th=np.array([th0]); et,co=fit(th,am)
   else:
    # optimize half-aperture; lower bound keeps distinct directions but allows clustering
    def obj(b): return fit(fan(th0,b,p),am)[0]
    opt=minimize_scalar(obj,bounds=(np.deg2rad(.25*(p-1)),np.deg2rad(100)),method='bounded',options={'xatol':1e-11})
    beta=opt.x;th=fan(th0,beta,p);et,co=fit(th,am)
   gc,rk=cond_rank(th)
   Phi=np.exp(1j*K*(X-cK)@np.c_[np.cos(th),np.sin(th)].T); ev=np.sqrt(np.sum(W*abs(u-Phi@co)**2))/den
   # equispaced with optimized rotation
   base=2*np.pi*np.arange(p)/p
   def er(rot):return fit((base+rot)%(2*np.pi),am)[0]
   op=minimize_scalar(er,bounds=(0,2*np.pi/p),method='bounded');the=(base+op.x)%(2*np.pi);ee,ce=fit(the,am);gce,rke=cond_rank(the)
   Phie=np.exp(1j*K*(X-cK)@np.c_[np.cos(the),np.sin(the)].T); eve=np.sqrt(np.sum(W*abs(u-Phie@ce)**2))/den
   ang=np.degrees(abs(np.angle(np.exp(1j*(th0-truth)))))
   rows += [dict(source_x=sx,element=j,p=p,method='ESPRIT-centered fan',trace_error=et,volume_L2_error=ev,gram_cond=gc,rank=rk,principal_angle_error_deg=ang,half_aperture_deg=np.degrees(beta)),dict(source_x=sx,element=j,p=p,method='equispaced',trace_error=ee,volume_L2_error=eve,gram_cond=gce,rank=rke,principal_angle_error_deg=np.nan,half_aperture_deg=np.nan)]
  arr.append(dict(source_x=sx,element=j,cx=cK[0],cy=cK[1],truth=truth,recovered=th0,angle_error_deg=np.degrees(abs(np.angle(np.exp(1j*(th0-truth)))))))

df=pd.DataFrame(rows)
out=Path(__file__).resolve().parents[1] / 'data'
out.mkdir(exist_ok=True)
df.to_csv(out/'hankel_pw_fan_results.csv',index=False)
pd.DataFrame(arr).to_csv(out/'hankel_pw_fan_arrows.csv',index=False)
agg=df.groupby(['source_x','p','method']).agg(trace_error=('trace_error',lambda x:np.sqrt(np.mean(np.array(x)**2))),volume_L2_error=('volume_L2_error',lambda x:np.sqrt(np.mean(np.array(x)**2))),max_gram_cond=('gram_cond','max'),min_rank=('rank','min'),max_principal_angle_error_deg=('principal_angle_error_deg','max'),mean_half_aperture_deg=('half_aperture_deg','mean')).reset_index();agg.to_csv(out/'hankel_pw_fan_summary.csv',index=False);print(agg.to_string(index=False))
