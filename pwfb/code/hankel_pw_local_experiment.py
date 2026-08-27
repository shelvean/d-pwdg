import numpy as np
from scipy.special import jv,jvp,hankel1,h1vp
from scipy.optimize import least_squares
from scipy.linalg import svd
import pandas as pd
from pathlib import Path

K=16.0; a=0.5; R=1.0; h=0.44
cent_angles=np.deg2rad(np.arange(22.5,360,45.0)); rc=0.75
centers=np.c_[rc*np.cos(cent_angles),rc*np.sin(cent_angles)]
sources=[0.0,0.15,0.25,0.35]
ps=[1,3,5,7,9,11]
Mfit=30
mfit=np.arange(-Mfit,Mfit+1)
tau2=2*np.pi*h*K*(jv(mfit,K*h)**2+jvp(mfit,K*h)**2)
# normalized weights; very small modes contribute negligibly to trace
w=np.sqrt(tau2/tau2.max())

# volume quadrature in exact annular sectors: tensor Gauss in r,theta with area r dr dtheta
from numpy.polynomial.legendre import leggauss
ng=36; gx,gw=leggauss(ng)

def sector_points(j):
    th0=j*np.pi/4; th1=(j+1)*np.pi/4
    rr=(a+R)/2+(R-a)/2*gx; rw=(R-a)/2*gw
    tt=(th0+th1)/2+(th1-th0)/2*gx; tw=(th1-th0)/2*gw
    RR,TT=np.meshgrid(rr,tt,indexing='ij'); WW=np.outer(rw,tw)*RR
    X=np.c_[ (RR*np.cos(TT)).ravel(), (RR*np.sin(TT)).ravel()]
    return X, WW.ravel()

def exact_u_grad(X,sx):
    S=np.array([sx,0.0]); V=X-S; r=np.linalg.norm(V,axis=1)
    u=hankel1(0,K*r)
    # d/dr H0 = -H1; use h1vp robustly
    ur=K*h1vp(0,K*r,1)
    g=(ur/r)[:,None]*V
    return u,g

def modal_coeff(center,sx):
    # Graf expansion valid when source lies outside local disk
    V=np.array([sx,0.0])-center; D=np.linalg.norm(V); phi=np.arctan2(V[1],V[0])
    if D <= h: return None,D
    # H0(|rho eitheta - D eiphi|)= sum H_m(kD) J_m(krho) exp(i m(theta-phi))
    am=hankel1(mfit,K*D)*np.exp(-1j*mfit*phi)
    return am,D

def pw_modal(theta):
    # columns modal coefficients of exp(ik d(theta).(x-center))
    return (1j**mfit[:,None])*np.exp(-1j*mfit[:,None]*theta[None,:])

def fit_coeff(theta,am):
    A=w[:,None]*pw_modal(theta); b=w*am
    c=np.linalg.lstsq(A,b,rcond=1e-13)[0]
    return c

def trace_error(theta,am):
    c=fit_coeff(theta,am); r=w*(am-pw_modal(theta)@c)
    den=np.linalg.norm(w*am)
    return np.linalg.norm(r)/den,c

def esprit_init(am,q):
    # Use a central consecutive modal window of length >= 2q+3 but avoid tiny trace modes.
    gamma=am/(1j**mfit)
    # select modes with trace weights above 1e-10 and centered contiguous block
    good=np.where(w>1e-7)[0]
    lo,hi=good[0],good[-1]
    # guarantee enough samples by expanding if necessary
    need=2*q+3
    if hi-lo+1 < need:
        mid=len(mfit)//2; lo=max(0,mid-need//2); hi=min(len(mfit)-1,lo+need-1); lo=hi-need+1
    seq=gamma[lo:hi+1]
    N=len(seq); L=N//2+1; KK=N-L+1
    H=np.empty((L,KK),complex)
    for i in range(L): H[i,:]=seq[i:i+KK]
    U,S,Vh=svd(H,full_matrices=False)
    Uq=U[:,:q]
    U0=Uq[:-1,:]; U1=Uq[1:,:]
    z=np.linalg.eigvals(np.linalg.lstsq(U0,U1,rcond=1e-13)[0])
    theta=(-np.angle(z))%(2*np.pi)
    return np.sort(theta), (S[0]/S[q-1] if S[q-1]>0 else np.inf), np.abs(z)

def optimize(theta0,am):
    q=len(theta0)
    # variable projection residual as real vector; angles unconstrained modulo 2pi
    def fun(th):
        th=np.mod(th,2*np.pi); A=w[:,None]*pw_modal(th); b=w*am
        c=np.linalg.lstsq(A,b,rcond=1e-13)[0]; r=A@c-b
        return np.r_[r.real,r.imag]
    res=least_squares(fun,theta0,method='trf',max_nfev=500,xtol=1e-12,ftol=1e-12,gtol=1e-12)
    th=np.mod(res.x,2*np.pi); err,c=trace_error(th,am)
    return th,c,err,res.nfev

def gram_cond(theta):
    A=w[:,None]*pw_modal(theta)
    G=A.conj().T@A
    ev=np.linalg.eigvalsh(G)
    tol=1e-12*ev.max()
    rk=int(np.sum(ev>tol)); cond=(ev.max()/ev[ev>tol].min()) if rk else np.inf
    return cond,rk

rows=[]; arrow_rows=[]
for sx in sources:
  for j,cK in enumerate(centers):
    am,D=modal_coeff(cK,sx)
    if am is None:
      print('invalid local disk source',sx,j,D); continue
    exactang=np.arctan2(cK[1],cK[0]-sx)%(2*np.pi)
    for p in ps:
      th0,kH,mods=esprit_init(am,p)
      th,c,et,nfev=optimize(th0,am)
      cond,rk=gram_cond(th)
      # equispaced with best global rotation optimized as 1D for fairness
      base=2*np.pi*np.arange(p)/p
      best=(1e99,None,None)
      for rot in np.linspace(0,2*np.pi/p,73,endpoint=False):
        ee,cc=trace_error((base+rot)%(2*np.pi),am)
        if ee<best[0]: best=(ee,(base+rot)%(2*np.pi),cc)
      ee,the,ce=best; conde,rke=gram_cond(the)
      # volume L2 errors on sector
      X,W=sector_points(j); u,_=exact_u_grad(X,sx)
      den=np.sqrt(np.sum(W*np.abs(u)**2))
      def vol(thv,cv):
        Phi=np.exp(1j*K*(X-centers[j])@np.c_[np.cos(thv),np.sin(thv)].T)
        return np.sqrt(np.sum(W*np.abs(u-Phi@cv)**2))/den
      ev=vol(th,c); eve=vol(the,ce)
      # closest recovered direction to geometric local phase
      dang=np.min(np.abs(np.angle(np.exp(1j*(th-exactang)))))
      rows.append(dict(source_x=sx,element=j,p=p,method='ESPRIT+VP',trace_error=et,volume_L2_error=ev,
                       gram_cond=cond,rank=rk,angle_error_deg=np.degrees(dang),esprit_hankel_cond=kH,nfev=nfev))
      rows.append(dict(source_x=sx,element=j,p=p,method='equispaced',trace_error=ee,volume_L2_error=eve,
                       gram_cond=conde,rank=rke,angle_error_deg=np.nan,esprit_hankel_cond=np.nan,nfev=0))
      if p==3:
        arrow_rows.append(dict(source_x=sx,element=j,cx=cK[0],cy=cK[1],exact_angle=exactang,
                               recovered_angle=th[np.argmin(np.abs(np.angle(np.exp(1j*(th-exactang)))))]))

df=pd.DataFrame(rows)
out=Path(__file__).resolve().parents[1] / 'data'
out.mkdir(exist_ok=True)
df.to_csv(out/'hankel_pw_local_results.csv',index=False)
pd.DataFrame(arrow_rows).to_csv(out/'hankel_pw_arrows.csv',index=False)
# aggregate RMS errors across elements; geometric mean cond, max angle
agg=df.groupby(['source_x','p','method']).agg(trace_error=('trace_error',lambda x: np.sqrt(np.mean(np.asarray(x)**2))),
    volume_L2_error=('volume_L2_error',lambda x: np.sqrt(np.mean(np.asarray(x)**2))),
    max_gram_cond=('gram_cond','max'),min_rank=('rank','min'),max_angle_error_deg=('angle_error_deg','max')).reset_index()
agg.to_csv(out/'hankel_pw_local_summary.csv',index=False)
print(agg.to_string(index=False))
