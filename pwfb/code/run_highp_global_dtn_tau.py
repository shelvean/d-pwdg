from pathlib import Path
import numpy as np
from scipy.special import jv, jvp, hankel1, h1vp
from numpy.polynomial.legendre import leggauss
from scipy.linalg import block_diag, eigh, cholesky, solve_triangular, svdvals
import pandas as pd, time

ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'data').mkdir(exist_ok=True)

k=16.0; a=0.5; R=1.0; ne=8; alpha=beta=delta=0.5; h=0.44; rc=0.75; tau=float(__import__('os').environ.get('TAU','1e-12')); NdtN=70
plist=[97]

def dirs(p):
    th=2*np.pi*np.arange(p)/p
    return np.c_[np.cos(th),np.sin(th)]

def centers():
    th=(np.arange(ne)+0.5)*2*np.pi/ne
    return rc*np.c_[np.cos(th),np.sin(th)]
CEN=centers()

def disk_gram(p):
    # exact circulant Gram using pairwise angle differences and quadrature modal formula
    d=dirs(p)
    # Direct boundary periodic quadrature stable for Gram entries, then Hermitian eig.
    nq=max(2048,32*p)
    t=2*np.pi*np.arange(nq)/nq
    x=h*np.c_[np.cos(t),np.sin(t)]
    n=np.c_[np.cos(t),np.sin(t)]
    V=np.exp(1j*k*(x@d.T))
    DN=1j*k*(n@d.T)*V
    w=h*2*np.pi/nq
    G=w*(k*(V.conj().T@V)+(1/k)*(DN.conj().T@DN))
    G=(G+G.conj().T)/2
    return G

def local_T(p):
    G=disk_gram(p)
    diag=np.sqrt(np.real(np.diag(G)))
    D=np.diag(1/diag)
    Gn=D@G@D
    lam,U=eigh(Gn)
    lmax=lam.max(); keep=lam>tau*lmax
    Tk=D@U[:,keep]@np.diag(1/np.sqrt(lam[keep]))
    Gf=Tk.conj().T@G@Tk
    s=svdvals(Gf)
    return Tk, keep.sum(), lam, s[0]/s[-1], np.linalg.cond(G)

def edge_quad_line(x0,x1,nq=60):
    z,w=leggauss(nq); pts=.5*((1-z)[:,None]*x0+(1+z)[:,None]*x1); ww=.5*np.linalg.norm(x1-x0)*w
    return pts,ww

def arc_quad(r,t0,t1,nq=80):
    z,w=leggauss(nq); th=.5*((t1-t0)*z+(t1+t0)); ww=.5*(t1-t0)*r*w
    pts=r*np.c_[np.cos(th),np.sin(th)]
    return th,pts,ww

def basis_vals(K,p,x):
    d=dirs(p); ph=np.exp(1j*k*((x-CEN[K])@d.T)); grad=1j*k*ph[:,:,None]*d[None,:,:]
    return ph,grad

def assemble_raw(p,nq_line=60,nq_arc=80):
    N=ne*p; Kmat=np.zeros((N,N),complex); f=np.zeros(N,complex)
    # interior radial edges at theta=2pi*j/ne, between left sector j-1 and right sector j
    for j in range(ne):
        th=2*np.pi*j/ne
        x0=a*np.array([np.cos(th),np.sin(th)]); x1=R*np.array([np.cos(th),np.sin(th)])
        x,w=edge_quad_line(x0,x1,nq_line)
        KL=(j-1)%ne; KR=j
        # normal from KL (sector before boundary) to KR (CCW): -e_theta? Check KL upper edge -> outward +e_theta, actually sector KL spans prev to th, outward at upper = +e_theta.
        nL=np.array([-np.sin(th),np.cos(th)]); nR=-nL
        for Kside,nside in [(KL,nL),(KR,nR)]:
            VK,GK=basis_vals(Kside,p,x)
            idxK=slice(Kside*p,(Kside+1)*p)
            avgUK=.5*VK; avgGK=.5*GK; jumpUK=VK[:,:,None]*nside[None,None,:]; jumpGK=np.einsum('qpd,d->qp',GK,nside)
            for Lside,ntest in [(KL,nL),(KR,nR)]:
                VL,GL=basis_vals(Lside,p,x)
                idxL=slice(Lside*p,(Lside+1)*p)
                avgUL=.5*VL; avgGL=.5*GL; jumpUL=VL[:,:,None]*ntest[None,None,:]; jumpGL=np.einsum('qpd,d->qp',GL,ntest)
                # row=test L, col=trial K
                A1=np.einsum('q,qi,qj->ij',w,np.conj(jumpGL),avgUK)
                # - avg grad trial dot conj jumpU test
                A2=-np.einsum('q,qid,qjd->ij',w,np.conj(jumpUL),avgGK)
                A3=-1j*alpha*k*np.einsum('q,qid,qjd->ij',w,np.conj(jumpUL),jumpUK)
                A4=(beta/(1j*k))*np.einsum('q,qi,qj->ij',w,np.conj(jumpGL),jumpGK)
                Kmat[idxL,idxK]+=A1+A2+A3+A4
    # inner Dirichlet arcs
    for K in range(ne):
        t0=2*np.pi*K/ne; t1=2*np.pi*(K+1)/ne
        th,x,w=arc_quad(a,t0,t1,nq_arc); n=-np.c_[np.cos(th),np.sin(th)]
        V,G=basis_vals(K,p,x); DN=np.einsum('qpd,qd->qp',G,n)
        idx=slice(K*p,(K+1)*p)
        Kmat[idx,idx]+= -np.einsum('q,qi,qj->ij',w,np.conj(V),DN) -1j*alpha*k*np.einsum('q,qi,qj->ij',w,np.conj(V),V)
        gd=hankel1(0,k*a)*np.ones(len(w),complex)
        f[idx]+= -np.einsum('q,q,qi->i',w,gd,np.conj(DN)+alpha*1j*k*np.conj(V))
    # outer DtN globally. Build trace matrices supported on each arc quadrature.
    Q=ne*nq_arc; Vg=np.zeros((Q,N),complex); Dg=np.zeros((Q,N),complex); wg=np.zeros(Q); thg=np.zeros(Q)
    q0=0
    for K in range(ne):
        t0=2*np.pi*K/ne; t1=2*np.pi*(K+1)/ne
        th,x,w=arc_quad(R,t0,t1,nq_arc); n=np.c_[np.cos(th),np.sin(th)]
        V,G=basis_vals(K,p,x); DN=np.einsum('qpd,qd->qp',G,n)
        slq=slice(q0,q0+nq_arc); sli=slice(K*p,(K+1)*p)
        Vg[slq,sli]=V; Dg[slq,sli]=DN; wg[slq]=w; thg[slq]=th; q0+=nq_arc
    # Fourier coeffs c_m = 1/(2pi) int v e^-imtheta dtheta (R=1)
    ms=np.arange(-NdtN,NdtN+1)
    E=np.exp(-1j*np.outer(thg,ms))
    coeff=(E.conj()*0) # dummy
    # coeff m x basis
    C=(E.T @ (wg[:,None]*Vg))/(2*np.pi)
    mult=k*h1vp(ms,k*R,1)/hankel1(ms,k*R)
    TV=(np.exp(1j*np.outer(thg,ms)) @ (mult[:,None]*C))
    Res=Dg-TV
    W=wg[:,None]
    Kmat += Dg.conj().T@(W*Vg) - Vg.conj().T@(W*TV) + delta/(1j*k)*(Res.conj().T@(W*Res))
    return Kmat,f

def solve_p(p):
    t=time.time(); Tk,rk,lam,korth,krawloc=local_T(p)
    T=block_diag(*([Tk]*ne))
    Kraw,fraw=assemble_raw(p)
    Kc=T.conj().T@Kraw@T; fc=T.conj().T@fraw
    G=(Kc.conj().T-Kc)/(2j); G=(G+G.conj().T)/2
    eg=eigh(G,eigvals_only=True); gmin=eg[0]; gmax=eg[-1]
    # Cholesky factor G=L L^H. Want B=L^{-H}, B^H G B = I.
    L=cholesky(G,lower=True,check_finite=False)
    B=solve_triangular(L.conj().T,np.eye(L.shape[0]),lower=False,check_finite=False)
    Kb=B.conj().T@Kc@B
    sgr=svdvals(Kb); kGR=sgr[0]/sgr[-1]
    y=np.linalg.solve(Kb,B.conj().T@fc); acoef=B@y; craw=T@acoef
    # L2 error on sectors
    z,wz=leggauss(90); zt,wt=leggauss(90); num=0.; den=0.
    for K in range(ne):
        r=.5*((R-a)*z+(R+a)); wr=.5*(R-a)*wz
        t0=2*np.pi*K/ne; t1=2*np.pi*(K+1)/ne; th=.5*((t1-t0)*zt+(t1+t0)); wth=.5*(t1-t0)*wt
        RR,TT=np.meshgrid(r,th,indexing='ij'); W2=np.outer(wr,wth)*RR
        x=np.c_[ (RR*np.cos(TT)).ravel(), (RR*np.sin(TT)).ravel()]
        V,_=basis_vals(K,p,x); uh=V@craw[K*p:(K+1)*p]
        ue=hankel1(0,k*np.hypot(x[:,0],x[:,1]))
        num+=np.sum(W2.ravel()*np.abs(uh-ue)**2); den+=np.sum(W2.ravel()*np.abs(ue)**2)
    err=np.sqrt(num/den)
    algres=np.linalg.norm(Kc@acoef-fc)/np.linalg.norm(fc)
    return dict(p=p,nominal=ne*p,local_rank=rk,retained=ne*rk,E_L2=err,kappa_tr_raw=krawloc,kappa_tr_final=korth,kappa_Kc=np.linalg.cond(Kc),kappa_GR=kGR,G_min=gmin,G_max=gmax,alg_res=algres,time=time.time()-t)

rows=[]
for p in plist:
    print('RUN',p,flush=True)
    try:
      r=solve_p(p); rows.append(r); print(r,flush=True)
    except Exception as e:
      print('FAIL',p,repr(e),flush=True); rows.append(dict(p=p,error=str(e)))
    pd.DataFrame(rows).to_csv(ROOT / 'data' / 'highp_global_dtn_results.csv',index=False)
print(pd.DataFrame(rows))
