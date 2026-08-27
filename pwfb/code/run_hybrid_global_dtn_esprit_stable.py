from pathlib import Path
import numpy as np, pandas as pd, time
from scipy.special import jv,jvp,hankel1,h1vp
from scipy.linalg import block_diag,eigh,cholesky,solve_triangular,svdvals
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares
from scipy.linalg import svd

ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'data').mkdir(exist_ok=True)

k=16.0; a=0.5; R=1.0; ne=8; alpha=beta=delta=0.5; h=0.44; rc=0.75; tau_rank=1e-12; NdtN=70
budgets=[21,27,33,39,45,53,61]
q_candidates=[0,2,4,6]
M_MODAL=110
ms_all=np.arange(-M_MODAL,M_MODAL+1)
tau_m=np.sqrt(2*np.pi*h*k*(np.abs(jv(ms_all,k*h))**2+np.abs(jvp(ms_all,k*h,1))**2))
a_m=((-1.0)**ms_all)*hankel1(ms_all,k*rc)
gamma=(1j)**(-ms_all)*a_m
trace_norm=np.linalg.norm(tau_m*gamma)
cent_th=(np.arange(ne)+0.5)*2*np.pi/ne
CEN=rc*np.c_[np.cos(cent_th),np.sin(cent_th)]

def gamma_at(m):
    return gamma[m+M_MODAL]

def esprit_init(M,q,N=24):
    mm=np.arange(M+1,M+1+N)
    seq=np.array([gamma_at(int(m)) for m in mm])
    L=N//2+1; K=N-L+1
    H=np.empty((L,K),complex)
    for r in range(L): H[r,:]=seq[r:r+K]
    U,s,_=svd(H,full_matrices=False)
    Uq=U[:,:q]
    S=np.linalg.pinv(Uq[:-1,:])@Uq[1:,:]
    z=np.linalg.eigvals(S)
    th=(-np.angle(z)+np.pi)%(2*np.pi)-np.pi
    return np.sort(th)

def fit_candidate(p,q):
    M=(p-q-1)//2
    mask=np.abs(ms_all)>M
    mt=ms_all[mask]; wt=tau_m[mask]; b=wt*gamma[mask]
    if q==0:
        return M,np.array([]),np.linalg.norm(b)/trace_norm
    th0=esprit_init(M,q,N=24)
    def solve_c(th):
        A=wt[:,None]*np.exp(-1j*np.outer(mt,th))
        c=np.linalg.lstsq(A,b,rcond=1e-13)[0]
        return c,A@c-b
    def fun(th):
        c,r=solve_c(th); return np.r_[r.real,r.imag]
    res=least_squares(fun,th0,method='trf',max_nfev=400,xtol=1e-12,ftol=1e-12,gtol=1e-12)
    th=(res.x+np.pi)%(2*np.pi)-np.pi
    c,r=solve_c(th)
    # Remove fitted rays carrying negligible amplitude, then keep the same FB order.
    scale=np.max(np.abs(c)) if len(c) else 0.0
    if scale>0:
        keep=np.abs(c)>1e-10*scale
        th=th[keep]
        if keep.sum()!=q:
            q2=int(keep.sum())
            if q2:
                c,r=solve_c(th)
            else:
                r=-b
    return M,th,np.linalg.norm(r)/trace_norm

def select(p):
    out=[]
    for q in q_candidates:
        M,th,e=fit_candidate(p,q)
        out.append((e,len(th),M,th,q))
    # Stability filter: require the complete equilibrated hybrid trace Gram
    # to retain its nominal dimension at tau_rank before global assembly.
    admiss=[]
    for item in out:
        e,qactive,M,th,qrequested=item
        Ttmp,rk,_,_,_=local_gram(th,M)
        nominal=qactive+2*M+1
        if rk==nominal:
            admiss.append(item)
    if not admiss:
        raise RuntimeError("no locally full-rank hybrid candidate")
    e,qactive,M,th,qrequested=min(admiss,key=lambda z:z[0])
    return (e,qactive,M,th),out

def edge_quad_line(x0,x1,nq=100):
    z,w=leggauss(nq); pts=.5*((1-z)[:,None]*x0+(1+z)[:,None]*x1); ww=.5*np.linalg.norm(x1-x0)*w
    return pts,ww

def arc_quad(r,t0,t1,nq=128):
    z,w=leggauss(nq); th=.5*((t1-t0)*z+(t1+t0)); ww=.5*(t1-t0)*r*w
    return th,r*np.c_[np.cos(th),np.sin(th)],ww

def spec_for_elem(K,rel_angles,M):
    dirs=np.c_[np.cos(rel_angles+cent_th[K]),np.sin(rel_angles+cent_th[K])] if len(rel_angles) else np.zeros((0,2))
    modes=np.arange(-M,M+1,dtype=int)
    return dirs,modes

def basis_vals(K,rel_angles,M,x):
    dirs,modes=spec_for_elem(K,rel_angles,M)
    parts=[]; grads=[]
    if len(dirs):
        ph=np.exp(1j*k*((x-CEN[K])@dirs.T)); gr=1j*k*ph[:,:,None]*dirs[None,:,:]
        parts.append(ph); grads.append(gr)
    y=x-CEN[K]; rho=np.hypot(y[:,0],y[:,1]); phi=np.arctan2(y[:,1],y[:,0])-cent_th[K]
    # local rotated polar angle; phase rotation does not alter span and improves covariance
    rr=np.where(rho>1e-14,rho,1e-14)
    E=np.exp(1j*np.outer(phi,modes)); Z=k*rho[:,None]
    J=jv(modes[None,:],Z); JP=jvp(modes[None,:],Z,1)
    V=J*E
    er=np.c_[np.cos(phi+cent_th[K]),np.sin(phi+cent_th[K])]
    et=np.c_[-np.sin(phi+cent_th[K]),np.cos(phi+cent_th[K])]
    rad=k*JP*E
    tang=(1j*modes[None,:]/rr[:,None])*V
    G=rad[:,:,None]*er[:,None,:]+tang[:,:,None]*et[:,None,:]
    # center limiting values not encountered by edge quadrature; volume quadrature can be arbitrarily close but not equal
    parts.append(V); grads.append(G)
    return np.concatenate(parts,axis=1),np.concatenate(grads,axis=1)

def local_gram(rel_angles,M,nq=4096):
    t=2*np.pi*np.arange(nq)/nq
    x=CEN[0]+h*np.c_[np.cos(t+cent_th[0]),np.sin(t+cent_th[0])]
    n=np.c_[np.cos(t+cent_th[0]),np.sin(t+cent_th[0])]
    V,G=basis_vals(0,rel_angles,M,x); DN=np.einsum('qpd,qd->qp',G,n)
    w=h*2*np.pi/nq
    Gram=w*(k*(V.conj().T@V)+(1/k)*(DN.conj().T@DN)); Gram=(Gram+Gram.conj().T)/2
    # equilibrate all columns first; then retain numerically resolvable hybrid trace subspace
    d=np.sqrt(np.maximum(np.real(np.diag(Gram)),1e-300)); D=np.diag(1/d)
    Geq=D@Gram@D; lam,U=eigh(Geq); keep=lam>tau_rank*lam.max()
    T=D@U[:,keep]@np.diag(1/np.sqrt(lam[keep]))
    Gf=T.conj().T@Gram@T; sf=svdvals(Gf)
    return T,int(keep.sum()),np.linalg.cond(Gram),sf[0]/sf[-1],lam

def assemble_raw(rel_angles,M,nq_line=100,nq_arc=128):
    p=len(rel_angles)+2*M+1; N=ne*p; Kmat=np.zeros((N,N),complex); f=np.zeros(N,complex)
    for j in range(ne):
        th=2*np.pi*j/ne; x0=a*np.array([np.cos(th),np.sin(th)]); x1=R*np.array([np.cos(th),np.sin(th)])
        x,w=edge_quad_line(x0,x1,nq_line); KL=(j-1)%ne; KR=j
        nL=np.array([-np.sin(th),np.cos(th)]); nR=-nL
        cache={}
        for Kside,nside in [(KL,nL),(KR,nR)]:
            V,G=basis_vals(Kside,rel_angles,M,x); cache[Kside]=(V,G,nside)
        for Kside in [KL,KR]:
            VK,GK,nside=cache[Kside]; idxK=slice(Kside*p,(Kside+1)*p)
            avgUK=.5*VK; avgGK=.5*GK; jumpUK=VK[:,:,None]*nside[None,None,:]; jumpGK=np.einsum('qpd,d->qp',GK,nside)
            for Lside in [KL,KR]:
                VL,GL,ntest=cache[Lside]; idxL=slice(Lside*p,(Lside+1)*p)
                jumpUL=VL[:,:,None]*ntest[None,None,:]; jumpGL=np.einsum('qpd,d->qp',GL,ntest)
                A1=np.einsum('q,qi,qj->ij',w,np.conj(jumpGL),avgUK)
                A2=-np.einsum('q,qid,qjd->ij',w,np.conj(jumpUL),avgGK)
                A3=-1j*alpha*k*np.einsum('q,qid,qjd->ij',w,np.conj(jumpUL),jumpUK)
                A4=(beta/(1j*k))*np.einsum('q,qi,qj->ij',w,np.conj(jumpGL),jumpGK)
                Kmat[idxL,idxK]+=A1+A2+A3+A4
    for K in range(ne):
        t0=2*np.pi*K/ne; t1=2*np.pi*(K+1)/ne; th,x,w=arc_quad(a,t0,t1,nq_arc); n=-np.c_[np.cos(th),np.sin(th)]
        V,G=basis_vals(K,rel_angles,M,x); DN=np.einsum('qpd,qd->qp',G,n); idx=slice(K*p,(K+1)*p)
        Kmat[idx,idx]+=-np.einsum('q,qi,qj->ij',w,np.conj(V),DN)-1j*alpha*k*np.einsum('q,qi,qj->ij',w,np.conj(V),V)
        gd=hankel1(0,k*a)*np.ones(len(w),complex)
        f[idx]+=-np.einsum('q,q,qi->i',w,gd,np.conj(DN)+alpha*1j*k*np.conj(V))
    Q=ne*nq_arc; Vg=np.zeros((Q,N),complex); Dg=np.zeros((Q,N),complex); wg=np.zeros(Q); thg=np.zeros(Q)
    q0=0
    for K in range(ne):
        t0=2*np.pi*K/ne; t1=2*np.pi*(K+1)/ne; th,x,w=arc_quad(R,t0,t1,nq_arc); n=np.c_[np.cos(th),np.sin(th)]
        V,G=basis_vals(K,rel_angles,M,x); DN=np.einsum('qpd,qd->qp',G,n); slq=slice(q0,q0+nq_arc); sli=slice(K*p,(K+1)*p)
        Vg[slq,sli]=V; Dg[slq,sli]=DN; wg[slq]=w; thg[slq]=th; q0+=nq_arc
    ms=np.arange(-NdtN,NdtN+1); E=np.exp(-1j*np.outer(thg,ms)); C=(E.T@(wg[:,None]*Vg))/(2*np.pi)
    mult=k*h1vp(ms,k*R,1)/hankel1(ms,k*R); TV=np.exp(1j*np.outer(thg,ms))@(mult[:,None]*C); Res=Dg-TV; W=wg[:,None]
    Kmat+=Dg.conj().T@(W*Vg)-Vg.conj().T@(W*TV)+delta/(1j*k)*(Res.conj().T@(W*Res))
    return Kmat,f

def solve_budget(p,best):
    pred,q,M,rel_angles=best; t0=time.time(); praw=q+2*M+1
    Tloc,rk,kraw,kfinal,lam=local_gram(rel_angles,M)
    T=block_diag(*([Tloc]*ne)); Kraw,fraw=assemble_raw(rel_angles,M); Kc=T.conj().T@Kraw@T; fc=T.conj().T@fraw
    G=(Kc.conj().T-Kc)/(2j); G=(G+G.conj().T)/2; eg=eigh(G,eigvals_only=True)
    if eg[0] <= 0: raise RuntimeError(f'graph matrix not PD: {eg[0]}')
    L=cholesky(G,lower=True,check_finite=False); B=solve_triangular(L.conj().T,np.eye(L.shape[0]),lower=False,check_finite=False)
    Kb=B.conj().T@Kc@B; sgr=svdvals(Kb); kGR=sgr[0]/sgr[-1]
    y=np.linalg.solve(Kb,B.conj().T@fc); coef=T@(B@y)
    z,wz=leggauss(70); zt,wt=leggauss(70); num=den=0.0
    for K in range(ne):
        rr=.5*((R-a)*z+(R+a)); wr=.5*(R-a)*wz; tA=2*np.pi*K/ne; tB=2*np.pi*(K+1)/ne; th=.5*((tB-tA)*zt+(tB+tA)); wth=.5*(tB-tA)*wt
        RR,TT=np.meshgrid(rr,th,indexing='ij'); W2=np.outer(wr,wth)*RR; x=np.c_[(RR*np.cos(TT)).ravel(),(RR*np.sin(TT)).ravel()]
        V,_=basis_vals(K,rel_angles,M,x); uh=V@coef[K*praw:(K+1)*praw]; ue=hankel1(0,k*np.hypot(x[:,0],x[:,1]))
        num+=np.sum(W2.ravel()*np.abs(uh-ue)**2); den+=np.sum(W2.ravel()*np.abs(ue)**2)
    err=np.sqrt(num/den)
    return dict(p=p,qPW=q,MFB=M,nominal_local=praw,retained_local=rk,pred_trace=pred,E_L2=err,kappa_tr_raw=kraw,kappa_tr_final=kfinal,kappa_K=np.linalg.cond(Kc),kappa_GR=kGR,G_min=eg[0],alg_res=np.linalg.norm(Kc@(B@y)-fc)/np.linalg.norm(fc),time_s=time.time()-t0,angles_deg_rel=';'.join(f'{x:.8f}' for x in np.rad2deg(rel_angles)))

rows=[]; selrows=[]
for p in budgets:
    print('SELECT',p,flush=True); best,allc=select(p); print(' best',best[1],best[2],best[0],np.rad2deg(best[3]),flush=True)
    for e,qactive,M,th,qrequested in allc: selrows.append(dict(p=p,q_requested=qrequested,qPW=qactive,MFB=M,pred_trace=e,angles_deg_rel=';'.join(f'{x:.8f}' for x in np.rad2deg(th))))
    print('SOLVE',p,flush=True)
    r=solve_budget(p,best); rows.append(r); print(r,flush=True)
    pd.DataFrame(rows).to_csv(ROOT / 'data' / 'hybrid_global_dtn_results.csv',index=False)
    pd.DataFrame(selrows).to_csv(ROOT / 'data' / 'hybrid_selector_candidates_reproduced.csv',index=False)
print(pd.DataFrame(rows))
