import sys, numpy as np, scipy.linalg as la
from pathlib import Path
from scipy.optimize import least_squares
from scipy.stats import qmc
from exact import Hankel, Transmission
from pwdg import square_mesh
from direction_adaptive_automatic import (PWDGDictionary, q_complex_angle, l2_error, graph_data,
    physical_skeleton_residual_vector, physical_skeleton_residual,
    optimize_region_directions_ls, solve_region_space, uniform_dictionary)

OUT=Path(__file__).resolve().parent.parent/'generated'; OUT.mkdir(exist_ok=True)

def solve_exact(A,F):
    A=A.toarray() if hasattr(A,'toarray') else np.asarray(A)
    G=(A-A.conj().T)/(2j); G=(G+G.conj().T)/2
    L=la.cholesky(G,lower=True)
    X=la.solve_triangular(L,A,lower=True)
    Ah=la.solve_triangular(L,X.conj().T,lower=True).conj().T
    fh=la.solve_triangular(L,F,lower=True)
    ch=la.solve(Ah,fh,assume_a='gen')
    return la.solve_triangular(L.conj().T,ch,lower=False)

def solve_pwdg_q(n, ex, q_by_element, nq=18):
    s=PWDGDictionary(n,ex,q_by_element,nq=nq); A,F=s.assemble()
    try: s.u=solve_exact(A,F)
    except Exception: s.u=la.solve(A.toarray(),F,assume_a='gen')
    return s

def trace_ls_system(s):
    # complex D c - b whose squared norm is J with current quadrature
    rows=[]; rhs=[]; N=s.ndof
    for f in range(s.et.shape[0]):
        pts,wq,nrm0,L=s.edge(f); e0,e1=s.et[f]
        if e1>=0:
            n0=s._outward(e0,nrm0,pts); xi=0.5*(s.kap[e0]+s.kap[e1])
            p0=s._basis(e0,pts); p1=s._basis(e1,pts)
            # u jump residual
            R=np.zeros((len(pts),N),complex)
            R[:,s.sl(e0)]=p0; R[:,s.sl(e1)]=-p1
            R*=np.sqrt((0.5*xi)*wq)[:,None]
            rows.append(R); rhs.append(np.zeros(len(pts),complex))
            # normal flux jump residual
            R=np.zeros((len(pts),N),complex)
            R[:,s.sl(e0)]=p0*(1j*(s.q[e0]@n0))[None,:]
            R[:,s.sl(e1)]=p1*(1j*(s.q[e1]@(-n0)))[None,:]
            R*=np.sqrt((0.5/xi)*wq)[:,None]
            rows.append(R); rhs.append(np.zeros(len(pts),complex))
        else:
            e=e0; p=s._basis(e,pts)
            R=np.zeros((len(pts),N),complex); R[:,s.sl(e)]=p
            R*=np.sqrt((0.5*s.kap[e])*wq)[:,None]
            b=np.sqrt((0.5*s.kap[e])*wq)*s.exact(pts[:,0],pts[:,1])
            rows.append(R); rhs.append(b)
    return np.vstack(rows), np.concatenate(rhs)

def solve_ls_q(n, ex, q_by_element, nq=18, rcond=None):
    s=PWDGDictionary(n,ex,q_by_element,nq=nq); s.assemble()
    D,b=trace_ls_system(s)
    c,resid,rank,sv=la.lstsq(D,b,cond=rcond,lapack_driver='gelsy')
    s.u=c; s.ls_rank=rank; s.D=D; s.b=b
    return s

def one_ray_q(n, ex, theta):
    v,t=square_mesh(n); kap=np.asarray(ex.wavenumber(v[t].mean(axis=1)[:,1]),float)
    theta=np.asarray(theta); return [np.array([kap[e]*np.array([np.cos(theta[e]),np.sin(theta[e])])],complex) for e in range(len(t))]

def continue_cyl(method='pwdg'):
    n=2; source=(-1.6,0.3); nt=len(square_mesh(n)[1])
    sob=qmc.Sobol(1,scramble=True,seed=19).random_base2(int(np.ceil(np.log2(nt))))[:nt,0]
    x=2*np.pi*sob; rows=[]; total_calls=0
    for k in [0.5,1.,1.5,2.,3.,4.,5.,6.,7.,8.]:
        ex=Hankel(k,source=source)
        def fun(xx):
            nonlocal total_calls
            total_calls+=1
            q=one_ray_q(n,ex,np.asarray(xx)%(2*np.pi))
            s=solve_pwdg_q(n,ex,q,nq=14) if method=='pwdg' else solve_ls_q(n,ex,q,nq=14)
            if method=='pwdg': return physical_skeleton_residual_vector(s)
            rr=s.D@s.u-s.b; return np.concatenate([rr.real,rr.imag])
        rr=least_squares(fun,x,method='trf',x_scale='jac',ftol=5e-12,xtol=5e-12,gtol=5e-12,max_nfev=220,diff_step=2e-5)
        x=rr.x
        q=one_ray_q(n,ex,x%(2*np.pi)); s=solve_pwdg_q(n,ex,q,nq=24) if method=='pwdg' else solve_ls_q(n,ex,q,nq=24)
        J=physical_skeleton_residual(s)**2
        rows.append([method,k,rr.nfev,total_calls,l2_error(s,16),J,*np.degrees(x%(2*np.pi))])
        print(method,k,rr.nfev,total_calls,l2_error(s,16),J)
    return x,rows

def radial_reference():
    n=2; ex=Hankel(8,source=(-1.6,0.3)); v,t=square_mesh(n); cent=v[t].mean(axis=1)
    d=cent-np.array(ex.x0); th=np.arctan2(d[:,1],d[:,0])%(2*np.pi)
    q=one_ray_q(n,ex,th); sp=solve_pwdg_q(n,ex,q,nq=24); sl=solve_ls_q(n,ex,q,nq=24)
    return [('radial-PWDG',l2_error(sp,16),physical_skeleton_residual(sp)**2,graph_data(sp.A)['cond_raw'],graph_data(sp.A)['cond_gr']),
            ('radial-LS',l2_error(sl,16),physical_skeleton_residual(sl)**2,np.nan,np.nan)]

def uniform_sweep():
    ex=Hankel(8,source=(-1.6,0.3)); rows=[]
    for p in range(10,25):
        q=uniform_dictionary(2,ex,p); s=solve_pwdg_q(2,ex,q,nq=24); gd=graph_data(s.A)
        rows.append([p,s.ndof,l2_error(s,16),gd['cond_raw'],gd['cond_gr'],gd['gmin']])
        print('uniform',rows[-1])
    return rows

def trans_mesh_check():
    rows=[]
    starts={69: np.array([[np.deg2rad(251.786),0],[np.deg2rad(103.766),0],[np.deg2rad(65.676),0]]),
            29: np.array([[np.deg2rad(263.749),-.6095],[np.deg2rad(45.757),.2579],[np.deg2rad(351.014),1.2004]])}
    for n in [2,4]:
      for ang in [69,29]:
        ex=Transmission(4,ang); z0=starts[ang]
        z,s,rr=optimize_region_directions_ls(n,ex,z0[:2],z0[2:],nq=12,max_nfev=250)
        gd=graph_data(s.A); rows.append([n,ang,s.ndof,rr.nfev,l2_error(s,14),physical_skeleton_residual(s)**2,gd['cond_raw'],gd['cond_gr'],*z.ravel()])
        print('trans',n,ang,rows[-1][:8])
    return rows

if __name__=='__main__':
    import pandas as pd
    xp,rp=continue_cyl('pwdg'); xl,rl=continue_cyl('ls')
    cols=['method','k','nfev_stage','full_residual_calls_cumulative','l2','J']+[f'theta{i}' for i in range(8)]
    pd.DataFrame(rp+rl,columns=cols).to_csv(OUT/'cyl_one_ray_pwdg_vs_ls.csv',index=False)
    pd.DataFrame(radial_reference(),columns=['method','l2','J','cond_raw','cond_gr']).to_csv(OUT/'cyl_radial_reference.csv',index=False)
    pd.DataFrame(uniform_sweep(),columns=['p','ndof','l2','cond_raw','cond_gr','gmin']).to_csv(OUT/'cyl_uniform_extended.csv',index=False)
    # Transmission at omega=12 is reproduced by transmission_omega12_experiments.py.
