import sys, numpy as np, scipy.linalg as la, pandas as pd
from scipy.optimize import least_squares
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from direction_adaptive_automatic import PWDGDictionary, q_complex_angle, l2_error, physical_skeleton_residual
from revision_experiments import trace_ls_system
from test_threewave_auto import ThreeWave
from exact import Transmission
from pwdg import square_mesh, ALPHA, BETA

OUT=Path(__file__).resolve().parent.parent/'generated'; OUT.mkdir(exist_ok=True)

def q_and_qprime(k,z):
    # z complex, q=k(cos z,sin z), dq/dz=k(-sin z,cos z)
    q=k*np.array([np.cos(z),np.sin(z)],complex)
    qp=k*np.array([-np.sin(z),np.cos(z)],complex)
    return q,qp

def build_state(n,ex,z,kind='threewave',nq=12):
    z=np.asarray(z,complex)
    v,t=square_mesh(n); cent=v[t].mean(axis=1)
    kap=np.asarray(ex.wavenumber(cent[:,1]),float)
    q=[]; active=[]; qp=[]
    if kind=='threewave':
        for e,k in enumerate(kap):
            qe=[]; qpe=[]; ae=[]
            for j,zz in enumerate(z):
                qq,qqp=q_and_qprime(k,zz); qe.append(qq); qpe.append(qqp); ae.append(j)
            q.append(np.array(qe)); qp.append(np.array(qpe)); active.append(ae)
        m=len(z)
    elif kind=='elementwise':
        m=len(t)
        if len(z)!=m: raise ValueError('elementwise needs one z per element')
        for e,k in enumerate(kap):
            qq,qqp=q_and_qprime(k,z[e]); q.append(np.array([qq])); qp.append(np.array([qqp])); active.append([e])
    elif kind=='transmission':
        # z[0:2] are lower-material directions, z[2] is upper
        m=3
        for e,k in enumerate(kap):
            ids=[0,1] if cent[e,1] < 0 else [2]
            qe=[];qpe=[]
            for j in ids:
                qq,qqp=q_and_qprime(k,z[j]);qe.append(qq);qpe.append(qqp)
            q.append(np.array(qe));qp.append(np.array(qpe));active.append(ids)
    else: raise ValueError
    s=PWDGDictionary(n,ex,q,nq=nq)
    # no PWDG assembly needed for the LS residual
    D,b=trace_ls_system(s)
    return s,D,b,qp,active,m

def complex_residual_and_Kz(n,ex,z,c,kind='threewave',nq=12):
    s,D,b,qp,active,m=build_state(n,ex,z,kind,nq)
    c=np.asarray(c,complex)
    r=D@c-b
    # derivative wrt each complex z_j, using product rule c_local * d(phi)/dz
    kz=[[] for _ in range(m)]
    # We'll assemble each residual derivative in the same block ordering as trace_ls_system.
    blocks=[[] for _ in range(m)]
    for f in range(s.et.shape[0]):
        pts,wq,nrm0,L=s.edge(f); e0,e1=s.et[f]; sw=np.sqrt(wq)
        if e1>=0:
            n0=s._outward(e0,nrm0,pts); xi=0.5*(s.kap[e0]+s.kap[e1])
            # direction sensitivities of u and grad per element and global direction
            du=[]; dg=[]
            for e in (e0,e1):
                R=pts-s.cent[e]
                ph=s._basis(e,pts)
                dume={}; dgme={}
                cl=c[s.sl(e)]
                for loc,j in enumerate(active[e]):
                    qq=s.q[e][loc]; qqp=qp[e][loc]
                    dot=R@qqp
                    dphi=1j*dot*ph[:,loc]
                    dgrad=(1j*qqp[None,:]-dot[:,None]*qq[None,:])*ph[:,loc,None]
                    dume[j]=cl[loc]*dphi
                    dgme[j]=cl[loc]*dgrad
                du.append(dume);dg.append(dgme)
            for j in range(m):
                du0=du[0].get(j,np.zeros(len(pts),complex)); du1=du[1].get(j,np.zeros(len(pts),complex))
                dg0=dg[0].get(j,np.zeros((len(pts),2),complex)); dg1=dg[1].get(j,np.zeros((len(pts),2),complex))
                bru=np.sqrt(ALPHA*xi)*sw*(du0-du1)
                brf=np.sqrt(BETA/xi)*sw*(dg0@n0 + dg1@(-n0))
                blocks[j].extend([bru,brf])
        else:
            e=e0; R=pts-s.cent[e]; ph=s._basis(e,pts); cl=c[s.sl(e)]
            dume={}
            for loc,j in enumerate(active[e]):
                qqp=qp[e][loc]; dot=R@qqp
                dphi=1j*dot*ph[:,loc]
                dume[j]=cl[loc]*dphi
            for j in range(m):
                du=dume.get(j,np.zeros(len(pts),complex))
                br=np.sqrt(ALPHA*s.kap[e])*sw*du
                blocks[j].append(br)
    Kz=np.column_stack([np.concatenate(blocks[j]) for j in range(m)])
    assert Kz.shape[0]==len(r)
    return s,D,b,r,Kz

def pack(c,z):
    # [Re c, Im c, theta, eta]
    z=np.asarray(z,complex); c=np.asarray(c,complex)
    return np.r_[c.real,c.imag,z.real,z.imag]

def unpack(x,N,m):
    c=x[:N]+1j*x[N:2*N]
    z=x[2*N:2*N+m]+1j*x[2*N+m:2*N+2*m]
    return c,z

def real_residual(r): return np.r_[r.real,r.imag]

def real_jac(D,Kz):
    K=np.column_stack([D,Kz]) # complex jac wrt w=(c,z)
    # columns ordered Re c, Re z / Im c, Im z would differ. Build desired [Re c, Im c, Re z, Im z]
    N=D.shape[1]; m=Kz.shape[1]
    def col_real(M): return np.vstack([M.real,M.imag])
    def col_imag(M): return np.vstack([-M.imag,M.real])
    return np.hstack([col_real(D),col_imag(D),col_real(Kz),col_imag(Kz)])

def solve_joint(n,ex,z0,kind,nq=12,c_init='qr',seed=1,max_nfev=500,eta_max=1.8,verbose=False):
    s,D,b,qp,active,m=build_state(n,ex,z0,kind,nq)
    if c_init=='qr': c0=la.lstsq(D,b,lapack_driver='gelsy')[0]
    elif c_init=='random':
        rng=np.random.default_rng(seed); c0=0.2*(rng.normal(size=D.shape[1])+1j*rng.normal(size=D.shape[1]))
    elif c_init=='zero': c0=np.zeros(D.shape[1],complex)
    else: raise ValueError
    N=len(c0); x0=pack(c0,np.asarray(z0,complex))
    counts={'fun':0,'jac':0}
    def fun(x):
        counts['fun']+=1
        c,z=unpack(x,N,m)
        _,_,_,r,_=complex_residual_and_Kz(n,ex,z,c,kind,nq)
        return real_residual(r)
    def jac(x):
        counts['jac']+=1
        c,z=unpack(x,N,m)
        _,D,_,r,Kz=complex_residual_and_Kz(n,ex,z,c,kind,nq)
        return real_jac(D,Kz)
    # theta unbounded locally; eta bounded
    lb=np.full_like(x0,-np.inf); ub=np.full_like(x0,np.inf)
    # eta block last m
    lb[-m:]=-eta_max;ub[-m:]=eta_max
    rr=least_squares(fun,x0,jac=jac,bounds=(lb,ub),method='trf',x_scale='jac',
                     ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=max_nfev,verbose=0)
    c,z=unpack(rr.x,N,m)
    s,D,b,r,Kz=complex_residual_and_Kz(n,ex,z,c,kind,max(nq,20))
    s.u=c
    grad=np.column_stack([D,Kz]).conj().T@r
    return dict(result=rr,c=c,z=z,s=s,J=float(np.vdot(r,r).real),l2=l2_error(s,14),
                grad_norm=float(la.norm(grad)),counts=counts,rank=int(np.linalg.matrix_rank(D)))

def variable_projection(n,ex,z0,kind,nq=12,max_nfev=300,eta_max=1.8):
    z0=np.asarray(z0,complex);m=len(z0)
    x0=np.r_[z0.real,z0.imag]
    def uz(x): return x[:m]+1j*x[m:]
    def fun(x):
        z=uz(x);s,D,b,qp,active,m2=build_state(n,ex,z,kind,nq)
        c=la.lstsq(D,b,lapack_driver='gelsy')[0]
        return real_residual(D@c-b)
    lb=np.r_[np.full(m,-np.inf),np.full(m,-eta_max)];ub=np.r_[np.full(m,np.inf),np.full(m,eta_max)]
    rr=least_squares(fun,x0,bounds=(lb,ub),method='trf',x_scale='jac',ftol=1e-13,xtol=1e-13,gtol=1e-13,
                     max_nfev=max_nfev,diff_step=1e-6)
    z=uz(rr.x);s,D,b,qp,active,m=build_state(n,ex,z,kind,max(nq,20));c=la.lstsq(D,b,lapack_driver='gelsy')[0];s.u=c
    r=D@c-b
    return dict(result=rr,c=c,z=z,s=s,J=float(np.vdot(r,r).real),l2=l2_error(s,14),rank=int(np.linalg.matrix_rank(D)))

def fd_check(n,ex,z,c,kind,nq=10,h=2e-7):
    s,D,b,r,Kz=complex_residual_and_Kz(n,ex,z,c,kind,nq);J=real_jac(D,Kz);x=pack(c,z);N=len(c);m=len(z)
    errs=[]
    for j in [0,N,2*N,2*N+m-1]:
        xp=x.copy();xm=x.copy();xp[j]+=h;xm[j]-=h
        cp,zp=unpack(xp,N,m);cm,zm=unpack(xm,N,m)
        rp=complex_residual_and_Kz(n,ex,zp,cp,kind,nq)[3]
        rm=complex_residual_and_Kz(n,ex,zm,cm,kind,nq)[3]
        fd=(real_residual(rp)-real_residual(rm))/(2*h)
        errs.append((j,la.norm(fd-J[:,j])/max(la.norm(fd),1e-15)))
    return errs

if __name__=='__main__':
    rows=[]
    # derivative check at generic state
    ex=ThreeWave(4.); z0=np.deg2rad(np.array([43.929,130.738,269.353]))+1j*np.array([.05,-.03,.04])
    s,D,b,qp,active,m=build_state(2,ex,z0,'threewave',10);c=la.lstsq(D,b,lapack_driver='gelsy')[0]
    print('FD three',fd_check(2,ex,z0,c,'threewave',10))
    # Joint from QR and random c
    for init in ['qr','random','zero']:
        o=solve_joint(2,ex,z0,'threewave',nq=10,c_init=init,seed=4,max_nfev=600)
        print('three',init,'nfev',o['result'].nfev,'J',o['J'],'l2',o['l2'],'grad',o['grad_norm'],'angles',np.degrees(o['z'].real),'eta',o['z'].imag)
        rows.append(['threewave',init,o['result'].nfev,o['J'],o['l2'],o['grad_norm'],*np.degrees(o['z'].real),*o['z'].imag])
    vp=variable_projection(2,ex,z0,'threewave',10,500)
    print('three VP',vp['result'].nfev,vp['J'],vp['l2'],np.degrees(vp['z'].real),vp['z'].imag)
    # Transmission
    for ang,zdeg,eta in [(69,[251.786,103.766,65.676],[0,0,0]),(29,[263.749,45.757,351.014],[-.6095,.2579,1.2004])]:
        ex=Transmission(12,ang);z0=np.deg2rad(zdeg)+1j*np.array(eta)
        s,D,b,qp,active,m=build_state(2,ex,z0,'transmission',10);c=la.lstsq(D,b,lapack_driver='gelsy')[0]
        print('FD trans',ang,fd_check(2,ex,z0,c,'transmission',10))
        for init in ['qr','random']:
            o=solve_joint(2,ex,z0,'transmission',nq=10,c_init=init,seed=7,max_nfev=500)
            print('trans',ang,init,'nfev',o['result'].nfev,'J',o['J'],'l2',o['l2'],'grad',o['grad_norm'],'angles',np.degrees(o['z'].real)%360,'eta',o['z'].imag)
            rows.append([f'trans{ang}',init,o['result'].nfev,o['J'],o['l2'],o['grad_norm'],* (np.degrees(o['z'].real)%360),*o['z'].imag])
    cols=['case','init','nfev','J','l2','grad_norm','theta1','theta2','theta3','eta1','eta2','eta3']
    pd.DataFrame(rows,columns=cols).to_csv(OUT/'joint_ls_summary.csv',index=False)
