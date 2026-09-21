import sys, os, time
import numpy as np
import scipy.linalg as la
from scipy.optimize import least_squares
from pwdg import square_mesh, ALPHA, BETA
from exact import Transmission
from direction_adaptive_automatic import PWDGDictionary, physical_skeleton_residual_vector, l2_error, graph_data


def d_and_derivs(k, theta, eta):
    z = theta + 1j*eta
    d = np.array([np.cos(z), np.sin(z)], complex)
    dz = np.array([-np.sin(z), np.cos(z)], complex)
    q = k*d
    q_theta = k*dz
    q_eta = 1j*k*dz
    return q, q_theta, q_eta


def make_family_state(n, exact, z, families_by_element, nq=16):
    v,t = square_mesh(n)
    cent=v[t].mean(axis=1)
    kap=np.asarray(exact.wavenumber(cent[:,1]),float)
    z=np.asarray(z,float).reshape(-1,2)
    q_by=[]; fam_by=[]
    for e,fams in enumerate(families_by_element):
        arr=[]
        for f in fams:
            arr.append(d_and_derivs(kap[e], z[f,0], z[f,1])[0])
        q_by.append(np.array(arr,complex).reshape(-1,2))
        fam_by.append(np.array(fams,int))
    s=PWDGDictionary(n, exact, q_by, nq=nq)
    A,F=s.assemble(); s.u=la.solve(A.toarray(),F)
    return s, fam_by


def parameter_dq(s, fam_by, z, p):
    f=p//2; comp=p%2
    dq=[]
    for e in range(s.nt):
        de=np.zeros_like(s.q[e])
        for j,fam in enumerate(fam_by[e]):
            if fam==f:
                _,qt,qe=d_and_derivs(s.kap[e], z[f,0], z[f,1])
                de[j]=qt if comp==0 else qe
        dq.append(de)
    return dq


def assemble_directional_derivative(s, dq):
    """Exact real-parameter directional derivative dA,dF for specified dq[e][j]."""
    N=s.ndof
    dF=np.zeros(N,complex)
    rows=[];cols=[];vals=[]
    def push(test,trial,block_test_trial):
        rt=np.arange(s.off[test],s.off[test+1]); ct=np.arange(s.off[trial],s.off[trial+1])
        if len(rt)==0 or len(ct)==0:return
        rows.append(np.repeat(rt,len(ct))); cols.append(np.tile(ct,len(rt))); vals.append(np.asarray(block_test_trial).ravel())
    for ff in range(s.et.shape[0]):
        pts,wq,nrm0,L=s.edge(ff); e0,e1=s.et[ff]
        if e1>=0:
            n0=s._outward(e0,nrm0,pts); xi=.5*(s.kap[e0]+s.kap[e1])
            for eA,nA in ((e0,n0),(e1,-n0)):
                for eB,nB in ((e0,n0),(e1,-n0)):
                    if s.nloc[eA]==0 or s.nloc[eB]==0:continue
                    XA=pts-s.cent[eA]; XB=pts-s.cent[eB]
                    pa=s._basis(eA,pts); pb=s._basis(eB,pts)
                    dpa=1j*(XA@dq[eA].T)*pa
                    dcpb=-1j*(XB@np.conj(dq[eB]).T)*np.conj(pb)
                    G=np.einsum('q,ql,qm->lm',wq,pa,np.conj(pb),optimize=True)
                    dG=(np.einsum('q,ql,qm->lm',wq,dpa,np.conj(pb),optimize=True)
                        +np.einsum('q,ql,qm->lm',wq,pa,dcpb,optimize=True))
                    qA,qB=s.q[eA],s.q[eB]; dAq,dBq=dq[eA],dq[eB]
                    qA_nA=qA@nA; qA_nB=qA@nB; cqB_nB=np.conj(qB@nB)
                    dqA_nA=dAq@nA; dqA_nB=dAq@nB; dcqB_nB=np.conj(dBq@nB)
                    C=(-.5j*cqB_nB[None,:]-.5j*qA_nB[:,None]
                       +1j*(BETA/xi)*qA_nA[:,None]*cqB_nB[None,:]
                       +1j*xi*ALPHA*float(nA@nB))
                    dC=(-.5j*dcqB_nB[None,:]-.5j*dqA_nB[:,None]
                        +1j*(BETA/xi)*(dqA_nA[:,None]*cqB_nB[None,:]
                                      +qA_nA[:,None]*dcqB_nB[None,:]))
                    push(eB,eA,(dC*G+C*dG).T)
        else:
            e=e0
            if s.nloc[e]==0:continue
            nrm=s._outward(e,nrm0,pts); k=s.kap[e]
            X=pts-s.cent[e]; pa=s._basis(e,pts); qe=s.q[e]; dqe=dq[e]
            dpa=1j*(X@dqe.T)*pa
            dcpa=-1j*(X@np.conj(dqe).T)*np.conj(pa)
            qn=qe@nrm; dqn=dqe@nrm
            G=np.einsum('q,ql,qm->lm',wq,pa,np.conj(pa),optimize=True)
            dG=(np.einsum('q,ql,qm->lm',wq,dpa,np.conj(pa),optimize=True)
                +np.einsum('q,ql,qm->lm',wq,pa,dcpa,optimize=True))
            C=(-1j*qn[:,None]+1j*ALPHA*k)*np.ones((1,s.nloc[e]),complex)
            dC=(-1j*dqn[:,None])*np.ones((1,s.nloc[e]),complex)
            push(e,e,(dC*G+C*dG).T)
            g=s.exact(pts[:,0],pts[:,1])
            rhs=np.einsum('q,q,qm->m',wq,g,np.conj(pa),optimize=True)
            drhs=np.einsum('q,q,qm->m',wq,g,dcpa,optimize=True)
            dF[s.sl(e)] += 1j*np.conj(dqn)*rhs + (1j*np.conj(qn)+1j*ALPHA*k)*drhs
    if rows:
        import scipy.sparse as sp
        dA=sp.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(N,N)).toarray()
    else:dA=np.zeros((N,N),complex)
    return dA,dF


def solution_sensitivity(s,dA,dF):
    A=s.A.toarray(); c=s.u
    return la.solve(A, dF-dA@c)


def residual_vector_and_jac_column(s,dq,dc):
    vals=[]; dvals=[]
    for ff in range(s.et.shape[0]):
        pts,wq,nrm0,L=s.edge(ff); e0,e1=s.et[ff]; sw=np.sqrt(wq)
        def u_du(e):
            X=pts-s.cent[e]; B=s._basis(e,pts); c=s.u[s.sl(e)]; dce=dc[s.sl(e)]
            dB=1j*(X@dq[e].T)*B
            u=B@c; du=B@dce+dB@c
            grad=(B*(1j*c)[None,:])@s.q[e]
            dgrad=(dB*(1j*c)[None,:]+B*(1j*dce)[None,:])@s.q[e] + (B*(1j*c)[None,:])@dq[e]
            return u,du,grad,dgrad
        if e1>=0:
            n0=s._outward(e0,nrm0,pts); xi=.5*(s.kap[e0]+s.kap[e1])
            u0,du0,g0,dg0=u_du(e0); u1,du1,g1,dg1=u_du(e1)
            ru=np.sqrt(ALPHA*xi)*sw*(u0-u1); dru=np.sqrt(ALPHA*xi)*sw*(du0-du1)
            rf=np.sqrt(BETA/xi)*sw*(g0@n0 + g1@(-n0)); drf=np.sqrt(BETA/xi)*sw*(dg0@n0 + dg1@(-n0))
            for r,dr in ((ru,dru),(rf,drf)):
                vals.extend([r.real,r.imag]);dvals.extend([dr.real,dr.imag])
        else:
            e=e0; u,du,g,dg=u_du(e)
            r=np.sqrt(ALPHA*s.kap[e])*sw*(u-s.exact(pts[:,0],pts[:,1]))
            dr=np.sqrt(ALPHA*s.kap[e])*sw*du
            vals.extend([r.real,r.imag]); dvals.extend([dr.real,dr.imag])
    return np.concatenate(vals),np.concatenate(dvals)


def all_analytic(s,fam_by,z):
    npar=2*len(z); r0=physical_skeleton_residual_vector(s)
    J=np.empty((len(r0),npar)); sens=[]; dAs=[];dFs=[]
    for p in range(npar):
        dq=parameter_dq(s,fam_by,z,p)
        dA,dF=assemble_directional_derivative(s,dq)
        dc=solution_sensitivity(s,dA,dF)
        r,dr=residual_vector_and_jac_column(s,dq,dc)
        assert np.linalg.norm(r-r0) < 1e-9*(1+np.linalg.norm(r0))
        J[:,p]=dr; sens.append(dc);dAs.append(dA);dFs.append(dF)
    return r0,J,sens,dAs,dFs


def fd_state(n,exact,z,fams,nq,p,h):
    zz=np.array(z,float,copy=True); f=p//2;c=p%2
    zp=zz.copy();zm=zz.copy();zp[f,c]+=h;zm[f,c]-=h
    sp,_=make_family_state(n,exact,zp,fams,nq); sm,_=make_family_state(n,exact,zm,fams,nq)
    return sp,sm


def rel(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))


def derivative_check(name,n,exact,z,fams,nq=16,h=2e-6):
    s,fam_by=make_family_state(n,exact,z,fams,nq)
    r,J,sens,dAs,dFs=all_analytic(s,fam_by,np.asarray(z))
    rows=[]
    for p in range(2*len(z)):
        sp,sm=fd_state(n,exact,z,fams,nq,p,h)
        dAfd=(sp.A.toarray()-sm.A.toarray())/(2*h); dFfd=(sp.F-sm.F)/(2*h)
        dcfd=(sp.u-sm.u)/(2*h)
        drfd=(physical_skeleton_residual_vector(sp)-physical_skeleton_residual_vector(sm))/(2*h)
        grad=2*np.dot(r,J[:,p]); gradfd=(np.dot(physical_skeleton_residual_vector(sp),physical_skeleton_residual_vector(sp))-np.dot(physical_skeleton_residual_vector(sm),physical_skeleton_residual_vector(sm)))/(2*h)
        rows.append([p,rel(dAs[p],dAfd),rel(dFs[p],dFfd),rel(sens[p],dcfd),rel(J[:,p],drfd),abs(grad-gradfd)/max(abs(gradfd),1e-14)])
    return s,r,J,np.array(rows)

class ThreeWave:
    def __init__(self,k=8.,angles_deg=(17.,123.,251.),amps=(1.,0.7*np.exp(.4j),.45*np.exp(-.8j))):
        self.kappa=float(k); a=np.deg2rad(np.array(angles_deg,float));self.amps=np.array(amps,complex);self.d=np.stack([np.cos(a),np.sin(a)],1)
    def wavenumber(self,y):return np.full(np.shape(y),self.kappa,float)
    def __call__(self,x,y):
        X=np.stack([np.asarray(x,float),np.asarray(y,float)],-1); ph=np.exp(1j*self.kappa*np.einsum('...j,mj->...m',X,self.d));return ph@self.amps
    def grad(self,x,y):
        X=np.stack([np.asarray(x,float),np.asarray(y,float)],-1);ph=np.exp(1j*self.kappa*np.einsum('...j,mj->...m',X,self.d));return 1j*self.kappa*np.einsum('...m,mj->...j',ph*self.amps,self.d)


def fams_all(n,m):
    _,t=square_mesh(n);return [list(range(m)) for _ in range(len(t))]

def fams_trans(n):
    v,t=square_mesh(n); c=v[t].mean(1);return [[0,1] if x[1]<0 else [2] for x in c]


def optimize_with_jac(name,n,exact,z0,fams,nq=12,max_nfev=200):
    z0=np.asarray(z0,float); x0=z0.ravel(); npar=len(x0)
    def unpack(x):
        z=np.asarray(x,float).reshape(-1,2).copy(); z[:,0]%=2*np.pi; return z
    counts={'fun':0,'jac':0}
    def fun(x):
        counts['fun']+=1; z=unpack(x);s,fb=make_family_state(n,exact,z,fams,nq);return physical_skeleton_residual_vector(s)
    def jac(x):
        counts['jac']+=1; z=unpack(x);s,fb=make_family_state(n,exact,z,fams,nq);r,J,*_=all_analytic(s,fb,z);return J
    lb=[];ub=[]
    for j in range(len(z0)):
        lb += [x0[2*j]-2*np.pi,-1.8];ub += [x0[2*j]+2*np.pi,1.8]
    t0=time.perf_counter();rr=least_squares(fun,x0,jac=jac,bounds=(lb,ub),method='trf',x_scale='jac',ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=max_nfev);ta=time.perf_counter()-t0
    za=unpack(rr.x);sa,_=make_family_state(n,exact,za,fams,max(nq,18))
    countsfd={'fun':0}
    def funfd(x):
        countsfd['fun']+=1; z=unpack(x);s,fb=make_family_state(n,exact,z,fams,nq);return physical_skeleton_residual_vector(s)
    t0=time.perf_counter();rrfd=least_squares(funfd,x0,bounds=(lb,ub),method='trf',x_scale='jac',ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=max_nfev,diff_step=2e-5);tf=time.perf_counter()-t0
    zf=unpack(rrfd.x);sf,_=make_family_state(n,exact,zf,fams,max(nq,18))
    return dict(name=name,analytic=rr,za=za,sa=sa,counts=counts,time=ta,fd=rrfd,zf=zf,sf=sf,countsfd=countsfd,timefd=tf)

if __name__=='__main__':
    import pandas as pd
    out=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'generated');os.makedirs(out,exist_ok=True)
    checks=[]
    # deliberately non-optimal states
    ex=ThreeWave(8.)
    z=np.array([[.4,.12],[2.0,-.08],[4.6,.05]])
    s,r,J,R=derivative_check('threewave',2,ex,z,fams_all(2,3),nq=16)
    for row in R:checks.append(['threewave',*row])
    ex69=Transmission(12,69);z69=np.array([[1.0,.07],[4.8,-.05],[.9,.11]])
    s,r,J,R=derivative_check('trans69',2,ex69,z69,fams_trans(2),nq=16)
    for row in R:checks.append(['trans69',*row])
    ex29=Transmission(12,29);z29=np.array([[.55,.1],[5.7,-.07],[.1,1.0]])
    s,r,J,R=derivative_check('trans29',2,ex29,z29,fams_trans(2),nq=16)
    for row in R:checks.append(['trans29',*row])
    df=pd.DataFrame(checks,columns=['case','parameter','rel_dA','rel_dF','rel_dc','rel_dr','rel_gradJ']);df.to_csv(out+'/derivative_checks.csv',index=False)
    print(df.groupby('case')[['rel_dA','rel_dF','rel_dc','rel_dr','rel_gradJ']].max())

    # optimizer comparisons. Use known generic starts from manuscript for transmission; threewave generic.
    opts=[]
    starts=[('threewave',ThreeWave(4.),np.deg2rad([[269.353,0],[130.738,0],[43.929,0]]),fams_all(2,3)),
            ('trans69',ex69,np.array([[np.deg2rad(251.786),0],[np.deg2rad(103.766),0],[np.deg2rad(65.676),0]]),fams_trans(2)),
            ('trans29',ex29,np.array([[np.deg2rad(263.749),-.6095],[np.deg2rad(45.757),.2579],[np.deg2rad(351.014),1.2004]]),fams_trans(2))]
    for nm,exx,z0,ff in starts:
        # deg2rad above on nested array wrongly scales eta zero only okay; for threewave fix theta/eta shape
        if nm=='threewave':
            z0=np.array([[np.deg2rad(269.353),0],[np.deg2rad(130.738),0],[np.deg2rad(43.929),0]])
        res=optimize_with_jac(nm,2,exx,z0,ff,nq=12,max_nfev=120)
        for typ,rr,zf,sf,cnt,tm in [('analytic',res['analytic'],res['za'],res['sa'],res['counts'],res['time']),('finite-diff',res['fd'],res['zf'],res['sf'],res['countsfd'],res['timefd'])]:
            opts.append(dict(case=nm,method=typ,nfev=rr.nfev,njev=getattr(rr,'njev',np.nan),actual_fun_calls=cnt.get('fun',0),actual_jac_calls=cnt.get('jac',0),seconds=tm,J=float(np.dot(physical_skeleton_residual_vector(sf),physical_skeleton_residual_vector(sf))),l2=l2_error(sf,12),angles_deg=';'.join(f'{np.degrees(a):.6f},{b:.6f}' for a,b in zf)))
            print(nm,typ,'nfev',rr.nfev,'fun',cnt.get('fun',0),'jac',cnt.get('jac',0),'J',opts[-1]['J'],'l2',opts[-1]['l2'],'z',zf)
    odf=pd.DataFrame(opts);odf.to_csv(out+'/optimizer_comparison.csv',index=False)
