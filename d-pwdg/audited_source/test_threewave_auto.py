import sys, numpy as np, scipy.linalg as la
from scipy.optimize import least_squares
from scipy.stats import qmc
from pwdg import PWDG, square_mesh
from errors import l2_error as std_l2
from direction_adaptive_automatic import PWDGDictionary,q_complex_angle,l2_error,graph_data,physical_skeleton_residual_vector,physical_skeleton_residual
from graph_riesz import solve_exact

class ThreeWave:
    def __init__(self,k=8.,angles_deg=(17.,123.,251.),amps=(1.,0.7*np.exp(0.4j),0.45*np.exp(-0.8j))):
        self.kappa=float(k); self.angles=np.deg2rad(np.array(angles_deg,float)); self.amps=np.array(amps,complex)
        self.d=np.stack([np.cos(self.angles),np.sin(self.angles)],axis=1)
    def wavenumber(self,y): return np.full(np.shape(y),self.kappa,float)
    def __call__(self,x,y):
        x=np.asarray(x,float); y=np.asarray(y,float)
        X=np.stack([x,y],axis=-1)
        ph=np.exp(1j*self.kappa*np.einsum('...j,mj->...m',X,self.d))
        return ph@self.amps
    def grad(self,x,y):
        x=np.asarray(x,float); y=np.asarray(y,float); X=np.stack([x,y],axis=-1)
        ph=np.exp(1j*self.kappa*np.einsum('...j,mj->...m',X,self.d))
        c=ph*self.amps
        return 1j*self.kappa*np.einsum('...m,mj->...j',c,self.d)

def solve_angles(n,ex,z,nq=18):
    z=np.asarray(z,float).reshape(-1,2); q0=np.array([q_complex_angle(ex.kappa,a,b) for a,b in z])
    v,t=square_mesh(n); q=[q0.copy() for _ in range(len(t))]
    s=PWDGDictionary(n,ex,q,nq=nq); A,F=s.assemble()
    try:s.u=solve_exact(A,F)
    except:s.u=la.solve(A.toarray(),F)
    return s

def optimize(n,ex,m=3,complex_angles=True,seed=47,max_nfev=500,nq=10):
    sob=qmc.Sobol(1,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(m))))[:m,0]
    th=2*np.pi*sob; eta=np.zeros(m); z0=np.stack([th,eta],axis=1); x0=z0.ravel()
    if complex_angles:
        lb=[];ub=[]
        for j in range(m): lb += [x0[2*j]-2*np.pi,-1.0]; ub += [x0[2*j]+2*np.pi,1.0]
    else:
        x0=th.copy(); lb=None;ub=None
    def unpack(x):
        if complex_angles:
            z=np.asarray(x).reshape(m,2).copy(); z[:,0]%=2*np.pi
        else:
            z=np.zeros((m,2));z[:,0]=np.asarray(x)%(2*np.pi)
        return z
    def fun(x):
        try:
            z=unpack(x);s=solve_angles(n,ex,z,nq=nq);r=physical_skeleton_residual_vector(s)
            return np.concatenate([r,1e-7*z[:,1]]) if complex_angles else r
        except Exception:
            s0=solve_angles(n,ex,z0,nq=nq);return np.ones(len(physical_skeleton_residual_vector(s0))+ (m if complex_angles else 0))*1e3
    kw=dict(method='trf',x_scale='jac',ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=max_nfev,diff_step=1e-5)
    if complex_angles: rr=least_squares(fun,x0,bounds=(np.array(lb),np.array(ub)),**kw)
    else: rr=least_squares(fun,x0,**kw)
    z=unpack(rr.x);s=solve_angles(n,ex,z,nq=24);return z,s,rr,z0

if __name__=='__main__':
    n=2;ex=ThreeWave()
    print('exact angles',np.degrees(ex.angles))
    print('uniform baselines')
    for p in [2,3,4,5,7,9]:
        s=PWDG(n,ex,p=p,nq=24);A,F=s.assemble();s.solve();print(p,s.ndof,std_l2(s,12),np.linalg.cond(A.toarray()))
    for comp in [False,True]:
        z,s,rr,z0=optimize(n,ex,3,complex_angles=comp,seed=47,max_nfev=700,nq=10)
        print('complex',comp,'start deg',np.degrees(z0[:,0]),'nfev',rr.nfev,'success',rr.success)
        print('z',[(float(np.degrees(a)),float(b)) for a,b in z])
        print('err',l2_error(s,12),'skel',physical_skeleton_residual(s),'cond',graph_data(s.A))
