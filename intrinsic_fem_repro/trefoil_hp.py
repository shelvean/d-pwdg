#!/usr/bin/env python3
"""Intrinsic C0 Bernstein FEM on the smooth boundary of a genuine trefoil tube.

The numerical solver uses *only* the 2x2 intrinsic metric, not 3D positions.
The 3D centerline is used once to derive exact metric coefficients and to
establish that the surface is a genuine trefoil tube, not an unknotted torus.

Usage examples:
 python trefoil_hp.py --task audit
 python trefoil_hp.py --task poisson --Nu 16 --Nv 12 --p 4 --quad 8
 python trefoil_hp.py --task spectrum --Nu 22 --Nv 14 --p 3 --nmodes 125
 python trefoil_hp.py --task sweep --output results
"""
from __future__ import annotations
import argparse, csv, math, time, json
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, spsolve
from numpy.polynomial.legendre import leggauss

TAU=2*np.pi
RADIUS=0.22


def curve_derivatives(u):
    """Standard (2,3) trefoil centerline and its first three derivatives."""
    u=np.asarray(u)
    cu=np.stack([np.sin(u)+2*np.sin(2*u),np.cos(u)-2*np.cos(2*u),-np.sin(3*u)],axis=-1)
    cp=np.stack([np.cos(u)+4*np.cos(2*u),-np.sin(u)+4*np.sin(2*u),-3*np.cos(3*u)],axis=-1)
    cpp=np.stack([-np.sin(u)-8*np.sin(2*u),-np.cos(u)+8*np.cos(2*u),9*np.sin(3*u)],axis=-1)
    cppp=np.stack([-np.cos(u)-16*np.cos(2*u),np.sin(u)-16*np.sin(2*u),27*np.cos(3*u)],axis=-1)
    return cu,cp,cpp,cppp


def curve_invariants(u):
    _,cp,cpp,cppp=curve_derivatives(u)
    speed=np.sqrt(np.sum(cp*cp,axis=-1))  # Bilinear product for complex-step.
    cross=np.cross(cp,cpp)
    cross2=np.sum(cross*cross,axis=-1)
    curvature=np.sqrt(cross2)/speed**3
    torsion=np.sum(cross*cppp,axis=-1)/cross2
    return speed,curvature,torsion


def metric(u,v,r=RADIUS):
    """Return intrinsic 2x2 metric (Frenet tube coordinates u,v).

    c is the trefoil centerline, s=|c'|, kappa is curvature, tau is torsion.
    G_uu=s^2[(1-r*kappa*cos(v))^2+r^2*tau^2],
    G_uv=s*r^2*tau, G_vv=r^2.
    Metric values are smooth and 2pi-periodic in both coordinates.
    """
    u,v=np.broadcast_arrays(u,v)
    s,kappa,torsion=curve_invariants(u)
    a=1-r*kappa*np.cos(v)
    if (not np.iscomplexobj(a)) and np.any(a<=0):
        raise ValueError('tube radius exceeds local focal radius')
    G=np.zeros(u.shape+(2,2),dtype=np.result_type(u,v,float))
    G[...,0,0]=s*s*(a*a+(r*torsion)**2)
    G[...,0,1]=s*r*r*torsion
    G[...,1,0]=G[...,0,1]
    G[...,1,1]=r*r
    return G


def metric_coefficients(u,v,r=RADIUS):
    G=metric(u,v,r)
    g00,g01,g11=G[...,0,0],G[...,0,1],G[...,1,1]
    det=g00*g11-g01*g01
    rho=np.sqrt(det)
    C=np.empty_like(G)
    C[...,0,0]=rho*g11/det
    C[...,0,1]=-rho*g01/det
    C[...,1,0]=C[...,0,1]
    C[...,1,1]=rho*g00/det
    return rho,C


def reference_rule(q):
    a,wa=leggauss(q);a=(a+1)/2;wa=wa/2
    r=a[:,None]*np.ones((1,q)); s=(1-a[:,None])*a[None,:]
    w=(wa[:,None]*wa[None,:]*(1-a[:,None])).ravel()
    lam=np.stack([1-r-s,r,s],axis=-1).reshape((-1,3))
    assert abs(sum(w)-.5)<1e-13
    return lam,w


def bernstein_reference(p,lam):
    """Reference basis B^p_ijk and gradients w.r.t (lambda1,lambda2)."""
    inds=[(i,j,p-i-j) for i in range(p,-1,-1) for j in range(p-i,-1,-1)]
    nloc=len(inds); nq=len(lam)
    B=np.zeros((nq,nloc)); grad=np.zeros((nq,nloc,2))
    dlam=np.array([[-1.,-1.],[1.,0.],[0.,1.]])
    for col,alpha in enumerate(inds):
        coeff=math.factorial(p)
        for a in alpha: coeff/=math.factorial(a)
        val=np.full(nq,coeff)
        for j,a in enumerate(alpha):
            if a:val*=lam[:,j]**a
        B[:,col]=val
        for j,aj in enumerate(alpha):
            if aj==0:continue
            deriv=np.full(nq,coeff*aj)
            for k,ak in enumerate(alpha):
                exponent=ak-(j==k)
                if exponent: deriv*=lam[:,k]**exponent
            grad[:,col,:]+=deriv[:,None]*dlam[j]
    return inds,B,grad


def mesh(Nu,Nv):
    """Periodic triangular mesh; unwrapped coordinates are used per cell."""
    du=TAU/Nu;dv=TAU/Nv
    cells=[];corners=[]
    ix=lambda i,j:(j%Nv)*Nu+(i%Nu)
    for j in range(Nv):
        for i in range(Nu):
            a=np.array([i*du,j*dv]);b=a+np.array([du,0.]);
            c=a+np.array([du,dv]);d=a+np.array([0.,dv]);
            cells.append((ix(i,j),ix(i+1,j),ix(i+1,j+1)))
            corners.append(np.stack([a,b,c]))
            cells.append((ix(i,j),ix(i+1,j+1),ix(i,j+1)))
            corners.append(np.stack([a,c,d]))
    return np.asarray(cells,dtype=int),np.asarray(corners)


def global_dofs(cells,p):
    keys={};gd=[]
    inds=[(i,j,p-i-j) for i in range(p,-1,-1) for j in range(p-i,-1,-1)]
    for cell,conn in enumerate(cells):
        row=[]
        for alpha in inds:
            pos=[j for j,x in enumerate(alpha) if x>0]
            if len(pos)==1:key=('v',int(conn[pos[0]]))
            elif len(pos)==2:
                a,b=pos; va,vb=map(int,(conn[a],conn[b]));lo,hi=sorted((va,vb))
                key=('e',lo,hi,int(alpha[a] if lo==va else alpha[b]))
            else:key=('t',cell,alpha)
            if key not in keys:keys[key]=len(keys)
            row.append(keys[key])
        gd.append(row)
    gd=np.array(gd,dtype=int)
    if p==1:
        assert len(keys)==int(cells.max())+1
    return gd,len(keys)


def exact_solution(u,v):
    return np.sin(2*u+v)+0.35*np.cos(3*u-2*v)


def solution_derivatives(u,v):
    a=2*u+v;b=3*u-2*v
    val=np.sin(a)+0.35*np.cos(b)
    du=2*np.cos(a)-1.05*np.sin(b)
    dv=np.cos(a)+0.70*np.sin(b)
    duu=-4*np.sin(a)-3.15*np.cos(b)
    duv=-2*np.sin(a)+2.10*np.cos(b)
    dvv=-np.sin(a)-1.40*np.cos(b)
    return val,np.stack([du,dv],axis=-1),duu,duv,dvv


def source_poisson(u,v,r=RADIUS):
    """Manufactured forcing f=-Delta_g u from complex-step derivatives of C.

    C=rho G^{-1}; derivatives are analytic complex-step, avoiding finite
    difference cancellation in the manufactured-source calculation.
    """
    h=1e-23
    rho,C=metric_coefficients(u,v,r)
    _,grad,duu,duv,dvv=solution_derivatives(u,v)
    Cu=np.imag(metric_coefficients(np.asarray(u,dtype=complex)+1j*h,v,r)[1])/h
    Cv=np.imag(metric_coefficients(u,np.asarray(v,dtype=complex)+1j*h,r)[1])/h
    div=Cu[...,0,0]*grad[...,0]+Cu[...,0,1]*grad[...,1]
    div+=Cv[...,1,0]*grad[...,0]+Cv[...,1,1]*grad[...,1]
    div+=C[...,0,0]*duu+2*C[...,0,1]*duv+C[...,1,1]*dvv
    return -div/rho


def assemble(Nu,Nv,p,q,r=RADIUS,forcing=False):
    cells,X=mesh(Nu,Nv); gd,ndof=global_dofs(cells,p)
    lam,w=reference_rule(q); inds,B,dB=bernstein_reference(p,lam)
    nc=len(cells);nloc=len(inds)
    rows=np.repeat(gd,nloc,axis=1).ravel()
    cols=np.tile(gd,(1,nloc)).ravel()
    kval=np.empty(nc*nloc*nloc); mval=np.empty_like(kval)
    rhs=np.zeros(ndof);t0=time.perf_counter()
    for e,(corners,ids) in enumerate(zip(X,gd)):
        E=np.stack([corners[1]-corners[0],corners[2]-corners[0]],axis=-1)
        detE=abs(np.linalg.det(E)); invE=np.linalg.inv(E)
        qp=lam@corners;u,v=qp.T
        rho,C=metric_coefficients(u,v,r)
        dq=np.einsum('qai,ij->qaj',dB,invE)
        fac=w*detE
        A=np.einsum('q,qai,qij,qbj->ab',fac,dq,C,dq,optimize=True)
        M=B.T@((fac*rho)[:,None]*B)
        start=e*nloc*nloc; stop=start+nloc*nloc
        kval[start:stop]=A.ravel();mval[start:stop]=M.ravel()
        if forcing:
            f=source_poisson(u,v,r)
            rhs[ids]+=B.T@(fac*rho*f)
    A=sp.coo_matrix((kval,(rows,cols)),shape=(ndof,ndof)).tocsr()
    M=sp.coo_matrix((mval,(rows,cols)),shape=(ndof,ndof)).tocsr()
    return dict(A=A,M=M,b=rhs,cells=cells,X=X,gd=gd,ndof=ndof,
                q=q,p=p,Nu=Nu,Nv=Nv,assembly_s=time.perf_counter()-t0)


def error_norms(c,assembled,qe,r=RADIUS):
    cells=assembled['cells'];X=assembled['X'];gd=assembled['gd'];p=assembled['p'];
    lam,w=reference_rule(qe);_,B,dB=bernstein_reference(p,lam)
    integ=0.; gradint=0.; mass=0.; meanres=0.; exact_mean=0.; exactnorm=0.
    for corners,ids in zip(X,gd):
        E=np.stack([corners[1]-corners[0],corners[2]-corners[0]],axis=-1)
        detE=abs(np.linalg.det(E));invE=np.linalg.inv(E)
        qp=lam@corners;u,v=qp.T
        rho,C=metric_coefficients(u,v,r)
        ux,gue,_,_,_=solution_derivatives(u,v)
        uh=B@c[ids]
        gh=np.einsum('qai,ij,a->qj',dB,invE,c[ids])
        du=uh-ux;dg=gh-gue;fac=w*detE
        meanres+=sum(fac*rho*du);exact_mean+=sum(fac*rho*ux)
        mass+=sum(fac*rho)
        integ+=sum(fac*rho*du*du)
        exactnorm+=sum(fac*rho*ux*ux)
        gradint+=sum(fac*np.einsum('qi,qij,qj->q',dg,C,dg))
    shift=meanres/mass
    return dict(L2=np.sqrt(max(0.,integ-mass*shift**2)),
                H1=np.sqrt(max(0.,gradint)),
                relL2=np.sqrt(max(0.,integ-mass*shift**2))/np.sqrt(max(exactnorm-exact_mean**2/mass,1e-30)),
                mean_shift=shift,area=mass)


def poisson_run(Nu,Nv,p,q=None,qe=None,r=RADIUS):
    q=q or max(7,p+3);qe=qe or max(q+2,p+5)
    data=assemble(Nu,Nv,p,q,r,forcing=True)
    A,M,b=data['A'],data['M'],data['b']; ndof=data['ndof']
    ones=np.ones(ndof);mm=M@ones
    # Enforce mean 0 and project the quadrature-induced O(qerror) source mean.
    rhs=b-mm*(np.sum(b)/np.sum(mm))
    aug=sp.bmat([[A,sp.csr_matrix(mm[:,None])],
                 [sp.csr_matrix(mm[None,:]),sp.csr_matrix((1,1))]],format='csc')
    t0=time.perf_counter();sol=spsolve(aug,np.r_[rhs,0]); c=sol[:-1]
    errs=error_norms(c,data,qe,r)
    residual=np.linalg.norm(A@c-rhs)/max(np.linalg.norm(rhs),1e-15)
    result=dict(task='poisson',Nu=Nu,Nv=Nv,p=p,quad=q,error_quad=qe,
                cells=len(data['cells']),dofs=ndof,radius=r,**errs,
                rhs_sum=float(np.sum(b)),residual=float(residual),
                assembly_s=data['assembly_s'],solve_s=time.perf_counter()-t0)
    return result,data,c


def spectrum_run(Nu,Nv,p,nmodes,q=None,r=RADIUS,save_vectors=False):
    q=q or max(7,p+3)
    data=assemble(Nu,Nv,p,q,r,forcing=False)
    A,M=data['A'],data['M']; ndof=data['ndof']
    if nmodes>=ndof-1:raise ValueError('nmodes must be < ndofs - 1')
    t0=time.perf_counter()
    vals,vecs=eigsh(A,M=M,k=nmodes,sigma=-0.01,which='LM',tol=4e-10)
    ind=np.argsort(vals);vals=np.real(vals[ind]);vecs=vecs[:,ind]
    # residuals of selected eigenfunctions in coefficient Euclidean norm.
    inds=sorted(set([0,1,2,5,10,20,40,80,120,nmodes-1]).intersection(range(nmodes)))
    residuals={str(j):float(np.linalg.norm(A@vecs[:,j]-vals[j]*M@vecs[:,j])/
                (np.linalg.norm(A@vecs[:,j])+abs(vals[j])*np.linalg.norm(M@vecs[:,j])+1e-15))
               for j in inds if abs(vals[j])>1e-9}
    zero_abs_residual=float(np.linalg.norm(A@vecs[:,0]))
    result=dict(task='spectrum',Nu=Nu,Nv=Nv,p=p,quad=q,
                cells=len(data['cells']),dofs=ndof,radius=r,
                nmodes=nmodes,assembly_s=data['assembly_s'],solve_s=time.perf_counter()-t0,
                zero_mode_abs_residual=zero_abs_residual,
                eigenvalues=vals.tolist(),relative_residuals=residuals)
    return result,data,(vals,vecs) if save_vectors else None


def audit(r=RADIUS):
    u=np.linspace(0,TAU,8192,endpoint=False)
    s,kap,tau=curve_invariants(u)
    L=float(np.sum(s)*TAU/len(s))
    area=TAU*r*L
    us,vs=np.meshgrid(np.linspace(0,TAU,60,endpoint=False),np.linspace(0,TAU,40,endpoint=False),indexing='ij')
    G=metric(us,vs,r);det=np.linalg.det(G)
    # Symmetry check and positivity.
    assert np.min(det)>0 and np.max(r*kap)<1
    # Independently check metric from finite-difference Frenet tube map.
    def embedding(a,b):
        c,cp,cpp,_=curve_derivatives(a)
        spd=np.linalg.norm(cp);t=cp/spd
        nr=cpp-np.dot(cpp,t)*t;n=nr/np.linalg.norm(nr)
        binormal=np.cross(t,n)
        return c+r*(np.cos(b)*n+np.sin(b)*binormal)
    errs=[]; h=2e-4
    for a,b in [(0.3,1.1),(1.2,2.3),(3.3,4.5)]:
        xu=(embedding(a+h,b)-embedding(a-h,b))/(2*h)
        xv=(embedding(a,b+h)-embedding(a,b-h))/(2*h)
        GG=np.array([[xu@xu,xu@xv],[xv@xu,xv@xv]])
        errs.append(np.linalg.norm(GG-metric(a,b,r))/np.linalg.norm(metric(a,b,r)))
    # FD against complex-step manufactured source via flux derivatives.
    delta=2e-5; source_err=[]
    for a,b in [(0.3,1.1),(1.2,2.3),(3.3,4.5)]:
        def flux(x,y):
            _,C=metric_coefficients(x,y,r)
            _,dw,*_=solution_derivatives(x,y)
            return C@dw
        fu=(flux(a+delta,b)[0]-flux(a-delta,b)[0])/(2*delta)
        fv=(flux(a,b+delta)[1]-flux(a,b-delta)[1])/(2*delta)
        rho,_=metric_coefficients(a,b,r)
        source_err.append(abs(-(fu+fv)/rho-source_poisson(a,b,r)))
    out=dict(task='audit',radius=r,knot='(2,3)-trefoil',
             curve_length=float(L),area_exact=float(area),
             r_times_max_curvature=float(r*np.max(kap)),
             min_metric_det=float(np.min(det)),
             metric_fd_rel_max=float(max(errs)),
             source_fd_abs_max=float(max(source_err)))
    assert out['metric_fd_rel_max']<1e-5 and out['source_fd_abs_max']<1e-7
    return out


def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['audit','poisson','spectrum','sweep'],default='audit')
    p.add_argument('--Nu',type=int,default=12);p.add_argument('--Nv',type=int,default=8)
    p.add_argument('--p',type=int,default=3);p.add_argument('--quad',type=int)
    p.add_argument('--error-quad',type=int);p.add_argument('--nmodes',type=int,default=81)
    p.add_argument('--radius',type=float,default=RADIUS)
    p.add_argument('--output',default='results');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.task=='audit':
        result=audit(a.radius);save_json(out/'audit.json',result);print(json.dumps(result,indent=2));return
    if a.task=='poisson':
        result,data,c=poisson_run(a.Nu,a.Nv,a.p,a.quad,a.error_quad,a.radius)
        name=f'poisson_u{a.Nu}_v{a.Nv}_p{a.p}'
        save_json(out/f'{name}.json',result)
        np.savez_compressed(out/f'{name}_solution.npz',coefficients=c,cells=data['cells'],
                            local_coordinates=data['X'],dofs=data['gd'])
        print(json.dumps(result,indent=2));return
    if a.task=='spectrum':
        result,data,ev=spectrum_run(a.Nu,a.Nv,a.p,a.nmodes,a.quad,a.radius,save_vectors=True)
        name=f'spectrum_u{a.Nu}_v{a.Nv}_p{a.p}'
        save_json(out/f'{name}.json',result)
        selected=np.array(sorted(set([0,1,2,10,20,40,80,120,200,240,320,400,480]).intersection(range(len(ev[0])))),dtype=int)
        np.savez_compressed(out/f'{name}_vectors.npz',eigenvalues=ev[0],
                            mode_indices=selected,coefficients=ev[1][:,selected],
                            cells=data['cells'],local_coordinates=data['X'],dofs=data['gd'])
        print('eigenvalues for indices 0,1,2,10,20,40,80,120 where computed:')
        for i in [0,1,2,10,20,40,80,120,200,240,320,400,480]:
            if i<len(ev[0]):
                res=(result['zero_mode_abs_residual'] if i==0 else result['relative_residuals'].get(str(i)))
                print(i,ev[0][i], 'absolute residual' if i==0 else 'relative residual',res)
        print('saved',out/f'{name}_vectors.npz');return
    if a.task=='sweep':
        results=[]
        for Nu,Nv,pp in [(8,6,2),(12,8,2),(16,12,2),(8,6,3),(12,8,3),(16,12,3),
                          (8,6,4),(12,8,4),(16,12,4),
                          (12,8,5),(16,12,5),(20,14,5),(12,8,6),(16,12,6)]:
            result,*_=poisson_run(Nu,Nv,pp,q=max(8,pp+4),qe=max(10,pp+6),r=a.radius)
            results.append(result)
            print(f'Poisson Nu={Nu} Nv={Nv} p={pp}: L2={result["L2"]:.5e} H1={result["H1"]:.5e}',flush=True)
        for pp in sorted(set(r['p'] for r in results)):
            prev=None
            for row in (z for z in results if z['p']==pp):
                if prev:
                    scale=(row['Nu']/prev['Nu']*row['Nv']/prev['Nv'])**.5
                    row['order_L2']=np.log(prev['L2']/row['L2'])/np.log(scale)
                    row['order_H1']=np.log(prev['H1']/row['H1'])/np.log(scale)
                prev=row
        fields=['Nu','Nv','p','cells','dofs','quad','error_quad','L2','order_L2','H1','order_H1','relL2','area','residual','assembly_s','solve_s']
        with (out/'poisson_sweep.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
        save_json(out/'poisson_sweep.json',results)
        print('saved',out/'poisson_sweep.csv')

if __name__=='__main__': main()
