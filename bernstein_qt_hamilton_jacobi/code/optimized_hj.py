from __future__ import annotations
import math, time
from types import SimpleNamespace
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import lsmr

from eikonal_c0_pminus1 import (
    multiindices2, bernstein_values, bernstein_gradients,
    triangle_quadrature, tri_geometry, build_c0_assembly,
)
from high_frequency_benchmarks import scaled_structured_tri_mesh, boundary_edges, ZeroFactor


def elevate_local_once(c, p):
    """Exact triangular Bernstein degree elevation p -> p+1."""
    old = multiindices2(p)
    new = multiindices2(p+1)
    pos = {a:i for i,a in enumerate(old)}
    out = np.zeros(len(new), dtype=np.asarray(c).dtype)
    q = p+1
    for j,beta in enumerate(new):
        s = 0.0
        for i in range(3):
            if beta[i] > 0:
                a = list(beta); a[i] -= 1
                s += (beta[i]/q) * c[pos[tuple(a)]]
        out[j] = s
    return out


def elevate_local(c, p_old, p_new):
    c = np.asarray(c)
    for p in range(p_old, p_new):
        c = elevate_local_once(c, p)
    return c


def elevate_global(old_model, old_c, new_model):
    """Degree-elevate a globally C0 field between models on the same mesh."""
    if len(old_model.tris) != len(new_model.tris) or not np.allclose(old_model.verts, new_model.verts):
        raise ValueError('models must share the same mesh')
    if new_model.p < old_model.p:
        raise ValueError('new degree must be >= old degree')
    out = np.zeros(len(new_model.asm.keys), float)
    cnt = np.zeros(len(new_model.asm.keys), float)
    for K in range(len(old_model.tris)):
        loc = old_c[old_model.asm.l2g[K]]
        el = elevate_local(loc, old_model.p, new_model.p)
        gids = new_model.asm.l2g[K]
        np.add.at(out, gids, el)
        np.add.at(cnt, gids, 1.0)
    out /= np.maximum(cnt, 1.0)
    if hasattr(new_model, 'fixed'):
        out[new_model.fixed] = new_model.fixed_vals
    return out


class FastFactoredEikonalPminus1:
    """Vectorized sparse-Jacobian global C0 p-1 factored eikonal solver.

    Same mathematical formulation as FactoredC0EikonalPminus1, but the element
    quadrature is batched and the analytic Jacobian is assembled directly as CSR.
    The default nonlinear iteration is column-scaled sparse Gauss-Newton with LSMR
    and backtracking.  This avoids dense global Jacobians and supports p well into
    the low teens on the benchmark meshes.
    """
    def __init__(self, n, p, problem, factor=None, bounds=((0.,1.),(0.,1.)),
                 quad_order=None):
        self.n=int(n); self.p=int(p); self.problem=problem
        self.factor=factor if factor is not None else ZeroFactor(); self.bounds=bounds
        self.verts,self.tris=scaled_structured_tri_mesh(n,bounds)
        self.asm=build_c0_assembly(self.verts,self.tris,p)
        self.inds=multiindices2(p); self.Nd=len(self.inds)
        self.r=p-1; self.test_inds=multiindices2(self.r); self.Nr=len(self.test_inds)
        # For a quadratic Hamiltonian and degree-p derivatives, 2p+1 is enough
        # for the polynomial part; retain a modest safety margin for nonpolynomial factors.
        self.qorder=quad_order or max(8, 2*p+3)
        ref,wref=triangle_quadrature(self.qorder)
        self.ref=ref; self.wref=wref
        lam=np.column_stack([1-ref[:,0]-ref[:,1],ref[:,0],ref[:,1]])
        self.B=bernstein_values(p,lam)
        self.Q=bernstein_values(self.r,lam)
        E=len(self.tris); nq=len(ref)
        self.PTS=np.empty((E,nq,2),float)
        self.G=np.empty((E,nq,self.Nd,2),float)
        self.W=np.empty((E,nq),float)
        for K,tri in enumerate(self.tris):
            V=self.verts[tri]; _,gl=tri_geometry(V)
            J=np.column_stack((V[1]-V[0],V[2]-V[0])); det=abs(np.linalg.det(J))
            self.PTS[K]=V[0]+ref@J.T
            self.G[K]=bernstein_gradients(p,lam,gl)
            self.W[K]=wref*det
        self.GFAC=self.factor.grad(self.PTS.reshape(-1,2)).reshape(E,nq,2)
        self.N2=self.problem.n2(self.PTS.reshape(-1,2)).reshape(E,nq)
        self.fixed,self.fixed_vals=self._boundary_coefficients()
        allidx=np.arange(len(self.asm.keys)); mask=np.ones(len(allidx),bool); mask[self.fixed]=False
        self.free=allidx[mask]
        self.g2free=np.full(len(allidx),-1,int); self.g2free[self.free]=np.arange(len(self.free))
        self.local_free=self.g2free[self.asm.l2g]
        # constant sparse pattern for all free local columns
        e_idx,b_idx,j_idx=[] ,[] ,[]
        for K in range(E):
            js=np.nonzero(self.local_free[K]>=0)[0]
            if len(js)==0: continue
            # local matrix block Nr x len(js)
            bb=np.repeat(np.arange(self.Nr),len(js))
            jj=np.tile(js,self.Nr)
            e_idx.append(np.full(len(bb),K,int)); b_idx.append(bb); j_idx.append(jj)
        self._pat_e=np.concatenate(e_idx) if e_idx else np.array([],int)
        self._pat_b=np.concatenate(b_idx) if b_idx else np.array([],int)
        self._pat_j=np.concatenate(j_idx) if j_idx else np.array([],int)
        self._pat_rows=self._pat_e*self.Nr+self._pat_b
        self._pat_cols=self.local_free[self._pat_e,self._pat_j]
        self._base=None
        self._cache_z=None; self._cache_R=None; self._cache_gh=None; self._cache_J=None

    def tau_exact(self,x): return self.problem.u(x)-self.factor.u(x)
    def grad_tau_exact(self,x): return self.problem.grad(x)-self.factor.grad(x)

    def _boundary_coefficients(self):
        # Use exact zero detection first (important for focal factors), otherwise
        # stable oversampled least squares on Chebyshev-Lobatto nodes.
        b_edges=boundary_edges(self.tris); fixed_vals={}
        key_to_gid={k:i for i,k in enumerate(self.asm.keys)}
        bverts=sorted(set(v for e in b_edges for v in e))
        for v in bverts:
            gid=key_to_gid.get(('v',int(v)))
            if gid is not None: fixed_vals[gid]=float(self.tau_exact(self.verts[[v]])[0])
        # Chebyshev-Lobatto interpolation nodes in [0,1]
        k=np.arange(self.p+1)
        ts=0.5*(1-np.cos(np.pi*k/self.p)) if self.p>0 else np.array([0.])
        from high_frequency_benchmarks import bernstein_1d_matrix
        B1=bernstein_1d_matrix(self.p,ts)
        for a,b in b_edges:
            va,vb=self.verts[a],self.verts[b]
            pts=(1-ts[:,None])*va+ts[:,None]*vb
            vals=self.tau_exact(pts)
            if np.max(np.abs(vals)) < 5e-15:
                coeff=np.zeros(self.p+1)
            else:
                coeff=la.solve(B1,vals,assume_a='gen',check_finite=False)
            for kk in range(1,self.p):
                gid=key_to_gid.get(('e',int(a),int(b),kk))
                if gid is not None: fixed_vals[gid]=float(coeff[kk])
        fixed=np.array(sorted(fixed_vals),int)
        vals=np.array([fixed_vals[g] for g in fixed],float)
        return fixed,vals

    def initial_coefficients(self):
        c=self.tau_exact(self.asm.dof_coords)
        c[self.fixed]=self.fixed_vals
        return c

    def unpack(self,z):
        c=self._base.copy(); c[self.free]=z; return c

    def _residual_state(self,z):
        if self._cache_z is not None and np.array_equal(z,self._cache_z):
            return self._cache_R,self._cache_gh
        c=self.unpack(z); C=c[self.asm.l2g]
        gtau=np.einsum('eqjd,ej->eqd',self.G,C,optimize=True)
        gh=gtau+self.GFAC
        F=0.5*(np.einsum('eqd,eqd->eq',gh,gh,optimize=True)-self.N2)
        Rloc=np.einsum('qb,eq,eq->eb',self.Q,self.W,F,optimize=True)
        R=Rloc.ravel()
        self._cache_z=z.copy(); self._cache_R=R; self._cache_gh=gh; self._cache_J=None
        return R,gh

    def residual(self,z): return self._residual_state(z)[0]

    def jacobian(self,z):
        R,gh=self._residual_state(z)
        if self._cache_J is not None: return self._cache_J
        adv=np.einsum('eqd,eqjd->eqj',gh,self.G,optimize=True)
        Jloc=np.einsum('qb,eq,eqj->ebj',self.Q,self.W,adv,optimize=True)
        data=Jloc[self._pat_e,self._pat_b,self._pat_j]
        J=sp.csr_matrix((data,(self._pat_rows,self._pat_cols)),shape=(len(R),len(self.free)))
        self._cache_J=J
        return J

    def residual_and_jac(self,z,want_jac=True):
        R=self.residual(z)
        return R,(self.jacobian(z) if want_jac else None)

    def solve(self,max_nfev=60,verbose=0,initial_full=None,rtol=2e-11,gtol=5e-11,
              lsmr_tol=2e-10,lsmr_maxiter=None):
        self._base=self.initial_coefficients() if initial_full is None else np.asarray(initial_full,float).copy()
        self._base[self.fixed]=self.fixed_vals
        z=self._base[self.free].copy()
        self._cache_z=None
        t0=time.perf_counter(); nfev=0; njev=0; success=False; message='maximum iterations reached'
        R=self.residual(z); nfev+=1
        cost0=0.5*np.dot(R,R); cost=cost0
        if lsmr_maxiter is None: lsmr_maxiter=min(800,max(80,4*len(self.free)))
        opt=np.inf
        for it in range(max_nfev):
            J=self.jacobian(z); njev+=1
            g=np.asarray(J.T@R).ravel()
            col=np.sqrt(np.asarray(J.power(2).sum(axis=0)).ravel())
            col=np.maximum(col,1e-14)
            # Variable-scaled first-order condition.  Raw Bernstein moments shrink
            # rapidly with p, so an unscaled gradient gives false convergence.
            opt=float(np.linalg.norm(g/col,np.inf)) if len(g) else 0.0
            if verbose: print(f'it={it:2d} cost={cost:.3e} rms={np.linalg.norm(R)/math.sqrt(len(R)):.3e} opt_scaled={opt:.3e}')
            if np.linalg.norm(R) <= rtol*max(1.0,math.sqrt(2*cost0)) or opt <= gtol:
                success=True; message='residual/gradient tolerance reached'; break
            S=sp.diags(1.0/col)
            Js=J@S
            y=lsmr(Js,-R,atol=lsmr_tol,btol=lsmr_tol,maxiter=lsmr_maxiter)[0]
            step=(1.0/col)*y
            sn=float(np.linalg.norm(step))
            if sn <= 1e-13*(1+np.linalg.norm(z)):
                success=True; message='step tolerance reached'; break
            # Armijo-like backtracking on the actual residual norm.
            alpha=1.0; accepted=False
            pred=max(1e-30,float(np.dot(R,R)-np.dot(R+J@step,R+J@step)))
            for _ in range(12):
                zt=z+alpha*step
                Rt=self.residual(zt); nfev+=1
                ct=0.5*np.dot(Rt,Rt)
                if ct < cost - 1e-4*alpha*max(pred,0.0) or ct < cost:
                    z,R,cost=zt,Rt,ct; accepted=True; break
                alpha*=0.5
            if not accepted:
                message='line search stalled'; break
        elapsed=time.perf_counter()-t0
        self._base[self.free]=z
        sol=SimpleNamespace(x=z.copy(),success=success,nfev=nfev,njev=njev,cost=float(cost),
                            optimality=float(opt),message=message)
        return self._base.copy(),sol,elapsed

    def errors(self,c,order=None):
        order=order or max(12,2*self.p+5)
        ref,wref=triangle_quadrature(order)
        lam=np.column_stack([1-ref[:,0]-ref[:,1],ref[:,0],ref[:,1]])
        B=bernstein_values(self.p,lam)
        L2=H1=Nu=Ng=0.0; Linf=0.0
        for K,tri in enumerate(self.tris):
            V=self.verts[tri]; _,gl=tri_geometry(V)
            JJ=np.column_stack((V[1]-V[0],V[2]-V[0])); det=abs(np.linalg.det(JJ))
            pts=V[0]+ref@JJ.T; W=wref*det; G=bernstein_gradients(self.p,lam,gl)
            ck=c[self.asm.l2g[K]]; tau=B@ck; gt=np.einsum('qjd,j->qd',G,ck,optimize=True)
            uh=self.factor.u(pts)+tau; gh=self.factor.grad(pts)+gt
            ue=self.problem.u(pts); ge=self.problem.grad(pts); de=uh-ue
            L2+=np.sum(W*de*de); H1+=np.sum(W*np.sum((gh-ge)**2,axis=1))
            Nu+=np.sum(W*ue*ue); Ng+=np.sum(W*np.sum(ge*ge,axis=1)); Linf=max(Linf,float(np.max(np.abs(de))))
        return math.sqrt(L2/max(Nu,1e-30)),math.sqrt(H1/max(Ng,1e-30)),Linf

    def summary(self,c,sol,seconds):
        e2,e1,ei=self.errors(c)
        self._base=c.copy(); self._cache_z=None
        R=self.residual(c[self.free])
        return dict(n=self.n,ntri=len(self.tris),p=self.p,test_degree=self.p-1,
                    full_c0_dofs=len(self.asm.keys),fixed_boundary_dofs=len(self.fixed),free_dofs=len(self.free),
                    residual_rows=len(self.tris)*self.Nr,retained_local_dim=self.p+1,
                    relL2=e2,relH1=e1,absLinf=ei,residual_rms=float(np.linalg.norm(R)/math.sqrt(len(R))),
                    success=bool(sol.success),nfev=int(sol.nfev),njev=int(sol.njev),seconds=float(seconds),
                    solver='vectorized sparse Gauss-Newton/LSMR')

class FastHamiltonJacobiPminus1:
    """Vectorized sparse-Jacobian solver for H(x,grad u)=0 with exact boundary trace."""
    def __init__(self,n,p,problem,quad_order=None):
        self.n=int(n);self.p=int(p);self.problem=problem
        from eikonal_c0_pminus1 import structured_tri_mesh
        self.verts,self.tris=structured_tri_mesh(n)
        self.asm=build_c0_assembly(self.verts,self.tris,p)
        self.inds=multiindices2(p);self.Nd=len(self.inds);self.r=p-1;self.Nr=len(multiindices2(self.r))
        self.qorder=quad_order or max(10,2*p+5)
        ref,wref=triangle_quadrature(self.qorder);self.ref=ref;self.wref=wref
        lam=np.column_stack([1-ref[:,0]-ref[:,1],ref[:,0],ref[:,1]])
        self.B=bernstein_values(p,lam);self.Q=bernstein_values(self.r,lam)
        E=len(self.tris);nq=len(ref)
        self.PTS=np.empty((E,nq,2));self.G=np.empty((E,nq,self.Nd,2));self.W=np.empty((E,nq))
        for K,tri in enumerate(self.tris):
            V=self.verts[tri];_,gl=tri_geometry(V);JJ=np.column_stack((V[1]-V[0],V[2]-V[0]));det=abs(np.linalg.det(JJ))
            self.PTS[K]=V[0]+ref@JJ.T;self.G[K]=bernstein_gradients(p,lam,gl);self.W[K]=wref*det
        # Exact boundary values are converted to Bernstein edge coefficients stably.
        self.fixed,self.fixed_vals=self._boundary_coefficients()
        allidx=np.arange(len(self.asm.keys));mask=np.ones(len(allidx),bool);mask[self.fixed]=False;self.free=allidx[mask]
        self.g2free=np.full(len(allidx),-1,int);self.g2free[self.free]=np.arange(len(self.free));self.local_free=self.g2free[self.asm.l2g]
        pe=[];pb=[];pj=[]
        for K in range(E):
            js=np.nonzero(self.local_free[K]>=0)[0]
            bb=np.repeat(np.arange(self.Nr),len(js));jj=np.tile(js,self.Nr)
            pe.append(np.full(len(bb),K,int));pb.append(bb);pj.append(jj)
        self._pat_e=np.concatenate(pe);self._pat_b=np.concatenate(pb);self._pat_j=np.concatenate(pj)
        self._pat_rows=self._pat_e*self.Nr+self._pat_b;self._pat_cols=self.local_free[self._pat_e,self._pat_j]
        self._base=None;self._cache_z=None;self._cache_R=None;self._cache_gh=None;self._cache_J=None

    def _boundary_coefficients(self):
        from high_frequency_benchmarks import bernstein_1d_matrix
        b_edges=boundary_edges(self.tris); kv={}; key_to_gid={k:i for i,k in enumerate(self.asm.keys)}
        bverts=sorted(set(v for e in b_edges for v in e))
        for v in bverts:
            gid=key_to_gid.get(('v',int(v)));kv[gid]=float(self.problem.u(self.verts[[v]])[0])
        k=np.arange(self.p+1);ts=0.5*(1-np.cos(np.pi*k/self.p));B1=bernstein_1d_matrix(self.p,ts)
        for a,b in b_edges:
            va,vb=self.verts[a],self.verts[b];pts=(1-ts[:,None])*va+ts[:,None]*vb;vals=self.problem.u(pts)
            coeff=la.solve(B1,vals,check_finite=False)
            for kk in range(1,self.p):
                gid=key_to_gid.get(('e',int(a),int(b),kk))
                if gid is not None:kv[gid]=float(coeff[kk])
        fixed=np.array(sorted(kv),int);vals=np.array([kv[g] for g in fixed])
        return fixed,vals

    def initial_coefficients(self):
        # Domain-point values are a robust high-order seed; boundary B-coefficients override.
        c=self.problem.u(self.asm.dof_coords).copy();c[self.fixed]=self.fixed_vals;return c
    def unpack(self,z):c=self._base.copy();c[self.free]=z;return c
    def _state(self,z):
        if self._cache_z is not None and np.array_equal(z,self._cache_z):return self._cache_R,self._cache_gh
        c=self.unpack(z);C=c[self.asm.l2g];gh=np.einsum('eqjd,ej->eqd',self.G,C,optimize=True)
        pts=self.PTS.reshape(-1,2);F=self.problem.H(pts,gh.reshape(-1,2)).reshape(len(self.tris),-1)
        R=np.einsum('qb,eq,eq->eb',self.Q,self.W,F,optimize=True).ravel()
        self._cache_z=z.copy();self._cache_R=R;self._cache_gh=gh;self._cache_J=None;return R,gh
    def residual(self,z):return self._state(z)[0]
    def jacobian(self,z):
        R,gh=self._state(z)
        if self._cache_J is not None:return self._cache_J
        a=self.problem.Hp(self.PTS.reshape(-1,2),gh.reshape(-1,2)).reshape(gh.shape)
        adv=np.einsum('eqd,eqjd->eqj',a,self.G,optimize=True)
        Jloc=np.einsum('qb,eq,eqj->ebj',self.Q,self.W,adv,optimize=True)
        data=Jloc[self._pat_e,self._pat_b,self._pat_j]
        J=sp.csr_matrix((data,(self._pat_rows,self._pat_cols)),shape=(len(R),len(self.free)));self._cache_J=J;return J
    def solve(self,max_nfev=50,initial_full=None,gtol=1e-14,rtol=2e-12,lsmr_tol=3e-12,lsmr_maxiter=None,verbose=0):
        self._base=self.initial_coefficients() if initial_full is None else np.asarray(initial_full,float).copy();self._base[self.fixed]=self.fixed_vals
        z=self._base[self.free].copy();self._cache_z=None;t0=time.perf_counter();nfev=0;njev=0;success=False;msg='maximum iterations reached'
        R=self.residual(z);nfev+=1;cost0=0.5*np.dot(R,R);cost=cost0;opt=np.inf
        if lsmr_maxiter is None:lsmr_maxiter=min(1000,max(100,4*len(self.free)))
        for it in range(max_nfev):
            J=self.jacobian(z);njev+=1;g=np.asarray(J.T@R).ravel();col=np.sqrt(np.asarray(J.power(2).sum(axis=0)).ravel());col=np.maximum(col,1e-14)
            opt=float(np.linalg.norm(g/col,np.inf)) if len(g) else 0.0
            if verbose:print('it',it,'cost',cost,'rms',np.linalg.norm(R)/math.sqrt(len(R)),'opt',opt)
            if np.linalg.norm(R)<=rtol*max(1.0,math.sqrt(2*cost0)) or opt<=gtol:success=True;msg='residual/gradient tolerance reached';break
            Js=J@sp.diags(1/col);y=lsmr(Js,-R,atol=lsmr_tol,btol=lsmr_tol,maxiter=lsmr_maxiter)[0];step=y/col
            if np.linalg.norm(step)<=1e-13*(1+np.linalg.norm(z)):success=True;msg='step tolerance reached';break
            pred=max(1e-30,float(np.dot(R,R)-np.dot(R+J@step,R+J@step)));alpha=1.;accepted=False
            for _ in range(12):
                zt=z+alpha*step;Rt=self.residual(zt);nfev+=1;ct=.5*np.dot(Rt,Rt)
                if ct<cost-1e-4*alpha*max(pred,0.) or ct<cost:z,R,cost=zt,Rt,ct;accepted=True;break
                alpha*=.5
            if not accepted:msg='line search stalled';break
        sec=time.perf_counter()-t0;self._base[self.free]=z
        sol=SimpleNamespace(x=z.copy(),success=success,nfev=nfev,njev=njev,cost=float(cost),optimality=float(opt),message=msg)
        return self._base.copy(),sol,sec
    def errors(self,c,order=None):
        q=order or max(12,2*self.p+6);ref,wref=triangle_quadrature(q);lam=np.column_stack([1-ref[:,0]-ref[:,1],ref[:,0],ref[:,1]])
        B=bernstein_values(self.p,lam);l2=l2u=h1=h1u=0.
        for K,tri in enumerate(self.tris):
            V=self.verts[tri];_,gl=tri_geometry(V);JJ=np.column_stack((V[1]-V[0],V[2]-V[0]));det=abs(np.linalg.det(JJ));pts=V[0]+ref@JJ.T;W=wref*det
            G=bernstein_gradients(self.p,lam,gl);ck=c[self.asm.l2g[K]];uh=B@ck;gh=np.einsum('qjd,j->qd',G,ck,optimize=True);ue=self.problem.u(pts);ge=self.problem.grad(pts)
            l2+=np.sum(W*(uh-ue)**2);l2u+=np.sum(W*ue**2);h1+=np.sum(W*np.sum((gh-ge)**2,axis=1));h1u+=np.sum(W*np.sum(ge*ge,axis=1))
        return math.sqrt(l2/l2u),math.sqrt(h1/h1u)

class FastMultiplicativeFactoredBB:
    """Sparse/vectorized solver for tau=r psi_h with only psi(source) fixed."""
    def __init__(self,nx,ny,p,problem,bounds=((0.,4.),(0.,8.)),quad_order=None):
        from treister_haber_benchmarks import rectangular_tri_mesh
        self.nx=int(nx);self.ny=int(ny);self.p=int(p);self.problem=problem;self.bounds=bounds
        self.verts,self.tris=rectangular_tri_mesh(nx,ny,bounds);self.asm=build_c0_assembly(self.verts,self.tris,p)
        self.inds=multiindices2(p);self.Nd=len(self.inds);self.rdeg=p-1;self.Nr=len(multiindices2(self.rdeg))
        self.qorder=quad_order or max(8,2*p+3);ref,wref=triangle_quadrature(self.qorder);self.ref=ref;self.wref=wref
        lam=np.column_stack([1-ref[:,0]-ref[:,1],ref[:,0],ref[:,1]])
        self.B=bernstein_values(p,lam);self.Q=bernstein_values(self.rdeg,lam)
        E=len(self.tris);nq=len(ref);self.PTS=np.empty((E,nq,2));self.G=np.empty((E,nq,self.Nd,2));self.W=np.empty((E,nq))
        self.RR=np.empty((E,nq));self.GR=np.empty((E,nq,2));self.N2=np.empty((E,nq))
        for K,tri in enumerate(self.tris):
            V=self.verts[tri];_,gl=tri_geometry(V);JJ=np.column_stack((V[1]-V[0],V[2]-V[0]));det=abs(np.linalg.det(JJ));pts=V[0]+ref@JJ.T
            self.PTS[K]=pts;self.G[K]=bernstein_gradients(p,lam,gl);self.W[K]=wref*det
            z=pts-self.problem.source;rr=np.linalg.norm(z,axis=1);gr=np.zeros_like(z);m=rr>1e-14;gr[m]=z[m]/rr[m,None]
            self.RR[K]=rr;self.GR[K]=gr;self.N2[K]=self.problem.n2(pts)
        d=np.linalg.norm(self.asm.dof_coords-self.problem.source[None,:],axis=1);gid=int(np.argmin(d))
        if d[gid]>1e-11:raise ValueError('source must be a global domain point')
        self.fixed=np.array([gid],int);self.fixed_vals=np.array([self.problem.psi_source()],float)
        allidx=np.arange(len(self.asm.keys));mask=np.ones(len(allidx),bool);mask[self.fixed]=False;self.free=allidx[mask]
        self.g2free=np.full(len(allidx),-1,int);self.g2free[self.free]=np.arange(len(self.free));self.local_free=self.g2free[self.asm.l2g]
        pe=[];pb=[];pj=[]
        for K in range(E):
            js=np.nonzero(self.local_free[K]>=0)[0];bb=np.repeat(np.arange(self.Nr),len(js));jj=np.tile(js,self.Nr)
            pe.append(np.full(len(bb),K,int));pb.append(bb);pj.append(jj)
        self._pat_e=np.concatenate(pe);self._pat_b=np.concatenate(pb);self._pat_j=np.concatenate(pj)
        self._pat_rows=self._pat_e*self.Nr+self._pat_b;self._pat_cols=self.local_free[self._pat_e,self._pat_j]
        self._base=None;self._cache_z=None;self._cache_R=None;self._cache_gtau=None;self._cache_J=None
    def initial_coefficients(self):
        c=np.full(len(self.asm.keys),self.problem.psi_source(),float);c[self.fixed]=self.fixed_vals;return c
    def unpack(self,z):c=self._base.copy();c[self.free]=z;c[self.fixed]=self.fixed_vals;return c
    def _state(self,z):
        if self._cache_z is not None and np.array_equal(z,self._cache_z):return self._cache_R,self._cache_gtau
        c=self.unpack(z);C=c[self.asm.l2g];psi=np.einsum('qj,ej->eq',self.B,C,optimize=True);gpsi=np.einsum('eqjd,ej->eqd',self.G,C,optimize=True)
        gtau=psi[:,:,None]*self.GR+self.RR[:,:,None]*gpsi
        F=.5*(np.einsum('eqd,eqd->eq',gtau,gtau,optimize=True)-self.N2)
        R=np.einsum('qb,eq,eq->eb',self.Q,self.W,F,optimize=True).ravel()
        self._cache_z=z.copy();self._cache_R=R;self._cache_gtau=gtau;self._cache_J=None;return R,gtau
    def residual(self,z):return self._state(z)[0]
    def jacobian(self,z):
        R,gtau=self._state(z)
        if self._cache_J is not None:return self._cache_J
        dgtau=self.B[None,:,:,None]*self.GR[:,:,None,:]+self.RR[:,:,None,None]*self.G
        adv=np.einsum('eqd,eqjd->eqj',gtau,dgtau,optimize=True)
        Jloc=np.einsum('qb,eq,eqj->ebj',self.Q,self.W,adv,optimize=True)
        data=Jloc[self._pat_e,self._pat_b,self._pat_j]
        J=sp.csr_matrix((data,(self._pat_rows,self._pat_cols)),shape=(len(R),len(self.free)));self._cache_J=J;return J
    def solve(self,max_nfev=50,initial_full=None,gtol=1e-13,rtol=1e-11,lsmr_tol=1e-10,lsmr_maxiter=None,verbose=0):
        self._base=self.initial_coefficients() if initial_full is None else np.asarray(initial_full,float).copy();self._base[self.fixed]=self.fixed_vals
        z=self._base[self.free].copy();self._cache_z=None;t0=time.perf_counter();R=self.residual(z);nfev=1;njev=0;cost0=.5*np.dot(R,R);cost=cost0;opt=np.inf;success=False;msg='maximum iterations reached'
        if lsmr_maxiter is None:lsmr_maxiter=min(1200,max(100,3*len(self.free)))
        for it in range(max_nfev):
            J=self.jacobian(z);njev+=1;g=np.asarray(J.T@R).ravel();col=np.sqrt(np.asarray(J.power(2).sum(axis=0)).ravel());col=np.maximum(col,1e-14);opt=float(np.linalg.norm(g/col,np.inf))
            if verbose:print(it,np.linalg.norm(R)/math.sqrt(len(R)),opt)
            if np.linalg.norm(R)<=rtol*max(1.,math.sqrt(2*cost0)) or opt<=gtol:success=True;msg='residual/gradient tolerance reached';break
            Js=J@sp.diags(1/col);y=lsmr(Js,-R,atol=lsmr_tol,btol=lsmr_tol,maxiter=lsmr_maxiter)[0];step=y/col
            alpha=1.;accepted=False
            for _ in range(12):
                zt=z+alpha*step;Rt=self.residual(zt);nfev+=1;ct=.5*np.dot(Rt,Rt)
                if ct<cost:z,R,cost=zt,Rt,ct;accepted=True;break
                alpha*=.5
            if not accepted:msg='line search stalled';break
        sec=time.perf_counter()-t0;self._base[self.free]=z
        sol=SimpleNamespace(x=z.copy(),success=success,nfev=nfev,njev=njev,cost=float(cost),optimality=float(opt),message=msg)
        return self._base.copy(),sol,sec
    def errors(self,c,order=None):
        order=order or max(12,2*self.p+5);ref,wref=triangle_quadrature(order);lam=np.column_stack([1-ref[:,0]-ref[:,1],ref[:,0],ref[:,1]]);B=bernstein_values(self.p,lam)
        e2=u2=area=0.;linf=0.
        for K,tri in enumerate(self.tris):
            V=self.verts[tri];JJ=np.column_stack((V[1]-V[0],V[2]-V[0]));det=abs(np.linalg.det(JJ));pts=V[0]+ref@JJ.T;W=wref*det;rr=np.linalg.norm(pts-self.problem.source,axis=1)
            psi=B@c[self.asm.l2g[K]];uh=rr*psi;ue=self.problem.tau(pts);e=uh-ue;e2+=np.sum(W*e*e);u2+=np.sum(W*ue*ue);area+=np.sum(W);linf=max(linf,float(np.max(np.abs(e))))
        return linf,math.sqrt(e2/area),math.sqrt(e2/max(u2,1e-30))
    def summary(self,c,sol,seconds):
        linf,rms,rel=self.errors(c);self._base=c.copy();self._cache_z=None;R=self.residual(c[self.free]);hx=(self.bounds[0][1]-self.bounds[0][0])/self.nx;hy=(self.bounds[1][1]-self.bounds[1][0])/self.ny
        return dict(case=self.problem.name,nx=self.nx,ny=self.ny,h=max(hx,hy),ntri=len(self.tris),p=self.p,c0_dofs=len(self.asm.keys),free_dofs=len(self.free),residual_rows=len(R),absLinf=linf,meanL2=rms,relL2=rel,residual_rms=float(np.linalg.norm(R)/math.sqrt(len(R))),success=sol.success,nfev=sol.nfev,seconds=float(seconds),solver='vectorized sparse Gauss-Newton/LSMR')
