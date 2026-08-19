import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from pwdg import square_mesh, edges, ALPHA, BETA
from quadrature import gauss01, triangle_rule


class PWDGDictionary:
    """PWDG with an element-dependent dictionary of (possibly complex) wave vectors.

    Each local atom is exp(i q . (x-x_K)), with q.q = k_K^2 (bilinear dot,
    no conjugation). For real q this is an ordinary plane wave; complex q
    allows evanescent Trefftz atoms.
    """
    def __init__(self, n, exact, q_by_element, nq=24, box=(-1.,1.), mesh=None):
        if mesh is None:
            self.v, self.t = square_mesh(n, box)
        else:
            self.v, self.t = mesh
        self.ev, self.et = edges(self.t)
        self.nt = self.t.shape[0]
        self.exact = exact
        self.cent = self.v[self.t].mean(axis=1)
        self.kap = np.asarray(exact.wavenumber(self.cent[:,1]), dtype=float)
        self.q = [np.asarray(qe, dtype=complex).reshape(-1,2) for qe in q_by_element]
        if len(self.q) != self.nt:
            raise ValueError('q_by_element must have one array per element')
        self.nloc = np.array([len(qe) for qe in self.q], dtype=int)
        self.off = np.concatenate([[0], np.cumsum(self.nloc)])
        self.ndof = int(self.off[-1])
        self.gs, self.gw = gauss01(nq)
        self.h = (box[1]-box[0])/n if n else np.nan

    def sl(self,e):
        return slice(self.off[e], self.off[e+1])

    def _basis(self,e,pts):
        qe=self.q[e]
        return np.exp(1j * (pts-self.cent[e]) @ qe.T)

    def uh(self,e,pts):
        if self.nloc[e]==0:
            return np.zeros(len(pts),dtype=complex)
        return self._basis(e,pts) @ self.u[self.sl(e)]

    def grad_uh(self,e,pts):
        if self.nloc[e]==0:
            return np.zeros((len(pts),2),dtype=complex)
        c=self.u[self.sl(e)]
        return (self._basis(e,pts) * (1j*c)[None,:]) @ self.q[e]

    def edge(self,f):
        va,vb=self.v[self.ev[f]]
        L=float(np.linalg.norm(vb-va))
        pts=va[None,:]+self.gs[:,None]*(vb-va)[None,:]
        tang=(vb-va)/L
        return pts,self.gw*L,np.array([tang[1],-tang[0]]),L

    def _outward(self,e,nrm,pts):
        return nrm if nrm @ (pts.mean(axis=0)-self.cent[e]) > 0 else -nrm

    def element_area(self,e):
        a,b,c=self.v[self.t[e]]
        return 0.5*abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))

    def assemble(self):
        N=self.ndof
        F=np.zeros(N,dtype=complex)
        rows=[]; cols=[]; vals=[]
        def push(test,trial,block_test_trial):
            rt=np.arange(self.off[test],self.off[test+1])
            ct=np.arange(self.off[trial],self.off[trial+1])
            if len(rt)==0 or len(ct)==0: return
            rr=np.repeat(rt,len(ct)); cc=np.tile(ct,len(rt))
            rows.append(rr); cols.append(cc); vals.append(np.asarray(block_test_trial).ravel())

        for f in range(self.et.shape[0]):
            pts,wq,nrm0,L=self.edge(f)
            e0,e1=self.et[f]
            if e1>=0:
                n0=self._outward(e0,nrm0,pts)
                xi=0.5*(self.kap[e0]+self.kap[e1])
                for eA,nA in ((e0,n0),(e1,-n0)):
                    for eB,nB in ((e0,n0),(e1,-n0)):
                        if self.nloc[eA]==0 or self.nloc[eB]==0: continue
                        pa,pb=self._basis(eA,pts),self._basis(eB,pts)
                        G=np.einsum('q,ql,qm->lm',wq,pa,np.conj(pb),optimize=True)
                        qA,qB=self.q[eA],self.q[eB]
                        qA_nA=qA@nA
                        qA_nB=qA@nB
                        cqB_nB=np.conj(qB@nB)
                        C=(-0.5j*cqB_nB[None,:]
                           -0.5j*qA_nB[:,None]
                           +1j*(BETA/xi)*qA_nA[:,None]*cqB_nB[None,:]
                           +1j*xi*ALPHA*float(nA@nB))
                        # C/G are (trial,test); push expects (test,trial)
                        push(eB,eA,(C*G).T)
            else:
                e=e0
                if self.nloc[e]==0: continue
                nrm=self._outward(e,nrm0,pts)
                k=self.kap[e]
                pa=self._basis(e,pts)
                qe=self.q[e]
                qn=qe@nrm
                G=np.einsum('q,ql,qm->lm',wq,pa,np.conj(pa),optimize=True)
                C=(-1j*qn[:,None] + 1j*ALPHA*k) * np.ones((1,self.nloc[e]),dtype=complex)
                # above broadcasting creates (trial,test), constant in test
                push(e,e,(C*G).T)
                g=self.exact(pts[:,0],pts[:,1])
                rhs=np.einsum('q,q,qm->m',wq,g,np.conj(pa),optimize=True)
                # test normal derivative factor is conjugate(q_test.n)
                F[self.sl(e)] += 1j*np.conj(qn)*rhs + 1j*ALPHA*k*rhs
        if rows:
            A=sp.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(N,N)).tocsc()
        else:
            A=sp.csc_matrix((N,N),dtype=complex)
        self.A,self.F=A,F
        return A,F

    def solve(self):
        if not hasattr(self,'A'): self.assemble()
        self.u=la.solve(self.A.toarray(),self.F,assume_a='gen')
        return self.u


def q_real(k,theta):
    return k*np.array([np.cos(theta),np.sin(theta)],dtype=complex)


def uniform_dictionary(n, exact, m, angle_offset=0.0):
    v,t=square_mesh(n)
    cent=v[t].mean(axis=1)
    kap=np.asarray(exact.wavenumber(cent[:,1]),dtype=float)
    th=angle_offset+2*np.pi*np.arange(1,m+1)/m
    ds=np.stack([np.cos(th),np.sin(th)],axis=1)
    return [k*ds.astype(complex) for k in kap]


def transmission_exact_direction_reference(n, exact):
    v,t=square_mesh(n)
    cent=v[t].mean(axis=1)
    out=[]
    k1=exact.omega*exact.n1
    for c in cent:
        if c[1] < 0:
            out.append(np.array([[k1*exact.d1,k1*exact.d2],
                                 [k1*exact.d1,-k1*exact.d2]],dtype=complex))
        else:
            out.append(np.array([[exact.K1,exact.K2]],dtype=complex))
    return out


def l2_error(s,nsub=10):
    lam,w=triangle_rule(nsub)
    num=den=0.0
    for e in range(s.nt):
        P=lam@s.v[s.t[e]]
        a=s.element_area(e)
        ue=s.exact(P[:,0],P[:,1])
        num += a*np.sum(w*np.abs(s.uh(e,P)-ue)**2)
        den += a*np.sum(w*np.abs(ue)**2)
    return float(np.sqrt(num/den))


def graph_data(A):
    A=A.toarray() if hasattr(A,'toarray') else np.asarray(A)
    G=(A-A.conj().T)/(2j); G=(G+G.conj().T)/2
    H=(A+A.conj().T)/2; H=(H+H.conj().T)/2
    eg=la.eigvalsh(G)
    out={'gmin':float(eg.min()),'gmax':float(eg.max()),'cond_raw':float(np.linalg.cond(A))}
    if eg.min()>1e-13*max(eg.max(),1):
        lam=la.eigvalsh(H,G)
        sig=np.sqrt(1+lam*lam)
        out.update(cond_gr=float(sig.max()/sig.min()),gamma=float(np.max(np.abs(lam))))
    else:
        out.update(cond_gr=np.nan,gamma=np.nan)
    return out


def reduced_solution_from_dictionary(full, selected):
    """Solve Galerkin system on index subset; return coefficient vector in full dictionary."""
    A=full.A.toarray(); F=full.F
    S=np.array(sorted(selected),dtype=int)
    c=np.zeros(full.ndof,dtype=complex)
    if len(S):
        c[S]=la.solve(A[np.ix_(S,S)],F[S],assume_a='gen')
    full.u=c
    return c


def greedy_graph_select(full, initial, max_add=30, rank_tol=1e-11, candidates=None, batch=1):
    """Global graph-residual greedy selection on a preassembled dictionary.

    Candidate j gets score |r_j| / ||(I-P_G)e_j||_G. The denominator is the
    G-Schur complement relative to the currently selected set. Returns history.
    """
    A=full.A.toarray(); F=full.F
    G=(A-A.conj().T)/(2j); G=(G+G.conj().T)/2
    allidx=np.arange(full.ndof)
    cand_mask=np.ones(full.ndof,dtype=bool) if candidates is None else np.zeros(full.ndof,dtype=bool)
    if candidates is not None: cand_mask[np.asarray(candidates,dtype=int)]=True
    S=list(dict.fromkeys(int(i) for i in initial))
    history=[]
    for it in range(max_add+1):
        Sarr=np.array(sorted(S),dtype=int)
        c=np.zeros(full.ndof,dtype=complex)
        if len(Sarr):
            AS=A[np.ix_(Sarr,Sarr)]
            try: c[Sarr]=la.solve(AS,F[Sarr],assume_a='gen')
            except la.LinAlgError: c[Sarr]=la.lstsq(AS,F[Sarr])[0]
        full.u=c
        r=F-A@c
        # subset metrics
        cond_raw=np.linalg.cond(A[np.ix_(Sarr,Sarr)]) if len(Sarr)>0 else np.nan
        cond_gr=np.nan
        if len(Sarr)>0:
            GS=G[np.ix_(Sarr,Sarr)]; HS=(A[np.ix_(Sarr,Sarr)]+A[np.ix_(Sarr,Sarr)].conj().T)/2
            eg=la.eigvalsh((GS+GS.conj().T)/2)
            if eg.min()>1e-13*max(eg.max(),1):
                try:
                    lam=la.eigvalsh((HS+HS.conj().T)/2,(GS+GS.conj().T)/2)
                    sig=np.sqrt(1+lam*lam); cond_gr=float(sig.max()/sig.min())
                except Exception: pass
        history.append(dict(iter=it,ndof=len(Sarr),selected=Sarr.copy(),l2=l2_error(full),
                            residual=float(la.norm(r)),cond_raw=float(cond_raw),cond_gr=cond_gr))
        if it==max_add: break
        avail=np.where(cand_mask)[0]
        avail=np.setdiff1d(avail,Sarr,assume_unique=False)
        if len(avail)==0: break
        # novelty via Schur complement
        if len(Sarr):
            GS=(G[np.ix_(Sarr,Sarr)]+G[np.ix_(Sarr,Sarr)].conj().T)/2
            # pseudoinverse is deliberate: selected dictionary may be close to its numerical rank limit
            GSinv=la.pinvh(GS,rtol=rank_tol)
            cross=G[np.ix_(Sarr,avail)]
            delta=np.real(np.diag(G[np.ix_(avail,avail)]) - np.einsum('ij,ji->i',cross.conj().T,GSinv@cross))
        else:
            delta=np.real(np.diag(G)[avail])
        scale=np.maximum(np.real(np.diag(G)[avail]),1e-300)
        novelty=delta/scale
        score=np.full(len(avail),-np.inf)
        ok=(delta>0)&(novelty>rank_tol)
        score[ok]=np.abs(r[avail[ok]])/np.sqrt(delta[ok])
        if not np.any(np.isfinite(score)): break
        order=np.argsort(score)[::-1]
        chosen=[]
        for jj in order:
            if np.isfinite(score[jj]):
                chosen.append(int(avail[jj]))
                if len(chosen)>=batch: break
        if not chosen: break
        S.extend(chosen)
        history[-1]['chosen_next']=chosen
        history[-1]['score_next']=[float(score[np.where(avail==j)[0][0]]) for j in chosen]
        history[-1]['novelty_next']=[float(novelty[np.where(avail==j)[0][0]]) for j in chosen]
    return history

def build_region_dictionary(n, exact, lower_q, upper_q):
    v,t=square_mesh(n)
    cent=v[t].mean(axis=1)
    return [np.asarray(lower_q if c[1]<0 else upper_q,dtype=complex).reshape(-1,2) for c in cent]


def solve_region_space(n, exact, lower_q, upper_q, nq=30):
    q=build_region_dictionary(n,exact,lower_q,upper_q)
    s=PWDGDictionary(n,exact,q,nq=nq)
    A,F=s.assemble()
    if s.ndof:
        try:
            from graph_riesz import solve_exact
            s.u=solve_exact(A,F)
        except Exception:
            try: s.u=la.solve(A.toarray(),F,assume_a='gen')
            except la.LinAlgError: s.u=la.lstsq(A.toarray(),F)[0]
    else:
        s.u=np.zeros(0,dtype=complex)
    return s


def _embed_old_coeffs(old, ext):
    c=np.zeros(ext.ndof,dtype=complex)
    old_idx=[]; new_idx=[]
    for e in range(ext.nt):
        no=old.nloc[e]
        if no:
            oi=np.arange(old.off[e],old.off[e+1])
            ei=np.arange(ext.off[e],ext.off[e]+no)
            c[ei]=old.u[oi]
            old_idx.extend(ei.tolist())
        if ext.nloc[e]>no:
            new_idx.extend(np.arange(ext.off[e]+no,ext.off[e+1]).tolist())
    return c,np.array(old_idx,dtype=int),np.array(new_idx,dtype=int)


def candidate_block_score(old, n, exact, lower_q, upper_q, region, q_candidate, nq=30, rank_tol=1e-12):
    lq=list(np.asarray(lower_q,dtype=complex).reshape(-1,2))
    uq=list(np.asarray(upper_q,dtype=complex).reshape(-1,2))
    if region=='lower': lq.append(np.asarray(q_candidate,dtype=complex))
    else: uq.append(np.asarray(q_candidate,dtype=complex))
    ext=PWDGDictionary(n,exact,build_region_dictionary(n,exact,lq,uq),nq=nq)
    A,F=ext.assemble(); Ad=A.toarray()
    c,O,C=_embed_old_coeffs(old,ext)
    r=F-Ad@c
    G=(Ad-Ad.conj().T)/(2j); G=(G+G.conj().T)/2
    if len(C)==0: return -np.inf,0.0
    GCC=G[np.ix_(C,C)]
    if len(O):
        GOO=G[np.ix_(O,O)]
        GOC=G[np.ix_(O,C)]
        Delta=GCC-GOC.conj().T@la.pinvh(GOO,rtol=rank_tol)@GOC
    else:
        Delta=GCC
    Delta=(Delta+Delta.conj().T)/2
    ev=la.eigvalsh(Delta)
    if ev.max()<=0: return -np.inf,0.0
    novelty=float(max(ev.min(),0)/ev.max())
    Dinv=la.pinvh(Delta,rtol=rank_tol)
    val=np.real(np.vdot(r[C],Dinv@r[C]))
    score=float(np.sqrt(max(val,0)))
    return score,novelty


def region_graph_greedy(n, exact, steps=5, allow_evan=False, coarse_deg=10.0, nq=30, verbose=False, min_novelty=1e-3):
    """Direction adaptation with one shared direction per material region.

    The search is graph-residual greedy. Real directions are scanned coarsely
    then refined continuously. For an evanescent upper region, optional decaying
    atoms q=(qx, i sqrt(qx^2-k^2)) are also searched over qx.
    """
    from scipy.optimize import minimize_scalar
    lower=[]; upper=[]; hist=[]
    for it in range(steps+1):
        old=solve_region_space(n,exact,lower,upper,nq=nq)
        A=old.A.toarray() if old.ndof else np.empty((0,0),complex)
        err=l2_error(old,nsub=12)
        gd=graph_data(old.A) if old.ndof else {'cond_raw':np.nan,'cond_gr':np.nan}
        hist.append(dict(iter=it,lower=np.array(lower),upper=np.array(upper),ndof=old.ndof,
                         l2=err,cond_raw=gd.get('cond_raw',np.nan),cond_gr=gd.get('cond_gr',np.nan)))
        if it==steps: break
        candidates=[]
        # real directions in both regions
        degs=np.arange(0.0,360.0,coarse_deg)
        for region,k in [('lower',exact.omega*exact.n1),('upper',exact.omega*exact.n2)]:
            vals=[]
            for deg in degs:
                q=q_real(k,np.deg2rad(deg))
                sc,nov=candidate_block_score(old,n,exact,lower,upper,region,q,nq=nq)
                vals.append(sc)
            j=int(np.argmax(vals)); d0=float(degs[j])
            # refine score in an unwrapped interval around d0
            def obj(d):
                q=q_real(k,np.deg2rad(d%360.0))
                sc,_=candidate_block_score(old,n,exact,lower,upper,region,q,nq=nq)
                return -sc
            rr=minimize_scalar(obj,bounds=(d0-coarse_deg,d0+coarse_deg),method='bounded',options={'xatol':2e-3})
            d=float(rr.x%360.0); q=q_real(k,np.deg2rad(d)); sc,nov=candidate_block_score(old,n,exact,lower,upper,region,q,nq=nq)
            candidates.append((sc,region,'real',d,q,nov))
        # decaying upper evanescent atoms parametrized by tangential qx
        if allow_evan:
            k=exact.omega*exact.n2
            # broad physically meaningful tangential range, both signs
            qmax=max(2.5*k, exact.omega*exact.n1*1.15)
            xs=np.concatenate([np.linspace(-qmax,-1.001*k,41),np.linspace(1.001*k,qmax,41)])
            vals=[]
            for qx in xs:
                q=np.array([qx,1j*np.sqrt(qx*qx-k*k)],dtype=complex)
                sc,nov=candidate_block_score(old,n,exact,lower,upper,'upper',q,nq=nq)
                vals.append(sc)
            j=int(np.argmax(vals)); x0=float(xs[j]); step=abs(xs[1]-xs[0])
            # stay on same sign branch
            if x0>0: bounds=(max(1.000001*k,x0-2*step),min(qmax,x0+2*step))
            else: bounds=(max(-qmax,x0-2*step),min(-1.000001*k,x0+2*step))
            def objx(qx):
                q=np.array([qx,1j*np.sqrt(qx*qx-k*k)],dtype=complex)
                sc,_=candidate_block_score(old,n,exact,lower,upper,'upper',q,nq=nq)
                return -sc
            rr=minimize_scalar(objx,bounds=bounds,method='bounded',options={'xatol':2e-4})
            qx=float(rr.x); q=np.array([qx,1j*np.sqrt(qx*qx-k*k)],dtype=complex)
            sc,nov=candidate_block_score(old,n,exact,lower,upper,'upper',q,nq=nq)
            candidates.append((sc,'upper','evan',qx,q,nov))
        viable=[z for z in candidates if z[5] >= min_novelty]
        if not viable:
            break
        best=max(viable,key=lambda z:z[0])
        sc,region,kind,param,q,nov=best
        if region=='lower': lower.append(q)
        else: upper.append(q)
        hist[-1].update(chosen_region=region,chosen_kind=kind,chosen_param=float(param),chosen_q=np.array(q),score=float(sc),novelty=float(nov))
        if verbose:
            print('it',it,'err',err,'choose',region,kind,param,'q',q,'score',sc,'nov',nov)
    return hist


def q_complex_angle(k, theta, eta):
    """Unified 2D Helmholtz wave vector q=k(cos z,sin z), z=theta+i eta.

    q.q=k^2 for all real theta,eta. eta=0 gives a propagating plane wave;
    eta!=0 gives an evanescent/inhomogeneous plane wave without changing family.
    """
    z = float(theta) + 1j*float(eta)
    return k*np.array([np.cos(z), np.sin(z)], dtype=complex)


def _automatic_complex_candidate(old, n, exact, lower_q, upper_q, region,
                                 nq=24, eta_max=1.8, nseed=32,
                                 min_novelty=1e-3, rng_seed=0):
    """Continuous automatic search over the complex-angle Trefftz manifold."""
    from scipy.optimize import minimize
    from scipy.stats import qmc
    k = exact.omega*(exact.n1 if region=='lower' else exact.n2)
    # Generic low-discrepancy starts; these are optimizer seeds, not retained directions.
    m = int(np.ceil(np.log2(max(2,nseed))))
    uv = qmc.Sobol(2, scramble=True, seed=rng_seed).random_base2(m)[:nseed]
    starts = np.column_stack([2*np.pi*uv[:,0], (2*uv[:,1]-1)*eta_max])
    # Always include a few eta=0 starts so propagating modes are not disadvantaged.
    th0 = 2*np.pi*(np.arange(8)+0.5)/8
    starts = np.vstack([starts, np.column_stack([th0, np.zeros_like(th0)])])

    scored=[]
    for th,et in starts:
        q=q_complex_angle(k,th,et)
        sc,nov=candidate_block_score(old,n,exact,lower_q,upper_q,region,q,nq=nq)
        if np.isfinite(sc): scored.append((sc,th,et,nov))
    if not scored: return None
    scored.sort(reverse=True,key=lambda x:x[0])

    best=None
    bounds=[(0.0,2*np.pi),(-eta_max,eta_max)]
    # Refine several best basins. Objective softly excludes graph-redundant candidates.
    for _,th0,et0,_ in scored[:5]:
        def obj(x):
            q=q_complex_angle(k,x[0],x[1])
            sc,nov=candidate_block_score(old,n,exact,lower_q,upper_q,region,q,nq=nq)
            if not np.isfinite(sc) or nov < min_novelty:
                return 1e6 + 1e3*(min_novelty-max(nov,0.0))
            return -sc
        rr=minimize(obj, np.array([th0,et0]), method='Nelder-Mead',
                    options={'maxiter':80,'xatol':2e-4,'fatol':2e-5})
        th=float(rr.x[0]%(2*np.pi)); et=float(np.clip(rr.x[1],-eta_max,eta_max))
        q=q_complex_angle(k,th,et)
        sc,nov=candidate_block_score(old,n,exact,lower_q,upper_q,region,q,nq=nq)
        if nov >= min_novelty and np.isfinite(sc):
            item=(float(sc),region,th,et,q,float(nov))
            if best is None or item[0]>best[0]: best=item
    return best


def region_graph_greedy_auto(n, exact, steps=8, nq=24, eta_max=1.8,
                             nseed=24, min_novelty=1e-3, verbose=False):
    """Automatic direction adaptation on q=k(cos(theta+i eta),sin(theta+i eta)).

    No physical directions and no separate propagating/evanescent dictionary are supplied.
    The optimizer searches theta and eta continuously and selects the region by score.
    """
    lower=[]; upper=[]; hist=[]
    for it in range(steps+1):
        old=solve_region_space(n,exact,lower,upper,nq=nq)
        err=l2_error(old,nsub=12)
        gd=graph_data(old.A) if old.ndof else {'cond_raw':np.nan,'cond_gr':np.nan}
        hist.append(dict(iter=it,lower=np.array(lower),upper=np.array(upper),ndof=old.ndof,
                         l2=err,cond_raw=gd.get('cond_raw',np.nan),cond_gr=gd.get('cond_gr',np.nan)))
        if it==steps: break
        cands=[]
        for j,region in enumerate(('lower','upper')):
            z=_automatic_complex_candidate(old,n,exact,lower,upper,region,nq=nq,
                                           eta_max=eta_max,nseed=nseed,
                                           min_novelty=min_novelty,rng_seed=7919*it+97*j+11)
            if z is not None: cands.append(z)
        if not cands: break
        sc,region,th,et,q,nov=max(cands,key=lambda x:x[0])
        if region=='lower': lower.append(q)
        else: upper.append(q)
        hist[-1].update(chosen_region=region,theta=float(th),eta=float(et),
                        chosen_q=np.array(q),score=float(sc),novelty=float(nov))
        if verbose:
            print('it',it,'err',err,'choose',region,'theta_deg',np.degrees(th),
                  'eta',et,'q',q,'score',sc,'nov',nov)
    return hist


def physical_skeleton_residual(s):
    """Weighted strong skeleton mismatch for a solved Trefftz field.

    Interior: continuity of u and normal flux. Boundary: Dirichlet trace u-g.
    This is zero for the exact transmission benchmark and is independent of
    the particular PWDG test space used to obtain s.u.
    """
    val=0.0
    for f in range(s.et.shape[0]):
        pts,wq,nrm0,L=s.edge(f)
        e0,e1=s.et[f]
        if e1>=0:
            n0=s._outward(e0,nrm0,pts)
            u0=s.uh(e0,pts); u1=s.uh(e1,pts)
            g0=s.grad_uh(e0,pts); g1=s.grad_uh(e1,pts)
            ju=u0-u1
            jf=g0@n0 + g1@(-n0)
            xi=0.5*(s.kap[e0]+s.kap[e1])
            val += np.sum(wq*(ALPHA*xi*np.abs(ju)**2 + (BETA/xi)*np.abs(jf)**2))
        else:
            e=e0
            u=s.uh(e,pts); g=s.exact(pts[:,0],pts[:,1])
            val += np.sum(wq*(ALPHA*s.kap[e]*np.abs(u-g)**2))
    return float(np.sqrt(max(val,0.0)))


def optimize_region_directions(n, exact, lower_z, upper_z, nq=20, eta_max=1.8,
                               maxiter=250, verbose=False):
    """Jointly move all active complex-angle directions by minimizing the
    physical skeleton residual after a graph-Riesz PWDG solve.

    lower_z/upper_z are arrays of (theta,eta); no exact/physical directions enter.
    """
    from scipy.optimize import minimize
    lower_z=np.asarray(lower_z,dtype=float).reshape(-1,2)
    upper_z=np.asarray(upper_z,dtype=float).reshape(-1,2)
    nlo=len(lower_z); nup=len(upper_z)
    x0=np.vstack([lower_z,upper_z]).ravel()
    def unpack(x):
        z=np.asarray(x).reshape(-1,2).copy()
        z[:,0]=np.mod(z[:,0],2*np.pi)
        z[:,1]=np.clip(z[:,1],-eta_max,eta_max)
        lq=[q_complex_angle(exact.omega*exact.n1,*z[j]) for j in range(nlo)]
        uq=[q_complex_angle(exact.omega*exact.n2,*z[nlo+j]) for j in range(nup)]
        return z,lq,uq
    def obj(x):
        z,lq,uq=unpack(x)
        try:
            s=solve_region_space(n,exact,lq,uq,nq=nq)
            r=physical_skeleton_residual(s)
            # tiny regularizer selects the least-evanescent representation among ties
            return np.log10(r*r+1e-30) + 2e-5*np.sum(z[:,1]**2)
        except Exception:
            return 10.0
    rr=minimize(obj,x0,method='Nelder-Mead',options={'maxiter':maxiter,'xatol':2e-5,'fatol':2e-6,'adaptive':True})
    z,lq,uq=unpack(rr.x)
    s=solve_region_space(n,exact,lq,uq,nq=nq)
    if verbose:
        print('opt success',rr.success,'fun',rr.fun,'skeleton',physical_skeleton_residual(s),'l2',l2_error(s,nsub=12))
        print('lower z deg/eta',[(np.degrees(a),b) for a,b in z[:nlo]])
        print('upper z deg/eta',[(np.degrees(a),b) for a,b in z[nlo:]])
    return z,s,rr


def physical_skeleton_residual_vector(s):
    """Real residual vector whose Euclidean norm is the strong graph skeleton mismatch."""
    parts=[]
    for f in range(s.et.shape[0]):
        pts,wq,nrm0,L=s.edge(f)
        e0,e1=s.et[f]
        sw=np.sqrt(wq)
        if e1>=0:
            n0=s._outward(e0,nrm0,pts)
            u0=s.uh(e0,pts); u1=s.uh(e1,pts)
            g0=s.grad_uh(e0,pts); g1=s.grad_uh(e1,pts)
            xi=0.5*(s.kap[e0]+s.kap[e1])
            ru=np.sqrt(ALPHA*xi)*sw*(u0-u1)
            rf=np.sqrt(BETA/xi)*sw*(g0@n0 + g1@(-n0))
            for r in (ru,rf):
                parts.extend([r.real,r.imag])
        else:
            e=e0
            r=np.sqrt(ALPHA*s.kap[e])*sw*(s.uh(e,pts)-s.exact(pts[:,0],pts[:,1]))
            parts.extend([r.real,r.imag])
    return np.concatenate(parts) if parts else np.zeros(0)


def optimize_region_directions_ls(n, exact, lower_z, upper_z, nq=12, eta_max=1.8,
                                  max_nfev=600, verbose=False):
    """Joint automatic complex-angle optimization using nonlinear least squares."""
    from scipy.optimize import least_squares
    lower_z=np.asarray(lower_z,dtype=float).reshape(-1,2)
    upper_z=np.asarray(upper_z,dtype=float).reshape(-1,2)
    nlo=len(lower_z); nup=len(upper_z)
    x0=np.vstack([lower_z,upper_z]).ravel()
    # theta is left locally unconstrained around start; eta is physically/numerically bounded.
    lb=[]; ub=[]
    for j in range(nlo+nup):
        lb += [x0[2*j]-2*np.pi, -eta_max]
        ub += [x0[2*j]+2*np.pi, eta_max]
    def unpack(x):
        z=np.asarray(x,dtype=float).reshape(-1,2).copy()
        z[:,0]=np.mod(z[:,0],2*np.pi)
        lq=[q_complex_angle(exact.omega*exact.n1,*z[j]) for j in range(nlo)]
        uq=[q_complex_angle(exact.omega*exact.n2,*z[nlo+j]) for j in range(nup)]
        return z,lq,uq
    def fun(x):
        try:
            z,lq,uq=unpack(x)
            s=solve_region_space(n,exact,lq,uq,nq=nq)
            r=physical_skeleton_residual_vector(s)
            # weak eta regularization appended as residuals, not a separate branch selection
            return np.concatenate([r, 1e-6*z[:,1]])
        except Exception:
            # fixed length obtained from initial valid solve
            z,lq,uq=unpack(x0)
            s0=solve_region_space(n,exact,lq,uq,nq=nq)
            return np.full(len(physical_skeleton_residual_vector(s0))+len(z),1e3)
    rr=least_squares(fun,x0,bounds=(np.array(lb),np.array(ub)),method='trf',
                     x_scale='jac',ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=max_nfev,
                     diff_step=2e-5)
    z,lq,uq=unpack(rr.x)
    s=solve_region_space(n,exact,lq,uq,nq=max(nq,16))
    if verbose:
        print('LS success',rr.success,'cost',rr.cost,'nfev',rr.nfev,
              'skeleton',physical_skeleton_residual(s),'l2',l2_error(s,nsub=12))
        print('lower z deg/eta',[(float(np.degrees(a)),float(b)) for a,b in z[:nlo]])
        print('upper z deg/eta',[(float(np.degrees(a)),float(b)) for a,b in z[nlo:]])
    return z,s,rr
