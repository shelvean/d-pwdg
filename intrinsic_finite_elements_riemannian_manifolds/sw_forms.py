"""Coexact 1-form spectrum of the Seifert-Weber dodecahedral space with the trimmed polynomial
complex P_r^- Lambda^k of riemannfem.  Mesh, pairings and refinement come from hypforms.py; the
element metric is the pullback of the hyperboloid metric through the chart y = V b, x = y/r."""
import sys,time,argparse,numpy as np,scipy.sparse.linalg as sla
sys.path.insert(0,'.')
sys.path.insert(0,'.')
import hypforms as H
from riemannfem.metric import CallableMetric
from riemannfem.global_complex import FacetPairing
from riemannfem.entity_assembly import SparseEntityComplex
from riemannfem.forms import local_hodge_mass
from riemannfem.integration import mesh_volume
Q=np.diag([-1.,1,1,1])
class HyperboloidRadialMetric(CallableMetric):
    def __init__(self,V):
        V=np.asarray(V,float);Dy=V[:,1:]-V[:,[0]]
        def fn(xhat):
            b=np.r_[1-np.sum(xhat),xhat];y=V@b;Qy=Q@y;r2=-y@Qy
            G=Dy.T@(Q/r2+np.outer(Qy,Qy)/r2**2)@Dy;return 0.5*(G+G.T)
        super().__init__(3,fn)
def pair_facets(cells,transforms,dec=8,tol=1e-7):
    halves=[];table={}
    key=lambda P:tuple(sorted(map(tuple,np.round(P,dec).tolist())))
    for ci,V in enumerate(cells):
        for f in range(4):
            loc=[j for j in range(4) if j!=f];face=V.T[loc];halves.append((ci,f,face));table.setdefault(key(face+0.0),[]).append(len(halves)-1)
    partner={}
    for h,(ci,f,face) in enumerate(halves):
        if h in partner:continue
        for L in transforms:
            for g in table.get(key(face@L.T+0.0),[]):
                if g!=h and g not in partner:partner[h]=(g,L);partner[g]=(h,None);break
            if h in partner:break
        if h not in partner:raise RuntimeError(f'half-facet {h} has no partner')
    out=[]
    for h,(g,L) in partner.items():
        if L is None:continue
        ca,fa,A=halves[h];cb,fb,B=halves[g];img=A@L.T          # L carries A onto B
        p=[int(np.argmin(np.linalg.norm(B-a,axis=1))) for a in img];assert sorted(p)==[0,1,2]
        out.append(FacetPairing(ca,fa,cb,fb,tuple(p)))
    return out
def seifert_weber(level):
    a,rho,U,Fd,hv,hf,faces=H.geometry();O=np.array([1.,0,0,0]);cells=[]
    for k,idx in enumerate(faces):
        for i in range(5):cells.append(np.stack([O,hf[k],hv[idx[i]],hv[idx[(i+1)%5]]],1))
    for _ in range(level):cells=H.refine(cells)
    T=[np.eye(4)]+[H.pairing(f,a) for f in Fd]
    return cells,pair_facets(cells,T),[HyperboloidRadialMetric(V) for V in cells]
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--r',type=int,default=2);ap.add_argument('--level',type=int,default=0)
    ap.add_argument('--quad',type=int,default=0);ap.add_argument('--sigma',type=float,default=1.0);ap.add_argument('--k',type=int,default=30)
    ap.add_argument('--save',action='store_true');ap.add_argument('--dense',action='store_true');a_=ap.parse_args();t=time.perf_counter()
    cells,pairings,metrics=seifert_weber(a_.level);q=a_.quad or a_.r+4
    C=SparseEntityComplex(len(cells),a_.r,pairings)
    print(f'cells={len(cells)} pairings={len(pairings)} r={a_.r} dims={C.dimensions} [{time.perf_counter()-t:.0f}s]',flush=True)
    M1=C.assemble_mass(1,[local_hodge_mass(C.spaces[1],g,q=q) for g in metrics])
    M2=C.assemble_mass(2,[local_hodge_mass(C.spaces[2],g,q=q) for g in metrics])
    print(f'mass matrices assembled [{time.perf_counter()-t:.0f}s]',flush=True)
    A=(C.D[1].T@M2@C.D[1]).tocsc()
    if a_.dense:
        import scipy.linalg as la
        lam,vec=la.eigh(A.toarray(),M1.toarray());np.savez_compressed(f'run/swforms_dense_r{a_.r}_L{a_.level}.npz',lam=lam);sla=None
    else:
      lam,vec=sla.eigsh(A,M=M1.tocsc(),k=min(a_.k,C.dimensions[1]-2),sigma=a_.sigma,which='LM',tol=1e-10)
    o=np.argsort(lam);lam=lam[o];vec=vec[:,o];pos=lam>1e-6
    print('coexact eigenvalues:',np.array2string(lam[pos][:24],precision=6,max_line_width=150))
    print(f'seconds {time.perf_counter()-t:.0f}')
    if a_.save:np.savez_compressed(f'run/swforms_r{a_.r}_L{a_.level}.npz',lam=lam,vec=vec)
