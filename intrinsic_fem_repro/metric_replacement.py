"""Tables on metric replacement (three-torus): first positive eigenvalue with the prescribed metric
g = exp(2 phi) B and with phi frozen at each cell barycenter, on the same Bernstein space; the sampled
discrepancy; and the ratio interval exp(t_- - 3 t_+), exp(t_+ - 3 t_-) from interval bounds of phi on
each coordinate cube.   usage: python metric_replacement.py [--anisotropic]"""
import sys,itertools,numpy as np,scipy.sparse.linalg as sla
from riemannfem.periodic_scalar import periodic_freudenthal_mesh,periodic_bernstein_numbering,reference_evaluations
from riemannfem.torus_metric import AffineTetrahedron,conformal_factor
from riemannfem.assembly import boolean_constraint,assemble_constrained
from oneform_high_order import B_ANISO
def pencil(N,p,q,Bm,frozen):
    cells=periodic_freudenthal_mesh(N);ld,lab=periodic_bernstein_numbering(cells,N,p);xh,w,Bv,dB=reference_evaluations(p,q)
    Binv=np.linalg.inv(Bm);sf=np.sqrt(np.linalg.det(Bm));Ms=[];Ks=[];eps=0.
    for V in cells:
        c=AffineTetrahedron(V/N);J=abs(np.linalg.det(c.E));Ei=np.linalg.inv(c.E);phi,_=conformal_factor(c.physical_coordinates(xh))
        if frozen:
            pc,_=conformal_factor(c.physical_coordinates(np.full((1,3),.25)));eps=max(eps,np.abs(np.exp(2*(pc[0]-phi))-1).max());phi=np.full_like(phi,pc[0])
        gp=np.einsum('qai,ij->qaj',dB,Ei);gB=np.einsum('qai,ij->qaj',gp,Binv)
        Ms.append(Bv.T@((w*J*sf*np.exp(3*phi))[:,None]*Bv));Ks.append(sum(gB[:,:,j].T@((w*J*sf*np.exp(phi))[:,None]*gp[:,:,j]) for j in range(3)))
    P=boolean_constraint(ld,len(lab));return assemble_constrained(Ks,P).tocsc(),assemble_constrained(Ms,P).tocsc(),eps,len(lab)
def mu1(K,M):
    v=np.sort(sla.eigsh(K,M=M,k=3,sigma=-1.,which='LM',return_eigenvectors=False));return v[1]
def srange(a,b):   # range of sin(2 pi x) on [a,b]
    c=[np.sin(2*np.pi*a),np.sin(2*np.pi*b)]+[s for k in range(-2,6) for x0,s in ((0.25+k,1.),(0.75+k,-1.)) if a<=x0<=b];return min(c),max(c)
def crange(a,b):   # range of cos(2 pi s) on [a,b]
    c=[np.cos(2*np.pi*a),np.cos(2*np.pi*b)]+[s for k in range(-2,6) for x0,s in ((k,1.),(0.5+k,-1.)) if a<=x0<=b];return min(c),max(c)
def tbounds(N):
    tm,tp=0.,0.;cells=periodic_freudenthal_mesh(N)
    for V in cells:
        lo=V.min(0)/N;hi=lo+1/N;r=[srange(lo[i],hi[i]) for i in range(3)];pr=[a*b*c for a in r[0] for b in r[1] for c in r[2]]
        cr=crange(lo[0]+lo[1],hi[0]+hi[1]);plo=0.12*min(pr)+0.08*cr[0];phi_=0.12*max(pr)+0.08*cr[1]
        pc,_=conformal_factor((V/N).mean(0)[None,:]);tm=min(tm,pc[0]-phi_);tp=max(tp,pc[0]-plo)
    return tm,tp
if __name__=='__main__':
    an='--anisotropic' in sys.argv;Bm=B_ANISO if an else np.eye(3)
    runs=[(2,2),(3,2),(4,2)] if an else [(2,2),(3,2),(4,2),(6,2),(8,2),(3,3),(4,3)]
    print('N p dofs eps_sampled mu1(exact) mu1(frozen) |diff| ratio  t- t+  [interval]')
    for N,p in runs:
        q=max(p+4,6);K,M,_,nd=pencil(N,p,q,Bm,False);Kf,Mf,eps,_=pencil(N,p,q,Bm,True);a,b=mu1(K,M),mu1(Kf,Mf);tm,tp=tbounds(N)
        print(f'{N} {p} {nd} {eps:.3f} {a:.6f} {b:.6f} {abs(a-b):.6f} {b/a:.6f}  {tm:.4f} {tp:.4f}  [{np.exp(tm-3*tp):.3f}, {np.exp(tp-3*tm):.3f}]',flush=True)
