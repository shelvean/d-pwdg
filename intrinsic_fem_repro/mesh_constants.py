"""Chart regularity numbers for the meshes of the paper: generalized eigenvalues of G_K(x) against
G_K(barycenter) over sample points (variation of the chart inside a cell), and the anisotropy
lambda_max/lambda_min of G_K at the barycenter (shape of the cell in its own metric)."""
import itertools,numpy as np,scipy.linalg as la
g=np.linspace(0.02,0.96,7);X=np.array([x for x in itertools.product(g,repeat=3) if sum(x)<=0.98]);bc=np.array([[.25,.25,.25]])
def report(name,Gfun,nc):
    lo,hi,an=np.inf,0,0
    for c in range(nc):
        G0=Gfun(c,bc)[0];w0=np.linalg.eigvalsh(G0);an=max(an,w0[-1]/w0[0])
        for G in Gfun(c,X):
            w=la.eigh(G,G0,eigvals_only=True);lo=min(lo,w[0]);hi=max(hi,w[-1])
    print(f'{name:34s} cells {nc:5d}  variation in a cell [{lo:.3f}, {hi:.3f}]  anisotropy at barycenter {an:.2f}')
import pds;report('S^3/Gamma, five cells',lambda c,x:pds.metric_fn([c],x)[0],5)
import sw
for L in (0,1):
    cells,_,_=sw.cells_and_gens(L);E=np.stack([cells[:,:,j]-cells[:,:,0] for j in (1,2,3)],-1)
    def Gf(c,x,cells=cells,E=E):
        y=cells[c,:,0][None,:]+x@E[c].T;Qy=y*np.diag(sw.Qm);r2=-(y*Qy).sum(-1)
        P=sw.Qm[None]/r2[:,None,None]+Qy[:,:,None]*Qy[:,None,:]/(r2**2)[:,None,None];return np.einsum('ki,qkl,lj->qij',E[c],P,E[c])
    report(f'Seifert-Weber, {len(cells)} cells',Gf,len(cells))
