import numpy as np, matplotlib.pyplot as plt
from optimized_hj import FastFactoredEikonalPminus1
from conforming_embedded_qt import ConformingEmbeddedQT
from qt_projection_utils import project_unknown
from high_frequency_benchmarks import RadialFactor
from eikonal_c0_pminus1 import bernstein_values
from pathlib import Path
from plot_style import ELEGANT_CMAP, MESH_GRAY
ROOT=Path(__file__).resolve().parents[1]
class FocalCaustic:
    def __init__(self, center=(0.,0.), amp=.035): self.center=np.asarray(center,float);self.amp=amp
    def u(self,x):
        z=x-self.center;r=np.linalg.norm(z,axis=1);X=x[:,0];Y=x[:,1];return -r+self.amp*np.sin(np.pi*X)*np.sin(np.pi*Y)
    def grad(self,x):
        z=x-self.center;r=np.linalg.norm(z,axis=1);g=np.zeros_like(z);m=r>1e-13;g[m]=-z[m]/r[m,None];X=x[:,0];Y=x[:,1];a=self.amp
        g[:,0]+=a*np.pi*np.cos(np.pi*X)*np.sin(np.pi*Y);g[:,1]+=a*np.pi*np.sin(np.pi*X)*np.cos(np.pi*Y);return g
    def n2(self,x):g=self.grad(x);return np.sum(g*g,axis=1)
prob=FocalCaustic();fac=RadialFactor((0,0),-1.)
m=FastFactoredEikonalPminus1(4,5,prob,fac,bounds=((-1,1),(-1,1)));seed,_=project_unknown(m,lambda x: prob.u(x)-fac.u(x));qt=ConformingEmbeddedQT(m,seed);c,sol=qt.solve_trace(initial_full=seed,max_iter=40)
N=301;xs=np.linspace(-1,1,N);ys=np.linspace(-1,1,N);Z=np.full((N,N),np.nan)
for K,tri in enumerate(m.tris):
 V=m.verts[tri];A=np.column_stack((V[1]-V[0],V[2]-V[0]));inv=np.linalg.inv(A);xmin,xmax=V[:,0].min(),V[:,0].max();ymin,ymax=V[:,1].min(),V[:,1].max();ii=np.where((xs>=xmin-1e-12)&(xs<=xmax+1e-12))[0];jj=np.where((ys>=ymin-1e-12)&(ys<=ymax+1e-12))[0]
 for j in jj:
  P=np.column_stack([xs[ii],np.full(len(ii),ys[j])]);rs=(P-V[0])@inv.T;l1,l2=rs[:,0],rs[:,1];l0=1-l1-l2;inside=(l0>=-1e-10)&(l1>=-1e-10)&(l2>=-1e-10)
  if inside.any():
   lam=np.column_stack([l0[inside],l1[inside],l2[inside]]);B=bernstein_values(m.p,lam);tau=B@c[m.asm.l2g[K]];Z[j,ii[inside]]=fac.u(P[inside])+tau
fig,ax=plt.subplots(figsize=(6.2,5.3));cf=ax.contourf(xs,ys,Z,levels=35,cmap=ELEGANT_CMAP);ax.contour(xs,ys,Z,levels=24,linewidths=.45,colors='white',alpha=.5)
for tri in m.tris:
 P=m.verts[np.r_[tri,tri[0]]];ax.plot(P[:,0],P[:,1],linewidth=.3,color=MESH_GRAY,alpha=.24)
ax.plot([0],[0],marker='x',markersize=7);ax.set_aspect('equal');ax.set_xlabel('$x$');ax.set_ylabel('$y$');ax.set_title('Computed factored focal phase, $p=5$');fig.colorbar(cf,ax=ax,label='$u_h$');fig.tight_layout();fig.savefig(ROOT/'figures'/'focal_caustic_p5.pdf');plt.close(fig)
