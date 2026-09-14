from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import ELEGANT_CMAP, LINE_BLUE, LINE_RED, LINE_GOLD, MESH_GRAY, LIGHT_GRID

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures'; DATA=ROOT/'data'; FIG.mkdir(exist_ok=True)

# 1. Smooth p convergence: six odd-degree points.
df=pd.read_csv(DATA/'smooth_p_odd_embedded_qt.csv')
fig,ax=plt.subplots(figsize=(5.8,4.4))
ax.semilogy(df.p,df.relL2,'o-',color=LINE_BLUE,label=r'relative $L^2$')
ax.semilogy(df.p,df.relH1,'s--',color=LINE_RED,label=r'relative $H^1$')
ax.set_xlabel(r'degree $p$');ax.set_ylabel('relative error');ax.set_xticks(df.p)
ax.grid(True,which='both',alpha=LIGHT_GRID);ax.legend();fig.tight_layout();fig.savefig(FIG/'smooth_p_convergence.pdf');plt.close(fig)

# 2. Smooth h convergence: six mesh levels, p=5.
dh=pd.read_csv(DATA/'smooth_h_p5_embedded_qt.csv')
fig,ax=plt.subplots(figsize=(5.8,4.4))
ax.loglog(dh.ntri,dh.relL2,'o-',color=LINE_BLUE,label=r'relative $L^2$')
ax.loglog(dh.ntri,dh.relH1,'s--',color=LINE_RED,label=r'relative $H^1$')
ax.set_xlabel('number of triangles');ax.set_ylabel('relative error')
ax.grid(True,which='both',alpha=LIGHT_GRID);ax.legend();fig.tight_layout();fig.savefig(FIG/'smooth_h_convergence.pdf');plt.close(fig)

# 3. Focal p convergence: six odd-degree points.
dfoc=pd.read_csv(DATA/'focal_caustic_p_sweep_embedded_qt.csv')
fig,ax=plt.subplots(figsize=(5.8,4.4))
ax.semilogy(dfoc.p,dfoc.relL2,'o-',color=LINE_BLUE,label=r'relative $L^2$')
ax.semilogy(dfoc.p,dfoc.relH1,'s--',color=LINE_RED,label=r'relative $H^1$')
ax.set_xlabel(r'degree $p$');ax.set_ylabel('relative error');ax.set_xticks(dfoc.p)
ax.grid(True,which='both',alpha=LIGHT_GRID);ax.legend();fig.tight_layout();fig.savefig(FIG/'focal_p_convergence.pdf');plt.close(fig)

# 4. Carrier-frequency graph: nine points.
dom=pd.read_csv(DATA/'carrier_frequency_sweep.csv')
fig,ax=plt.subplots(figsize=(5.9,4.4))
ax.loglog(dom.omega,dom.relative_field_L2,'o-',color=LINE_BLUE)
ax.set_xlabel(r'carrier frequency $\omega$');ax.set_ylabel(r'relative $L^2$ field error')
ax.grid(True,which='both',alpha=LIGHT_GRID);fig.tight_layout();fig.savefig(FIG/'carrier_frequency.pdf');plt.close(fig)

# 5. Stratified p=5 field, n=4 mesh.
z=np.load(DATA/'stratified_n4_p5_solution.npz');c=z['c'];verts=z['verts'];tris=z['tris'];l2g=z['l2g'];p=int(z['p'])
from eikonal_c0_pminus1 import bernstein_values
from heterogeneous_stratified import StratifiedEikonal
prob=StratifiedEikonal(alpha=.8)
N=301;xs=np.linspace(0,1,N);ys=np.linspace(0,1,N);Z=np.full((N,N),np.nan)
for K,tri in enumerate(tris):
    V=verts[tri];A=np.column_stack((V[1]-V[0],V[2]-V[0]));inv=np.linalg.inv(A)
    xmin,xmax=V[:,0].min(),V[:,0].max();ymin,ymax=V[:,1].min(),V[:,1].max()
    ii=np.where((xs>=xmin-1e-12)&(xs<=xmax+1e-12))[0];jj=np.where((ys>=ymin-1e-12)&(ys<=ymax+1e-12))[0]
    for j in jj:
        P=np.column_stack([xs[ii],np.full(len(ii),ys[j])]);rs=(P-V[0])@inv.T;l1,l2=rs[:,0],rs[:,1];l0=1-l1-l2;inside=(l0>=-1e-10)&(l1>=-1e-10)&(l2>=-1e-10)
        if inside.any():
            lam=np.column_stack([l0[inside],l1[inside],l2[inside]]);B=bernstein_values(p,lam);Z[j,ii[inside]]=B@c[l2g[K]]
fig,ax=plt.subplots(figsize=(7.1,5.5));cf=ax.contourf(xs,ys,Z,levels=36,cmap=ELEGANT_CMAP);ax.contour(xs,ys,Z,levels=14,linewidths=.55,alpha=.55,colors='white')
for tri in tris:
    P=verts[np.r_[tri,tri[0]]];ax.plot(P[:,0],P[:,1],linewidth=.3,color=MESH_GRAY,alpha=.24)
# medium layer reference lines
for yy in [0.22,0.58]:ax.axhline(yy,linestyle='--',linewidth=.7,color=MESH_GRAY,alpha=.5)
ax.set_aspect('equal');ax.set_xlabel('$x$');ax.set_ylabel('$y$');ax.set_title('Stratified eikonal phase, $p=5$')
fig.colorbar(cf,ax=ax,label='$u_h$');fig.tight_layout();fig.savefig(FIG/'stratified_p5_solution.png',dpi=220);plt.close(fig)

# 6. Two-source p=5 solution.
g=np.load(DATA/'linear_speed_two_source_p5_grid.npz');X=g['X'];Y=g['Y'];U=g['U']
fig,ax=plt.subplots(figsize=(7.1,5.5));cf=ax.contourf(X,Y,U,levels=36,cmap=ELEGANT_CMAP);ax.contour(X,Y,U,levels=14,linewidths=.6,colors='white',alpha=.6)
ax.scatter([0,.8],[0,0],marker='*',s=90);ax.set_aspect('equal');ax.set_xlabel('$x$');ax.set_ylabel('$y$');ax.set_title('Two-source first-arrival travel time, $p=5$')
fig.colorbar(cf,ax=ax,label='$u_h$');fig.tight_layout();fig.savefig(FIG/'linear_speed_two_source_p5.png',dpi=220);plt.close(fig)
