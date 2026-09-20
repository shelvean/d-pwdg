import numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt,matplotlib.gridspec as gs
from matplotlib.collections import PolyCollection
from tubeviz import *
from cmcrameri import cm as ccm
plt.rcParams.update({'font.family':'cmr10','mathtext.fontset':'cm','axes.unicode_minus':False,'axes.formatter.use_mathtext':True});CMAP=ccm.vik
sel=[('neumann',1),('neumann',49),('neumann',52),('neumann',63),('neumann',72),('dirichlet',0),('dirichlet',3),('dirichlet',13)]
data={}
for bc in ('neumann','dirichlet'):
    f=f'run/tube_trefoil_96_3_3_{bc[0]}.npz';d=np.load(f);data[bc]=(Tube(f,96,3,3),d[bc+'_vals'],d[bc+'_vecs'])
tb=data['neumann'][0]
ns_,nr_=1600,29;t=np.linspace(0,TAU,ns_+1);rho=np.linspace(-tb.rin*.995,tb.rin*.995,nr_)
c0,tt,n,b=frame(t);w=np.cross(tt,[0,0,1.]);w/=np.linalg.norm(w,axis=1,keepdims=True)
S=np.repeat(t[:,None],nr_,1);U=rho[None,:]*(w*n).sum(1)[:,None];Vv=rho[None,:]*(w*b).sum(1)[:,None]
X=c0[:,None,:]+rho[None,:,None]*w[:,None,:]
quad=lambda Z:0.25*(Z[:-1,:-1]+Z[1:,:-1]+Z[1:,1:]+Z[:-1,1:])
P=np.stack([X[:-1,:-1,:2],X[1:,:-1,:2],X[1:,1:,:2],X[:-1,1:,:2]],2).reshape(-1,4,2)
zc=quad(X[...,2]).ravel();order=np.argsort(zc);depth=(0.78+0.22*(zc-zc.min())/(zc.max()-zc.min()))[order]
edge=(np.abs(quad(np.repeat(rho[None,:],ns_+1,0)))>0.9*tb.rin).ravel()[order]
nd=8;g=np.linspace(-a,a,61);UU,VV=np.meshgrid(g,g,indexing='ij')
Sd=np.concatenate([np.full(UU.shape,i*TAU/nd) for i in range(nd)],0);Ud=np.tile(UU,(nd,1));Vd=np.tile(VV,(nd,1))
fig=plt.figure(figsize=(14,10.2));outer=gs.GridSpec(2,4,figure=fig,left=.01,right=.875,top=.92,bottom=.02,wspace=.04,hspace=.22)
for cell,(bc,k) in zip(outer,sel):
    tbx,vals,vecs=data[bc];inner=cell.subgridspec(2,1,height_ratios=[4.2,1.15],hspace=.03)
    F=tbx.ev(vecs[:,k],S,U,Vv);Fd=tbx.ev(vecs[:,k],Sd,Ud,Vd);m=max(np.nanmax(abs(F)),np.nanmax(abs(Fd)))
    ax=fig.add_subplot(inner[0]);col=CMAP(0.5+0.5*quad(F/m).ravel()[order]);col[:,:3]*=depth[:,None];col[edge,:3]=0.25
    ax.add_collection(PolyCollection(P[order],facecolors=col,edgecolors='none',antialiaseds=False,rasterized=True))
    ax.set_xlim(-3.5,3.5);ax.set_ylim(-3.7,3.1);ax.set_aspect('equal');ax.axis('off')
    ax.set_title(f'{bc.capitalize()} mode {k}\neigenvalue {vals[k]:.4g}',fontsize=19,pad=6,linespacing=1.15)
    ax2=fig.add_subplot(inner[1]);ax2.imshow(Fd.T/m,cmap=CMAP,vmin=-1,vmax=1,origin='lower',aspect='equal',interpolation='bilinear');ax2.axis('off')
cax=fig.add_axes([.888,.2,.018,.6]);cb=fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP,norm=plt.Normalize(-1,1)),cax=cax)
cb.set_label('scaled eigenfunction value',fontsize=21);cb.ax.tick_params(labelsize=19)
fig.savefig('figures/solid_trefoil_tube_modes.png',dpi=240);plt.close(fig)
