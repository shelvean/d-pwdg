import numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt,matplotlib.gridspec as gs
from matplotlib.collections import PolyCollection
from cmcrameri import cm as ccm
from pdsviz import *
plt.rcParams.update({'font.family':'cmr10','mathtext.fontset':'cm','axes.unicode_minus':False,'axes.formatter.use_mathtext':True});CMAP=ccm.vik
def inv_U(C,elems,npts=1200):
    rng=np.random.default_rng(0);x=rng.normal(size=(npts,4));x/=np.linalg.norm(x,axis=1,keepdims=True)
    Phi=ev(C,x);A=np.zeros((C.shape[1],)*2)
    for g in elems:A+=np.linalg.lstsq(Phi,ev(C,conj(g,x)),rcond=None)[0]
    A/=len(elems);A=(A+A.T)/2;w,U=np.linalg.eigh(A);return U[:,w>0.5]
half=[V[i] for i in range(120) if tuple(np.round(-V[i],9))>tuple(np.round(V[i],9))]
nb=np.where(adj[one])[0];n5=V[nb[0]];ax5=n5[1:]/np.linalg.norm(n5[1:])
def powers(g,n):
    out=[np.array([1.,0,0,0])]
    for _ in range(n-1):out.append(qmul(out[-1][None,:],g[None,:])[0])
    return out
g3=next(V[i] for i in range(120) if abs(V[i][0]-.5)<1e-9 and abs(V[i]@n5-np.cos(np.pi/5)*.5)<.5 and V[i][1:]@ax5>0.5)
ax3=g3[1:]/np.linalg.norm(g3[1:]);C5=powers(n5,5);C3=powers(g3,3)
def nonA(lev,elems,pick=0):
    C,v=level(lev);uA=inv_U(C,half);U=inv_U(C,elems);U=U-uA@(uA.T@U);u,s,_=np.linalg.svd(U,full_matrices=False)
    return C@u[:,pick],v.mean(),int((s>0.5).sum())
def Amode(lev):
    C,v=level(lev);return (C@inv_U(C,half))[:,0],v.mean()
# dodecahedron surface (Voronoi cell of the vertex 1), fine triangulation
def face_tris(ni,n=36):
    cen=[V[list(t)].sum(0) for t in tets if one in t and ni in t];cen=[c/np.linalg.norm(c) for c in cen];assert len(cen)==5
    m=V[one]+V[ni];m/=np.linalg.norm(m);axis=V[ni][1:]-V[one][1:]
    e1=np.cross(axis,[.3,.5,.8]);e1/=np.linalg.norm(e1);e2=np.cross(axis,e1);e2/=np.linalg.norm(e2)
    ang=[np.arctan2((c[1:]-m[1:])@e2,(c[1:]-m[1:])@e1) for c in cen];cen=[cen[i] for i in np.argsort(ang)]
    T=[];L=[]
    for a in range(5):
        A,B_=cen[a],cen[(a+1)%5]
        pt=lambda i,j:(m*(n-i-j)+A*i+B_*j)/n
        for i in range(n):
            for j in range(n-i):
                T.append([pt(i,j),pt(i+1,j),pt(i,j+1)]);L.append((n-i-j-1/3)/n)
                if i+j<n-1:T.append([pt(i+1,j),pt(i+1,j+1),pt(i,j+1)]);L.append((n-i-j-5/3)/n)
    T=np.array(T);return T/np.linalg.norm(T,axis=2,keepdims=True),np.array(L)
TR=[];LM_=[]
for ni in nb:
    t,l=face_tris(ni);TR.append(t);LM_.append(l)
TR=np.concatenate(TR);LM_=np.concatenate(LM_);CT=TR.mean(1);CT/=np.linalg.norm(CT,axis=1,keepdims=True)
st=lambda x:x[...,1:]/(1+x[...,[0]]);XT=st(TR);XC=XT.mean(1)
NRM=np.cross(XT[:,1]-XT[:,0],XT[:,2]-XT[:,0]);NRM/=np.linalg.norm(NRM,axis=1,keepdims=True);NRM*=np.sign((NRM*XC).sum(1))[:,None]
def basis(d):
    d=d/np.linalg.norm(d);e1=np.cross([0.1,0.2,1.],d);e1/=np.linalg.norm(e1);return d,e1,np.cross(d,e1)
def tilt(axis,deg=22):
    d,e1,e2=basis(axis);t=np.radians(deg);return np.cos(t)*d+np.sin(t)*(0.6*e1+0.8*e2)
panels=[('A',168,'fully symmetric'),('A',440,'fully symmetric'),('A',624,'fully symmetric'),('A',960,'fully symmetric'),
        ('5',168,'five-fold axis'),('3',168,'three-fold axis'),('5',440,'five-fold axis'),('3',624,'three-fold axis')]
fig=plt.figure(figsize=(14,12.6));outer=gs.GridSpec(2,4,figure=fig,left=.01,right=.875,top=.925,bottom=.01,wspace=.03,hspace=.16)
G=np.linspace(-.2,.2,221)
for cell,(kind,lev,txt) in zip(outer,panels):
    if kind=='A':c,val=Amode(lev);d=tilt(ax5)
    elif kind=='5':c,val,_=nonA(lev,C5);d=ax5
    else:c,val,_=nonA(lev,C3);d=ax3
    d,e1,e2=basis(d);F=ev(c[:,None],CT)[:,0]
    a,b=np.meshgrid(G,G,indexing='ij');X=a[...,None]*e1+b[...,None]*e2;r2=(X*X).sum(-1)
    x=np.concatenate([((1-r2)/(1+r2))[...,None],2*X/(1+r2)[...,None]],-1).reshape(-1,4)
    inside=(x@V[one][:,None]>=(x@V[nb].T).max(1,keepdims=True)-1e-12)[:,0];Fs=np.full(len(x),np.nan);Fs[inside]=ev(c[:,None],x[inside])[:,0]
    m=max(np.abs(F).max(),np.nanmax(np.abs(Fs)));sgn=np.sign(Fs[len(Fs)//2]) or 1.;F=F/m*sgn;Fs=Fs/m*sgn
    inner=cell.subgridspec(2,1,height_ratios=[1.35,1],hspace=.0);ax=fig.add_subplot(inner[0])
    vis=(NRM@d)>0;o=np.argsort((XC@d)[vis]);P=np.stack([XT[vis]@e1,XT[vis]@e2],-1)[o]
    L=np.array([-.35,.45,.82]);L/=np.linalg.norm(L);nv=np.stack([NRM[vis]@e1,NRM[vis]@e2,NRM[vis]@d],-1)[o]
    col=CMAP(.5+.5*F[vis][o]);col[:,:3]*=(0.55+0.45*np.clip(nv@L,0,1))[:,None];col[LM_[vis][o]<0.035,:3]=0.2
    ax.add_collection(PolyCollection(P,facecolors=col,edgecolors='none',antialiaseds=False,rasterized=True))
    ax.set_xlim(-.21,.21);ax.set_ylim(-.21,.21);ax.set_aspect('equal');ax.axis('off')
    ax.set_title(f'eigenvalue {val:.4f}\n{txt}',fontsize=18,pad=2,linespacing=1.15)
    ax2=fig.add_subplot(inner[1]);ax2.imshow(Fs.reshape(a.shape).T,origin='lower',cmap=CMAP,vmin=-1,vmax=1,extent=[-.2,.2,-.2,.2],interpolation='bilinear')
    ax2.contour(G,G,np.isfinite(Fs).reshape(a.shape).T.astype(float),levels=[.5],colors='0.2',linewidths=1.2);ax2.axis('off')
cax=fig.add_axes([.888,.2,.018,.6]);cb=fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP,norm=plt.Normalize(-1,1)),cax=cax)
cb.set_label('scaled eigenfunction value',fontsize=21);cb.ax.tick_params(labelsize=19)
fig.savefig('figures/poincare_dodecahedral_space_modes.png',dpi=220);plt.close(fig)
