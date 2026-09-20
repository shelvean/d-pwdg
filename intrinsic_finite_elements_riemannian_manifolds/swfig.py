import numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt,matplotlib.gridspec as gs
from matplotlib.collections import PolyCollection
from cmcrameri import cm as ccm
from swviz import *
plt.rcParams.update({'font.family':'cmr10','mathtext.fontset':'cm','axes.unicode_minus':False,'axes.formatter.use_mathtext':True});CMAP=ccm.vik
def face_tris(k,n=34):
    T=[];L=[];idx=faces[k];m=hf[k]
    for i in range(5):
        A,B_=hv[idx[i]],hv[idx[(i+1)%5]];pt=lambda i_,j_:(m*(n-i_-j_)+A*i_+B_*j_)/n
        for i_ in range(n):
            for j_ in range(n-i_):
                T.append([pt(i_,j_),pt(i_+1,j_),pt(i_,j_+1)]);L.append((n-i_-j_-1/3)/n)
                if i_+j_<n-1:T.append([pt(i_+1,j_),pt(i_+1,j_+1),pt(i_,j_+1)]);L.append((n-i_-j_-5/3)/n)
    return np.array(T),np.array(L)
TR=[];LM_=[]
for k in range(12):
    t,l=face_tris(k);TR.append(t);LM_.append(l)
TR=np.concatenate(TR);LM_=np.concatenate(LM_);CT=TR.mean(1)
nrm=lambda y:y/np.sqrt(-(y*y*np.diag(Qm)).sum(-1))[...,None];ball=lambda x:x[...,1:]/(1+x[...,[0]])
XT=ball(nrm(TR));XC=XT.mean(1);NRM=np.cross(XT[:,1]-XT[:,0],XT[:,2]-XT[:,0]);NRM/=np.linalg.norm(NRM,axis=1,keepdims=True);NRM*=np.sign((NRM*XC).sum(1))[:,None]
NF=np.c_[np.full(12,np.sinh(a_in)),np.cosh(a_in)*Fd]                         # face planes: <x,n_f> <= 0 inside
def basis(d):
    d=d/np.linalg.norm(d);e1=np.cross([0.1,0.2,1.],d);e1/=np.linalg.norm(e1);return d,e1,np.cross(d,e1)
panels=[(9.5701,'5',0),(9.5701,'3',0),(15.3637,'5',0),(19.328,'3',0),(19.4929,'3',0),(32.8616,'5',1),(34.3539,'5',0),(45.7429,'3',0)]
fig=plt.figure(figsize=(14,12.6));outer=gs.GridSpec(2,4,figure=fig,left=.01,right=.875,top=.925,bottom=.01,wspace=.03,hspace=.16)
R0=np.tanh(rho/2)*1.04;Gr=np.linspace(-R0,R0,241)
for cell,(lam,kind,pick) in zip(outer,panels):
    C,val=level_of(lam);Uu,_=inv_U(C,C5 if kind=='5' else C3);c=(C@Uu)[:,[pick]];d,e1,e2=basis(AX5 if kind=='5' else AX3)
    F=ev(c,CT)[:,0];aa,bb=np.meshgrid(Gr,Gr,indexing='ij');X=aa[...,None]*e1+bb[...,None]*e2;r2=(X*X).sum(-1);ok=r2<0.98
    x=np.concatenate([((1+r2)/(1-r2))[...,None],2*X/(1-r2)[...,None]],-1).reshape(-1,4)
    inside=ok.ravel()&((x*np.diag(Qm))@NF.T<=1e-12).all(1);Fs=np.full(len(x),np.nan);Fs[inside]=ev(c,x[inside])[:,0]
    m=max(np.abs(F).max(),np.nanmax(np.abs(Fs)));F/=m;Fs/=m
    inner=cell.subgridspec(2,1,height_ratios=[1.35,1],hspace=.0);ax=fig.add_subplot(inner[0])
    vis=(NRM@d)>0;o=np.argsort((XC@d)[vis]);P=np.stack([XT[vis]@e1,XT[vis]@e2],-1)[o]
    Lt=np.array([-.35,.45,.82]);Lt/=np.linalg.norm(Lt);nv=np.stack([NRM[vis]@e1,NRM[vis]@e2,NRM[vis]@d],-1)[o]
    col=CMAP(.5+.5*F[vis][o]);col[:,:3]*=(0.55+0.45*np.clip(nv@Lt,0,1))[:,None];col[LM_[vis][o]<0.035,:3]=0.2
    ax.add_collection(PolyCollection(P,facecolors=col,edgecolors='none',antialiaseds=False,rasterized=True))
    ax.set_xlim(-R0,R0);ax.set_ylim(-R0,R0);ax.set_aspect('equal');ax.axis('off')
    ax.set_title(f'eigenvalue {val:.4f}\n{"five" if kind=="5" else "three"}-fold axis',fontsize=18,pad=2,linespacing=1.15)
    ax2=fig.add_subplot(inner[1]);ax2.imshow(Fs.reshape(aa.shape).T,origin='lower',cmap=CMAP,vmin=-1,vmax=1,extent=[-R0,R0,-R0,R0],interpolation='bilinear')
    ax2.contour(Gr,Gr,np.isfinite(Fs).reshape(aa.shape).T.astype(float),levels=[.5],colors='0.2',linewidths=1.2);ax2.axis('off')
cax=fig.add_axes([.888,.2,.018,.6]);cb=fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP,norm=plt.Normalize(-1,1)),cax=cax)
cb.set_label('scaled eigenfunction value',fontsize=21);cb.ax.tick_params(labelsize=19)
fig.savefig('figures/seifert_weber_space_modes.png',dpi=220);plt.close(fig)
