"""Figure: eight Laplace-Beltrami eigenfunctions on the trefoil tube surface (needs run/spec_144_12_5.npz from run_spec.py 144 12 5 12)."""
import sys,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
sys.path.insert(0,'.')
import trefoil_hp as T
try:
    from cmcrameri import cm as ccm; CMAP=ccm.vik
except Exception: CMAP=plt.get_cmap('BrBG_r')
try:
    plt.rcParams.update({'text.usetex':True,'font.family':'serif'})
    f=plt.figure();f.text(.5,.5,'test 1.5');f.savefig('run/_usetex_probe.png');plt.close(f)
except Exception as e:
    print('usetex off',e);plt.rcParams.update({'text.usetex':False,'font.family':'serif','mathtext.fontset':'cm'})
Nu,Nv,p=144,12,5; su,sv=10,3
D=np.load(f'run/spec_{Nu}_{Nv}_{p}.npz');vals,vecs,gd=D['vals'],D['vecs'],D['gd']
# local sample points on a cell square, split by triangle
A,Bq=np.meshgrid(np.linspace(0,1,su+1),np.linspace(0,1,sv+1),indexing='ij')
a=A.ravel();b=Bq.ravel();low=a>=b
lamL=np.stack([1-a,a-b,b],-1)[low];lamU=np.stack([1-b,a,b-a],-1)[~low]
_,BL,_=T.bernstein_reference(p,lamL);_,BU,_=T.bernstein_reference(p,lamU)
def sample(c):
    F=np.zeros((Nu*su+1,Nv*sv+1))
    cl=c[gd[0::2]];cu=c[gd[1::2]]          # (ncell_sq,nloc), ordering j*Nu+i
    loc=np.empty((Nu*Nv,len(a)));loc[:,low]=cl@BL.T;loc[:,~low]=cu@BU.T
    loc=loc.reshape(Nv,Nu,su+1,sv+1)
    for j in range(Nv):
        for i in range(Nu):
            F[i*su:i*su+su+1,j*sv:j*sv+sv+1]=loc[j,i]
    return F
# embedding for display only
u=np.linspace(0,T.TAU,Nu*su+1);v=np.linspace(0,T.TAU,Nv*sv+1)
c0,cp,cpp,_=T.curve_derivatives(u)
t=cp/np.linalg.norm(cp,axis=1)[:,None];nr=cpp-np.sum(cpp*t,1)[:,None]*t;n=nr/np.linalg.norm(nr,axis=1)[:,None];bn=np.cross(t,n)
nrm=np.cos(v)[None,:,None]*n[:,None,:]+np.sin(v)[None,:,None]*bn[:,None,:]
X=c0[:,None,:]+T.RADIUS*nrm
quad=lambda Z:0.25*(Z[:-1,:-1]+Z[1:,:-1]+Z[1:,1:]+Z[:-1,1:])
P=np.stack([X[:-1,:-1,:2],X[1:,:-1,:2],X[1:,1:,:2],X[:-1,1:,:2]],2).reshape(-1,4,2)
zc=quad(X[...,2]).ravel();nc=quad(nrm).reshape(-1,3)
L=np.array([-0.35,0.45,0.82]);L/=np.linalg.norm(L)
shade=0.50+0.50*np.clip(nc@L,0,1)
vis=nc[:,2]>-0.05;order=np.argsort(zc[vis])
Pv=P[vis][order];sh=shade[vis][order]
import matplotlib.gridspec as gs
plt.rcParams.update({'font.family':'cmr10','mathtext.fontset':'cm','axes.unicode_minus':False,'axes.formatter.use_mathtext':True})
sel=[1,45,59,170,233,465,470,480]
fig=plt.figure(figsize=(14,10.2))
outer=gs.GridSpec(2,4,figure=fig,left=.01,right=.875,top=.92,bottom=.02,wspace=.04,hspace=.22,)
for cell,k in zip(outer,sel):
    inner=cell.subgridspec(2,1,height_ratios=[4.2,1],hspace=.03)
    F=sample(vecs[:,k]);F/=np.abs(F).max()
    ax=fig.add_subplot(inner[0])
    col=CMAP(0.5+0.5*quad(F).ravel()[vis][order]);col[:,:3]*=sh[:,None]
    ax.add_collection(PolyCollection(Pv,facecolors=col,edgecolors='none',antialiaseds=False,rasterized=True))
    ax.set_xlim(-3.4,3.4);ax.set_ylim(-3.6,3.0);ax.set_aspect('equal');ax.axis('off')
    ax.set_title(f'mode {k}\neigenvalue {vals[k]:.4g}',fontsize=20,pad=6,linespacing=1.15)
    ax2=fig.add_subplot(inner[1])
    ax2.imshow(F.T,origin='lower',cmap=CMAP,vmin=-1,vmax=1,aspect='auto',extent=[0,T.TAU,0,T.TAU],interpolation='bilinear')
    ax2.set_xticks([]);ax2.set_yticks([])
cax=fig.add_axes([.888,.2,.018,.6]);sm=plt.cm.ScalarMappable(cmap=CMAP,norm=plt.Normalize(-1,1))
cb=fig.colorbar(sm,cax=cax);cb.set_label('scaled eigenfunction value',fontsize=21);cb.ax.tick_params(labelsize=19)
fig.savefig('figures/trefoil_selected_modes.png',dpi=260);plt.close(fig)
