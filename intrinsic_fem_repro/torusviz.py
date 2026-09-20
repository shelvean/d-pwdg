import itertools,numpy as np,bb3d
from skimage.measure import marching_cubes
N,p=6,4;D=np.load(f'run/torus3_{N}_{p}.npz');vals,vecs,gd=D['vals'],D['vecs'],D['gd']
PERMS=list(itertools.permutations(range(3)));PIDX={pm:i for i,pm in enumerate(PERMS)}
def ev(c,x):
    x=x%1.0;ijk=np.minimum((x*N).astype(int),N-1);loc=x*N-ijk;o=np.argsort(-loc,axis=1,kind='stable')
    pid=np.array([PIDX[tuple(r)] for r in o]);s=np.take_along_axis(loc,o,1)
    lam=np.c_[1-s[:,0],s[:,0]-s[:,1],s[:,1]-s[:,2],s[:,2]];cell=((ijk[:,0]*N+ijk[:,1])*N+ijk[:,2])*6+pid
    out=np.empty(len(x))
    for a in range(0,len(x),20000):
        sl=slice(a,a+20000);out[sl]=(bb3d.bern(p,lam[sl],grad=False)*c[gd[cell[sl]]]).sum(1)
    return out
def field(k,n=56):
    g=np.linspace(0,1,n+1);X=np.stack(np.meshgrid(g,g,g,indexing='ij'),-1).reshape(-1,3)
    F=ev(vecs[:,k],X).reshape(n+1,n+1,n+1);return F/np.abs(F).max()
def view(az=-58,el=24):
    az,el=np.radians(az),np.radians(el);d=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    e1=np.cross([0,0,1.],d);e1/=np.linalg.norm(e1);return d,e1,np.cross(d,e1)
def draw(ax,F,CMAP,level=0.45):
    from matplotlib.collections import PolyCollection
    d,e1,e2=view();L=np.array([0.3,-0.5,0.8]);L/=np.linalg.norm(L);n=F.shape[0]-1;P=[];C=[];Z=[]
    for sg in (1,-1):
        if (sg*F).max()<level:continue
        v,f,nr,_=marching_cubes(sg*F,level,spacing=(1/n,)*3);tri=v[f];nn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        nn/=np.linalg.norm(nn,axis=1,keepdims=True)+1e-30;sh=0.35+0.65*np.abs(nn@L)
        col=np.array(CMAP(0.5+0.5*sg*0.8))[None,:]*np.ones((len(tri),1));col[:,:3]*=sh[:,None]
        P.append(np.stack([tri@e1,tri@e2],-1));C.append(col);Z.append(tri.mean(1)@d)
    P=np.concatenate(P);C=np.concatenate(C);o=np.argsort(np.concatenate(Z))
    cor=np.array(list(itertools.product((0,1),repeat=3)),float)
    edges=[(a,b) for a in range(8) for b in range(a+1,8) if np.abs(cor[a]-cor[b]).sum()==1]
    back=cor[np.argmin(cor@d)]
    for a,b in edges:
        hidden=(cor[a]==back).all() or (cor[b]==back).all()
        ax.plot(*zip(*[(cor[i]@e1,cor[i]@e2) for i in (a,b)]),color='0.35',lw=1.0,zorder=0 if hidden else 3,ls=':' if hidden else '-')
    ax.add_collection(PolyCollection(P[o],facecolors=C[o],edgecolors='none',antialiaseds=False,rasterized=True,zorder=1))
    ax.set_aspect('equal');ax.autoscale_view();ax.axis('off')
