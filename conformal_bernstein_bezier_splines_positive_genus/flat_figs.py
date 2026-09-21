"""3D renderings of the flat quotients with the criss-cross mesh mapped by a
standard embedding/immersion: torus, Mobius strip, Klein bottle (figure-8)."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from flatquot import crisscross_mesh
plt.rcParams.update({'font.family':'serif','font.serif':['cmr10','DejaVu Serif'],'mathtext.fontset':'cm','font.size':9})

def torus(x,y,R=2.0,r=0.8):
    u,v=2*np.pi*x,2*np.pi*y
    return np.array([(R+r*np.cos(v))*np.cos(u),(R+r*np.cos(v))*np.sin(u),r*np.sin(v)])
def mobius(x,y,R=2.0,wd=0.9):
    u=2*np.pi*x; s=wd*(y-0.5)
    return np.array([(R+s*np.cos(u/2))*np.cos(u),(R+s*np.cos(u/2))*np.sin(u),s*np.sin(u/2)])
def klein8(x,y,R=2.2,r=0.9):
    # figure-8 immersion; x along the tube, y around; the y-glide (x-> a-x at y=b) is realized
    u,v=2*np.pi*x,2*np.pi*y
    return np.array([(R+r*np.cos(u/2)*np.sin(v)-r*np.sin(u/2)*np.sin(2*v))*np.cos(u),
                     (R+r*np.cos(u/2)*np.sin(v)-r*np.sin(u/2)*np.sin(2*v))*np.sin(u),
                     r*np.sin(u/2)*np.sin(v)+r*np.cos(u/2)*np.sin(2*v)])

def render(name, emb, n, m, elev, azim, out, sub=6):
    V,T=crisscross_mesh(n,m)
    fig=plt.figure(figsize=(4.2,3.6)); ax=fig.add_subplot(111,projection='3d')
    polys=[]; shade=[]
    for tri in T:
        P=V[list(tri)]
        # curved element: subdivide in barycentric coordinates
        pts=[]
        for i in range(sub+1):
            for j in range(sub+1-i):
                b=np.array([i,j,sub-i-j])/sub; p=b@P; pts.append(emb(p[0],p[1]))
        pts=np.array(pts)
        # triangles of the subgrid
        idx=lambda i,j: sum(sub+1-k for k in range(i))+j
        for i in range(sub):
            for j in range(sub-i):
                polys.append([pts[idx(i,j)],pts[idx(i+1,j)],pts[idx(i,j+1)]])
                if j<sub-i-1: polys.append([pts[idx(i+1,j)],pts[idx(i+1,j+1)],pts[idx(i,j+1)]])
    polys=np.array(polys)
    nrm=np.cross(polys[:,1]-polys[:,0],polys[:,2]-polys[:,0]); nrm/=np.linalg.norm(nrm,axis=1)[:,None]+1e-15
    light=np.array([0.3,-0.5,0.8]); light/=np.linalg.norm(light)
    sh=0.55+0.45*np.abs(nrm@light)
    col=np.array([0.80,0.83,0.90])
    pc=Poly3DCollection(polys,facecolors=np.clip(col[None,:]*sh[:,None],0,1),edgecolor='none',linewidth=0)
    ax.add_collection3d(pc)
    # mesh edges as curves
    segs=[]
    for tri in T:
        P=V[list(tri)]
        for a,b in [(0,1),(1,2),(2,0)]:
            ts=np.linspace(0,1,12); pts=np.array([emb(*(P[a]*(1-t)+P[b]*t)) for t in ts]); segs.append(pts)
    ax.add_collection3d(Line3DCollection(segs,colors='0.25',linewidths=0.5))
    allp=polys.reshape(-1,3); c=allp.mean(0); rr=np.abs(allp-c).max()
    ax.set_xlim(c[0]-rr,c[0]+rr); ax.set_ylim(c[1]-rr,c[1]+rr); ax.set_zlim(c[2]-rr,c[2]+rr)
    ax.set_box_aspect((1,1,1)); ax.view_init(elev=elev,azim=azim); ax.set_axis_off()
    fig.subplots_adjust(0,0,1,1); fig.savefig(out,dpi=220); plt.close(fig)
    print(out)

if __name__ == '__main__':
    render('torus',torus,8,8,35,-60,'/home/claude/work/hyp/paper/fig_torus_mesh.png')
    render('mobius',mobius,12,3,40,-50,'/home/claude/work/hyp/paper/fig_mobius_mesh.png')
    render('klein',klein8,10,10,28,-55,'/home/claude/work/hyp/paper/fig_klein_mesh.png')
