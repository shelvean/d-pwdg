"""Physical rendering of the distorted torus in R^3 with the criss-cross mesh
of the flat torus [0,U]x[0,2pi] carried through the conformal embedding, the
edges drawn as curves lying on the surface. Raw Mayavi panels, no text."""
import numpy as np
from mayavi import mlab
mlab.options.offscreen = True
from flatquot import crisscross_mesh
from dtorus import U, rho, zz, t_of, weight
from flatquot import Quotient

def emb(u, phi):
    t = t_of(u)
    return np.array([rho(t)*np.cos(phi), rho(t)*np.sin(phi), zz(t)])

def render(n, m, out, scalars=None, sub=8, tube=0.013, size=(1500,1100)):
    V, T = crisscross_mesh(n, m, U, 2*np.pi)
    X=[]; F=[]; S=[]; base=0
    for tri in T:
        P = V[list(tri)]; idx={}
        for i in range(sub+1):
            for j in range(sub+1-i):
                b=np.array([i,j,sub-i-j])/sub; p=b@P
                idx[(i,j)]=len(X); X.append(emb(p[0],p[1]))
                if scalars is not None: S.append(scalars(p))
        for i in range(sub):
            for j in range(sub-i):
                F.append([idx[(i,j)],idx[(i+1,j)],idx[(i,j+1)]])
                if j<sub-i-1: F.append([idx[(i+1,j)],idx[(i+1,j+1)],idx[(i,j+1)]])
        base=len(X)
    X=np.array(X); F=np.array(F)
    key=np.round(X,6); _,inv=np.unique(key,axis=0,return_inverse=True)
    Xw=np.zeros((inv.max()+1,3)); Xw[inv]=X; Fw=inv[F]
    fig=mlab.figure(bgcolor=(1,1,1),size=size)
    if scalars is None:
        s=mlab.triangular_mesh(Xw[:,0],Xw[:,1],Xw[:,2],Fw,color=(0.86,0.88,0.93))
    else:
        Sw=np.zeros(inv.max()+1); Sw[inv]=np.array(S)
        s=mlab.triangular_mesh(Xw[:,0],Xw[:,1],Xw[:,2],Fw,scalars=Sw,colormap='RdBu',
                               vmin=-np.abs(Sw).max(),vmax=np.abs(Sw).max())
        s.module_manager.scalar_lut_manager.reverse_lut=True
    s.actor.property.specular=0.1
    # mesh edges as curves on the surface, one line source
    px,py,pz,conn=[],[],[],[]; k=0
    for tri in T:
        P=V[list(tri)]
        for a,b in [(0,1),(1,2),(2,0)]:
            ts=np.linspace(0,1,14)
            pts=np.array([emb(*(P[a]*(1-t)+P[b]*t)) for t in ts])
            px+=list(pts[:,0]); py+=list(pts[:,1]); pz+=list(pts[:,2])
            conn+=[(k+i,k+i+1) for i in range(len(ts)-1)]; k+=len(ts)
    src=mlab.pipeline.scalar_scatter(np.array(px),np.array(py),np.array(pz))
    src.mlab_source.dataset.lines=np.array(conn)
    lines=mlab.pipeline.stripper(src)
    tb=mlab.pipeline.tube(lines,tube_radius=tube); tb.filter.number_of_sides=8
    mlab.pipeline.surface(tb,color=(0.22,0.22,0.28))
    mlab.view(azimuth=-62,elevation=62,distance='auto'); fig.scene.parallel_projection=True
    mlab.savefig(out,size=size); mlab.close(fig); print(out,flush=True)

if __name__=='__main__':
    render(10,10,'/home/claude/work/hyp/paper/p_dtorus_mesh.png')
    # an eigenfunction on the surface
    Q=Quotient('torus',10,10,6,1,a=U,b=2*np.pi)
    vals,vecs,Z=Q.eigen(k=8,w=weight,q=10)
    c=vecs[:,5]
    render(10,10,'/home/claude/work/hyp/paper/p_dtorus_mode.png',
           scalars=lambda p: Q.evaluate(c,np.array([p]))[0], sub=6)
    print('mu =', vals[5])
