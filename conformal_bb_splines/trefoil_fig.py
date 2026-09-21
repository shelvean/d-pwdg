import numpy as np
from mayavi import mlab
mlab.options.offscreen = True
from flatquot import Quotient, crisscross_mesh
from metric_assembly import assemble_metric
from trefoil import emb, metric, L, CIRC
import scipy.sparse.linalg as spla
def surf_grid(nu=681, nv=41):
    us=np.linspace(0,L,nu); vs=np.linspace(0,CIRC,nv); U,Vg=np.meshgrid(us,vs,indexing='ij')
    P=np.array([U.ravel(),Vg.ravel()]).T; X=np.array([emb(u,v) for u,v in P])
    tris=[]
    for i in range(nu-1):
        for j in range(nv-1):
            a,b,c,d=i*nv+j,(i+1)*nv+j,i*nv+j+1,(i+1)*nv+j+1; tris+=[(a,b,d),(a,d,c)]
    return P,X,np.array(tris)
P,X,tris=surf_grid()
# eigenfunction panel
Q=Quotient('torus',68,6,6,1,a=L,b=CIRC); K,M=assemble_metric(Q,metric,q=9); Z=Q.nullspace()
vals,vecs=spla.eigsh((Z.T@K@Z).tocsc(),k=130,M=(Z.T@M@Z).tocsc(),sigma=-1.0,which='LM'); o=np.argsort(vals); vals=vals[o]; vecs=Z@vecs[:,o]
from flatquot import bernstein
def fast_eval(c,pts):
    hs=L/68; ht=CIRC/6; out=np.empty(len(pts)); cells={}
    for t,T_ in enumerate(Q.tris):
        ctr=Q.V[list(T_)].mean(0); cells.setdefault((min(int(ctr[0]//hs),67),min(int(ctr[1]//ht),5)),[]).append(t)
    for q,p in enumerate(pts):
        i=min(int((p[0]%L)//hs),67); j=min(int((p[1]%CIRC)//ht),5)
        for t in cells[(i,j)]:
            V3=Q.V[list(Q.tris[t])]; A=np.vstack([V3.T,np.ones(3)]); b=np.linalg.solve(A,np.array([p[0],p[1],1.0]))
            if b.min()>-1e-9: out[q]=bernstein(Q.d,b[:,None])[:,0]@c[t*Q.Nloc:(t+1)*Q.Nloc]; break
        else: out[q]=0.0
    return out
for k in [40,80,120]:
    u=fast_eval(vecs[:,k],P); u/=np.abs(u).max()
    fig=mlab.figure(bgcolor=(1,1,1),size=(1500,1100))
    s=mlab.triangular_mesh(X[:,0],X[:,1],X[:,2],tris,scalars=u,colormap='RdBu',vmin=-1,vmax=1); s.module_manager.scalar_lut_manager.reverse_lut=True; s.actor.property.specular=0.08
    mlab.view(azimuth=30,elevation=55,distance='auto'); fig.scene.parallel_projection=True
    mlab.savefig(f'/home/claude/work/hyp/paper/p_trefoil_m{k}.png',size=(1500,1100)); mlab.close(fig); print(k,vals[k],flush=True)
