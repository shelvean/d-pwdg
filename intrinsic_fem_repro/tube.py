import sys,time,numpy as np,scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.spatial import Delaunay
sys.path.insert(0,'.');import trefoil_hp as T
import bb3d
a=0.35;TAU=2*np.pi
def disk(nr):
    pts=[(0.,0.)]
    for k in range(1,nr+1):
        th=np.arange(6*k)*TAU/(6*k);pts+=list(zip(a*k/nr*np.cos(th),a*k/nr*np.sin(th)))
    pts=np.array(pts);tri=Delaunay(pts).simplices
    return pts,np.sort(tri,axis=1)
def build(Ns,nr):
    pts,tri=disk(nr);nv=len(pts);ds=TAU/Ns;cells=[];coords=[]
    for k in range(Ns):
        for (i0,i1,i2) in tri:
            b=[k*nv+i for i in (i0,i1,i2)];t=[((k+1)%Ns)*nv+i for i in (i0,i1,i2)]
            xb=[np.r_[k*ds,pts[i]] for i in (i0,i1,i2)];xt=[np.r_[(k+1)*ds,pts[i]] for i in (i0,i1,i2)]
            for conn,xx in (((b[0],b[1],b[2],t[2]),(xb[0],xb[1],xb[2],xt[2])),
                            ((b[0],b[1],t[1],t[2]),(xb[0],xb[1],xt[1],xt[2])),
                            ((b[0],t[0],t[1],t[2]),(xb[0],xt[0],xt[1],xt[2]))):
                cells.append(conn);coords.append(np.array(xx))
    return np.array(cells),np.array(coords),pts,tri
def g_trefoil(x):
    s,u,v=x[...,0],x[...,1],x[...,2];q,kap,tau=T.curve_invariants(s)
    G=np.zeros(x.shape[:-1]+(3,3))
    G[...,0,0]=q*q*((1-kap*u)**2+tau*tau*(u*u+v*v));G[...,0,1]=G[...,1,0]=-q*tau*v
    G[...,0,2]=G[...,2,0]=q*tau*u;G[...,1,1]=1;G[...,2,2]=1
    return G
def g_torus(x,R=2.2):
    G=np.zeros(x.shape[:-1]+(3,3));G[...,0,0]=(R+x[...,1])**2;G[...,1,1]=1;G[...,2,2]=1;return G
if __name__=='__main__':
    which,Ns,nr,p,nm=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])
    gfun=g_trefoil if which=='trefoil' else g_torus
    cells,coords,pts,tri=build(Ns,nr);gd,ndof,keys=bb3d.dofs_by_vertex_ids(cells,p)
    E=np.stack([coords[:,j]-coords[:,0] for j in (1,2,3)],-1)     # (nc,3,3) columns
    def mf(idx,X):
        x=coords[idx,0][:,None,:]+np.einsum('cij,qj->cqi',E[idx],X)
        return np.einsum('cki,cqkl,clj->cqij',E[idx],gfun(x),E[idx])
    t=time.time();K,M,vol=bb3d.assemble(gd,ndof,p,p+5,mf);print('cells',len(cells),'dofs',ndof,'volume',vol,'asm',time.time()-t,flush=True)
    out={}
    for bc in sys.argv[6].split(','):
        t=time.time()
        if bc=='dirichlet':
            bd=bb3d.boundary_dofs(cells,keys);free=np.setdiff1d(np.arange(ndof),bd)
            vals,v=eigsh(K[free][:,free].tocsc(),M=M[free][:,free].tocsc(),k=nm,sigma=40.,which='LM');vecs=np.zeros((ndof,nm));vecs[free]=v
        else:vals,vecs=eigsh(K.tocsc(),M=M.tocsc(),k=nm,sigma=-0.01,which='LM')
        o=np.argsort(vals);vals=vals[o];vecs=vecs[:,o];print(bc,time.time()-t,'s');print(np.round(vals[:40],4),flush=True)
        out[bc+'_vals']=vals;out[bc+'_vecs']=vecs
    np.savez_compressed(f'run/tube_{which}_{Ns}_{nr}_{p}_{sys.argv[6][0]}.npz',gd=gd,cells=cells,coords=coords,pts=pts,tri=tri,**out)
