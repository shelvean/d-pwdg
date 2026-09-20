"""Intrinsic finite elements on swept 3D volume meshes (no polar singularity).

The computational coordinates are (s,u,v), with a periodic s-axis and a
triangulated Cartesian disk in (u,v).  The metrics are supplied directly in
these coordinates and pulled back to affine reference tetrahedra.

The visualization map is an optional, separate geometry provider.  FEM
assembly never calls it.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from .reference import SimplexBernstein, simplex_duffy

PI2=2.0*np.pi

@dataclass
class TubeMesh:
    coords: np.ndarray           # global vertices, (s,u,v), with s in [0,2pi)
    tets: np.ndarray             # tetrahedra modulo periodic s seam
    local_coords: np.ndarray     # unwrapped local chart coordinates (ntet,4,3)
    faces_boundary: np.ndarray   # global boundary facet vertex ids
    disk_xy: np.ndarray
    disk_tri: np.ndarray
    disk_outer: np.ndarray
    ns: int
    ndisk: int
    radius: float
    

def disk_triangulation(radius=.35, nr=2, nt=12):
    if nr<1 or nt<5: raise ValueError('require nr>=1 and nt>=5')
    pos=[(0.,0.)]
    for j in range(1,nr+1):
        rr=radius*j/nr
        for i in range(nt):
            a=PI2*i/nt
            pos.append((rr*np.cos(a),rr*np.sin(a)))
    tri=[]
    for i in range(nt):
        tri.append((0,1+i,1+(i+1)%nt))
    for j in range(1,nr):
        lo=1+(j-1)*nt; hi=1+j*nt
        for i in range(nt):
            ip=(i+1)%nt
            tri.extend(((lo+i,hi+i,hi+ip),(lo+i,hi+ip,lo+ip)))
    tri=np.asarray(tri,int)
    # Canonical global numbering ensures compatible Freudenthal prism faces.
    tri=np.sort(tri,axis=1)
    outer=np.arange(1+(nr-1)*nt,1+nr*nt,dtype=int)
    return np.asarray(pos,float),tri,outer


def swept_mesh(ns=12, nr=2, nt=12, radius=.35):
    if ns<5: raise ValueError('ns>=5 required')
    xy, disk_tets, outer=disk_triangulation(radius,nr,nt)
    nd=len(xy)
    svals=PI2*np.arange(ns)/ns
    coords=np.column_stack((np.repeat(svals,nd),np.tile(xy,(ns,1))))
    tet=[]; lc=[]
    for j in range(ns):
        a=PI2*j/ns; b=PI2*(j+1)/ns
        lower=np.arange(nd,dtype=int)+j*nd
        upper=np.arange(nd,dtype=int)+((j+1)%ns)*nd
        for q in disk_tets:
            i0,i1,i2=map(int,q)
            a0,a1,a2=map(int,lower[q])
            b0,b1,b2=map(int,upper[q])
            base=[(a0,a1,a2,b2),(a0,a1,b1,b2),(a0,b0,b1,b2)]
            # Get unwrapped s locally, so the last slab uses upper s=2*pi.
            for g in base:
                p=coords[list(g)].copy()
                for n,node in enumerate(g):
                    if node in upper: p[n,0]=b
                    else: p[n,0]=a
                tet.append(g);lc.append(p)
    tet=np.asarray(tet,int);lc=np.asarray(lc,float)
    # Every periodic seam is an interior face.  Boundary facets are found
    # from tetrahedron face incidence, not inferred from vertex classes.
    from collections import Counter
    face_count=Counter()
    for t in tet:
        for omit in range(4):
            face_count[tuple(sorted(np.delete(t,omit)))] += 1
    bad={v for v in face_count.values() if v not in (1,2)}
    if bad: raise RuntimeError(f'non-manifold face incidences: {bad}')
    bf=np.asarray([k for k,v in face_count.items() if v==1],int)
    if len(bf)!=2*ns*nt: raise RuntimeError(f'boundary mismatch: {len(bf)} vs {2*ns*nt}')
    if np.any(np.min(np.linalg.det(lc[:,1:,:]-lc[:,[0],:]),axis=0)==0):
        pass
    return TubeMesh(coords,tet,lc,bf,xy,disk_tets,outer,ns,nd,radius)

class RoundTorus:
    name='round_solid_torus'
    def __init__(self,R=2.2): self.R=float(R)
    def metric(self,s,u,v):
        H=self.R+u
        if np.any(H<=0): raise ValueError('tube intersects its center axis')
        z=np.zeros_like(H);o=np.ones_like(H)
        return np.stack((np.stack((H*H,z,z),axis=-1),
                         np.stack((z,o,z),axis=-1),
                         np.stack((z,z,o),axis=-1)),axis=-2)
    def physical(self,s,u,v):
        s=np.asarray(s)
        return np.stack(((self.R+u)*np.cos(s),
                         (self.R+u)*np.sin(s),v),axis=-1)
    def centerline_length(self): return PI2*self.R

class TwistedRoundTorus(RoundTorus):
    """The same circular solid torus in an s-dependent rotating disk chart.

    For integer winding m, the map is globally periodic. This is an intrinsic
    coordinate covariance test: its continuum spectrum is EXACTLY that of the
    standard round solid torus.  Finite-mesh spectra can differ because the
    two parameter-space meshes represent different physical partitions.
    """
    name='twisted_chart_solid_torus'
    def __init__(self,R=2.2,winding=1):
        super().__init__(R)
        if int(winding)!=winding: raise ValueError('periodic frame winding must be integer')
        self.winding=int(winding)
    def _rotation(self,s,u,v):
        c=np.cos(self.winding*s); t=np.sin(self.winding*s)
        return c*u-t*v,t*u+c*v
    def metric(self,s,u,v):
        m=self.winding
        uu,vv=self._rotation(s,u,v)
        Q=(self.R+uu)**2 + m*m*(u*u+v*v)
        su=-m*v;sv=m*u
        z=np.zeros_like(Q);o=np.ones_like(Q)
        return np.stack((np.stack((Q,su,sv),axis=-1),
                         np.stack((su,o,z),axis=-1),
                         np.stack((sv,z,o),axis=-1)),axis=-2)
    def physical(self,s,u,v):
        uu,vv=self._rotation(s,u,v)
        return super().physical(s,uu,vv)

class TrefoilTube:
    name='trefoil_solid_tube'
    def __init__(self, radius=.35):
        self.radius=float(radius)
        test=np.linspace(0,PI2,4097)
        q,ka,ta,_=self.frenet_data(test)
        if self.radius*np.max(ka)>=.8:
            raise ValueError('tube radius too large for the selected centerline curvature')
    @staticmethod
    def c(s):
        return np.stack((np.sin(s)+2*np.sin(2*s),
                         np.cos(s)-2*np.cos(2*s), -np.sin(3*s)),axis=-1)
    @staticmethod
    def cp(s):
        return np.stack((np.cos(s)+4*np.cos(2*s),
                         -np.sin(s)+4*np.sin(2*s), -3*np.cos(3*s)),axis=-1)
    @staticmethod
    def cpp(s):
        return np.stack((-np.sin(s)-8*np.sin(2*s),
                         -np.cos(s)+8*np.cos(2*s), 9*np.sin(3*s)),axis=-1)
    @staticmethod
    def cppp(s):
        return np.stack((-np.cos(s)-16*np.cos(2*s),
                         np.sin(s)-16*np.sin(2*s),27*np.cos(3*s)),axis=-1)
    @classmethod
    def frenet_data(cls,s):
        cp=cls.cp(s); cpp=cls.cpp(s); cppp=cls.cppp(s)
        q=np.linalg.norm(cp,axis=-1)
        T=cp/q[...,None]
        cross=np.cross(cp,cpp)
        crossn=np.linalg.norm(cross,axis=-1)
        ka=crossn/q**3
        tors=np.einsum('...i,...i->...',cross,cppp)/crossn**2
        B=cross/crossn[...,None]
        N=np.cross(B,T)
        return q,ka,tors,(T,N,B)
    def metric(self,s,u,v):
        q,k,t,_=self.frenet_data(s)
        Q=q*q*((1-k*u)**2+t*t*(u*u+v*v))
        su=-q*t*v; sv=q*t*u
        z=np.zeros_like(Q);o=np.ones_like(Q)
        return np.stack((np.stack((Q,su,sv),axis=-1),
                         np.stack((su,o,z),axis=-1),
                         np.stack((sv,z,o),axis=-1)),axis=-2)
    def physical(self,s,u,v):
        _,_,_,(T,N,B)=self.frenet_data(s)
        return self.c(s)+u[...,None]*N+v[...,None]*B
    def centerline_length(self):
        from scipy.integrate import quad
        return quad(lambda ss: float(np.linalg.norm(self.cp(np.array(ss)))),0,PI2,
                    epsabs=1e-12,epsrel=1e-12)[0]


def global_bernstein_dofs(mesh,p):
    space=SimplexBernstein(3,p)
    # Barycentric domain points with zero coefficients omitted. A sorted
    # tuple (vertex id, alpha) is invariant under local vertex reordering.
    allkeys={}; maps=np.empty((len(mesh.tets),space.nloc),int)
    boundary_set={frozenset(map(int,f)) for f in mesh.faces_boundary}
    boundary_support=set()
    from itertools import combinations
    for face in boundary_set:
        for m in (1,2,3):
            for subset in combinations(sorted(face),m):
                boundary_support.add(tuple(subset))
    boundary_key=[]
    for i,tet in enumerate(mesh.tets):
        for b,alpha in enumerate(space.indices):
            key=tuple(sorted((int(v),int(a)) for v,a in zip(tet,alpha) if a>0))
            if key not in allkeys:
                allkeys[key]=len(allkeys)
                boundary_key.append(tuple(v for v,a in key) in boundary_support)
            maps[i,b]=allkeys[key]
    if len(set(map(tuple,maps.tolist())))==0: raise RuntimeError('missing dofs')
    return space,maps,np.asarray(boundary_key,bool)


def tabulate(space,q):
    L,W=simplex_duffy(3,q)
    V=[];D=[]
    for lam in L:
        b,g=space.values_grads(lam)
        V.append(b);D.append(g)
    return L,W,np.asarray(V),np.asarray(D)


def local_matrices(mesh, geom, p=1, q=3, progress=False):
    """Assemble standard constrained C^0 volume FEM, including periodic seam."""
    space, ids, bdy=global_bernstein_dofs(mesh,p)
    L,W,V,D=tabulate(space,q)
    nloc=space.nloc; n=len(bdy)
    ii=[];jj=[];Km=[];Mm=[];volume=0.
    for it,pcoords in enumerate(mesh.local_coords):
        # map from barycentric reference tetra to intrinsic chart coordinates
        xyz=L@pcoords
        A=(pcoords[1:]-pcoords[0]).T
        detA=float(np.linalg.det(A))
        if abs(detA)<1e-13: raise RuntimeError('degenerate chart tetrahedron')
        Gc=geom.metric(xyz[:,0],xyz[:,1],xyz[:,2])
        G=np.einsum('ia,qij,jb->qab',A,Gc,A,optimize=True)
        detG=np.linalg.det(G)
        if np.any(detG<1e-20): raise RuntimeError('nonpositive pullback metric')
        weight=W*np.sqrt(detG)
        invG=np.linalg.inv(G)
        # Matrix-valued metric is evaluated on each reference tetra;
        # neither assembly nor reference space accesses geom.physical().
        M=np.einsum('q,qi,qj->ij',weight,V,V,optimize=True)
        K=np.einsum('q,qia,qab,qjb->ij',weight,D,invG,D,optimize=True)
        volume+=np.dot(weight,np.ones_like(weight))
        localids=ids[it]
        ii.extend(np.repeat(localids,nloc));jj.extend(np.tile(localids,nloc))
        Km.extend(K.reshape(-1));Mm.extend(M.reshape(-1))
        if progress and it%2000==0: print('cell',it,'/',len(mesh.tets),flush=True)
    shape=(n,n)
    K=sp.coo_matrix((Km,(ii,jj)),shape=shape).tocsr()
    M=sp.coo_matrix((Mm,(ii,jj)),shape=shape).tocsr()
    K=(K+K.T)*.5;M=(M+M.T)*.5
    return dict(mesh=mesh,geom=geom,p=p,ids=ids,boundary=bdy,space=space,K=K,M=M,
                volume=float(volume),ndof=n)


def solve_modes(A,bc='neumann',npositive=10,tol=1e-8):
    """Return all nonzero eigenpairs and report numerical zero for Neumann."""
    boundary=A['boundary']
    if bc not in ('neumann','dirichlet'):raise ValueError(bc)
    kept=np.arange(A['ndof']) if bc=='neumann' else np.flatnonzero(~boundary)
    K=A['K'][kept,:][:,kept]
    M=A['M'][kept,:][:,kept]
    # A small negative shift makes K-sigma*M positive definite for Neumann.
    nreq=npositive+(bc=='neumann')
    k=min(len(kept)-2,nreq+2)
    vals,vec=sla.eigsh(K,k=k,M=M,sigma=-1e-3,which='LM',tol=tol)
    sort=np.argsort(vals); vals=vals[sort];vec=vec[:,sort]
    if bc=='neumann':
        if abs(vals[0])>5e-5: raise RuntimeError(f'no Neumann zero mode: {vals[0]}')
        # first positive computed modes
        pick=np.where(vals>max(1e-5,1000*abs(vals[0])))[0][:npositive]
    else:pick=np.arange(min(npositive,len(vals)))
    full=np.zeros((A['ndof'],len(pick)))
    full[kept]=vec[:,pick]
    return dict(bc=bc,zero_eigenvalue=float(vals[0]) if bc=='neumann' else None,
                eigenvalues=vals[pick].copy(),eigenvectors=full,kept=len(kept),
                all_low=vals)


def nodal_values(A,coeff):
    """Get P1 vertex values from any polynomial order via Bernstein dofs."""
    p=A['p'];res=np.empty(len(A['mesh'].coords));seen=np.zeros(len(res),bool)
    for t,ids in zip(A['mesh'].tets,A['ids']):
        for i,alpha in enumerate(A['space'].indices):
            nz=[k for k,a in enumerate(alpha) if a]
            if len(nz)==1 and alpha[nz[0]]==p:
                v=t[nz[0]]
                if not seen[v]:res[v]=coeff[ids[i]];seen[v]=True
    if not np.all(seen): raise RuntimeError('not all vertices represented')
    return res
