"""
Contains routines for handling meshes and geometry

by Shelvean Kapita 2022


"""
import time
import numpy as np
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt,\
    path as mpltPath, colors as mc
from scipy.sparse import coo_matrix
import numpy.linalg as la


def initmesh(p):
# =============================================================================
# # creates initial course mesh
# # input mesh points e.g. p = [[0,0],[1,0],[1,1],[0,1]]
# # output vertices and triangles 
# # v = mesh.points    (x and y coordinates of vertices)
# # t = mesh.simplices  (triangles)   
# =============================================================================
    points = np.array(p)
    mesh =  Delaunay(points)
    v = mesh.points
    t = mesh.simplices
    t = orientcc(v,t) # orient all triangles cc
    return v, t

def plotmesh(v,t):
# =============================================================================
#     # plot the mesh 
# =============================================================================
    plt.axes(aspect='equal')
    plt.triplot(v[:,0], v[:,1], t)
    plt.title('Mesh')
    
def plotpatch(v,t,p):
# =============================================================================
#     # plots mesh with selected elements colored
#     # p is a vector of selected element numbers
# =============================================================================
    plt.axes(aspect='equal')
    plt.triplot(v[:,0],v[:,1],t)
    c = np.ones(p.size)
    cmap = mc.ListedColormap("red")
    plt.tripcolor(v[:,0],v[:,1],t[p,:],c,lw=2,cmap=cmap)
    plt.title('Marked for refinement')
     
def plotbdy(b,v): 
# =============================================================================
#     #plot the boundary of the mesh
#     # b is the list of boundary nodes
# =============================================================================
    x = v[b,0]; x = np.append(x,x[0])
    y = v[b,1]; y = np.append(y,y[0])
    plt.plot(x,y,'m-')
    plt.axes(aspect='equal')
    plt.title('Domain boundary')

    
def triarea(v1, v2, v3):
# =============================================================================
#     # computes the signed area of a triangle with vertices v1, v2, v3
#     # cc oriented triangles have positive signed area
#     # clockwise oriented triangles return negative signed area
# =============================================================================
    v1 = v1.reshape(1,2)
    v2 = v2.reshape(1,2)
    v3 = v3.reshape(1,2)    
    x = v1[0,0];  y = v1[0,1]
    a = v2[0,0];  b = v2[0,1]
    c = v3[0,0];  d = v3[0,1]    
    A = ((a-x)*(d-y)-(c-x)*(b-y))/2
    return A  
    
def area(v,t):
# =============================================================================
#     # returns vector of areas of the triangulation
# =============================================================================
    n = t.shape[0]
    A = np.zeros(shape=(n,1)).flatten()
    for i in np.arange(n):
        v1 = v[t[i,0],:]
        v2 = v[t[i,1],:]
        v3 = v[t[i,2],:]
        A[i] = np.abs(triarea(v1,v2,v3))
    return A

def meshsize(v,t):
# =============================================================================
#     # returns length of the longest edge of each triangle
# =============================================================================
    n = t.shape[0]
    H = np.zeros(shape=(n,1)).flatten()
    for i in np.arange(n):
        v1 = v[t[i,0],:]
        v2 = v[t[i,1],:]
        v3 = v[t[i,2],:]
        e1 = np.sqrt((v1[0]-v2[0])**2+(v1[1]-v2[1])**2)
        e2 = np.sqrt((v1[0]-v3[0])**2+(v1[1]-v3[1])**2)
        e3 = np.sqrt((v3[0]-v2[0])**2+(v3[1]-v2[1])**2)
        H[i] = np.max([e1,e2,e3])
    return H.flatten()

def orientcc(v,t):
# =============================================================================
#     # orient the mesh counterclockwise
# =============================================================================
    ntri = len(t[:,0])
    for i in range(ntri):
        v1 = v[t[i,0],:]
        v2 = v[t[i,1],:]
        v3 = v[t[i,2],:] 
        if triarea(v1,v2,v3) < 0:
            a = t[i,1]
            b = t[i,2]
            t[i,1] = b
            t[i,2] = a
    return t            

 
def refine(v0, t0):
# =============================================================================
#    # uniformly refine mesh   
# =============================================================================
    nvert = v0.shape[0]
    ntri =  t0.shape[0]
    v = v0
    t = []    
    for i in range(ntri):        
        J = t0[i,:]
        x0 = v0[J[0],0]; y0 = v0[J[0],1]
        x1 = v0[J[1],0]; y1 = v0[J[1],1]
        x2 = v0[J[2],0]; y2 = v0[J[2],1]
        x01 = 0.5*(x0 + x1); y01 = 0.5*(y0 + y1)
        x02 = 0.5*(x0 + x2); y02 = 0.5*(y0 + y2)
        x12 = 0.5*(x1 + x2); y12 = 0.5*(y1 + y2)        
        vertx = v[:,0]; verty = v[:,1]        
        p01 = np.argwhere((vertx==x01)&(verty==y01))
        p02 = np.argwhere((vertx==x02)&(verty==y02))
        p12 = np.argwhere((vertx==x12)&(verty==y12))
        if p01.size == 0:
            nvert = nvert + 1
            p01 = nvert-1
            nc0 = np.array([x01,y01])
            v = np.append(v,nc0).reshape(nvert,2)
        else:
            p01 = int(p01.flat[0])
        if p02.size == 0:
            nvert = nvert + 1
            p02 = nvert-1
            nc1 = np.array([x02,y02])
            v = np.append(v,nc1).reshape(nvert,2)
        else:
            p02 = int(p02.flat[0])
        if p12.size == 0:
            nvert = nvert + 1
            p12 = nvert-1
            nc2 = np.array([x12,y12])
            v = np.append(v,nc2).reshape(nvert,2)
        else:
            p12 = int(p12.flat[0])
        av = np.array([J[0], p01, p02], dtype = int)
        bv = np.array([p01,  p02, p12], dtype = int)
        cv = np.array([J[1], p01, p12], dtype = int)
        dv = np.array([J[2], p02, p12], dtype = int)        
        t = np.append(t,av); t = t.reshape(int(len(t)/3),3)
        t = np.append(t,bv); t = t.reshape(int(len(t)/3),3)
        t = np.append(t,cv); t = t.reshape(int(len(t)/3),3)
        t = np.append(t,dv); t = t.reshape(int(len(t)/3),3)
        t = np.array(t, dtype=int)
        t = orientcc(v, t)
    return v, t

def nrefine(n,v,t):
# =============================================================================
#     # refine mesh uniformly n times
# =============================================================================
    for j in np.arange(n):
        v,t = refine(v,t)
    return v,t


def meshdata(v, t):
# =============================================================================
#    # returns triangulation relations
#    # e: edges
#    # te: triangles to edges sparse matrix 
#    # tv: triangles to vertices sparse matrix
#    # b: boundary nodes arranged in cc order
#    # ev: edges to vertices sparse matrix
# =============================================================================
    m = t.shape[0] 
    n = v.shape[0]
    ne = m+n-1   # Euler's relation for number of edges assume no holes
    edges = []
    idx_tei = []  # initialize sparse te matrix
    idx_tej = []
    val_te = []
    nedges = 0
    for i in range(m):
        ti = t[i,:]
        for j in range(3):
            e1 = min(ti[j],ti[np.mod(j+1,3)])
            e2 = max(ti[j],ti[np.mod(j+1,3)])
            edg = np.array([e1, e2], dtype=int)
            if  len(edges) != 0:
                f1 = np.argwhere(edg[0]==edges[:,0])
                f2 = np.argwhere(edg[1]==edges[:,1])
                edgnum = np.intersect1d(f1,f2)
                if len(edgnum)==1:
                    idx_tei = np.append(idx_tei,i).astype(int)
                    idx_tej = np.append(idx_tej,edgnum[0]).astype(int)
                    val_te = np.append(val_te,1).astype(int)                   
            else:
                edgnum = []
            if len(edgnum) == 0:
                edges = np.append(edges,edg)
                ne = int(len(edges)/2)
                edges = edges.reshape(ne,2); 
                nedges = nedges + 1
                edgnum = nedges 
                idx_tei = np.append(idx_tei,i).astype(int)
                idx_tej = np.append(idx_tej,edgnum-1).astype(int)
                val_te = np.append(val_te,1).astype(int)
    nvert = len(v[:,0])
    idx_tvi = []  # initialize sparse tv matrix
    idx_tvj = []
    val_tv = []
    for i in range(m):
        idx_tvi = np.append(idx_tvi,[i,i,i]).astype(int)
        idx_tvj = np.append(idx_tvj,t[i,:]).astype(int)
        val_tv = np.append(val_tv,[1,1,1]).astype(int)
    idx_evi = []  # initialize sparse ev matrix
    idx_evj = []
    val_ev = []
    for i in range(nedges):
        edges = np.array(edges, dtype=int)
        idx_evi = np.append(idx_evi,[i,i]).astype(int)
        idx_evj = np.append(idx_evj,edges[i,:]).astype(int)
        val_ev = np.append(val_ev,[1,1]).astype(int)
    e = edges
    te = coo_matrix((val_te,(idx_tei,idx_tej)),shape=(m,ne)).tocsr() # sparse te matrix
    tv = coo_matrix((val_tv,(idx_tvi,idx_tvj)),shape=(m,nvert)).tocsr() # sparse tv matrix
    ev = coo_matrix((val_ev,(idx_evi,idx_evj)),shape=(nedges,nvert)).tocsr() # sparse ev matrix
    b = findbdn(t,v,e,te,ev) # boundary nodes in counter-clockwise order
    return e, b, te, tv, ev


def findbdn(t,v,e,te,ev):
# =============================================================================
#     # return a set b of boundary nodes
#     # boundary nodes are arranged in cc order
# =============================================================================
    be = np.argwhere(sum(te)==1)[:,1]
    be = be.flatten()
    b = []
    while (len(be)!=0):
        enum = be[0]
        bee = e[be,:]
        ed = e[enum,:]
        tet = np.argwhere(te[:,enum])
        tet = tet.flatten()[0]
        tri = t[tet,:]
        tri = tri.flatten()
        v1 = v[tri[0],:]
        v2 = v[tri[1],:]
        v3 = v[tri[2],:]
        p = (v1+v2+v3)/3
        vl = v[ed[0],:]
        vr = v[ed[1],:]
        if triarea(vl, vr, p)>0:
            b1 = ed
        else:
            b1 = np.array([ed[1],ed[0]])
        l1 = np.array([0])
        loop = 0
        i = 0
        while (loop==0):
            u1 = b1[i]
            u2 = b1[i+1]
            next_ed = np.argwhere(((u2==bee[:,0])&(u1!=bee[:,1]))| \
                                  ((u2==bee[:,1])&(u1!= bee[:,0])))
            next_ed = next_ed.flatten()
            if (bee[next_ed,0]==u2):
                u3 = bee[next_ed,1]
            else:
                u3 = bee[next_ed,0]
            if u3 != b1[0]:
                b1 = np.append(b1,u3)
                l1 = np.append(l1,next_ed)
                i = i+1
            else:
                l1 = np.append(l1,next_ed)
                l1 = l1.flatten()                
                loop = 1
        be = np.setdiff1d(be,be[l1])
        b = np.append(b,b1)
        b = np.array(b,dtype=int)
    return b

def tictocgen():
# =============================================================================
#     # equivalent to MATLAB timing [tic; dosomething; toc]
# =============================================================================
    ti = 0
    tf = time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti

tictoc = tictocgen()

def toc(tempbool=True):
    tempTimeInterval = next(tictoc)
    if tempbool:
        print("Elapsed time: %.4f seconds \n" %tempTimeInterval)

def tic():
    toc(False)
# =============================================================================
#     # tic(); dosomething; toc()
# =============================================================================

def inpoly(x,y,polygon):
# =============================================================================
#     # find points that lie inside or on boundary of polygon 
#     # x,y are n by 1 vectors
#     # polygon is an m by 2 array of boundary vertices of polygon
#     # returns x[inside, y[inside] coords of inside points
#     # returns vector of positions of inside points
# =============================================================================
    points = np.hstack((x,y))
    path = mpltPath.Path(polygon)
    inside = path.contains_points(points,radius=1e-6)
    return x[inside], y[inside], inside

def vgrid(b,v,n):
# =============================================================================
#     # construct grid to remove points outside the domain
#     # for rectangular domains this is just meshgrid
#     # returns x,y n**2 by 1 coord vectors
# =============================================================================
    polygon = v[b,:]
    xmax = max(v[:,0])
    xmin = min(v[:,0])
    ymax = max(v[:,1])
    ymin = min(v[:,1])
    xi = np.linspace(xmin,xmax,n)
    yi = np.linspace(ymin,ymax,n)
    xx,yy = np.meshgrid(xi,yi)
    xx = xx.flatten()[:,np.newaxis]
    yy = yy.flatten()[:,np.newaxis]
    x,y,inside = inpoly(xx,yy,polygon)
    return x,y

def meshnodes(v,t,d):
    from barynets import indices
# =============================================================================
#     # returns x,y: coordinate vectors of the domain points of triangulation [v,t]
#     # returns tri: a vector of triangle numbers 
# =============================================================================
    vt = v[t[:,:],:]
    I,J,K = indices(d)
    p = np.multiply.outer(vt[:,0,:],I)+\
        np.multiply.outer(vt[:,1,:],J)+\
        np.multiply.outer(vt[:,2,:],K)
    p = p/d
    x = p[:,0,:].flatten()
    y = p[:,1,:].flatten()
    m = int(0.5*(d+1)*(d+2))
    tr = np.floor(np.arange(x.size)/m).astype(int)
    return x,y,tr

def normals(v,t):
    
# =============================================================================
#     # computes unit outward normals associated with each triangle
#     # outputs: array of normal vectors N
#     #          array of lengths of the edges
# =============================================================================
    t = orientcc(v,t)    # orient mesh triangles counterclockwise
    v1 = v[t[:,2],:]-v[t[:,1],:] # directed edge vectors
    v2 = v[t[:,0],:]-v[t[:,2],:]
    v3 = v[t[:,1],:]-v[t[:,0],:]
    a1 = la.norm(v1,axis=1) # lengths of edges opposite node 1/all triangles
    a2 = la.norm(v2,axis=1) # lengths of edges opposite node 2/all triangles
    a3 = la.norm(v3,axis=1) # lengths of edges opposite node 3/all triangles
    m = a1.size
    a1 = a1.reshape(m,1)
    a2 = a2.reshape(m,1)
    a3 = a3.reshape(m,1)
    L = np.hstack((a1,a2,a3)) # matrix of edge lengths 
    u = np.ones((2,1))
    l1 = np.multiply.outer(a1,u).reshape(m,2) # extend the edge length vector
    l2 = np.multiply.outer(a2,u).reshape(m,2) # extension equalizes dimensions
    l3 = np.multiply.outer(a3,u).reshape(m,2) # of the edge vectors & lengths
    w1 = v1/l1
    w2 = v2/l2
    w3 = v3/l3
    z = np.dstack((w1,w2,w3)) # stack the unit directed edge vectors
    rot = np.array([[0.,1.],[-1.,0.]]).reshape(1,2,2)   # rotation matrix
    N = np.tensordot(z,rot,axes=([1],[2])).reshape(m,3,2) # array of normals
    return N,L    



def splot(x,y,z,*args):
# =============================================================================
#     # surface plot of z over meshgrid points x,y
#     # if args not empty, also plots obstacle
# =============================================================================
    if x.shape[1]==1:
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x,y,z,linewidth=0.2,antialiased=False,color='b')
    if len(args)==1:
        psi = args[0].flatten()
        ax.plot_trisurf(x,y,psi,linewidth=0.2,antialiased=True,color='m')
    plt.show()


def activeplot(i,b,v,t,d):
# =============================================================================
#     # plot the active nodes
# =============================================================================
    plt.axes(aspect='equal')
    x = v[b,0]; x = np.append(x,x[0])
    y = v[b,1]; y = np.append(y,y[0])
    plt.plot(x,y,'m-')
    x,y,tri = meshnodes(v,t,d)
    plt.plot(x[i],y[i],'r.')
    plt.title('Active nodes')

    
def mayaviplot(x,y,z,*args):
    from mayavi import mlab
# =============================================================================
#     # plots a 3d surface using mayavi
# =============================================================================
    n = args[0]
    x = x.reshape(n,n)
    y = y.reshape(n,n)
    z = z.reshape(n,n)
    mlab.clf()
    mlab.surf(x,y,z,warp_scale='auto',colormap='jet')
    u = z[~np.isnan(z)]
    x1 = np.min(x)
    x2 = np.max(x) 
    y1 = np.min(y) 
    y2 = np.max(y) 
    z1 = np.min(u)
    z2 = np.max(u)
    if len(args)==2:
        psi = args[1]
        psi = psi.reshape(n,n)
        mlab.surf(x,y,psi,warp_scale='auto',colormap='jet')
    mlab.axes(nb_labels=5,ranges=(x1,x2,y1,y2,z1,z2))








    

   




                
            
            
            
    



        
    
        
    
    
            
  
  
                
                
                

    
    
    



        
            
        
        
        
        
        
        
    
