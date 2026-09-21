import numpy as np
from mayavi import mlab
mlab.options.offscreen = True
import dtorus_rbc as R
from flatquot import Quotient
def emb(u, phi):
    t = R.t_of(u); return np.array([R.rho(t)*np.cos(phi), R.rho(t)*np.sin(phi), R.zz(t)])
PHI0, PHI1 = 0.55, 1.75
def cutaway(out, nu=241, nphi=221, size=(1500,1000), tube=0.030):
    us=np.linspace(0,R.U,nu); ps=np.linspace(PHI1,PHI0+2*np.pi,nphi)
    Ug,Pg=np.meshgrid(us,ps,indexing='ij'); X=np.array([emb(u,p) for u,p in zip(Ug.ravel(),Pg.ravel())])
    tris=[]
    for i in range(nu-1):
        for j in range(nphi-1):
            a,b,c,d=i*nphi+j,(i+1)*nphi+j,i*nphi+j+1,(i+1)*nphi+j+1; tris+=[(a,b,d),(a,d,c)]
    fig=mlab.figure(bgcolor=(1,1,1),size=size)
    s=mlab.triangular_mesh(X[:,0],X[:,1],X[:,2],np.array(tris),color=(0.87,0.89,0.93)); s.actor.property.specular=0.12
    for p in (PHI0,PHI1):
        uu=np.linspace(0,R.U,240); P=np.array([emb(u,p) for u in uu]); ctr=P.mean(0)
        V=[];F=[];nr=26
        for k in range(nr+1):
            for q in range(240): V.append(ctr+(k/nr)*(P[q]-ctr))
        for k in range(nr):
            for q in range(240):
                a=k*240+q;b=k*240+(q+1)%240;c=(k+1)*240+q;d=(k+1)*240+(q+1)%240; F+=[(a,b,d),(a,d,c)]
        V=np.array(V); cap=mlab.triangular_mesh(V[:,0],V[:,1],V[:,2],np.array(F),color=(0.72,0.74,0.79)); cap.actor.property.specular=0.04
        pts=np.array([emb(u,p) for u in np.linspace(0,R.U,400)]); w=np.array([R.rho(R.t_of(u))**2 for u in np.linspace(0,R.U,400)])
        l=mlab.plot3d(pts[:,0],pts[:,1],pts[:,2],w,colormap='RdBu',tube_radius=tube,vmin=w.min(),vmax=w.max()); l.module_manager.scalar_lut_manager.reverse_lut=True
    for p in np.linspace(PHI1,PHI0+2*np.pi,13)[1:-1]:
        pts=np.array([emb(u,p) for u in np.linspace(0,R.U,200)]); mlab.plot3d(pts[:,0],pts[:,1],pts[:,2],color=(0.25,0.28,0.42),tube_radius=0.010)
    mlab.view(azimuth=66,elevation=60,distance='auto'); fig.scene.camera.zoom(1.6); fig.scene.parallel_projection=True
    mlab.savefig(out,size=size); mlab.close(fig); print(out,flush=True)
if __name__=='__main__':
    cutaway('/home/claude/work/hyp/paper/p_rbctorus_cut.png')
    Q=Quotient('torus',8,8,6,1,a=R.U,b=2*np.pi); vals,_,Z=Q.eigen(k=6,w=R.weight,q=10)
    K,M,F,c1=Q.assemble(w=R.weight,q=10); print('dim',Z.shape[1],'area',c1@(M@c1),'mu1..3',vals[1:4])
