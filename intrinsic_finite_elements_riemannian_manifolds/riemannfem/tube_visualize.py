"""Render ACTUAL computed nodal 3D FEM mode geometry, not synthetic imagery."""
from __future__ import annotations
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import ScalarMappable
from pathlib import Path


def _isopoly(mesh,geom,values,level):
    """Marching tetrahedra: piecewise linear FE isosurface on the volume mesh."""
    edges=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
    tris=[]
    for t,coords in zip(mesh.tets,mesh.local_coords):
        vals=values[t]
        pts=[]
        for i,j in edges:
            vi,vj=vals[i]-level,vals[j]-level
            if (vi<0 and vj>0) or (vj<0 and vi>0):
                w=vi/(vi-vj)
                pts.append(coords[i]+w*(coords[j]-coords[i]))
        if len(pts)<3:continue
        pts=np.asarray(pts)
        if len(pts)>3:
            c=pts.mean(axis=0)
            _,_,v=np.linalg.svd(pts-c,full_matrices=False)
            z=(pts-c)@v[:2].T
            order=np.argsort(np.arctan2(z[:,1],z[:,0]))
            pts=pts[order]
        pp=geom.physical(pts[:,0],pts[:,1],pts[:,2])
        for j in range(1,len(pp)-1):tris.append([pp[0],pp[j],pp[j+1]])
    return np.asarray(tris,float)


def render_mode(mesh,geom,values,lam,path):
    """Two actual FE isosurfaces ±0.50*||u||∞, plus faint volume boundary."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    norm=np.max(np.abs(values));val=values/max(norm,1e-300)
    fig=plt.figure(figsize=(10,6.4),dpi=165)
    ax=fig.add_subplot(111,projection='3d',computed_zorder=False)
    for level,color in ((-.45,'#1b5c96'),(.45,'#b02f3d')):
        tris=_isopoly(mesh,geom,val,level)
        if len(tris):
            col=Poly3DCollection(tris,facecolors=color,edgecolors='none',
                                 alpha=.83,zsort='average')
            ax.add_collection3d(col)
    face=mesh.faces_boundary
    xyz=geom.physical(mesh.coords[:,0],mesh.coords[:,1],mesh.coords[:,2])
    surf=Poly3DCollection(xyz[face],facecolors=(.73,.75,.80,.045),
                          edgecolors=(.32,.36,.44,.045),linewidths=.12)
    ax.add_collection3d(surf)
    # use body bounding box to avoid matplotlib aspect distortions
    mins=xyz.min(axis=0);maxs=xyz.max(axis=0)
    extent=np.maximum(maxs-mins,1e-6)
    ax.set(xlim=(mins[0]-.07*extent[0],maxs[0]+.07*extent[0]),
           ylim=(mins[1]-.07*extent[1],maxs[1]+.07*extent[1]),
           zlim=(mins[2]-.07*extent[2],maxs[2]+.07*extent[2]))
    ax.set_box_aspect(extent,zoom=1.18)
    ax.view_init(elev=37 if geom.name.startswith("round") else 30,azim=-58)
    ax.set_axis_off()
    ax.set_title(f'{geom.name.replace("_"," ")} | volume mode  λ={lam:.5f}',fontsize=13,pad=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#b02f3d',label='+0.45 max|u|'),
                       Patch(color='#1b5c96',label='−0.45 max|u|')],
              loc='lower center', bbox_to_anchor=(.5,-.03),frameon=False,ncol=2)
    fig.tight_layout()
    fig.savefig(path,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def render_disk_slices(mesh,geom,values,lam,path):
    """Exact traces of P1 volume FE function on s=const triangulated disks."""
    import matplotlib.tri as mtri
    path=Path(path)
    ns=mesh.ns;nd=mesh.ndisk;disk=mesh.disk_xy
    tri=mtri.Triangulation(disk[:,0],disk[:,1],triangles=mesh.disk_tri)
    mx=max(np.max(np.abs(values)),1e-30)
    fig,axs=plt.subplots(1,4,figsize=(13,3.7),constrained_layout=True)
    for a,j in zip(axs,[0,ns//4,ns//2,3*ns//4]):
        z=values[j*nd:(j+1)*nd]/mx
        pic=a.tricontourf(tri,z,levels=np.linspace(-1,1,33),cmap='RdBu_r',extend='both')
        a.triplot(tri,color='k',linewidth=.25,alpha=.18)
        a.set_aspect('equal');a.set_xticks([]);a.set_yticks([])
        a.set_title(f's={2*np.pi*j/ns:.2f}')
    cb=fig.colorbar(pic,ax=axs,shrink=.8,pad=.016)
    cb.set_label('u / max|u|')
    fig.suptitle(f'{geom.name.replace("_"," ")}: internal cross-sectional traces, λ={lam:.6f}',fontsize=12)
    fig.savefig(path,dpi=180,facecolor='white')
    plt.close(fig)
