"""Reproduce intrinsic 3D volume spectra and interior isosurface figures.

Examples are *representative* geometries; the supplied reference images do not
specify enough metric/mesh information to reconstruct their exact volumes.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from riemannfem.volume_tubes import (RoundTorus,TwistedRoundTorus,TrefoilTube,swept_mesh,
       local_matrices,solve_modes,nodal_values)


def run(kind='both',ns=12,nr=2,nt=12,radius=.35,p=1,q=3,which='both',npositive=10,plot=True,out='results_volume'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    mesh=swept_mesh(ns=ns,nr=nr,nt=nt,radius=radius)
    geomlist={'round':[RoundTorus()], 'twisted':[TwistedRoundTorus()], 'trefoil':[TrefoilTube(radius)], 'both':[RoundTorus(),TwistedRoundTorus()], 'all':[RoundTorus(),TwistedRoundTorus(),TrefoilTube(radius)]}[kind]
    report=[]
    from riemannfem.tube_visualize import render_mode, render_disk_slices
    for geom in geomlist:
        start=time.perf_counter()
        print(f'Assembling {geom.name}: {len(mesh.tets)} tets, p={p}, q={q}',flush=True)
        A=local_matrices(mesh,geom,p=p,q=q)
        row={'geometry':geom.name,'p':p,'q':q,'ns':ns,'nr':nr,'nt':nt,'radius':radius,
             'vertices':len(mesh.coords),'tetrahedra':len(mesh.tets),
             'dofs':A['ndof'],'boundary_dofs':int(np.sum(A['boundary'])),
             'volume_quadrature':A['volume'],
             'circle_centerline_length':geom.centerline_length(),
             'periodic':True}
        diskarea=float(sum(abs(np.linalg.det(np.column_stack((mesh.disk_xy[t[1]]-mesh.disk_xy[t[0]],mesh.disk_xy[t[2]]-mesh.disk_xy[t[0]]))))*.5 for t in mesh.disk_tri))
        row['mesh_exact_volume']=geom.centerline_length()*diskarea
        row['mesh_volume_relative_error']=abs(A['volume']/row['mesh_exact_volume']-1)
        row['boundary_triangle_count']=len(mesh.faces_boundary)
        print(' DOFs',A['ndof'],'volume error',row['mesh_volume_relative_error'],flush=True)
        for bc in (['neumann','dirichlet'] if which=='both' else [which]):
            t=time.perf_counter()
            sol=solve_modes(A,bc=bc,npositive=npositive)
            eig=sol['eigenvalues']
            row[bc]={'eigenvalues':[float(v) for v in eig],
                     'zero_eigenvalue':sol['zero_eigenvalue'],
                     'free_dofs':sol['kept'], 'seconds':time.perf_counter()-t}
            np.savez_compressed(out/f'{geom.name}_{bc}_p{p}.npz',
                eigenvalues=eig,eigenvectors=sol['eigenvectors'],
                local_to_global=A['ids'],vertices=mesh.coords,tetrahedra=mesh.tets,
                local_chart_coordinates=mesh.local_coords,
                boundary_faces=mesh.faces_boundary)
            print(' ',bc, np.round(eig,7),flush=True)
            if plot and p==1:
                for modeidx in [0, min(3,len(eig)-1), min(7,len(eig)-1)]:
                    vec=nodal_values(A,sol['eigenvectors'][:,modeidx]); lam=eig[modeidx]
                    stem=out/f'{geom.name}_{bc}_mode{modeidx+1:02d}'
                    render_mode(mesh,geom,vec,lam,stem.with_suffix('.png'))
                    if modeidx==0:render_disk_slices(mesh,geom,vec,lam,Path(str(stem)+'_slices.png'))
        row['total_seconds']=time.perf_counter()-start
        report.append(row)
    (out/'results.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--kind',choices=['round','twisted','trefoil','both','all'],default='both')
    ap.add_argument('--ns',type=int,default=12)
    ap.add_argument('--nr',type=int,default=2)
    ap.add_argument('--nt',type=int,default=12)
    ap.add_argument('--radius',type=float,default=.35)
    ap.add_argument('--p',type=int,default=1)
    ap.add_argument('--q',type=int,default=3)
    ap.add_argument('--npositive',type=int,default=10)
    ap.add_argument('--which',choices=['neumann','dirichlet','both'],default='both')
    ap.add_argument('--out',default='results_volume')
    ap.add_argument('--no-plot',action='store_true')
    aa=ap.parse_args()
    run(**{k:v for k,v in vars(aa).items() if k!='no_plot'},plot=not aa.no_plot)
