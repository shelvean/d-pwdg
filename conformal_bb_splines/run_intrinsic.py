"""Table 'tab:intrinsic': Bolza from faces and hyperbolic edge lengths only,
each triangle placed on its own, transitions computed per edge."""
import numpy as np, json, warnings; warnings.filterwarnings("ignore")
from scipy.sparse.linalg import eigsh
from hypmesh import octagon_mesh, build_dofs
from hypgeom import dist
import intrinsic as it
LAM1 = 3.8388872588421995
def bolza_intrinsic(level):
    tris, V0, gens = octagon_mesh(level)
    dof1, _ = build_dofs(tris, V0, gens, 1); faces = dof1.reshape(len(tris), 3)
    dof2, _ = build_dofs(tris, V0, gens, 2); mid = [4, 2, 1]
    key_of = lambda f, m: int(dof2[f*6+mid[m]])
    l = np.zeros((len(tris), 3))
    for t, T in enumerate(tris):
        for a in range(3):
            b, c = (a+1) % 3, (a+2) % 3; l[t, a] = dist(T[:, b], T[:, c])
    return faces, l, key_of
rows = []
for level, nq in [(1, 24), (2, 16)]:
    faces, l, key_of = bolza_intrinsic(level)
    V = it.place(l); edges = it.edge_table(faces, key_of); G = it.transitions(faces, V, edges)
    res = 0.0
    for e, (f, m, f2, m2) in enumerate(edges):
        g1 = faces[f][(m+1) % 3]; g2 = faces[f][(m+2) % 3]; j1, j2 = it._slots(faces[f2], g1, g2, m2)
        res = max(res, np.max(np.abs(G[e]@V[f2][:, j1]-V[f][:, (m+1) % 3])), np.max(np.abs(G[e]@V[f2][:, j2]-V[f][:, (m+2) % 3])))
    for d in (4, 6):
        dof, ndof = it.dof_map(faces, edges, d)
        Mg, Kg, area = it.assemble(faces, V, dof, ndof, d, nq)
        vals = np.sort(eigsh(Kg.tocsc(), k=6, M=Mg.tocsc(), sigma=-1.0, which='LM', return_eigenvectors=False).real)
        rows.append(dict(level=level, d=d, ndof=int(ndof), res=float(res), area=float(area), mu0=float(vals[0]), mu1=float(vals[1]), rel=float(abs(vals[1]-LAM1)/LAM1)))
        print(rows[-1], flush=True)
json.dump(rows, open('bolza_intrinsic.json', 'w'), indent=1)
