"""Sparse entity-moment assembly for high-order trimmed differential forms.

Each local DOF is an integral of a tangential trace wedged with a polynomial
complementary form on a subsimplex.  Facet pairings identify vertex-, edge-,
face-, and volume-entity *instances*, not just vertex tuples.  A sparse map
injects independent global entity moments into local moments.  This module
never forms a global dense constraint matrix or its nullspace.

The cell maps need supply only their intrinsic reference-coordinate metric.
The sparse assembly does not ask for an ambient embedding.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import comb
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from .forms import (TrimmedPolynomialFormSpace, monomial_indices, wedge_indices,
                    polynomial_pullback, simplex_vertex_permutation_affine)
from .reference import simplex_duffy


def _wedge_sign(i, j, d):
    seq = tuple(i) + tuple(j)
    if len(set(seq)) != d or len(seq) != d:
        return 0
    return -1 if sum(seq[a] > seq[b] for a in range(d) for b in range(a+1,d)) % 2 else 1


def _reference_vertices(d):
    if d == 0:
        return np.zeros((1,0))
    return np.vstack((np.zeros(d), np.eye(d)))


def _embed_entity(vertices, d):
    V = _reference_vertices(3)
    if d == 0:
        return np.zeros((3,0)), V[vertices[0]]
    b = V[vertices[0]]
    A = np.column_stack([V[v]-b for v in vertices[1:]])
    return A,b


def _test_keys(d,r,k):
    if d < k:
        return []
    s = r+k-d-1
    if s < 0:
        return []
    if d == 0:
        return [((),())] if k == 0 else []
    return [(alpha,I) for alpha in monomial_indices(d,s)
            for I in wedge_indices(d,d-k)]


@lru_cache(maxsize=None)
def reference_entity_dofs(r,k,q=None):
    """Reference tetrahedron local DOF matrix E: moments = E @ coefficients.

    Rows are indexed by pairs (local subsimplex vertex tuple, moment test).
    Integrates with a high-enough Gauss rule for polynomial integrands.
    """
    V = TrimmedPolynomialFormSpace(3,r,k)
    q = int(q or max(5,r+3))
    entries=[]
    rows=[]
    for d in range(k,4):
        keys=_test_keys(d,r,k)
        if not keys:
            continue
        for vertex_tuple in combinations(range(4),d+1):
            A,b=_embed_entity(vertex_tuple,d)
            start=len(rows)
            if d==0:
                rows.append(V.values(b)[:,0].copy())
            else:
                lambdas,weights=simplex_duffy(d,q)
                wedges_k=wedge_indices(d,k)
                ambient_wedges=wedge_indices(3,k)
                minors=np.array([
                    [np.linalg.det(A[np.ix_(I,J)]) if k else 1.
                     for J in wedges_k]
                    for I in ambient_wedges
                ])
                signs=np.array([[_wedge_sign(I,J,d) for J in wedge_indices(d,d-k)]
                                for I in wedges_k])
                block=np.zeros((len(keys),V.nloc))
                for lam,w in zip(lambdas,weights):
                    y=lam[1:]
                    x=A@y+b
                    tangent=V.values(x)@minors
                    tests=np.array([np.prod([y[j]**a for j,a in enumerate(alpha)])
                                    for alpha,_ in keys],float)
                    Jkeys=wedge_indices(d,d-k)
                    Jidx={J:t for t,J in enumerate(Jkeys)}
                    for l,(_,J) in enumerate(keys):
                        block[l] += w*tests[l]*(tangent@signs[:,Jidx[J]])
                rows.extend(block)
            if len(rows)-start != len(keys):
                raise RuntimeError('entity moment size mismatch')
            entries.append((vertex_tuple,tuple(keys),start,len(rows)))
    E=np.array(rows)
    if E.shape!=(V.nloc,V.nloc):
        raise RuntimeError(f'wrong count in moment DOFs {(r,k)}: {E.shape}')
    singular=la.svdvals(E)
    if singular[-1] <= 1e-12*singular[0]:
        raise RuntimeError(f'unisolvence failure {(r,k)}, condition {singular[0]/singular[-1]:.3g}')
    E_inv=la.solve(E,np.eye(V.nloc))
    return E,E_inv,tuple(entries),float(singular[0]/singular[-1])


@lru_cache(maxsize=None)
def entity_moment_transform(d,r,k,permutation):
    """Map canonical entity moments to local entity moments.

    The tuple permutation[j] is the *local* position of canonical vertex j.
    A is the affine map canonical -> local.  Pullback of complementary test
    forms, with orientation sign, acts on moment DOFs of the same entity.
    """
    keys=_test_keys(d,r,k)
    m=len(keys)
    if not m:
        return np.zeros((0,0))
    if d==0:
        return np.ones((1,1))
    A,b=simplex_vertex_permutation_affine(permutation)
    orient=int(round(np.linalg.det(A)))
    if abs(orient)!=1:
        raise ValueError('entity map is not a vertex permutation')
    idx={q:i for i,q in enumerate(keys)}
    T=np.zeros((m,m))
    for j,(alpha,I) in enumerate(keys):
        poly=polynomial_pullback(alpha,A,b)
        for J in wedge_indices(d,d-k):
            det=np.linalg.det(A[np.ix_(I,J)]) if d>k else 1.
            if abs(det)<1e-13:
                continue
            for beta,value in poly.items():
                if abs(value)<1e-13:
                    continue
                t=(beta,J)
                if t not in idx:
                    raise RuntimeError(f'test pullback left polynomial space {t}')
                T[j,idx[t]] += orient*det*value
    if np.linalg.matrix_rank(T)!=m:
        raise RuntimeError('singular entity moment transformation')
    return T


@dataclass(frozen=True)
class EntityInstance:
    cell: int
    vertices: tuple[int,...]


class EntityOrbits:
    """Quotient cells using full entity incidences and vertex transports."""

    def __init__(self,ncells,pairings):
        self.ncells=int(ncells)
        self.entities=[EntityInstance(c,vs) for d in range(4)
                       for c in range(ncells)
                       for vs in combinations(range(4),d+1)]
        self.index={e:i for i,e in enumerate(self.entities)}
        self.edges=defaultdict(list)
        self.parent=list(range(len(self.entities)))

        def find(i):
            while self.parent[i]!=i:
                self.parent[i]=self.parent[self.parent[i]]
                i=self.parent[i]
            return i
        def union(a,b):
            a,b=find(a),find(b)
            if a!=b:self.parent[b]=a

        for pf in pairings:
            plus=[j for j in range(4) if j!=pf.facet_plus]
            minus=[j for j in range(4) if j!=pf.facet_minus]
            if sorted(pf.vertex_permutation)!=[0,1,2]:
                raise ValueError('invalid facet vertex permutation')
            fwd={plus[j]:minus[pf.vertex_permutation[j]] for j in range(3)}
            for d in range(3):
                for subs in combinations(plus,d+1):
                    src=EntityInstance(pf.cell_plus,tuple(sorted(subs)))
                    dst=EntityInstance(pf.cell_minus,tuple(sorted(fwd[v] for v in subs)))
                    a,b=self.index[src],self.index[dst]
                    map_src_dst=tuple(dst.vertices.index(fwd[v]) for v in src.vertices)
                    inv=[None]*(d+1)
                    for q,p in enumerate(map_src_dst):inv[p]=q
                    self.edges[a].append((b,map_src_dst))
                    self.edges[b].append((a,tuple(inv)))
                    union(a,b)

        # Group quotient entities separately by their full cell/entity orbits.
        groups=defaultdict(list)
        for i,e in enumerate(self.entities):
            groups[find(i)].append(i)
        self.orbits=[]
        self.info={}
        for ids in sorted(groups.values(),key=lambda ids:min(ids)):
            root=min(ids)
            root_entity=self.entities[root]
            degree=len(root_entity.vertices)-1
            labels={root:tuple(range(degree+1))}
            queue=deque([root])
            while queue:
                u=queue.popleft()
                for v,mp in self.edges[u]:
                    label=[None]*(degree+1)
                    for i,j in enumerate(mp):
                        label[j]=labels[u][i]
                    label=tuple(label)
                    if v not in labels:
                        labels[v]=label
                        queue.append(v)
                    elif labels[v]!=label:
                        raise ValueError(f'nontrivial stabilizer / inconsistent vertex transport '
                                         f'on quotient {degree}-entity: {self.entities[v]}')
            if set(labels)!=set(ids):
                raise RuntimeError('disconnected entity orbit')
            orbit=len(self.orbits)
            self.orbits.append(tuple(ids))
            for i in ids:
                self.info[i]=(orbit,labels[i])

        self.orbits_by_dim={d:[] for d in range(4)}
        for j,ids in enumerate(self.orbits):
            d=len(self.entities[ids[0]].vertices)-1
            self.orbits_by_dim[d].append(j)

    def counts(self):
        return tuple(len(self.orbits_by_dim[d]) for d in range(4))

    def identification(self,cell,vertices):
        return self.info[self.index[EntityInstance(cell,tuple(vertices))]]


class SparseEntityComplex:
    """Assemble P_r^- Lambda^k via sparse entity-moment injections P_k.

    Global moment numbering follows quotient entity orbits.  The method
    stores only sparse injections, sparse differential matrices, and small
    dense reference-cell changes of basis.  It does not form a global dense
    matrix/nullspace.  The sparse complexes can be reused with any intrinsic
    cell metrics without changing the topological assembly.
    """

    def __init__(self,ncells,r,pairings,check=True,reference_basis='algebraic'):
        if reference_basis not in ('algebraic','bernstein'):
            raise ValueError('reference_basis must be algebraic or bernstein')
        self.reference_basis=reference_basis
        self.ncells=int(ncells)
        self.r=int(r)
        self.pairings=list(pairings)
        self.orbits=EntityOrbits(ncells,pairings)
        self.spaces=[]
        self.E=[]
        self.Einv=[]
        self.entity_info=[]
        self.P=[]
        self.R=[]
        self.conds=[]
        self.num_global=[]
        for k in range(4):
            V=TrimmedPolynomialFormSpace(3,r,k)
            if self.reference_basis=='bernstein':
                from .bernstein_forms import reference_entity_dofs_bernstein
                E,Ein,entries,condition=reference_entity_dofs_bernstein(r,k)
            else:
                E,Ein,entries,condition=reference_entity_dofs(r,k)
            self.spaces.append(V)
            self.E.append(E)
            self.Einv.append(Ein)
            self.entity_info.append(entries)
            self.conds.append(condition)
            nloc=V.nloc
            sizes={d:len(_test_keys(d,r,k)) for d in range(4)}
            global_blocks={}
            cursor=0
            for orbit,ids in enumerate(self.orbits.orbits):
                d=len(self.orbits.entities[ids[0]].vertices)-1
                if sizes[d]:
                    global_blocks[orbit]=cursor
                    cursor+=sizes[d]
            self.num_global.append(cursor)
            rows=[];cols=[];data=[]
            restrict_rows=[];restrict_cols=[]
            for c in range(ncells):
                for verts,keys,start,end in entries:
                    orbit,labels=self.orbits.identification(c,verts)
                    offset=global_blocks[orbit]
                    # labels[local_vertex_position] = representative position
                    permutation=tuple(labels.index(j) for j in range(len(labels)))
                    T=entity_moment_transform(len(verts)-1,r,k,permutation)
                    for j in range(end-start):
                        for i,t in enumerate(T[j]):
                            if abs(t)>1e-13:
                                rows.append(c*nloc+start+j)
                                cols.append(offset+i)
                                data.append(float(t))
                    # The root instance has identity transform in its local
                    # ordering and gives an exact sparse left inverse R P=I.
                    rep=self.orbits.entities[self.orbits.orbits[orbit][0]]
                    if rep.cell==c and rep.vertices==verts:
                        for j in range(end-start):
                            restrict_rows.append(offset+j)
                            restrict_cols.append(c*nloc+start+j)
            P=sp.coo_matrix((data,(rows,cols)),shape=(ncells*nloc,cursor)).tocsr()
            R=sp.coo_matrix((np.ones(len(restrict_rows)),
                (restrict_rows,restrict_cols)),shape=(cursor,ncells*nloc)).tocsr()
            ident=R@P-sp.eye(cursor,format='csr')
            if ident.nnz and np.max(np.abs(ident.data))>1e-11:
                raise RuntimeError(f'R_{k} P_{k} is not identity')
            self.P.append(P)
            self.R.append(R)

        self.D=[]
        self.defects=[]
        for k in range(3):
            if self.reference_basis=='bernstein':
                from .bernstein_forms import reference_differential_bernstein
                Dloc=reference_differential_bernstein(3,r,k)
            else:
                Dloc=self.spaces[k].exterior_derivative_matrix()
            Dref=self.E[k+1]@Dloc@self.Einv[k]
            Dbr=sp.kron(sp.eye(ncells,format='csr'),sp.csr_matrix(Dref),format='csr')
            Dgl=(self.R[k+1]@Dbr@self.P[k]).tocsr()
            Dgl.eliminate_zeros()
            self.D.append(Dgl)
            if check:
                Z=Dbr@self.P[k]-self.P[k+1]@Dgl
                self.defects.append(float(np.max(np.abs(Z.data))) if Z.nnz else 0.)
            else:
                self.defects.append(float('nan'))

    @property
    def dimensions(self):return tuple(self.num_global)

    @property
    def d2_defects(self):
        return tuple(float(np.max(np.abs(Z.data))) if Z.nnz else 0.
                     for Z in (self.D[1]@self.D[0],self.D[2]@self.D[1]))

    def betti_numbers(self,rank_tol=1e-8):
        # Diagnostic only: dense rank calculation on small complexes.
        ranks=[np.linalg.matrix_rank(D.toarray(),tol=rank_tol) for D in self.D]
        return tuple(int(self.num_global[k]-(ranks[k-1] if k>0 else 0)
                     -(ranks[k] if k<3 else 0)) for k in range(4))

    def assemble_mass(self,k,local_masses,local_basis='algebraic'):
        """Assemble metric Hodge mass from local matrices in a named basis.

        The intrinsic metric is unchanged by a local reference basis change.
        Default keeps compatibility with existing element providers.
        """
        if local_basis not in ('algebraic','bernstein'):
            raise ValueError('local_basis must be algebraic or bernstein')
        if len(local_masses)!=self.ncells:raise ValueError('incorrect cell count')
        Ein=self.Einv[k]
        if local_basis != self.reference_basis:
            from .bernstein_forms import bernstein_change
            C,R,_=bernstein_change(3,self.r,k)
            if local_basis=='algebraic':
                matrices=[C.T@M@C for M in local_masses]
            else:
                matrices=[R.T@M@R for M in local_masses]
        else:
            matrices=local_masses
        blocks=[sp.csr_matrix(Ein.T @ M @ Ein) for M in matrices]
        Mbr=sp.block_diag(blocks,format='csr')
        P=self.P[k]
        out=(P.T@Mbr@P).tocsr()
        return 0.5*(out+out.T)

    def audit_against_dense_constraints(self,k):
        """For small test meshes, check that every sparse basis function
        obeys the original algebraic trace constraint system."""
        from .global_complex import trace_constraint_matrix
        C=trace_constraint_matrix(self.ncells,self.r,k,self.pairings)
        if self.reference_basis=='bernstein':
            from .bernstein_forms import bernstein_change
            Change,_,_=bernstein_change(3,self.r,k)
            back=Change@self.Einv[k]
        else:
            back=self.Einv[k]
        X=sp.kron(sp.eye(self.ncells,format='csr'),sp.csr_matrix(back))@self.P[k]
        if not C.size:return 0.
        defect=C@X.toarray()
        return float(np.max(np.abs(defect)))
