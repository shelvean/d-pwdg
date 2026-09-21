"""
Plane wave discontinuous Galerkin method on a square with a piecewise
constant wavenumber.

The scheme is the ultra weak variational formulation of Cessenat and
Despres in the discontinuous Galerkin form of Hiptmair, Moiola and Perugia,
as written in equation (3.2) of

    S. Kapita, P. Monk and T. Warburton, Residual-based adaptivity and PWDG
    methods for the Helmholtz equation, SIAM J. Sci. Comput. 37 (2015),
    A1525-A1553

with the UWVF flux parameters alpha = beta = delta = 1/2.  The boundary is
entirely Dirichlet, with data taken from the exact solution, which is the
setting of section 6.3 there.

Discrete space.  On element K the local space is spanned by p plane waves
with the local wavenumber kappa_K and equally spaced directions,

    v(x) = sum_l c_l exp(i kappa_K d_l . (x - x_K)),
    d_l  = (cos(2 pi l / p), sin(2 pi l / p)),

with x_K the centroid, subtracted only for conditioning.  Every basis
function solves the Helmholtz equation exactly on its element, so the
scheme has no volume term and the whole system is assembled on the mesh
skeleton.

Interior edge block.  For a trial function on element A with outward normal
nA and a test function on element B with outward normal nB, the four terms
of (3.2) contribute

    C[l,m] = -i kB (d_m.nB) / 2 - i kA (d_l.nB) / 2
             + i (beta / xi) kA kB (d_l.nA)(d_m.nB)
             + i xi alpha (nA.nB),

with xi the mean of the two local wavenumbers.  This reduces to the usual
constant-wavenumber block when kA = kB = xi.  The edge integral of the two
basis functions multiplies C entrywise.

Mesh.  A structured triangulation of (-1,1)^2 with an even number of cells
per direction, so that the line y = 0 is a mesh line and the wavenumber is
constant on every element.  Nothing below assumes the mesh is structured
except the constructor square_mesh itself.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from quadrature import gauss01

ALPHA = BETA = 0.5          # UWVF flux parameters


# ----------------------------------------------------------------------
# mesh
# ----------------------------------------------------------------------

def square_mesh(n, box=(-1.0, 1.0)):
    """
    Structured triangulation of box x box with n cells per direction.

    Returns vertices v of shape (nv, 2) and triangles t of shape (nt, 3).
    For a problem whose wavenumber jumps across y = 0, n must be even so
    that the interface is a mesh line; PWDG checks this.
    """
    xs = np.linspace(box[0], box[1], n + 1)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    v = np.stack([X.ravel(), Y.ravel()], axis=1)
    idx = lambda i, j: i * (n + 1) + j
    t = []
    for i in range(n):
        for j in range(n):
            a, b = idx(i, j), idx(i + 1, j)
            c, d = idx(i + 1, j + 1), idx(i, j + 1)
            t.append([a, b, c])
            t.append([a, c, d])
    return v, np.array(t, dtype=int)


def lshape_mesh(n, box=(-1.0, 1.0)):
    """
    Structured triangulation of the L-shaped domain

        box x box  minus  the quadrant  x > 0, y < 0,

    which for box = (-1, 1) is the domain of sections 6.1 and 6.2 of Kapita,
    Monk and Warburton (2015).  The reentrant corner sits at the origin and
    the interior angle is 3 pi / 2.

    Built by discarding the cells of the full square whose centroid lies in
    the removed quadrant, so n must be even for the cut to follow mesh
    lines.  Unused vertices are left in the vertex array; nothing downstream
    reads them.
    """
    if n % 2:
        raise ValueError('n must be even so that the reentrant corner is a '
                         'mesh vertex')
    v, t = square_mesh(n, box)
    mid = 0.5 * (box[0] + box[1])
    c = v[t].mean(axis=1)
    keep = ~((c[:, 0] > mid) & (c[:, 1] < mid))
    return v, t[keep]


def edges(t):
    """
    Edge list of a triangulation.

    Returns ev of shape (ne, 2), the two vertices of each edge, and et of
    shape (ne, 2), the two adjacent elements, the second being -1 on the
    boundary.
    """
    local = [(1, 2), (2, 0), (0, 1)]
    seen = {}
    for e in range(t.shape[0]):
        for (i, j) in local:
            seen.setdefault(tuple(sorted((t[e, i], t[e, j]))), []).append(e)
    ev = np.array(list(seen.keys()), dtype=int)
    et = np.array([[a[0], a[1] if len(a) == 2 else -1]
                   for a in seen.values()], dtype=int)
    return ev, et


# ----------------------------------------------------------------------
# the method
# ----------------------------------------------------------------------

class PWDG:
    """
    Parameters
    ----------
    n     : cells per direction of the structured mesh.  Must be even when
            the wavenumber jumps across y = 0, so that the interface is a
            mesh line.
    exact : object with __call__(x, y), grad(x, y) and wavenumber(y).
            Supplies the Dirichlet data and the local wavenumber.
    p     : number of plane wave directions per element.  With p = 2q + 1
            the space approximates like polynomials of degree q, so the
            expected rates are h^(q+1) in L2 and h^q in the energy norm.
    nq    : points in the Gauss rule on each edge.
    box   : the square is box[0] < x, y < box[1].
    domain: 'square', or 'lshape' for the square with the quadrant
            x > 0, y < 0 removed, whose reentrant corner is at the origin.
    mesh  : an explicit (v, t) pair, which overrides n, box and domain.
            Nothing below assumes the mesh is structured, so any conforming
            triangulation works; see mesh_skfem.py for reading one from a
            Gmsh file or refining one adaptively.

    The boundary is entirely Dirichlet, so the discrete problem is uniquely
    solvable provided kappa^2 is not a Dirichlet eigenvalue of the square;
    for box = (-1, 1) those are (pi^2/4)(m^2 + n^2).
    """

    def __init__(self, n, exact, p=9, nq=20, box=(-1.0, 1.0),
                 domain='square', mesh=None):
        if mesh is not None:
            self.v, self.t = mesh
            self.domain = 'given'
        elif domain == 'square':
            self.v, self.t = square_mesh(n, box)
            self.domain = domain
        elif domain == 'lshape':
            self.v, self.t = lshape_mesh(n, box)
            self.domain = domain
        else:
            raise ValueError("domain must be 'square' or 'lshape'")
        self.ev, self.et = edges(self.t)
        self.nt = self.t.shape[0]
        self.p = p
        # nominal mesh size, only used for reporting; the estimators
        # use the length of each edge and never this
        self.h = (box[1] - box[0]) / n if n else float(
            np.max([np.max(np.linalg.norm(
                self.v[self.t[:, i]] - self.v[self.t[:, (i + 1) % 3]],
                axis=1)) for i in range(3)]))
        self.exact = exact
        self.cent = self.v[self.t].mean(axis=1)
        self.kap = np.asarray(exact.wavenumber(self.cent[:, 1]), dtype=float)
        if mesh is None and self.kap.max() > self.kap.min() and n % 2:
            raise ValueError('n must be even when the wavenumber jumps at y = 0')
        ang = 2.0 * np.pi * np.arange(1, p + 1) / p
        self.d = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        self.gs, self.gw = gauss01(nq)

    # ---- local basis -------------------------------------------------
    def _basis(self, e, pts):
        """Values of the p basis functions of element e at pts, (nq, p)."""
        return np.exp(1j * self.kap[e] * (pts - self.cent[e]) @ self.d.T)

    def uh(self, e, pts):
        """Discrete solution restricted to element e, evaluated at pts."""
        return self._basis(e, pts) @ self.u[e * self.p:(e + 1) * self.p]

    def grad_uh(self, e, pts):
        """Gradient of the discrete solution on element e, shape (nq, 2)."""
        c = self.u[e * self.p:(e + 1) * self.p]
        return (self._basis(e, pts) * (1j * self.kap[e] * c)[None, :]) @ self.d

    # ---- edge geometry -----------------------------------------------
    def edge(self, f):
        """
        Quadrature data for edge f.

        Returns the physical points, the weights carrying the edge length,
        a unit normal, and the edge length.  The normal is oriented later,
        by _outward, since only the relative orientation matters.
        """
        va, vb = self.v[self.ev[f]]
        L = float(np.linalg.norm(vb - va))
        pts = va[None, :] + self.gs[:, None] * (vb - va)[None, :]
        tang = (vb - va) / L
        return pts, self.gw * L, np.array([tang[1], -tang[0]]), L

    def _outward(self, e, nrm, pts):
        """Flip nrm if needed so that it points out of element e."""
        return nrm if nrm @ (pts.mean(axis=0) - self.cent[e]) > 0 else -nrm

    # ---- assembly ----------------------------------------------------
    def assemble(self):
        p = self.p
        ndof = self.nt * p
        F = np.zeros(ndof, dtype=complex)
        rows, cols, vals = [], [], []
        ii = np.arange(p)

        def push(test, trial, block):
            """Add block[test index, trial index] to the sparse triplets."""
            rows.append(np.repeat(test * p + ii, p))
            cols.append(np.tile(trial * p + ii, p))
            vals.append(np.asarray(block).ravel())

        for f in range(self.et.shape[0]):
            pts, wq, nrm0, L = self.edge(f)
            e0, e1 = self.et[f]

            if e1 >= 0:
                # interior edge: four blocks, trial and test on either side
                n0 = self._outward(e0, nrm0, pts)
                xi = 0.5 * (self.kap[e0] + self.kap[e1])
                for eA, nA in ((e0, n0), (e1, -n0)):
                    for eB, nB in ((e0, n0), (e1, -n0)):
                        pa, pb = self._basis(eA, pts), self._basis(eB, pts)
                        G = np.einsum('q,ql,qm->lm', wq, pa, np.conj(pb),
                                      optimize=True)
                        kA, kB = self.kap[eA], self.kap[eB]
                        dA, dB = self.d @ nA, self.d @ nB
                        C = (-0.5j * kB * dB[None, :]
                             - 0.5j * kA * dB[:, None]
                             + 1j * (BETA / xi) * kA * kB
                             * dA[:, None] * dB[None, :]
                             + 1j * xi * ALPHA * float(nA @ nB))
                        push(eB, eA, (C * G).T)
            else:
                # Dirichlet edge: fluxes uhat = g, i kappa sigmahat . n =
                # grad u . n - alpha i kappa (u - g)
                e = e0
                nrm = self._outward(e, nrm0, pts)
                k = self.kap[e]
                pa = self._basis(e, pts)
                dn = self.d @ nrm
                G = np.einsum('q,ql,qm->lm', wq, pa, np.conj(pa),
                              optimize=True)
                Gd = np.einsum('q,l,ql,qm->lm', wq, dn, pa, np.conj(pa),
                               optimize=True)
                push(e, e, (-1j * k * Gd + 1j * ALPHA * k * G).T)
                g = self.exact(pts[:, 0], pts[:, 1])
                rhs = np.einsum('q,q,qm->m', wq, g, np.conj(pa),
                                optimize=True)
                rhsd = np.einsum('q,q,m,qm->m', wq, g, dn, np.conj(pa),
                                 optimize=True)
                F[e * p:(e + 1) * p] += 1j * k * rhsd + 1j * ALPHA * k * rhs

        self.A = sp.coo_matrix((np.concatenate(vals),
                                (np.concatenate(rows), np.concatenate(cols))),
                               shape=(ndof, ndof)).tocsc()
        self.F = F
        return self.A, self.F

    def solve(self):
        if not hasattr(self, 'A'):
            self.assemble()
        self.u = spla.spsolve(self.A, self.F)
        return self.u

    # ---- convenience --------------------------------------------------
    @property
    def ndof(self):
        return self.nt * self.p

    def element_area(self, e):
        a, b, c = self.v[self.t[e]]
        return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1])
                         - (b[1] - a[1]) * (c[0] - a[0]))
