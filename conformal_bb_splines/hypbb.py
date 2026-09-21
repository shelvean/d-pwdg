"""
hypbb.py -- Bernstein-Bezier layer on hyperbolic geodesic triangles.

DERIVATION (the one piece that is not a transcription of the spherical case).

Write the forward light cone in polar form x = rho * u, u in H^2, rho > 0.  Then
    <dx,dx> = -drho^2 + rho^2 g_{H^2},
so the flat Minkowski metric on the cone is a LORENTZIAN cone metric in which rho is
a time coordinate.  The d'Alembertian of -drho^2 + rho^2 g in three dimensions is

    box = -( d_rhorho + (2/rho) d_rho ) + rho^{-2} Delta_{H^2},

which is the spherical formula with the radial part negated.  For the homogeneous
extension F(x) = rho^d f(x/rho) of degree d this gives

    box F = rho^{d-2} ( Delta_{H^2} f  -  d(d+1) f ),                        (*)

against the sphere's  Delta_{R^3} F = r^{d-2}( Delta_{S^2} f + d(d+1) f ).  The SIGN
of the zeroth-order term flips.  So for a homogeneous polynomial P of degree d,

    Delta_{H^2} ( P|_{H^2} ) = ( box P )|_{H^2} + d(d+1) P|_{H^2},

an exact algebraic identity: box P is again a homogeneous polynomial, of degree d-2.

GRADIENT.  With the Euler relation x . grad F = d F,
    grad_{H^2} f = J grad F + d F x        at x in H^2,
(the sphere has a minus sign there), and <grad_{H^2} f, x> = 0 identically.

CONSTANTS.  <x,x>^{d/2} = (-1)^{d/2} on H^2, so the constant function lies in the
degree-d space if and only if d is EVEN.  In barycentric form <x,x> = b^T G b with
G_{ij} = <v_i,v_j>, so the B-coefficients of the constant 1 are those of the
polynomial (-b^T G b)^{d/2}.  This matters for eigenvalue work: with odd d the
constant is not representable and lambda_0 = 0 is not attained exactly.

NOTE.  Barycentric coordinates are invariant under the isometry group: if x = V b then
gamma x = (gamma V) b for gamma in SO^+(2,1).  Hence a side pairing carries the
B-coefficients of a triangle to the SAME coefficients on the image triangle, and
gluing reduces to the ordinary smoothness conditions computed with the transformed
vertices.  The hyperbolic gluing is therefore no more expensive than the flat one.
"""

import numpy as np
from math import factorial
from hypgeom import J, mink, to_klein, from_klein, klein_area_weight, duffy_rule

# ---------------------------------------------------------------- index sets


def bern_indices(d):
    """Triangular index set [(i,j,k)] with i+j+k = d, lexicographic."""
    return [(i, j, d - i - j) for i in range(d, -1, -1) for j in range(d - i, -1, -1)]


def multinomial(d, i, j, k):
    return factorial(d) // (factorial(i) * factorial(j) * factorial(k))


# ---------------------------------------------------------------- barycentric


def vertex_matrix(v0, v1, v2):
    """V with the three hyperboloid vertices as columns."""
    return np.column_stack([np.asarray(v, float) for v in (v0, v1, v2)])


def gram(V):
    """Minkowski Gram matrix G_ij = <v_i, v_j>; diagonal is -1."""
    return V.T @ J @ V


def bary(V, X):
    """Barycentric coordinates b with V b = x.  Linear in x.  X is (...,3)."""
    return np.linalg.solve(V, np.atleast_2d(X).T).T


# ---------------------------------------------------------------- Bernstein basis


def bern_eval(d, B):
    """Values of all B^d_{ijk} at barycentric points B, shape (npts, ndof)."""
    B = np.atleast_2d(B)
    idx = bern_indices(d)
    out = np.empty((B.shape[0], len(idx)))
    for m, (i, j, k) in enumerate(idx):
        out[:, m] = multinomial(d, i, j, k) * B[:, 0] ** i * B[:, 1] ** j * B[:, 2] ** k
    return out


def bern_grad_bary(d, B):
    """Derivatives of the basis w.r.t. barycentric coordinates, shape (npts, ndof, 3)."""
    B = np.atleast_2d(B)
    idx = bern_indices(d)
    out = np.zeros((B.shape[0], len(idx), 3))
    for m, e in enumerate(idx):
        c = multinomial(d, *e)
        for a in range(3):
            if e[a] == 0:
                continue
            p = list(e)
            p[a] -= 1
            out[:, m, a] = c * e[a] * (B[:, 0] ** p[0] * B[:, 1] ** p[1] * B[:, 2] ** p[2])
    return out


def basis_and_grad(V, d, X):
    """Basis values and hyperbolic surface gradients at points X on H^2.

    Returns (phi (npts,ndof), gphi (npts,ndof,3)) with gphi tangent to H^2:
    grad_{H^2} phi = J grad_x phi + d phi x."""
    X = np.atleast_2d(X)
    B = bary(V, X)
    phi = bern_eval(d, B)
    gb = bern_grad_bary(d, B)
    Vinv = np.linalg.inv(V)
    gx = np.einsum("pma,ab->pmb", gb, Vinv)          # d/dx = (V^{-1})^T d/db
    g = np.einsum("ab,pmb->pma", J, gx) + d * phi[:, :, None] * X[:, None, :]
    return phi, g


# ---------------------------------------------------------------- constants


def constant_coeffs(V, d):
    """B-coefficients of the constant function 1.  Requires d even.

    Expands (-b^T G b)^{d/2} in the degree-d Bernstein basis by solving a small
    interpolation system on the domain points, which is exact since both sides are
    homogeneous polynomials of degree d in b."""
    if d % 2:
        raise ValueError("constants lie in the degree-d space only for even d")
    G = gram(V)
    idx = bern_indices(d)
    # collocate at the domain points b = (i,j,k)/d, plus enough extra points
    pts = np.array([(i / d, j / d, k / d) for (i, j, k) in idx])
    rhs = (-np.einsum("pa,ab,pb->p", pts, G, pts)) ** (d // 2)
    A = bern_eval(d, pts)
    return np.linalg.solve(A, rhs)


# ---------------------------------------------------------------- local matrices


def quad_points(V, n, jacobi=False):
    """Klein-model quadrature on the geodesic triangle: points on H^2 and
    weights. The collapsed vertex of the rule is placed at the corner farthest
    from the origin (largest x_0), where the area density is steepest."""
    Vp = np.array(V.T)                                  # rows = vertices
    far = int(np.argmax(Vp[:, 0]))
    order = [(far + 2) % 3, far, (far + 1) % 3]         # collapsed vertex -> local index 1
    Y = to_klein(Vp[order])
    a, b, w = duffy_rule(n, jacobi=jacobi)
    y = Y[0][None, :] + a[:, None] * (Y[1] - Y[0])[None, :] + b[:, None] * (Y[2] - Y[0])[None, :]
    jac = abs(np.cross(Y[1] - Y[0], Y[2] - Y[0]))
    return from_klein(y), w * klein_area_weight(y) * jac


def local_matrices(V, d, n=20):
    """Local mass and stiffness on one geodesic triangle."""
    X, w = quad_points(V, n)
    phi, g = basis_and_grad(V, d, X)
    M = np.einsum("p,pm,pl->ml", w, phi, phi)
    gg = np.einsum("ab,pmb->pma", J, g)              # lower the index for <.,.>
    K = np.einsum("p,pma,pla->ml", w, g, gg)
    return M, K


# ---------------------------------------------------------------- second order


def bern_hess_bary(d, B):
    """Second derivatives of the basis w.r.t. barycentric coordinates, (npts,ndof,3,3)."""
    B = np.atleast_2d(B)
    idx = bern_indices(d)
    out = np.zeros((B.shape[0], len(idx), 3, 3))
    for m, e in enumerate(idx):
        c = multinomial(d, *e)
        for a in range(3):
            for b_ in range(3):
                p = list(e)
                fac = e[a]
                p[a] -= 1
                if p[a] < 0:
                    continue
                fac *= p[b_]
                p[b_] -= 1
                if p[b_] < 0:
                    continue
                out[:, m, a, b_] = c * fac * (B[:, 0] ** p[0] * B[:, 1] ** p[1]
                                              * B[:, 2] ** p[2])
    return out


def lap_basis(V, d, X):
    """Laplace-Beltrami of each basis function at points X on H^2.

    Uses the exact identity  Delta_H (P|) = (box P)| + d(d+1) P|  together with
        box P = tr( G^{-1} Hess_b P ),      G = V^T J V,
    since Hess_x = V^{-T} Hess_b V^{-1} and tr(J V^{-T} H V^{-1}) = tr(G^{-1} H).
    So the whole second-order operator is the Minkowski Gram matrix contracted
    against the barycentric Hessian: no metric evaluation, no Christoffel symbols."""
    X = np.atleast_2d(X)
    B = bary(V, X)
    phi = bern_eval(d, B)
    H = bern_hess_bary(d, B)
    Ginv = np.linalg.inv(gram(V))
    box = np.einsum("pq,pmqr->pm", np.zeros((0, 0)), H) if False else \
        np.einsum("qr,pmqr->pm", Ginv, H)
    return phi, box + d * (d + 1) * phi
