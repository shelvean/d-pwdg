"""
hypgeom.py -- Minkowski (hyperboloid model) geometry layer for hyperbolic splines.

Model:  H^2 = { x in R^{2,1} : <x,x> = -1, x_0 > 0 },  <u,v> = -u0 v0 + u1 v1 + u2 v2.
Isometries: SO^+(2,1) acting LINEARLY on R^3, hence preserving homogeneous
trivariate polynomials of degree d.  This is the reason the spherical-spline
Bernstein-Bezier machinery transfers: barycentric coordinates are linear and the
C^r smoothness conditions are statements about homogeneous polynomials.

Klein model is used for quadrature: geodesics are straight chords, so a hyperbolic
geodesic triangle is a Euclidean triangle with weight (1-|y|^2)^{-3/2}.

Self-contained (numpy only) so it can be dropped into the spline-obstacle codebase.
"""

import numpy as np

J = np.diag([-1.0, 1.0, 1.0])

# ---------------------------------------------------------------- Minkowski algebra


def mink(u, v):
    """Minkowski inner product <u,v> = -u0 v0 + u1 v1 + u2 v2 (vectorized on last axis)."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    return -u[..., 0] * v[..., 0] + u[..., 1] * v[..., 1] + u[..., 2] * v[..., 2]


def mnorm(u):
    """Norm of a spacelike vector."""
    return np.sqrt(np.maximum(mink(u, u), 0.0))


def mcross(u, v):
    """Minkowski cross product: J (u x v).  Orthogonal to both u and v in <,>."""
    return J @ np.cross(u, v)


def normalize_point(x):
    """Scale x inside the forward light cone onto the hyperboloid <x,x> = -1, x0 > 0."""
    x = np.asarray(x, float)
    q = -mink(x, x)
    if np.any(q <= 0):
        raise ValueError("point is not timelike")
    x = x / np.sqrt(q)[..., None]
    return np.where(x[..., :1] > 0, x, -x)


def on_hyperboloid(x, tol=1e-12):
    x = np.atleast_2d(x)
    return np.all(np.abs(mink(x, x) + 1.0) < tol) and np.all(x[..., 0] > 0)


def dist(p, q):
    """Hyperbolic distance, in the stable form d = 2 arcsinh(|p-q|/2).

    Since <p-q,p-q> = 2(cosh d - 1) = 4 sinh^2(d/2), this avoids the sqrt-amplified
    cancellation of arccosh(-<p,q>) for nearby points, which costs about half the
    digits and would contaminate every geometric quantity built on top of it."""
    w = np.asarray(p, float) - np.asarray(q, float)
    return 2.0 * np.arcsinh(0.5 * np.sqrt(np.maximum(mink(w, w), 0.0)))


def log_map(p, q):
    """Tangent vector at p pointing to q, with |u| = dist(p,q).  <u,p> = 0."""
    d = dist(p, q)
    w = q + mink(p, q)[..., None] * p          # project q onto T_p
    n = mnorm(w)
    small = n < 1e-14
    n = np.where(small, 1.0, n)
    return (d / n)[..., None] * w * (~small)[..., None] if w.ndim > 1 else (
        np.zeros(3) if small else (d / n) * w)


def exp_map(p, u):
    """Geodesic exponential at p in tangent direction u (u spacelike, <u,p>=0)."""
    t = mnorm(u)
    if t < 1e-15:
        return np.array(p, float)
    return np.cosh(t) * np.asarray(p, float) + np.sinh(t) * np.asarray(u, float) / t


def geodesic_point(p, q, s):
    """Point at parameter s in [0,1] along the geodesic from p to q."""
    return exp_map(p, s * log_map(p, q))


# ---------------------------------------------------------------- frames and SO(2,1)


def frame(p, q):
    """Minkowski-orthonormal frame F = [e0 e1 e2] with e0 = p, e1 pointing to q.

    Satisfies F^T J F = J exactly (to round-off), so F is in O(2,1)."""
    e0 = np.asarray(p, float)
    u = log_map(e0, np.asarray(q, float))
    e1 = u / mnorm(u)
    e2 = mcross(e0, e1)
    e2 = e2 / mnorm(e2)
    return np.column_stack([e0, e1, e2])


def inv_lorentz(A):
    """Exact inverse of a matrix in O(2,1):  A^{-1} = J A^T J."""
    return J @ A.T @ J


def is_lorentz(A, tol=1e-10):
    """Check A^T J A = J, det = 1, and A preserves the forward cone."""
    ortho = np.max(np.abs(A.T @ J @ A - J))
    return ortho < tol, ortho, float(np.linalg.det(A)), float(A[0, 0])


def pairing_map(p0, p1, q0, q1):
    """The element of SO^+(2,1) carrying the geodesic segment p0->p1 to q0->q1.

    Requires dist(p0,p1) == dist(q0,q1); constructed from Minkowski frames, so it
    is exact linear algebra with no optimization."""
    return frame(q0, q1) @ inv_lorentz(frame(p0, p1))


# ---------------------------------------------------------------- triangles


def angle_at(v, a, b):
    """Interior angle of the geodesic triangle at vertex v, between edges v->a, v->b."""
    u1 = log_map(v, a)
    u2 = log_map(v, b)
    c = mink(u1, u2) / (mnorm(u1) * mnorm(u2))
    return np.arccos(np.clip(c, -1.0, 1.0))


def triangle_area_defect(v0, v1, v2):
    """Exact area by the Gauss-Bonnet angle defect: pi - (A + B + C)."""
    return np.pi - (angle_at(v0, v1, v2) + angle_at(v1, v2, v0) + angle_at(v2, v0, v1))


def barycentric(V, x):
    """Minkowski barycentric coordinates: solve [v0 v1 v2] b = x.  Linear in x.

    V is 3x3 with the hyperboloid vertices as columns."""
    return np.linalg.solve(V, np.asarray(x, float))


def from_barycentric(V, b):
    """Homogeneous point V b, renormalized onto the hyperboloid."""
    return normalize_point(V @ np.asarray(b, float))


# ---------------------------------------------------------------- Klein model


def to_klein(x):
    """Central projection onto the Klein disk: y = (x1/x0, x2/x0)."""
    x = np.asarray(x, float)
    return x[..., 1:] / x[..., :1]


def from_klein(y):
    """Inverse: x = (1, y1, y2) / sqrt(1 - |y|^2)."""
    y = np.asarray(y, float)
    s = 1.0 - np.sum(y * y, axis=-1)
    return np.concatenate([np.ones(y.shape[:-1] + (1,)), y], axis=-1) / np.sqrt(s)[..., None]


def klein_area_weight(y):
    """Hyperbolic area element in Klein coordinates: (1 - |y|^2)^{-3/2}."""
    return (1.0 - np.sum(y * y, axis=-1)) ** (-1.5)


def duffy_rule(n, jacobi=False):
    """Quadrature on the reference triangle {(a,b): a,b>=0, a+b<=1} by the
    collapsed-coordinate map (u,v) -> (a,b) = (u, v(1-u)), whose Jacobian
    (1-u) is a Jacobi weight. With jacobi=True the u-direction uses the
    Gauss-Jacobi rule for the weight (1-u), which absorbs the Jacobian
    (Karniadakis-Sherwin); with jacobi=False the classical Duffy rule with
    Gauss-Legendre in both directions. The collapsed vertex is (a,b) = (1,0)."""
    from scipy.special import roots_jacobi
    xg, wg = np.polynomial.legendre.leggauss(n)
    xg = 0.5 * (xg + 1.0); wg = 0.5 * wg
    if jacobi:
        xj, wj = roots_jacobi(n, 1.0, 0.0)          # weight (1-x) on [-1,1]
        u = 0.5 * (xj + 1.0); wu = 0.25 * wj            # int_0^1 f(u)(1-u) du
        U, Vv = np.meshgrid(u, xg, indexing="ij"); WU, WV = np.meshgrid(wu, wg, indexing="ij")
        w = (WU * WV).ravel()
    else:
        U, Vv = np.meshgrid(xg, xg, indexing="ij"); WU, WV = np.meshgrid(wg, wg, indexing="ij")
        w = (WU * WV * (1.0 - U)).ravel()
    a = U.ravel(); b = (Vv * (1.0 - U)).ravel()
    return a, b, w


def klein_triangle_area(v0, v1, v2, n=24):
    """Hyperbolic area of a geodesic triangle by quadrature in the Klein model.

    Geodesic edges are straight chords in the Klein disk, so the integration domain
    is the Euclidean triangle spanned by the projected vertices.  This validates the
    quadrature route that later carries the mass and stiffness assembly."""
    y0, y1, y2 = to_klein(v0), to_klein(v1), to_klein(v2)
    a, b, w = duffy_rule(n)
    pts = y0[None, :] + a[:, None] * (y1 - y0)[None, :] + b[:, None] * (y2 - y0)[None, :]
    jac = abs(np.cross(y1 - y0, y2 - y0))
    return float(np.sum(w * klein_area_weight(pts)) * jac)


# ---------------------------------------------------------------- regular polygons


def regular_polygon(n, interior_angle):
    """Vertices of a regular hyperbolic n-gon centred at the origin (1,0,0).

    Circumradius from the right-triangle relation  cosh R = cot(pi/n) cot(theta/2)."""
    coshR = (1.0 / np.tan(np.pi / n)) * (1.0 / np.tan(0.5 * interior_angle))
    R = np.arccosh(coshR)
    th = 2.0 * np.pi * np.arange(n) / n
    return np.column_stack([
        np.cosh(R) * np.ones(n),
        np.sinh(R) * np.cos(th),
        np.sinh(R) * np.sin(th),
    ]), R


def bolza_octagon():
    """Bolza surface data: regular octagon with interior angle pi/4, opposite-side pairing.

    Returns (vertices 8x3, generators list of 4 matrices, circumradius R).
    Generator g_j maps side (v_j -> v_{j+1}) onto side (v_{j+5} -> v_{j+4})."""
    V, R = regular_polygon(8, np.pi / 4)
    gens = [pairing_map(V[j], V[(j + 1) % 8], V[(j + 5) % 8], V[(j + 4) % 8])
            for j in range(4)]
    return V, gens, R


def side_maps(gens, n=8):
    """Full list g[0..n-1] with g[j] mapping side j onto side j+4, g[j+4] = g[j]^{-1}."""
    return list(gens) + [inv_lorentz(m) for m in gens]


def vertex_cycle(n=8, step=5, start=0):
    """Corner cycle of the opposite-side pairing.

    A corner at v_j leaves through side s_j; g_j sends v_j to v_{j+5}, which is the
    corner between s_{j+4} and s_{j+5}, so the next index is j+5.  gcd(5,8)=1, so the
    cycle has length 8: all vertices are one point on the quotient."""
    return [(start + step * k) % n for k in range(n)]


def cycle_word(gens, n=8):
    """Product of side maps around the vertex cycle.  Equals the identity.

    This is the concrete form of the vertex compatibility condition: the C^r
    conditions imposed around the identified vertex close up precisely because this
    product is I.  Note the relation is NOT prod[a_i,b_i]; for the opposite-side
    pairing it is g0 g1^{-1} g2 g3^{-1} g0^{-1} g1 g2^{-1} g3."""
    g = side_maps(gens, n)
    P = np.eye(3)
    for j in vertex_cycle(n):
        P = P @ g[j]
    return P
