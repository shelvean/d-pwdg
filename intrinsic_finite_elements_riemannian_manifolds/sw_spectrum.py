"""Complex length spectrum of the Seifert-Weber space from the face pairings in
hypforms.py.  No census and no SnapPy.

Hyperboloid model in R^{3,1} with <x,y> = -x0 y0 + x.y and basepoint O = e_0.  The
fundamental domain is the regular dodecahedron of dihedral angle 2 pi / 5, which is the
Dirichlet domain of Gamma at O, so d(O, g O) = arccosh(g[0,0]).

A closed geodesic of M is a Gamma-orbit of axes.  Every geodesic of translation length
at most R has a lift whose axis meets the dodecahedron, so its closest point x to O
satisfies d(O, x) <= circumradius and the element satisfies d(O, gO) <= R + 2 rho.
Two such lifts are Gamma-equivalent through an element of the ball of radius 2 rho,
which makes the identification a hash lookup rather than a pairwise search.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hypforms import geometry, pairing, Q, mdot, TWIST

a, rho, U, Fd, hv, hf, faces = geometry()
O = np.array([1.0, 0, 0, 0])
GENS = np.array([pairing(f, a) for f in Fd])
GENS = np.concatenate([GENS, np.linalg.inv(GENS)])


def ball(Dmax, verbose=True):
    """{ g : d(O, gO) <= Dmax } by levels, vectorized."""
    cmax = np.cosh(Dmax)
    I = np.eye(4)
    seen = {I.round(6).tobytes()}
    out = [I[None]]
    frontier = I[None]
    t0 = time.time()
    while len(frontier):
        prod = np.einsum('gij,njk->ngik', GENS, frontier).reshape(-1, 4, 4)
        keep = prod[:, 0, 0] <= cmax
        prod = prod[keep]
        keys = np.round(prod, 6).reshape(len(prod), -1)
        new, newkeys = [], []
        for i in range(len(prod)):
            k = keys[i].tobytes()
            if k not in seen:
                seen.add(k)
                new.append(prod[i])
        if not new:
            break
        frontier = np.array(new)
        out.append(frontier)
        if verbose:
            print(f"   level {len(out)-1}: +{len(frontier)} (total {len(seen)})"
                  f" [{time.time()-t0:.0f}s]", flush=True)
    return np.concatenate(out)


def axis_of(G):
    """complex length and closest point / direction of the axis, for each element."""
    w, V = np.linalg.eig(G)
    n = len(G)
    out = []
    for i in range(n):
        wi, Vi = w[i], V[i]
        real = np.where(np.abs(wi.imag) < 1e-8)[0]
        if len(real) < 2:
            out.append(None); continue
        lam = wi[real].real
        if lam.max() < 1 + 1e-8:
            out.append(None); continue
        ip_, im_ = real[np.argmax(lam)], real[np.argmin(lam)]
        ell = float(np.log(lam.max()))
        rot = wi[np.where(np.abs(wi.imag) > 1e-8)[0]]
        theta = float(abs(np.angle(rot[0]))) if len(rot) else 0.0
        vp, vm = Vi[:, ip_].real.copy(), Vi[:, im_].real.copy()
        ipd = mdot(vp, vm)
        if abs(ipd) < 1e-12:
            out.append(None); continue
        sc = np.sqrt(abs(-0.5 / ipd))
        vp, vm = vp * sc, vm * sc
        if -mdot(O, vp + vm) < 0:
            vp, vm = -vp, -vm
        A_, B_ = -mdot(O, vp), -mdot(O, vm)
        if A_ <= 0 or B_ <= 0:
            out.append(None); continue
        s = 0.5 * np.log(B_ / A_)
        x = np.exp(s) * vp + np.exp(-s) * vm
        x = x / np.sqrt(max(1e-300, -mdot(x, x)))
        y = G[i] @ x
        u = y + mdot(x, y) * x
        nu = np.sqrt(max(1e-300, mdot(u, u)))
        out.append((ell, theta, x, u / nu))
    return out


def canon(x, u):
    """slide (x, u) along its geodesic to the point closest to O"""
    A, B = -mdot(O, x), -mdot(O, u)
    t = np.arctanh(np.clip(-B / A, -0.999999999999, 0.999999999999))
    xs = x * np.cosh(t) + u * np.sinh(t)
    us = x * np.sinh(t) + u * np.cosh(t)
    xs = xs / np.sqrt(max(1e-300, -mdot(xs, xs)))
    us = us / np.sqrt(max(1e-300, mdot(us, us)))
    return xs, us


def key_of(x, u, dec=5):
    """orientation-free key of an axis through x with direction u"""
    ku = tuple(np.round(u, dec)), tuple(np.round(-u, dec))
    return (tuple(np.round(x, dec)), min(ku))


def spectrum(R, verbose=True):
    Dmax = R + 2 * rho + 0.05
    t0 = time.time()
    G = ball(Dmax, verbose)
    if verbose:
        print(f"ball to displacement {Dmax:.2f}: {len(G)} elements [{time.time()-t0:.0f}s]",
              flush=True)
    sel = np.arccosh(np.clip(G[:, 0, 0], 1, None)) <= 2 * rho + 0.05
    small = G[sel]
    if verbose:
        print(f"conjugators in the ball of radius {2*rho+0.05:.2f}: {len(small)}", flush=True)

    dat = [d for d in axis_of(G) if d is not None]
    items = {}
    for ell, theta, x, u in dat:
        if ell < 1e-6 or ell > R + 1e-9:
            continue
        if np.arccosh(max(1.0, -mdot(O, x))) > rho + 1e-7:
            continue
        items.setdefault(key_of(x, u), (ell, theta, x, u))
    items = list(items.values())
    if verbose:
        print(f"lifts with axis meeting the dodecahedron: {len(items)}", flush=True)

    from scipy.spatial import cKDTree
    P = np.array([np.r_[it[2], it[3]] for it in items]
                 + [np.r_[it[2], -it[3]] for it in items])
    owner = np.r_[np.arange(len(items)), np.arange(len(items))]
    tree = cKDTree(P)
    parent = list(range(len(items)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, (ell, theta, x, u) in enumerate(items):
        for g in small:
            gx, gu = canon(g @ x, g @ u)
            q = np.r_[gx, gu]
            for j in tree.query_ball_point(q, 1e-7):
                j = owner[j]
                if find(j) != find(i):
                    parent[find(j)] = find(i)
    classes = {}
    for i, (ell, theta, x, u) in enumerate(items):
        if find(i) == i:
            k = (round(ell, 6), round(theta, 6))
            classes[k] = classes.get(k, 0) + 1
    return sorted(classes.items()), len(G)


if __name__ == "__main__":
    R = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    print(f"twist {TWIST:.9f} = 3 pi / 5 ? {abs(TWIST - 3*np.pi/5) < 1e-12}")
    print(f"inradius {a:.9f}  circumradius {rho:.9f}")
    cl, nG = spectrum(R)
    print(f"{'length':>14} {'|holonomy|':>14} {'mult':>6}")
    tot = 0
    for (ell, th), m in cl:
        tot += m
        print(f"{ell:14.9f} {th:14.9f} {m:6d}")
    print(f"classes {len(cl)}, total multiplicity {tot}")
