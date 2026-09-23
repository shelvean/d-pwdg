"""Complex length spectrum of the Seifert-Weber space from the face pairings in
hypforms.py, with a memory-lean group enumeration.

Changes over sw_spectrum.py:
  * the visited set stores one int64 hash per element (8 bytes) in a sorted array
    instead of the full matrix in a Python set (about 200 bytes), so the ball to
    displacement 8.9 fits in a few hundred megabytes;
  * products are formed in chunks, so the transient (frontier x 12) array stays small;
  * the translation length is filtered by traces before any eigendecomposition.
    For g in SO+(3,1) with eigenvalues e^{l}, e^{-l}, e^{i t}, e^{-i t},
        tr(g)   = 2 cosh(l) + 2 cos(t),
        tr(g^2) = 4 (cosh^2(l) + cos^2(t)) - 4,
    so cosh(l) and cos(t) are the roots of z^2 - (tr/2) z + pq, with cosh(l) the larger.
"""
import os, sys, time
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hypforms import geometry, pairing, Q, mdot, TWIST

a, rho, U, Fd, hv, hf, faces = geometry()
O = np.array([1.0, 0, 0, 0])
GENS = np.array([pairing(f, a) for f in Fd])
GENS = np.concatenate([GENS, np.linalg.inv(GENS)])

_rng = np.random.default_rng(20260923)
_W = _rng.integers(1, 2**31, size=16).astype(np.int64)
_MOD = np.int64((1 << 61) - 1)


def hashes(M, dec=6):
    """one int64 per 4x4 matrix, from entries rounded to dec decimals"""
    K = np.rint(M.reshape(len(M), 16) * 10 ** dec).astype(np.int64)
    return (K * _W).sum(axis=1) % _MOD


def cosh_length(M):
    """cosh of the translation length, from tr(g) and tr(g^2)"""
    tr = np.einsum('nii->n', M)
    tr2 = np.einsum('nij,nji->n', M, M)
    s = tr / 2.0                      # cosh l + cos t
    q2 = (tr2 + 4.0) / 4.0            # cosh^2 l + cos^2 t
    prod = (s * s - q2) / 2.0         # cosh l * cos t
    disc = np.maximum(s * s - 4.0 * prod, 0.0)
    return 0.5 * (s + np.sqrt(disc))


def ball(Dmax, Rmax, chunk=40000, verbose=True):
    """enumerate { g : d(O,gO) <= Dmax }, returning only what is needed:
    candidates with translation length <= Rmax, and the ball of radius 2 rho."""
    cmax = np.cosh(Dmax)
    csmall = np.cosh(2 * rho + 0.05)
    cR = np.cosh(Rmax) + 1e-9
    I = np.eye(4)[None]
    seen = np.sort(hashes(I))
    frontier = I
    cand, small = [I[0:0]], [I]
    total = 1
    t0 = time.time()
    level = 0
    while len(frontier):
        level += 1
        new_parts, new_hash = [], []
        for s0 in range(0, len(frontier), chunk):
            blk = frontier[s0:s0 + chunk]
            P = np.einsum('gij,njk->ngik', GENS, blk).reshape(-1, 4, 4)
            P = P[P[:, 0, 0] <= cmax]
            if not len(P):
                continue
            h = hashes(P)
            o = np.argsort(h, kind='stable')
            h, P = h[o], P[o]
            uniq = np.r_[True, h[1:] != h[:-1]]
            h, P = h[uniq], P[uniq]
            pos = np.searchsorted(seen, h)
            fresh = ~((pos < len(seen)) & (seen[np.minimum(pos, len(seen) - 1)] == h))
            h, P = h[fresh], P[fresh]
            if not len(P):
                continue
            new_parts.append(P)
            new_hash.append(h)
        if not new_parts:
            break
        P = np.concatenate(new_parts)
        h = np.concatenate(new_hash)
        o = np.argsort(h, kind='stable')
        h, P = h[o], P[o]
        uniq = np.r_[True, h[1:] != h[:-1]]
        h, P = h[uniq], P[uniq]
        seen = np.sort(np.concatenate([seen, h]))
        total += len(P)
        cl = cosh_length(P)
        cand.append(P[cl <= cR])
        small.append(P[P[:, 0, 0] <= csmall])
        frontier = P
        if verbose:
            print(f"   level {level}: +{len(P)} (total {total}, candidates "
                  f"{sum(len(c) for c in cand)}) [{time.time()-t0:.0f}s]", flush=True)
    return np.concatenate(cand), np.concatenate(small), total


def axis_of(G):
    w, V = np.linalg.eig(G)
    out = []
    for i in range(len(G)):
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
        out.append((ell, theta, x, u / np.sqrt(max(1e-300, mdot(u, u)))))
    return out


def canon(x, u):
    A, B = -mdot(O, x), -mdot(O, u)
    t = np.arctanh(np.clip(-B / A, -1 + 1e-14, 1 - 1e-14))
    xs = x * np.cosh(t) + u * np.sinh(t)
    us = x * np.sinh(t) + u * np.cosh(t)
    xs = xs / np.sqrt(max(1e-300, -mdot(xs, xs)))
    us = us / np.sqrt(max(1e-300, mdot(us, us)))
    return xs, us


def key_of(x, u, dec=5):
    ku = tuple(np.round(u, dec)), tuple(np.round(-u, dec))
    return (tuple(np.round(x, dec)), min(ku))


def spectrum(R, verbose=True):
    t0 = time.time()
    cand, small, total = ball(R + 2 * rho + 0.05, R, verbose=verbose)
    if verbose:
        print(f"ball {total} elements, candidates {len(cand)}, conjugators {len(small)} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    items = {}
    for d in axis_of(cand):
        if d is None:
            continue
        ell, theta, x, u = d
        if ell < 1e-6 or ell > R + 1e-9:
            continue
        if np.arccosh(max(1.0, -mdot(O, x))) > rho + 1e-7:
            continue
        items.setdefault(key_of(x, u), (ell, theta, x, u))
    items = list(items.values())
    if verbose:
        print(f"lifts with axis meeting the dodecahedron: {len(items)}", flush=True)
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
            for j in tree.query_ball_point(np.r_[gx, gu], 1e-7):
                j = owner[j]
                if find(j) != find(i):
                    parent[find(j)] = find(i)
    classes = {}
    for i, it in enumerate(items):
        if find(i) == i:
            k = (round(it[0], 6), round(it[1], 6))
            classes[k] = classes.get(k, 0) + 1
    return sorted(classes.items()), total


if __name__ == "__main__":
    R = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    print(f"twist {TWIST:.9f}, inradius {a:.9f}, circumradius {rho:.9f}")
    cl, total = spectrum(R)
    print(f"{'length':>14} {'|holonomy|':>14} {'mult':>6}")
    tot = 0
    for (ell, th), m in cl:
        tot += m
        print(f"{ell:14.9f} {th:14.9f} {m:6d}")
    print(f"classes {len(cl)}, total multiplicity {tot}, group ball {total}")
    import pickle
    pickle.dump([(complex(e, t), m) for (e, t), m in cl], open(f"own_ls_R{R:g}.pkl", "wb"))
