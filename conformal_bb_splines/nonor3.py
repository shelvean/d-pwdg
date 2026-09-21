"""Corner walk with a DIRECTION flag, required once gluings may reverse orientation.

State (vertex j, dir d).  d=+1 exits through side j, d=-1 through side j-1.
   P, d=+1:  (j,+1) -> (p(j)+1, +1)        P, d=-1: (j,-1) -> (p(j-1),   -1)
   R, d=+1:  (j,+1) -> (p(j),   -1)        R, d=-1: (j,-1) -> (p(j-1)+1, +1)
With only P pairings the flag never changes and this reduces to the earlier walk,
which is why the orientable case never exposed the need for it.
"""
import numpy as np, itertools
from hypgeom import regular_polygon, frame, inv_lorentz

def frame_r(p, q):
    F = frame(p, q); F[:, 2] *= -1.0; return F

def matchings(items):
    if not items: yield []; return
    a = items[0]
    for i in range(1, len(items)):
        rest = items[1:i] + items[i+1:]
        for m in matchings(rest): yield [(a, items[i])] + m

def walk(n, pm, ty):
    st = {}
    for j in range(n):
        s = j; p = pm[s]
        st[(j, 1)] = (( (p+1) % n, 1) if ty[s] == 'P' else ((p % n), -1))
        s2 = (j-1) % n; p2 = pm[s2]
        st[(j, -1)] = ((p2 % n, -1) if ty[s2] == 'P' else (((p2+1) % n), 1))
    seen, cycles = set(), []
    for s0 in st:
        if s0 in seen: continue
        c, s = [], s0
        while s not in seen:
            seen.add(s); c.append(s); s = st[s]
        cycles.append(c)
    return cycles

def analyse(n, pairs, types):
    pm = {}; ty = {}
    for (a, b), t in zip(pairs, types):
        pm[a] = b; pm[b] = a; ty[a] = t; ty[b] = t
    cyc = walk(n, pm, ty)
    classes = [sorted({j for j, _ in c}) for c in cyc]
    # merge state-cycles that share vertices
    merged = []
    for cl in classes:
        hit = [m for m in merged if set(m) & set(cl)]
        for h in hit: merged.remove(h)
        merged.append(sorted(set(cl).union(*[set(h) for h in hit]) if hit else set(cl)))
    sizes = {len(m) for m in merged}
    if len(sizes) != 1: return None
    c = len(merged); theta = 2*np.pi/ list(sizes)[0]
    if theta >= np.pi*(n-2)/n or theta <= 0: return None
    V, R = regular_polygon(n, theta)
    g = [None]*n
    for (a, b), t in zip(pairs, types):
        if t == 'P': g[a] = frame(V[(b+1) % n], V[b]) @ inv_lorentz(frame(V[a], V[(a+1) % n]))
        else:        g[a] = frame_r(V[b], V[(b+1) % n]) @ inv_lorentz(frame(V[a], V[(a+1) % n]))
        g[b] = inv_lorentz(g[a])
    dev = 0.0
    for cc in cyc:
        W = np.eye(3)
        for (j, d) in cc:
            s = j if d == 1 else (j-1) % n
            W = g[s] @ W
        dev = max(dev, float(np.max(np.abs(W - np.eye(3)))))
    chi = c - n//2 + 1
    return dict(classes=c, theta=theta, chi=chi, dev=dev, area=(n-2)*np.pi - n*theta,
                orientable=all(t == 'P' for t in types), types=''.join(types),
                pairs=pairs, V=V, g=g)

if __name__ == "__main__":
    for n in (6, 8):
        best = {}
        for pairs in matchings(list(range(n))):
            for types in itertools.product('PR', repeat=n//2):
                r = analyse(n, pairs, list(types))
                if r is None or r['dev'] > 1e-6: continue
                k = (r['classes'], r['orientable'])
                if k not in best: best[k] = r
        print(f"n={n}")
        for (c, o), r in sorted(best.items()):
            s = (f"orientable genus {(2-r['chi'])//2}" if o
                 else f"NONORIENTABLE N_{2-r['chi']}")
            print(f"   classes {c} theta {r['theta']:.6f} chi {r['chi']:+d} "
                  f"area {r['area']:.6f} (-2pi chi {-2*np.pi*r['chi']:.6f}) dev {r['dev']:.1e} "
                  f"types {r['types']} pairs {r['pairs']}  {s}")
