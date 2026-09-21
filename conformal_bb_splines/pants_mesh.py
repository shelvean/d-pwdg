"""Right-angled hexagons and pants decomposition in the Poincare disk.
Builds a right-angled hexagon from three alternate side lengths, places it
in H^2 (hyperboloid model, then Poincare disk for drawing), triangulates it
by a fan with midpoint refinement, and lays out (i) one hexagon, (ii) a pair
of pants = two hexagons across a seam, (iii) four hexagons around a common
right-angled vertex, the fundamental domain of a genus-two surface.
Raw matplotlib panels, no text; annotate in LaTeX."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

Q = np.diag([-1.0, 1.0, 1.0])
def mink(x, y): return x @ Q @ y
def geod(p, v, s):                     # walk s along unit tangent v from p
    return np.cosh(s)*p + np.sinh(s)*v, np.sinh(s)*p + np.cosh(s)*v
def turn(p, v, ang):                   # rotate tangent v at p by ang (left)
    w = np.array([p[1]*v[2]-p[2]*v[1], -(p[0]*v[2]-p[2]*v[0]) , p[0]*v[1]-p[1]*v[0]])
    w = Q @ w; w = w/np.sqrt(abs(mink(w, w)))
    if mink(w, w) < 0: w = -w
    # ensure right-handed: pick sign so that (v,w) is positively oriented
    return np.cos(ang)*v + np.sin(ang)*w
def disk(p): return p[1:]/(1+p[0])
def hexagon(a, b, c):
    """right-angled hexagon with alternate sides a, b, c; returns 6 vertices
    (hyperboloid) in cyclic order a, c', b, a', c, b'"""
    ch = lambda x: np.cosh(x); sh = lambda x: np.sinh(x)
    ap = np.arccosh((ch(b)*ch(c)+ch(a))/(sh(b)*sh(c)))
    bp = np.arccosh((ch(a)*ch(c)+ch(b))/(sh(a)*sh(c)))
    cp = np.arccosh((ch(a)*ch(b)+ch(c))/(sh(a)*sh(b)))
    sides = [a, cp, b, ap, c, bp]
    p = np.array([1.0, 0, 0]); v = np.array([0, 1.0, 0])
    V = [p]
    for s in sides:
        p, v = geod(p, v, s); V.append(p); v = turn(p, v, np.pi/2)
    return np.array(V[:6]), sides, np.linalg.norm(disk(V[6]) - disk(V[0]))
def midpoint(p, q):
    m = p + q; return m/np.sqrt(-mink(m, m))
def geodesic_pts(p, q, n=40):
    d = np.arccosh(-mink(p, q)); out = []
    for t in np.linspace(0, 1, n):
        x = (np.sinh((1-t)*d)*p + np.sinh(t*d)*q)/np.sinh(d) if d > 1e-12 else p
        out.append(disk(x))
    return np.array(out)
def centroid(V):
    c = V.sum(0); return c/np.sqrt(-mink(c, c))
def tri_mesh(V, level):
    c = centroid(V); tris = [(c, V[i], V[(i+1) % 6]) for i in range(6)]
    for _ in range(level):
        new = []
        for (p, q, r) in tris:
            m1, m2, m3 = midpoint(p, q), midpoint(q, r), midpoint(r, p)
            new += [(p, m1, m3), (m1, q, m2), (m3, m2, r), (m1, m2, m3)]
        tris = new
    return tris
def reflect_across(V, i):
    """reflection of the hexagon across side i (from V[i] to V[i+1])"""
    p, q = V[i], V[(i+1) % 6]
    # normal to the geodesic plane through p, q (Minkowski cross product)
    n = np.array([-(p[1]*q[2]-p[2]*q[1]), (p[0]*q[2]-p[2]*q[0]), -(p[0]*q[1]-p[1]*q[0])])
    n = Q @ n; n = n/np.sqrt(mink(n, n))
    return lambda x: x - 2*mink(x, n)*n

def draw(ax, hexes, level=2, seams=(1, 3, 5), edge_lw=0.5):
    th = np.linspace(0, 2*np.pi, 400); ax.plot(np.cos(th), np.sin(th), color='0.6', lw=0.8)
    for V in hexes:
        for (p, q, r) in tri_mesh(V, level):
            for a_, b_ in ((p, q), (q, r), (r, p)):
                g = geodesic_pts(a_, b_, 12); ax.plot(g[:, 0], g[:, 1], color='0.35', lw=edge_lw)
        for i in range(6):
            g = geodesic_pts(V[i], V[(i+1) % 6])
            if i in seams:   # seams: thick solid dark red
                ax.plot(g[:, 0], g[:, 1], color='#8b2020', lw=3.2, solid_capstyle='round')
            else:            # boundary halves: thinner, dashed, blue
                ax.plot(g[:, 0], g[:, 1], color='#2b4a8b', lw=1.6, ls=(0, (4, 2)))
    ax.set_aspect('equal'); ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02); ax.axis('off')

if __name__ == '__main__':
    a = b = c = 1.2                       # half the three boundary lengths
    V, sides, closure = hexagon(a, b, c)
    print('closure defect', closure, 'sides', np.round(sides, 4))
    # (i) one hexagon
    fig, ax = plt.subplots(figsize=(4, 4)); draw(ax, [V]); fig.savefig('/home/claude/work/hyp/paper/p_hex1.png', dpi=220, bbox_inches='tight'); plt.close(fig)
    # (ii) pair of pants: hexagon and its reflection across the seam side 1
    R1 = reflect_across(V, 1); V2 = np.array([R1(x) for x in V])
    fig, ax = plt.subplots(figsize=(4, 4)); draw(ax, [V, V2]); fig.savefig('/home/claude/work/hyp/paper/p_hex2.png', dpi=220, bbox_inches='tight'); plt.close(fig)
    # (iii) four hexagons around the right-angled vertex V[1]: reflect across sides 0 and 1
    R0 = reflect_across(V, 0)
    V3 = np.array([R0(x) for x in V]); V4 = np.array([R0(x) for x in V2])
    fig, ax = plt.subplots(figsize=(4, 4)); draw(ax, [V, V2, V3, V4]); fig.savefig('/home/claude/work/hyp/paper/p_hex4.png', dpi=220, bbox_inches='tight'); plt.close(fig)
    print('done')
