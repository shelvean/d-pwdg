"""Gauss-Bonnet ladder, metric-free rungs, on the flat quotients: Euler count on
the quotient mesh, vertex-cycle angle sums at the polygon corners."""
import numpy as np
from flatquot import Quotient
for kind in ['torus','klein','mobius']:
    Q = Quotient(kind, 4, 4, 2, 0)
    # vertices modulo identification
    keys = {}
    for v in Q.V:
        keys.setdefault(Q._canonical(tuple(v)), 0); keys[Q._canonical(tuple(v))] += 1
    Vq = len(keys)
    # edges modulo identification
    ek = set()
    for T in Q.tris:
        P = Q.V[list(T)]
        for e in range(3):
            q0, q1 = P[(e+1)%3], P[(e+2)%3]
            ek.add(frozenset([Q._canonical(tuple(q0)), Q._canonical(tuple(q1))]))
    Eq = len(ek); Fq = len(Q.tris)
    # angle sum at the corner cycle(s): each square corner contributes pi/2
    corners = [Q._canonical(p) for p in [(0,0),(1,0),(0,1),(1,1)]]
    cyc = {}
    for c in corners: cyc[c] = cyc.get(c, 0) + np.pi/2
    print(f"{kind:8s} V={Vq:3d} E={Eq:3d} F={Fq:3d}  V-E+F={Vq-Eq+Fq:+d}  corner cycles: " +
          ", ".join(f"{k}: {v/np.pi:.2f} pi" for k, v in cyc.items()))
