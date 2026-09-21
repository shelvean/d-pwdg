"""
prolong.py -- rank-revealing local prolongation for the smoothness constraints.

The dense SVD null space is O(n^3) in time and O(n^2) in memory, which caps the
hyperbolic runs at a few thousand coefficients.  The smoothness matrix is extremely
sparse instead: each C^1 condition couples exactly four Bernstein coefficients across
one edge.  So we run a rank-revealing sparse Gauss-Jordan elimination with Markowitz
pivoting and a relative threshold for stability.  This

  - reveals the rank of S (the smoothness conditions are redundant at vertices, so S
    is genuinely rank deficient and a plain LU would fail),
  - splits the coefficients into DEPENDENT (pivot) and FREE columns,
  - expresses every dependent coefficient as a combination of free ones,

which is exactly a prolongation c = Z alpha with Z sparse.  Redundant rows reduce to
zero and are dropped, which is how the rank is revealed rather than assumed.
"""

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix


def rr_prolongation(S, tol=1e-10, pivot_thresh=0.1, verbose=False):
    """Return (Z, rank, free, pivots) with S @ Z = 0 and Z of full column rank."""
    S = csr_matrix(S)
    m, n = S.shape

    rows = {}
    for i in range(m):
        s, e = S.indptr[i], S.indptr[i + 1]
        d = {int(c): float(v) for c, v in zip(S.indices[s:e], S.data[s:e]) if v != 0.0}
        if d:
            rows[i] = d
    col_rows = {}
    for i, d in rows.items():
        for c in d:
            col_rows.setdefault(c, set()).add(i)

    scale = {i: max(abs(v) for v in d.values()) for i, d in rows.items()}
    pivots = {}          # pivot column -> row
    active = set(rows)
    dropped = 0

    while active:
        # Markowitz: shortest remaining row, then the column touching fewest rows
        i = min(active, key=lambda r: len(rows[r]))
        d = rows[i]
        if not d:
            del rows[i]
            active.discard(i)
            dropped += 1
            continue
        big = max(abs(v) for v in d.values())
        if big <= tol * max(scale[i], 1.0):
            for c in list(d):
                col_rows[c].discard(i)
            del rows[i]
            active.discard(i)
            dropped += 1
            continue
        cand = [c for c, v in d.items()
                if abs(v) >= pivot_thresh * big and c not in pivots]
        if not cand:
            cand = [c for c in d if c not in pivots]
        if not cand:
            for c in list(d):
                col_rows[c].discard(i)
            del rows[i]
            active.discard(i)
            dropped += 1
            continue
        p = min(cand, key=lambda c: len(col_rows.get(c, ())))

        pv = d[p]
        for c in list(d):
            d[c] /= pv
        d[p] = 1.0

        # Gauss-Jordan: eliminate column p from every other row
        for j in list(col_rows.get(p, ())):
            if j == i:
                continue
            dj = rows[j]
            f = dj.pop(p, 0.0)
            col_rows[p].discard(j)
            if f == 0.0:
                continue
            for c, v in d.items():
                if c == p:
                    continue
                nv = dj.get(c, 0.0) - f * v
                if abs(nv) <= 1e-14 * max(scale[j], 1.0):
                    if c in dj:
                        del dj[c]
                        col_rows[c].discard(j)
                else:
                    if c not in dj:
                        col_rows.setdefault(c, set()).add(j)
                    dj[c] = nv
        pivots[p] = i
        active.discard(i)

    rank = len(pivots)
    free = np.array(sorted(set(range(n)) - set(pivots)), dtype=int)
    fpos = {c: k for k, c in enumerate(free)}

    r_, c_, v_ = [], [], []
    for k, c in enumerate(free):
        r_.append(c)
        c_.append(k)
        v_.append(1.0)
    for p, i in pivots.items():
        for c, v in rows[i].items():
            if c == p:
                continue
            if c not in fpos:
                raise RuntimeError("elimination left a pivot-pivot coupling")
            r_.append(p)
            c_.append(fpos[c])
            v_.append(-v)
    Z = coo_matrix((v_, (r_, c_)), shape=(n, len(free))).tocsr()
    if verbose:
        print(f"    rank {rank}/{min(m, n)}, {dropped} redundant rows dropped, "
              f"Z is {Z.shape[0]}x{Z.shape[1]} with {Z.nnz} nonzeros "
              f"({Z.nnz / (Z.shape[0] * Z.shape[1]) * 100:.3f}% dense)")
    return Z, rank, free, pivots
