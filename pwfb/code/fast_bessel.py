"""Fast consecutive integer-order Bessel J sequences.

The routine is designed for Fourier--Bessel bases where all orders
0,...,M+1 are needed at the same set of arguments.  It uses a vectorized
Miller continued-ratio recurrence and one J_0 normalization call.  Integer
negative orders and derivatives are then obtained algebraically.

For the recurrence see the classical Miller/Gautschi treatment of
three-term recurrences.  The ratio form avoids the large intermediate
numbers produced by unscaled backward Miller recurrence.
"""
from __future__ import annotations

import numpy as np
from scipy.special import jv


def bessel_j_sequence_miller(M: int, z, extra: int = 36):
    """Return J_0(z),...,J_{M+1}(z) for an array of real nonnegative z.

    Parameters
    ----------
    M : int
        Highest order whose derivative may be needed.  The returned array
        includes order M+1 for the derivative identity.
    z : array_like
        Real nonnegative arguments.  Arbitrary shape is accepted.
    extra : int
        Backward-recurrence safety margin beyond max(M+1, ceil(max(z))).

    Returns
    -------
    J : ndarray, shape (M+2, *z.shape)
        Consecutive integer-order Bessel values.
    """
    if M < 0:
        raise ValueError("M must be nonnegative")
    z = np.asarray(z, dtype=float)
    shp = z.shape
    zz = z.ravel()
    out = np.zeros((M + 2, zz.size), dtype=float)

    zero = zz == 0.0
    nz = ~zero
    out[0, zero] = 1.0
    if not np.any(nz):
        return out.reshape((M + 2,) + shp)

    x = zz[nz]
    # Miller truncation must lie beyond both requested order and argument.
    L = max(M + 1, int(np.ceil(np.max(x)))) + int(extra)

    # r_m = J_{m+1}/J_m.  Start with r_L ~ 0 and recurse downward:
    # r_{m-1} = 1 / (2m/x - r_m).
    ratios = np.empty((M + 1, x.size), dtype=float)
    r = np.zeros_like(x)
    tiny = np.finfo(float).tiny
    for m in range(L, 0, -1):
        den = (2.0 * m / x) - r
        # Exact poles correspond to zeros of J_{m-1}; quadrature points will
        # almost never hit them, but guard division to keep the kernel finite.
        den = np.where(np.abs(den) < tiny, np.copysign(tiny, den + (den == 0)), den)
        r = 1.0 / den
        if m - 1 <= M:
            ratios[m - 1] = r

    vals = np.empty((M + 2, x.size), dtype=float)
    vals[0] = jv(0, x)
    for m in range(0, M + 1):
        vals[m + 1] = ratios[m] * vals[m]

    out[:, nz] = vals
    return out.reshape((M + 2,) + shp)


def bessel_j_and_derivative_integer_modes(M: int, z, extra: int = 36):
    """Return J_m(z), J'_m(z) for m=-M,...,M.

    Values for negative integer orders use J_{-m}=(-1)^m J_m; derivatives
    obey the same parity relation.  Only nonnegative orders are evaluated.
    """
    seq = bessel_j_sequence_miller(M, z, extra=extra)
    # seq[n] = J_n, n=0,...,M+1
    dpos = np.empty_like(seq[: M + 1])
    dpos[0] = -seq[1]
    if M:
        # J'_m = (J_{m-1} - J_{m+1})/2
        dpos[1:] = 0.5 * (seq[:M] - seq[2 : M + 2])

    modes = np.arange(-M, M + 1, dtype=int)
    idx = np.abs(modes)
    parity = np.where(modes < 0, np.where(idx % 2, -1.0, 1.0), 1.0)
    J = parity[(slice(None),) + (None,) * (seq.ndim - 1)] * seq[idx]
    dJ = parity[(slice(None),) + (None,) * (seq.ndim - 1)] * dpos[idx]
    return modes, J, dJ


def bessel_j_and_derivative_amos_block(M: int, z):
    """Conservative block evaluator using one AMOS J block and no jvp calls.

    This is slower than Miller but follows SciPy's AMOS evaluation for every
    nonnegative order, then obtains derivatives and negative orders exactly
    from integer-order identities.  It is useful when a nearly singular raw
    trace Gramian makes 1e-13-level changes in J_m visible downstream.
    """
    z = np.asarray(z, dtype=float)
    pos = np.arange(M + 2, dtype=int)
    # scipy.special.jv broadcasts as (point,order); transpose to (order,point)
    S = jv(pos[None, :], z.reshape(-1, 1)).T.reshape((M + 2,) + z.shape)
    dpos = np.empty_like(S[: M + 1])
    dpos[0] = -S[1]
    if M:
        dpos[1:] = 0.5 * (S[:M] - S[2 : M + 2])
    modes = np.arange(-M, M + 1, dtype=int)
    idx = np.abs(modes)
    parity = np.where(modes < 0, np.where(idx % 2, -1.0, 1.0), 1.0)
    fac = parity[(slice(None),) + (None,) * z.ndim]
    return modes, fac * S[idx], fac * dpos[idx]
