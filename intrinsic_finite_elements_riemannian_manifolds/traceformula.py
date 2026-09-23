"""Python 3 port of the Lin-Lipnowski Sage code for the coexact 1-form trace
formula on a closed hyperbolic 3-manifold, plus Booker's method.

Conventions follow the original file:
  spectralsideregularpart(F, vol, ls, b1, R, F'')  =  (1/2) sum_n Fhat(t_n),
where the coexact 1-form eigenvalues of *d are the t_n.
"""
import numpy as np


# ---------------------------------------------------------------- test function
def convolutionfourth(d):
    """4th convolution power of the indicator of [-d, d]; support [-4d, 4d]."""
    def f(x):
        if x >= 0:
            x = -x
        if x <= -4 * d:
            return 0.0
        if x <= -2 * d:
            return (4 * d + x) ** 3 / 6.0
        return (32 * d ** 3 - 12 * d * x * x - 3 * x ** 3) / 6.0
    return f


def secondderivativeshift(n, d):
    """second derivative at 0 of x -> convolutionfourth(d)(x + n d)"""
    y = abs(n) * d
    if abs(n) <= 2:
        return -4 * d + 3 * y
    if abs(n) <= 4:
        return 4 * d - y
    return 0.0


# ---------------------------------------------------------------- geometric side
def contributiononeprimitivegeodesic(F, cl, mult, R):
    l = cl.real
    hol = cl.imag
    N = int(np.floor(R / l))
    s = 0.0
    for k in range(N, 0, -1):
        denom = abs(np.exp(k * cl / 2) - np.exp(-k * cl / 2)) ** 2
        s += np.cos(k * hol) * F(k * l) / denom
    return 2.0 * mult * l * s      # factor 2: SnapPy lumps g with g^{-1}


def totalregularcontribution(F, ls, R):
    terms = [contributiononeprimitivegeodesic(F, cl, m, R) for cl, m in ls
             if cl.real < R]
    terms.sort(reverse=True)
    return float(np.sum(terms))


def identitycontribution(F, vol, secondderivative):
    return (vol / (2 * np.pi)) * (F(0.0) - secondderivative)


def trivialrepresentation_exact(d, b1):
    """integral of the 4th convolution power of 1_[-d,d] is (2d)^4, shift invariant"""
    return 0.5 * (b1 - 1) * 16.0 * d ** 4


def spectralsideregularpart(F, vol, ls, b1, R, secondderivative, d=None):
    return (totalregularcontribution(F, ls, R)
            + identitycontribution(F, vol, secondderivative)
            - trivialrepresentation_exact(d, b1))


# ---------------------------------------------------------------- Booker matrix
def bookermatrix(vol, ls, R, n, b1=0):
    """A with A x . x = spectral side for the test function built from x."""
    d = R / (2 * n + 4)
    F = convolutionfourth(d)
    scale = 2.0 * (1.0 / (2 * d)) ** 4
    spectralsums = {}
    for shift in range(-2 * n, 2 * n + 1):
        Fs = (lambda sh: (lambda x: F(x + sh * d)))(shift)
        spectralsums[shift] = scale * spectralsideregularpart(
            Fs, vol, ls, b1, R, secondderivativeshift(shift, d), d=d)
    A = np.empty((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            A[i, j] = 0.25 * (spectralsums[i + j] + spectralsums[i - j]
                              + spectralsums[-i + j] + spectralsums[-i - j])
    return A, d


def booker_function(A, d):
    """f(r) = min { x^T A x : Fhat_x(r) = 1 } = 1 / [ (sin(dr)/(dr))^4 c^T A^-1 c ].

    An eigenvalue parameter r with multiplicity m forces f(r) >= m, so f(r) < m
    rules out multiplicity m at r, and f(r) < 1 rules out r altogether.
    """
    n = A.shape[0]
    Ainv = np.linalg.inv(A)
    def f(r):
        c = np.array([np.cos(j * d * r) for j in range(n)])
        q = c @ (Ainv @ c)
        sinc = np.sin(d * r) / (d * r) if r != 0 else 1.0
        return 1.0 / (sinc ** 4 * q)
    return f
