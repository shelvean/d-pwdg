import time
import numpy as np
from scipy.special import jv, jvp
from fast_bessel import bessel_j_and_derivative_integer_modes


def bench(M, n=4096, zmax=24.0, repeats=8):
    # Include tiny arguments and values around several Bessel zeros.
    z = np.r_[np.geomspace(1e-10, 1e-2, n//8), np.linspace(1e-2, zmax, n-n//8)]
    modes = np.arange(-M, M+1)

    # Accuracy against SciPy/AMOS.
    _, Jf, Df = bessel_j_and_derivative_integer_modes(M, z)
    Jf = np.moveaxis(Jf, 0, 1)
    Df = np.moveaxis(Df, 0, 1)
    Jr = jv(modes[None,:], z[:,None])
    Dr = jvp(modes[None,:], z[:,None], 1)
    absJ = np.max(np.abs(Jf-Jr))
    absD = np.max(np.abs(Df-Dr))
    maskJ = np.abs(Jr) > 1e-12
    maskD = np.abs(Dr) > 1e-12
    relJ = np.max(np.abs((Jf-Jr)[maskJ]/Jr[maskJ])) if np.any(maskJ) else 0.0
    relD = np.max(np.abs((Df-Dr)[maskD]/Dr[maskD])) if np.any(maskD) else 0.0

    # Warm up
    jv(modes[None,:], z[:,None]); jvp(modes[None,:], z[:,None],1)
    bessel_j_and_derivative_integer_modes(M,z)

    t0=time.perf_counter()
    for _ in range(repeats):
        jv(modes[None,:], z[:,None]); jvp(modes[None,:], z[:,None],1)
    tref=(time.perf_counter()-t0)/repeats

    t0=time.perf_counter()
    for _ in range(repeats):
        bessel_j_and_derivative_integer_modes(M,z)
    tfast=(time.perf_counter()-t0)/repeats

    print(f'M={M:2d}, n={len(z):5d}: scipy jv+jvp {tref:.6f}s, Miller {tfast:.6f}s, speedup {tref/tfast:.2f}x')
    print(f'   max abs J={absJ:.3e}, dJ={absD:.3e}; max rel(|ref|>1e-12) J={relJ:.3e}, dJ={relD:.3e}')

if __name__ == '__main__':
    for M in (10,20,30,40,60):
        bench(M)
