"""Reproduce Table: uniform PWDG vs graph--Riesz conditioning."""
import numpy as np
from scipy.linalg import cholesky
from exact import Transmission
from pwdg import PWDG

ex = Transmission(12, 69)
for p in (3,5,7,9,11,13,15,17,19,21,23,25,27,29):
    solver = PWDG(4, ex, p=p, nq=30)
    A, _ = solver.assemble()
    A = A.toarray()
    G = (A - A.conj().T)/(2j)
    G = (G + G.conj().T)/2
    eig = np.linalg.eigvalsh(G)
    kA = np.linalg.cond(A)
    try:
        L = cholesky(G, lower=True, check_finite=False)
        B = np.linalg.solve(L.conj().T, np.eye(L.shape[0]))
        kGR = np.linalg.cond(B.conj().T @ A @ B)
        sgr = f"{kGR:.6e}"
    except Exception:
        sgr = "FAIL"
    print(p, A.shape[0], f"{kA:.6e}", sgr, f"{eig[0]:.6e}")
