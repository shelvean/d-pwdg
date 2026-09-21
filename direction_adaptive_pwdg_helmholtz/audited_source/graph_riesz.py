import numpy as np
import scipy.linalg as la

def solve_exact(A,F):
    A=A.toarray() if hasattr(A,'toarray') else np.asarray(A)
    G=(A-A.conj().T)/(2j); G=(G+G.conj().T)/2
    L=la.cholesky(G,lower=True)
    X=la.solve_triangular(L,A,lower=True)
    Ah=la.solve_triangular(L,X.conj().T,lower=True).conj().T
    fh=la.solve_triangular(L,F,lower=True)
    ch=la.solve(Ah,fh,assume_a='gen')
    return la.solve_triangular(L.conj().T,ch,lower=False)
