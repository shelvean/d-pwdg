
from __future__ import annotations
import time
import numpy as np

from riemannfem.forms import TrimmedPolynomialFormSpace, local_hodge_mass
from riemannfem.global_complex import DenseConstrainedComplex
from riemannfem.integration import mesh_volume
from riemannfem.spherical import poincare_homology_sphere


def run(r=3, level=0, quadrature=None, spectra=True):
    data = poincare_homology_sphere(level=level)
    nc = len(data["cells"])

    t0 = time.time()
    C = DenseConstrainedComplex(nc, r, data["pairings"])

    q = quadrature or max(7,r+4)
    local_masses = []
    for k in range(4):
        V = TrimmedPolynomialFormSpace(3,r,k)
        local_masses.append(
            [local_hodge_mass(V,g,q=q) for g in data["metrics"]]
        )

    M = C.assemble_hodge_masses(local_masses)
    volume = mesh_volume(data["metrics"],q=max(q,10))

    result = {
        "r": r,
        "level": level,
        "cells": nc,
        "dimensions": C.dimensions,
        "betti": C.betti_numbers(),
        "d_invariance": tuple(C.derivative_invariance_defect),
        "d2": C.d2_defects,
        "volume": volume,
        "volume_exact": data["exact_volume"],
        "seconds": time.time()-t0,
    }

    if spectra:
        result["spectra"] = [
            C.hodge_spectrum(M,k) for k in range(4)
        ]

    return result


def report(R):
    print(
        f"Poincare homology sphere: r={R['r']} level={R['level']} "
        f"cells={R['cells']}"
    )
    print("global dimensions:", R["dimensions"])
    print("Betti numbers:", R["betti"])
    print("d-invariance defects:", [f"{x:.3e}" for x in R["d_invariance"]])
    print("d^2 defects:", [f"{x:.3e}" for x in R["d2"]])
    print(
        "volume:",
        f"{R['volume']:.15f}",
        " exact:",
        f"{R['volume_exact']:.15f}",
        " error:",
        f"{abs(R['volume']-R['volume_exact']):.3e}"
    )
    if "spectra" in R:
        for k,lam in enumerate(R["spectra"]):
            print(f"k={k} first eigenvalues:")
            print(np.array2string(lam[:min(12,len(lam))],precision=8))
    print(f"time: {R['seconds']:.2f}s")


if __name__ == "__main__":
    import sys
    r = int(sys.argv[1]) if len(sys.argv)>1 else 3
    lev = int(sys.argv[2]) if len(sys.argv)>2 else 0
    R = run(r=r,level=lev)
    report(R)
