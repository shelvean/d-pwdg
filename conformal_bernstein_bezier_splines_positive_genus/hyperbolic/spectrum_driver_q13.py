"""Convenience driver for the quotient spectra described in quotients-13.pdf.

This driver was assembled from the recovered hyperbolic implementation. It is
not an archived original driver for the September 20 manuscript, and numerical
results must be checked against Tables 10--12 before claiming reproduction.

Examples (run from the hyperbolic directory):
 python spectrum_driver_q13.py bolza --degree 4 --level 1 --quadrature 44
 python spectrum_driver_q13.py bolza --degree 6 --level 2 --quadrature 30 --count 11
 python spectrum_driver_q13.py genus --genus 3 --degree 4 --level 1 --quadrature 24
"""
import argparse
import numpy as np
from scipy.sparse.linalg import eigsh
from hypmesh import assemble
from hypsurf import assemble_n

BOLZA_REFERENCE = 3.8388872588421995

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task", required=True)
    for name in ("bolza", "genus"):
        p = sub.add_parser(name)
        p.add_argument("--degree", type=int, default=4)
        p.add_argument("--level", type=int, default=1)
        p.add_argument("--quadrature", type=int, default=24)
        p.add_argument("--count", type=int, default=6, help="include zero eigenvalue")
        if name == "genus":
            p.add_argument("--genus", type=int, required=True)
    args = parser.parse_args()
    if args.degree % 2:
        parser.error("even homogeneous degree is required for the exact zero mode")
    if args.task == "bolza":
        M, K, area, triangles, ndof = assemble(args.level, args.degree, nquad=args.quadrature)
        genus = 2
    else:
        if args.genus < 2:
            parser.error("genus must be at least two")
        genus = args.genus
        M, K, area, triangles, ndof, _ = assemble_n(4 * genus, args.level,
                                                   args.degree, nquad=args.quadrature)
    vals = np.sort(eigsh(K, k=min(args.count,ndof-2), M=M, sigma=-1.0,
                         which="LM", return_eigenvectors=False).real)
    expected_area = 4 * np.pi * (genus - 1)
    print(f"genus={genus} degree={args.degree} level={args.level} "
          f"quadrature={args.quadrature} triangles={triangles} ndof={ndof}")
    print(f"area={area:.14g} absolute_area_defect={abs(area-expected_area):.4e}")
    for j, val in enumerate(vals):
        print(f"eigenvalue[{j}]={val:.12g}")
    if genus == 2 and len(vals)>1:
        print(f"relative_first_positive_error={abs(vals[1]-BOLZA_REFERENCE)/BOLZA_REFERENCE:.4e}")

if __name__ == "__main__":
    main()
