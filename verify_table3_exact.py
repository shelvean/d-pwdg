"""Exact numerical verification of Table 3 in the submitted manuscript.

This script intentionally calls the audited analytic-Jacobian implementation,
not the readable finite-difference reference implementation in ``core/``.

Author: Shelvean Kapita
Date: August 2026
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDITED = ROOT / "audited_source"
sys.path.insert(0, str(AUDITED))

from transmission_omega12_experiments import run_part_a, OUT  # noqa: E402

EXPECTED = {
    (2, 69): {
        "ndof": 12,
        "cum_nfev": 15,
        "l2": 1.405206341707785e-15,
        "cond_raw": 10.056802149815764,
        "cond_gr": 1.8409802568446776,
    },
    (2, 29): {
        "ndof": 12,
        "cum_nfev": 21,
        "l2": 1.9963819848785935e-15,
        "cond_raw": 1217646507.6359737,
        "cond_gr": 2.251317950098825,
    },
    (4, 69): {
        "ndof": 48,
        "cum_nfev": 15,
        "l2": 1.126507600830482e-15,
        "cond_raw": 22.945859328178646,
        "cond_gr": 2.9191129753291714,
    },
    (4, 29): {
        "ndof": 48,
        "cum_nfev": 20,
        "l2": 4.257703805636485e-15,
        "cond_raw": 42816.719563132974,
        "cond_gr": 3.520113725717814,
    },
}


def close(a: float, b: float, rtol: float = 5e-13, atol: float = 5e-28) -> bool:
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def main() -> None:
    run_part_a()
    csv_path = OUT / "transmission_omega12_analytic_final.csv"
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    print("Table 3 exact-reproduction check")
    print("source:", csv_path)
    failures = []
    for row in rows:
        key = (int(row["N"]), int(row["angle"]))
        if key not in EXPECTED:
            continue
        exp = EXPECTED[key]
        got_l2 = float(row["l2"])
        got_raw = float(row["cond_raw"])
        got_gr = float(row["cond_gr"])
        got_ndof = int(row["ndof"])
        got_nfev = int(row["cum_nfev"])

        ok = (
            got_ndof == exp["ndof"]
            and got_nfev == exp["cum_nfev"]
            and close(got_l2, exp["l2"])
            and close(got_raw, exp["cond_raw"])
            and close(got_gr, exp["cond_gr"])
        )
        print(
            f"N={key[0]}, angle={key[1]:2d}: "
            f"L2={got_l2:.16e}, nfev={got_nfev}, "
            f"cond(A)={got_raw:.15g}, cond_GR={got_gr:.15g} "
            f"[{'PASS' if ok else 'FAIL'}]"
        )
        if not ok:
            failures.append(key)

    if failures:
        raise SystemExit(f"Table 3 verification failed for: {failures}")
    print("\nPASS: all four submitted Table 3 rows are reproduced.")
    print("Rounded manuscript L2 values:")
    print("  N=2, 69 deg: 1.41e-15")
    print("  N=2, 29 deg: 2.00e-15")
    print("  N=4, 69 deg: 1.13e-15")
    print("  N=4, 29 deg: 4.26e-15")


if __name__ == "__main__":
    main()
