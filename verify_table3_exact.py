"""Numerical verification of Table 3 in the submitted manuscript.

This script calls the audited analytic-Jacobian implementation, not the
readable finite-difference reference implementation in ``core/``.

Two modes.

Default (manuscript accuracy, machine independent).  Asserts the claims
Table 3 and Section "Transmission at omega=12" actually make:

  * problem sizes (``ndof``) are exact;
  * both relative L2 errors per mesh are at machine precision (< 1e-14)
    and of the same roundoff order as the reference record;
  * the recovered transmitted direction matches tangential phase matching
    analytically: theta_t = arccos(2 cos 69 deg) at 69 degrees incidence,
    eta_t = arccosh(2 cos 29 deg) at 29 degrees incidence;
  * the incident and reflected directions are recovered exactly;
  * kappa_2(A) agrees within its own roundoff floor (a condition number
    kappa computed in double precision carries relative noise of order
    eps * kappa, so digits below that floor are not portable);
  * kappa_GR agrees to 10 digits;
  * nonlinear evaluation counts agree within +-2 (optimizer stopping
    tests may cross one step earlier or later under different floating
    point kernels).

``--bitwise``.  The original exact-digit comparison (rtol = 5e-13 on
every float, evaluation counts exact).  This is a record of the
validation machine (see ``ENVIRONMENT_TESTED.txt`` and
``validation/table3_exact_verification.txt``) and is expected to pass
only there: BLAS kernel dispatch differs across CPUs, so the last bits
of roundoff-limited quantities, and with them the L2 residuals at 1e-15
and the trailing digits of large condition numbers, are hardware
dependent even with identical numpy/scipy versions.

Author: Shelvean Kapita
Date: August 2026
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDITED = ROOT / "audited_source"
sys.path.insert(0, str(AUDITED))

from transmission_omega12_experiments import run_part_a, OUT  # noqa: E402

EPS = sys.float_info.epsilon

# Reference record from the validation machine (18 Aug 2026, byte-identical
# to validation/transmission_omega12_analytic_final.csv).
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

# Analytic transmission targets (n1 = 2, n2 = 1): tangential phase matching.
THETA_T_69 = math.degrees(math.acos(2.0 * math.cos(math.radians(69.0))))
ETA_T_29 = math.acosh(2.0 * math.cos(math.radians(29.0)))

# Machine-independent tolerances.
L2_CEILING = 1e-14        # "machine precision" claim of Table 3
L2_ORDER_BAND = (0.2, 5)  # same roundoff order as the reference record
COND_GR_RTOL = 1e-10
COND_RAW_RTOL_FLOOR = 1e-9
COND_RAW_NOISE_FACTOR = 200.0  # rtol = max(floor, factor * eps * kappa_ref)
NFEV_SLACK = 2
ANGLE_TOL_DEG = 1e-8
ETA_TOL = 1e-10
ROUNDOFF_ETA = 1e-12      # etas that the theory says are exactly zero


def wrap360(theta: float) -> float:
    """Angle in [0, 360), with values a hair below 360 folded back through 0.

    The evanescent case recovers a transmitted angle of exactly zero, so the
    sign of its roundoff must not decide whether it lands near 0 or near 360.
    """
    t = theta % 360.0
    return t - 360.0 if t > 360.0 - ANGLE_TOL_DEG else t


def close(a: float, b: float, rtol: float = 5e-13, atol: float = 5e-28) -> bool:
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def check_bitwise(row: dict, exp: dict) -> list[str]:
    bad = []
    if int(row["ndof"]) != exp["ndof"]:
        bad.append("ndof")
    if int(row["cum_nfev"]) != exp["cum_nfev"]:
        bad.append("cum_nfev")
    for field in ("l2", "cond_raw", "cond_gr"):
        if not close(float(row[field]), exp[field]):
            bad.append(field)
    return bad


def check_manuscript(row: dict, exp: dict) -> list[str]:
    key = (int(row["N"]), int(row["angle"]))
    bad = []

    if int(row["ndof"]) != exp["ndof"]:
        bad.append("ndof")

    dnfev = abs(int(row["cum_nfev"]) - exp["cum_nfev"])
    if dnfev > NFEV_SLACK:
        bad.append("cum_nfev")

    l2 = float(row["l2"])
    ratio = l2 / exp["l2"]
    if not (l2 < L2_CEILING and L2_ORDER_BAND[0] <= ratio <= L2_ORDER_BAND[1]):
        bad.append("l2")

    raw = float(row["cond_raw"])
    raw_rtol = max(COND_RAW_RTOL_FLOOR,
                   COND_RAW_NOISE_FACTOR * EPS * exp["cond_raw"])
    if abs(raw - exp["cond_raw"]) > raw_rtol * exp["cond_raw"]:
        bad.append("cond_raw")

    gr = float(row["cond_gr"])
    if abs(gr - exp["cond_gr"]) > COND_GR_RTOL * exp["cond_gr"]:
        bad.append("cond_gr")

    # Recovered directions: the physical claim of the section.
    thetas = sorted(wrap360(float(row[c])) for c in ("theta1", "theta2", "theta3"))
    etas = [float(row[c]) for c in ("eta1", "eta2", "eta3")]
    if key[1] == 69:
        targets = sorted((69.0, 291.0, THETA_T_69))
        if any(abs(t - s) > ANGLE_TOL_DEG for t, s in zip(thetas, targets)):
            bad.append("angles")
        if any(abs(e) > ROUNDOFF_ETA for e in etas):
            bad.append("eta_roundoff")
    else:  # 29 degrees: evanescent transmitted wave, real angle 0
        targets = sorted((29.0, 331.0, 0.0))
        if any(abs(t - s) > ANGLE_TOL_DEG for t, s in zip(thetas, targets)):
            bad.append("angles")
        eta_ev = max(etas, key=abs)
        if abs(eta_ev - ETA_T_29) > ETA_TOL:
            bad.append("eta_evanescent")
        if sum(abs(e) > ROUNDOFF_ETA for e in etas) != 1:
            bad.append("eta_roundoff")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bitwise", action="store_true",
                    help="original exact-digit check; expected to pass only "
                         "on the validation machine")
    args = ap.parse_args()

    run_part_a()
    csv_path = OUT / "transmission_omega12_analytic_final.csv"
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    mode = "bitwise (validation machine)" if args.bitwise else "manuscript accuracy"
    print(f"Table 3 reproduction check [{mode}]")
    print("source:", csv_path)
    print(f"analytic targets: theta_t = {THETA_T_69:.10f} deg, "
          f"eta_t = {ETA_T_29:.10f}")

    failures = []
    for row in rows:
        key = (int(row["N"]), int(row["angle"]))
        if key not in EXPECTED:
            continue
        exp = EXPECTED[key]
        bad = (check_bitwise if args.bitwise else check_manuscript)(row, exp)
        note = ""
        dnfev = int(row["cum_nfev"]) - exp["cum_nfev"]
        if not args.bitwise and dnfev != 0 and "cum_nfev" not in bad:
            note = f" (nfev {dnfev:+d} vs reference, within +-{NFEV_SLACK})"
        print(
            f"N={key[0]}, angle={key[1]:2d}: "
            f"L2={float(row['l2']):.16e}, nfev={int(row['cum_nfev'])}, "
            f"cond(A)={float(row['cond_raw']):.15g}, "
            f"cond_GR={float(row['cond_gr']):.15g} "
            f"[{'PASS' if not bad else 'FAIL: ' + ','.join(bad)}]{note}"
        )
        if bad:
            failures.append(key)

    if failures:
        raise SystemExit(f"Table 3 verification failed for: {failures}")
    if args.bitwise:
        print("\nPASS: all four submitted Table 3 rows reproduced to the digit.")
    else:
        print("\nPASS: all four Table 3 rows reproduced at manuscript accuracy:")
        print("  machine-precision L2 on both meshes, Snell/evanescent")
        print("  recovery to analytic targets, conditioning within its")
        print("  roundoff floor, kappa_GR to 10 digits.")
        print("For the validation-machine digit record, rerun with --bitwise.")


if __name__ == "__main__":
    main()
