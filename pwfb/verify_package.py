#!/usr/bin/env python3
"""Integrity/smoke check for the JCP reproduction package.

Default mode is fast and validates the shipped independently generated audit output.
Use --full to rerun the high-precision audit from scratch.
"""
from __future__ import annotations

import argparse
import compileall
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
DATA = ROOT / "data"
FIG = ROOT / "figures"

ap = argparse.ArgumentParser()
ap.add_argument("--full", action="store_true", help="rerun the high-precision Hankel audit")
args = ap.parse_args()

required_data = [
    "coarse_trace_selector.csv",
    "esprit_capacity_N401.csv",
    "highp_global_dtn_pwdg.csv",
    "highp_rank_threshold_sensitivity.csv",
    "hybrid_selector_candidates.csv",
    "hybrid_selector_global_results.csv",
    "timing_levels.csv",
]
required_figures = [
    "modal_weights_and_pw_gram_k16.pdf",
    "basis_visuals_k16.pdf",
    "herglotz_dtn_selection_k16.pdf",
    "global_error_pw_hybrid.pdf",
    "retained_rank_vs_p_full.pdf",
    "global_kappaGR_vs_p_full.pdf",
    "transmission_field.pdf",
    "esprit_capacity_N401.pdf",
]

print(f"Python: {sys.version.split()[0]}")
for name in ("numpy", "scipy", "pandas", "mpmath", "matplotlib"):
    mod = importlib.import_module(name)
    print(f"{name}: {getattr(mod, '__version__', 'unknown')}")

if not compileall.compile_dir(str(CODE), quiet=1, force=True):
    raise SystemExit("Python syntax check FAILED")
print("Python syntax check: PASSED")

missing = [str(DATA / f) for f in required_data if not (DATA / f).is_file()]
missing += [str(FIG / f) for f in required_figures if not (FIG / f).is_file()]
if missing:
    raise SystemExit("Missing required reference files:\n  " + "\n  ".join(missing))
print("Reference-file check: PASSED")

hp = CODE / "reproduction_highp_hankel"
if args.full:
    proc = subprocess.run([sys.executable, "verify.py"], cwd=hp)
    if proc.returncode:
        raise SystemExit("High-precision audit verification FAILED")
else:
    got = pd.read_csv(hp / "_verify_results" / "highp_hankel_reproduced.csv")
    exp = pd.read_csv(hp / "expected_audited_results.csv")
    for c in ("relative_trace_error", "broken_relative_L2_error", "raw_trace_gram_condition"):
        rel = np.max(np.abs(got[c] - exp[c]) / np.maximum(np.abs(exp[c]), 1e-300))
        if rel >= 5e-8:
            raise SystemExit(f"Shipped audit check FAILED in {c}: relative discrepancy {rel}")
    r = got["orth_trace_gram_condition_minus_1"] / exp["orth_trace_gram_condition_minus_1"]
    if not np.all((r > 0.8) & (r < 1.2)):
        raise SystemExit("Shipped audit orthogonality check FAILED")
    print("Shipped high-precision audit check: PASSED")
    print("Use 'python verify_package.py --full' to recompute that audit from scratch.")

print("PACKAGE CHECK PASSED")
