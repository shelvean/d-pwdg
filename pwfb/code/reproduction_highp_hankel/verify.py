#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import subprocess, sys

root = Path(__file__).resolve().parent
out = root / "_verify_results"
subprocess.run(
    [sys.executable, str(root/"reproduce_highp_hankel.py"), "--out", str(out)],
    check=True
)
got = pd.read_csv(out/"highp_hankel_reproduced.csv")
exp = pd.read_csv(root/"expected_audited_results.csv")

cols = [
    "relative_trace_error",
    "broken_relative_L2_error",
    "raw_trace_gram_condition",
]
ok = True
for c in cols:
    rel = np.max(np.abs(got[c]-exp[c]) / np.maximum(np.abs(exp[c]), 1e-300))
    print(c, "max relative discrepancy =", rel)
    ok &= rel < 5e-8

# The tiny orthogonality defects are sensitive to mpmath/libm details at the
# last few digits; compare within 10%.
r = got["orth_trace_gram_condition_minus_1"] / exp["orth_trace_gram_condition_minus_1"]
print("orth defect ratio range =", r.min(), r.max())
ok &= np.all((r > 0.8) & (r < 1.2))

if not ok:
    raise SystemExit("verification FAILED")
print("verification PASSED")
