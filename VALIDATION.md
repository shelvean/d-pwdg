# Validation of the cleaned source

**Shelvean Kapita — August 2026**

The cleanup was checked without changing the numerical algorithms.

## Syntax check

All Python files pass `python -m compileall`.

## Numerical spot checks

The cleaned and pre-cleanup sources were both run for the transmission and crossing-wave experiments.  The printed numerical results agreed exactly at the displayed precision, including nonlinear evaluation counts, relative errors, residuals, and conditioning diagnostics.

A separate finite-direction `M=1` check also gave the same values before and after cleanup:

```text
adaptive L2 error     8.356033873343226e-16
adaptive J            6.855957695518156e-29
nonlinear evaluations 6
recovered direction   17.000000 degrees
raw condition number  6.333689464604073
graph-Riesz condition 1.4567687965954845
```

The cleaned compressed DtN implementation was also checked at the audited point `p=27`, `tau_rank=1e-12`, with the manuscript quadrature settings.  It returned

```text
effective dofs          184
local retained rank     23
relative L2 error       7.89506491216754e-4
nonpositive trace eigs  0
linear residual         1.164615124181679e-13
```

This agrees with the audited table (effective dimension 184, local rank 23, relative L2 error `7.90e-4`).

These checks are not replacements for the audited manuscript data.  The published numerical record remains the CSV files under `data/audited/`.
